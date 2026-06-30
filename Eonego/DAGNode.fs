/// Eonego - DAG node table (Phase 2 of the DAG+TT plan).
///
/// Each slot describes one position's *in-flight* search node: a Zobrist key, an `alpha`/`beta` window the node
/// was searched under, a partial `score`, a `move` (best-so-far), a `depth`, and a `status` word
/// (`Empty` / `Expanding` / `Done`). The full node `<key|status|move|score|alpha|beta|depth|gen>` is encoded
/// as a single 64-bit `Data` plus a 64-bit `Key` (= `realKey ^^^ Data`), in the XOR-lockless idiom of
/// `Transposition.fs`. A 64-bit aligned read is atomic on x64 and the XOR ties the two words together.
///
/// WHY NOT THE TT? The TT already encodes (move, score, depth, bound, eval) and is the completed-result
/// store. The DAG layer adds the partial-result + ownership metadata that the TT does not carry:
///   - `Status` distinguishes "no one is searching this" (Empty) from "a worker owns this now" (Expanding)
///     from "this node is finished" (Done - at that point the result is ALSO written to the TT).
///   - `alpha`/`beta` broaden the bound description: a TT entry whose `bound` permits a cutoff is enough;
///     a DagNode whose stored `[storedAlpha, storedBeta]` contains the new window likewise permits reuse.
///
/// CONCURRENCY MODEL - best-effort lock-free:
///   - Aligned 64-bit reads/writes are atomic on x64; the XOR self-check rejects torn/racing reads as Miss.
///   - Claim/complete/cancel transitions take an atomic reservation for the target cluster. If the reservation
///     is busy, the caller returns `NoClaim`/`false` immediately and falls back to normal search; no thread
///     waits for another writer.
///   - The cluster reservation closes the DATA-before-KEY publication window: two workers cannot both claim
///     different slots for the same key before either KEY becomes visible.
///   - `TryClaim` returns the claimed slot token. `Complete`/`Cancel` must present that token and validate
///     the slot is still `(key, Expanding, current-generation)`, so a racing/non-owner caller cannot publish
///     into a stranger's slot.
///   - Writer order: DATA first, then KEY. A reader that sees the fresh KEY is guaranteed to see the fresh
///     DATA (the TT's "Data before Key" rule).
///   - Resize/Clear/NewSearch are NOT concurrent with searches (UCI driver guarantees this).
///
/// ENCODING (LSB -> MSB of `Data`):
///   bits 0..1   status   (0 = Empty, 1 = Expanding, 2 = Done)
///   bits 2..17  move:16                                  (matches TT's move field layout)
///   bits 18..33 score:int16  (mate-ply-corrected by the caller, like the TT)
///   bits 34..41 alpha:int8  (signed; window-relative offset in [-127,127])
///   bits 42..49 beta:int8  (signed)
///   bits 50..57 depth:uint8
///   bits 58..62 generation:5  (mirrors the TT's 5-bit age; bumped by `NewSearch`)
///
/// `alpha`/`beta` are 8-bit signed offsets used in the common case where the search window is narrow
/// (aspiration windows +/-50..200 cp). When the true window exceeds [-127,127], the caller stores the
/// clamped `WideSentinel` (-128); a probe seeing the sentinel treats the bound as inconclusive and never
/// deduces a cutoff from it. This keeps the slot at 16 bytes - identical granularity and replacement
/// behaviour to the TT.
module Eonego.DagNode

open System
open System.Runtime.CompilerServices
open System.Threading
open Eonego.Move

[<assembly: System.Runtime.CompilerServices.InternalsVisibleTo("Eonego.Tests")>]
do ()

[<Literal>]
let StatusEmpty = 0 // slot free / evictable

[<Literal>]
let StatusExpanding = 1 // a worker owns this node and is searching it under (storedAlpha, storedBeta, depth)

[<Literal>]
let StatusDone = 2 // the node's (move, score, depth) are FINAL under (storedAlpha, storedBeta); result also in TT

[<Literal>]
let ClusterSize = 4

/// Physical stride (in `int`s) between two clusters' reservation flags. 16 ints = 64 bytes = one x64 cache
/// line, so each cluster's CAS flag gets its own line: two threads claiming/completing/cancelling unrelated
/// (non-colliding) clusters never bounce the same cache line via the reservation array, only via genuine
/// key collisions on `entries`. Pure perf padding; the logical cluster index (`ClusterIndex`) is unaffected.
[<Literal>]
let ReservationStride = 16

/// Bounded-retry tuning for `Complete`/`Cancel`'s reservation CAS (see `TryReserveSpin`): a handful of short
/// spins is enough to ride out the sub-microsecond overlap of a sibling claim/complete/cancel on the same
/// cluster without ever turning this into a blocking wait.
[<Literal>]
let ReservationSpinAttempts = 8

[<Literal>]
let ReservationSpinIterations = 16

[<Literal>]
let NoClaim = -1

/// Sentinel for an alpha/beta byte that cannot faithfully encode the real window. A probe reading this
/// treats the bound as the unsafe (inconclusive) edge, so it never proves a cutoff.
[<Literal>]
let WideSentinel = -128 // int8 -128

let inline packData
    (status: int)
    (move: Move)
    (score: int)
    (alpha: int)
    (beta: int)
    (depth: int)
    (genBound: int)
    : uint64 =
    (uint64 (status &&& 3))
    ||| (uint64 (uint16 move) <<< 2)
    ||| (uint64 (uint16 (int16 score)) <<< 18)
    ||| (uint64 (byte (int8 alpha)) <<< 34)
    ||| (uint64 (byte (int8 beta)) <<< 42)
    ||| (uint64 (byte depth) <<< 50)
    ||| (uint64 (byte genBound) <<< 58)

let inline dStatus (d: uint64) : int = int (d &&& 3UL)
let inline dMove (d: uint64) : Move = int (uint16 (d >>> 2))
let inline dScore (d: uint64) : int = int (int16 (uint16 (d >>> 18)))
let inline dAlpha (d: uint64) : int = int (int8 (byte (d >>> 34)))
let inline dBeta (d: uint64) : int = int (int8 (byte (d >>> 42)))
let inline dDepth (d: uint64) : int = int (byte (d >>> 50))
let inline dGen (d: uint64) : int = int (byte (d >>> 58))

/// True iff the stored alpha/beta encodes a window that CONTAINS [alpha..beta]. A `WideSentinel` on either
/// side fails this test (treated as inconclusive), so we never cut off based on a degenerate encoding.
let inline windowContains (storedAlpha: int) (storedBeta: int) (alpha: int) (beta: int) : bool =
    storedAlpha <> WideSentinel
    && storedBeta <> WideSentinel
    && storedAlpha <= alpha
    && storedBeta >= beta

/// Clamp a true window edge into the 8-bit signed range; return `WideSentinel` when it cannot fit.
let inline clampWindowEdge (v: int) : int =
    if v < -127 || v > 127 then WideSentinel else v

let private clustersFor (mb: int) : int =
    let bytes = int64 (max 1 mb) * 1024L * 1024L
    let perCluster = int64 (ClusterSize * 16)
    let mutable nc = 1L

    while (nc * 2L) * perCluster <= bytes do
        nc <- nc * 2L

    int nc

[<Sealed>]
[<AllowNullLiteral>]
type DagNodeTable(mb: int) =
    let clusterCount = clustersFor mb
    let entries: uint64[] = Array.zeroCreate (clusterCount * ClusterSize * 2) // pairs: key, data
    // One physical `int` per cache line per cluster (see ReservationStride doc) to avoid false-sharing
    // between unrelated clusters' CAS flags under concurrent LazySMP claim/complete/cancel traffic.
    let reservations: int[] = Array.zeroCreate (clusterCount * ReservationStride)
    let mutable clusterMask: int = clusterCount - 1
    let mutable clusterMask64: uint64 = uint64 clusterMask
    let mutable generation: int = 0

    member private _.ClusterIndex(key: uint64) : int =
        int ((key >>> 33) &&& clusterMask64)

    member private this.Base(key: uint64) : int =
        this.ClusterIndex(key) * (ClusterSize * 2)

    member private _.BaseFromCluster(cluster: int) : int =
        cluster * (ClusterSize * 2)

    member private _.Current(d: uint64) : bool =
        dStatus d <> StatusEmpty && dGen d = generation

    member private _.TryReserve(cluster: int) : bool =
        Interlocked.CompareExchange(&reservations.[cluster * ReservationStride], 1, 0) = 0

    /// Bounded-retry reservation acquire for `Complete`/`Cancel`: unlike `TryClaim` (which is allowed to walk
    /// away from contention - the caller just falls back to plain search), a caller of `Complete`/`Cancel`
    /// already owns the claimed slot and is obligated to either publish or release it. A single lost CAS here
    /// (e.g. racing a sibling worker's claim/complete/cancel on a colliding cluster index) would otherwise
    /// permanently strand the slot in `StatusExpanding` for the rest of the search generation. A handful of
    /// `SpinWait` rounds resolves the brief (sub-microsecond) overlap without ever blocking a thread.
    member private this.TryReserveSpin(cluster: int) : bool =
        let mutable acquired = this.TryReserve cluster
        let mutable attempt = 0

        while not acquired && attempt < ReservationSpinAttempts do
            Thread.SpinWait(ReservationSpinIterations)
            acquired <- this.TryReserve cluster
            attempt <- attempt + 1

        acquired

    member private _.Release(cluster: int) : unit =
        Volatile.Write(&reservations.[cluster * ReservationStride], 0)

    member _.Clear() : unit =
        Array.Clear(entries, 0, entries.Length)
        Array.Clear(reservations, 0, reservations.Length)
        generation <- 0

    member _.NewSearch() : unit =
        generation <- (generation + 1) &&& 0x1F
        Array.Clear(entries, 0, entries.Length)
        Array.Clear(reservations, 0, reservations.Length)

    member _.SizeMb: int = (entries.Length * 8) / (1024 * 1024)

    /// Generation byte shifted into the encoded offset (bits 58..62). The low 3 bits of `genBound` are
    /// currently unused; the TT packs (gen5<<3)|(ttPv<<2)|bound into them. Here we keep all 5 bits as gen.
    member private _.GenByte : int = generation &&& 0x1F

    /// Probe a slot. Returns `struct (status, move, score, alpha, beta, depth)` for the matching key, or
    /// `StatusEmpty` on a miss / torn read / stale generation. The caller applies `valueFromTt` to `score`
    /// if it plans to compare it against a root-relative bound.
    member this.Probe(key: uint64) : struct (int * Move * int * int * int * int) =
        let b = this.Base key
        let mutable i = 0
        let mutable result = struct (StatusEmpty, MoveNone, 0, 0, 0, 0)
        let mutable found = false

        while i < ClusterSize && not found do
            let k = Volatile.Read(&entries.[b + i * 2])
            let d = Volatile.Read(&entries.[b + i * 2 + 1])

            if (k ^^^ d) = key && this.Current d then
                result <- struct (dStatus d, dMove d, dScore d, dAlpha d, dBeta d, dDepth d)
                found <- true

            i <- i + 1

        result

    /// Attempt to claim an `Empty` slot for `key` under `(alpha,beta,depth)`; on success the returned slot
    /// token identifies the owner and the slot's new `Data` carries `(Expanding, MoveNone, 0, alpha, beta,
    /// depth, gen)`. Returns `NoClaim` if the key was already Expanding/Done, no empty/stale slot was
    /// available, or the cluster reservation was busy.
    member this.TryClaim (key: uint64) (alpha: int) (beta: int) (depth: int) : int =
        let cluster = this.ClusterIndex key

        if not (this.TryReserve cluster) then
            NoClaim
        else
            try
                let b = this.BaseFromCluster cluster
                let mutable i = 0
                let mutable alreadyOwned = false
                let mutable slot = NoClaim

                while i < ClusterSize && not alreadyOwned do
                    let idx = b + i * 2
                    let k = Volatile.Read(&entries.[idx])
                    let d = Volatile.Read(&entries.[idx + 1])
                    let status = dStatus d

                    if (k ^^^ d) = key && this.Current d then
                        alreadyOwned <- true
                    elif slot = NoClaim && (status = StatusEmpty || dGen d <> generation) then
                        slot <- idx

                    i <- i + 1

                if alreadyOwned || slot = NoClaim then
                    NoClaim
                else
                    let claimD =
                        packData StatusExpanding MoveNone 0 (clampWindowEdge alpha) (clampWindowEdge beta) depth this.GenByte

                    Volatile.Write(&entries.[slot + 1], claimD)
                    Volatile.Write(&entries.[slot], key ^^^ claimD)
                    slot
            finally
                this.Release cluster

    member private this.ValidClaim(claim: int, key: uint64) : bool =
        let b = this.Base key
        claim >= b && claim < b + ClusterSize * 2 && (claim &&& 1) = 0 && claim + 1 < entries.Length

    /// Mark a previously-claimed node `Done` with its final `(move, score)` under the original window. The
    /// caller MUST pass the slot token returned by `TryClaim`. Returns `false` if the cluster reservation is
    /// busy, the token is stale, or the slot no longer belongs to this claimant.
    member this.Complete
        (claim: int)
        (key: uint64)
        (move: Move)
        (score: int)
        (alpha: int)
        (beta: int)
        (depth: int)
        : bool =
        if not (this.ValidClaim(claim, key)) then
            false
        else
            let cluster = this.ClusterIndex key

            if not (this.TryReserveSpin cluster) then
                false
            else
                try
                    let k = Volatile.Read(&entries.[claim])
                    let d = Volatile.Read(&entries.[claim + 1])

                    if (k ^^^ d) = key && dStatus d = StatusExpanding && dGen d = generation then
                        let doneD =
                            packData StatusDone move score (clampWindowEdge alpha) (clampWindowEdge beta) depth this.GenByte

                        Volatile.Write(&entries.[claim + 1], doneD)
                        Volatile.Write(&entries.[claim], key ^^^ doneD)
                        true
                    else
                        false
                finally
                    this.Release cluster

    /// Abandon a previously-claimed node without publishing a reusable result. Used for heuristic exits that
    /// do not go through the normal TT-store path.
    member this.Cancel(claim: int, key: uint64) : bool =
        if not (this.ValidClaim(claim, key)) then
            false
        else
            let cluster = this.ClusterIndex key

            if not (this.TryReserveSpin cluster) then
                false
            else
                try
                    let k = Volatile.Read(&entries.[claim])
                    let d = Volatile.Read(&entries.[claim + 1])

                    if (k ^^^ d) = key && dStatus d = StatusExpanding && dGen d = generation then
                        Volatile.Write(&entries.[claim + 1], 0UL)
                        Volatile.Write(&entries.[claim], 0UL)
                        true
                    else
                        false
                finally
                    this.Release cluster

    // Test-only probes.
    member internal _.RawEntries: uint64[] = entries
    member internal _.RawReservations: int[] = reservations
    member internal this.ClusterBase(key: uint64) : int = this.Base key
    // Physical index into RawReservations (cluster index * ReservationStride; see TryReserve/Release).
    member internal this.ReservationIndex(key: uint64) : int = this.ClusterIndex key * ReservationStride
    member internal _.Generation: int = generation

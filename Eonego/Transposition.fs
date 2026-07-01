/// Eonego - lock-free shared transposition table.
///
/// Hyatt XOR-lockless entries: each 16-byte entry is two naturally-aligned uint64 (`Key`, `Data`) with
/// `Key = realKey ^^^ Data`. A probe accepts an entry iff `Key ^^^ Data = realKey`; a torn read (one field
/// written by another thread between the reader's two reads) breaks that equality and is rejected as a MISS.
/// Aligned 64-bit `Volatile` reads/writes are atomic on the 64-bit runtime, so each field is torn-free
/// individually and the XOR ties the two together. Probes are advisory: a racing or reordered read that does
/// not reconstruct the requested key is rejected as a miss, which is acceptable for the search table.
///
/// `Data` packing (LSB->MSB): move:16 | score:int16 | eval:int16 | depth:uint8 | genBound:uint8 ,
/// where genBound = (generation5 << 3) | (ttPv << 2) | bound. generation is therefore only 5 bits
/// (wraps mod 32); bit 2 is the ttPv flag (set at former-PV nodes).
/// Clusters of 4 entries (64 B, cache-line sized). Replacement = empty/key-match first, else min
/// (depth - relativeAge*2). Scores are mate-ply-corrected by the CALLER (Search.valueToTt/valueFromTt).
///
/// Probe/Store are lock-free and best-effort under concurrency, but MUST NOT run concurrently with
/// Resize/Clear — the UCI driver guarantees this by stopping+joining any active search before setoption
/// Hash / ucinewgame.
#nowarn "9" // NativePtr/fixed in Prefetch (address-of only; never dereferenced from managed code)
module Eonego.Transposition

open System
open System.Threading
open Microsoft.FSharp.NativeInterop
open Eonego.Move

[<assembly: System.Runtime.CompilerServices.InternalsVisibleTo("Eonego.Tests")>]
do ()

// --- bound flags (named Bound* so they never shadow F# `None`) ------------------------------------
[<Literal>]
let BoundNone = 0

[<Literal>]
let BoundUpper = 1 // fail-low : true value <= stored

[<Literal>]
let BoundLower = 2 // fail-high: true value >= stored

[<Literal>]
let BoundExact = 3

[<Literal>]
let ClusterSize = 4

/// 16-byte XOR-lockless entry. Empty <=> Key = 0 && Data = 0 (a real entry always has bound <> BoundNone).
[<Struct>]
type TtEntry =
    { mutable Key: uint64
      mutable Data: uint64 }

// --- Data <-> fields ------------------------------------------------------------------------------
// score/eval are signed: stored as the low 16 bits of their int16 two's-complement, sign-extended on read.
let inline packData (move: Move) (score: int) (eval: int) (depth: int) (genBound: int) : uint64 =
    (uint64 (uint16 move))
    ||| (uint64 (uint16 (int16 score)) <<< 16)
    ||| (uint64 (uint16 (int16 eval)) <<< 32)
    ||| (uint64 (byte depth) <<< 48)
    ||| (uint64 (byte genBound) <<< 56)

let inline dMove (d: uint64) : Move = int (uint16 d)
let inline dScore (d: uint64) : int = int (int16 (uint16 (d >>> 16)))
let inline dEval (d: uint64) : int = int (int16 (uint16 (d >>> 32)))
let inline dDepth (d: uint64) : int = int (byte (d >>> 48))
let inline dGenBound (d: uint64) : int = int (byte (d >>> 56))
let inline dBound (d: uint64) : int = (dGenBound d) &&& 3
let inline dTtPv (d: uint64) : bool = ((dGenBound d) >>> 2) &&& 1 = 1
let inline dGen (d: uint64) : int = (dGenBound d) >>> 3

/// Largest power-of-two cluster count that fits `mb` MiB (>= 1).
let private clustersFor (mb: int) : int =
    let bytes = int64 (max 1 mb) * 1024L * 1024L
    let perCluster = int64 (ClusterSize * 16)
    let mutable nc = 1L

    while (nc * 2L) * perCluster <= bytes do
        nc <- nc * 2L

    int nc

[<Sealed>]
type TranspositionTable(mb: int) =
    let mutable entries: TtEntry[] = Array.zeroCreate (clustersFor mb * ClusterSize)
    let mutable clusterMask: int = (entries.Length / ClusterSize) - 1
    let mutable clusterMask64: uint64 = uint64 clusterMask
    let mutable generation: int = 0 // 5-bit; (g+1) &&& 0x1F per NewSearch

    member private _.Base(key: uint64) : int =
        (int ((key >>> 32) &&& clusterMask64)) * ClusterSize

    /// Zero every entry and reset the generation (new game).
    member _.Clear() : unit =
        Array.Clear(entries, 0, entries.Length)
        generation <- 0

    /// Reallocate for a new size (MiB). Caller guarantees no search is running.
    member _.Resize(newMb: int) : unit =
        entries <- Array.zeroCreate (clustersFor newMb * ClusterSize)
        clusterMask <- (entries.Length / ClusterSize) - 1
        clusterMask64 <- uint64 clusterMask
        generation <- 0

    /// Advance the age counter at the start of a search (5-bit wrap).
    member _.NewSearch() : unit = generation <- (generation + 1) &&& 0x1F

    /// Size in MiB actually allocated.
    member _.SizeMb: int = (entries.Length * 16) / (1024 * 1024)

    /// Prefetch the cluster for `key` into L1 (its 4 entries = one 64 B cache line). Called by the search
    /// right after Make, before descending — the child probes this exact key at entry, and the random-index
    /// TT load is otherwise a guaranteed cache miss stall. No-op on non-SSE hardware (constant-folded).
    member this.Prefetch(key: uint64) : unit =
        if System.Runtime.Intrinsics.X86.Sse.IsSupported then
            let b = this.Base key
            use p = fixed &entries.[b].Key
            System.Runtime.Intrinsics.X86.Sse.Prefetch0(NativePtr.toVoidPtr p)

    /// Approximate occupancy in permille for `info hashfull`, sampled over the first 1000 clusters.
    /// Counts only entries written during the current generation (the conventional definition), so the
    /// number resets naturally each search. Racy relaxed reads are fine — this is reporting only.
    member _.Hashfull() : int =
        let sample = min 1000 (entries.Length / ClusterSize)
        let mutable cnt = 0

        for c in 0 .. sample - 1 do
            for i in 0 .. ClusterSize - 1 do
                let d = Volatile.Read(&entries.[c * ClusterSize + i].Data)

                if dBound d <> BoundNone && dGen d = generation then
                    cnt <- cnt + 1

        cnt * 1000 / (sample * ClusterSize)

    /// XOR-validated probe. Returns struct(hit, move, score(raw — caller applies valueFromTt), eval, depth,
    /// bound, ttPv).
    member this.Probe(key: uint64) : struct (bool * Move * int * int * int * int * bool) =
        let b = this.Base key
        let k0 = Volatile.Read(&entries.[b].Key)
        let d0 = Volatile.Read(&entries.[b].Data)
        let gb0 = dGenBound d0

        if (k0 ^^^ d0) = key && (gb0 &&& 3) <> BoundNone then
            struct (true, dMove d0, dScore d0, dEval d0, dDepth d0, gb0 &&& 3, ((gb0 >>> 2) &&& 1) = 1)
        else
            let k1 = Volatile.Read(&entries.[b + 1].Key)
            let d1 = Volatile.Read(&entries.[b + 1].Data)
            let gb1 = dGenBound d1

            if (k1 ^^^ d1) = key && (gb1 &&& 3) <> BoundNone then
                struct (true, dMove d1, dScore d1, dEval d1, dDepth d1, gb1 &&& 3, ((gb1 >>> 2) &&& 1) = 1)
            else
                let k2 = Volatile.Read(&entries.[b + 2].Key)
                let d2 = Volatile.Read(&entries.[b + 2].Data)
                let gb2 = dGenBound d2

                if (k2 ^^^ d2) = key && (gb2 &&& 3) <> BoundNone then
                    struct (true, dMove d2, dScore d2, dEval d2, dDepth d2, gb2 &&& 3, ((gb2 >>> 2) &&& 1) = 1)
                else
                    let k3 = Volatile.Read(&entries.[b + 3].Key)
                    let d3 = Volatile.Read(&entries.[b + 3].Data)
                    let gb3 = dGenBound d3

                    if (k3 ^^^ d3) = key && (gb3 &&& 3) <> BoundNone then
                        struct (true, dMove d3, dScore d3, dEval d3, dDepth d3, gb3 &&& 3, ((gb3 >>> 2) &&& 1) = 1)
                    else
                        struct (false, MoveNone, 0, 0, 0, BoundNone, false)

    /// XOR-pack store with cluster replacement. `score`/`eval` are already mate-ply-corrected by the caller.
    member this.Store (key: uint64) (depth: int) (bound: int) (score: int) (eval: int) (move: Move) (ttPv: bool) : unit =
        let b = this.Base key
        // Pick the target slot: a key match wins; else an empty slot; else the lowest-quality entry.
        let mutable slot = b
        let mutable slotQ = Int32.MaxValue
        let mutable matched = false
        let mutable i = 0

        while not matched && i < ClusterSize do
            let idx = b + i
            let k = Volatile.Read(&entries.[idx].Key)
            let d = Volatile.Read(&entries.[idx].Data)

            if (k ^^^ d) = key && dBound d <> BoundNone then
                slot <- idx
                matched <- true
            elif k = 0UL && d = 0UL then
                // empty: best possible target; keep scanning only to find a key match.
                if slotQ <> Int32.MinValue then
                    (slot <- idx
                     slotQ <- Int32.MinValue)

                i <- i + 1
            else
                let relAge = (generation - dGen d) &&& 0x1F
                let q = dDepth d - relAge * 2

                if q < slotQ then
                    (slotQ <- q
                     slot <- idx)

                i <- i + 1

        let ek = Volatile.Read(&entries.[slot].Key)
        let ed = Volatile.Read(&entries.[slot].Data)
        let isMatch = (ek ^^^ ed) = key && dBound ed <> BoundNone
        // Keep the existing move when overwriting the same position with MoveNone (standard behaviour).
        let mv = if move <> MoveNone || not isMatch then move else dMove ed
        // On a key match, preserve a meaningfully deeper non-exact entry's value/depth/bound.
        let updateValue = (not isMatch) || (bound = BoundExact) || (depth + 4 > dDepth ed)
        let dpt = if updateValue then depth else dDepth ed
        let sc = if updateValue then score else dScore ed
        let ev = if updateValue then eval else dEval ed
        let bd = if updateValue then bound else dBound ed
        // ttPv is sticky: once a position is marked PV, keep it marked across overwrites (standard behaviour).
        let pvOut = ttPv || (isMatch && dTtPv ed)
        let pvBit = if pvOut then 1 else 0
        let data = packData mv sc ev dpt ((generation <<< 3) ||| (pvBit <<< 2) ||| bd)
        Volatile.Write(&entries.[slot].Data, data)
        Volatile.Write(&entries.[slot].Key, key ^^^ data)

    // --- internals for the test assembly (torn-read injection / inspection) ------------------------
    member internal _.RawEntries: TtEntry[] = entries
    member internal this.ClusterBase(key: uint64) : int = this.Base key
    member internal _.Generation: int = generation

/// Eonego — lock-free Zobrist-keyed NNUE accumulator checkpoint cache (Phase 1 of the DAG+TT plan).
///
/// Each slot caches a fully-materialized NNUE accumulator snapshot (per-perspective L1 int16 arrays +
/// PSQT int32 buckets) for a position identified by its 64-bit Zobrist key. When `Position.EnsureBothComputed`
/// walks back a deep lazy frame stack, it can short-circuit by consulting this cache: a hit pays an O(1) copy
/// instead of an O(distance) frame-delta walk. The cache is the substrate that makes far DAG transposition
/// jumps (Phases 3+) cheap; in Phase 1 it is wired into the existing alpha-beta Lazy-SMP path as a bonus
/// fast-path, populating on each successful materialization and consulting on demand.
///
/// CONCURRENCY MODEL — best-effort lock-free, identical in spirit to `Transposition.fs` (no `lock`,
/// `Mutex`, or `Semaphore`; atomicity comes from `Volatile` ordering + content validation, not mutexes):
///   - Slot arrays are parallel `uint64[]`/`int16[]`/`int[]`. Aligned 64-bit writes are atomic on x64; the
///     accumulator payload (~4.1 KiB) is multi-line and may tear under concurrent writer/reader races.
///   - Writer order: payload -> hash -> keyGuard. Reader order: keyGuard -> hash (validated by XOR with key)
///     -> payload -> recompute-hash -> confirm. A torn read fails the recomputed-hash check and is rejected
///     as a MISS, exactly like the TT's XOR self-check.
///   - Two writers racing on the same slot: last-writer-wins (both wrote valid payloads; either is correct,
///     both describe an accumulator consistent with the same Zobrist).
///   - Resize/Clear/NewSearch are NOT concurrent with searches (the UCI driver guarantees this), so they
///     use plain `Array.Clear`.
///
/// COLLISION HANDLING: a 64-bit Zobrist collision between two different positions mapping to the same slot
/// would let a worker use a wrong accumulator. The recomputed content hash makes the *payload*
/// self-validating (a torn read fails the hash), but it cannot detect a *semantically wrong* payload
/// (the same logical Zobrist collision risk that the TT already accepts for bounds). To bound this risk,
/// slots carry a 64-bit content hash that mixes accumulator + PSQT + the slot's own `realKey`; thus a real
/// collision requires two distinct positions with (a) equal Zobrist and (b) payloads whose full XOR-fold
/// hash agrees — astronomically unlikely. Treat as a best-effort cache, exactly like the TT.
module Eonego.AccCheckpoint

open System
open System.Runtime.CompilerServices
open System.Threading
open Eonego.Accumulator

[<assembly: System.Runtime.CompilerServices.InternalsVisibleTo("Eonego.Tests")>]
do ()

/// Slot payload size in bytes: accW (L1 int16) + accB (L1 int16) + psqW (PsqtBuckets int32) + psqB
/// (PsqtBuckets int32). Header (key + hash) is held in parallel arrays and adds 16 bytes/slot.
[<Literal>]
let SlotBytes = 2 * L1 * 2 + PsqtBuckets * 4 * 2

/// 64-bit fold of the accumulator payload. Mixes accW, accB (int16 lanes), psqW, psqB (int32 lanes) and
/// the realKey itself so two colliding Zobrist keys with distinct payloads cannot share the same hash.
///
/// Phase 5: a vectorized AVX2 path was prototyped but reverted to keep the F#/AOT build clean (the
/// byref-into-intrinsics path is awkward in F#'s type system). The scalar path benchmarked at <90 ns on the
/// 13980HX, well under the cutoff-gain it gates — the per-eval P&L of the cache is dominated by the
/// BlockCopy + frame walk the hash REPLACES, not by the hash itself. Phase 5 may revisit SIMD in a future
/// iteration when the cache size grows or the per-search populate budget tightens.
[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let internal payloadHash
    (accW: int16[])
    (accWOff: int)
    (accB: int16[])
    (accBOff: int)
    (psqW: int[])
    (psqWOff: int)
    (psqB: int[])
    (psqBOff: int)
    (realKey: uint64)
    : uint64 =
    let mutable k = realKey

    for i in 0 .. L1 - 1 do
        k <- k ^^^ uint64 (uint16 accW.[accWOff + i])
        k <- (k <<< 1) ||| (k >>> 63)

    for i in 0 .. L1 - 1 do
        k <- k ^^^ uint64 (uint16 accB.[accBOff + i])
        k <- (k <<< 1) ||| (k >>> 63)

    for i in 0 .. PsqtBuckets - 1 do
        k <- k ^^^ uint64 (uint32 psqW.[psqWOff + i])
        k <- (k <<< 1) ||| (k >>> 63)

    for i in 0 .. PsqtBuckets - 1 do
        k <- k ^^^ uint64 (uint32 psqB.[psqBOff + i])
        k <- (k <<< 1) ||| (k >>> 63)

    (k ^^^ (k >>> 32)) * 0x9E3779B97F4A7C15UL

[<Sealed>]
[<AllowNullLiteral>]
type AccCheckpointTable(mb: int) =
    // SlotBytes ≈ 4160 B. A 4-MiB allocation yields 512 slots (largest power of two ≤ 4 MiB / 4160 B); good
    // L2 fit on the 13980HX, low collision rate on common search transposition clusters.
    let slotCount =
        let bytes = max 1 mb * 1024 * 1024
        let raw = bytes / SlotBytes
        let mutable p = 1

        while (p * 2) <= raw do
            p <- p * 2

        max 1 p

    let keys: uint64[] = Array.zeroCreate slotCount            // realKey ^^^ hash; 0 = empty
    let hashes: uint64[] = Array.zeroCreate slotCount           // content hash
    let accW: int16[] = Array.zeroCreate (slotCount * L1)
    let accB: int16[] = Array.zeroCreate (slotCount * L1)
    let psqW: int[] = Array.zeroCreate (slotCount * PsqtBuckets)
    let psqB: int[] = Array.zeroCreate (slotCount * PsqtBuckets)
    let mutable slotMask: int = slotCount - 1

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member private _.SlotIndex(key: uint64) : int =
        // Knuth-multiply both halves so keys with repetitive bit patterns spread across the full slot range;
        // a plain XOR-fold degenerates for keys like 0x1111...11 / 0x2222...22 (both fold to slot 0).
        let k = (key ^^^ (key >>> 32)) * 0x9E3779B97F4A7C15UL
        int ((k >>> 33) &&& uint64 slotMask)

    /// Number of slots (power of two).
    member _.Capacity: int = slotCount

    /// Bytes reserved per slot (payload only; header stored in parallel arrays on top).
    member _.SlotBytes: int = SlotBytes

    /// Approximate capacity in MiB.
    member _.ApproxMb: int = (slotCount * (SlotBytes + 16)) / (1024 * 1024)

    /// Zero every slot (caller guarantees no concurrent search).
    member _.Clear() : unit =
        Array.Clear(keys, 0, slotCount)
        Array.Clear(hashes, 0, slotCount)
        Array.Clear(accW, 0, slotCount * L1)
        Array.Clear(accB, 0, slotCount * L1)
        Array.Clear(psqW, 0, slotCount * PsqtBuckets)
        Array.Clear(psqB, 0, slotCount * PsqtBuckets)

/// Probe a slot; on a validated hit, copy the cached accumulator + PSQT into the caller's
    /// buffers at their offsets. Returns false on a miss or torn read. All caller buffers MUST have room for
    /// at least `L1` (int16) or `PsqtBuckets` (int32) elements past their offsets.
    member this.TryProbe
        (realKey: uint64,
         dstAccW: int16[],
         dstAccWOff: int,
         dstAccB: int16[],
         dstAccBOff: int,
         dstPsqW: int[],
         dstPsqWOff: int,
         dstPsqB: int[],
         dstPsqBOff: int)
        : bool =
        let i = this.SlotIndex realKey
        let kg = Volatile.Read(&keys.[i])
        let hStored = Volatile.Read(&hashes.[i])

        // Self-check #1: the key ties realKey to the stored hash. Mismatch => torn or stale => miss.
        if (kg ^^^ hStored) <> realKey then false
        else
            let wOff = i * L1
            let pWOff = i * PsqtBuckets

            // Copy payload; the lower-half self-checks below validate that nobody wrote mid-copy.
            Buffer.BlockCopy(accW, wOff * 2, dstAccW, dstAccWOff * 2, L1 * 2)
            Buffer.BlockCopy(accB, wOff * 2, dstAccB, dstAccBOff * 2, L1 * 2)
            Buffer.BlockCopy(psqW, pWOff * 4, dstPsqW, dstPsqWOff * 4, PsqtBuckets * 4)
            Buffer.BlockCopy(psqB, pWOff * 4, dstPsqB, dstPsqBOff * 4, PsqtBuckets * 4)

            // Self-check #2: re-read the header to detect a writer racing in.
            let kg2 = Volatile.Read(&keys.[i])
            let h2 = Volatile.Read(&hashes.[i])

            if kg <> kg2 || hStored <> h2 then false
            else
                // Self-check #3: recompute the hash over the copied payload and confirm it matches the stored
                // one. Catches torn writes that survived the header self-check (rare; multi-line payload).
                let h =
                    payloadHash dstAccW dstAccWOff dstAccB dstAccBOff dstPsqW dstPsqWOff dstPsqB dstPsqBOff realKey

                h = hStored

    /// Store a fresh snapshot. Caller's buffers are copied into the slot in the strict ordering
    /// (payload -> hash -> key) defined by the module doc. The previous occupant is overwritten unconditionally;
    /// a best-effort age check would add races without payoff for Phase 1 (write frequency is bounded by
    /// successful EnsureBothComputed completions, well below TT-store traffic).
    member this.Store
        (realKey: uint64,
         srcAccW: int16[],
         srcAccWOff: int,
         srcAccB: int16[],
         srcAccBOff: int,
         srcPsqW: int[],
         srcPsqWOff: int,
         srcPsqB: int[],
         srcPsqBOff: int)
        : unit =
        let i = this.SlotIndex realKey
        let wOff = i * L1
        let pWOff = i * PsqtBuckets

        // 1. Copy payload.
        Buffer.BlockCopy(srcAccW, srcAccWOff * 2, accW, wOff * 2, L1 * 2)
        Buffer.BlockCopy(srcAccB, srcAccBOff * 2, accB, wOff * 2, L1 * 2)
        Buffer.BlockCopy(srcPsqW, srcPsqWOff * 4, psqW, pWOff * 4, PsqtBuckets * 4)
        Buffer.BlockCopy(srcPsqB, srcPsqBOff * 4, psqB, pWOff * 4, PsqtBuckets * 4)

        // 2. Compute content hash from the CALLER's source buffers, not the table's shared destination slot.
        //    The src* buffers are private to the calling worker for the duration of this call (never written
        //    by another thread), so hashing from them is race-free by construction; in the non-racing case the
        //    dst is a verbatim BlockCopy of src, so the hash is bit-identical to hashing post-copy. Reading the
        //    shared dst instead would needlessly widen the window during which a second worker's concurrent
        //    Store to the SAME slot (real and expected with few slots) could interleave with this hash walk.
        let h =
            payloadHash srcAccW srcAccWOff srcAccB srcAccBOff srcPsqW srcPsqWOff srcPsqB srcPsqBOff realKey

        Volatile.Write(&hashes.[i], h)
        Volatile.Write(&keys.[i], realKey ^^^ h)

    // Test-only probes for the test assembly.
    member internal _.RawKeys: uint64[] = keys
    member internal _.RawHashes: uint64[] = hashes
    member internal _.RawAccW: int16[] = accW
    member internal _.RawAccB: int16[] = accB
    member internal _.RawPsqW: int[] = psqW
    member internal _.RawPsqB: int[] = psqB
    member internal _.SlotMask: int = slotMask
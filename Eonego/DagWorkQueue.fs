/// Eonego - bounded MPMC work-stealing queue (Phase 3 substrate of the DAG+TT plan).
///
/// Vyukov-style bounded multi-producer multi-consumer ring. Producers/consumers reserve a global position
/// with CAS, then publish the slot-local sequence after the payload write/read has happened. The queue is
/// BEST-EFFORT: a failed push means the producer keeps the work itself; a failed pop means the worker falls
/// back to its local search.
module Eonego.DagWorkQueue

open System
open System.Runtime.CompilerServices
open System.Threading

[<assembly: System.Runtime.CompilerServices.InternalsVisibleTo("Eonego.Tests")>]
do ()

[<Sealed>]
type DagWorkQueue(capacity: int) =
    // Capacity must be a positive power of two; the ctor rounds up. ~16 bytes/slot (key uint64 + seq int64).
    let cap =
        let mutable p = 1
        while p < capacity do
            p <- p * 2
        max 1 p

    let mask: int = cap - 1
    let keys: uint64[] = Array.zeroCreate cap
    let seqs: int64[] =
        // Initial state: slot i has its "ready to write" sequence = i (Vyukov convention).
        Array.init cap int64

    let mutable enqPos: int64 = 0L
    let mutable deqPos: int64 = 0L

    member _.Capacity: int = cap

    member _.Count: int =
        // Best-effort snapshot of (enqueued - dequeued), non-negative by monotonic advance.
        let e = Volatile.Read(&enqPos)
        let d = Volatile.Read(&deqPos)
        max 0 (int (e - d))

    /// Try to enqueue `key` as a stealable work unit. Returns `false` when the queue is full or contention
    /// wins the bounded retry budget.
    member _.TryPush(key: uint64) : bool =
        let mutable pos = Volatile.Read(&enqPos)
        let mutable ok = false
        let mutable doneTrying = false
        let mutable spins = 0

        while not ok && not doneTrying && spins < 64 do
            let i = int (pos &&& int64 mask)
            let seq = Volatile.Read(&seqs.[i])
            let dif = seq - pos

            if dif = 0L then
                // Reserve this producer position first. The slot is not readable until seq is published below.
                let prev = Interlocked.CompareExchange(&enqPos, pos + 1L, pos)

                if prev = pos then
                    Volatile.Write(&keys.[i], key)
                    Volatile.Write(&seqs.[i], pos + 1L)
                    ok <- true
                else
                    pos <- Volatile.Read(&enqPos)
            elif dif < 0L then
                doneTrying <- true // full at this producer position
            else
                pos <- Volatile.Read(&enqPos)

            spins <- spins + 1

        ok

    /// Try to dequeue a stealable work unit. Returns `Some key` on success or `None` when the queue is empty
    /// or contention wins the bounded retry budget.
    member _.TryPop() : uint64 option =
        let mutable pos = Volatile.Read(&deqPos)
        let mutable result = 0UL
        let mutable ok = false
        let mutable doneTrying = false
        let mutable spins = 0

        while not ok && not doneTrying && spins < 64 do
            let i = int (pos &&& int64 mask)
            let seq = Volatile.Read(&seqs.[i])
            let target = pos + 1L
            let dif = seq - target

            if dif = 0L then
                // Reserve this consumer position first. The slot is not writable again until seq is advanced.
                let prev = Interlocked.CompareExchange(&deqPos, pos + 1L, pos)

                if prev = pos then
                    result <- Volatile.Read(&keys.[i])
                    Volatile.Write(&seqs.[i], pos + int64 cap)
                    ok <- true
                else
                    pos <- Volatile.Read(&deqPos)
            elif dif < 0L then
                if pos >= Volatile.Read(&enqPos) then
                    doneTrying <- true // empty at this consumer position
                else
                    pos <- Volatile.Read(&deqPos)
            else
                pos <- Volatile.Read(&deqPos)

            spins <- spins + 1

        if ok then Some result else None

    // Test-only probes.
    member internal _.RawKeys: uint64[] = keys
    member internal _.RawSeqs: int64[] = seqs
    member internal _.EnqueuePos: int64 = Volatile.Read(&enqPos)
    member internal _.DequeuePos: int64 = Volatile.Read(&deqPos)

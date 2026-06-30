/// Phase 3 - Vyukov bounded MPMC queue tests. Covers sequential producer/consumer behavior, concurrent
/// producer/consumer stress, FIFO order under sequential ownership, and full/empty best-effort boundaries.
module Eonego.Tests.DagWorkQueueTests

open System.Threading.Tasks
open Xunit
open Eonego.DagWorkQueue

[<Fact>]
let ``capacity rounds up to a power of two`` () =
    let q1 = DagWorkQueue(3)
    let q2 = DagWorkQueue(256)
    let q3 = DagWorkQueue(1023)
    Assert.Equal(4, q1.Capacity)
    Assert.Equal(256, q2.Capacity)
    Assert.Equal(1024, q3.Capacity)

[<Fact>]
let ``single producer + single consumer FIFO`` () =
    let q = DagWorkQueue(8)
    for i in 0 .. 6 do
        Assert.True(q.TryPush(uint64 i))
    Assert.Equal(7, q.Count)

    for i in 0 .. 6 do
        match q.TryPop() with
        | Some k -> Assert.Equal(uint64 i, k)
        | None -> Assert.Fail(sprintf "TryPop returned None for i=%d" i)
    Assert.Equal(0, q.Count)

[<Fact>]
let ``slot sequence is published after payload write and after payload read`` () =
    let q = DagWorkQueue(2)
    Assert.True(q.TryPush 0xFEEDUL)
    Assert.Equal(1L, q.RawSeqs.[0])
    match q.TryPop() with
    | Some k -> Assert.Equal(0xFEEDUL, k)
    | None -> Assert.Fail("TryPop returned None on a single-element queue")
    Assert.Equal(2L, q.RawSeqs.[0])

[<Fact>]
let ``empty queue TryPop returns None`` () =
    let q = DagWorkQueue(4)
    Assert.Equal(None, q.TryPop())

[<Fact>]
let ``full queue TryPush returns false`` () =
    let q = DagWorkQueue(4)
    for i in 0 .. 3 do
        Assert.True(q.TryPush(uint64 i))

    Assert.False(q.TryPush(0xFFFFFFFFUL))
    Assert.Equal(4, q.Count)

[<Fact>]
let ``FIFO order around the wrap`` () =
    let q = DagWorkQueue(4)
    for round in 0 .. 15 do
        Assert.True(q.TryPush(uint64 (round * 2)))
        Assert.True(q.TryPush(uint64 (round * 2 + 1)))
        match q.TryPop() with
        | Some k -> Assert.Equal(uint64 (round * 2), k)
        | None -> Assert.Fail(sprintf "round %d: pop1 returned None" round)
        match q.TryPop() with
        | Some k -> Assert.Equal(uint64 (round * 2 + 1), k)
        | None -> Assert.Fail(sprintf "round %d: pop2 returned None" round)
        Assert.Equal(0, q.Count)

[<Fact>]
let ``concurrent 4P / 4C stress: no ops lost or duplicated`` () =
    // Keys are deliberately never zero. A stale read of a default payload is therefore caught as garbage.
    let q = DagWorkQueue(256)
    let perProducer = 256

    let pushed =
        [| for tid in 0 .. 3 ->
               Task.Run(fun () ->
                   let mutable lost = 0
                   for i in 0 .. perProducer - 1 do
                       let key = (uint64 (tid + 1) <<< 32) ||| uint64 (i + 1)
                       if not (q.TryPush key) then
                           lost <- lost + 1
                   lost) |]

    let popped =
        [| for _ in 0 .. 3 ->
               Task.Run(fun () ->
                   let bag = System.Collections.Generic.List<uint64>()
                   let mutable spins = 0
                   while (bag.Count < perProducer) && spins < 4_000_000 do
                       match q.TryPop() with
                       | Some k -> bag.Add k
                       | None -> spins <- spins + 1
                   bag) |]

    let lostPushes = pushed |> Array.sumBy (fun t -> t.Result)
    for t in pushed do
        t.Wait()

    let perConsumer = popped |> Array.map (fun t -> t.Result)
    for t in popped do
        t.Wait()

    let totalCount = perConsumer |> Array.sumBy (fun b -> b.Count)
    Assert.True(totalCount + lostPushes = 4 * perProducer, sprintf "lost = %d, popped = %d (total expected %d)" lostPushes totalCount (4 * perProducer))

    let seen = System.Collections.Generic.HashSet<uint64>()
    for bag in perConsumer do
        for k in bag do
            Assert.NotEqual(0UL, k)
            let tid = int (k >>> 32) - 1
            let i = int (k &&& 0xFFFFFFFFUL) - 1
            Assert.InRange(tid, 0, 3)
            Assert.InRange(i, 0, perProducer - 1)
            Assert.True(seen.Add k, sprintf "duplicate popped key tid=%d i=%d" tid i)

[<Fact>]
let ``TryPush/TryPop are safe to interleave on the same thread`` () =
    let q = DagWorkQueue(4)
    Assert.True(q.TryPush 100UL)
    match q.TryPop() with
    | Some k -> Assert.Equal(100UL, k)
    | None -> Assert.Fail("TryPop returned None on a single-element queue")
    Assert.True(q.TryPush 101UL)
    match q.TryPop() with
    | Some k -> Assert.Equal(101UL, k)
    | None -> Assert.Fail("second TryPop returned None")

/// Phase 2 - DAG node table tests.
///
/// Two layers:
///   1. Pure-table unit tests: pack round-trip, tokenized claim/complete/cancel, generation reset, and
///      concurrent same-key ownership.
///   2. Integration through `searchToDepthNet`: score + PV parity DAG-on vs DAG-off on the golden FEN set.
module Eonego.Tests.DagNodeTests

open System
open System.Threading
open System.Threading.Tasks
open Xunit
open Eonego.Move
open Eonego.Transposition
open Eonego.Search
open Eonego.DagNode

// ---------------------------------------------------------------------------
// 1. Pure-table unit tests.
// ---------------------------------------------------------------------------
let private assertClaim (dag: DagNodeTable) (key: uint64) (alpha: int) (beta: int) (depth: int) : int =
    let claim = dag.TryClaim key alpha beta depth
    Assert.NotEqual(NoClaim, claim)
    claim

[<Theory>]
[<InlineData(0x1F3F, 100, -50, 60, 7, 3)>]
[<InlineData(0x0041, -12345, -100, 100, 8, 0)>]
[<InlineData(0, 0, -127, 127, 0, 1)>]
let ``packData round-trips every field`` (move: int) (score: int) (alpha: int) (beta: int) (depth: int) (gen: int) =
    let d = packData StatusDone move score alpha beta depth gen
    Assert.Equal(StatusDone, dStatus d)
    Assert.Equal(move, dMove d)
    Assert.Equal(score, dScore d)
    Assert.Equal(alpha, dAlpha d)
    Assert.Equal(beta, dBeta d)
    Assert.Equal(depth, dDepth d)
    Assert.Equal(gen, dGen d)

[<Fact>]
let ``clampWindowEdge returns WideSentinel outside int8 representable range`` () =
    Assert.Equal(-127, clampWindowEdge -127)
    Assert.Equal(127, clampWindowEdge 127)
    Assert.Equal(WideSentinel, clampWindowEdge -128)
    Assert.Equal(WideSentinel, clampWindowEdge 128)
    Assert.Equal(WideSentinel, clampWindowEdge Int32.MaxValue)
    Assert.Equal(WideSentinel, clampWindowEdge Int32.MinValue)

[<Fact>]
let ``windowContains is inclusive on both edges and rejects WideSentinel`` () =
    Assert.True(windowContains -50 60 -50 60)
    Assert.True(windowContains -50 60 -10 20)
    Assert.False(windowContains -50 60 -60 70)
    Assert.False(windowContains -50 60 -51 60)
    Assert.False(windowContains WideSentinel 60 -50 60)
    Assert.False(windowContains -50 WideSentinel -50 60)
    Assert.False(windowContains WideSentinel WideSentinel -50 60)

[<Fact>]
let ``claim EMPTY slot then probe reads Expanding`` () =
    let dag = DagNodeTable(1)
    let key = 0xDEADBEEFCAFEBABEUL
    let _ = assertClaim dag key -50 60 7
    let struct (st, _, sc, da, db, dd) = dag.Probe key
    Assert.Equal(StatusExpanding, st)
    Assert.Equal(0, sc)
    Assert.True(windowContains da db -50 60)
    Assert.Equal(7, dd)

[<Fact>]
let ``claim twice on the same key: second returns NoClaim`` () =
    let dag = DagNodeTable(1)
    let key = 0xAAAA1111BBBB2222UL
    let _ = assertClaim dag key -100 100 8
    Assert.Equal(NoClaim, dag.TryClaim key -100 100 8)

[<Fact>]
let ``concurrent claim of the same key admits one owner`` () =
    let dag = DagNodeTable(1)
    let key = 0xBADC0FFEE0DDF00DUL
    let workers = 8
    use barrier = new Barrier(workers)

    let tasks =
        [| for _ in 1 .. workers ->
               Task.Run(fun () ->
                   barrier.SignalAndWait()
                   dag.TryClaim key -10 10 4) |]

    for t in tasks do
        t.Wait()
    let claims = tasks |> Array.map (fun t -> t.Result) |> Array.filter ((<>) NoClaim)
    Assert.Equal(1, claims.Length)

[<Fact>]
let ``busy cluster reservation makes TryClaim fail without modifying slots`` () =
    let dag = DagNodeTable(1)
    let key = 0xCAFEBABE12345678UL
    let r = dag.ReservationIndex key
    dag.RawReservations.[r] <- 1
    Assert.Equal(NoClaim, dag.TryClaim key -10 10 4)
    let struct (st, _, _, _, _, _) = dag.Probe key
    Assert.Equal(StatusEmpty, st)
    dag.RawReservations.[r] <- 0
    let _ = assertClaim dag key -10 10 4
    ()

[<Fact>]
let ``busy cluster reservation makes Complete and Cancel fail without blocking`` () =
    let dag = DagNodeTable(1)
    let key = 0xD00DFEEDABCDEF01UL
    let claim = assertClaim dag key -20 20 5
    let r = dag.ReservationIndex key
    dag.RawReservations.[r] <- 1
    Assert.False(dag.Complete claim key 0x1234 77 -20 20 5)
    Assert.False(dag.Cancel(claim, key))
    let struct (stBusy, _, _, _, _, _) = dag.Probe key
    Assert.Equal(StatusExpanding, stBusy)
    dag.RawReservations.[r] <- 0
    Assert.True(dag.Cancel(claim, key))
    let struct (st, _, _, _, _, _) = dag.Probe key
    Assert.Equal(StatusEmpty, st)
[<Fact>]
let ``complete transitions only the claimed slot to Done with final move+score`` () =
    let dag = DagNodeTable(1)
    let key = 0x0123456789ABCDEFUL
    let claim = assertClaim dag key -50 60 7
    Assert.True(dag.Complete claim key 0x1ABC 1500 -50 60 7)
    let struct (st, mv, sc, da, db, dd) = dag.Probe key
    Assert.Equal(StatusDone, st)
    Assert.Equal(0x1ABC, mv)
    Assert.Equal(1500, sc)
    Assert.Equal(7, dd)
    Assert.True(windowContains da db -50 60)

[<Fact>]
let ``complete with wrong token fails and leaves node Expanding`` () =
    let dag = DagNodeTable(1)
    let key = 0x1200340056007800UL
    let claim = assertClaim dag key -25 25 3
    Assert.False(dag.Complete (claim + 2) key 0x2222 99 -25 25 3)
    let struct (st, _, _, _, _, _) = dag.Probe key
    Assert.Equal(StatusExpanding, st)

[<Fact>]
let ``cancel releases an abandoned claim`` () =
    let dag = DagNodeTable(1)
    let key = 0x0BADF00D12345678UL
    let claim = assertClaim dag key -20 20 5
    Assert.True(dag.Cancel(claim, key))
    let struct (st, _, _, _, _, _) = dag.Probe key
    Assert.Equal(StatusEmpty, st)
    let _ = assertClaim dag key -20 20 5
    ()

[<Fact>]
let ``Done result feeds back a probe cutoff-eligible bound`` () =
    let dag = DagNodeTable(1)
    let key = 0xF1F2F3F4F5F6F7F8UL
    let claim = assertClaim dag key -100 100 5
    Assert.True(dag.Complete claim key 0xBEEF 42 -100 100 5)
    let struct (st, _, _, da, db, dd) = dag.Probe key
    Assert.Equal(StatusDone, st)
    Assert.True(windowContains da db -50 60)
    Assert.Equal(5, dd)

[<Fact>]
let ``NewSearch invalidates old Done entries and permits a fresh claim`` () =
    let dag = DagNodeTable(1)
    let key = 0x1234567890ABCDEFUL
    let claim = assertClaim dag key 0 50 6
    Assert.True(dag.Complete claim key 0x1111 12 0 50 6)
    dag.NewSearch()
    let struct (st, _, _, _, _, _) = dag.Probe key
    Assert.Equal(StatusEmpty, st)
    let _ = assertClaim dag key 0 50 6
    ()

[<Fact>]
let ``Clear empties every slot`` () =
    let dag = DagNodeTable(1)
    let key = 0x1234567890ABCDEFUL
    let _ = assertClaim dag key 0 50 6
    dag.Clear()
    let struct (st, _, _, _, _, _) = dag.Probe key
    Assert.Equal(StatusEmpty, st)

[<Fact>]
let ``two distinct keys in the same cluster: both claim independently when slots are free`` () =
    let dag = DagNodeTable(1)
    let keyA = 0xDEADBEEFCAFEBABEUL
    let keyB = 0x0123456789ABCDEFUL
    let claimA = assertClaim dag keyA -50 50 6
    let claimB = assertClaim dag keyB -40 40 4
    Assert.True(dag.Complete claimA keyA 0x1111 10 -50 50 6)
    Assert.True(dag.Complete claimB keyB 0x2222 20 -40 40 4)
    let struct (stA, mvA, _, _, _, _) = dag.Probe keyA
    Assert.Equal(StatusDone, stA)
    Assert.Equal(0x1111, mvA)
    let struct (stB, mvB, _, _, _, _) = dag.Probe keyB
    Assert.Equal(StatusDone, stB)
    Assert.Equal(0x2222, mvB)

[<Fact>]
let ``probe of a key not stored misses with StatusEmpty`` () =
    let dag = DagNodeTable(1)
    let struct (st, _, _, _, _, _) = dag.Probe 0xCAFED00DDEADBEEFUL
    Assert.Equal(StatusEmpty, st)

// ---------------------------------------------------------------------------
// 2. Integration: searchToDepthNet parity, DAG-on vs DAG-off.
// ---------------------------------------------------------------------------
let private goldenFens =
    [| "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
       "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
       "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"
       "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3" |]

let private cfgOff =
    { defaultConfig with
        Threads = 1
        UseTt = true
        UsePruning = true
        DagHashMb = 0 }

let private cfgOn =
    { defaultConfig with
        Threads = 1
        UseTt = true
        UsePruning = true
        DagHashMb = 2 }

[<Theory>]
[<InlineData(0, 8)>]
[<InlineData(1, 8)>]
[<InlineData(2, 8)>]
[<InlineData(3, 7)>]
let ``DAG-on score == DAG-off score at depth on golden FEN set`` (fenIndex: int) (depth: int) =
    let fen = goldenFens.[fenIndex]
    let struct (sOff, _, mvOff) = searchToDepthNet fen [||] depth cfgOff None
    let struct (sOn, _, mvOn) = searchToDepthNet fen [||] depth cfgOn None
    Assert.Equal(sOff, sOn)
    Assert.Equal(mvOff, mvOn)

[<Fact>]
let ``DAG-on completes a node-budget search and returns a legal move (smoke)`` () =
    let fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"

    let struct (_, _, mvOff) = searchToNodesNet fen [||] 100_000L cfgOff None
    let struct (sOn, nOn, mvOn) = searchToNodesNet fen [||] 100_000L cfgOn None

    Assert.True(nOn > 0L, "DAG-on search must explore at least one node")
    Assert.True(mvOff <> MoveNone, "DAG-off best move must be legal (not MoveNone)")
    Assert.True(mvOn <> MoveNone, "DAG-on best move must be legal (not MoveNone)")
    Assert.True(sOn > -INF && sOn < INF, "DAG-on root score must be in-range")



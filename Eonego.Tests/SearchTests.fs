/// Search correctness: the pruning-off/TT-off oracle (== an in-test minimax that shares the engine's
/// qsearch leaf), single-thread node determinism, TT invariance, mate suites, qsearch/eval consistency,
/// and the mate-ply TT round-trip.
module Eonego.Tests.SearchTests

open Xunit
open Eonego.Move
open Eonego.Position
open Eonego.Evaluation
open Eonego.Transposition
open Eonego.Search
open Eonego.Tests.TestFixtures

let private oracleCfg =
    { defaultConfig with
        UseTt = false
        UsePruning = false
        Threads = 1 }

let private makeWorker (fen: string) (cfg: SearchConfig) : Worker =
    let tt = TranspositionTable(max 1 cfg.HashMb)
    let control = SearchControl(cfg, defaultLimits, tt, fen, [||])
    let w = Worker(0, true, control)
    w.SetupRoot()
    control.Reset()
    control.StartClock 0L 0L
    w

/// Reference: exhaustive negamax (NO alpha-beta, NO TT, NO ordering) that tops each depth-0 node with the
/// engine's OWN qsearch (full window) and uses identical draw/terminal/mate semantics. Full-window fail-soft
/// alpha-beta must return the same score.
let rec private refMinimax (w: Worker) (pos: Position) (depth: int) (ply: int) : int =
    if ply > 0 && isImmediateDraw pos then
        0
    elif depth = 0 then
        qsearch w pos (-INF) INF ply
    elif ply >= MaxSearchPly then
        (if pos.InCheck then 0 else eval pos)
    else
        let moves = collectLegal pos

        if moves.Length = 0 then
            (if pos.InCheck then -MATE + ply else 0)
        else
            let mutable best = -INF

            for m in moves do
                pos.Make m
                let v = -(refMinimax w pos (depth - 1) (ply + 1))
                pos.Unmake m

                if v > best then
                    best <- v

            best

[<Theory>]
[<InlineData("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 3)>]
[<InlineData("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 2)>]
[<InlineData("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 3)>]
[<InlineData("8/8/4k3/8/8/4K3/4P3/8 w - - 0 1", 4)>]
let ``oracle: pruning-off TT-off negamax score equals minimax`` (fen: string) (depth: int) =
    let struct (engine, _, _) = searchToDepth fen [||] depth oracleCfg
    let w = makeWorker fen oracleCfg
    let refScore = refMinimax w w.Pos depth 0
    Assert.Equal(refScore, engine)

[<Fact>]
let ``single-thread node count is deterministic`` () =
    let struct (_, n1, _) = searchToDepth StartPosFen [||] 5 oracleCfg
    let struct (_, n2, _) = searchToDepth StartPosFen [||] 5 oracleCfg
    Assert.Equal(n1, n2)

[<Theory>]
[<InlineData("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 4)>]
[<InlineData("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 5)>]
let ``TT-on score equals TT-off score at fixed depth (pruning off)`` (fen: string) (depth: int) =
    let cfgOff =
        { defaultConfig with
            UseTt = false
            UsePruning = false }

    let cfgOn =
        { defaultConfig with
            UseTt = true
            UsePruning = false }

    let struct (s1, _, _) = searchToDepth fen [||] depth cfgOff
    let struct (s2, _, _) = searchToDepth fen [||] depth cfgOn
    Assert.Equal(s1, s2)

[<Fact>]
let ``mate in one is found with the correct ply-adjusted score`` () =
    let struct (score, _, m) =
        searchToDepth "k7/8/1K6/8/8/8/8/7R w - - 0 1" [||] 2 defaultConfig

    Assert.True(score >= MATE_IN_MAX_PLY)
    Assert.Equal(MATE - 1, score)
    Assert.Equal("h1h8", toUci m)

[<Fact>]
let ``mate in two is found with the correct ply-adjusted score`` () =
    // Two mates-in-2 exist here (1.Kg6 Kg8 2.Ra8# and 1.Kf7 Kh7 2.Rh1#) — assert the ply-adjusted mate
    // SCORE (mate in 2 = 3 plies), not a specific first move.
    let struct (score, _, m) =
        searchToDepth "7k/8/5K2/8/8/8/8/R7 w - - 0 1" [||] 6 defaultConfig

    Assert.True(score >= MATE_IN_MAX_PLY)
    Assert.Equal(MATE - 3, score)
    Assert.NotEqual(MoveNone, m)

[<Fact>]
let ``qsearch on a quiet position equals static eval`` () =
    let cfg =
        { defaultConfig with
            UseTt = false
            UsePruning = false }

    let w = makeWorker "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1" cfg
    Assert.Equal(eval w.Pos, qsearch w w.Pos (-INF) INF 0)

[<Theory>]
[<InlineData(0)>]
[<InlineData(1)>]
[<InlineData(50)>]
[<InlineData(245)>]
let ``mate score round-trips through store and probe at every ply`` (ply: int) =
    let tt = TranspositionTable(1)
    let key = 0xABCDEF1234567890UL
    let vWin = MATE - 7
    let vLose = -MATE + 7
    tt.Store key 4 BoundExact (valueToTt vWin ply) 0 0x10
    let struct (h1, _, sc1, _, _, _) = tt.Probe key
    Assert.True h1
    Assert.Equal(vWin, valueFromTt sc1 ply)
    tt.Store key 4 BoundExact (valueToTt vLose ply) 0 0x11
    let struct (h2, _, sc2, _, _, _) = tt.Probe key
    Assert.True h2
    Assert.Equal(vLose, valueFromTt sc2 ply)

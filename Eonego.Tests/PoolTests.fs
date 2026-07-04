/// Worker-pool path (goPooled/goPooledArmed): consecutive searches through one pool must return
/// legal moves, and the warm gravity-history tables must actually flow across searches — the pool's
/// entire point for LazySMP divergence (the fresh-worker default zeroes every table per `go`).
/// First-search equivalence with the fresh path was verified end-to-end over UCI (identical
/// 126,995-node depth-12 startpos searches, 2026-07-04): keepHistory=true on brand-new zeroed
/// tables is state-identical to Clear().
module Eonego.Tests.PoolTests

open Xunit
open Eonego.Move
open Eonego.Position
open Eonego.Transposition
open Eonego.Search
open Eonego.Tests.TestFixtures

let private AfterE4E5 = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"

let private mkControl (tt: TranspositionTable) (fen: string) (depth: int) (threads: int) =
    let cfg =
        { defaultConfig with
            Threads = threads
            UseRetro = false }

    SearchControl(cfg, { defaultLimits with Depth = depth }, tt, fen, [||])

let private legalIn (fen: string) (m: Move) =
    collectLegal (Position.OfFen fen) |> Array.contains m

[<Fact>]
let ``pooled search returns legal moves across consecutive searches (2 threads)`` () =
    let tt = TranspositionTable(8) // persists across the two controls, like UCIState.Tt
    let c1 = mkControl tt StartPosFen 8 2
    let pool = Array.init 2 (fun i -> Worker(i, (i = 0), c1))
    let m1 = goPooled c1 pool
    Assert.True(legalIn StartPosFen m1, "first pooled search: illegal move " + toUci m1)
    // New control (new fen, same TT), same pool — the UCI per-move flow.
    let c2 = mkControl tt AfterE4E5 8 2
    let m2 = goPooled c2 pool
    Assert.True(legalIn AfterE4E5 m2, "second pooled search: illegal move " + toUci m2)

[<Fact>]
let ``pooled worker keeps warm history that reorders the next search`` () =
    // Warm arm: the pool searches startpos first, then AfterE4E5 with a FRESH TT — isolating the
    // history tables as the only carried-over state. Cold arm: a new worker, same fen, fresh TT.
    // 1T fixed-depth searches are deterministic; the node inequality is measured-then-pinned — if
    // it ever becomes equality, keepHistory stopped carrying the tables across searches.
    let cWarm1 = mkControl (TranspositionTable(8)) StartPosFen 10 1
    let pool = [| Worker(0, true, cWarm1) |]
    goPooled cWarm1 pool |> ignore
    let cWarm2 = mkControl (TranspositionTable(8)) AfterE4E5 10 1
    let mWarm = goPooled cWarm2 pool
    let nWarm = pool.[0].Nodes
    let cCold = mkControl (TranspositionTable(8)) AfterE4E5 10 1
    let cold = [| Worker(0, true, cCold) |]
    let mCold = goPooled cCold cold
    let nCold = cold.[0].Nodes
    Assert.True(legalIn AfterE4E5 mWarm, "warm pooled search: illegal move " + toUci mWarm)
    Assert.True(legalIn AfterE4E5 mCold, "cold search: illegal move " + toUci mCold)
    Assert.NotEqual(nCold, nWarm)

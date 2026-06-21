/// LazySMP: at fixed depth with pruning OFF, the TT only speeds the search, so N threads must return the
/// SAME score as a single thread, never crash, all join, and emit a legal best move. (nps scaling and
/// 0-B/op are benchmark concerns, not flaky xUnit assertions.)
module Eonego.Tests.LazySmpTests

open Xunit
open Eonego.Move
open Eonego.Position
open Eonego.Transposition
open Eonego.Search
open Eonego.Tests.TestFixtures

let private runFixed (fen: string) (depth: int) (threads: int) : struct (int * Move) =
    let cfg =
        { defaultConfig with
            Threads = threads
            UseTt = true
            UsePruning = false }

    let tt = TranspositionTable(16)

    let control =
        SearchControl(cfg, { defaultLimits with Depth = depth }, tt, fen, [||])

    let best = go control
    struct (control.LastScore, best)

[<Fact>]
let ``multi-thread pruning-off search matches single-thread score and returns a legal move`` () =
    let fen = StartPosFen
    let depth = 6
    let struct (s1, m1) = runFixed fen depth 1
    let struct (s2, m2) = runFixed fen depth 2
    let struct (s4, m4) = runFixed fen depth 4
    Assert.Equal(s1, s2)
    Assert.Equal(s1, s4)
    let legal = collectLegal (Position.OfFen fen)
    Assert.Contains(m1, legal)
    Assert.Contains(m2, legal)
    Assert.Contains(m4, legal)

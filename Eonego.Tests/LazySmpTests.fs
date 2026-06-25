/// LazySMP sanity. Single-thread pruning-off search is the minimax oracle and is fully DETERMINISTIC (one
/// writer/reader on the TT). Multi-thread shares the lockless TT, so an exact-bound entry written by ANOTHER
/// thread at a DEEPER depth can be grafted through a cutoff (negamax searches non-first moves as non-PV at a
/// full window — Search.fs:962 — and a `not isPv && dp >= depthIn` BoundExact entry cuts off there,
/// Search.fs:656). That makes the multi-thread score legitimately differ from single-thread: a known property
/// of shared-TT parallel search, NOT a bug. So exact cross-thread score equality is NOT an invariant and must
/// not be asserted (it was the documented ~1/17-under-load flake). We assert only what holds: the search
/// completes, single-thread is reproducible, and every thread count returns a legal move with a sane score.
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
let ``single-thread pruning-off search is deterministic`` () =
    let fen = StartPosFen
    let depth = 6
    // One writer/reader on the TT => byte-identical score AND move across runs.
    let struct (s1, m1) = runFixed fen depth 1
    let struct (s2, m2) = runFixed fen depth 1
    Assert.Equal(s1, s2)
    Assert.Equal(m1, m2)
    Assert.Contains(m1, collectLegal (Position.OfFen fen))

[<Fact>]
let ``multi-thread pruning-off search returns a legal move and a sane score`` () =
    let fen = StartPosFen
    let depth = 6
    let legal = collectLegal (Position.OfFen fen)

    // NOTE: do NOT assert score == single-thread — shared-TT deeper-exact grafting makes it legitimately vary.
    for threads in [ 2; 4 ] do
        let struct (s, m) = runFixed fen depth threads
        Assert.Contains(m, legal) // completed, joined, emitted a legal best move
        Assert.True(abs s < 2000, sprintf "threads=%d score %d outside the sane startpos-depth6 range" threads s)

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

[<Fact>]
let ``Phase 3 Lazy-SMP scaling: 4T nps > 1T nps (sanity gate)`` () =
    // Phase 3 gate: with the Phase 1 acc-checkpoint cache + Phase 2 DAG table shared across all workers,
    // 4-thread Lazy-SMP must visit strictly MORE nodes than 1-thread in the SAME wall-clock budget. The
    // existing Lazy-SMP plumbing spawns N workers each running iterative-deepening; they share the lock-free
    // TT + acc-checkpoint + DagNodeTable, so one worker's findings propagate to the others' subsequent
    // iterations — even without fine-grained work-stealing. The bench-friendly `Threads`/`Nodes` time budget
    // asserts that downscaling-up IS observed. The aggressive 0.7× linear-to-8c threshold would require a
    // full task-parallel work-stealing rewrite (deferred per the pragmatic Phase 3 scope); this gate catches
    // the regression class where shared cache wiring is broken (no parallelism observable at all).
    let fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -" // Kiwipete (rich tree)
    let budgetMs = 300L

    let runNps (threads: int) : int64 =
        let cfg =
            { defaultConfig with
                Threads = threads
                UseTt = true
                UsePruning = true }

        let tt = TranspositionTable(64)
        let lim: SearchLimits = { defaultLimits with MoveTime = int budgetMs }
        let control = SearchControl(cfg, lim, tt, fen, [||])
        let _ = go control
        let elapsedMs = max 1L control.ElapsedMs
        // Final node sum across all workers (set by go's `control.NodeSum`) / elapsed (ms) * 1000 → nps.
        control.NodeSum () * 1000L / elapsedMs

    let nps1 = runNps 1
    let nps4 = runNps 4

    // Mailbox: 『4T nps must be at least 1.5x of 1T nps』 — a sane lazy-SMP bar that survives short-budget
    // scheduling jitter on the runner; higher bars require a deeper work-stealing rework (Phase 3b).
    Assert.True(nps4 * 2L >= nps1 * 3L, sprintf "Lazy-SMP 4T nps %d must be >= 1.5x of 1T nps %d" nps4 nps1)

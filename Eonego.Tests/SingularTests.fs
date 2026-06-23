/// Singular / double extension integration tests.
///
/// Like ProbCut/LMR, singular extensions are a *heuristic* (they only change WHICH depth a move is searched
/// to), so exact correctness is the job of the pruning-OFF oracle (`SearchTests`). These tests assert the
/// two behaviours SE must have: it never *hides* a known tactic, and it never blows up the tree (double
/// extensions stay capped). The exclusion plumbing (skip-the-move, dedicated buffers, no TT clobber) is
/// exercised indirectly by every full-stack search here and by the whole suite staying green.
module Eonego.Tests.SingularTests

open Xunit
open Eonego.Bitboard
open Eonego.Move
open Eonego.Search

let private cfgOn = defaultConfig                                  // UseSingular = true (default)
let private cfgOff = { defaultConfig with UseSingular = false }

// ---------------------------------------------------------------------------
// Tactic not hidden: a winning capture and a forced mate are still found with SE enabled.
// ---------------------------------------------------------------------------
[<Fact>]
let ``singular extension does not hide a winning capture`` () =
    let struct (score, _, m) = searchToDepth "4k3/8/8/3q4/4P3/8/8/3RK3 w - - 0 1" [||] 10 cfgOn
    Assert.Equal(mkSquare 3 4, toSq m) // e4xd5 takes the queen
    Assert.True(score > 500, "expected a decisive material win, got " + string score)

[<Fact>]
let ``singular extension does not hide a mate`` () =
    let struct (score, _, _) = searchToDepth "7k/8/5K2/8/8/8/8/R7 w - - 0 1" [||] 12 cfgOn
    Assert.True(score >= MATE_IN_MAX_PLY, "expected a mate score, got " + string score)

// ---------------------------------------------------------------------------
// Search-explosion guard: with SE on, the (capped) extensions must not run the tree away versus SE off.
// ---------------------------------------------------------------------------
[<Fact>]
let ``singular extension does not explode the node count`` () =
    let fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1" // Kiwipete
    let depth = 10
    let struct (_, nOn, _) = searchToDepth fen [||] depth cfgOn
    let struct (_, nOff, _) = searchToDepth fen [||] depth cfgOff
    Assert.True(nOn < 5L * nOff, "SE blew up the tree: nOn=" + string nOn + " nOff=" + string nOff)

// Deep-search regression: a *cross-ply* singular exclusion search (ply P's exclusion search recurses into a
// child that triggers its OWN exclusion search) must not share move buffers — the original single-buffer
// version corrupted the board and threw IndexOutOfRange in pos.Make at depth ~18. Depth 18 is needed: the
// halved exclusion depth (depth-1)/2 must stay >= 8 (the SE threshold) at the nested child. Must return a
// legal move without throwing.
[<Fact>]
let ``deep search with nested singular extensions does not corrupt the board`` () =
    let fen = "rnbqkb1r/pp2pppp/3p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R b KQkq - 0 5"
    let struct (_, _, m) = searchToDepth fen [||] 18 cfgOn
    Assert.NotEqual(MoveNone, m)

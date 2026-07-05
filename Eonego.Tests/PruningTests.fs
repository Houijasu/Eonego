/// Forward-pruning integration tests (IIR, razoring, history pruning, delta pruning).
///
/// IMPORTANT: like ProbCut/null-move/LMR these are *heuristic* accelerators — they can perturb the
/// backed-up score, so we do NOT assert score invariance between on and off. Exact correctness is
/// guaranteed by the pruning-OFF oracle (`SearchTests`, `negamax == minimax`), under which all four
/// are provably disabled (every one is gated on `usePruning`). These tests assert the two things the
/// heuristics must do: never *hide* a known tactic, and actually *reduce* nodes (they are accelerators).
module Eonego.Tests.PruningTests

open Xunit
open Eonego.Bitboard
open Eonego.Move
open Eonego.Search
open Eonego.Tests.TestFixtures

// cfgOn: all four new heuristics on (defaults). cfgOff: the four off, but the legacy pruning suite
// (RFP/null-move/LMP/futility/SEE/LMR/ProbCut) stays on — so node deltas isolate the new heuristics.
// UseRetro pinned OFF in both arms: retrograde values publish process-wide once any parallel test
// class solves a signature, and the queen-grab fixture reaches 3-man subtrees at depth 8 — its
// measured-then-pinned best-move tie-break must not depend on WHEN that publication happens.
let private cfgOn = { defaultConfig with UseRetro = false }

let private cfgOff =
    { defaultConfig with
        UseIir = false
        UseRazoring = false
        UseHistoryPruning = false
        UseDeltaPruning = false
        UseRetro = false }

// ---------------------------------------------------------------------------
// (a) Tactics not hidden — with the full pruning suite on, known wins/mates are still found.
// ---------------------------------------------------------------------------
[<Fact>]
let ``pruning does not hide a mate in one`` () =
    let struct (score, _, m) = searchToDepth "k7/8/1K6/8/8/8/8/7R w - - 0 1" [||] 4 cfgOn
    Assert.Equal(mkSquare 7 7, toSq m) // Rh1-h8#
    Assert.True(score >= MATE_IN_MAX_PLY, "expected a mate score, got " + string score)

[<Fact>]
let ``pruning does not hide a mate in two`` () =
    let struct (score, _, _) = searchToDepth "7k/8/5K2/8/8/8/8/R7 w - - 0 1" [||] 6 cfgOn
    Assert.True(score >= MATE_IN_MAX_PLY, "expected a mate score, got " + string score)

[<Fact>]
let ``pruning does not hide a winning capture`` () =
    match tryLoadNet () with
    | None -> () // soft-skip: NNUE net absent
    | Some net ->
        let struct (score, _, m) = searchToDepthNet "4k3/8/8/3q4/4P3/8/8/3RK3 w - - 0 1" [||] 8 cfgOn (Some net)
        Assert.Equal(mkSquare 3 4, toSq m) // exd5 captures the queen on d5
        Assert.True(score > 500, "expected a decisive material win, got " + string score)

// ---------------------------------------------------------------------------
// (b) Measured node reduction — the heuristics are accelerators, so the full suite must not search
// MORE nodes than the legacy suite on a representative middlegame. Measured-then-pinned (relation
// observed during implementation), à la ProbCut — NOT a score-invariance assertion.
// ---------------------------------------------------------------------------
[<Theory>]
[<InlineData("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 9)>] // Kiwipete
[<InlineData("r1bqkb1r/pp2pppp/2np1n2/2p5/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 0 1", 9)>] // quiet-ish
let ``new pruning does not increase nodes`` (fen: string) (depth: int) =
    let struct (_, nOn, _) = searchToDepth fen [||] depth cfgOn
    let struct (_, nOff, _) = searchToDepth fen [||] depth cfgOff
    Assert.True(nOn < nOff, "expected new pruning to strictly reduce nodes: nOn=" + string nOn + " nOff=" + string nOff)

// ---------------------------------------------------------------------------
// (c) Best-move agreement on clear positions (move only, not score — the score may shift a few cp).
// ---------------------------------------------------------------------------
[<Theory>]
[<InlineData("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 8)>] // Kiwipete
[<InlineData("4k3/8/8/3q4/4P3/8/8/3RK3 w - - 0 1", 8)>]                                    // queen grab
let ``new pruning keeps the same best move on clear positions`` (fen: string) (depth: int) =
    let struct (_, _, mOn) = searchToDepth fen [||] depth cfgOn
    let struct (_, _, mOff) = searchToDepth fen [||] depth cfgOff
    Assert.Equal(toUCI mOff, toUCI mOn)

/// ProbCut integration tests.
///
/// IMPORTANT: ProbCut is a *heuristic* forward-pruning technique — like null-move/LMR, it can slightly
/// perturb the backed-up score (a reduced-depth verification may cause a speculative cutoff). It is NOT
/// score-preserving the way the TT is, so we do NOT assert score invariance between ProbCut-on and -off.
/// Exact correctness is guaranteed by the pruning-OFF oracle (`SearchTests`, `negamax == minimax`), under
/// which ProbCut is provably disabled. These tests assert the two things ProbCut must do: never *hide* a
/// known tactic, and actually *reduce* nodes (it is an accelerator).
module Eonego.Tests.ProbCutTests

open Xunit
open Eonego.Bitboard
open Eonego.Move
open Eonego.Transposition
open Eonego.Search
open Eonego.Tests.TestFixtures

// Isolate ProbCut against the legacy search: hold the later forward-pruning heuristics (IIR/razoring/
// history/delta) OFF in both arms so these fixtures measure ProbCut alone. (The "same best move"
// fixtures are coincidental for any heuristic — turning the newer ones on can flip a tie-break between
// equally-scored moves, e.g. =Q vs =R that both win the queen. The full default stack is exercised by
// PruningTests instead.)
let private cfgBase =
    { defaultConfig with
        UseIir = false
        UseRazoring = false
        UseHistoryPruning = false
        UseDeltaPruning = false
        UseContHist = false
        UseSingular = false
        UseNmpVerify = false
        UseLmrTweaks = false
        UseAspTweaks = false
        UseQsTt = false
        UseTtEvalAdjust = false
        UseQsEvasionCap = false
        UseCorrHist = false }

let private cfgOn = cfgBase                                         // UseProbCut = true
let private cfgOff = { cfgBase with UseProbCut = false }

// ---------------------------------------------------------------------------
// Tactic not hidden: a genuinely winning capture (rook backs the pawn -> KR vs K, a real win, not a
// KPvK draw) must still be found, with a decisive score, when ProbCut is on.
// ---------------------------------------------------------------------------
[<Fact>]
let ``ProbCut does not hide a winning capture`` () =
    match tryLoadNet () with
    | None -> () // soft-skip: NNUE net absent
    | Some net ->
        let struct (score, _, m) = searchToDepthNet "4k3/8/8/3q4/4P3/8/8/3RK3 w - - 0 1" [||] 8 cfgOn (Some net)
        Assert.Equal(mkSquare 3 4, toSq m)                          // the move captures the queen on d5
        Assert.True(score > 500, "expected a decisive material win, got " + string score)

[<Fact>]
let ``ProbCut does not hide a mate`` () =
    // Mate in two is still found (and scored as a mate) with ProbCut enabled.
    let struct (score, _, _) = searchToDepth "7k/8/5K2/8/8/8/8/R7 w - - 0 1" [||] 6 cfgOn
    Assert.True(score >= MATE_IN_MAX_PLY, "expected a mate score, got " + string score)

// ---------------------------------------------------------------------------
// ProbCut fires & saves: strict node reduction on a capture-rich position (Kiwipete), and here it keeps
// the same best move. (Same-move is not guaranteed in general for a heuristic, but holds on this fixture.)
// ---------------------------------------------------------------------------
[<Fact>]
let ``ProbCut reduces nodes on a capture-rich position`` () =
    match tryLoadNet () with
    | None -> () // soft-skip: NNUE net absent
    | Some net ->
        let fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"   // Kiwipete
        let depth = 9
        let struct (_, nOn, mOn) = searchToDepthNet fen [||] depth cfgOn (Some net)
        let struct (_, nOff, mOff) = searchToDepthNet fen [||] depth cfgOff (Some net)
        Assert.True(nOn < nOff, "expected ProbCut to cut nodes: nOn=" + string nOn + " nOff=" + string nOff)
        Assert.Equal(toUci mOff, toUci mOn)

// ---------------------------------------------------------------------------
// Best-move agreement on a few clear positions at fixed depth: ProbCut should not change the chosen move
// on these (the score may shift by a few cp, which is why we compare the MOVE, not the score).
// ---------------------------------------------------------------------------
[<Theory>]
[<InlineData("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 8)>]   // Kiwipete
[<InlineData("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 0 1", 8)>]              // tactical
[<InlineData("rnbqkb1r/pppp1ppp/4pn2/8/2PP4/6P1/PP2PP1P/RNBQKBNR b KQkq - 0 1", 9)>]        // opening line
let ``ProbCut keeps the same best move on clear positions`` (fen: string) (depth: int) =
    let struct (_, _, mOn) = searchToDepth fen [||] depth cfgOn
    let struct (_, _, mOff) = searchToDepth fen [||] depth cfgOff
    Assert.Equal(toUci mOff, toUci mOn)

// ---------------------------------------------------------------------------
// Mate score round-trips through a BoundLower store (the bound ProbCut writes on a cutoff).
// ---------------------------------------------------------------------------
[<Theory>]
[<InlineData(0)>]
[<InlineData(7)>]
[<InlineData(120)>]
let ``mate score round-trips through a BoundLower ProbCut store`` (ply: int) =
    let tt = TranspositionTable(1)
    let key = 0x51A2B3C4D5E6F708UL
    let vWin = MATE - 11
    tt.Store key 6 BoundLower (valueToTt vWin ply) 0 0x123 false
    let struct (hit, _, sc, _, _, bd, _) = tt.Probe key
    Assert.True hit
    Assert.Equal(BoundLower, bd)
    Assert.Equal(vWin, valueFromTt sc ply)

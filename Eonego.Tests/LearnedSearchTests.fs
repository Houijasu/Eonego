/// LearnedSearch (Phase A) correctness: the full-expansion backup must equal an explicit NNUE-leaf
/// minimax (refLearnedMinimax — NOT the qsearch-leaf Search oracle); terminal-at-creation (mate/stalemate
/// scored at child creation, not as a stub-eval leaf); draw semantics; legal bestmove; node determinism;
/// arena-cap safety. Net-free tests inject a deterministic stub leaf-eval; net-gated tests soft-skip.
module Eonego.Tests.LearnedSearchTests

open Xunit
open Eonego.Move
open Eonego.Position
open Eonego.MoveGeneration
open Eonego.Nnue
open Eonego.Search
open Eonego.LearnedSearch
open Eonego.Tests.TestFixtures

let private startFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
let private mate1Fen = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1" // Ra8#
let private stalemateFen = "7k/8/6Q1/8/8/8/8/6K1 b - - 0 1" // black is stalemated
let private kvkFen = "8/8/8/4k3/8/4K3/8/8 w - - 0 1" // insufficient material ⇒ draw
let private midFen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1"

let private cap = 50_000 // ample for these shallow trees; small so parallel test runs stay light on memory

/// Deterministic, bounded (|.| ≤ 10000, well clear of mate thresholds) leaf eval — the net-free stub.
let private stubEval (pos: Position) : int = int (pos.Key % 20001UL) - 10000

/// Explicit NNUE-leaf negamax with the SAME draw/terminal/mate rules as the arena (draw checked at
/// ply>0, then mate/stalemate, then the depth cutoff). This is the Phase-A oracle.
let rec private refLM (pos: Position) (ply: int) (maxDepth: int) (eval: Position -> int) : int =
    if ply > 0 && isImmediateDraw pos then
        0
    else
        let buf = Array.zeroCreate<Move> 256
        let n = generateLegal pos (System.Span<Move>(buf))

        if n = 0 then
            if pos.InCheck then -MATE + ply else 0
        elif ply >= maxDepth then
            eval pos
        else
            let mutable best = System.Int32.MinValue

            for i in 0 .. n - 1 do
                let m = buf.[i]
                pos.Make m
                let v = -(refLM pos (ply + 1) maxDepth eval)
                pos.Unmake m
                if v > best then best <- v

            best

let private legalMoves (fen: string) : Move[] =
    let p = Position()
    p.LoadFen fen
    let buf = Array.zeroCreate<Move> 256
    let n = generateLegal p (System.Span<Move>(buf))
    buf.[0 .. n - 1]

// (1) THE ORACLE: full-expansion backup == explicit NNUE-leaf minimax (stub eval), across positions/depths.
[<Theory>]
[<InlineData("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 1)>]
[<InlineData("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 2)>]
[<InlineData("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 3)>]
[<InlineData("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1", 2)>]
[<InlineData("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", 2)>]
[<InlineData("8/8/8/4k3/8/4K3/8/8 w - - 0 1", 2)>]
let ``full expansion equals explicit minimax`` (fen: string) (depth: int) =
    let struct (lsScore, _, _) = searchToDepthEval fen [||] depth cap None stubEval
    let refPos = Position()
    refPos.LoadFen fen
    let refScore = refLM refPos 0 depth stubEval
    Assert.Equal(refScore, lsScore)

// (2) Terminal-at-creation: mate-in-1 is found at depth 1 (the mate child is scored at creation,
// not as a stub leaf — the stub is bounded to ±10000, far below the mate score).
[<Fact>]
let ``mate in one is found at depth 1`` () =
    let struct (score, _, move) = searchToDepthEval mate1Fen [||] 1 cap None stubEval
    Assert.True(score >= MATE_IN_MAX_PLY, "expected a mate score, got " + string score)
    Assert.Equal("a1a8", toUci move)

// (3) Stalemate at the root ⇒ score 0.
[<Fact>]
let ``stalemate scores zero`` () =
    let struct (score, _, _) = searchToDepthEval stalemateFen [||] 1 cap None stubEval
    Assert.Equal(0, score)

// (4) Insufficient material ⇒ draw (children are all draws ⇒ backed-up 0).
[<Fact>]
let ``insufficient material is a draw`` () =
    let struct (score, _, _) = searchToDepthEval kvkFen [||] 2 cap None stubEval
    Assert.Equal(0, score)

// (5) Best-first plays a legal move.
[<Fact>]
let ``best-first returns a legal move`` () =
    let struct (_, _, move) = searchToNodesEval startFen [||] 2000L cap None stubEval
    Assert.Contains(move, legalMoves startFen)

// (6) Determinism: identical results across repeated runs.
[<Fact>]
let ``best-first is deterministic`` () =
    let run () = searchToNodesEval midFen [||] 3000L cap None stubEval
    let struct (s0, n0, m0) = run ()

    for _ in 1..9 do
        let struct (s, n, m) = run ()
        Assert.Equal(s0, s)
        Assert.Equal(n0, n)
        Assert.Equal(m0, m)

// (7) Arena cap: a tiny cap still yields a legal move and does not hang/crash.
[<Fact>]
let ``tiny arena cap yields a legal move`` () =
    let struct (_, count, move) = searchToNodesEval startFen [||] 0L 500 None stubEval
    Assert.True(int count <= 500, "node count exceeded cap")
    Assert.Contains(move, legalMoves startFen)

// (8) NET-GATED: with the real SF NNUE leaf eval, the arena backup still equals explicit minimax,
// and mate-in-1 is found. Soft-skips when no net is present.
[<Fact>]
let ``net-backed full expansion equals explicit minimax`` () =
    match tryLoadSfNet () with
    | None -> () // soft-skip: no embedded net
    | Some net ->
        let eval = fun (p: Position) -> evalCp net p
        // Reference position must bind the net so evalCp uses the same (incremental) path.
        let refPos = Position()
        refPos.LoadFen midFen
        bindNnue net refPos
        let refScore = refLM refPos 0 2 eval
        let struct (lsScore, _, _) = searchToDepthEval midFen [||] 2 cap (Some net) eval
        Assert.Equal(refScore, lsScore)

[<Fact>]
let ``net-backed mate in one`` () =
    match tryLoadSfNet () with
    | None -> ()
    | Some net ->
        let struct (score, _, move) = searchToDepthEval mate1Fen [||] 1 cap (Some net) (fun (p: Position) -> evalCp net p)
        Assert.True(score >= MATE_IN_MAX_PLY, "expected a mate score, got " + string score)
        Assert.Equal("a1a8", toUci move)

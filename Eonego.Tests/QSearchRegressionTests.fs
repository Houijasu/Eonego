/// Focused regressions for qsearch's move-domain and TT-domain contracts.
module Eonego.Tests.QSearchRegressionTests

#nowarn "9" // NativePtr.stackalloc in drainQSearch

open System
open Microsoft.FSharp.NativeInterop
open Xunit
open Eonego.Bitboard
open Eonego.Move
open Eonego.Position
open Eonego.MoveGeneration
open Eonego.History
open Eonego.MovePick
open Eonego.Transposition
open Eonego.Search

let private sq (file: int) (rank: int) : Square = mkSquare file rank

let private makeWorker (fen: string) (cfg: SearchConfig) (tt: TranspositionTable) : Worker =
    let control = SearchControl(cfg, defaultLimits, tt, fen, [||])
    let w = Worker(0, true, control)
    w.SetupRoot()
    control.Reset()
    control.StartClock 0L 0L
    w

let private drainQSearch (pos: Position) (ttMove: Move) : Move[] =
    let pMoves = NativePtr.stackalloc<Move> MaxMoves
    let moves = Span<Move>(NativePtr.toVoidPtr pMoves, MaxMoves)
    let pScores = NativePtr.stackalloc<int> MaxMoves
    let scores = Span<int>(NativePtr.toVoidPtr pScores, MaxMoves)
    let mutable picker = mkQSearch pos (Tables()) ttMove moves scores
    let result = ResizeArray<Move>()
    let mutable m = nextMove &picker false

    while m <> MoveNone do
        result.Add m
        m <- nextMove &picker false

    result.ToArray()

let private qsearchCfg =
    { defaultConfig with
        Threads = 1
        UseTt = true
        UseQsTt = true
        UsePruning = true
        UseQsChecks = true
        // Keep the fixture about qsearch domains, not TT-as-static-eval or another heuristic.
        UseTtEvalAdjust = false
        UseDeltaPruning = false
        UseCorrHist = false
        UseR50Damp = false
        UseRetro = false }

// White has the quiet 1.Rh8#. With no NNUE attached, captures-only qsearch returns stand-pat 0,
// while checks-enabled qsearch sees the mate and returns MATE - 1.
let private quietMateFen = "k7/8/1K6/8/8/8/8/7R w - - 0 1"

let private runFresh (qsDepth: int) : int =
    let tt = TranspositionTable(1)
    let w = makeWorker quietMateFen qsearchCfg tt
    qsearch w w.Pos 0 1 0 qsDepth

[<Fact>]
let ``captures-only qsearch picker rejects a quiet TT move`` () =
    let pos = Position.OfFen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    let quietTt = mkMove (sq 4 1) (sq 4 3) // e2e4
    Assert.True(pos.IsPseudoLegal quietTt, "fixture TT move must be pseudo-legal")

    let drained = drainQSearch pos quietTt
    Assert.DoesNotContain(quietTt, drained)

    Assert.True(
        drained
        |> Array.forall (fun m ->
            pos.PieceOn(toSq m) <> NoPiece || isEnPassant m || isPromotion m),
        "captures-only qsearch emitted a plain quiet move"
    )

[<Fact>]
let ``qsearch keeps a quiet TT evasion while in check`` () =
    let pos = Position.OfFen "4r1k1/8/8/8/8/8/8/4K3 w - - 0 1"
    let quietEvasion = mkMove (sq 4 0) (sq 3 0) // Ke1-d1
    Assert.True(pos.InCheck)
    Assert.True(isLegal pos quietEvasion)

    let drained = drainQSearch pos quietEvasion
    Assert.NotEmpty(drained)
    Assert.Equal(quietEvasion, drained.[0])

[<Fact>]
let ``captures-only qsearch does not search a quiet TT move`` () =
    let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    let cfg =
        { qsearchCfg with
            UsePruning = false
            UseQsChecks = false }

    let tt = TranspositionTable(1)
    let w = makeWorker fen cfg tt
    let quietTt = mkMove (sq 4 1) (sq 4 3) // e2e4

    // Full-window qsearch cannot take a TT cutoff. The entry exists only to feed the picker a move.
    tt.Store w.Pos.Key 0 BoundUpper 0 VALUE_NONE quietTt false
    let score = qsearch w w.Pos (-INF) INF 0 -1

    Assert.Equal(0, score)
    Assert.Equal(1L, w.Nodes) // root only: the quiet TT child must never be entered

[<Fact>]
let ``checks-enabled qsearch TT result is not reused by captures-only qsearch`` () =
    let expectedCapturesOnly = runFresh -1
    Assert.Equal(0, expectedCapturesOnly)

    let tt = TranspositionTable(1)
    let w = makeWorker quietMateFen qsearchCfg tt
    let checksScore = qsearch w w.Pos 0 1 0 0
    Assert.Equal(MATE - 1, checksScore)

    // Same position and null window, but a strictly smaller move domain. The checks result must not cut.
    let capturesOnlyAfterChecks = qsearch w w.Pos 0 1 0 -1
    Assert.Equal(expectedCapturesOnly, capturesOnlyAfterChecks)

[<Fact>]
let ``captures-only qsearch TT result is not reused by checks-enabled qsearch`` () =
    let expectedWithChecks = runFresh 0
    Assert.Equal(MATE - 1, expectedWithChecks)

    let tt = TranspositionTable(1)
    let w = makeWorker quietMateFen qsearchCfg tt
    let capturesOnlyScore = qsearch w w.Pos 0 1 0 -1
    Assert.Equal(0, capturesOnlyScore)

    // The captures-only upper bound cannot suppress the quiet mate searched by the wider domain.
    let checksAfterCapturesOnly = qsearch w w.Pos 0 1 0 0
    Assert.Equal(expectedWithChecks, checksAfterCapturesOnly)

[<Fact>]
let ``pruning-off qsearch is tagged as captures-only even when checks are configured`` () =
    let tt = TranspositionTable(1)
    let noPruning = makeWorker quietMateFen { qsearchCfg with UsePruning = false } tt

    // Quiet checks live behind UsePruning, so this first search has the captures-only move domain.
    Assert.Equal(0, qsearch noPruning noPruning.Pos 0 1 0 0)

    // Enabling pruning activates the configured quiet-check layer. The prior upper bound must not
    // masquerade as a checks-domain hit and hide Rh8#.
    let withChecks = makeWorker quietMateFen qsearchCfg tt
    Assert.Equal(MATE - 1, qsearch withChecks withChecks.Pos 0 1 0 0)

[<Fact>]
let ``non-checking en passant is searched by qsearch`` () =
    let fen = "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"
    let cfg =
        { qsearchCfg with
            UseTt = false
            UseQsTt = false
            UseQsChecks = false }

    let w = makeWorker fen cfg (TranspositionTable(1))
    let ep = mkEnPassant (sq 4 4) (sq 3 5) // e5xd6 e.p.
    Assert.True(isLegal w.Pos ep, "fixture en passant must be legal")
    Assert.False(w.Pos.GivesCheck ep, "fixture en passant must be non-checking")

    let score = qsearch w w.Pos (-INF) INF 0 -1
    Assert.Equal(0, score)
    Assert.True(w.Nodes > 1L, "qsearch categorically pruned the only en-passant capture")

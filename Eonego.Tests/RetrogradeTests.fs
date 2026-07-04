/// Retrograde search: index/encoding round-trips, the succToPred closed forms, arithmetic index
/// legality per signature, and (later tasks) solver terminals, the full self-consistency proof,
/// publication, and probe behavior.
module Eonego.Tests.RetrogradeTests

open System.Text
open Xunit
open Eonego.Bitboard
open Eonego.Move
open Eonego.Position
open Eonego.Retrograde
open Eonego.Search
open Eonego.Tests.TestFixtures

// LERF square shorthands used across the fixtures (a1 = 0 .. h8 = 63).
let private A1 = 0
let private D1 = 3
let private E1 = 4
let private H1 = 7
let private H2 = 15
let private B3 = 17
let private E4 = 28
let private D5 = 35
let private E5 = 36
let private A8 = 56
let private D8 = 59
let private E8 = 60

[<Fact>]
let ``index packs and decomposes losslessly`` () =
    for stm in 0..1 do
        for wk in [ 0; 7; 28; 35; 56; 63 ] do
            for bk in [ 0; 9; 27; 44; 63 ] do
                for pc in [ 0; 17; 36; 59; 63 ] do
                    let idx = idxOf stm wk bk pc
                    Assert.True(idx >= 0 && idx < RetroSize)
                    Assert.Equal(stm, idxStm idx)
                    Assert.Equal(wk, idxWk idx)
                    Assert.Equal(bk, idxBk idx)
                    Assert.Equal(pc, idxPc idx)

[<Fact>]
let ``succToPred closed forms`` () =
    // Child WinIn1 (+2) -> parent LossIn2 (-3); child mated-now (-1) -> parent WinIn1 (+2).
    Assert.Equal(-3y, succToPred 2y)
    Assert.Equal(2y, succToPred -1y)
    Assert.Equal(0y, succToPred 0y)
    Assert.Equal(-5y, succToPred 4y)
    Assert.Equal(4y, succToPred -3y)

[<Fact>]
let ``retroOrd orders faster mates above slower above draw above losses`` () =
    // +1 is unproducible by the solver (minimum win encoding is +2) — pure comparator check.
    Assert.True(retroOrd 1y > retroOrd 3y)
    Assert.True(retroOrd 3y > retroOrd 0y)
    Assert.True(retroOrd 0y > retroOrd -5y)
    Assert.True(retroOrd -5y > retroOrd -1y)

[<Fact>]
let ``square collisions are illegal`` () =
    let wq = makePiece White Queen
    Assert.False(arithLegal wq White E4 E4 H1) // wk = bk
    Assert.False(arithLegal wq White E4 E8 E4) // pc = wk
    Assert.False(arithLegal wq White E4 E8 E8) // pc = bk

[<Fact>]
let ``adjacent kings are illegal for both sides to move`` () =
    let wq = makePiece White Queen
    Assert.False(arithLegal wq White E4 E5 H1)
    Assert.False(arithLegal wq Black E4 E5 H1)

[<Fact>]
let ``pawns on rank 1 or 8 are illegal for either owner`` () =
    let wp = makePiece White Pawn
    let bp = makePiece Black Pawn
    Assert.False(arithLegal wp White E1 A8 D1) // white pawn, rank 1 (its own back rank)
    Assert.False(arithLegal wp White E1 A8 D8) // white pawn, rank 8 (would be promoted)
    Assert.False(arithLegal bp White E1 A8 D1) // black pawn, rank 1 (would be promoted)
    Assert.False(arithLegal bp White E1 A8 D8) // black pawn, rank 8 (its own back rank)

[<Fact>]
let ``piece attacking the bare king: illegal only when the owner is to move`` () =
    // White Qd5 attacks bKd8 up the open d-file (wKa1 doesn't block).
    let wq = makePiece White Queen
    Assert.False(arithLegal wq White A1 D8 D5) // Black in check with White to move: illegal
    Assert.True(arithLegal wq Black A1 D8 D5) // Black to move, in check, must evade: legal

[<Fact>]
let ``black-owner signature swaps the roles`` () =
    // Black qd5 attacks wKd1 down the open d-file (bKa8 doesn't block).
    let bq = makePiece Black Queen
    Assert.False(arithLegal bq Black D1 A8 D5) // White in check with Black to move: illegal
    Assert.True(arithLegal bq White D1 A8 D5) // White to move, in check, must evade: legal

[<Fact>]
let ``a quiet legal placement is legal for both sides to move`` () =
    // Qb3 attacks neither e8 nor anything adjacent to the kings' geometry rules.
    let wq = makePiece White Queen
    Assert.True(arithLegal wq White E1 E8 B3)
    Assert.True(arithLegal wq Black E1 E8 B3)

// ---------------------------------------------------------------------------
// FEN builder + init pass
// ---------------------------------------------------------------------------

let private G6 = 46
let private C7 = 50
let private G7 = 54
let private B6 = 41
let private H8 = 63

[<Fact>]
let ``fenOf round-trips through Position for sampled indices`` () =
    let sb = StringBuilder(80)

    for (pce, stm, wk, bk, pc) in
        [ (makePiece White Queen, White, E1, E8, B3)
          (makePiece White Queen, Black, G6, H8, G7)
          (makePiece Black Queen, White, D1, A8, D5)
          (makePiece Black Pawn, Black, E4, A8, D5)
          (makePiece White Rook, White, A1, H8, D8) ] do
        let pos = Position.OfFen(fenOf sb pce stm wk bk pc)
        Assert.Equal(stm, pos.SideToMove)
        Assert.Equal(wk, pos.KingSquare White)
        Assert.Equal(bk, pos.KingSquare Black)
        Assert.Equal(pce, pos.PieceOn pc)
        Assert.Equal(3, popCount pos.Occupied)
        Assert.Equal(0, pos.Rule50)
        Assert.Equal(0, pos.CastlingRights)
        Assert.Equal(NoSquare, pos.EpSquare)

/// Shared one-shot init of the White-queen signature (~1M index scan; lazy so the cost is paid once).
let private wqInit =
    lazy
        (let values = Array.create RetroSize RetroUnknown
         let counter: byte[] = Array.zeroCreate RetroSize
         let lossQ0 = ResizeArray<int>()
         initSignature (makePiece White Queen) Array.empty values counter lossQ0 Array.empty
         struct (values, counter, lossQ0))

[<Fact>]
let ``init classifies checkmate as LossIn0 and seeds the level-0 queue`` () =
    let struct (values, _, lossQ0) = wqInit.Force()
    // wKg6, Qg7, bKh8, Black to move: Qg7 is protected mate.
    let idx = idxOf Black G6 H8 G7
    Assert.Equal(-1y, values.[idx])
    Assert.Contains(idx, lossQ0)

[<Fact>]
let ``init classifies stalemate as a finalized draw`` () =
    let struct (values, _, _) = wqInit.Force()
    // wKb6, Qc7, bKa8, Black to move: no legal move, not in check.
    Assert.Equal(0y, values.[idxOf Black B6 A8 C7])

[<Fact>]
let ``init marks arithmetically illegal indices RetroIllegal`` () =
    let struct (values, _, _) = wqInit.Force()
    Assert.Equal(RetroIllegal, values.[idxOf White E4 E5 H1]) // adjacent kings
    Assert.Equal(RetroIllegal, values.[idxOf White A1 D8 D5]) // bare king in check, owner to move

[<Fact>]
let ``init counter equals the legal move count`` () =
    let struct (values, counter, _) = wqInit.Force()
    let idx = idxOf White E1 E8 B3
    Assert.Equal(RetroUnknown, values.[idx]) // non-terminal, untouched by init
    let pos = Position.OfFen "4k3/8/8/8/8/1Q6/8/4K3 w - - 0 1"
    Assert.Equal((collectLegal pos).Length, int counter.[idx])

// ---------------------------------------------------------------------------
// Queen-signature solve: BFS retraction + the full self-consistency proof
// ---------------------------------------------------------------------------

let private wqSolved = lazy (solveSignature (makePiece White Queen) Array.empty)
let private bqSolved = lazy (solveSignature (makePiece Black Queen) Array.empty)

[<Fact>]
let ``solved queen table finds mate in one`` () =
    // wKg6, Qb3, bKh8, White to move: Qb8# (b-file to b8, mate along rank 8; g7/h7 covered by wKg6).
    let values = wqSolved.Force()
    Assert.Equal(2y, values.[idxOf White G6 H8 B3]) // WinIn 1 ply

[<Fact>]
let ``solved queen table keeps terminals and proves losses and quiet stalemates`` () =
    let values = wqSolved.Force()
    Assert.Equal(-1y, values.[idxOf Black G6 H8 G7]) // checkmate stays LossIn 0
    Assert.Equal(0y, values.[idxOf Black B6 A8 C7]) // stalemate stays draw
    // wKg6/Qb3 vs bKh8 with BLACK to move is stalemate (Qb3 covers g8 diagonally; g7/h7 are next
    // to the White king) — the solver must prove the 0, not a loss.
    Assert.Equal(0y, values.[idxOf Black G6 H8 B3])
    // wKg6/Qh2+ vs bKh8: Black in check, Kg8 forced, cornered against the KQ mating net — a loss.
    let v = values.[idxOf Black G6 H8 H2]
    Assert.True(v < 0y, "expected a loss, got " + string (int v))

[<Fact>]
let ``queen signature stats match the literature`` () =
    // KQK: longest win is mate in 10 moves = 19 plies.
    let struct (legal, wins, losses, maxWin, maxLoss) = statsOf (wqSolved.Force())
    Assert.Equal(19, maxWin)
    Assert.True(wins > 0 && losses > 0 && legal > wins + losses) // wins, losses, and real draws exist
    Assert.True(maxLoss = maxWin + 1 || maxLoss = maxWin - 1) // loss parity brackets the win depth

[<Fact>]
let ``black-owner queen signature mirrors the white twin exactly`` () =
    let struct (lW, wW, sW, mwW, mlW) = statsOf (wqSolved.Force())
    let struct (lB, wB, sB, mwB, mlB) = statsOf (bqSolved.Force())
    Assert.Equal(lW, lB)
    Assert.Equal(wW, wB)
    Assert.Equal(sW, sB)
    Assert.Equal(mwW, mwB)
    Assert.Equal(mlW, mlB)

[<Fact>]
[<Trait("Category", "Slow")>]
let ``white queen signature passes the full self-consistency proof`` () =
    Assert.Equal(None, verifySignature (makePiece White Queen) (wqSolved.Force()) Array.empty)

[<Fact>]
[<Trait("Category", "Slow")>]
let ``black queen signature passes the full self-consistency proof`` () =
    Assert.Equal(None, verifySignature (makePiece Black Queen) (bqSolved.Force()) Array.empty)

// ---------------------------------------------------------------------------
// Rook / bishop / knight signatures — same generic solver, piece-specific retraction sets
// ---------------------------------------------------------------------------

let private wrSolved = lazy (solveSignature (makePiece White Rook) Array.empty)
let private brSolved = lazy (solveSignature (makePiece Black Rook) Array.empty)
let private wbSolved = lazy (solveSignature (makePiece White Bishop) Array.empty)
let private wnSolved = lazy (solveSignature (makePiece White Knight) Array.empty)

[<Fact>]
let ``solved rook table finds the existing mate-in-one fixture`` () =
    // k7/8/1K6/8/8/8/8/7R w — the search suite's mate-in-1 (Rh8#): wKb6, bKa8, Rh1.
    let values = wrSolved.Force()
    Assert.Equal(2y, values.[idxOf White B6 A8 H1])

[<Fact>]
let ``rook signature stats match the literature and the black twin mirrors them`` () =
    // KRK: longest win is mate in 16 moves = 31 plies.
    let struct (lW, wW, sW, mwW, mlW) = statsOf (wrSolved.Force())
    Assert.Equal(31, mwW)
    Assert.True(wW > 0 && sW > 0 && lW > wW + sW)
    let struct (lB, wB, sB, mwB, mlB) = statsOf (brSolved.Force())
    Assert.Equal((lW, wW, sW, mwW, mlW), (lB, wB, sB, mwB, mlB))

[<Fact>]
let ``bishop and knight signatures are proven all-draw`` () =
    // The retrograde proof of insufficient material — no shortcut, the fixpoint must show it.
    let struct (_, wB, sB, mwB, mlB) = statsOf (wbSolved.Force())
    Assert.Equal((0, 0, 0, 0), (wB, sB, mwB, mlB))
    let struct (_, wN, sN, mwN, mlN) = statsOf (wnSolved.Force())
    Assert.Equal((0, 0, 0, 0), (wN, sN, mwN, mlN))

[<Fact>]
[<Trait("Category", "Slow")>]
let ``rook signature passes the full self-consistency proof`` () =
    Assert.Equal(None, verifySignature (makePiece White Rook) (wrSolved.Force()) Array.empty)

[<Fact>]
[<Trait("Category", "Slow")>]
let ``bishop signature passes the full self-consistency proof`` () =
    Assert.Equal(None, verifySignature (makePiece White Bishop) (wbSolved.Force()) Array.empty)

[<Fact>]
[<Trait("Category", "Slow")>]
let ``knight signature passes the full self-consistency proof`` () =
    Assert.Equal(None, verifySignature (makePiece White Knight) (wnSolved.Force()) Array.empty)

// ---------------------------------------------------------------------------
// Pawn signatures — un-pushes (single + double) and the promotion dependency
// ---------------------------------------------------------------------------

let private bbSolved = lazy (solveSignature (makePiece Black Bishop) Array.empty)
let private bnSolved = lazy (solveSignature (makePiece Black Knight) Array.empty)

let private promoTablesWhite =
    lazy
        (let t: sbyte[][] = Array.zeroCreate 6
         t.[Knight] <- wnSolved.Force()
         t.[Bishop] <- wbSolved.Force()
         t.[Rook] <- wrSolved.Force()
         t.[Queen] <- wqSolved.Force()
         t)

let private promoTablesBlack =
    lazy
        (let t: sbyte[][] = Array.zeroCreate 6
         t.[Knight] <- bnSolved.Force()
         t.[Bishop] <- bbSolved.Force()
         t.[Rook] <- brSolved.Force()
         t.[Queen] <- bqSolved.Force()
         t)

let private wpSolved = lazy (solveSignature (makePiece White Pawn) (promoTablesWhite.Force()))
let private bpSolved = lazy (solveSignature (makePiece Black Pawn) (promoTablesBlack.Force()))

let private E2 = 12
let private E3 = 20
let private E6 = 44
let private E7 = 52
let private G8 = 62

[<Fact>]
let ``solved pawn table proves promotion mate in one`` () =
    // wKg6, Pe7, bKg8, White to move: e8=Q# (rank-8 check; g7/f7/h7 covered by wKg6).
    let values = wpSolved.Force()
    Assert.Equal(2y, values.[idxOf White G6 G8 E7])

[<Fact>]
let ``king ahead on the sixth wins even with the defender to move`` () =
    // Textbook: Ke6 ahead of Pe5 vs Ke8 is winning regardless of the move; Black to move loses.
    let values = wpSolved.Force()
    let v = values.[idxOf Black E6 E8 E5]
    Assert.True(v < 0y, "expected a loss, got " + string (int v))

[<Fact>]
let ``defender blockading the pawn with the attacker king behind is drawn`` () =
    // Textbook: bKe3 directly in front of Pe2 with wKe1 stuck behind — dead draw.
    let values = wpSolved.Force()
    Assert.Equal(0y, values.[idxOf White E1 E3 E2])

[<Fact>]
let ``pawn signature stats match the literature and the black twin mirrors them`` () =
    // KPK: longest win is mate in 28 moves = 55 plies (through promotion).
    let struct (lW, wW, sW, mwW, mlW) = statsOf (wpSolved.Force())
    Assert.Equal(55, mwW)
    Assert.True(wW > 0 && sW > 0 && lW > wW + sW)
    let struct (lB, wB, sB, mwB, mlB) = statsOf (bpSolved.Force())
    Assert.Equal((lW, wW, sW, mwW, mlW), (lB, wB, sB, mwB, mlB))

[<Fact>]
[<Trait("Category", "Slow")>]
let ``white pawn signature passes the full self-consistency proof`` () =
    Assert.Equal(None, verifySignature (makePiece White Pawn) (wpSolved.Force()) (promoTablesWhite.Force()))

[<Fact>]
[<Trait("Category", "Slow")>]
let ``black pawn signature passes the full self-consistency proof`` () =
    Assert.Equal(None, verifySignature (makePiece Black Pawn) (bpSolved.Force()) (promoTablesBlack.Force()))

// ---------------------------------------------------------------------------
// Publication, probe, root trigger
// ---------------------------------------------------------------------------

[<Fact>]
let ``probe returns solved values through the public path`` () =
    ensureSolved (makePiece White Queen)
    // wKg6/Qb3 vs bKh8: mate in 1 with White to move, stalemate with Black to move.
    Assert.Equal(ValueSome 2y, probe (Position.OfFen "7k/8/6K1/8/8/1Q6/8/8 w - - 0 1"))
    Assert.Equal(ValueSome 0y, probe (Position.OfFen "7k/8/6K1/8/8/1Q6/8/8 b - - 0 1"))

[<Fact>]
let ``probe handles a black-owner signature directly`` () =
    ensureSolved (makePiece Black Queen)
    // The rank-mirrored twin of the mate-in-1: bKg3/qb6 vs wKh1, Black to move.
    Assert.Equal(ValueSome 2y, probe (Position.OfFen "8/8/1q6/8/8/6k1/8/7K b - - 0 1"))

[<Fact>]
let ``probe declines castling rights, wrong man counts, and unsolved signatures`` () =
    // Live castling rights: O-O is legal there and un-modeled by the index.
    Assert.Equal(ValueNone, probe (Position.OfFen "4k3/8/8/8/8/8/8/4K2R w K - 0 1"))
    // 4 men and bare kings are out of probe scope.
    Assert.Equal(ValueNone, probe (Position.OfFen "4k3/8/8/3q4/8/8/3R4/4K3 w - - 0 1"))
    Assert.Equal(ValueNone, probe (Position.OfFen "8/8/4k3/8/8/4K3/8/8 w - - 0 1"))
    // Black-knight signature: nothing in this suite solves it through the PUBLIC slots (its only
    // public solver is ensureSolved of the black PAWN — keep it that way or this goes order-
    // dependent). The slot must be null and the probe must fall through.
    Assert.Equal(ValueNone, probe (Position.OfFen "4k3/8/8/8/4n3/8/8/4K3 w - - 0 1"))

[<Fact>]
let ``signatureClosure lists the signatures one capture away`` () =
    Assert.Equal<int list>([ makePiece White Queen ], signatureClosure (Position.OfFen "7k/8/6K1/8/8/1Q6/8/8 w - - 0 1"))
    // KQ vs KR, 4 men: either capture leaves the other piece's signature.
    let closure = signatureClosure (Position.OfFen "4k3/8/8/3q4/8/8/3R4/4K3 w - - 0 1")
    Assert.Equal(2, closure.Length)
    Assert.Contains(makePiece Black Queen, closure)
    Assert.Contains(makePiece White Rook, closure)
    Assert.Equal<int list>([], signatureClosure (Position.OfFen StartPosFen))

[<Fact>]
let ``concurrent ensureSolved is idempotent and publishes one table`` () =
    let pce = makePiece White Rook

    let tasks =
        Array.init 8 (fun _ -> System.Threading.Tasks.Task.Run(fun () -> ensureSolved pce))

    System.Threading.Tasks.Task.WaitAll tasks
    Assert.True(isSolved pce)
    Assert.Same(solvedTable pce, solvedTable pce)
    Assert.Equal(ValueSome 2y, probe (Position.OfFen "k7/8/1K6/8/8/8/8/7R w - - 0 1"))

[<Fact>]
let ``root trigger solves the signature in the background`` () =
    let pos = Position.OfFen "k7/8/1K6/8/8/8/8/7R w - - 0 1"
    requestSolveFor pos
    let sw = System.Diagnostics.Stopwatch.StartNew()

    while not (isSolved (makePiece White Rook)) && sw.ElapsedMilliseconds < 30000L do
        System.Threading.Thread.Sleep 25

    Assert.True(isSolved (makePiece White Rook))

// ---------------------------------------------------------------------------
// Search integration — the negamax/qsearch probe arms return exact DTM scores
// ---------------------------------------------------------------------------

[<Fact>]
let ``search returns the exact retro mate score at depth 2`` () =
    ensureSolved (makePiece White Rook)
    let struct (score, _, m) = searchToDepth "k7/8/1K6/8/8/8/8/7R w - - 0 1" [||] 2 defaultConfig
    Assert.Equal(MATE - 1, score)
    Assert.Equal("h1h8", toUci m)

[<Fact>]
let ``search backs up the table's exact DTM from probed children`` () =
    // The root itself is never probed (ply 0) — its children return exact values, so even depth 2
    // must reproduce the table's DTM to the ply.
    ensureSolved (makePiece White Queen)
    let v = (solvedTable (makePiece White Queen)).[idxOf White E1 E8 B3]
    Assert.True(v > 0y)
    let struct (score, _, m) = searchToDepth "4k3/8/8/8/8/1Q6/8/4K3 w - - 0 1" [||] 2 defaultConfig
    Assert.Equal(MATE - retroDtm v, score)
    Assert.NotEqual(MoveNone, m)

[<Fact>]
let ``search scores the blockade draw exactly zero`` () =
    ensureSolved (makePiece White Pawn)
    let struct (score, _, _) = searchToDepth "8/8/8/8/8/4k3/4P3/4K3 w - - 0 1" [||] 2 defaultConfig
    Assert.Equal(0, score)

[<Fact>]
let ``rule-50 budget guard declines unreachable wins but keeps draws`` () =
    ensureSolved (makePiece White Queen)
    let v = (solvedTable (makePiece White Queen)).[idxOf White E1 E8 B3]
    Assert.True(retroDtm v > 4) // the decline below relies on the win being longer than the budget
    // Winning KQK with dtm beyond the remaining rule-50 budget: declined, search falls through.
    Assert.Equal(VALUE_NONE, retroScoreAt (Position.OfFen "4k3/8/8/8/8/1Q6/8/4K3 w - - 96 1") 1)
    Assert.True(retroScoreAt (Position.OfFen "4k3/8/8/8/8/1Q6/8/4K3 w - - 0 1") 1 >= MATE_IN_MAX_PLY)
    // Draws are unconditionally safe (the Qb3 stalemate fixture at a high counter).
    Assert.Equal(0, retroScoreAt (Position.OfFen "7k/8/6K1/8/8/1Q6/8/8 b - - 96 1") 1)

[<Fact>]
let ``depth-1 search agrees with the solved table on a stride sample`` () =
    ensureSolved (makePiece White Queen)
    let values = solvedTable (makePiece White Queen)
    let pce = makePiece White Queen
    let cfg = { defaultConfig with HashMb = 1 } // fresh TT per sample; keep it tiny
    let sb = StringBuilder(80)
    let mutable idx = 0

    while idx < RetroSize do
        if values.[idx] <> RetroIllegal then
            let fen = fenOf sb pce (idxStm idx) (idxWk idx) (idxBk idx) (idxPc idx)
            let struct (score, _, _) = searchToDepth fen [||] 1 cfg
            let v = values.[idx]

            let expected =
                if v = 0y then 0
                elif v > 0y then MATE - retroDtm v
                else -MATE + retroDtm v

            Assert.True(
                (expected = score),
                "table/search disagree at " + fen + ": table " + string expected + " search " + string score
            )

        idx <- idx + 4093 // odd stride, ~128 samples across both stm halves

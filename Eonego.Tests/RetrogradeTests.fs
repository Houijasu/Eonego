/// Retrograde search: index/encoding round-trips, the succToPred closed forms, arithmetic index
/// legality per signature, and (later tasks) solver terminals, the full self-consistency proof,
/// publication, and probe behavior.
module Eonego.Tests.RetrogradeTests

open System.Text
open Xunit
open Eonego.Bitboard
open Eonego.Position
open Eonego.Retrograde
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

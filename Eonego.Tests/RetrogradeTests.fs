/// Retrograde search: index/encoding round-trips, the succToPred closed forms, arithmetic index
/// legality per signature, and (later tasks) solver terminals, the full self-consistency proof,
/// publication, and probe behavior.
module Eonego.Tests.RetrogradeTests

open Xunit
open Eonego.Bitboard
open Eonego.Retrograde

// LERF square shorthands used across the fixtures (a1 = 0 .. h8 = 63).
let private A1 = 0
let private D1 = 3
let private E1 = 4
let private H1 = 7
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

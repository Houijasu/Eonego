/// SEE (Position.SeeGe) exact fixtures. Every expected value is hand-computed against the canonical
/// PieceValue table {Pawn=100, Knight=320, Bishop=330, Rook=500, Queen=900, King=0}. Boundary cases pin
/// "SEE value exactly == threshold => true". Non-NORMAL moves must short-circuit to (0 >= threshold).
module Eonego.Tests.SeeTests

open Xunit
open Eonego.Bitboard
open Eonego.Move
open Eonego.Position

let private sq (f: int) (r: int) : Square = mkSquare f r // file a=0..h=7 ; rank 1=0..8=7
let private see (fen: string) (m: Move) (thr: int) : bool = (Position.OfFen fen).SeeGe m thr

// --- A. free hanging capture (undefended pawn): e4xd5 = +100 ---
[<Fact>]
let ``free hanging capture is winning`` () =
    let m = mkMove (sq 4 3) (sq 3 4) // e4 -> d5
    Assert.True(see "4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1" m 0)

// --- B. equal trade RxR (defended): SEE = 0 ---
[<Fact>]
let ``equal rook trade meets threshold 0 but not +1`` () =
    let fen = "4k3/8/4r3/4r3/8/8/4R3/4K3 w - - 0 1" // Re2 x Re5, Re5 defended by Re6
    let m = mkMove (sq 4 1) (sq 4 4) // e2 -> e5
    Assert.True(see fen m 0) // SEE 0 == 0
    Assert.False(see fen m 1) // SEE 0 < 1

// --- C. losing capture QxP, pawn defended by a pawn: SEE = 100 - 900 = -800 ---
[<Fact>]
let ``queen takes pawn-defended pawn is losing`` () =
    let fen = "4k3/8/5p2/4p3/8/8/8/Q3K3 w - - 0 1" // Qa1 x e5 (a1-h8 diag), e5 defended by f6
    let m = mkMove (sq 0 0) (sq 4 4) // a1 -> e5
    Assert.False(see fen m 0) // -800 < 0
    Assert.True(see fen m -800) // boundary: -800 == -800

// --- D. boundary at the free pawn's exact value (+100) ---
[<Fact>]
let ``free pawn capture boundary at its exact value`` () =
    let fen = "4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1"
    let m = mkMove (sq 4 3) (sq 3 4) // e4 -> d5, SEE = +100
    Assert.True(see fen m 100) // 100 == 100
    Assert.False(see fen m 101) // 100 < 101

// --- E. x-ray battery (stacked rooks): d2xd5 chain = +100-500+500-500 = -400 ---
[<Fact>]
let ``stacked-rook battery capture is losing through x-ray`` () =
    let fen = "3rk3/3r4/8/3p4/8/8/3R4/3RK3 w - - 0 1" // Rd2,Rd1 vs Rd7,Rd8 over pawn d5
    let m = mkMove (sq 3 1) (sq 3 4) // d2 -> d5
    Assert.False(see fen m 0) // -400 < 0
    Assert.True(see fen m -400) // boundary

// --- E2. x-ray reveals a queen behind the rook -> flips the sign to +100 ---
[<Fact>]
let ``x-ray queen behind rook flips capture to winning`` () =
    let fen = "3rk3/8/8/3p4/8/8/3R4/3QK3 w - - 0 1" // Rd2 with Qd1 behind, vs Rd8, pawn d5
    let m = mkMove (sq 3 1) (sq 3 4) // d2 -> d5
    Assert.True(see fen m 0) // +100 >= 0

// --- F. recapture chain (knight initiates, pawn-defended): +100-320 = -220 ---
[<Fact>]
let ``knight capture into a pawn defender is losing`` () =
    let fen = "4k3/8/2p5/3p4/8/2N5/8/4K3 w - - 0 1" // Nc3 x d5, d5 defended by c6 pawn
    let m = mkMove (sq 2 2) (sq 3 4) // c3 -> d5
    Assert.False(see fen m 0) // -220 < 0
    Assert.True(see fen m -220) // boundary

// --- G. defended-by-pawn but attacker cheaper than victim: PxR still winning ---
[<Fact>]
let ``pawn takes rook is winning even when defended`` () =
    let undef = "4k3/8/8/4r3/3P4/8/8/4K3 w - - 0 1" // Pd4 x Re5 (undefended)
    let defd = "4k3/8/5p2/4r3/3P4/8/8/4K3 w - - 0 1" // same but Re5 defended by f6 pawn
    let m = mkMove (sq 3 3) (sq 4 4) // d4 -> e5
    Assert.True(see undef m 0) // +500
    Assert.True(see defd m 0) // +500 - 100 = +400 >= 0

// --- H. KING-terminate: king is the only remaining recapturer but an enemy defender remains ---
[<Fact>]
let ``king cannot recapture when a defender remains (king-terminate)`` () =
    // Rd1xd5: +100(P), -500(Rd7 recaptures); white's only further attacker is Ke4, but Rd8 still defends
    // d5 -> the king cannot recapture -> white stays down a rook. Net -400.
    let fen = "3rk3/3r4/8/3p4/4K3/8/8/3R4 w - - 0 1"
    let m = mkMove (sq 3 0) (sq 3 4) // d1 -> d5
    Assert.False(see fen m 0) // -400 < 0 (exercises the KING branch)
    Assert.True(see fen m -400) // boundary

// --- I. non-NORMAL moves short-circuit to (0 >= threshold) ---
[<Fact>]
let ``promotion short-circuits to zero-vs-threshold`` () =
    let fen = "4k3/4P3/8/8/8/8/8/4K3 w - - 0 1"
    let m = mkPromotion (sq 4 6) (sq 4 7) Queen // e7e8=Q
    Assert.True(see fen m 0)
    Assert.False(see fen m 1)
    Assert.True(see fen m -1)

[<Fact>]
let ``castling short-circuits to zero-vs-threshold`` () =
    let fen = "4k3/8/8/8/8/8/8/4K2R w K - 0 1"
    let m = mkCastling (sq 4 0) (sq 6 0) // e1g1
    Assert.True(see fen m 0)
    Assert.False(see fen m 1)
    Assert.True(see fen m -1)

[<Fact>]
let ``en passant short-circuits to zero-vs-threshold`` () =
    let fen = "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"
    let m = mkEnPassant (sq 4 4) (sq 3 5) // e5xd6 e.p.
    Assert.True(see fen m 0)
    Assert.False(see fen m 1)
    Assert.True(see fen m -1)

/// Position.IsPseudoLegal cross-check. The generator (compiled after Position) is the oracle: build the
/// pseudo-legal set (Evasions when in check, else NonEvasions) and the fully-legal set, then assert:
///   (a) every fully-legal move is pseudo-legal;
///   (b) every generated NON-king candidate is pseudo-legal;
///   (c) for KING moves, IsPseudoLegal <=> membership in the LEGAL set (IsPseudoLegal keeps the reference's
///       king-into-check test, which raw generate() does not cull — the "split oracle");
///   (d) no NORMAL move from a side-to-move piece is a false positive.
/// Plus a battery of crafted negatives.
module Eonego.Tests.IsPseudoLegalTests

#nowarn "9" // NativePtr.stackalloc

open System
open System.Collections.Generic
open Microsoft.FSharp.NativeInterop
open Xunit
open Eonego.Bitboard
open Eonego.Move
open Eonego.Position
open Eonego.MoveGeneration
open Eonego.Tests.TestFixtures

let private sq (f: int) (r: int) : Square = mkSquare f r

let private pseudoSet (p: Position) : HashSet<int> =
    let pm = NativePtr.stackalloc<Move> MaxMoves
    let buf = Span<Move>(NativePtr.toVoidPtr pm, MaxMoves)
    let n = generate p buf (if p.InCheck then Evasions else NonEvasions)
    let s = HashSet<int>()

    for i in 0 .. n - 1 do
        s.Add buf.[i] |> ignore

    s

let private legalSet (p: Position) : HashSet<int> =
    let s = HashSet<int>()

    for m in collectLegal p do
        s.Add m |> ignore

    s

let private testFens =
    [ yield! perftFens
      "4r2k/8/8/8/8/8/8/4K3 w - - 0 1" // Ke1 in check from Re8 (king-split cases incl. Ke1e2)
      "k7/6b1/8/3pP3/8/2K5/8/8 w - d6 0 1" // ep available under a diagonal-pin context
      "k7/8/8/3pP3/4K3/8/8/8 w - d6 0 1" // ep as a check evasion
      "5rk1/8/8/8/8/8/8/4K2R w K - 0 1" // O-O through the attacked f1
      "1r5k/8/8/8/8/8/8/R3K3 w Q - 0 1" // O-O-O with b1 attacked-but-empty
      "7k/4P3/8/8/8/8/8/4K3 w - - 0 1" ] // a promotion is available (king off the e-file)

[<Fact>]
let ``IsPseudoLegal agrees with the generator across positions`` () =
    for fen in testFens do
        let p = Position.OfFen fen
        let pl = pseudoSet p
        let legal = legalSet p
        let ksq = p.KingSquare p.SideToMove

        for m in legal do
            Assert.True(p.IsPseudoLegal m, sprintf "legal move not pseudo-legal: %s in %s" (toUci m) fen)

        for m in pl do
            if fromSq m = ksq then
                Assert.Equal(legal.Contains m, p.IsPseudoLegal m) // king split oracle
            else
                Assert.True(p.IsPseudoLegal m, sprintf "generated non-king not pseudo-legal: %s in %s" (toUci m) fen)
        // no false positives among NORMAL moves from any side-to-move piece
        let mutable occ = p.ColorBB p.SideToMove

        while occ <> 0UL do
            let from = popLsb &occ

            for t in 0..63 do
                if t <> from then
                    let m = mkMove from t

                    if p.IsPseudoLegal m then
                        Assert.True(
                            pl.Contains m || (from = ksq && legal.Contains m),
                            sprintf "false-positive NORMAL pseudo-legal: %s in %s" (toUci m) fen
                        )

[<Fact>]
let ``crafted illegal/foreign moves are rejected`` () =
    let isPL (fen: string) (m: Move) = (Position.OfFen fen).IsPseudoLegal m
    let start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    // sentinels
    Assert.False(isPL start MoveNone)
    Assert.False(isPL start MoveNull)
    // wrong/empty piece on from (e4 empty in startpos)
    Assert.False(isPL start (mkMove (sq 4 3) (sq 4 4)))
    // own-piece capture (Nb1 x own d2 pawn)
    Assert.False(isPL start (mkMove (sq 1 0) (sq 3 1)))
    // enemy piece on from (white to move, e7 is a black pawn)
    Assert.False(isPL start (mkMove (sq 4 6) (sq 4 5)))
    // blocked slider: Ra1 cannot pass Pb1
    Assert.False(isPL "4k3/8/8/8/8/8/8/RP2K3 w - - 0 1" (mkMove (sq 0 0) (sq 7 0)))
    // pawn diagonal to an empty square
    Assert.False(isPL "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1" (mkMove (sq 4 1) (sq 5 2)))
    // single push onto an enemy / double push over a blocker / from a non-start rank
    Assert.False(isPL "4k3/8/8/8/8/4p3/4P3/4K3 w - - 0 1" (mkMove (sq 4 1) (sq 4 2))) // e2e3 blocked
    Assert.False(isPL "4k3/8/8/8/8/4p3/4P3/4K3 w - - 0 1" (mkMove (sq 4 1) (sq 4 3))) // e2e4 over blocker
    Assert.False(isPL "4k3/8/8/8/8/4P3/8/4K3 w - - 0 1" (mkMove (sq 4 2) (sq 4 4))) // e3e5 not from start
    Assert.True(isPL start (mkMove (sq 4 1) (sq 4 3))) // e2e4 valid (positive)
    // en passant: none set, and wrong target
    Assert.False(isPL "4k3/8/8/3pP3/8/8/8/4K3 w - - 0 1" (mkEnPassant (sq 4 4) (sq 3 5))) // no ep square
    Assert.False(isPL "k7/8/8/3pP3/8/2K5/8/8 w - d6 0 1" (mkEnPassant (sq 4 4) (sq 5 5))) // ep but wrong target
    Assert.True(isPL "k7/8/8/3pP3/8/2K5/8/8 w - d6 0 1" (mkEnPassant (sq 4 4) (sq 3 5))) // valid ep (positive)
    // castling: no right / through check / in check / rook missing
    Assert.False(isPL "4k3/8/8/8/8/8/8/4K2R w - - 0 1" (mkCastling (sq 4 0) (sq 6 0))) // no K right
    Assert.False(isPL "5rk1/8/8/8/8/8/8/4K2R w K - 0 1" (mkCastling (sq 4 0) (sq 6 0))) // through attacked f1
    Assert.False(isPL "4r2k/8/8/8/8/8/8/4K2R w K - 0 1" (mkCastling (sq 4 0) (sq 6 0))) // while in check
    Assert.False(isPL "4k3/8/8/8/8/8/8/4K3 w K - 0 1" (mkCastling (sq 4 0) (sq 6 0))) // rook missing
    Assert.True(isPL "4k3/8/8/8/8/8/8/4K2R w K - 0 1" (mkCastling (sq 4 0) (sq 6 0))) // valid O-O (positive)
    // promotion flag on a non-last-rank move ; NORMAL flag on a last-rank pawn move
    Assert.False(isPL start (mkPromotion (sq 4 1) (sq 4 3) Queen)) // e2e4 with promo flag
    Assert.False(isPL "7k/4P3/8/8/8/8/8/4K3 w - - 0 1" (mkMove (sq 4 6) (sq 4 7))) // e7e8 NORMAL (needs promo)
    Assert.True(isPL "7k/4P3/8/8/8/8/8/4K3 w - - 0 1" (mkPromotion (sq 4 6) (sq 4 7) Rook)) // valid under-promo
    // a move legal only in a different position
    Assert.False(isPL "4k3/8/8/8/8/8/8/4K3 w - - 0 1" (mkMove (sq 4 1) (sq 4 3))) // e2 empty here

module Eonego.Tests.EvaluationTests

open System
open Xunit
open Eonego.Bitboard
open Eonego.Move
open Eonego.Position
open Eonego.MoveGeneration
open Eonego.Evaluation
open Eonego.Tests.TestFixtures

let private boardField (fen: string) = fen.Split(' ').[0]

let private assertSymmetric (p: Position) =
    let mirrored = Position.OfFen(mirrorFen (p.ToFen()))
    Assert.Equal(eval p, -(eval mirrored))

[<Fact>]
let ``mirrorFen flips vertically + swaps colors (startpos is a fixed point, Kiwipete is not)`` () =
    Assert.Equal(boardField StartPosFen, boardField (mirrorFen StartPosFen))
    Assert.True(boardField perftFens.[1] <> boardField (mirrorFen perftFens.[1]))

[<Fact>]
let ``eval is exactly anti-symmetric under the board mirror`` () =
    for fen in perftFens do
        assertSymmetric (Position.OfFen fen)

    let rng = Random(20260621)
    let mutable p = Position.OfFen StartPosFen
    let mutable ply = 0

    for _ in 1..1000 do
        assertSymmetric p
        let moves = collectLegal p

        if moves.Length = 0 || ply >= 80 then
            p <- Position.OfFen StartPosFen
            ply <- 0
        else
            p.Make moves.[rng.Next moves.Length]
            ply <- ply + 1

[<Fact>]
let ``startpos evaluates to 0`` () =
    let p = Position.OfFen StartPosFen
    Assert.Equal(0, eval p)
    let struct (mg, eg, phase) = evalTrace p
    Assert.Equal(struct (0, 0, 0), struct (mg, eg, phase))

[<Fact>]
let ``up a queen is strongly positive for the side to move`` () =
    let p = Position.OfFen "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    Assert.True(eval p > 600, sprintf "expected > 600, got %d" (eval p))

[<Fact>]
let ``materialEval matches eval`` () =
    for fen in perftFens do
        let p = Position.OfFen fen
        Assert.Equal(eval p, materialEval p)

[<Fact>]
let ``pinned exact totals: KQ vs K`` () =
    let p = Position.OfFen "7k/8/8/8/8/8/8/QK6 w - -"
    Assert.Equal(struct (900, 900, 0), evalTrace p)
    Assert.Equal(900, eval p)

[<Fact>]
let ``pinned exact totals: pawn endgame`` () =
    let p = Position.OfFen "8/8/8/3k4/8/3K4/4P3/8 w - -"
    Assert.Equal(struct (100, 100, 0), evalTrace p)
    Assert.Equal(100, eval p)

module Eonego.Tests.NnueEvalRangeTests

open Xunit
open Eonego.Position
open Eonego.MoveGeneration
open Eonego.Nnue
open Eonego.Tests.TestFixtures
open Eonego.Tests.NnueTestFixtures

let private enabled (fen: string) : Position =
    let p = Position()
    p.LoadFen fen
    p.EnableNnue true
    p

[<Fact>]
let ``NNUE eval is within ±EvalMax across the CPW positions`` () =
    let net = loadOrFail (buildSeededRefNet 7)

    for fen in perftFens do
        let e = evaluate net (enabled fen)
        Assert.True(abs e <= EvalMax, sprintf "%d out of range for %s" e fen)

[<Fact>]
let ``NNUE eval is restored after make + unmake (full accumulator->assemble->forward pipeline)`` () =
    let net = loadOrFail (buildSeededRefNet 7)
    let p = enabled perftFens.[1] // Kiwipete: captures, castling, promotions

    let rec walk depth =
        let before = evaluate net p

        if depth > 0 then
            for m in collectLegal p do
                p.Make m
                walk (depth - 1)
                p.Unmake m
                Assert.Equal(before, evaluate net p)

    walk 2

[<Fact>]
let ``a mis-scaled net is clamped to ±EvalMax (no int16 TT wrap)`` () =
    // all weights 0, L5 bias enormous, quantScale 1 -> raw forward = 5,000,000 >> EvalMax.
    let net = loadOrFail { zeroNet 1 with L5B = 5_000_000 }
    let p = enabled perftFens.[0] // startpos, white to move -> positive clamp
    Assert.Equal(EvalMax, evaluate net p)

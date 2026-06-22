module Eonego.Tests.NnueSearchTests

open Xunit
open Eonego.Move
open Eonego.Nnue
open Eonego.Search
open Eonego.Tests.NnueTestFixtures

let private oracleNnue =
    { defaultConfig with
        UsePruning = false
        UseNnue = true }

let private net = loadOrFail (buildSeededRefNet 7)

[<Fact>]
let ``NNUE search finds mate in one with the correct ply-adjusted score`` () =
    let struct (score, _, m) =
        searchToDepthNet "k7/8/1K6/8/8/8/8/7R w - - 0 1" [||] 2 oracleNnue (Some net)

    Assert.True(score >= MATE_IN_MAX_PLY)
    Assert.Equal(MATE - 1, score)
    Assert.Equal("h1h8", toUci m)

[<Fact>]
let ``NNUE search finds mate in two with the correct ply-adjusted score`` () =
    let struct (score, _, m) =
        searchToDepthNet "7k/8/5K2/8/8/8/8/R7 w - - 0 1" [||] 6 oracleNnue (Some net)

    Assert.True(score >= MATE_IN_MAX_PLY)
    Assert.Equal(MATE - 3, score)
    Assert.NotEqual(MoveNone, m)

[<Fact>]
let ``NNUE search returns a finite quiet-position score`` () =
    let struct (score, _, m) =
        searchToDepthNet "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1" [||] 1 oracleNnue (Some net)

    Assert.True(abs score <= EvalMax, sprintf "score %d exceeded NNUE eval clamp" score)
    Assert.NotEqual(MoveNone, m)

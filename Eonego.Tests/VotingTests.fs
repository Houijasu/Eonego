/// Thread-vote correctness (Search.voteBest — pure over parallel arrays, so no workers needed).
/// The vote weight is (score − minScore + 40) × depth per voter; mate proofs override consensus.
module Eonego.Tests.VotingTests

open Xunit
open Eonego.Move
open Eonego.Search

let private m1 = mkMove 12 28 // e2e4
let private m2 = mkMove 6 21 // g1f3
let private m3 = mkMove 11 27 // d2d4

[<Fact>]
let ``unanimous vote picks the deepest worker for the move`` () =
    let moves = [| m1; m1; m1; m1 |]
    let scores = [| 10; 12; 8; 11 |]
    let depths = [| 14; 17; 13; 15 |]
    Assert.Equal(1, voteBest moves scores depths 4)

[<Fact>]
let ``deep high-scoring minority outvotes a shallow majority`` () =
    // minority: (50-10+40)*20 = 1600; majority: 3 * (10-10+40)*10 = 1200
    let moves = [| m2; m1; m1; m1 |]
    let scores = [| 50; 10; 10; 10 |]
    let depths = [| 20; 10; 10; 10 |]
    Assert.Equal(0, voteBest moves scores depths 4)

[<Fact>]
let ``a broad deep majority beats a single deeper voter`` () =
    // majority: 3 * (30-10+40)*15 = 2700; single: (10-10+40)*20 = 800
    let moves = [| m1; m1; m1; m2 |]
    let scores = [| 30; 30; 30; 10 |]
    let depths = [| 15; 15; 15; 20 |]
    let idx = voteBest moves scores depths 4
    Assert.Equal(m1, moves.[idx])

[<Fact>]
let ``a proven mate overrides any consensus`` () =
    let moves = [| m1; m1; m1; m3 |]
    let scores = [| 100; 100; 100; MATE - 5 |]
    let depths = [| 25; 25; 25; 8 |]
    Assert.Equal(3, voteBest moves scores depths 4)

[<Fact>]
let ``the shortest proven mate wins among mate voters`` () =
    let moves = [| m1; m2; m3 |]
    let scores = [| MATE - 9; MATE - 3; MATE - 5 |]
    let depths = [| 12; 8; 20 |]
    Assert.Equal(1, voteBest moves scores depths 3)

[<Fact>]
let ``zero voters falls back to worker 0`` () =
    let moves = [| MoveNone; MoveNone |]
    let scores = [| 0; 0 |]
    let depths = [| 0; 0 |]
    Assert.Equal(0, voteBest moves scores depths 2)

[<Fact>]
let ``a single voter wins regardless of index`` () =
    let moves = [| MoveNone; MoveNone; m2 |]
    let scores = [| 0; 0; -30 |]
    let depths = [| 0; 0; 9 |]
    Assert.Equal(2, voteBest moves scores depths 3)

[<Fact>]
let ``workers without a completed iteration cannot vote`` () =
    // Worker 1 has a huge score but depth 0 (no completed iteration) — must be ignored.
    let moves = [| m1; m2 |]
    let scores = [| 5; 5000 |]
    let depths = [| 10; 0 |]
    Assert.Equal(0, voteBest moves scores depths 2)

[<Fact>]
let ``full ties resolve to the lowest worker index`` () =
    let moves = [| m1; m1; m1 |]
    let scores = [| 10; 10; 10 |]
    let depths = [| 12; 12; 12 |]
    Assert.Equal(0, voteBest moves scores depths 3)

module Eonego.Tests.NnueL1AccumulatorTests

#nowarn "9" // `fixed` pin in the recompute oracle

open Xunit
open Eonego.Bitboard
open Eonego.Move
open Eonego.Position
open Eonego.MoveGeneration
open Eonego.NnueNetwork
open Eonego.Nnue
open Eonego.Tests.TestFixtures
open Eonego.Tests.NnueTestFixtures

/// Local clamp matching Nnue.clampEval (private there) — keeps eval within ±EvalMax.
let private clampEval (x: int) : int =
    if x > EvalMax then EvalMax
    elif x < -EvalMax then -EvalMax
    else x

// ---------------------------------------------------------------------------
// Oracle A: the incremental L1 pre-activation accumulator must equal a slow
// from-scratch recomputation of the 2592x64 first-layer GEMV at every node.
// Oracle B: end-to-end eval must stay bit-identical to the old recompute path.
// ---------------------------------------------------------------------------

/// From-scratch L1 pre-activation: L1B[o] + sum_i assembleInput(p)[i] * L1W[o*PaddedL1+i].
let private fromScratchL1 (net: Network) (p: Position) : int[] =
    let input = assembleInput p
    let acc = Array.copy net.L1B

    for o in 0 .. L1Size - 1 do
        let mutable sum = 0
        let row = o * PaddedL1

        for i in 0 .. PaddedL1 - 1 do
            sum <- sum + int input.[i] * int net.L1W.[row + i]

        acc.[o] <- acc.[o] + sum

    acc

let private enabledAndBound (net: Network) (fen: string) : Position =
    let p = Position()
    p.LoadFen fen
    p.EnableNnue true
    bind net p
    p

let private seeds = [ 1; 2; 3; 7; 42 ]

[<Fact>]
let ``L1 accumulator == from-scratch after EnableNnue + bind for all CPW positions and seeds`` () =
    for seed in seeds do
        let net = loadOrFail (buildSeededRefNet seed)

        for fen in perftFens do
            let p = enabledAndBound net fen
            Assert.True((p.L1Accumulator = fromScratchL1 net p), "static L1 mismatch for " + fen)

let rec private walkL1 (net: Network) (p: Position) (depth: int) =
    Assert.True((p.L1Accumulator = fromScratchL1 net p), "incremental L1 != from-scratch mid-walk")

    if depth > 0 then
        for m in collectLegal p do
            let before = Array.copy (p.L1Accumulator)
            p.Make m
            walkL1 net p (depth - 1)
            p.Unmake m
            Assert.True((before = p.L1Accumulator), "L1 accumulator not restored after unmake: " + toUci m)

[<Fact>]
let ``incremental L1 accumulator matches from-scratch at every node (depth 2, all CPW positions)`` () =
    let net = loadOrFail (buildSeededRefNet 7)

    for fen in perftFens do
        walkL1 net (enabledAndBound net fen) 2

[<Fact>]
let ``L1 accumulator toggles STM correctly across null moves`` () =
    let net = loadOrFail (buildSeededRefNet 7)
    let p = enabledAndBound net perftFens.[1] // Kiwipete

    let rec walkNull depth =
        Assert.True((p.L1Accumulator = fromScratchL1 net p), "L1 mismatch before null move")

        if depth > 0 && not p.InCheck then
            let before = Array.copy (p.L1Accumulator)
            p.MakeNull()
            Assert.True((p.L1Accumulator = fromScratchL1 net p), "L1 mismatch after MakeNull")
            walkNull (depth - 1)
            p.UnmakeNull()
            Assert.True((before = p.L1Accumulator), "L1 accumulator not restored after UnmakeNull")

    walkNull 3

// ---------------------------------------------------------------------------
// Oracle B: end-to-end eval bit-identical to the old assembleInto+forwardWith path.
// ---------------------------------------------------------------------------

let private recomputeEval (net: Network) (p: Position) : int =
    let input = assembleInput p
    use ptr = fixed input
    let cp = clampEval (forwardWith useAvx2Default net ptr)
    if p.SideToMove = White then cp else -cp

[<Fact>]
let ``incremental evaluate matches recompute across CPW positions (AVX2 path)`` () =
    if not useAvx2Default then () // skip when scalar is forced; covered by scalar variant
    let net = loadOrFail (buildSeededRefNet 7)

    for fen in perftFens do
        let p = enabledAndBound net fen
        Assert.Equal(recomputeEval net p, evaluate net p)

[<Fact>]
let ``incremental evaluate matches recompute across CPW positions (scalar path)`` () =
    let net = loadOrFail (buildSeededRefNet 7)

    for fen in perftFens do
        let p = enabledAndBound net fen
        let input = assembleInput p
        use ptr = fixed input
        let cp = clampEval (forwardWith false net ptr)
        let expected = if p.SideToMove = White then cp else -cp
        Assert.Equal(expected, evaluate net p)

[<Fact>]
let ``incremental evaluate matches recompute at every node of a Kiwipete depth-2 walk (all seeds)`` () =
    for seed in seeds do
        let net = loadOrFail (buildSeededRefNet seed)
        let p = enabledAndBound net perftFens.[1]

        let rec walk depth =
            Assert.Equal(recomputeEval net p, evaluate net p)

            if depth > 0 then
                for m in collectLegal p do
                    p.Make m
                    walk (depth - 1)
                    p.Unmake m
                    Assert.Equal(recomputeEval net p, evaluate net p)

        walk 2

// ---------------------------------------------------------------------------
// Hardening cases: reload-while-bound (C6), bound-but-disabled fallback (B5),
// and null moves interleaved with real moves in one walk.
// ---------------------------------------------------------------------------

[<Fact>]
let ``L1 accumulator is rebuilt correctly when LoadFen reloads a bound position`` () =
    let net = loadOrFail (buildSeededRefNet 7)
    let p = enabledAndBound net perftFens.[0] // bound to one position...
    p.LoadFen perftFens.[3] // ...then reloaded to another while still bound + active
    let fresh = enabledAndBound net perftFens.[3]
    Assert.True((p.L1Accumulator = fresh.L1Accumulator), "reloaded L1 != freshly-built L1")
    Assert.Equal(evaluate net fresh, evaluate net p)

[<Fact>]
let ``evaluate falls back to the recompute path when NNUE is active but L1 is not bound`` () =
    let net = loadOrFail (buildSeededRefNet 7)
    let p = Position()
    p.LoadFen perftFens.[0]
    p.EnableNnue true // active, but never bound -> L1Active is false
    Assert.False(p.L1Bound)
    Assert.False(p.L1Active)

    for fen in perftFens do
        p.LoadFen fen
        p.EnableNnue true
        Assert.Equal(recomputeEval net p, evaluate net p)

[<Fact>]
let ``L1 accumulator stays exact with null moves interleaved in a Make/Unmake walk`` () =
    let net = loadOrFail (buildSeededRefNet 7)

    let rec walk (p: Position) depth =
        Assert.True((p.L1Accumulator = fromScratchL1 net p), "L1 mismatch in interleaved walk")

        if depth > 0 then
            if not p.InCheck then
                let before = Array.copy (p.L1Accumulator)
                p.MakeNull()
                Assert.True((p.L1Accumulator = fromScratchL1 net p), "L1 mismatch after interleaved null")
                walk p (depth - 1)
                p.UnmakeNull()
                Assert.True((before = p.L1Accumulator), "L1 not restored after interleaved null")

            for m in collectLegal p do
                p.Make m
                walk p (depth - 1)
                p.Unmake m

    walk (enabledAndBound net perftFens.[1]) 2

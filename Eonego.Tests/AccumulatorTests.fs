module Eonego.Tests.AccumulatorTests

open Xunit
open Eonego.Bitboard
open Eonego.Accumulator
open Eonego.Move
open Eonego.Position
open Eonego.MoveGeneration
open Eonego.Tests.TestFixtures

// Two hand-verified indices: white pawn e2 with WK e1; black pawn e7 with BK e8.
[<Fact>]
let ``makeIndex matches hand-computed HalfKAv2_hm indices`` () =
    Assert.Equal(21836, makeIndex White 0 12 4)   // White persp, WP(0) on e2(12), WK e1(4)
    Assert.Equal(21836, makeIndex Black 6 52 60)  // Black persp, BP(6) on e7(52), BK e8(60)

[<Fact>]
let ``makeIndex is always in [0, 22528)`` () =
    for pColor in [ White; Black ] do
        for pc in 0 .. 11 do
            for sq in 0 .. 63 do
                for ksq in 0 .. 63 do
                    let idx = makeIndex pColor pc sq ksq
                    Assert.True(idx >= 0 && idx < 22528, sprintf "idx %d out of range (pc=%d sq=%d ksq=%d)" idx sq pc ksq)

// The Position-maintained accumulator is now lazy; NNUE tests cover full raw/eval parity when the net exists.

[<Fact>]
let ``addFeature AVX2 equals scalar over random rows`` () =
    let rng = System.Random(12345)
    let nFeatures = 64
    let ftWeights = Array.init (nFeatures * L1) (fun _ -> int16 (rng.Next(-32768, 32768)))
    let ftPsqt = Array.init (nFeatures * PsqtBuckets) (fun _ -> rng.Next(-100000, 100000))
    let accA = Array.init L1 (fun _ -> rng.Next(-1000000, 1000000))
    let accB = Array.copy accA
    let psqtA = Array.zeroCreate<int> PsqtBuckets
    let psqtB = Array.zeroCreate<int> PsqtBuckets
    for trial in 0 .. 200 do
        let idx = rng.Next(0, nFeatures)
        let sign = if rng.Next(0, 2) = 0 then 1 else -1
        addFeature accA psqtA ftWeights ftPsqt idx sign false  // scalar
        addFeature accB psqtB ftWeights ftPsqt idx sign true   // avx2
    Assert.Equal<int[]>(accA, accB)
    Assert.Equal<int[]>(psqtA, psqtB)

[<Fact>]
let ``addThreat AVX2 equals scalar over random rows`` () =
    let rng = System.Random(67890)
    let nFeatures = 64
    let threatWeights = Array.init (nFeatures * L1) (fun _ -> sbyte (rng.Next(-128, 128)))
    let threatPsqt = Array.init (nFeatures * PsqtBuckets) (fun _ -> rng.Next(-100000, 100000))
    let accA = Array.init L1 (fun _ -> rng.Next(-1000000, 1000000))
    let accB = Array.copy accA
    let psqtA = Array.zeroCreate<int> PsqtBuckets
    let psqtB = Array.zeroCreate<int> PsqtBuckets

    for _ in 0 .. 200 do
        let idx = rng.Next(0, nFeatures)
        let sign = if rng.Next(0, 2) = 0 then 1 else -1
        addThreatAt accA 0 psqtA 0 threatWeights threatPsqt idx sign false
        addThreatAt accB 0 psqtB 0 threatWeights threatPsqt idx sign true

    Assert.Equal<int[]>(accA, accB)
    Assert.Equal<int[]>(psqtA, psqtB)

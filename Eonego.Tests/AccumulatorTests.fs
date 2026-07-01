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
    let accA = Array.init L1 (fun _ -> int16 (rng.Next(-32768, 32768))) // int16 accumulator (full range)
    let accB = Array.copy accA
    let psqtA = Array.zeroCreate<int> PsqtBuckets
    let psqtB = Array.zeroCreate<int> PsqtBuckets
    for trial in 0 .. 200 do
        let idx = rng.Next(0, nFeatures)
        let sign = if rng.Next(0, 2) = 0 then 1 else -1
        addFeature accA psqtA ftWeights ftPsqt idx sign false  // scalar
        addFeature accB psqtB ftWeights ftPsqt idx sign true   // avx2
    Assert.Equal<int16[]>(accA, accB) // scalar==AVX2 wrap agreement through int16 overflow
    Assert.Equal<int[]>(psqtA, psqtB)

[<Fact>]
let ``addThreat AVX2 equals scalar over random rows`` () =
    let rng = System.Random(67890)
    let nFeatures = 64
    let threatWeights = Array.init (nFeatures * L1) (fun _ -> sbyte (rng.Next(-128, 128)))
    let threatPsqt = Array.init (nFeatures * PsqtBuckets) (fun _ -> rng.Next(-100000, 100000))
    let accA = Array.init L1 (fun _ -> int16 (rng.Next(-32768, 32768))) // int16 accumulator (full range)
    let accB = Array.copy accA
    let psqtA = Array.zeroCreate<int> PsqtBuckets
    let psqtB = Array.zeroCreate<int> PsqtBuckets

    for _ in 0 .. 200 do
        let idx = rng.Next(0, nFeatures)
        let sign = if rng.Next(0, 2) = 0 then 1 else -1
        addThreatAt accA 0 psqtA 0 threatWeights threatPsqt idx sign false
        addThreatAt accB 0 psqtB 0 threatWeights threatPsqt idx sign true

    Assert.Equal<int16[]>(accA, accB) // scalar==AVX2 wrap agreement through int16 overflow
    Assert.Equal<int[]>(psqtA, psqtB)

// The fused single-pass kernel must equal a sequence of the feature-outer reference kernels, for both SIMD
// modes, for in-place AND parent->child (src<>dst) variants, and across the multi-pass chunking threshold
// (row totals > FusedMaxRowsPerPass). Wrapping int16/int32 adds commute, so any mismatch is a kernel bug.
[<Fact>]
let ``applyFused equals sequential reference kernels (in-place, src->dst, chunked, both SIMD modes)`` () =
    let rng = System.Random(424242)
    let nHalfRows = 48
    let nThrRows = 160
    let halfW = Array.init (nHalfRows * L1) (fun _ -> int16 (rng.Next(-32768, 32768)))
    let halfPsqt = Array.init (nHalfRows * PsqtBuckets) (fun _ -> rng.Next(-100000, 100000))
    let thrW = Array.init (nThrRows * L1) (fun _ -> sbyte (rng.Next(-128, 128)))
    let thrPsqt = Array.init (nThrRows * PsqtBuckets) (fun _ -> rng.Next(-100000, 100000))

    for useAvx2 in [ false; true ] do
        if not useAvx2 || System.Runtime.Intrinsics.X86.Avx2.IsSupported then
            for trial in 0 .. 24 do
                // Row-list sizes sweep from tiny to well past FusedMaxRowsPerPass (chunked path).
                let nHA = rng.Next(0, 9)
                let nHS = rng.Next(0, 9)
                let nTA = rng.Next(0, 65)
                let nTS = rng.Next(0, 65)
                let halfAdd = Array.init nHA (fun _ -> rng.Next(0, nHalfRows))
                let halfSub = Array.init nHS (fun _ -> rng.Next(0, nHalfRows))
                let thrAdd = Array.init nTA (fun _ -> rng.Next(0, nThrRows))
                let thrSub = Array.init nTS (fun _ -> rng.Next(0, nThrRows))
                let src = Array.init L1 (fun _ -> int16 (rng.Next(-32768, 32768)))
                let srcPsq = Array.init PsqtBuckets (fun _ -> rng.Next(-1000000, 1000000))

                // Reference: copy src, then sequential feature-outer kernels.
                let refAcc = Array.copy src
                let refPsq = Array.copy srcPsq

                for idx in halfAdd do
                    addFeature refAcc refPsq halfW halfPsqt idx 1 useAvx2

                for idx in halfSub do
                    addFeature refAcc refPsq halfW halfPsqt idx -1 useAvx2

                for idx in thrAdd do
                    addThreatAt refAcc 0 refPsq 0 thrW thrPsqt idx 1 useAvx2

                for idx in thrSub do
                    addThreatAt refAcc 0 refPsq 0 thrW thrPsqt idx -1 useAvx2

                // Fused, parent->child: dst is a separate buffer.
                let dst = Array.zeroCreate<int16> L1
                let dstPsq = Array.zeroCreate<int> PsqtBuckets

                applyFused
                    src 0 dst 0 srcPsq 0 dstPsq 0 halfW halfPsqt thrW thrPsqt
                    halfAdd nHA halfSub nHS thrAdd nTA thrSub nTS useAvx2

                Assert.Equal<int16[]>(refAcc, dst)
                Assert.Equal<int[]>(refPsq, dstPsq)

                // Fused, in-place: src buffer updated in situ.
                let inPlace = Array.copy src
                let inPlacePsq = Array.copy srcPsq

                applyFused
                    inPlace 0 inPlace 0 inPlacePsq 0 inPlacePsq 0 halfW halfPsqt thrW thrPsqt
                    halfAdd nHA halfSub nHS thrAdd nTA thrSub nTS useAvx2

                Assert.Equal<int16[]>(refAcc, inPlace)
                Assert.Equal<int[]>(refPsq, inPlacePsq)

                ignore trial

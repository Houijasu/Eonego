module Eonego.Tests.NnueForwardPassTests

open System.Runtime.Intrinsics.X86
open Xunit
open Eonego.NnueNetwork
open Eonego.Tests.NnueTestFixtures

let private netSeeds = [ 1; 2; 3; 7; 42 ]
let private inputSeeds = [ 10; 11; 12 ]

[<Fact>]
let ``AVX2 and scalar forward passes are bit-identical`` () =
    if Avx2.IsSupported then
        for s in netSeeds do
            let net = loadOrFail (buildSeededRefNet s)
            for inSeed in inputSeeds do
                let input = makeInput inSeed
                Assert.Equal(runKernel false net input, runKernel true net input)

[<Fact>]
let ``forward pass matches the integer reference (default dispatch)`` () =
    for s in netSeeds do
        let raw = buildSeededRefNet s
        let net = loadOrFail raw
        for inSeed in inputSeeds do
            let input = makeInput inSeed
            Assert.Equal(refForward raw input, runKernel useAvx2Default net input)

[<Fact>]
let ``scalar path matches the integer reference (independent of AVX2 availability)`` () =
    for s in netSeeds do
        let raw = buildSeededRefNet s
        let net = loadOrFail raw
        for inSeed in inputSeeds do
            let input = makeInput inSeed
            Assert.Equal(refForward raw input, runKernel false net input)

[<Fact>]
let ``zero-weight net returns L5 bias / quantScale regardless of input`` () =
    // all weights/biases 0 except L5B=1000, quantScale=100 -> every layer clips to 0, output = 1000/100 = 10.
    let net = loadOrFail { zeroNet 100 with L5B = 1000 }
    Assert.Equal(10, runKernel useAvx2Default net (makeInput 5))
    Assert.Equal(10, runKernel useAvx2Default net (zeroInput ()))

[<Fact>]
let ``garbage in the input padding tail does not affect the result`` () =
    let net = loadOrFail (buildSeededRefNet 99)
    let clean = makeInput 33
    let dirty = Array.copy clean

    for i in InputSize .. PaddedL1 - 1 do
        dirty.[i] <- 255uy // garbage where the loader zero-filled the L1 weight columns

    Assert.Equal(runKernel useAvx2Default net clean, runKernel useAvx2Default net dirty)

[<Fact>]
let ``the all-127 input with full-range weights never saturates (kernel == reference)`` () =
    // edge case for the int16 no-saturation invariant: input=127, |weight| up to 128.
    let raw = buildSeededRefNet 4
    let net = loadOrFail raw
    let input = Array.zeroCreate PaddedL1

    for i in 0 .. InputSize - 1 do
        input.[i] <- 127uy

    Assert.Equal(refForward raw input, runKernel useAvx2Default net input)
    if Avx2.IsSupported then
        Assert.Equal(runKernel false net input, runKernel true net input)

[<Fact>]
let ``nonzero L1 shift preserves dynamic range instead of immediate clipping`` () =
    let raw =
        { zeroNet 1 with
            Shift1 = 12
            L1W =
                let a = Array.zeroCreate (L1Size * InputSize)
                for i in 0 .. InputSize - 1 do
                    a.[i] <- 1y
                a
            L2W =
                let a = Array.zeroCreate (L2Size * L1Size)
                a.[0] <- 1y
                a
            L3W =
                let a = Array.zeroCreate (L3Size * L2Size)
                a.[0] <- 1y
                a
            L4W =
                let a = Array.zeroCreate (L4Size * L3Size)
                a.[0] <- 1y
                a
            L5W =
                let a = Array.zeroCreate L4Size
                a.[0] <- 1s
                a }

    let net = loadOrFail raw
    let input = Array.zeroCreate PaddedL1

    for i in 0 .. InputSize - 1 do
        input.[i] <- 127uy

    let expected = ((InputSize * 127) + (1 <<< 11)) >>> 12
    Assert.InRange(expected, 1, 126)
    Assert.Equal(expected, refForward raw input)
    Assert.Equal(expected, runKernel false net input)
    Assert.Equal(expected, runKernel useAvx2Default net input)

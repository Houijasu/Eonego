module Eonego.Tests.NnueLoaderTests

open System.IO
open Xunit
open Eonego.NnueNetwork
open Eonego.Tests.NnueTestFixtures

let private isFailed =
    function
    | Failed _ -> true
    | Loaded _ -> false

/// Patch a little-endian uint32 at byte offset `off` in a fresh copy of `b`.
let private patchU32 (b: byte[]) (off: int) (v: uint32) : byte[] =
    let c = Array.copy b
    c.[off] <- byte (v &&& 0xFFu)
    c.[off + 1] <- byte ((v >>> 8) &&& 0xFFu)
    c.[off + 2] <- byte ((v >>> 16) &&& 0xFFu)
    c.[off + 3] <- byte ((v >>> 24) &&& 0xFFu)
    c

[<Fact>]
let ``loadBytes(serialize net) round-trips the header and pads weights into kernel layout`` () =
    let raw = buildSeededRefNet 5

    match loadBytes (serialize raw) with
    | Failed r -> Assert.Fail("expected Loaded, got Failed: " + r)
    | Loaded net ->
        Assert.Equal(Version, net.Version)
        Assert.Equal(raw.QuantScale, net.QuantScale)
        Assert.Equal(raw.Shift1, net.Shift1)
        Assert.Equal(raw.Shift2, net.Shift2)
        Assert.Equal(raw.Shift3, net.Shift3)
        Assert.Equal(raw.Shift4, net.Shift4)
        Assert.Equal(raw.L5B, net.L5B0)
        Assert.True((raw.L1B = net.L1B))
        Assert.True((raw.L2B = net.L2B))
        Assert.True((raw.L3B = net.L3B))
        Assert.True((raw.L4B = net.L4B))
        Assert.True((raw.L5W = net.L5W))
        // L1: row o cols [0,InputSize) copied, [InputSize,PaddedL1) zero-filled.
        for o in [ 0; 1; L1Size - 1 ] do
            for i in [ 0; 1; InputSize - 1 ] do
                Assert.Equal(raw.L1W.[o * InputSize + i], net.L1W.[o * PaddedL1 + i])
            for i in InputSize .. PaddedL1 - 1 do
                Assert.Equal(0y, net.L1W.[o * PaddedL1 + i])
        // L4: row o cols [0,L4Size) copied, [L4Size,L4PaddedIn) zero-filled.
        for o in [ 0; L4Size - 1 ] do
            for i in 0 .. L3Size - 1 do
                Assert.Equal(raw.L4W.[o * L3Size + i], net.L4W.[o * L4PaddedIn + i])
            for i in L3Size .. L4PaddedIn - 1 do
                Assert.Equal(0y, net.L4W.[o * L4PaddedIn + i])

[<Fact>]
let ``loader rejects bad magic (Failed, never throws)`` () =
    let bytes = serialize (buildSeededRefNet 1)
    bytes.[0] <- byte 'X'
    Assert.True(isFailed (loadBytes bytes))

[<Fact>]
let ``loader rejects every architecture mismatch and a non-positive quantScale`` () =
    let good = serialize (buildSeededRefNet 1)
    Assert.True(isFailed (loadBytes (patchU32 good 8 1u))) // v1 rejected (offset 8)
    Assert.True(isFailed (loadBytes (patchU32 good 8 3u))) // v3 rejected (offset 8)
    Assert.True(isFailed (loadBytes (patchU32 good 12 2576u))) // inputSize (12)
    Assert.True(isFailed (loadBytes (patchU32 good 16 63u))) // l1Size (16)
    Assert.True(isFailed (loadBytes (patchU32 good 20 31u))) // l2Size (20)
    Assert.True(isFailed (loadBytes (patchU32 good 24 15u))) // l3Size (24)
    Assert.True(isFailed (loadBytes (patchU32 good 28 17u))) // l4Size (28)
    Assert.True(isFailed (loadBytes (patchU32 good 32 2u))) // outputSize (32)
    Assert.True(isFailed (loadBytes (patchU32 good 36 0u))) // quantScale = 0 (36)
    Assert.True(isFailed (loadBytes (patchU32 good 40 32u))) // shift1 out of range (40)
    Assert.True(isFailed (loadBytes (patchU32 good 44 32u))) // shift2 out of range (44)
    Assert.True(isFailed (loadBytes (patchU32 good 48 32u))) // shift3 out of range (48)
    Assert.True(isFailed (loadBytes (patchU32 good 52 32u))) // shift4 out of range (52)
    // sanity: the unpatched buffer still loads.
    Assert.False(isFailed (loadBytes good))

[<Fact>]
let ``loader rejects a truncated payload`` () =
    let good = serialize (buildSeededRefNet 1)
    let truncated = good.[0 .. good.Length - 101] // chop 100 bytes off the weights
    Assert.True(isFailed (loadBytes truncated))

[<Fact>]
let ``loader rejects empty / too-short buffers without throwing`` () =
    Assert.True(isFailed (loadBytes [||]))
    Assert.True(isFailed (loadBytes (Array.zeroCreate 8))) // magic-length only

[<Fact>]
let ``load returns Failed for a missing file (never throws)`` () =
    Assert.True(isFailed (load "Z:\\no\\such\\path\\eonego_net.nnue"))

[<Fact>]
let ``load reads a valid net from disk`` () =
    let path = Path.GetTempFileName()

    try
        File.WriteAllBytes(path, serialize (buildSeededRefNet 3))
        Assert.False(isFailed (load path))
    finally
        File.Delete path

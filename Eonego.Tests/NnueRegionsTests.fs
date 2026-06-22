module Eonego.Tests.NnueRegionsTests

open Xunit
open Eonego.Bitboard
open Eonego.NnueRegions

// ---------------------------------------------------------------------------
// Step-1 guardrails for the NNUE region-feature table (the highest-risk math:
// the regionSizeOffset prefix sums and the per-square coverage lists). These
// pin the closed-form invariants so an off-by-one in initRegions fails the build
// before any Position wiring lands.
// ---------------------------------------------------------------------------

[<Fact>]
let ``region constants match the design (204 regions, 12 channels, 2448 accumulator)`` () =
    Assert.Equal(204, RegionCount)
    Assert.Equal(12, Channels)
    Assert.Equal(2448, AccSize)
    Assert.Equal(RegionCount * Channels, AccSize)

[<Fact>]
let ``incidence identity: total square coverage = sum k^2 (9-k)^2 = 1968`` () =
    let closedForm = [ for k in 1..8 -> k * k * (9 - k) * (9 - k) ] |> List.sum
    Assert.Equal(1968, closedForm) // pin the closed form itself
    let mutable total = 0
    for sq in 0..63 do
        total <- total + (regionsAt sq).Length
    Assert.Equal(closedForm, total) // builder matches it

[<Fact>]
let ``corner a1 touches 8 regions, center d4 touches 60`` () =
    Assert.Equal(8, (regionsAt (mkSquare 0 0)).Length) // a1 = file 0, rank 0
    Assert.Equal(60, (regionsAt (mkSquare 3 3)).Length) // d4 = file 3, rank 3

[<Fact>]
let ``every region index is in [0,204) and channel index = region*12 + piece`` () =
    for sq in 0..63 do
        for r in regionsAt sq do
            Assert.InRange(r, 0, RegionCount - 1)
    Assert.Equal(0, channelIndex 0 0) // region 0, white pawn (pc=0)
    Assert.Equal(11, channelIndex 0 11) // region 0, black king (pc=11)
    Assert.Equal(12, channelIndex 1 0) // region 1, white pawn

[<Fact>]
let ``activate then deactivate restores a zeroed accumulator`` () =
    let acc: sbyte[] = Array.zeroCreate AccSize
    let pc = makePiece White Knight
    let sq = mkSquare 4 4 // e5
    activate acc pc sq
    Assert.True(Array.exists (fun (v: sbyte) -> v <> 0y) acc) // something changed
    deactivate acc pc sq
    Assert.True(Array.forall (fun (v: sbyte) -> v = 0y) acc) // fully restored

[<Fact>]
let ``move equals deactivate(from) + activate(dst) on the same channel`` () =
    let pc = makePiece Black Rook
    let from = mkSquare 0 0 // a1
    let dst = mkSquare 7 7 // h8
    let viaMove: sbyte[] = Array.zeroCreate AccSize
    let viaPair: sbyte[] = Array.zeroCreate AccSize
    move viaMove pc from dst
    deactivate viaPair pc from
    activate viaPair pc dst
    Assert.True((viaMove = viaPair)) // F# structural array equality

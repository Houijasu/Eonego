module Eonego.Tests.SfAccumulatorTests

open Xunit
open Eonego.Bitboard
open Eonego.SfAccumulator
open Eonego.Move
open Eonego.Position
open Eonego.MoveGeneration
open Eonego.SfNnue
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

let private assertAccMatches (net: SfNetwork) (pos: Position) =
    let wA, wP = accumulatorOf net pos White
    let bA, bP = accumulatorOf net pos Black
    Assert.Equal<int[]>(wA, pos.SfWhiteAcc)
    Assert.Equal<int[]>(wP, pos.SfWhitePsqt)
    Assert.Equal<int[]>(bA, pos.SfBlackAcc)
    Assert.Equal<int[]>(bP, pos.SfBlackPsqt)

// Depth 2 keeps the from-scratch oracle cost reasonable while hitting castling/promotion/EP/king-refresh
// (all occur at ply 1 in the chosen FENs; Kiwipete has legal O-O / O-O-O).
let rec private walk (net: SfNetwork) (pos: Position) (depth: int) =
    assertAccMatches net pos
    if depth > 0 then
        let buf = Array.zeroCreate<Move> 256
        let span = System.Span<Move>(buf, 0, 256)
        let n = generateLegal pos span
        for i in 0 .. n - 1 do
            pos.Make buf.[i]
            walk net pos (depth - 1)
            pos.Unmake buf.[i]
            assertAccMatches net pos

[<Fact>]
let ``incremental accumulator equals from-scratch over a make/unmake walk`` () =
    match tryLoadSfNet () with
    | None -> ()
    | Some net ->
        let fens =
            [ "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
              "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
              "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1"
              "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3" ]
        for fen in fens do
            let pos = Position.OfFen fen
            pos.EnableSfNnue net.FtWeights net.FtPsqt net.FtBiases
            walk net pos 2

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

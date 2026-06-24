/// Stockfish HalfKAv2_hm `.nnue` loader tests. Validates that Eonego parses a REAL Stockfish net
/// file (SF16-era `nn-5af11540bbfe.nnue`, version 0x7AF32F20, L1=1536) byte-for-byte: the header,
/// the COMPRESSED_LEB128 feature-transformer arrays, and the 8 fc-stacks all consume to EOF.
///
/// The 40 MB net is NOT committed (CC0 but large) — these tests SOFT-SKIP when it is absent, so CI
/// without the file stays green. Drop the net at `<repo>/nets/sf16.nnue` to exercise them.
module Eonego.Tests.SfNnueTests

open System
open System.IO
open System.Text
open Xunit
open Eonego.Bitboard
open Eonego.Position
open Eonego.SfNnue

/// Walk up from the test assembly to the repo root (the dir holding Eonego.slnx), then nets/sf16.nnue.
let private netPath () : string option =
    let mutable dir = DirectoryInfo(AppContext.BaseDirectory)
    let mutable found = None

    while found.IsNone && not (isNull dir) do
        if File.Exists(Path.Combine(dir.FullName, "Eonego.slnx")) then
            found <- Some dir.FullName

        dir <- dir.Parent

    match found with
    | Some root ->
        let p = Path.Combine(root, "nets", "sf16.nnue")
        if File.Exists p then Some p else None
    | None -> None

let private withNet (f: SfNetwork -> unit) =
    match netPath () with
    | None -> () // soft-skip: net not present
    | Some p ->
        match load p with
        | Failed reason -> Assert.Fail("SF net failed to load: " + reason)
        | Loaded net -> f net

[<Fact>]
let ``loads the SF16 header (version + feature-transformer hash)`` () =
    withNet (fun net ->
        Assert.Equal(SfVersion, net.Version)
        // ft_hash = HalfKAv2_hm feature-set hash (0x7F234CB8) XOR (L1*2) = 0x7F2340B8 for L1=1536.
        Assert.Equal(0x7F2340B8u, net.FtHash)
        Assert.Contains("nnue-pytorch", net.Desc))

[<Fact>]
let ``feature-transformer arrays have the exact HalfKAv2_hm dimensions`` () =
    withNet (fun net ->
        Assert.Equal(L1, net.FtBiases.Length)
        Assert.Equal(NumInputs * L1, net.FtWeights.Length)
        Assert.Equal(NumInputs * PsqtBuckets, net.FtPsqt.Length))

[<Fact>]
let ``parses all eight fc layer-stacks with the right shapes`` () =
    withNet (fun net ->
        Assert.Equal(LayerStacks, net.Stacks.Length)

        for s in net.Stacks do
            Assert.Equal(Fc0Out * L1, s.Fc0W.Length)
            Assert.Equal(Fc0Out, s.Fc0B.Length)
            Assert.Equal(Fc1Out * Fc1In, s.Fc1W.Length)
            Assert.Equal(Fc1Out, s.Fc1B.Length)
            Assert.Equal(Fc2In, s.Fc2W.Length)
            Assert.Equal(1, s.Fc2B.Length))

[<Fact>]
let ``weights are non-trivial (loader didn't zero-fill)`` () =
    withNet (fun net ->
        // A real net has a spread of feature-transformer weights; assert it isn't all zero.
        Assert.Contains(net.FtWeights, (fun w -> w <> 0s))
        Assert.Contains(net.Stacks.[0].Fc0W, (fun w -> w <> 0y)))

// ---------------------------------------------------------------------------
// Bit-exact inference parity vs real Stockfish 16 `eval` ("NNUE evaluation", white-side pawns).
// SF prints (psqt+positional)/16/NormalizeToPawnValue (328 for SF16), rounded to 0.01 pawn.
// ---------------------------------------------------------------------------
[<Literal>]
let private NormalizeToPawnValue = 328.0

let private cases =
    [ "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 0.26
      "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", 0.04
      "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", -1.20
      "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", 0.14
      "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 0.22
      "4k3/8/8/3q4/4P3/8/8/3RK3 w - - 0 1", -0.03 ]

[<Fact>]
let ``inference matches Stockfish 16 NNUE eval (white-side pawns, +/-0.02)`` () =
    withNet (fun net ->
        let sb = StringBuilder()
        let mutable allOk = true

        for (fen, expected) in cases do
            let pos = Position.OfFen fen
            let internalStm = evalInternal net pos
            let vStm = internalStm / 16 // SF: NNUE::evaluate(pos,false) = (psqt+positional)/OutputScale
            let vWhite = if pos.SideToMove = White then vStm else -vStm
            let pawns = float vWhite / NormalizeToPawnValue
            let diff = abs (pawns - expected)

            if diff > 0.02 then
                allOk <- false

            sb.AppendLine(sprintf "  %-66s mine=%+.3f  sf=%+.2f  diff=%.3f" fen pawns expected diff)
            |> ignore

        Assert.True(allOk, "\nNNUE inference parity (white-side pawns):\n" + sb.ToString()))

[<Fact>]
let ``evalCp is near zero at startpos and large-positive when up a rook`` () =
    withNet (fun net ->
        let start = Position.OfFen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        let s = evalCp net start
        Assert.True(abs s < 150, sprintf "startpos evalCp should be small, got %d" s)
        // White to move, Black is missing its a8 rook -> clearly winning for White (stm).
        let upRook = Position.OfFen "1nbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQk - 0 1"
        let u = evalCp net upRook
        Assert.True(u > 300, sprintf "up-a-rook evalCp should be large positive, got %d" u)
        // Clamp holds.
        Assert.True(abs (evalCp net start) <= EvalMax))

[<Fact>]
let ``evalInternal with the incremental accumulator equals the from-scratch eval`` () =
    withNet (fun net ->
        for fen in
            [ "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
              "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1" ] do
            let fromScratch = Position.OfFen fen          // SfActive = false -> buildAcc path
            let incremental = Position.OfFen fen
            incremental.EnableSfNnue net.FtWeights net.FtPsqt net.FtBiases  // SfActive = true -> maintained path
            Assert.Equal(evalInternal net fromScratch, evalInternal net incremental))

[<Fact>]
let ``fc0Gemv AVX2 equals scalar over random ft+weights`` () =
    let rng = System.Random(777)
    let ft = Array.init 1536 (fun _ -> byte (rng.Next(0, 128)))
    let w = Array.init (16 * 1536) (fun _ -> sbyte (rng.Next(-128, 128)))
    let b = Array.init 16 (fun _ -> rng.Next(-100000, 100000))
    let outScalar = Array.zeroCreate<int> 16
    let outAvx2 = Array.zeroCreate<int> 16
    fc0Gemv false ft w b outScalar
    fc0Gemv true ft w b outAvx2
    Assert.Equal<int[]>(outScalar, outAvx2)

[<Fact>]
let ``ftProduct AVX2 equals scalar over random accumulators`` () =
    let rng = System.Random(2024)
    let accUs = Array.init 1536 (fun _ -> rng.Next(-300, 300))
    let accThem = Array.init 1536 (fun _ -> rng.Next(-300, 300))
    let ftScalar = Array.zeroCreate<byte> 1536
    let ftAvx2 = Array.zeroCreate<byte> 1536
    ftProduct false accUs accThem ftScalar
    ftProduct true accUs accThem ftAvx2
    Assert.Equal<byte[]>(ftScalar, ftAvx2)

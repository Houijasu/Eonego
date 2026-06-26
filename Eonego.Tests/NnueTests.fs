/// Stockfish-master "FullThreats" NNUE loader + evaluator tests. The net (nn-f8a759c05f9f.nnue, version
/// 0x6A448AFA, ~90 MB) is NOT committed (CC0 but large) — these SOFT-SKIP when absent. Without a way to run
/// real Stockfish here we cannot bit-exact-verify the inference, so these are STRUCTURAL (loads-to-EOF +
/// dimensions) and SANITY (startpos balanced, up-a-rook large). The parity scaffold at the bottom is ready
/// for reference "NNUE evaluation" pawn values from real Stockfish on this net.
module Eonego.Tests.NnueTests

open System
open System.IO
open Xunit
open Eonego.Bitboard
open Eonego.Position
open Eonego.Nnue
open Eonego.Tests.TestFixtures

let private netPath () : string option =
    let mutable dir = DirectoryInfo(AppContext.BaseDirectory)
    let mutable found = None

    while found.IsNone && not (isNull dir) do
        if File.Exists(Path.Combine(dir.FullName, "Eonego.slnx")) then
            found <- Some dir.FullName

        dir <- dir.Parent

    match found with
    | Some root ->
        let p = Path.Combine(root, "nets", "nn-f8a759c05f9f.nnue")
        if File.Exists p then Some p else None
    | None -> None

let private withNet (f: SfNetwork -> unit) =
    match netPath () with
    | None -> () // soft-skip: net not present
    | Some p ->
        match load p with
        | Failed reason -> Assert.Fail("FullThreats net failed to load: " + reason)
        | Loaded net -> f net

[<Fact>]
let ``loads to EOF with the FullThreats version`` () =
    withNet (fun net ->
        // A clean `Loaded` already means the parse consumed exactly to EOF (layout/dimension proof).
        Assert.Equal(SfVersion, net.Version))

[<Fact>]
let ``feature-transformer arrays have the dual-input FullThreats dimensions`` () =
    withNet (fun net ->
        Assert.Equal(L1, net.FtBiases.Length)
        Assert.Equal(HalfKaDims * L1, net.Weights.Length)
        Assert.Equal(ThreatDims * L1, net.ThreatWeights.Length)
        Assert.Equal(HalfKaDims * PsqtBuckets, net.PsqtWeights.Length)
        Assert.Equal(ThreatDims * PsqtBuckets, net.ThreatPsqtWeights.Length))

[<Fact>]
let ``parses all eight fc layer-stacks with the new shapes`` () =
    withNet (fun net ->
        Assert.Equal(LayerStacks, net.Stacks.Length)

        for s in net.Stacks do
            Assert.Equal(Fc0Out * L1, s.Fc0W.Length)
            Assert.Equal(Fc0Out, s.Fc0B.Length)
            Assert.Equal(Fc1Out * Fc1In, s.Fc1W.Length)
            Assert.Equal(Fc1Out, s.Fc1B.Length)
            Assert.Equal(Fc2In, s.Fc2W.Length)
            Assert.Equal(1, s.Fc2B.Length))

// HARD self-consistency gate for Phase 2: the incremental accumulator (a BOUND position) must produce the
// EXACT same eval as the from-scratch oracle (an UNBOUND position replaying the same moves) at every node of
// a make/unmake walk — covering captures, castling, en passant, promotions, and king moves (full refresh).
[<Fact>]
let ``incremental accumulator equals from-scratch over a make/unmake walk`` () =
    withNet (fun net ->
        let fens =
            [ "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1" // kiwipete (castling/captures)
              "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3" // en passant available
              "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1" // promotions + king moves
              "8/8/8/3k4/8/3K4/4P3/8 w - - 0 1" ] // king-move endgame

        let rec walk (b: Position) (o: Position) (depth: int) =
            Assert.Equal(evalCp net o, evalCp net b) // from-scratch (unbound) == incremental (bound)

            if depth > 0 then
                for m in collectLegal b do
                    b.Make m
                    o.Make m
                    walk b o (depth - 1)
                    b.Unmake m
                    o.Unmake m

        for fen in fens do
            let bound = Position.OfFen fen
            bindNnue net bound // sfActive -> incremental
            let oracle = Position.OfFen fen // unbound -> from-scratch
            walk bound oracle 2)

// The AVX2 forward kernels (ftProduct/fc0/fc1/fc2) MUST be bit-identical to the scalar reference. evalInternal
// takes an explicit useAvx2 so both paths run in ONE process on the SAME maintained accumulator — any delta is
// a SIMD-kernel bug. Walks several positions to exercise varied fc0/conc/fc1 magnitudes.
[<Fact>]
let ``forward AVX2 path equals scalar path bit-exactly`` () =
    if System.Runtime.Intrinsics.X86.Avx2.IsSupported then
        withNet (fun net ->
            let fens =
                [ "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" // startpos
                  "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1" // kiwipete
                  "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3" // en passant
                  "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1" ] // sparse endgame

            let rec walk (b: Position) (depth: int) =
                Assert.Equal(evalInternal net b false, evalInternal net b true) // scalar == AVX2

                if depth > 0 then
                    for m in collectLegal b do
                        b.Make m
                        walk b (depth - 1)
                        b.Unmake m

            for fen in fens do
                let bound = Position.OfFen fen
                bindNnue net bound
                walk bound 2)

[<Fact>]
let ``weights are non-trivial (loader didn't zero-fill)`` () =
    withNet (fun net ->
        Assert.Contains(net.Weights, (fun w -> w <> 0s))
        Assert.Contains(net.ThreatWeights, (fun w -> w <> 0y))
        Assert.Contains(net.Stacks.[0].Fc0W, (fun w -> w <> 0y)))

[<Fact>]
let ``evalCp is roughly balanced at startpos and large-positive when up a rook`` () =
    withNet (fun net ->
        let start = Position.OfFen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        let s = evalCp net start
        // STRUCTURAL sanity only (NormalizeToPawnValue uncalibrated): assert sane magnitude. The message
        // surfaces the actual value for calibration.
        Assert.True(abs s < 300, sprintf "startpos evalCp should be small, got %d cp" s)

        // White to move, Black missing its a8 rook -> clearly winning for White (stm).
        let upRook = Position.OfFen "1nbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQk - 0 1"
        let u = evalCp net upRook
        Assert.True(u > 150, sprintf "up-a-rook evalCp should be large positive, got %d cp" u)
        Assert.True(abs (evalCp net start) <= EvalMax))

// ---------------------------------------------------------------------------
// PARITY SCAFFOLD (the only TRUE correctness gate). Fill `cases` with (FEN, white-side pawn value) from
// running real Stockfish (with this net): `position fen <FEN>` then `eval`, read the final "NNUE evaluation"
// pawn number. Then this asserts Eonego matches to +/-0.05 pawn. Empty => no-op until provided.
// ---------------------------------------------------------------------------
[<Literal>]
let private NormalizeToPawnValueF = 356.0

let private parityCases: (string * float) list = [] // (fen, sf_white_pawns)

[<Fact>]
let ``inference matches real Stockfish NNUE eval (white-side pawns) when reference values are provided`` () =
    withNet (fun net ->
        for (fen, expected) in parityCases do
            let pos = Position.OfFen fen
            let cpStm = float (evalCp net pos)
            let cpWhite = if pos.SideToMove = White then cpStm else -cpStm
            let pawns = cpWhite / 100.0
            Assert.True(abs (pawns - expected) < 0.05, sprintf "%s: mine=%.2f sf=%.2f" fen pawns expected))

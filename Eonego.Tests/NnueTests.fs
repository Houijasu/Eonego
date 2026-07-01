/// "FullThreats" NNUE loader + evaluator tests. The net (nn-f8a759c05f9f.nnue, version
/// 0x6A448AFA, ~90 MB) is NOT committed (CC0 but large) — these SOFT-SKIP when absent. Without a way to run
/// a reference engine here we cannot bit-exact-verify the inference, so these are STRUCTURAL (loads-to-EOF +
/// dimensions) and SANITY (startpos balanced, up-a-rook large). The parity scaffold at the bottom is ready
/// for reference "NNUE evaluation" pawn values from a reference engine on this net.
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

let private withNet (f: Network -> unit) =
    match netPath () with
    | None -> () // soft-skip: net not present
    | Some p ->
        match load p with
        | Failed reason -> Assert.Fail("FullThreats net failed to load: " + reason)
        | Loaded net -> f net

let private assertRawAccEqualsOracle (net: Network) (bound: Position) (oracle: Position) =
    for persp in [ White; Black ] do
        let incAcc = Array.zeroCreate<int16> L1 // int16 incremental accumulator
        let incPsqt = Array.zeroCreate PsqtBuckets
        let refAcc = Array.zeroCreate L1 // int32 "true value" oracle
        let refPsqt = Array.zeroCreate PsqtBuckets
        bound.ReadAccInto(persp, Span<int16>(incAcc), Span<int>(incPsqt))
        buildAccOracle net oracle persp (Span<int>(refAcc)) (Span<int>(refPsqt))
        // int16 incremental == int16(true value), AND the true value fits int16 (the overflow gate the old
        // both-int32 test could not express).
        for j in 0 .. L1 - 1 do
            Assert.True(
                refAcc.[j] >= -32768 && refAcc.[j] <= 32767,
                sprintf "acc[%d] = %d exceeds int16 range (perspective %A)" j refAcc.[j] persp
            )

            Assert.Equal(int16 refAcc.[j], incAcc.[j])

        Assert.Equal<int[]>(refPsqt, incPsqt)

// Wide overflow gate: over a perft-style legal-move corpus the int32 "true value" accumulator must stay within
// int16 range for BOTH perspectives — the realistic guard that the trained net's sums actually fit in search.
[<Fact>]
let ``int32 oracle accumulator stays within int16 range over a deep corpus`` () =
    withNet (fun net ->
        let cases =
            [ "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 3 // startpos
              "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 2 // kiwipete
              "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 3 ] // sparse endgame

        let refAcc = Array.zeroCreate L1
        let refPsqt = Array.zeroCreate PsqtBuckets

        let check (p: Position) =
            for persp in [ White; Black ] do
                buildAccOracle net p persp (Span<int>(refAcc)) (Span<int>(refPsqt))

                for j in 0 .. L1 - 1 do
                    Assert.True(refAcc.[j] >= -32768 && refAcc.[j] <= 32767, sprintf "acc[%d]=%d exceeds int16" j refAcc.[j])

        let rec walk (p: Position) (depth: int) =
            check p

            if depth > 0 then
                for m in collectLegal p do
                    p.Make m
                    walk p (depth - 1)
                    p.Unmake m

        for (fen, depth) in cases do
            walk (Position.OfFen fen) depth)

[<Fact>]
let ``loads to EOF with the FullThreats version`` () =
    withNet (fun net ->
        // A clean `Loaded` already means the parse consumed exactly to EOF (layout/dimension proof).
        Assert.Equal(Version, net.Version))

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
            assertRawAccEqualsOracle net b o
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
            bindNnue net bound // active -> incremental
            let oracle = Position.OfFen fen // unbound -> from-scratch
            walk bound oracle 2)

[<Fact>]
let ``lazy accumulator replays unevaluated real-move chains`` () =
    withNet (fun net ->
        let fens =
            [ "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
              "8/8/8/3k4/8/3K4/4P3/8 w - - 0 1" ]

        for fen in fens do
            let bound = Position.OfFen fen
            bindNnue net bound
            let oracle = Position.OfFen fen
            let m1 = (collectLegal bound).[0]
            bound.Make m1
            oracle.Make m1
            let m2 = (collectLegal bound).[0]
            bound.Make m2
            oracle.Make m2
            assertRawAccEqualsOracle net bound oracle
            Assert.Equal(evalCp net oracle, evalCp net bound)
            bound.Unmake m2
            oracle.Unmake m2
            bound.Unmake m1
            oracle.Unmake m1
            assertRawAccEqualsOracle net bound oracle)

[<Fact>]
let ``null moves preserve top and replay across following real move`` () =
    withNet (fun net ->
        let bound = Position.OfFen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        bindNnue net bound
        let oracle = Position.OfFen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        let top0 = bound.Top
        bound.MakeNull()
        oracle.MakeNull()
        Assert.Equal(top0, bound.Top)
        assertRawAccEqualsOracle net bound oracle
        let m = (collectLegal bound).[0]
        bound.Make m
        oracle.Make m
        Assert.Equal(top0 + 1, bound.Top)
        assertRawAccEqualsOracle net bound oracle
        bound.Unmake m
        oracle.Unmake m
        Assert.Equal(top0, bound.Top)
        bound.UnmakeNull()
        oracle.UnmakeNull()
        Assert.Equal(top0, bound.Top)
        assertRawAccEqualsOracle net bound oracle)

[<Fact>]
let ``changed threat indices are materialized eagerly during make`` () =
    withNet (fun net ->
        let fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
        let scout = Position.OfFen fen
        let dirty = Array.zeroCreate Eonego.Accumulator.MaxDirtyThreats
        let move =
            collectLegal scout
            |> Array.find (fun m -> scout.DebugCollectDirtyThreats(m, dirty) > 0)
        let bound = Position.OfFen fen
        let mutable changedCalls = 0
        bound.EnableNnue
            net.FtBiases
            net.Weights
            net.PsqtWeights
            net.ThreatWeights
            net.ThreatPsqtWeights
            (System.Func<Position, int, int[], int>(fun p persp buf -> Eonego.Threats.appendActiveThreats persp p buf))
            (System.Func<Position, int[], int[], int64>(fun p bw bb -> Eonego.Threats.appendActiveThreatsBoth p bw bb))
            (System.Func<Position, int[], int, int, int[], int[], int64>(fun p dirty off n bw bb ->
                changedCalls <- changedCalls + 1
                Eonego.Threats.appendChangedThreatsBothAt p dirty off n bw bb))
        // With eager accumulator updates, the changed-threat delegate fires DURING Make (not deferred to eval).
        bound.Make move
        Assert.True(changedCalls > 0, "eager Make must have converted dirty threats")
        evalCp net bound |> ignore
        bound.Unmake move)


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
                let scalar = evalInternal net b false false
                Assert.Equal(scalar, evalInternal net b true false) // scalar == AVX2

                if System.Runtime.Intrinsics.X86.AvxVnni.IsSupported then
                    Assert.Equal(scalar, evalInternal net b true true) // scalar == AVX2 + VNNI (vpdpbusd)

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
// running a reference engine (with this net): `position fen <FEN>` then `eval`, read the final "NNUE evaluation"
// pawn number. Then this asserts Eonego matches to +/-0.05 pawn. Empty => no-op until provided.
// ---------------------------------------------------------------------------
[<Literal>]
let private NormalizeToPawnValueF = 356.0

let private parityCases: (string * float) list = [] // (fen, white_pawns)

[<Fact>]
let ``inference matches a reference NNUE eval (white-side pawns) when reference values are provided`` () =
    withNet (fun net ->
        for (fen, expected) in parityCases do
            let pos = Position.OfFen fen
            let cpStm = float (evalCp net pos)
            let cpWhite = if pos.SideToMove = White then cpStm else -cpStm
            let pawns = cpWhite / 100.0
            Assert.True(abs (pawns - expected) < 0.05, sprintf "%s: mine=%.2f expected=%.2f" fen pawns expected))

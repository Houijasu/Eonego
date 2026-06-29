/// Golden regression for the NNUE evaluator's NUMERIC output. Locks `evalInternal` (the raw SF-Value,
/// side-to-move-relative, BEFORE the centipawn/NormalizeToPawnValue conversion) to fixed integers captured
/// from the current build. Every performance phase (VNNI, bounds-check elimination, int16 accumulator, ...)
/// MUST reproduce these exactly — this is the bit-exactness gate that proves a perf change did not alter eval.
///
/// Why `evalInternal` (not `evalCp`): it is independent of `NormalizeToPawnValue`, so a future calibration
/// tweak does not churn the golden values, and it is the value the SIMD kernels actually produce.
///
/// Capture protocol: when `golden` is empty (or length-mismatched) the test writes the computed values to
/// `<repoRoot>/eonego-golden-eval.txt` and SKIPS the assert. Run once, paste the file's integers into
/// `golden`, delete the file. SOFT-SKIPS entirely if the net is absent.
module Eonego.Tests.GoldenEvalTests

open System
open System.IO
open Xunit
open Eonego.Position
open Eonego.Nnue

let private repoRoot () : string option =
    let mutable dir = DirectoryInfo(AppContext.BaseDirectory)
    let mutable found = None

    while found.IsNone && not (isNull dir) do
        if File.Exists(Path.Combine(dir.FullName, "Eonego.slnx")) then
            found <- Some dir.FullName

        dir <- dir.Parent

    found

let private withNet (f: SfNetwork -> unit) =
    match repoRoot () with
    | None -> ()
    | Some root ->
        let p = Path.Combine(root, "nets", "nn-f8a759c05f9f.nnue")

        if File.Exists p then
            match load p with
            | Failed reason -> Assert.Fail("FullThreats net failed to load: " + reason)
            | Loaded net -> f net

// A spread of positions across occupancy buckets ((popcount-1)/4 selects the layer stack) and both side-to-move
// values, so a forward-pass bug in any one of the 8 stacks (or a perspective/STM bug) shows up here.
let private fens =
    [ "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" // startpos (32) bucket 7
      "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -" // kiwipete white
      "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq -" // kiwipete black (STM flip)
      "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq -" // Ruy Lopez (many pieces)
      "rnbq1rk1/pp2bppp/2p2n2/3p4/2PP4/2N1PN2/PP2BPPP/R1BQ1RK1 w - -" // closed middlegame
      "2r3k1/5ppp/8/8/8/8/5PPP/2R3K1 w - -" // rook endgame (10)
      "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -" // sparse endgame
      "6k1/5ppp/8/8/8/8/5PPP/6K1 w - -" // KP vs KP (8) bucket 1
      "8/8/8/3k4/8/3K4/4P3/8 w - -" // KPK (3) bucket 0
      "1nbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQk - 0 1" ] // white up the a1 rook

// Captured from the int32 AVX2 build on 2026-06-29 (branch nnue-fullthreats-perf, pre-optimization baseline).
// Empty => capture mode (writes eonego-golden-eval.txt, skips assert). Order matches `fens`.
let private golden: int list =
    [ 16 // startpos
      -312 // kiwipete white
      785 // kiwipete black
      -47 // Ruy Lopez black
      676 // closed middlegame
      -1 // rook endgame
      113 // sparse endgame
      -6 // KP vs KP
      41 // KPK
      1837 ] // up a rook

[<Fact>]
let ``golden evalInternal is stable across perf phases`` () =
    withNet (fun net ->
        let vals = fens |> List.map (fun fen -> evalInternal net (Position.OfFen fen) UseAvx2 UseVnni)

        if golden.Length = vals.Length then
            Assert.Equal<int list>(golden, vals)
        else
            // Capture mode: dump for baking into `golden`.
            match repoRoot () with
            | Some r -> File.WriteAllLines(Path.Combine(r, "eonego-golden-eval.txt"), vals |> List.map string)
            | None -> ())

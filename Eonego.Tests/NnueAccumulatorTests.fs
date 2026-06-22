module Eonego.Tests.NnueAccumulatorTests

open Xunit
open Eonego.Bitboard
open Eonego.Move
open Eonego.Position
open Eonego.MoveGeneration
open Eonego.NnueRegions
open Eonego.Tests.TestFixtures

// ---------------------------------------------------------------------------
// Step-2 guardrails for the Position-side NNUE accumulator wiring: the gated
// PutPiece/RemovePiece/MovePiece hooks, the snapshot-on-Make / restore-on-Unmake,
// and EnableNnue/RefreshNnueAcc. The oracle is a slow-but-obvious from-scratch
// recomputation over the board via public accessors.
// ---------------------------------------------------------------------------

/// From-scratch region accumulator straight off the board (the readable oracle).
let private fromScratch (p: Position) : sbyte[] =
    let acc: sbyte[] = Array.zeroCreate AccSize
    for sq in 0..63 do
        let pc = p.PieceOn sq
        if pc <> NoPiece then activate acc pc sq
    acc

let private enabledPos (fen: string) : Position =
    let p = Position()
    p.LoadFen fen // hooks off (nnueActive=false) -> accumulator stays zero
    p.EnableNnue true // allocates the slab + RefreshNnueAcc rebuilds from the loaded board
    p

[<Fact>]
let ``accumulator stays zero when NNUE is disabled (PeSTO path untouched)`` () =
    let p = Position()
    p.LoadFen perftFens.[0] // startpos, NNUE never enabled
    Assert.True(Array.forall (fun (v: sbyte) -> v = 0y) (p.NnueAccumulator))

[<Fact>]
let ``EnableNnue + LoadFen builds the accumulator == from-scratch for all CPW positions`` () =
    for fen in perftFens do
        let p = enabledPos fen
        Assert.True((p.NnueAccumulator = fromScratch p), "mismatch for " + fen)

let rec private walk (p: Position) (depth: int) =
    Assert.True((p.NnueAccumulator = fromScratch p), "incremental != from-scratch mid-walk")

    if depth > 0 then
        for m in collectLegal p do
            let before = Array.copy (p.NnueAccumulator)
            p.Make m
            walk p (depth - 1)
            p.Unmake m
            Assert.True((before = p.NnueAccumulator), "accumulator not restored after unmake: " + toUci m)

[<Fact>]
let ``incremental accumulator matches from-scratch at every node (depth 2, all CPW positions)`` () =
    for fen in perftFens do
        walk (enabledPos fen) 2

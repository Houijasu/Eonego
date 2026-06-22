/// Eonego — NNUE evaluation seam (the only module that touches BOTH Position and the network weights).
///
/// Assembles the 2577-feature input from a Position — the 2448 incrementally-maintained region counts
/// (pos.NnueAccumulator) + 1 STM bit + 128 king one-hot — into the kernel's padded 2592 buffer, runs the
/// quantized forward pass (NnueNetwork), clamps to ±EvalMax, and applies the negamax sign. Same contract as
/// Evaluation.eval: returns centipawns from the side-to-move's perspective. Compiled AFTER NnueNetwork.fs
/// (needs the kernel) and Position.fs (reads the board).
module Eonego.Nnue

#nowarn "9" // NativePtr.stackalloc / fixed — AllowUnsafeBlocks is set in the .fsproj

open Microsoft.FSharp.NativeInterop
open Eonego.Bitboard
open Eonego.Position
open Eonego.NnueNetwork

/// Clamp bound: keeps NNUE eval well below the mate band (MATE_IN_MAX_PLY = 31754) and the int16 TT eval
/// field, so a mis-scaled / malformed net can never wrap a stored score or collide with mate values.
[<Literal>]
let EvalMax = 10000

let private StmIndex = NnueRegions.AccSize // 2448
let private WKingBase = StmIndex + 1 // 2449
let private BKingBase = WKingBase + 64 // 2513

let inline private clampEval (x: int) : int =
    if x > EvalMax then EvalMax
    elif x < -EvalMax then -EvalMax
    else x

/// Fully initialize the kernel input `buf` (length PaddedL1): 2448 region counts from the accumulator, the
/// STM bit, the 128 king one-hot, and a zeroed king-block + [InputSize,PaddedL1) padding. The caller hands an
/// uninitialized stack buffer; every byte is defined here.
let assembleInto (pos: Position) (buf: nativeptr<byte>) : unit =
    let acc = pos.NnueAccumulator

    for i in 0 .. NnueRegions.AccSize - 1 do
        NativePtr.set buf i (byte acc.[i])

    NativePtr.set buf StmIndex (byte pos.SideToMove)

    for i in WKingBase .. PaddedL1 - 1 do // zero both king blocks + the padding tail
        NativePtr.set buf i 0uy

    NativePtr.set buf (WKingBase + pos.KingSquare White) 1uy
    NativePtr.set buf (BKingBase + pos.KingSquare Black) 1uy

/// NNUE static evaluation in centipawns from the side-to-move's perspective (negamax), clamped to ±EvalMax.
/// 0 B/op when the L1 accumulator is bound (hot path); falls back to assemble+forward when not.
let evaluate (net: Network) (pos: Position) : int =
    let cp =
        if pos.L1Active then
            clampEval (NnueNetwork.forwardFromL1Default net pos.L1Accumulator)
        else
            let buf = NativePtr.stackalloc<byte> PaddedL1
            assembleInto pos buf
            clampEval (NnueNetwork.forward net buf)

    if pos.SideToMove = White then cp else -cp

/// Bind precomputed L1 tables into the Position (root-only; call after EnableNnue).
let bind (net: Network) (pos: Position) : unit =
    pos.BindNnueWeights net.PieceColSum net.AuxCol net.L1B

/// Test / diagnostic seam: assemble into a fresh byte[] (allocates; NOT the hot path).
let assembleInput (pos: Position) : byte[] =
    let buf = Array.zeroCreate PaddedL1
    use p = fixed buf
    assembleInto pos p
    buf

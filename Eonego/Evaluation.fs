/// Material-only static evaluation: piece values summed white-relative, negamax sign at the leaf.
/// No PST, taper, king safety, tempo, or hand-crafted positional terms — the bootstrap eval for gen-0
/// self-play and the default when NNUE is off.
///
/// THREAD-SAFETY: no mutable state. Any number of search threads may call `eval` on distinct Position
/// instances concurrently.
module Eonego.Evaluation

open System.Runtime.CompilerServices
open Eonego.Bitboard
open Eonego.Position

[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let private materialWhite (pos: Position) : int =
    let mutable score = 0

    for pt in Pawn .. Queen do
        let v = pieceValueOf pt
        score <- score + v * popCount (pos.PiecesCT White pt)
        score <- score - v * popCount (pos.PiecesCT Black pt)

    score

/// White-relative (material, material, phase=0). Diagnostic hook; not on the hot path.
let evalTrace (pos: Position) : struct (int * int * int) =
    let m = materialWhite pos
    struct (m, m, 0)

/// Material-only eval from the side-to-move's perspective (negamax). Alias kept for tooling clarity.
let materialEval (pos: Position) : int =
    let m = materialWhite pos
    if pos.SideToMove = White then m else -m

/// Static evaluation in centipawns from the side-to-move's perspective (negamax). 0 B/op.
let eval (pos: Position) : int = materialEval pos

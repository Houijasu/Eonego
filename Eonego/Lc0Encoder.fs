/// Lc0 INPUT_CLASSICAL_112_PLANE board encoder. Position -> float32[112*64] (NCHW, plane-major).
/// Ported from lc0 src/neural/encoder.cc EncodePositionForNN (classical branch). See memory lc0-net-spec.
///
/// Layout: 8 history blocks x 13 planes (0-5 ours P..K, 6-11 theirs P..K, 12 repetition) + 8 aux:
///   104 we_can_000(queenside), 105 we_can_00(kingside), 106 they_000, 107 they_00,
///   108 black-to-move (all-ones), 109 rule50 (raw), 110 zeros, 111 all-ones.
/// Board is from the MOVER's perspective: vertical flip (sq^56 / ReverseEndianness) when black to move,
/// ours = side-to-move pieces. NO horizontal mirror (classical format). Eonego has <=1-position history on
/// the MCTS path, so h=1..7 are REPEAT-CURRENT (lc0's short-history padding); repetition planes left 0.
module Eonego.Lc0Encoder

open System
open System.Buffers.Binary
open Eonego.Bitboard
open Eonego.Position

[<Literal>]
let Planes = 112

[<Literal>]
let PlanesPerBoard = 13

[<Literal>]
let HistoryLen = 8

[<Literal>]
let AuxBase = 104

/// Fill `out` (length >= 112*64) with the 112-plane encoding of `pos`.
let encodeInto (pos: Position) (out: float32[]) : unit =
    Array.Clear(out, 0, Planes * 64)
    let stm = pos.SideToMove
    let opp = 1 - stm
    let blackToMove = (stm = Black)

    let flipbb (bb: Bitboard) : Bitboard =
        if blackToMove then BinaryPrimitives.ReverseEndianness bb else bb

    let writePlane (planeIdx: int) (bb0: Bitboard) =
        let basep = planeIdx * 64
        let mutable bb = bb0

        while bb <> 0UL do
            let sq = popLsb &bb
            out.[basep + sq] <- 1.0f

    // h=0 piece planes: ours (0..5), theirs (6..11). Plane 12 (repetition) stays 0.
    for pt in 0..5 do
        writePlane pt (flipbb (pos.PiecesCT stm pt))

    for pt in 0..5 do
        writePlane (6 + pt) (flipbb (pos.PiecesCT opp pt))

    // Repeat the current board into history slots h=1..7 (copy planes 0..12).
    for h in 1 .. HistoryLen - 1 do
        Array.blit out 0 out (h * PlanesPerBoard * 64) (PlanesPerBoard * 64)

    // Aux planes.
    let setAll (planeIdx: int) =
        let basep = planeIdx * 64
        for i in 0..63 do
            out.[basep + i] <- 1.0f

    // Mover = "we"; queenside / kingside are file-side (unchanged by the vertical flip).
    let ourQ = if stm = White then WQ else BQ
    let ourK = if stm = White then WK else BK
    let theirQ = if stm = White then BQ else WQ
    let theirK = if stm = White then BK else WK

    if pos.CanCastle ourQ then setAll (AuxBase + 0)
    if pos.CanCastle ourK then setAll (AuxBase + 1)
    if pos.CanCastle theirQ then setAll (AuxBase + 2)
    if pos.CanCastle theirK then setAll (AuxBase + 3)
    if blackToMove then setAll (AuxBase + 4)

    // rule50 (raw, broadcast); plane +6 stays zero; plane +7 all-ones.
    let r50 = float32 pos.Rule50
    let baseR = (AuxBase + 5) * 64

    for i in 0..63 do
        out.[baseR + i] <- r50

    setAll (AuxBase + 7)

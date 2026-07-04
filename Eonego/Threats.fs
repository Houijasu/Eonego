/// FullThreats feature enumeration for the NNUE ("FullThreats" net, version 0x6A448AFA).
/// Clean-room port of src/nnue/features/full_threats.{h,cpp}. Threat features encode, for each non-king
/// piece, every OCCUPIED square it attacks (own pieces = defences, enemy = attacks) plus pawn-blocked-by-pawn
/// contacts, indexed by (attacker, from, to, attacked, king-bucket, perspective). 60720 dims, <=128 active.
///
/// IMPORTANT: this module works entirely in the NNUE reference piece encoding (PieceType PAWN=1..KING=6; Piece
/// W_PAWN=1..W_KING=6, B_PAWN=9..B_KING=14, make_piece=(c<<3)+pt) because the index LUTs are generated in it.
/// Eonego pieces (color*6+type, 0..11) are converted at the enumeration boundary. Squares are LERF in both.
module Eonego.Threats

open Eonego.Bitboard
open Eonego.Position

[<Literal>]
let Dimensions = 60720

[<Literal>]
let MaxActive = 256 // The encoding caps at 128 active; 256 is a safe buffer size

// Encoded piece values present on a board (W_PAWN..W_KING, B_PAWN..B_KING). Order drives the cumulative offsets.
let private allPieces = [| 1; 2; 3; 4; 5; 6; 9; 10; 11; 12; 13; 14 |]

// numValidTargets[Piece] (PIECE_NB = 16); king has none.
let private numValidTargets = [| 0; 6; 10; 8; 8; 10; 0; 0; 0; 6; 10; 8; 8; 10; 0; 0 |]

// map[attackerType-1][attackedType-1] (PAWN..KING = type 1..6 -> row/col 0..5); -1 = excluded.
let private threatMap =
    array2D
        [ [ 0; 1; -1; 2; -1; -1 ] // PAWN
          [ 0; 1; 2; 3; 4; -1 ] // KNIGHT
          [ 0; 1; 2; 3; -1; -1 ] // BISHOP
          [ 0; 1; 2; 3; -1; -1 ] // ROOK
          [ 0; 1; 2; 3; 4; -1 ] // QUEEN
          [ -1; -1; -1; -1; -1; -1 ] ] // KING

// FullThreats OrientTBL: SQ_A1(0) for files a-d, SQ_H1(7) for files e-h. (Opposite of HalfKAv2_hm's table.)
let private orientTbl = Array.init 64 (fun sq -> if (sq % 8) < 4 then 0 else 7)

[<Literal>]
let private FileA = 0x0101010101010101UL

[<Literal>]
let private FileH = 0x8080808080808080UL

let inline private pieceType (piece: int) = piece &&& 7
let inline private color (piece: int) = piece >>> 3
let inline private makePiece (c: int) (pt: int) = (c <<< 3) + pt

let private pieceOfEonego = [| 1; 2; 3; 4; 5; 6; 9; 10; 11; 12; 13; 14 |]

/// Eonego piece (0..11) -> encoded piece. e = color*6 + type ; encoded = (color<<3) + type + 1.
let inline private ofEonego (e: int) = pieceOfEonego.[e]

/// Empty-board pseudo-attacks for an encoded non-pawn piece type (KNIGHT=2..KING=6).
let inline private pseudoAttacks (pt: int) (sq: int) : Bitboard =
    match pt with
    | 2 -> knightAttacks sq
    | 3 -> bishopAttacks sq 0UL
    | 4 -> rookAttacks sq 0UL
    | 5 -> queenAttacks sq 0UL
    | _ -> kingAttacks sq // 6

/// PawnPushOrAttacks[color][sq] = single-step push (geometric, on-board) | the two diagonal attacks.
let inline private pawnPushOrAttacks (color: int) (sq: int) : Bitboard =
    let atk = pawnAttacks color sq

    let push =
        if color = White then (if sq < 56 then 1UL <<< (sq + 8) else 0UL)
        else (if sq >= 8 then 1UL <<< (sq - 8) else 0UL)

    atk ||| push

/// The attack/push set used for LUT ordinal counting, per encoded piece.
let inline private attackSetFor (piece: int) (sq: int) : Bitboard =
    if pieceType piece = 1 then pawnPushOrAttacks (color piece) sq
    else pseudoAttacks (pieceType piece) sq

// ---------------------------------------------------------------------------
// Generated LUTs (constexpr equivalents from full_threats.cpp), built once at init.
// ---------------------------------------------------------------------------

// index_lut2[piece][from][to] = popcount of attack-targets strictly below `to`. Flat [16*64*64].
let private indexLut2 : int[] =
    let t = Array.zeroCreate (16 * 64 * 64)

    for p in allPieces do
        for from in 0..63 do
            let attacks = attackSetFor p from
            for too in 0..63 do
                let below = (if too = 0 then 0UL else ((1UL <<< too) - 1UL)) &&& attacks
                t.[(p * 64 + from) * 64 + too] <- popCount below

    t

// offsets[piece][from] (flat [16*64]) and helper cumulativePieceOffset / cumulativeOffset per piece.
let private offsets : int[] = Array.zeroCreate (16 * 64)
let private helperCumPiece : int[] = Array.zeroCreate 16
let private helperCumOffset : int[] = Array.zeroCreate 16

let private initOffsets () =
    let mutable cumulativeOffset = 0

    for piece in allPieces do
        let mutable cumPiece = 0
        let isPawn = pieceType piece = 1

        for from in 0..63 do
            offsets.[piece * 64 + from] <- cumPiece

            if not isPawn then
                cumPiece <- cumPiece + popCount (pseudoAttacks (pieceType piece) from)
            elif from >= 8 && from <= 55 then
                cumPiece <- cumPiece + popCount (pawnPushOrAttacks (color piece) from)

        helperCumPiece.[piece] <- cumPiece
        helperCumOffset.[piece] <- cumulativeOffset
        cumulativeOffset <- cumulativeOffset + numValidTargets.[piece] * cumPiece

initOffsets ()

// index_lut1[attacker][attacked][from<to ? 1 : 0]. Flat [16*16*2]. Excluded -> Dimensions (skipped).
let private indexLut1 : int[] =
    let t = Array.create (16 * 16 * 2) Dimensions

    for attacker in allPieces do
        for attacked in allPieces do
            let enemy = (attacker ^^^ attacked) = 8
            let aType = pieceType attacker
            let dType = pieceType attacked
            let m = threatMap.[aType - 1, dType - 1]
            let semiExcluded = (aType = dType) && (enemy || aType <> 1)

            let feature =
                helperCumOffset.[attacker]
                + (color attacked * (numValidTargets.[attacker] / 2) + m) * helperCumPiece.[attacker]

            let excluded = m < 0
            t.[(attacker * 16 + attacked) * 2 + 0] <- if excluded then Dimensions else feature
            t.[(attacker * 16 + attacked) * 2 + 1] <- if excluded || semiExcluded then Dimensions else feature

    t

let inline private makeIndexOriented (orientation: int) (swap: int) (attacker: int) (from: int) (too: int) (attacked: int) : int =
    let fromO = from ^^^ orientation
    let toO = too ^^^ orientation
    let attackerO = attacker ^^^ swap
    let attackedO = attacked ^^^ swap

    indexLut1.[(attackerO * 16 + attackedO) * 2 + (if fromO < toO then 1 else 0)]
    + offsets.[attackerO * 64 + fromO]
    + indexLut2.[(attackerO * 64 + fromO) * 64 + toO]

/// Feature index for a threat. `attacker`/`attacked` are encoded pieces; from/to/ksq squares.
let makeIndex (perspective: int) (attacker: int) (from: int) (too: int) (attacked: int) (ksq: int) : int =
    makeIndexOriented (orientTbl.[ksq] ^^^ (56 * perspective)) (8 * perspective) attacker from too attacked

// ---------------------------------------------------------------------------
// Active-feature enumeration (port of append_active_indices). Fills `buf`, returns count.
// ---------------------------------------------------------------------------
let appendActiveThreats (perspective: int) (pos: Position) (buf: int[]) : int =
    let ksq = pos.KingSquare perspective
    let orientation = orientTbl.[ksq] ^^^ (56 * perspective)
    let swap = 8 * perspective
    let occupied = pos.Occupied
    let allPawns = pos.Pieces Pawn
    let mutable n = 0

    let inline emit (attacker: int) (from: int) (too: int) =
        let attacked = ofEonego (pos.PieceOn too)
        let idx = makeIndexOriented orientation swap attacker from too attacked

        if idx < Dimensions && n < MaxActive then
            buf.[n] <- idx
            n <- n + 1

    for colorBit in 0..1 do
        let c = perspective ^^^ colorBit // us, then them
        // --- pawns: diagonal captures onto occupied squares + pawn blocked by a pawn in front ---
        let attackerP = makePiece c 1
        let cPawns = pos.PiecesCT c Pawn

        if c = White then
            let mutable ne = ((cPawns &&& ~~~FileH) <<< 9) &&& occupied
            while ne <> 0UL do
                let too = popLsb &ne
                emit attackerP (too - 9) too

            let mutable nw = ((cPawns &&& ~~~FileA) <<< 7) &&& occupied
            while nw <> 0UL do
                let too = popLsb &nw
                emit attackerP (too - 7) too

            let mutable push = ((allPawns >>> 8) &&& cPawns) <<< 8
            while push <> 0UL do
                let too = popLsb &push
                emit attackerP (too - 8) too
        else
            let mutable sw = ((cPawns &&& ~~~FileA) >>> 9) &&& occupied
            while sw <> 0UL do
                let too = popLsb &sw
                emit attackerP (too + 9) too

            let mutable se = ((cPawns &&& ~~~FileH) >>> 7) &&& occupied
            while se <> 0UL do
                let too = popLsb &se
                emit attackerP (too + 7) too

            let mutable push = ((allPawns <<< 8) &&& cPawns) >>> 8
            while push <> 0UL do
                let too = popLsb &push
                emit attackerP (too + 8) too

        // --- knight, bishop, rook, queen: every attack landing on an occupied square ---
        for pt in 2..5 do
            let attacker = makePiece c pt
            let mutable bb = pos.PiecesCT c (pt - 1) // encoded type -> Eonego type

            while bb <> 0UL do
                let from = popLsb &bb

                let attacks =
                    (match pt with
                     | 2 -> knightAttacks from
                     | 3 -> bishopAttacks from occupied
                     | 4 -> rookAttacks from occupied
                     | _ -> queenAttacks from occupied)
                    &&& occupied

                let mutable a = attacks
                while a <> 0UL do
                    let too = popLsb &a
                    emit attacker from too

    n

/// As `appendActiveThreats` but enumerates the (perspective-independent) physical threats ONCE and fills BOTH
/// perspective buffers, computing each perspective's `makeIndex` per threat. Returns `(nW <<< 32) ||| nB`.
/// The expensive part — slider attack-generation and the popLsb target loops — runs once instead of twice;
/// only the cheap `makeIndex` is doubled. bufW/bufB receive the SAME SETS as `appendActiveThreats White`/
/// `Black` would (physical enumeration order may differ, but the caller sorts before diffing, so the result
/// is bit-identical). Enumeration order here is fixed (white pieces then black) and perspective-free.
let appendActiveThreatsBoth (pos: Position) (bufW: int[]) (bufB: int[]) : int64 =
    let ksqW = pos.KingSquare White
    let ksqB = pos.KingSquare Black
    let orientW = orientTbl.[ksqW]
    let orientB = orientTbl.[ksqB] ^^^ 56
    let occupied = pos.Occupied
    let allPawns = pos.Pieces Pawn
    let mutable nW = 0
    let mutable nB = 0

    let inline emit (attacker: int) (from: int) (too: int) =
        let attacked = ofEonego (pos.PieceOn too)
        let wIdx = makeIndexOriented orientW 0 attacker from too attacked

        if wIdx < Dimensions && nW < MaxActive then
            bufW.[nW] <- wIdx
            nW <- nW + 1

        let bIdx = makeIndexOriented orientB 8 attacker from too attacked

        if bIdx < Dimensions && nB < MaxActive then
            bufB.[nB] <- bIdx
            nB <- nB + 1

    for c in 0..1 do
        let attackerP = makePiece c 1
        let cPawns = pos.PiecesCT c Pawn

        if c = White then
            let mutable ne = ((cPawns &&& ~~~FileH) <<< 9) &&& occupied
            while ne <> 0UL do
                let too = popLsb &ne
                emit attackerP (too - 9) too

            let mutable nw = ((cPawns &&& ~~~FileA) <<< 7) &&& occupied
            while nw <> 0UL do
                let too = popLsb &nw
                emit attackerP (too - 7) too

            let mutable push = ((allPawns >>> 8) &&& cPawns) <<< 8
            while push <> 0UL do
                let too = popLsb &push
                emit attackerP (too - 8) too
        else
            let mutable sw = ((cPawns &&& ~~~FileA) >>> 9) &&& occupied
            while sw <> 0UL do
                let too = popLsb &sw
                emit attackerP (too + 9) too

            let mutable se = ((cPawns &&& ~~~FileH) >>> 7) &&& occupied
            while se <> 0UL do
                let too = popLsb &se
                emit attackerP (too + 7) too

            let mutable push = ((allPawns <<< 8) &&& cPawns) >>> 8
            while push <> 0UL do
                let too = popLsb &push
                emit attackerP (too + 8) too

        for pt in 2..5 do
            let attacker = makePiece c pt
            let mutable bb = pos.PiecesCT c (pt - 1)

            while bb <> 0UL do
                let from = popLsb &bb

                let attacks =
                    (match pt with
                     | 2 -> knightAttacks from
                     | 3 -> bishopAttacks from occupied
                     | 4 -> rookAttacks from occupied
                     | _ -> queenAttacks from occupied)
                    &&& occupied

                let mutable a = attacks
                while a <> 0UL do
                    let too = popLsb &a
                    emit attacker from too

    (int64 nW <<< 32) ||| int64 nB

/// Convert perspective-agnostic dirty physical threat edges to signed FullThreats feature indices for both
/// perspectives. Positive encoded values mean add `(v - 1)`, negative values mean remove `(-v - 1)`.
let appendChangedThreatsBothAt (pos: Position) (dirty: int[]) (dirtyOff: int) (dirtyN: int) (bufW: int[]) (bufB: int[]) : int64 =
    let ksqW = pos.KingSquare White
    let ksqB = pos.KingSquare Black
    let orientW = orientTbl.[ksqW]
    let orientB = orientTbl.[ksqB] ^^^ 56
    let mutable nW = 0
    let mutable nB = 0

    let inline encodeSigned idx sign = if sign > 0 then idx + 1 else -idx - 1

    for i in 0 .. dirtyN - 1 do
        let signedEdge = dirty.[dirtyOff + i]
        let edge = Accumulator.dirtyThreatEdge signedEdge
        let sign = Accumulator.dirtyThreatSign signedEdge
        let attacker = ofEonego (Accumulator.dirtyThreatAttacker edge)
        let attacked = ofEonego (Accumulator.dirtyThreatAttacked edge)
        let from = Accumulator.dirtyThreatFrom edge
        let too = Accumulator.dirtyThreatTo edge
        let wIdx = makeIndexOriented orientW 0 attacker from too attacked

        if wIdx < Dimensions && nW < bufW.Length then
            bufW.[nW] <- encodeSigned wIdx sign
            nW <- nW + 1

        let bIdx = makeIndexOriented orientB 8 attacker from too attacked

        if bIdx < Dimensions && nB < bufB.Length then
            bufB.[nB] <- encodeSigned bIdx sign
            nB <- nB + 1

    (int64 nW <<< 32) ||| int64 nB

let appendChangedThreatsBoth (pos: Position) (dirty: int[]) (dirtyN: int) (bufW: int[]) (bufB: int[]) : int64 =
    appendChangedThreatsBothAt pos dirty 0 dirtyN bufW bufB

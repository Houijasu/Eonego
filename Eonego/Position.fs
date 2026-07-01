/// Eonego — mutable board state for the chess engine.
///
/// Position is a single [<Sealed>] mutable CLASS (one heap allocation reused across the whole search via
/// make/unmake — NOT copy-make). It owns Stockfish-style boards: `byTypeBB.[0..5]` = Pawn..King (both
/// colors), `byTypeBB.[AllPieces]` = full occupancy, `byColorBB.[2]`, and a `Piece[64]` mailbox. All
/// irreversible / derived undo data lives in a plain mutable [<Struct>] StateInfo stored in a preallocated
/// `StateInfo[MaxPly]` stack indexed by `stPly` (no per-node allocation, no linked list), read copy-free
/// via `let st = &states.[stPly]`.
///
/// CONTRACTS:
///  - A fresh `Position()` is EMPTY (no kings, all NoPiece, states.[0] zeroed). It MUST be initialized via
///    `LoadFen` / `SetStartPos` / `OfFen` before any `Make`/query. The ctor stays cheap on purpose.
///  - `Make` assumes `m` is a LEGAL move for the position (no self-check/duplicate-piece validation — that
///    is MoveGen's job). Passing an illegal move is undefined behaviour, as in every engine core.
///  - Position is NOT thread-safe — each search thread gets its own instance.
///  - HARD RULE: NEVER pass `&states.[stPly].Field` as a byref out-param (binds a copy / trips FS3228).
///    Compute into a `let mutable` local, then assign back into the field.
///
/// ZOBRIST KEY CONVENTION (see Zobrist.fs): key = XOR(zPiece) ^^^ zCastle rights ^^^ (zEp file iff ep)
/// ^^^ (Side iff Black to move). The three producers LoadFen / Make / RecomputeKey must agree bit-for-bit.
module Eonego.Position

open System.Diagnostics
open System.Runtime.CompilerServices
open Eonego.AccCheckpoint
open Eonego.Bitboard
open Eonego.Move
open Eonego.Zobrist

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
[<Literal>]
let AllPieces = 6 // index of the full-occupancy bitboard inside byTypeBB

[<Literal>]
let MaxPly = 1024 // undo-stack depth (search depth << this; Debug.Assert-guarded)

[<Literal>]
let StartPosFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

// ---------------------------------------------------------------------------
// Module-level static tables — built once via `do initTables ()`. These read ONLY Bitboard [<Literal>]
// consts (WK/WQ/BK/BQ), never Zobrist, so there is no cross-file static-ctor race.
//
// DUAL INDEXING (intentional): castleRookFrom/To are indexed by KING DST (make/unmake have the move's
// dst); castleKingPath/EmptyPath/RookOrigin/KingDest are indexed by the SINGLE RIGHT BIT (WK/WQ/BK/BQ,
// i.e. 1/2/4/8) because MoveGen iterates rights.
// ---------------------------------------------------------------------------
let private castlingRightsMask: int[] = Array.create 64 0xF // AND-out table; default = no right revoked
let private castleRookFrom: int[] = Array.create 64 NoSquare // by king dst
let private castleRookTo: int[] = Array.create 64 NoSquare // by king dst
let private castleKingPath: Bitboard[] = Array.zeroCreate 16 // by right bit; squares king passes through OR lands on
let private castleEmptyPath: Bitboard[] = Array.zeroCreate 16 // by right bit; squares that must be empty
let private castleRookOrigin: int[] = Array.create 16 NoSquare // by right bit; rook from square
let private castleKingDest: int[] = Array.create 16 NoSquare // by right bit; king to square
let private sfRayBeyond: Bitboard[] = Array.zeroCreate (64 * 64) // [from][through] -> squares beyond through

let private bbOf (sqs: int list) : Bitboard =
    List.fold (fun acc s -> acc ||| (1UL <<< s)) 0UL sqs

let private initTables () =
    let a1, b1, c1, d1, e1, f1, g1, h1 = 0, 1, 2, 3, 4, 5, 6, 7
    let a8, b8, c8, d8, e8, f8, g8, h8 = 56, 57, 58, 59, 60, 61, 62, 63
    // Rights revoked when a king/rook leaves OR a rook is captured on a home square.
    castlingRightsMask.[e1] <- ~~~(WK ||| WQ) &&& 0xF
    castlingRightsMask.[a1] <- ~~~WQ &&& 0xF
    castlingRightsMask.[h1] <- ~~~WK &&& 0xF
    castlingRightsMask.[e8] <- ~~~(BK ||| BQ) &&& 0xF
    castlingRightsMask.[a8] <- ~~~BQ &&& 0xF
    castlingRightsMask.[h8] <- ~~~BK &&& 0xF
    // Rook leg of each castle, keyed by king destination.
    castleRookFrom.[g1] <- h1
    castleRookTo.[g1] <- f1
    castleRookFrom.[c1] <- a1
    castleRookTo.[c1] <- d1
    castleRookFrom.[g8] <- h8
    castleRookTo.[g8] <- f8
    castleRookFrom.[c8] <- a8
    castleRookTo.[c8] <- d8
    // King transit (test each vs AttackedBy them) and must-be-empty path (O-O-O includes the b-file).
    castleKingPath.[WK] <- bbOf [ f1; g1 ]
    castleEmptyPath.[WK] <- bbOf [ f1; g1 ]
    castleKingPath.[WQ] <- bbOf [ d1; c1 ]
    castleEmptyPath.[WQ] <- bbOf [ d1; c1; b1 ]
    castleKingPath.[BK] <- bbOf [ f8; g8 ]
    castleEmptyPath.[BK] <- bbOf [ f8; g8 ]
    castleKingPath.[BQ] <- bbOf [ d8; c8 ]
    castleEmptyPath.[BQ] <- bbOf [ d8; c8; b8 ]
    castleRookOrigin.[WK] <- h1
    castleKingDest.[WK] <- g1
    castleRookOrigin.[WQ] <- a1
    castleKingDest.[WQ] <- c1
    castleRookOrigin.[BK] <- h8
    castleKingDest.[BK] <- g8
    castleRookOrigin.[BQ] <- a8
    castleKingDest.[BQ] <- c8

    for from in 0..63 do
        for throughSq in 0..63 do
            if from <> throughSq then
                let df = fileOf throughSq - fileOf from
                let dr = rankOf throughSq - rankOf from

                let step =
                    if dr = 0 && df <> 0 then (if df > 0 then 1 else -1)
                    elif df = 0 && dr <> 0 then (if dr > 0 then 8 else -8)
                    elif df = dr then (if dr > 0 then 9 else -9)
                    elif df = -dr then (if dr > 0 then 7 else -7)
                    else 0

                if step <> 0 then
                    let mutable sq = throughSq + step
                    let mutable b = 0UL

                    while sq >= 0 && sq < 64 && aligned from throughSq sq do
                        b <- b ||| (1UL <<< sq)
                        sq <- sq + step

                    sfRayBeyond.[(from <<< 6) + throughSq] <- b

do initTables ()

// ---------------------------------------------------------------------------
// PieceValue — the single SEE / MVV-LVA / capture-history value table (centipawns), indexed by
// PieceType (Pawn..King). King = 0 is the SEE sentinel: a king is never an SEE victim and its value is
// never subtracted (the KING branch of see_ge returns/terminates first). This is a literal-initialized
// immutable array that reads ONLY integer literals, so it joins the same already-once static init as the
// castling tables above — no Zobrist/MoveGen reference, no new cross-module cctor edge, no lock (shared
// immutable). MovePick / History reuse it via `pieceValueOf` so SEE and ordering can never disagree.
// ---------------------------------------------------------------------------
let private pieceValue: int[] =
    //   Pawn  Knight  Bishop  Rook  Queen  King
    [| 100; 320; 330; 500; 900; 0 |]

/// Material value of a PieceType (Pawn..King); King = 0. Single source of truth for SEE + move ordering.
[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let pieceValueOf (pt: PieceType) : int = pieceValue.[pt]

// ---------------------------------------------------------------------------
// StateInfo — plain mutable [<Struct>] (NOT IsReadOnly: make writes ~17 fields). Check-squares are flat
// fields (a managed array inside a struct would heap-allocate per node). Stored in a preallocated array.
// ---------------------------------------------------------------------------
[<Struct>]
type StateInfo =
    { // irreversible / restore-on-unmake
      mutable Key: uint64
      mutable CastlingRights: int
      mutable EpSquare: Square // landing square, or NoSquare (64)
      mutable Rule50: int
      mutable PliesFromNull: int
      mutable CapturedPiece: Piece // removed by the move producing this node; NoPiece if none
      mutable Repetition: int // 0 none; search-only (perft ignores)
      // cached check-info (SF set_check_info), for the side to move AT this node
      mutable Checkers: Bitboard
      mutable BlockersW: Bitboard // pieces (either color) blocking a check on the White king
      mutable BlockersB: Bitboard
      mutable PinnersW: Bitboard // enemy sliders pinning a White piece to the White king
      mutable PinnersB: Bitboard
      mutable CheckSqP: Bitboard
      mutable CheckSqN: Bitboard
      mutable CheckSqB: Bitboard
      mutable CheckSqR: Bitboard
      mutable CheckSqQ: Bitboard } // King never gives check -> no field

// ---------------------------------------------------------------------------
// Position
// ---------------------------------------------------------------------------
[<Sealed>]
type Position() =
    let byTypeBB: Bitboard[] = Array.zeroCreate 7 // [0..5]=Pawn..King, [6]=ALL_PIECES
    let byColorBB: Bitboard[] = Array.zeroCreate 2
    let board: Piece[] = Array.create 64 NoPiece // NOT zeroCreate (0 = white pawn!)
    let mutable sideToMove = White
    let mutable gamePly = 0
    let mutable currentKey = 0UL
    let states: StateInfo[] = Array.zeroCreate MaxPly
    let mutable stPly = 0

    // --- SF FullThreats NNUE lazy accumulator (gated by sfActive) ----------------------------------------
    // MERGED accumulator: HalfKA and FullThreats both add into the same L1 cells. Make records dirty frame
    // metadata; eval/read materializes the current sfTop frame on demand. Dirty threats are physical edges,
    // converted to perspective-dependent indices through delegates bound by NNUE.fs to avoid Position->Threats.
    [<Literal>]
    let SfMaxThreats = 256

    [<Literal>]
    let SfMaxPly = 246

    let mutable sfActive = false
    let mutable sfEagerUpdates = false
    let mutable sfTop = 0
    // Phase 1 — optional lock-free NNUE accumulator checkpoint cache. When non-null, `SfEnsureBothComputed`
    // consults it as a fast-path before walking the lazy frame stack, and populates it on a successful
    // materialization. Owned by `SearchControl`, bound per-worker via `SfBindCheckpoint`; cleared via the
    // owning table's `Clear()` between searches (NOT here — the position may outlive multiple searches).
    let mutable sfCheckpoint: AccCheckpointTable = null
    let mutable sfAccW: int16[] = Array.empty
    let mutable sfAccB: int16[] = Array.empty
    let mutable sfPsqW: int[] = Array.empty
    let mutable sfPsqB: int[] = Array.empty
    let mutable sfComputedW: bool[] = Array.empty
    let mutable sfComputedB: bool[] = Array.empty
    let sfTmpW: int[] = Array.zeroCreate SfMaxThreats // enumeration scratch, white perspective
    let sfTmpB: int[] = Array.zeroCreate SfMaxThreats // enumeration scratch, black perspective
    let sfChangedW: int[] = Array.zeroCreate Accumulator.MaxDirtyThreats
    let sfChangedB: int[] = Array.zeroCreate Accumulator.MaxDirtyThreats
    let mutable sfBiases: int16[] = Array.empty
    let mutable sfHalfWeights: int16[] = Array.empty
    let mutable sfHalfPsqt: int[] = Array.empty
    let mutable sfThreatWeights: sbyte[] = Array.empty
    let mutable sfThreatPsqt: int[] = Array.empty
    let mutable sfThreatFn: (System.Func<Position, int, int[], int>) | null = null
    let mutable sfThreatFnBoth: (System.Func<Position, int[], int[], int64>) | null = null
    let mutable sfThreatFnChangedBoth: (System.Func<Position, int[], int, int, int[], int[], int64>) | null = null
    let mutable sfFrameDirtyPc: int[] = Array.empty
    let mutable sfFrameDirtySq: int[] = Array.empty
    let mutable sfFrameDirtySign: int[] = Array.empty
    let mutable sfFrameDirtyN: int[] = Array.empty
    let mutable sfFrameThreats: int[] = Array.empty
    let mutable sfFrameThreatN: int[] = Array.empty
    let mutable sfFrameChangedW: int[] = Array.empty
    let mutable sfFrameChangedB: int[] = Array.empty
    let mutable sfFrameChangedNW: int[] = Array.empty
    let mutable sfFrameChangedNB: int[] = Array.empty
    let mutable sfFrameChangedValid: bool[] = Array.empty
    let mutable sfFrameChangedKsqW: int[] = Array.empty
    let mutable sfFrameChangedKsqB: int[] = Array.empty
    let mutable sfFrameWhiteKingMoved: bool[] = Array.empty
    let mutable sfFrameBlackKingMoved: bool[] = Array.empty
    let mutable sfFrameThreatOverflow: bool[] = Array.empty
    let sfDirtyPc: int[] = Array.zeroCreate Accumulator.MaxDirtyPieces
    let sfDirtySq: int[] = Array.zeroCreate Accumulator.MaxDirtyPieces
    let sfDirtySign: int[] = Array.zeroCreate Accumulator.MaxDirtyPieces
    let mutable sfDirtyN = 0
    let sfDirtyThreats: int[] = Array.zeroCreate Accumulator.MaxDirtyThreats
    let mutable sfDirtyThreatN = 0
    let mutable sfDirtyThreatOverflow = false

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member private _.SfAppendDirtyThreat(putPiece: bool, pc: Piece, attacked: Piece, from: Square, too: Square) =
        if pc <> NoPiece && attacked <> NoPiece && pieceType pc <> King && not sfDirtyThreatOverflow then
            if sfDirtyThreatN < Accumulator.MaxDirtyThreats then
                let edge = Accumulator.packDirtyThreatEdge pc from too attacked
                sfDirtyThreats.[sfDirtyThreatN] <- Accumulator.packSignedDirtyThreat edge (if putPiece then 1 else -1)
                sfDirtyThreatN <- sfDirtyThreatN + 1
            else
                sfDirtyThreatOverflow <- true

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member private _.SfRayBeyond(from: Square, throughSq: Square) : Bitboard =
        sfRayBeyond.[(from <<< 6) + throughSq]

    member private _.SfPawnPushOrAttacks(c: Color, sq: Square) : Bitboard =
        let atk = pawnAttacks c sq

        let push =
            if c = White then
                if sq < 56 then 1UL <<< (sq + 8) else 0UL
            else if sq >= 8 then
                1UL <<< (sq - 8)
            else
                0UL

        atk ||| push

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member private this.SfProcessSliders
        (
            sliders: Bitboard,
            putPiece: bool,
            pc: Piece,
            sq: Square,
            computeRay: bool,
            noRaysContaining: Bitboard,
            occupiedNoK: Bitboard,
            rAttacks: Bitboard,
            bAttacks: Bitboard,
            addDirectAttacks: bool
        ) =
        let mutable ss = sliders

        while ss <> 0UL do
            let sliderSq = popLsb &ss
            let slider = board.[sliderSq]

            if computeRay then
                let ray = this.SfRayBeyond(sliderSq, sq)
                let discovered = ray &&& (rAttacks ||| bAttacks) &&& occupiedNoK

                if discovered <> 0UL && ((ray &&& noRaysContaining) <> noRaysContaining) then
                    let threatenedSq = lsb discovered
                    this.SfAppendDirtyThreat(not putPiece, slider, board.[threatenedSq], sliderSq, threatenedSq)

            if addDirectAttacks then
                this.SfAppendDirtyThreat(putPiece, slider, pc, sliderSq, sq)

    member private this.SfUpdatePieceThreats(pc: Piece, putPiece: bool, sq: Square, computeRay: bool, noRaysContaining: Bitboard) =
        if pc <> NoPiece then
            let occupied = byTypeBB.[AllPieces]
            let rookQueens = byTypeBB.[Rook] ||| byTypeBB.[Queen]
            let bishopQueens = byTypeBB.[Bishop] ||| byTypeBB.[Queen]
            let rAttacks = rookAttacks sq occupied
            let bAttacks = bishopAttacks sq occupied
            let occupiedNoK = occupied ^^^ byTypeBB.[King]
            let sliders = (rookQueens &&& rAttacks) ||| (bishopQueens &&& bAttacks)

            if pieceType pc = King then
                if computeRay then
                    this.SfProcessSliders(sliders, putPiece, pc, sq, computeRay, noRaysContaining, occupiedNoK, rAttacks, bAttacks, false)
            else
                let pt = pieceType pc
                let c = pieceColor pc

                let mutable threatened =
                    (match pt with
                     | Pawn -> pawnAttacks c sq
                     | Knight -> knightAttacks sq
                     | Bishop -> bishopAttacks sq occupied
                     | Rook -> rookAttacks sq occupied
                     | _ -> queenAttacks sq occupied)
                    &&& occupiedNoK

                let mutable incoming = knightAttacks sq &&& byTypeBB.[Knight]

                if pt = Pawn then
                    let whiteAttacks = this.SfPawnPushOrAttacks(White, sq)
                    let blackAttacks = this.SfPawnPushOrAttacks(Black, sq)
                    threatened <- threatened ||| ((if c = White then whiteAttacks else blackAttacks) &&& byTypeBB.[Pawn])
                    incoming <- incoming ||| (whiteAttacks &&& byColorBB.[Black] &&& byTypeBB.[Pawn])
                    incoming <- incoming ||| (blackAttacks &&& byColorBB.[White] &&& byTypeBB.[Pawn])
                else
                    incoming <-
                        incoming
                        ||| (pawnAttacks White sq &&& byColorBB.[Black] &&& byTypeBB.[Pawn])
                        ||| (pawnAttacks Black sq &&& byColorBB.[White] &&& byTypeBB.[Pawn])

                let mutable t = threatened

                while t <> 0UL do
                    let too = popLsb &t
                    this.SfAppendDirtyThreat(putPiece, pc, board.[too], sq, too)

                if computeRay then
                    this.SfProcessSliders(sliders, putPiece, pc, sq, computeRay, noRaysContaining, occupiedNoK, rAttacks, bAttacks, true)
                else
                    incoming <- incoming ||| sliders

                while incoming <> 0UL do
                    let from = popLsb &incoming
                    let src = board.[from]

                    if src <> NoPiece && pieceType src <> King then
                        this.SfAppendDirtyThreat(putPiece, src, pc, from, sq)

    // --- mutation choke points (the ONLY board writers) ---------------------
    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member private this.PutPiece (pc: Piece) (sq: Square) =
        Debug.Assert(pc <> NoPiece, "PutPiece: NoPiece")
        let b = 1UL <<< sq
        let pt = pieceType pc
        let c = pieceColor pc
        byTypeBB.[pt] <- byTypeBB.[pt] ||| b
        byTypeBB.[AllPieces] <- byTypeBB.[AllPieces] ||| b
        byColorBB.[c] <- byColorBB.[c] ||| b
        board.[sq] <- pc
        currentKey <- currentKey ^^^ zPiece pc sq
        if sfActive then
            sfDirtyPc.[sfDirtyN] <- pc; sfDirtySq.[sfDirtyN] <- sq; sfDirtySign.[sfDirtyN] <- 1; sfDirtyN <- sfDirtyN + 1
            this.SfUpdatePieceThreats(pc, true, sq, true, System.UInt64.MaxValue)

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member private this.RemovePiece (pc: Piece) (sq: Square) =
        Debug.Assert(pc <> NoPiece, "RemovePiece: NoPiece")
        if sfActive then
            this.SfUpdatePieceThreats(pc, false, sq, true, System.UInt64.MaxValue)

        let b = 1UL <<< sq
        let pt = pieceType pc
        let c = pieceColor pc
        byTypeBB.[pt] <- byTypeBB.[pt] ^^^ b // bit known set -> XOR clears
        byTypeBB.[AllPieces] <- byTypeBB.[AllPieces] ^^^ b
        byColorBB.[c] <- byColorBB.[c] ^^^ b
        board.[sq] <- NoPiece
        currentKey <- currentKey ^^^ zPiece pc sq
        if sfActive then
            sfDirtyPc.[sfDirtyN] <- pc; sfDirtySq.[sfDirtyN] <- sq; sfDirtySign.[sfDirtyN] <- -1; sfDirtyN <- sfDirtyN + 1

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member private this.MovePiece (pc: Piece) (from: Square) (dst: Square) =
        Debug.Assert(pc <> NoPiece, "MovePiece: NoPiece")
        // PRE: dst is empty for pt and color c (captures call RemovePiece on dst first).
        let fromTo = (1UL <<< from) ^^^ (1UL <<< dst)
        if sfActive then
            this.SfUpdatePieceThreats(pc, false, from, true, fromTo)

        let pt = pieceType pc
        let c = pieceColor pc
        byTypeBB.[pt] <- byTypeBB.[pt] ^^^ fromTo
        byTypeBB.[AllPieces] <- byTypeBB.[AllPieces] ^^^ fromTo
        byColorBB.[c] <- byColorBB.[c] ^^^ fromTo
        board.[from] <- NoPiece
        board.[dst] <- pc
        currentKey <- currentKey ^^^ zPiece pc from ^^^ zPiece pc dst
        if sfActive then
            sfDirtyPc.[sfDirtyN] <- pc; sfDirtySq.[sfDirtyN] <- from; sfDirtySign.[sfDirtyN] <- -1; sfDirtyN <- sfDirtyN + 1
            sfDirtyPc.[sfDirtyN] <- pc; sfDirtySq.[sfDirtyN] <- dst;  sfDirtySign.[sfDirtyN] <- 1;  sfDirtyN <- sfDirtyN + 1
            this.SfUpdatePieceThreats(pc, true, dst, true, fromTo)

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member private this.SwapPiece (oldPc: Piece) (newPc: Piece) (sq: Square) =
        Debug.Assert(oldPc <> NoPiece && newPc <> NoPiece, "SwapPiece: NoPiece")
        Debug.Assert(board.[sq] = oldPc, "SwapPiece: old piece mismatch")
        let b = 1UL <<< sq
        byTypeBB.[pieceType oldPc] <- byTypeBB.[pieceType oldPc] ^^^ b
        byTypeBB.[AllPieces] <- byTypeBB.[AllPieces] ^^^ b
        byColorBB.[pieceColor oldPc] <- byColorBB.[pieceColor oldPc] ^^^ b
        board.[sq] <- NoPiece
        currentKey <- currentKey ^^^ zPiece oldPc sq
        if sfActive then
            sfDirtyPc.[sfDirtyN] <- oldPc; sfDirtySq.[sfDirtyN] <- sq; sfDirtySign.[sfDirtyN] <- -1; sfDirtyN <- sfDirtyN + 1
            this.SfUpdatePieceThreats(oldPc, false, sq, false, 0UL)

        byTypeBB.[pieceType newPc] <- byTypeBB.[pieceType newPc] ||| b
        byTypeBB.[AllPieces] <- byTypeBB.[AllPieces] ||| b
        byColorBB.[pieceColor newPc] <- byColorBB.[pieceColor newPc] ||| b
        board.[sq] <- newPc
        currentKey <- currentKey ^^^ zPiece newPc sq
        if sfActive then
            sfDirtyPc.[sfDirtyN] <- newPc; sfDirtySq.[sfDirtyN] <- sq; sfDirtySign.[sfDirtyN] <- 1; sfDirtyN <- sfDirtyN + 1
            this.SfUpdatePieceThreats(newPc, true, sq, false, 0UL)

    // Key-less siblings (used ONLY by unmake -> pure board restore, zero key-drift risk).
    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member private _.PutPieceNK (pc: Piece) (sq: Square) =
        let b = 1UL <<< sq
        byTypeBB.[pieceType pc] <- byTypeBB.[pieceType pc] ||| b
        byTypeBB.[AllPieces] <- byTypeBB.[AllPieces] ||| b
        byColorBB.[pieceColor pc] <- byColorBB.[pieceColor pc] ||| b
        board.[sq] <- pc

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member private _.RemovePieceNK (pc: Piece) (sq: Square) =
        let b = 1UL <<< sq
        byTypeBB.[pieceType pc] <- byTypeBB.[pieceType pc] ^^^ b
        byTypeBB.[AllPieces] <- byTypeBB.[AllPieces] ^^^ b
        byColorBB.[pieceColor pc] <- byColorBB.[pieceColor pc] ^^^ b
        board.[sq] <- NoPiece

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member private _.MovePieceNK (pc: Piece) (from: Square) (dst: Square) =
        let fromTo = (1UL <<< from) ^^^ (1UL <<< dst)
        byTypeBB.[pieceType pc] <- byTypeBB.[pieceType pc] ^^^ fromTo
        byTypeBB.[AllPieces] <- byTypeBB.[AllPieces] ^^^ fromTo
        byColorBB.[pieceColor pc] <- byColorBB.[pieceColor pc] ^^^ fromTo
        board.[from] <- NoPiece
        board.[dst] <- pc

    // --- board / scalar queries --------------------------------------------
    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member _.PieceOn(sq: Square) : Piece = board.[sq]

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member _.IsEmpty(sq: Square) : bool = board.[sq] = NoPiece

    member _.Occupied: Bitboard = byTypeBB.[AllPieces]

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member _.Pieces(pt: PieceType) : Bitboard = byTypeBB.[pt]

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member _.ColorBB(c: Color) : Bitboard = byColorBB.[c]

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member _.PiecesCT (c: Color) (pt: PieceType) : Bitboard = byTypeBB.[pt] &&& byColorBB.[c]

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member _.KingSquare(c: Color) : Square = lsb (byTypeBB.[King] &&& byColorBB.[c]) // PRE: king present

    member _.SideToMove: Color = sideToMove
    member _.GamePly: int = gamePly

    // --- SF FullThreats NNUE accumulator: public read + lazy materialization ---
    member _.SfActive: bool = sfActive

    member _.SfTop: int = sfTop

    member private _.SfEnsureStorage() =
        if Array.isEmpty sfAccW then
            sfAccW <- Array.zeroCreate<int16> (SfMaxPly * Accumulator.L1)
            sfAccB <- Array.zeroCreate<int16> (SfMaxPly * Accumulator.L1)
            sfPsqW <- Array.zeroCreate (SfMaxPly * Accumulator.PsqtBuckets)
            sfPsqB <- Array.zeroCreate (SfMaxPly * Accumulator.PsqtBuckets)
            sfComputedW <- Array.zeroCreate SfMaxPly
            sfComputedB <- Array.zeroCreate SfMaxPly
            sfFrameDirtyPc <- Array.zeroCreate Accumulator.MaxDirtyPieces
            sfFrameDirtySq <- Array.zeroCreate Accumulator.MaxDirtyPieces
            sfFrameDirtySign <- Array.zeroCreate Accumulator.MaxDirtyPieces
            sfFrameDirtyN <- Array.zeroCreate SfMaxPly
            sfFrameThreats <- Array.zeroCreate Accumulator.MaxDirtyThreats
            sfFrameThreatN <- Array.zeroCreate SfMaxPly
            sfFrameChangedW <- Array.zeroCreate Accumulator.MaxDirtyThreats
            sfFrameChangedB <- Array.zeroCreate Accumulator.MaxDirtyThreats
            sfFrameChangedNW <- Array.zeroCreate SfMaxPly
            sfFrameChangedNB <- Array.zeroCreate SfMaxPly
            sfFrameChangedValid <- Array.zeroCreate SfMaxPly
            sfFrameChangedKsqW <- Array.create SfMaxPly NoSquare
            sfFrameChangedKsqB <- Array.create SfMaxPly NoSquare
            sfFrameWhiteKingMoved <- Array.zeroCreate SfMaxPly
            sfFrameBlackKingMoved <- Array.zeroCreate SfMaxPly
            sfFrameThreatOverflow <- Array.zeroCreate SfMaxPly

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member private _.SfAccOff(frame: int) = frame * Accumulator.L1

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member private _.SfPsqOff(frame: int) = frame * Accumulator.PsqtBuckets

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member private _.SfDirtyOff(frame: int) = 0

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member private _.SfThreatOff(frame: int) = 0

    member private this.SfEnumThreats(pColor: Color, buf: int[]) : int =
        match sfThreatFn with
        | null -> 0
        | f -> f.Invoke(this, pColor, buf)

    member private this.SfBuildHalf(pColor: Color, frame: int) =
        let acc = if pColor = White then sfAccW else sfAccB
        let psq = if pColor = White then sfPsqW else sfPsqB
        let accOff = this.SfAccOff frame
        let psqOff = this.SfPsqOff frame
        let ksq = this.KingSquare pColor

        for j in 0 .. Accumulator.L1 - 1 do
            acc.[accOff + j] <- sfBiases.[j]

        System.Array.Clear(psq, psqOff, Accumulator.PsqtBuckets)

        for sq in 0 .. 63 do
            let pc = board.[sq]

            if pc <> NoPiece then
                Accumulator.addFeatureAt acc accOff psq psqOff sfHalfWeights sfHalfPsqt (Accumulator.makeIndex pColor pc sq ksq) 1 Accumulator.UseAvx2

    member private this.SfAddActiveThreats(pColor: Color, frame: int) =
        let acc = if pColor = White then sfAccW else sfAccB
        let psq = if pColor = White then sfPsqW else sfPsqB
        let accOff = this.SfAccOff frame
        let psqOff = this.SfPsqOff frame
        let n = this.SfEnumThreats(pColor, sfTmpW)

        for k in 0 .. n - 1 do
            Accumulator.addThreatAt acc accOff psq psqOff sfThreatWeights sfThreatPsqt sfTmpW.[k] 1 Accumulator.UseAvx2

    member private this.SfBuildFull(pColor: Color, frame: int) =
        this.SfBuildHalf(pColor, frame)
        this.SfAddActiveThreats(pColor, frame)
        if pColor = White then sfComputedW.[frame] <- true else sfComputedB.[frame] <- true

    // Both perspectives' active threats from a SINGLE physical enumeration (slider rays walked once),
    // emitting per-perspective indices into sfTmpW/sfTmpB. Bit-exact vs two SfAddActiveThreats calls:
    // same indices, and int16 accumulation is order-independent (modular add).
    member private this.SfAddActiveThreatsBoth(frame: int) =
        match sfThreatFnBoth with
        | null ->
            this.SfAddActiveThreats(White, frame)
            this.SfAddActiveThreats(Black, frame)
        | f ->
            let packed = f.Invoke(this, sfTmpW, sfTmpB)
            let nW = int (packed >>> 32)
            let nB = int (packed &&& 0xFFFFFFFFL)
            let accOff = this.SfAccOff frame
            let psqOff = this.SfPsqOff frame

            for k in 0 .. nW - 1 do
                Accumulator.addThreatAt sfAccW accOff sfPsqW psqOff sfThreatWeights sfThreatPsqt sfTmpW.[k] 1 Accumulator.UseAvx2

            for k in 0 .. nB - 1 do
                Accumulator.addThreatAt sfAccB accOff sfPsqB psqOff sfThreatWeights sfThreatPsqt sfTmpB.[k] 1 Accumulator.UseAvx2

    member private this.SfBuildFullBoth(frame: int) =
        this.SfBuildHalf(White, frame)
        this.SfBuildHalf(Black, frame)
        this.SfAddActiveThreatsBoth(frame)
        sfComputedW.[frame] <- true
        sfComputedB.[frame] <- true

    member private this.SfBeginFrame() =
        Debug.Assert((sfTop + 1 < SfMaxPly), "SfBeginFrame: stack overflow")
        sfTop <- sfTop + 1
        sfDirtyN <- 0
        sfDirtyThreatN <- 0
        sfDirtyThreatOverflow <- false
        sfFrameDirtyN.[sfTop] <- 0
        sfFrameThreatN.[sfTop] <- 0
        sfFrameChangedNW.[sfTop] <- 0
        sfFrameChangedNB.[sfTop] <- 0
        sfFrameChangedValid.[sfTop] <- false
        sfFrameChangedKsqW.[sfTop] <- NoSquare
        sfFrameChangedKsqB.[sfTop] <- NoSquare
        sfFrameWhiteKingMoved.[sfTop] <- false
        sfFrameBlackKingMoved.[sfTop] <- false
        sfFrameThreatOverflow.[sfTop] <- false
        sfComputedW.[sfTop] <- false
        sfComputedB.[sfTop] <- false

    member private this.SfCommitFrame() =
        let frame = sfTop
        let dOff = this.SfDirtyOff frame
        let mutable whiteKingMoved = false
        let mutable blackKingMoved = false

        Debug.Assert((sfDirtyN <= Accumulator.MaxDirtyPieces), "SfCommitFrame: dirty piece overflow")

        for i in 0 .. sfDirtyN - 1 do
            let pc = sfDirtyPc.[i]
            sfFrameDirtyPc.[dOff + i] <- pc
            sfFrameDirtySq.[dOff + i] <- sfDirtySq.[i]
            sfFrameDirtySign.[dOff + i] <- sfDirtySign.[i]

            if pieceType pc = King then
                if pieceColor pc = White then whiteKingMoved <- true else blackKingMoved <- true

        sfFrameDirtyN.[frame] <- sfDirtyN
        sfFrameWhiteKingMoved.[frame] <- whiteKingMoved
        sfFrameBlackKingMoved.[frame] <- blackKingMoved

        if sfDirtyThreatOverflow then
            sfFrameThreatOverflow.[frame] <- true
            sfFrameThreatN.[frame] <- 0
            sfFrameChangedNW.[frame] <- 0
            sfFrameChangedNB.[frame] <- 0
            sfFrameChangedValid.[frame] <- false
        else
            let threatN = min sfDirtyThreatN Accumulator.MaxDirtyThreats
            sfFrameThreatN.[frame] <- threatN
            System.Array.Copy(sfDirtyThreats, 0, sfFrameThreats, this.SfThreatOff frame, threatN)

            if threatN <> 0 then
                sfFrameChangedValid.[frame] <- false
            else
                sfFrameChangedNW.[frame] <- 0
                sfFrameChangedNB.[frame] <- 0
                sfFrameChangedValid.[frame] <- true

    member private _.SfFrameNeedsRefresh(pColor: Color, frame: int) : bool =
        sfFrameThreatOverflow.[frame]
        || (pColor = White && sfFrameWhiteKingMoved.[frame])
        || (pColor = Black && sfFrameBlackKingMoved.[frame])

    /// Eagerly materialize the current frame's accumulator during Make: copy the parent (sfTop-1) into sfTop,
    /// then apply this frame's dirty piece + threat deltas immediately. After this, sfComputedW/B.[sfTop] are
    /// true and eval is O(1) — no lazy frame walk, no cache probe, no deferred delegate conversion at eval time.
    /// King moves or threat overflow force a full from-scratch rebuild (all HalfKA indices change).
    member private this.SfEagerUpdate() =
        let frame = sfTop

        if sfFrameThreatOverflow.[frame] || sfFrameWhiteKingMoved.[frame] || sfFrameBlackKingMoved.[frame] then
            this.SfBuildFullBoth(frame)
        else
            this.SfCopyFrame(White, frame - 1, frame)
            this.SfCopyFrame(Black, frame - 1, frame)
            this.SfApplyFrameBoth(frame)
            sfComputedW.[frame] <- true
            sfComputedB.[frame] <- true

    member private this.SfCopyFrame(pColor: Color, src: int, dst: int) =
        let acc = if pColor = White then sfAccW else sfAccB
        let psq = if pColor = White then sfPsqW else sfPsqB
        System.Array.Copy(acc, this.SfAccOff src, acc, this.SfAccOff dst, Accumulator.L1)
        System.Array.Copy(psq, this.SfPsqOff src, psq, this.SfPsqOff dst, Accumulator.PsqtBuckets)

    member private this.SfApplySignedThreats(pColor: Color, frame: int, buf: int[], off: int, n: int) =
        let acc = if pColor = White then sfAccW else sfAccB
        let psq = if pColor = White then sfPsqW else sfPsqB
        let accOff = this.SfAccOff frame
        let psqOff = this.SfPsqOff frame

        if n <= 62 then
            let addIdxs = sfTmpW
            let subIdxs = sfTmpB
            let mutable nAdd = 0
            let mutable nSub = 0

            for i in 0 .. n - 1 do
                let v = buf.[off + i]

                if v > 0 then
                    addIdxs.[nAdd] <- v - 1
                    nAdd <- nAdd + 1
                else
                    subIdxs.[nSub] <- -v - 1
                    nSub <- nSub + 1

            let pairN = min nAdd nSub
            let mutable p = 0

            while p + 1 < pairN do
                Accumulator.addThreatPair2At
                    acc
                    accOff
                    psq
                    psqOff
                    sfThreatWeights
                    sfThreatPsqt
                    subIdxs.[p]
                    addIdxs.[p]
                    subIdxs.[p + 1]
                    addIdxs.[p + 1]
                    Accumulator.UseAvx2

                p <- p + 2

            if p < pairN then
                Accumulator.addThreatPairAt acc accOff psq psqOff sfThreatWeights sfThreatPsqt subIdxs.[p] addIdxs.[p] Accumulator.UseAvx2

            for i in pairN .. nAdd - 1 do
                Accumulator.addThreatAt acc accOff psq psqOff sfThreatWeights sfThreatPsqt addIdxs.[i] 1 Accumulator.UseAvx2

            for i in pairN .. nSub - 1 do
                Accumulator.addThreatAt acc accOff psq psqOff sfThreatWeights sfThreatPsqt subIdxs.[i] -1 Accumulator.UseAvx2
        else
            for i in 0 .. n - 1 do
                let v = buf.[off + i]
                let sign, idx = if v > 0 then 1, v - 1 else -1, -v - 1
                Accumulator.addThreatAt acc accOff psq psqOff sfThreatWeights sfThreatPsqt idx sign Accumulator.UseAvx2

    member private this.SfApplyFrame(pColor: Color, frame: int) =
        let acc = if pColor = White then sfAccW else sfAccB
        let psq = if pColor = White then sfPsqW else sfPsqB
        let accOff = this.SfAccOff sfTop
        let psqOff = this.SfPsqOff sfTop
        let dOff = this.SfDirtyOff frame
        let ksq = this.KingSquare pColor
        let ksqW = this.KingSquare White
        let ksqB = this.KingSquare Black

        let dirtyN = sfFrameDirtyN.[frame]
        let mutable usedDirty = 0

        for i in 0 .. dirtyN - 1 do
            if (usedDirty &&& (1 <<< i)) = 0 then
                let pc = sfFrameDirtyPc.[dOff + i]
                let sq = sfFrameDirtySq.[dOff + i]
                let sign = sfFrameDirtySign.[dOff + i]

                if sign < 0 then
                    let mutable j = i + 1
                    let mutable pair = -1

                    while pair < 0 && j < dirtyN do
                        if
                            (usedDirty &&& (1 <<< j)) = 0
                            && sfFrameDirtyPc.[dOff + j] = pc
                            && sfFrameDirtySign.[dOff + j] > 0
                        then
                            pair <- j
                        else
                            j <- j + 1

                    if pair >= 0 then
                        usedDirty <- usedDirty ||| (1 <<< i) ||| (1 <<< pair)
                        let addSq = sfFrameDirtySq.[dOff + pair]
                        Accumulator.addFeaturePairAt
                            acc
                            accOff
                            psq
                            psqOff
                            sfHalfWeights
                            sfHalfPsqt
                            (Accumulator.makeIndex pColor pc sq ksq)
                            (Accumulator.makeIndex pColor pc addSq ksq)
                            Accumulator.UseAvx2
                    else
                        usedDirty <- usedDirty ||| (1 <<< i)
                        Accumulator.addFeatureAt acc accOff psq psqOff sfHalfWeights sfHalfPsqt (Accumulator.makeIndex pColor pc sq ksq) sign Accumulator.UseAvx2
                else
                    usedDirty <- usedDirty ||| (1 <<< i)
                    Accumulator.addFeatureAt acc accOff psq psqOff sfHalfWeights sfHalfPsqt (Accumulator.makeIndex pColor pc sq ksq) sign Accumulator.UseAvx2

        let threatN = sfFrameThreatN.[frame]

        if threatN <> 0 then
            if sfFrameChangedValid.[frame] && sfFrameChangedKsqW.[frame] = ksqW && sfFrameChangedKsqB.[frame] = ksqB then
                let off = this.SfThreatOff frame
                if pColor = White then
                    this.SfApplySignedThreats(White, sfTop, sfFrameChangedW, off, sfFrameChangedNW.[frame])
                else
                    this.SfApplySignedThreats(Black, sfTop, sfFrameChangedB, off, sfFrameChangedNB.[frame])
            else
                match sfThreatFnChangedBoth with
                | null -> ()
                | f ->
                    let packed = f.Invoke(this, sfFrameThreats, this.SfThreatOff frame, threatN, sfChangedW, sfChangedB)
                    let nW = int (packed >>> 32)
                    let nB = int (packed &&& 0xFFFFFFFFL)
                    let off = this.SfThreatOff frame

                    System.Array.Copy(sfChangedW, 0, sfFrameChangedW, off, nW)
                    System.Array.Copy(sfChangedB, 0, sfFrameChangedB, off, nB)
                    sfFrameChangedNW.[frame] <- nW
                    sfFrameChangedNB.[frame] <- nB
                    sfFrameChangedKsqW.[frame] <- ksqW
                    sfFrameChangedKsqB.[frame] <- ksqB
                    sfFrameChangedValid.[frame] <- true

                    if pColor = White then
                        this.SfApplySignedThreats(White, sfTop, sfFrameChangedW, off, nW)
                    else
                        this.SfApplySignedThreats(Black, sfTop, sfFrameChangedB, off, nB)

    member private this.SfEnsureComputed(pColor: Color) =
        if sfActive then
            let computed = if pColor = White then sfComputedW else sfComputedB

            if not computed.[sfTop] then
                let mutable baseFrame = sfTop
                let mutable blocked = false

                while baseFrame > 0 && not blocked && not computed.[baseFrame] do
                    if this.SfFrameNeedsRefresh(pColor, baseFrame) then
                        blocked <- true
                    else
                        baseFrame <- baseFrame - 1

                if blocked || not computed.[baseFrame] then
                    this.SfBuildFull(pColor, sfTop)
                else
                    this.SfCopyFrame(pColor, baseFrame, sfTop)

                    for f in baseFrame + 1 .. sfTop do
                        this.SfApplyFrame(pColor, f)

                    computed.[sfTop] <- true

    // Replay one frame's deltas onto the sfTop accumulator for BOTH perspectives at once: one pass over the
    // dirty-piece list (each acc only takes its own perspective's features), one shared changed-threat
    // conversion. Bit-exact vs SfApplyFrame(White)+SfApplyFrame(Black).
    member private this.SfApplyFrameBoth(frame: int) =
        let accOff = this.SfAccOff sfTop
        let psqOff = this.SfPsqOff sfTop
        let dOff = this.SfDirtyOff frame
        let ksqW = this.KingSquare White
        let ksqB = this.KingSquare Black

        let dirtyN = sfFrameDirtyN.[frame]
        let mutable usedDirty = 0

        for i in 0 .. dirtyN - 1 do
            if (usedDirty &&& (1 <<< i)) = 0 then
                let pc = sfFrameDirtyPc.[dOff + i]
                let sq = sfFrameDirtySq.[dOff + i]
                let sign = sfFrameDirtySign.[dOff + i]

                if sign < 0 then
                    let mutable j = i + 1
                    let mutable pair = -1

                    while pair < 0 && j < dirtyN do
                        if
                            (usedDirty &&& (1 <<< j)) = 0
                            && sfFrameDirtyPc.[dOff + j] = pc
                            && sfFrameDirtySign.[dOff + j] > 0
                        then
                            pair <- j
                        else
                            j <- j + 1

                    if pair >= 0 then
                        usedDirty <- usedDirty ||| (1 <<< i) ||| (1 <<< pair)
                        let addSq = sfFrameDirtySq.[dOff + pair]
                        Accumulator.addFeaturePairAt
                            sfAccW
                            accOff
                            sfPsqW
                            psqOff
                            sfHalfWeights
                            sfHalfPsqt
                            (Accumulator.makeIndex White pc sq ksqW)
                            (Accumulator.makeIndex White pc addSq ksqW)
                            Accumulator.UseAvx2
                        Accumulator.addFeaturePairAt
                            sfAccB
                            accOff
                            sfPsqB
                            psqOff
                            sfHalfWeights
                            sfHalfPsqt
                            (Accumulator.makeIndex Black pc sq ksqB)
                            (Accumulator.makeIndex Black pc addSq ksqB)
                            Accumulator.UseAvx2
                    else
                        usedDirty <- usedDirty ||| (1 <<< i)
                        Accumulator.addFeatureAt sfAccW accOff sfPsqW psqOff sfHalfWeights sfHalfPsqt (Accumulator.makeIndex White pc sq ksqW) sign Accumulator.UseAvx2
                        Accumulator.addFeatureAt sfAccB accOff sfPsqB psqOff sfHalfWeights sfHalfPsqt (Accumulator.makeIndex Black pc sq ksqB) sign Accumulator.UseAvx2
                else
                    usedDirty <- usedDirty ||| (1 <<< i)
                    Accumulator.addFeatureAt sfAccW accOff sfPsqW psqOff sfHalfWeights sfHalfPsqt (Accumulator.makeIndex White pc sq ksqW) sign Accumulator.UseAvx2
                    Accumulator.addFeatureAt sfAccB accOff sfPsqB psqOff sfHalfWeights sfHalfPsqt (Accumulator.makeIndex Black pc sq ksqB) sign Accumulator.UseAvx2

        let threatN = sfFrameThreatN.[frame]

        if threatN <> 0 then
            let off = this.SfThreatOff frame

            if sfFrameChangedValid.[frame] && sfFrameChangedKsqW.[frame] = ksqW && sfFrameChangedKsqB.[frame] = ksqB then
                this.SfApplySignedThreats(White, sfTop, sfFrameChangedW, off, sfFrameChangedNW.[frame])
                this.SfApplySignedThreats(Black, sfTop, sfFrameChangedB, off, sfFrameChangedNB.[frame])
            else
                match sfThreatFnChangedBoth with
                | null -> ()
                | f ->
                    let packed = f.Invoke(this, sfFrameThreats, off, threatN, sfChangedW, sfChangedB)
                    let nW = int (packed >>> 32)
                    let nB = int (packed &&& 0xFFFFFFFFL)

                    System.Array.Copy(sfChangedW, 0, sfFrameChangedW, off, nW)
                    System.Array.Copy(sfChangedB, 0, sfFrameChangedB, off, nB)
                    sfFrameChangedNW.[frame] <- nW
                    sfFrameChangedNB.[frame] <- nB
                    sfFrameChangedKsqW.[frame] <- ksqW
                    sfFrameChangedKsqB.[frame] <- ksqB
                    sfFrameChangedValid.[frame] <- true

                    this.SfApplySignedThreats(White, sfTop, sfFrameChangedW, off, nW)
                    this.SfApplySignedThreats(Black, sfTop, sfFrameChangedB, off, nB)

    // Materialize BOTH perspectives at sfTop in one frame walk (evalInternal always needs both). Takes the
    // merged path only when the two perspectives' back-walks agree (the common no-king-move/no-overflow
    // case); falls back to the per-perspective SfEnsureComputed otherwise (byte-identical to that path).
    member this.SfEnsureBothComputed() =
        if sfActive && not (sfComputedW.[sfTop] && sfComputedB.[sfTop]) then
            // Phase 1 fast-path: best-effort checkpoint cache. A validated hit pays an O(1) snapshot copy
            // instead of the O(distance) frame-delta walk below. Stored snapshots are bit-exact for any given
            // position regardless of the make/unmake path that reached it, so a hit is provably equivalent
            // to re-running the lazy walk.
            let accOff = this.SfAccOff sfTop
            let psqOff = this.SfPsqOff sfTop

            let cached =
                match sfCheckpoint with
                | null -> false
                | cache ->
                    cache.TryProbe(this.Key, sfAccW, accOff, sfAccB, accOff, sfPsqW, psqOff, sfPsqB, psqOff)

            if cached then
                sfComputedW.[sfTop] <- true
                sfComputedB.[sfTop] <- true
            else
                this.SfEnsureBothComputedCore()

                // Best-effort populate. Checking both flags post-materialization guarantees we never cache a
                // partial snapshot, even on the mixed-rebuild branch + the per-perspective fallback path.
                if sfComputedW.[sfTop] && sfComputedB.[sfTop] then
                    match sfCheckpoint with
                    | null -> ()
                    | cache ->
                        cache.Store(this.Key, sfAccW, accOff, sfAccB, accOff, sfPsqW, psqOff, sfPsqB, psqOff)

    /// Bind the per-worker checkpoint cache. Pass `null` to disable (tests, from-scratch eval, etc.).
    /// `SearchControl` owns the table lifecycle; this Position merely holds a borrowed reference for the
    /// duration of a search.
    member _.SfBindCheckpoint(cache: AccCheckpointTable) : unit = sfCheckpoint <- cache

    /// Detach the cache (no-op if already detached). Called by `SearchControl` once the search has joined to
    /// release the worker's borrowed reference; the position can continue to be reused by tests/tools.
    member _.SfUnbindCheckpoint() : unit = sfCheckpoint <- null

    /// Unconditionally publish the current frame's computed accumulator snapshot to the bound checkpoint
    /// cache, if any. Used by `Worker.SetupRoot` to seed the root after `EnableNnue` has already set the
    /// `sfComputed` flags (so the early-return path inside `SfEnsureBothComputed` skips the populate).
    /// No-op when the accumulator is inactive, the current frame is not yet materialized, or no cache is bound.
    member this.SfSeedCheckpoint() : unit =
        if sfActive && sfComputedW.[sfTop] && sfComputedB.[sfTop] then
            match sfCheckpoint with
            | null -> ()
            | cache ->
                let accOff = this.SfAccOff sfTop
                let psqOff = this.SfPsqOff sfTop
                cache.Store(this.Key, sfAccW, accOff, sfAccB, accOff, sfPsqW, psqOff, sfPsqB, psqOff)

    /// Phase 1 — the unchanged frame-walk materialization used when the checkpoint cache misses (or is
    /// null). Byte-for-byte identical to the pre-Phase-1 `SfEnsureBothComputed`; retained verbatim so that
    /// benchmarks + parity tests can isolate Phase 1's perf contribution empirically (toggle the UCI option
    /// `EnableAccCheckpoint` off — Phase 1 Step 5 — to route all calls through this core).
    member private this.SfEnsureBothComputedCore() =
        if sfActive && not (sfComputedW.[sfTop] && sfComputedB.[sfTop]) then
            let mutable baseW = sfTop
            let mutable blockedW = false

            while baseW > 0 && not blockedW && not sfComputedW.[baseW] do
                if this.SfFrameNeedsRefresh(White, baseW) then blockedW <- true
                else baseW <- baseW - 1

            let mutable baseB = sfTop
            let mutable blockedB = false

            while baseB > 0 && not blockedB && not sfComputedB.[baseB] do
                if this.SfFrameNeedsRefresh(Black, baseB) then blockedB <- true
                else baseB <- baseB - 1

            let rebuildW = blockedW || not sfComputedW.[baseW]
            let rebuildB = blockedB || not sfComputedB.[baseB]

            if not rebuildW && not rebuildB && baseW = baseB then
                this.SfCopyFrame(White, baseW, sfTop)
                this.SfCopyFrame(Black, baseB, sfTop)

                for f in baseW + 1 .. sfTop do
                    this.SfApplyFrameBoth(f)

                sfComputedW.[sfTop] <- true
                sfComputedB.[sfTop] <- true
            elif rebuildW && rebuildB then
                this.SfBuildFullBoth(sfTop)
            else
                this.SfEnsureComputed White
                this.SfEnsureComputed Black

    /// Merged accumulator (biases + HalfKA + threats already summed) for a perspective, into caller spans.
    member this.SfReadAccInto(pColor: Color, acc: System.Span<int16>, psqt: System.Span<int>) =
        this.SfEnsureComputed pColor
        let m = if pColor = White then sfAccW else sfAccB
        let mp = if pColor = White then sfPsqW else sfPsqB
        System.Span<int16>(m, this.SfAccOff sfTop, Accumulator.L1).CopyTo(acc)
        System.Span<int>(mp, this.SfPsqOff sfTop, Accumulator.PsqtBuckets).CopyTo(psqt)

    member this.SfAccSpan(pColor: Color) : System.Span<int16> =
        this.SfEnsureComputed pColor
        let m = if pColor = White then sfAccW else sfAccB
        System.Span<int16>(m, this.SfAccOff sfTop, Accumulator.L1)

    member this.SfPsqtSpan(pColor: Color) : System.Span<int> =
        this.SfEnsureComputed pColor
        let mp = if pColor = White then sfPsqW else sfPsqB
        System.Span<int>(mp, this.SfPsqOff sfTop, Accumulator.PsqtBuckets)

    member this.SfAccSpanComputed(pColor: Color) : System.Span<int16> =
        let m = if pColor = White then sfAccW else sfAccB
        System.Span<int16>(m, this.SfAccOff sfTop, Accumulator.L1)

    member this.SfPsqtSpanComputed(pColor: Color) : System.Span<int> =
        let mp = if pColor = White then sfPsqW else sfPsqB
        System.Span<int>(mp, this.SfPsqOff sfTop, Accumulator.PsqtBuckets)

    /// Compatibility escape hatch for tests/probes that still want an array. Prefer SfAccSpan in hot code.
    member this.SfAccArray(pColor: Color) : int16[] =
        this.SfEnsureComputed pColor
        let src = if pColor = White then sfAccW else sfAccB
        src.[this.SfAccOff sfTop .. this.SfAccOff sfTop + Accumulator.L1 - 1]

    member this.SfPsqtArray(pColor: Color) : int[] =
        this.SfEnsureComputed pColor
        let src = if pColor = White then sfPsqW else sfPsqB
        src.[this.SfPsqOff sfTop .. this.SfPsqOff sfTop + Accumulator.PsqtBuckets - 1]

    /// Bind weights + threat enumerators + materialize root. ROOT ONLY.
    member this.EnableNnue
        (biases: int16[])
        (halfWeights: int16[])
        (halfPsqt: int[])
        (threatWeights: sbyte[])
        (threatPsqt: int[])
        (threatFn: System.Func<Position, int, int[], int>)
        (threatFnBoth: System.Func<Position, int[], int[], int64>)
        (threatFnChangedBoth: System.Func<Position, int[], int, int, int[], int[], int64>)
        =
        Debug.Assert((stPly = 0), "EnableNnue must be called at the root (stPly = 0)")
        this.SfEnsureStorage()
        sfBiases <- biases
        sfHalfWeights <- halfWeights
        sfHalfPsqt <- halfPsqt
        sfThreatWeights <- threatWeights
        sfThreatPsqt <- threatPsqt
        sfThreatFn <- threatFn
        sfThreatFnBoth <- threatFnBoth
        sfThreatFnChangedBoth <- threatFnChangedBoth
        sfTop <- 0
        sfDirtyN <- 0
        sfDirtyThreatN <- 0
        sfDirtyThreatOverflow <- false
        System.Array.Clear(sfComputedW, 0, sfComputedW.Length)
        System.Array.Clear(sfComputedB, 0, sfComputedB.Length)
        sfFrameDirtyN.[0] <- 0
        sfFrameThreatN.[0] <- 0
        sfFrameChangedNW.[0] <- 0
        sfFrameChangedNB.[0] <- 0
        sfFrameChangedValid.[0] <- false
        sfFrameChangedKsqW.[0] <- NoSquare
        sfFrameChangedKsqB.[0] <- NoSquare
        sfFrameWhiteKingMoved.[0] <- false
        sfFrameBlackKingMoved.[0] <- false
        sfFrameThreatOverflow.[0] <- false
        sfActive <- true
        sfEagerUpdates <- true
        this.SfBuildFull(White, 0)
        this.SfBuildFull(Black, 0)

    /// Test/debug hook: collect the physical dirty FullThreats edges that `Make m` would record, without
    /// requiring NNUE weights. Returns -1 if the frame hit the overflow fallback.
    member this.SfDebugCollectDirtyThreats(m: Move, dst: int[]) : int =
        let wasActive = sfActive
        let savedTop = sfTop
        this.SfEnsureStorage()

        if not wasActive then
            sfActive <- true
            sfTop <- 0
            sfDirtyN <- 0

        this.Make m
        let frame = sfTop
        let n = sfFrameThreatN.[frame]
        let overflow = sfDirtyThreatOverflow

        if not overflow then
            System.Array.Copy(sfFrameThreats, this.SfThreatOff frame, dst, 0, min n dst.Length)

        this.Unmake m

        if not wasActive then
            sfActive <- false
            sfTop <- savedTop
            sfDirtyN <- 0

        if overflow then -1 else min n dst.Length

    // --- StateInfo scalar accessors (byref-local read avoids the ~120 B struct copy; the getter itself
    //     is trivial and JIT-inlined — F# forbids MethodImpl on a parameterless property) -------------
    member _.EpSquare: Square = let st = &states.[stPly] in st.EpSquare
    member _.CastlingRights: int = let st = &states.[stPly] in st.CastlingRights
    member _.Rule50: int = let st = &states.[stPly] in st.Rule50
    // Plies since the last null move (search-only; bounds the null-safe repetition walk). Mirrors Rule50.
    member _.PliesFromNull: int = let st = &states.[stPly] in st.PliesFromNull
    member _.Key: uint64 = let st = &states.[stPly] in st.Key

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member _.CanCastle(right: int) : bool =
        let st = &states.[stPly] in (st.CastlingRights &&& right) <> 0

    // --- attackersTo: the single occupancy-parameterized attack primitive ---
    // Pawn colors are deliberately CROSSED: pawnAttacks White sq = where a white pawn ON sq attacks =
    // exactly where BLACK pawns attack sq from. occ affects only the sliders (leapers ignore it).
    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member _.AttackersTo (sq: Square) (occ: Bitboard) : Bitboard =
        let bishops = byTypeBB.[Bishop] ||| byTypeBB.[Queen]
        let rooks = byTypeBB.[Rook] ||| byTypeBB.[Queen]

        (pawnAttacks White sq &&& (byTypeBB.[Pawn] &&& byColorBB.[Black]))
        ||| (pawnAttacks Black sq &&& (byTypeBB.[Pawn] &&& byColorBB.[White]))
        ||| (knightAttacks sq &&& byTypeBB.[Knight])
        ||| (kingAttacks sq &&& byTypeBB.[King])
        ||| (bishopAttacks sq occ &&& bishops)
        ||| (rookAttacks sq occ &&& rooks)

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member this.AttackersToOcc(sq: Square) : Bitboard =
        this.AttackersTo sq byTypeBB.[AllPieces]

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member this.AttackedBy (c: Color) (sq: Square) : bool =
        (this.AttackersToOcc sq &&& byColorBB.[c]) <> 0UL

    /// Material value of a PieceType (same table SEE/ordering use); King = 0.
    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member _.PieceValueOf(pt: PieceType) : int = pieceValue.[pt]

    // --- SEE: static-exchange evaluation >= threshold (Stockfish Position::see_ge, faithful) ----------
    // v1 simplifications (see plan D5): non-NORMAL m short-circuits to (0 >= threshold) — exactly SF's
    // own early-out (promotion treated via that path); and the swap loop is NOT pin-aware (the
    // KING-terminate rule already covers the dominant illegal-recapture case; Pinners/BlockersForKing are
    // available for a v2 refinement). SEE drives only pruning/ordering, never legality. The control flow
    // (swap accumulator, res toggle, `swap < res` break, KING terminate, return bool(res)) mirrors SF
    // line-for-line so the unit fixtures pin to exact values.
    member this.SeeGe (m: Move) (threshold: int) : bool =
        if isSpecial m then
            0 >= threshold
        else
            let from = fromSq m
            let dst = toSq m
            let captured = board.[dst]
            // value[captured] - threshold ; captured may be NoPiece (quiet move) -> 0
            let mutable swap =
                (if captured = NoPiece then
                     0
                 else
                     pieceValue.[pieceType captured])
                - threshold

            if swap < 0 then
                false
            else
                // value[mover] - swap ; <=0 means the opponent cannot profitably recapture
                swap <- pieceValue.[pieceType board.[from]] - swap

                if swap <= 0 then
                    true
                else
                    let mutable occupied = byTypeBB.[AllPieces] ^^^ (bit from) ^^^ (bit dst)
                    let mutable stm = pieceColor board.[from]
                    let mutable attackers = this.AttackersTo dst occupied
                    let bishopsQueens = byTypeBB.[Bishop] ||| byTypeBB.[Queen]
                    let rooksQueens = byTypeBB.[Rook] ||| byTypeBB.[Queen]
                    let mutable res = 1
                    let mutable looping = true
                    let mutable result = false

                    while looping do
                        stm <- flipColor stm
                        attackers <- attackers &&& occupied // drop attackers whose square left occ
                        let stmAttackers = attackers &&& byColorBB.[stm]

                        if stmAttackers = 0UL then
                            looping <- false
                            result <- (res <> 0) // stm cannot recapture
                        else
                            res <- res ^^^ 1
                            // least-valuable attacker, in P,N,B,R,Q,K order; x-ray-refresh sliders through dst
                            let bbP = stmAttackers &&& byTypeBB.[Pawn]

                            if bbP <> 0UL then
                                swap <- pieceValue.[Pawn] - swap

                                if swap < res then
                                    (looping <- false
                                     result <- (res <> 0))
                                else
                                    occupied <- occupied ^^^ (bit (lsb bbP))
                                    attackers <- attackers ||| (bishopAttacks dst occupied &&& bishopsQueens)
                            else
                                let bbN = stmAttackers &&& byTypeBB.[Knight]

                                if bbN <> 0UL then
                                    swap <- pieceValue.[Knight] - swap

                                    if swap < res then
                                        (looping <- false
                                         result <- (res <> 0))
                                    else
                                        occupied <- occupied ^^^ (bit (lsb bbN))
                                else
                                    let bbB = stmAttackers &&& byTypeBB.[Bishop]

                                    if bbB <> 0UL then
                                        swap <- pieceValue.[Bishop] - swap

                                        if swap < res then
                                            (looping <- false
                                             result <- (res <> 0))
                                        else
                                            occupied <- occupied ^^^ (bit (lsb bbB))
                                            attackers <- attackers ||| (bishopAttacks dst occupied &&& bishopsQueens)
                                    else
                                        let bbR = stmAttackers &&& byTypeBB.[Rook]

                                        if bbR <> 0UL then
                                            swap <- pieceValue.[Rook] - swap

                                            if swap < res then
                                                (looping <- false
                                                 result <- (res <> 0))
                                            else
                                                occupied <- occupied ^^^ (bit (lsb bbR))
                                                attackers <- attackers ||| (rookAttacks dst occupied &&& rooksQueens)
                                        else
                                            let bbQ = stmAttackers &&& byTypeBB.[Queen]

                                            if bbQ <> 0UL then
                                                swap <- pieceValue.[Queen] - swap

                                                if swap < res then
                                                    (looping <- false
                                                     result <- (res <> 0))
                                                else
                                                    occupied <- occupied ^^^ (bit (lsb bbQ))

                                                    attackers <-
                                                        attackers
                                                        ||| (bishopAttacks dst occupied &&& bishopsQueens)
                                                        ||| (rookAttacks dst occupied &&& rooksQueens)
                                            else
                                                // KING: if the OTHER side still has an attacker, the king
                                                // cannot capture (would move into check) -> stm loses.
                                                looping <- false

                                                result <-
                                                    if (attackers &&& byColorBB.[flipColor stm]) <> 0UL then
                                                        (res ^^^ 1) <> 0
                                                    else
                                                        res <> 0

                    result

    // --- slider blockers / pinners (SF slider_blockers, verbatim) -----------
    // snipers are found on the EMPTY board so a far slider behind a blocker is still seen; occ ^^^ snipers
    // removes ALL snipers so one sniper isn't mis-counted as a "blocker" behind another. Tupled (byref out).
    member private _.SliderBlockers(kc: Color, ksq: Square, pinners: byref<Bitboard>) : Bitboard =
        let them = flipColor kc
        let occ = byTypeBB.[AllPieces]

        let mutable snipers =
            ((rookAttacks ksq 0UL &&& (byTypeBB.[Rook] ||| byTypeBB.[Queen]))
             ||| (bishopAttacks ksq 0UL &&& (byTypeBB.[Bishop] ||| byTypeBB.[Queen])))
            &&& byColorBB.[them]

        let occMinusSnipers = occ ^^^ snipers
        let mutable blockers = 0UL
        pinners <- 0UL

        while snipers <> 0UL do
            let s = popLsb &snipers
            let b = between ksq s &&& occMinusSnipers

            if b <> 0UL && not (moreThanOne b) then
                blockers <- blockers ||| b

                if (b &&& byColorBB.[kc]) <> 0UL then
                    pinners <- pinners ||| (1UL <<< s)

        blockers

    // --- set_check_info: cache checkers / blockers / pinners / checkSquares --
    // HARD RULE: never pass &states.[stPly].Field as a byref out-param — compute into a local, assign back.
    member private this.SetCheckInfo() =
        let us = sideToMove // side to move AFTER the move
        let them = flipColor us

        Debug.Assert(
            ((byTypeBB.[King] &&& byColorBB.[us]) <> 0UL
             && (byTypeBB.[King] &&& byColorBB.[them]) <> 0UL),
            "SetCheckInfo: both kings must be present"
        )

        let ourKing = this.KingSquare us
        let theirKing = this.KingSquare them
        let occ = byTypeBB.[AllPieces]
        let mutable pw = 0UL
        let mutable pb = 0UL
        let bw = this.SliderBlockers(White, this.KingSquare White, &pw) // local out-params ONLY
        let bb = this.SliderBlockers(Black, this.KingSquare Black, &pb)
        let checkers = this.AttackersToOcc ourKing &&& byColorBB.[them]
        let sqP = pawnAttacks them theirKing // where a `us` pawn checks `them` king (inversion)
        let sqN = knightAttacks theirKing
        let sqB = bishopAttacks theirKing occ
        let sqR = rookAttacks theirKing occ
        let st = &states.[stPly]
        st.Checkers <- checkers
        st.BlockersW <- bw
        st.BlockersB <- bb
        st.PinnersW <- pw
        st.PinnersB <- pb
        st.CheckSqP <- sqP
        st.CheckSqN <- sqN
        st.CheckSqB <- sqB
        st.CheckSqR <- sqR
        st.CheckSqQ <- sqB ||| sqR

    // --- check-info accessors (byref-local read) ----------------------------
    member _.Checkers: Bitboard = let st = &states.[stPly] in st.Checkers
    member _.InCheck: bool = let st = &states.[stPly] in st.Checkers <> 0UL

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member _.BlockersForKing(c: Color) : Bitboard =
        let st = &states.[stPly] in (if c = White then st.BlockersW else st.BlockersB)

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member _.Pinners(c: Color) : Bitboard =
        let st = &states.[stPly] in (if c = White then st.PinnersW else st.PinnersB)

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member _.CheckSquares(pt: PieceType) : Bitboard =
        let st = &states.[stPly]

        match pt with
        | Pawn -> st.CheckSqP
        | Knight -> st.CheckSqN
        | Bishop -> st.CheckSqB
        | Rook -> st.CheckSqR
        | Queen -> st.CheckSqQ
        | _ -> 0UL // King never gives check (SF: checkSquares[KING] = 0)

    // --- castling geometry for MoveGen (by single right bit WK/WQ/BK/BQ) ----
    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member _.CastleKingPath(right: int) : Bitboard = castleKingPath.[right]

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member _.CastleEmptyPath(right: int) : Bitboard = castleEmptyPath.[right]

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member _.CastleRookSquare(right: int) : Square = castleRookOrigin.[right]

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member _.CastleKingDest(right: int) : Square = castleKingDest.[right]

    // --- repetition / key history (search; perft ignores) -------------------
    member _.KeyAt(pliesBack: int) : uint64 =
        Debug.Assert((pliesBack >= 0 && pliesBack <= stPly), "KeyAt: out of range")
        let st = &states.[stPly - pliesBack] in
        st.Key

    member _.IsRepetition: bool = let st = &states.[stPly] in st.Repetition <> 0
    member _.SetRepetition() : unit = () // search-only stub; filled when search lands

    // --- from-scratch key oracle (the convention all 3 producers must match) -
    member _.RecomputeKey() : uint64 =
        let mutable key = 0UL
        let mutable occ = byTypeBB.[AllPieces]

        while occ <> 0UL do
            let sq = popLsb &occ
            key <- key ^^^ zPiece board.[sq] sq

        let st = &states.[stPly]
        key <- key ^^^ zCastle st.CastlingRights

        if st.EpSquare <> NoSquare then
            key <- key ^^^ zEp (fileOf st.EpSquare)

        if sideToMove = Black then
            key <- key ^^^ Side

        key

    // --- shared en-passant capturer test (the pawn-inversion, single source) -
    /// True iff a `capturer`-colored pawn can legally capture onto `epSq` (squares a `capturer` pawn
    /// attacks `epSq` from = pawnAttacks (flipColor capturer) epSq). Reused by Make / LoadFen / ToFen.
    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member private _.EpCapturerExists (epSq: Square) (capturer: Color) : bool =
        (pawnAttacks (flipColor capturer) epSq
         &&& byTypeBB.[Pawn]
         &&& byColorBB.[capturer])
        <> 0UL

    // --- initialization ----------------------------------------------------
    /// Set the standard chess starting position (folds onto LoadFen).
    member this.SetStartPos() = this.LoadFen StartPosFen

    // --- FEN (cold path; allocation OK; hand-parsed, no String.Split/regex) -
    member this.LoadFen(fen: string) : unit =
        System.Array.Fill(board, NoPiece)
        System.Array.Clear(byTypeBB, 0, byTypeBB.Length)
        System.Array.Clear(byColorBB, 0, byColorBB.Length)
        currentKey <- 0UL
        stPly <- 0
        gamePly <- 0
        // A bulk board load invalidates the incremental SF accumulator: disable it so the piece-placement
        // below records NO deltas (32 PutPiece calls would overflow the small dirty buffer). The caller
        // re-enables via EnableNnue, which rebuilds both perspectives from scratch (SfRefresh).
        sfActive <- false
        sfTop <- 0
        sfDirtyN <- 0
        sfDirtyThreatN <- 0
        sfDirtyThreatOverflow <- false
        let n = fen.Length
        let mutable i = 0
        // 1. piece placement (ranks 8->1, files a->h)
        let mutable rank = 7
        let mutable file = 0

        while i < n && fen.[i] <> ' ' do
            let ch = fen.[i]

            if ch = '/' then
                rank <- rank - 1
                file <- 0
            elif ch >= '1' && ch <= '8' then
                file <- file + (int ch - int '0')
            else
                let color = if System.Char.IsUpper ch then White else Black

                let pt =
                    match System.Char.ToLower ch with
                    | 'p' -> Pawn
                    | 'n' -> Knight
                    | 'b' -> Bishop
                    | 'r' -> Rook
                    | 'q' -> Queen
                    | _ -> King

                this.PutPiece (makePiece color pt) (mkSquare file rank)
                file <- file + 1

            i <- i + 1

        while i < n && fen.[i] = ' ' do
            i <- i + 1
        // 2. side to move
        let stm = if i < n && fen.[i] = 'b' then Black else White
        sideToMove <- stm

        if stm = Black then
            currentKey <- currentKey ^^^ Side

        if i < n then
            i <- i + 1

        while i < n && fen.[i] = ' ' do
            i <- i + 1
        // 3. castling rights
        let mutable rights = 0

        if i < n && fen.[i] = '-' then
            i <- i + 1
        else
            while i < n && fen.[i] <> ' ' do
                (match fen.[i] with
                 | 'K' -> rights <- rights ||| WK
                 | 'Q' -> rights <- rights ||| WQ
                 | 'k' -> rights <- rights ||| BK
                 | 'q' -> rights <- rights ||| BQ
                 | _ -> ())

                i <- i + 1

        currentKey <- currentKey ^^^ zCastle rights

        while i < n && fen.[i] = ' ' do
            i <- i + 1
        // 4. en-passant target (kept only if a real capturer exists — the SF gate)
        let mutable ep = NoSquare

        if i < n && fen.[i] = '-' then
            i <- i + 1
        elif i + 1 < n && fen.[i] >= 'a' && fen.[i] <= 'h' then
            let ef = int fen.[i] - int 'a'
            let er = int fen.[i + 1] - int '1'
            i <- i + 2
            let epSq = mkSquare ef er

            if this.EpCapturerExists epSq stm then
                ep <- epSq
                currentKey <- currentKey ^^^ zEp (fileOf epSq)
                Debug.Assert((er = (if stm = White then 5 else 2)), "LoadFen: ep rank wrong for side to move")
                Debug.Assert(board.[epSq] = NoPiece, "LoadFen: ep square not empty")

                Debug.Assert(
                    board.[epSq + (if stm = White then -8 else 8)] = makePiece (flipColor stm) Pawn,
                    "LoadFen: ep victim pawn missing"
                )

        while i < n && fen.[i] = ' ' do
            i <- i + 1
        // 5. halfmove (rule50) and fullmove (tolerant: abbreviated FENs omit them)
        let mutable rule50 = 0
        let mutable fullmove = 1

        if i < n && fen.[i] >= '0' && fen.[i] <= '9' then
            let mutable v = 0

            while i < n && fen.[i] >= '0' && fen.[i] <= '9' do
                v <- v * 10 + (int fen.[i] - int '0')
                i <- i + 1

            rule50 <- v

            while i < n && fen.[i] = ' ' do
                i <- i + 1

            if i < n && fen.[i] >= '0' && fen.[i] <= '9' then
                let mutable w = 0

                while i < n && fen.[i] >= '0' && fen.[i] <= '9' do
                    w <- w * 10 + (int fen.[i] - int '0')
                    i <- i + 1

                fullmove <- w

        gamePly <- (fullmove - 1) * 2 + (if stm = Black then 1 else 0)
        // 6. seed states.[0] (every scalar explicit — zeroCreate leaves EpSquare=0=a1, CapturedPiece=0=wP)
        let st = &states.[0]
        st.Key <- currentKey
        st.CastlingRights <- rights
        st.EpSquare <- ep
        st.Rule50 <- rule50
        st.PliesFromNull <- 0
        st.CapturedPiece <- NoPiece
        st.Repetition <- 0
        st.Checkers <- 0UL
        st.BlockersW <- 0UL
        st.BlockersB <- 0UL
        st.PinnersW <- 0UL
        st.PinnersB <- 0UL
        st.CheckSqP <- 0UL
        st.CheckSqN <- 0UL
        st.CheckSqB <- 0UL
        st.CheckSqR <- 0UL
        st.CheckSqQ <- 0UL
        // 7. compute cached check-info for the side to move
        Debug.Assert(
            ((byTypeBB.[King] &&& byColorBB.[White]) <> 0UL
             && (byTypeBB.[King] &&& byColorBB.[Black]) <> 0UL),
            "LoadFen: each side must have exactly one king"
        )

        this.SetCheckInfo()
        Debug.Assert((this.RecomputeKey() = currentKey), "LoadFen: incremental key != from-scratch")

    member _.ToFen() : string =
        let sb = System.Text.StringBuilder()

        for rank in 7..-1..0 do
            let mutable empty = 0

            for file in 0..7 do
                let pc = board.[mkSquare file rank]

                if pc = NoPiece then
                    empty <- empty + 1
                else
                    if empty > 0 then
                        sb.Append(empty) |> ignore

                    empty <- 0
                    let c = "pnbrqk".[pieceType pc]
                    sb.Append(if pieceColor pc = White then System.Char.ToUpper c else c) |> ignore

            if empty > 0 then
                sb.Append(empty) |> ignore

            if rank > 0 then
                sb.Append('/') |> ignore

        sb.Append(if sideToMove = White then " w " else " b ") |> ignore
        let st = &states.[stPly]

        if st.CastlingRights = 0 then
            sb.Append('-') |> ignore
        else
            if st.CastlingRights &&& WK <> 0 then
                sb.Append('K') |> ignore

            if st.CastlingRights &&& WQ <> 0 then
                sb.Append('Q') |> ignore

            if st.CastlingRights &&& BK <> 0 then
                sb.Append('k') |> ignore

            if st.CastlingRights &&& BQ <> 0 then
                sb.Append('q') |> ignore

        sb.Append(' ') |> ignore

        if st.EpSquare <> NoSquare then
            sb.Append("abcdefgh".[fileOf st.EpSquare]) |> ignore
            sb.Append("12345678".[rankOf st.EpSquare]) |> ignore
        else
            sb.Append('-') |> ignore

        let fullmove = gamePly / 2 + 1
        sb.Append(' ').Append(st.Rule50).Append(' ').Append(fullmove) |> ignore
        sb.ToString()

    /// Construct + load a FEN in one step.
    static member OfFen(fen: string) : Position =
        let p = Position()
        p.LoadFen fen
        p

    // --- make / unmake -----------------------------------------------------
    /// Apply a LEGAL move (see file header — no legality validation here).
    member this.Make(m: Move) : unit =
        Debug.Assert((stPly + 1 < MaxPly), "Make: stack overflow")
        let us = sideToMove
        let them = flipColor us
        let from = fromSq m
        let dst = toSq m
        let pc = board.[from]
        // 1. seed key from the parent frame, clear stale ep term, advance the stack
        let prev = &states.[stPly]
        currentKey <- prev.Key ^^^ Side

        if prev.EpSquare <> NoSquare then
            currentKey <- currentKey ^^^ zEp (fileOf prev.EpSquare)

        let pCastling = prev.CastlingRights
        let pRule50 = prev.Rule50
        let pPlies = prev.PliesFromNull
        // SF NNUE: push a lazy dirty frame for real moves only. Null moves intentionally do not affect sfTop.
        if sfActive then
            this.SfBeginFrame()
        stPly <- stPly + 1
        gamePly <- gamePly + 1
        let st = &states.[stPly]
        st.CastlingRights <- pCastling
        st.Rule50 <- pRule50 + 1
        st.PliesFromNull <- pPlies + 1
        st.EpSquare <- NoSquare
        st.CapturedPiece <- NoPiece
        st.Repetition <- 0
        // 2/3. apply (fast path first per Move.fs contract)
        if not (isSpecial m) then
            let captured = board.[dst]

            if captured <> NoPiece then
                st.CapturedPiece <- captured
                st.Rule50 <- 0
                this.RemovePiece pc from
                this.SwapPiece captured pc dst
            else
                this.MovePiece pc from dst

            if pieceType pc = Pawn then
                st.Rule50 <- 0

                if (dst - from = 16) || (from - dst = 16) then // double push
                    let ep = (from + dst) / 2

                    if this.EpCapturerExists ep them then
                        st.EpSquare <- ep
                        currentKey <- currentKey ^^^ zEp (fileOf ep)
        else
            match moveFlag m with
            | FlagEnPassant ->
                let capSq = dst + (if us = White then -8 else 8) // victim on a DIFFERENT square, never dst
                let victim = makePiece them Pawn
                this.RemovePiece victim capSq
                this.MovePiece pc from dst
                st.CapturedPiece <- victim
                st.Rule50 <- 0
            | FlagPromotion ->
                let captured = board.[dst]
                let promoted = makePiece us (promoType m)

                if captured <> NoPiece then
                    st.CapturedPiece <- captured
                    this.RemovePiece pc from
                    this.SwapPiece captured promoted dst
                else
                    this.RemovePiece pc from
                    this.PutPiece promoted dst

                st.Rule50 <- 0
            | _ -> // FlagCastling: move king then rook
                let rook = makePiece us Rook
                let rf = castleRookFrom.[dst]
                let rt = castleRookTo.[dst]
                this.RemovePiece pc from
                this.RemovePiece rook rf
                this.PutPiece pc dst
                this.PutPiece rook rt
        // 4. castling-rights update (uniform across all paths; mask[dst] handles rook-captured-on-home)
        let newR =
            st.CastlingRights &&& castlingRightsMask.[from] &&& castlingRightsMask.[dst]

        if newR <> st.CastlingRights then
            currentKey <- currentKey ^^^ zCastle st.CastlingRights ^^^ zCastle newR
            st.CastlingRights <- newR
        // 5. commit
        sideToMove <- them
        st.Key <- currentKey
        this.SetCheckInfo()
        // SF NNUE: persist dirty pieces + physical FullThreats deltas, then eagerly materialize the child
        // accumulator (copy parent + apply deltas). Eval becomes O(1) — no lazy frame walk at eval time.
        if sfActive then
            this.SfCommitFrame()
            if sfEagerUpdates then this.SfEagerUpdate()

    /// Undo the move applied by Make. Pure board restore — NO key math (the parent frame holds the key).
    member this.Unmake(m: Move) : unit =
        sideToMove <- flipColor sideToMove // back to the mover `us`
        let us = sideToMove
        let them = flipColor us
        let from = fromSq m
        let dst = toSq m
        let captured = (let st = &states.[stPly] in st.CapturedPiece)

        if not (isSpecial m) then
            this.MovePieceNK board.[dst] dst from

            if captured <> NoPiece then
                this.PutPieceNK captured dst
        else
            match moveFlag m with
            | FlagEnPassant ->
                this.MovePieceNK board.[dst] dst from
                let capSq = dst + (if us = White then -8 else 8)
                this.PutPieceNK (makePiece them Pawn) capSq
            | FlagPromotion ->
                this.RemovePieceNK board.[dst] dst // remove the promoted piece first
                this.PutPieceNK (makePiece us Pawn) from

                if captured <> NoPiece then
                    this.PutPieceNK captured dst
            | _ -> // FlagCastling: king back, then rook back
                this.MovePieceNK board.[dst] dst from
                this.MovePieceNK (makePiece us Rook) castleRookTo.[dst] castleRookFrom.[dst]

        Debug.Assert((stPly > 0), "Unmake: stack underflow")
        stPly <- stPly - 1
        gamePly <- gamePly - 1
        currentKey <- (let p = &states.[stPly] in p.Key)
        // SF NNUE: pop the lazy frame. Parent materialization, if needed, is computed on demand.
        if sfActive then
            Debug.Assert((sfTop > 0), "Unmake: sfTop underflow")
            sfTop <- sfTop - 1

    /// Play a null move (side passes). PRE: not in check.
    member this.MakeNull() : unit =
        Debug.Assert((stPly + 1 < MaxPly), "MakeNull: stack overflow")
        let prev = &states.[stPly]
        Debug.Assert((prev.Checkers = 0UL), "MakeNull: side to move must not be in check")
        currentKey <- prev.Key ^^^ Side

        if prev.EpSquare <> NoSquare then
            currentKey <- currentKey ^^^ zEp (fileOf prev.EpSquare)

        let pCastling = prev.CastlingRights
        let pRule50 = prev.Rule50
        // SF NNUE: a null move moves no pieces ⇒ the accumulator is identical ⇒ no snapshot/update needed.
        if sfActive then sfDirtyN <- 0
        stPly <- stPly + 1
        gamePly <- gamePly + 1
        let st = &states.[stPly]
        st.CastlingRights <- pCastling
        st.EpSquare <- NoSquare
        st.Rule50 <- pRule50 + 1
        st.PliesFromNull <- 0
        st.CapturedPiece <- NoPiece
        st.Repetition <- 0
        sideToMove <- flipColor sideToMove
        st.Key <- currentKey
        this.SetCheckInfo()

    member this.UnmakeNull() : unit =
        Debug.Assert((stPly > 0), "UnmakeNull: stack underflow")
        stPly <- stPly - 1
        gamePly <- gamePly - 1
        sideToMove <- flipColor sideToMove
        currentKey <- (let p = &states.[stPly] in p.Key)
        // SF NNUE: a null move left the accumulator unchanged ⇒ nothing to restore.

    // --- GivesCheck: does `m` give check? (search/movegen; perft does not use it) -------------------
    // Normal: direct via cached CheckSquares (valid because a `from`-blocked ray would mean the enemy king
    // was already in check = illegal). Special moves change occupancy on the checking ray, so each
    // recomputes against POST-MOVE occupancy. Discovered (any kind): the moved piece was a blocker for the
    // enemy king and leaves its line.
    member this.GivesCheck(m: Move) : bool =
        let us = sideToMove
        let them = flipColor us
        let theirKing = this.KingSquare them
        let from = fromSq m
        let dst = toSq m
        let pc = board.[from]
        let occ = byTypeBB.[AllPieces]
        // discovered check (the moved piece uncovers a slider on the enemy king)
        if (testBit (this.BlockersForKing them) from) && not (aligned from dst theirKing) then
            true
        elif not (isSpecial m) then
            testBit (this.CheckSquares(pieceType pc)) dst
        else
            match moveFlag m with
            | FlagPromotion ->
                let occ' = occ ^^^ (1UL <<< from) // pawn vacates from; promoted piece on dst
                (attacksFrom (promoType m) us dst occ' &&& (1UL <<< theirKing)) <> 0UL
            | FlagCastling ->
                let rf = castleRookFrom.[dst]
                let rt = castleRookTo.[dst]

                let occ' =
                    (occ ^^^ (1UL <<< from) ^^^ (1UL <<< rf)) ||| (1UL <<< dst) ||| (1UL <<< rt)

                (rookAttacks rt occ' &&& (1UL <<< theirKing)) <> 0UL
            | _ -> // EnPassant: post-EP occupancy recompute
                let capSq = dst + (if us = White then -8 else 8)
                let occ' = (occ ^^^ (1UL <<< from) ^^^ (1UL <<< capSq)) ||| (1UL <<< dst)
                (this.AttackersTo theirKing occ' &&& byColorBB.[us]) <> 0UL

    // --- in-check resolution shared by IsPseudoLegal's NORMAL/Promotion arms -----------------------
    // king=true  -> destination must be safe with the king REMOVED from occ (X-ray through the vacated
    //               square) — this is the one piece of legality IsPseudoLegal does (matching SF pseudo_legal).
    // king=false -> under single check the move must land on between(ksq,checker)|checkers (capture/interpose);
    //               under double check a non-king move is illegal. Pin legality for non-king pieces is left
    //               to isLegal (exactly as SF). PRE: the move already passed its shape check.
    member private this.ResolvesCheck (us: Color) (from: Square) (dst: Square) (isKing: bool) : bool =
        let them = flipColor us

        if isKing then
            (this.AttackersTo dst (byTypeBB.[AllPieces] ^^^ (bit from)) &&& byColorBB.[them]) = 0UL
        else
            let st = &states.[stPly]

            if st.Checkers = 0UL then
                true
            elif moreThanOne st.Checkers then
                false // double check, non-king -> illegal
            else
                let ksq = this.KingSquare us
                let checker = lsb st.Checkers
                testBit ((between ksq checker) ||| st.Checkers) dst

    // --- IsPseudoLegal (SF Position::pseudo_legal): is `m` playable in THIS position by sideToMove? --
    // Used by the MovePick to emit a TT/killer/counter move WITHOUT generating it (Make assumes
    // legality). Fully INLINE — it MUST NOT call MoveGeneration.generate (Position compiles first); the
    // three special-move validators are hand-rolled from Position's own accessors (mirroring tryCastle /
    // genPawnMoves / the evasion target so that IsPseudoLegal == membership in the generated set). Pin
    // legality for non-king pieces is deferred to isLegal, exactly as SF.
    member this.IsPseudoLegal(m: Move) : bool =
        if not (isOk m) then
            false // rejects MoveNone, MoveNull, from==to
        else
            let us = sideToMove
            let them = flipColor us
            let from = fromSq m
            let dst = toSq m
            let pc = board.[from]

            if pc = NoPiece || pieceColor pc <> us then
                false // empty/enemy/wrong piece on `from`
            else
                let occ = byTypeBB.[AllPieces]
                let pt = pieceType pc

                match moveFlag m with
                // ---- CASTLING: replicate genCastling + tryCastle from Position geometry --------------
                | FlagCastling ->
                    if pt <> King || from <> this.KingSquare us then
                        false
                    else
                        let right =
                            if us = White then
                                (if dst = castleKingDest.[WK] then WK
                                 elif dst = castleKingDest.[WQ] then WQ
                                 else 0)
                            else
                                (if dst = castleKingDest.[BK] then BK
                                 elif dst = castleKingDest.[BQ] then BQ
                                 else 0)

                        if right = 0 then
                            false
                        elif not (this.CanCastle right) then
                            false
                        elif this.InCheck then
                            false
                        elif (castleEmptyPath.[right] &&& occ) <> 0UL then
                            false
                        elif board.[castleRookOrigin.[right]] <> makePiece us Rook then
                            false
                        else
                            let mutable path = castleKingPath.[right]
                            let mutable safe = true

                            while path <> 0UL && safe do
                                if this.AttackedBy them (popLsb &path) then
                                    safe <- false

                            safe
                // ---- EN PASSANT --------------------------------------------------------------------
                | FlagEnPassant ->
                    let st = &states.[stPly]

                    if pt <> Pawn || st.EpSquare = NoSquare || dst <> st.EpSquare then
                        false
                    elif not (testBit (pawnAttacks them dst) from) then
                        false // an `us` pawn captures onto dst
                    else
                        let capSq = if us = White then dst - 8 else dst + 8

                        if board.[capSq] <> makePiece them Pawn then false
                        // generator's evasion EP gate: in check, the EP victim must be the checker.
                        elif st.Checkers <> 0UL then testBit st.Checkers capSq
                        else true
                // ---- PROMOTION (pawn onto the back rank by push-to-empty or capture-enemy) -----------
                | FlagPromotion ->
                    if pt <> Pawn then
                        false
                    else
                        let srcRank = if us = White then 6 else 1
                        let dstRank = if us = White then 7 else 0

                        if rankOf from <> srcRank || rankOf dst <> dstRank then
                            false
                        else
                            let one = if us = White then from + 8 else from - 8
                            let push = dst = one && board.[dst] = NoPiece

                            let cap =
                                testBit (pawnAttacks us from) dst
                                && board.[dst] <> NoPiece
                                && pieceColor board.[dst] = them

                            if not (push || cap) then
                                false
                            else
                                this.ResolvesCheck us from dst false
                // ---- NORMAL ------------------------------------------------------------------------
                | _ ->
                    let cap = board.[dst]

                    if cap <> NoPiece && pieceColor cap = us then
                        false // friendly capture
                    else
                        let shapeOk =
                            if pt = Pawn then
                                let backRank = if us = White then 7 else 0

                                if rankOf dst = backRank then
                                    false // NORMAL onto back rank -> must be Promotion
                                else
                                    let one = if us = White then from + 8 else from - 8
                                    let two = if us = White then from + 16 else from - 16
                                    let startRank = if us = White then 1 else 6

                                    if dst = one then
                                        board.[dst] = NoPiece
                                    elif dst = two then
                                        rankOf from = startRank && board.[one] = NoPiece && board.[dst] = NoPiece
                                    else
                                        testBit (pawnAttacks us from) dst && cap <> NoPiece // diagonal capture
                            else
                                testBit (attacksFrom pt us from occ) dst // non-pawn (blocked sliders fail here)

                        if not shapeOk then
                            false
                        else
                            this.ResolvesCheck us from dst (pt = King)

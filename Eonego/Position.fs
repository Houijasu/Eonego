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

    // --- SF NNUE incremental accumulator (gated by sfActive; bound raw weight arrays, no SfNetwork type) ---
    let mutable sfActive = false
    let sfWhiteAcc: int[] = Array.zeroCreate SfAccumulator.L1
    let sfWhitePsqt: int[] = Array.zeroCreate SfAccumulator.PsqtBuckets
    let sfBlackAcc: int[] = Array.zeroCreate SfAccumulator.L1
    let sfBlackPsqt: int[] = Array.zeroCreate SfAccumulator.PsqtBuckets
    let mutable sfFtWeights: int16[] = Array.empty
    let mutable sfFtPsqt: int[] = Array.empty
    let mutable sfFtBiases: int16[] = Array.empty
    let sfPlyStride = 2 * (SfAccumulator.L1 + SfAccumulator.PsqtBuckets)
    // Per-ply snapshot stack, lazy-allocated only when sfActive. Sized MaxPly because stPly reaches
    // (root moves replayed in SetupRoot) + (search depth). FOOTPRINT: MaxPly*sfPlyStride int32 ~= 12 MB
    // per Position, i.e. ~12 MB PER LazySMP Worker (each owns its own Position). Deferred memory mitigations
    // (this phase is correctness-first, int32 to stay bit-exact with the from-scratch oracle): narrow to
    // int16 (halves it, matches SF), or a forward-accumulator model that avoids the full per-ply snapshot.
    let mutable sfStack: int[] = Array.empty
    let sfDirtyPc: int[] = Array.zeroCreate 8
    let sfDirtySq: int[] = Array.zeroCreate 8
    let sfDirtySign: int[] = Array.zeroCreate 8
    let mutable sfDirtyN = 0

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

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member private this.RemovePiece (pc: Piece) (sq: Square) =
        Debug.Assert(pc <> NoPiece, "RemovePiece: NoPiece")
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

    // --- SF NNUE incremental accumulator public accessors + enable ---
    member _.SfActive: bool = sfActive
    member _.SfWhiteAcc: int[] = sfWhiteAcc
    member _.SfWhitePsqt: int[] = sfWhitePsqt
    member _.SfBlackAcc: int[] = sfBlackAcc
    member _.SfBlackPsqt: int[] = sfBlackPsqt

    /// Refresh one perspective from scratch (at enable + on a king move). Mirrors SfNnue.buildAcc.
    member private this.SfRefresh(pColor: Color) =
        let acc = if pColor = White then sfWhiteAcc else sfBlackAcc
        let psqt = if pColor = White then sfWhitePsqt else sfBlackPsqt
        let ksq = this.KingSquare pColor
        for j in 0 .. SfAccumulator.L1 - 1 do
            acc.[j] <- int sfFtBiases.[j]
        System.Array.Clear(psqt, 0, SfAccumulator.PsqtBuckets)
        for sq in 0 .. 63 do
            let pc = board.[sq]
            if pc <> NoPiece then
                SfAccumulator.addFeature acc psqt sfFtWeights sfFtPsqt (SfAccumulator.makeIndex pColor pc sq ksq) 1 SfAccumulator.UseAvx2

    /// Bind raw weight arrays + refresh both perspectives. ROOT ONLY.
    member this.EnableSfNnue (ftWeights: int16[]) (ftPsqt: int[]) (ftBiases: int16[]) =
        Debug.Assert((stPly = 0), "EnableSfNnue must be called at the root (stPly = 0)")
        sfFtWeights <- ftWeights
        sfFtPsqt <- ftPsqt
        sfFtBiases <- ftBiases
        if Array.isEmpty sfStack then sfStack <- Array.zeroCreate (MaxPly * sfPlyStride)
        sfActive <- true
        this.SfRefresh White
        this.SfRefresh Black

    /// Apply the dirty buffer to both perspectives (refresh a side iff its own king moved). Call AFTER the
    /// board mutation completes.
    member private this.SfUpdate() =
        let mutable whiteKingMoved = false
        let mutable blackKingMoved = false
        for i in 0 .. sfDirtyN - 1 do
            if pieceType sfDirtyPc.[i] = King then
                if pieceColor sfDirtyPc.[i] = White then whiteKingMoved <- true else blackKingMoved <- true
        if whiteKingMoved then this.SfRefresh White
        else
            let ksq = this.KingSquare White
            for i in 0 .. sfDirtyN - 1 do
                SfAccumulator.addFeature sfWhiteAcc sfWhitePsqt sfFtWeights sfFtPsqt (SfAccumulator.makeIndex White sfDirtyPc.[i] sfDirtySq.[i] ksq) sfDirtySign.[i] SfAccumulator.UseAvx2
        if blackKingMoved then this.SfRefresh Black
        else
            let ksq = this.KingSquare Black
            for i in 0 .. sfDirtyN - 1 do
                SfAccumulator.addFeature sfBlackAcc sfBlackPsqt sfFtWeights sfFtPsqt (SfAccumulator.makeIndex Black sfDirtyPc.[i] sfDirtySq.[i] ksq) sfDirtySign.[i] SfAccumulator.UseAvx2

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
        // re-enables via EnableSfNnue, which rebuilds both perspectives from scratch (SfRefresh).
        sfActive <- false
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
        // SF NNUE: snapshot parent accumulator before stPly advances; reset dirty buffer.
        if sfActive then
            let off = stPly * sfPlyStride
            System.Array.Copy(sfWhiteAcc, 0, sfStack, off, SfAccumulator.L1)
            System.Array.Copy(sfWhitePsqt, 0, sfStack, off + SfAccumulator.L1, SfAccumulator.PsqtBuckets)
            System.Array.Copy(sfBlackAcc, 0, sfStack, off + SfAccumulator.L1 + SfAccumulator.PsqtBuckets, SfAccumulator.L1)
            System.Array.Copy(sfBlackPsqt, 0, sfStack, off + 2 * SfAccumulator.L1 + SfAccumulator.PsqtBuckets, SfAccumulator.PsqtBuckets)
            sfDirtyN <- 0
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
                this.RemovePiece captured dst
                st.CapturedPiece <- captured
                st.Rule50 <- 0

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

                if captured <> NoPiece then
                    this.RemovePiece captured dst
                    st.CapturedPiece <- captured

                this.RemovePiece pc from
                this.PutPiece (makePiece us (promoType m)) dst
                st.Rule50 <- 0
            | _ -> // FlagCastling: move king then rook
                this.MovePiece pc from dst
                this.MovePiece (makePiece us Rook) castleRookFrom.[dst] castleRookTo.[dst]
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
        // SF NNUE: apply accumulated dirty buffer to both perspectives.
        if sfActive then this.SfUpdate()

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
        // SF NNUE: restore parent accumulator from snapshot (stPly is now back at parent ply).
        if sfActive then
            let off = stPly * sfPlyStride
            System.Array.Copy(sfStack, off, sfWhiteAcc, 0, SfAccumulator.L1)
            System.Array.Copy(sfStack, off + SfAccumulator.L1, sfWhitePsqt, 0, SfAccumulator.PsqtBuckets)
            System.Array.Copy(sfStack, off + SfAccumulator.L1 + SfAccumulator.PsqtBuckets, sfBlackAcc, 0, SfAccumulator.L1)
            System.Array.Copy(sfStack, off + 2 * SfAccumulator.L1 + SfAccumulator.PsqtBuckets, sfBlackPsqt, 0, SfAccumulator.PsqtBuckets)

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
        // SF NNUE: snapshot parent accumulator before stPly advances; no pieces move so no dirty entries.
        if sfActive then
            let off = stPly * sfPlyStride
            System.Array.Copy(sfWhiteAcc, 0, sfStack, off, SfAccumulator.L1)
            System.Array.Copy(sfWhitePsqt, 0, sfStack, off + SfAccumulator.L1, SfAccumulator.PsqtBuckets)
            System.Array.Copy(sfBlackAcc, 0, sfStack, off + SfAccumulator.L1 + SfAccumulator.PsqtBuckets, SfAccumulator.L1)
            System.Array.Copy(sfBlackPsqt, 0, sfStack, off + 2 * SfAccumulator.L1 + SfAccumulator.PsqtBuckets, SfAccumulator.PsqtBuckets)
            sfDirtyN <- 0
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
        // SF NNUE: restore parent accumulator from snapshot (stPly is now back at parent ply).
        if sfActive then
            let off = stPly * sfPlyStride
            System.Array.Copy(sfStack, off, sfWhiteAcc, 0, SfAccumulator.L1)
            System.Array.Copy(sfStack, off + SfAccumulator.L1, sfWhitePsqt, 0, SfAccumulator.PsqtBuckets)
            System.Array.Copy(sfStack, off + SfAccumulator.L1 + SfAccumulator.PsqtBuckets, sfBlackAcc, 0, SfAccumulator.L1)
            System.Array.Copy(sfStack, off + 2 * SfAccumulator.L1 + SfAccumulator.PsqtBuckets, sfBlackPsqt, 0, SfAccumulator.PsqtBuckets)

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

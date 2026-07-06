/// Eonego — mutable board state for the chess engine.
///
/// Position is a single [<Sealed>] mutable CLASS (one heap allocation reused across the whole search via
/// make/unmake — NOT copy-make). It owns boards: `byTypeBB.[0..5]` = Pawn..King (both
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

/// Absolute accumulator-stack depth cap. A node whose `stPly` (root moves already played + search ply from
/// root) reaches this MUST NOT enter its body / Make a child, or `Position.BeginFrame` overflows the
/// `AccMaxPly`-sized per-frame arrays. The search's RELATIVE `ply` cap (`Search.MaxSearchPly`) alone is
/// insufficient when the root position carries many played moves (UCI `position ... moves m1 m2 ...`), since
/// `stPly = ply + rootMoveCount` and the accumulator frames are indexed by the ABSOLUTE `top`/`stPly`.
/// MUST stay `<= AccMaxPly - 1` (the type-private `AccMaxPly` is 256); kept at 255 for a tight, safe bound.
[<Literal>]
let AccStackLimit = 255

[<Literal>]
let StartPosFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

/// Opt-in per-search phase profiling (`EONEGO_PROF=1`): coarse Stopwatch-tick counters over the NNUE
/// accumulator hot path, printed as one `info string prof ...` line by `Search.go`. Counters are plain
/// static mutables — meaningful at Threads = 1 only (racy-but-harmless otherwise). Every write is gated on
/// `Enabled` (a static readonly bool the JIT folds), so the disabled cost is one predictable branch.
/// This exists because dotnet-trace sampling misattributes this workload (PollGC artifact, audit 2026-07-01).
/// EONEGO_FINNY=0 falls back to the from-scratch BuildFull refresh path (A/B + reference oracle for the
/// finny-table refresh; see Position.RefreshFromFinny). Default ON.
let UseFinny =
    System.Environment.GetEnvironmentVariable("EONEGO_FINNY") <> "0"

module PosProf =
    let Enabled = System.Environment.GetEnvironmentVariable("EONEGO_PROF") = "1"
    let mutable tMake = 0L // Position.Make total (incl. any eager accumulator work)
    let mutable tEager = 0L // EagerUpdate total (subset of tMake)
    let mutable tEnsure = 0L // EnsureBothComputed non-trivial body (lazy catch-up walks + cache probes)
    let mutable tBuild = 0L // BuildFull/BuildFullBoth bodies (subset of tEager or tEnsure)
    let mutable tEval = 0L // full forward evals (Search.evalPos)
    let mutable nMake = 0L
    let mutable nEnsure = 0L
    let mutable nBuild = 0L
    let mutable nEval = 0L
    let mutable maxThreatN = 0L // high-water mark of per-move physical dirty-threat edges (sizing data)
    // Ensure-bucket sub-counters (2026-07-02 speed audit): where do the ~48%-of-wall catch-up walks go?
    let mutable tEnumThreats = 0L // Threats enumeration calls (subset of tBuild — BuildFull/finny refresh)
    let mutable tGather = 0L // EnsureChangedThreats (dirty-threat diff conversion; subset of tEnsure)
    let mutable tApply = 0L // ApplyFrameFusedTo (gather + fused kernel; subset of tEnsure walks)
    let mutable nEnumThreats = 0L
    let mutable nGather = 0L // calls that did WORK (missed the per-frame changed cache)
    let mutable nApply = 0L // perspective-frames materialized by walks
    // Tree-shape counters (2026-07-06, policy-net campaign Phase 0): how often does the picker actually
    // reach quiet scoring, and which move class causes beta cutoffs? These size the slice a policy-net
    // ordering prior could even address (a cutoff by TT move/capture/killer/countermove is emitted by
    // earlier picker stages regardless of quiet ordering). Written by Search.fs (nodes, cutoffs) and
    // MovePick.fs (StgQuietInit); same 1T semantics as the rest.
    let mutable nNodesMain = 0L // negamax nodes that reached the move loop (built a main picker)
    let mutable nNodesQs = 0L // qsearch nodes that reached the move loop
    let mutable nQuietInit = 0L // StgQuietInit executions that generated + scored quiets (not skipQuiets)
    let QuietInitByDepth: int64[] = Array.zeroCreate 64 // nQuietInit bucketed by picker depth (clamped)
    let mutable nCutoffs = 0L // negamax beta cutoffs
    let mutable nCutTT = 0L // ... caused by the TT move
    let mutable nCutCapture = 0L // ... caused by a capture/promotion (non-quiet)
    let mutable nCutRefutation = 0L // ... caused by a killer1/killer2/countermove quiet
    let mutable nCutQuietTail = 0L // ... caused by a NON-refutation quiet (the policy-addressable slice)
    let CutQuietIdx: int64[] = Array.zeroCreate 16 // nCutQuietTail by tried-quiet index (clamped to 15)

    let reset () =
        tMake <- 0L
        tEager <- 0L
        tEnsure <- 0L
        tBuild <- 0L
        tEval <- 0L
        nMake <- 0L
        nEnsure <- 0L
        nBuild <- 0L
        nEval <- 0L
        maxThreatN <- 0L
        tEnumThreats <- 0L
        tGather <- 0L
        tApply <- 0L
        nEnumThreats <- 0L
        nGather <- 0L
        nApply <- 0L
        nNodesMain <- 0L
        nNodesQs <- 0L
        nQuietInit <- 0L
        Array.fill QuietInitByDepth 0 QuietInitByDepth.Length 0L
        nCutoffs <- 0L
        nCutTT <- 0L
        nCutCapture <- 0L
        nCutRefutation <- 0L
        nCutQuietTail <- 0L
        Array.fill CutQuietIdx 0 CutQuietIdx.Length 0L

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
let private rayBeyond: Bitboard[] = Array.zeroCreate (64 * 64) // [from][through] -> squares beyond through

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

                    rayBeyond.[(from <<< 6) + throughSq] <- b

do initTables ()

// ---------------------------------------------------------------------------
// PieceValue — the single SEE / MVV-LVA / capture-history value table (centipawns), indexed by
// PieceType (Pawn..King). King = 0 is the SEE sentinel: a king is never an SEE victim and its value is
// never subtracted (the KING branch of SEE returns/terminates first). This is a literal-initialized
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
      // Pawn-only Zobrist key (both colors' pawns, piece-square terms only — no castle/ep/side). Feeds the
      // correction-history index; maintained by the same mutation choke points as Key.
      mutable PawnKey: uint64
      // Minor-piece Zobrist key (both colors' knights, bishops AND kings — minor corrhist keying;
      // the king term makes king-safety structures distinguishable). Same contract as PawnKey.
      mutable MinorKey: uint64
      mutable MajorKey: uint64
      mutable NonPawnKey: uint64
      mutable CastlingRights: int
      mutable EpSquare: Square // landing square, or NoSquare (64)
      mutable Rule50: int
      mutable PliesFromNull: int
      mutable CapturedPiece: Piece // removed by the move producing this node; NoPiece if none
      mutable Repetition: int // 0 none; search-only (perft ignores)
      // cached check-info, for the side to move AT this node
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
    let mutable currentPawnKey = 0UL
    let mutable currentMinorKey = 0UL
    let mutable currentMajorKey = 0UL
    let mutable currentNonPawnKey = 0UL
    let states: StateInfo[] = Array.zeroCreate MaxPly
    let mutable stPly = 0

    // --- FullThreats NNUE lazy accumulator (gated by active) ----------------------------------------
    // MERGED accumulator: HalfKA and FullThreats both add into the same L1 cells. Make records dirty frame
    // metadata; eval/read materializes the current top frame on demand. Dirty threats are physical edges,
    // converted to perspective-dependent indices through delegates bound by NNUE.fs to avoid Position->Threats.
    [<Literal>]
    let MaxThreats = 256

    [<Literal>]
    let AccMaxPly = 256

    let mutable active = false
    let mutable eagerUpdates = false
    let mutable top = 0
    // Phase 1 — optional lock-free NNUE accumulator checkpoint cache. When non-null, `EnsureBothComputed`
    // consults it as a fast-path before walking the lazy frame stack, and populates it on a successful
    // materialization. Owned by `SearchControl`, bound per-worker via `BindCheckpoint`; cleared via the
    // owning table's `Clear()` between searches (NOT here — the position may outlive multiple searches).
    let mutable checkpoint: AccCheckpointTable = null
    let mutable accW: int16[] = Array.empty
    let mutable accB: int16[] = Array.empty
    let mutable psqW: int[] = Array.empty
    let mutable psqB: int[] = Array.empty
    let mutable computedW: bool[] = Array.empty
    let mutable computedB: bool[] = Array.empty
    let tmpW: int[] = Array.zeroCreate MaxThreats // enumeration scratch, white perspective
    let tmpB: int[] = Array.zeroCreate MaxThreats // enumeration scratch, black perspective
    let changedW: int[] = Array.zeroCreate Accumulator.MaxDirtyThreats
    let changedB: int[] = Array.zeroCreate Accumulator.MaxDirtyThreats
    // Fused-apply gather scratch (dedicated — tmpW/tmpB are aliased by EnumThreats). HalfKA lists are sized
    // for the finny board-diff worst case (32 subs + 32 adds would only occur split across the two lists;
    // 40 gives slack + assert headroom). Threat lists hold decoded per-perspective row indices.
    let fusedHalfAdd: int[] = Array.zeroCreate 40
    let fusedHalfSub: int[] = Array.zeroCreate 40
    let fusedThrAdd: int[] = Array.zeroCreate Accumulator.MaxDirtyThreats
    let fusedThrSub: int[] = Array.zeroCreate Accumulator.MaxDirtyThreats
    // Finny table ("AccumulatorCaches"): per (king square x perspective) cached HalfKA-only accumulator
    // (bias + Σ HalfKA rows for the snapshot board) + the board snapshot it was built from. A refresh
    // (own-king move / threat overflow barrier) becomes a board DIFF against the entry instead of a
    // from-scratch rebuild; threats are NEVER cached (re-enumerated fresh — they depend on the live board).
    // Entry state is path-independent (add/sub are exact modular inverses), so entries survive unmakes and
    // whole searches; the only invalidation point is EnableNNUE (net rebind) via ResetFinny.
    // ~292 KB per Position: acc 64*2*L1 int16 + psqt 64*2*8 int32 + board 64*2*64 int32.
    let mutable finnyAcc: int16[] = Array.empty
    let mutable finnyPsqt: int[] = Array.empty
    let mutable finnyBoard: int[] = Array.empty
    let mutable biases: int16[] = Array.empty
    let mutable halfWeights: int16[] = Array.empty
    let mutable halfPsqt: int[] = Array.empty
    let mutable threatWeights: sbyte[] = Array.empty
    let mutable threatPsqt: int[] = Array.empty
    // 64B-alignment element offsets (2026-07-02): row 0 / frame 0 of the corresponding array starts at this
    // element so that its address is 64B-aligned (allocAligned64/copyAligned64). Frame and row strides are
    // all 64B multiples, so a single base offset aligns every frame/row. halfW/threatW come from the
    // Network (EnableNNUE); the accumulator-stack and finny bases are set by EnsureStorage.
    let mutable halfWBase = 0
    let mutable threatWBase = 0
    let mutable accBaseW = 0
    let mutable accBaseB = 0
    let mutable psqBaseW = 0
    let mutable psqBaseB = 0
    let mutable finnyAccBase = 0
    let mutable finnyPsqBase = 0
    let mutable threatFn: (System.Func<Position, int, int[], int>) | null = null
    let mutable threatFnBoth: (System.Func<Position, int[], int[], int64>) | null = null
    let mutable threatFnChangedBoth: (System.Func<Position, int[], int, int, int[], int[], int64>) | null = null
    let mutable frameDirtyPc: int[] = Array.empty
    let mutable frameDirtySq: int[] = Array.empty
    let mutable frameDirtySign: int[] = Array.empty
    let mutable frameDirtyN: int[] = Array.empty
    let mutable frameThreats: int[] = Array.empty
    let mutable frameThreatN: int[] = Array.empty
    let mutable frameChangedW: int[] = Array.empty
    let mutable frameChangedB: int[] = Array.empty
    let mutable frameChangedNW: int[] = Array.empty
    let mutable frameChangedNB: int[] = Array.empty
    let mutable frameChangedValid: bool[] = Array.empty
    let mutable frameChangedKsqW: int[] = Array.empty
    let mutable frameChangedKsqB: int[] = Array.empty
    let mutable frameWhiteKingMoved: bool[] = Array.empty
    let mutable frameBlackKingMoved: bool[] = Array.empty
    let mutable frameThreatOverflow: bool[] = Array.empty
    let dirtyPc: int[] = Array.zeroCreate Accumulator.MaxDirtyPieces
    let dirtySq: int[] = Array.zeroCreate Accumulator.MaxDirtyPieces
    let dirtySign: int[] = Array.zeroCreate Accumulator.MaxDirtyPieces
    let mutable dirtyN = 0
    let dirtyThreats: int[] = Array.zeroCreate Accumulator.MaxDirtyThreats
    let mutable dirtyThreatN = 0
    let mutable dirtyThreatOverflow = false

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member private _.AppendDirtyThreat(putPiece: bool, pc: Piece, attacked: Piece, from: Square, too: Square) =
        if pc <> NoPiece && attacked <> NoPiece && pieceType pc <> King && not dirtyThreatOverflow then
            if dirtyThreatN < Accumulator.MaxDirtyThreats then
                let edge = Accumulator.packDirtyThreatEdge pc from too attacked
                dirtyThreats.[dirtyThreatN] <- Accumulator.packSignedDirtyThreat edge (if putPiece then 1 else -1)
                dirtyThreatN <- dirtyThreatN + 1
            else
                dirtyThreatOverflow <- true

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member private _.RayBeyond(from: Square, throughSq: Square) : Bitboard =
        rayBeyond.[(from <<< 6) + throughSq]

    member private _.PawnPushOrAttacks(c: Color, sq: Square) : Bitboard =
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
    member private this.ProcessSliders
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
                let ray = this.RayBeyond(sliderSq, sq)
                let discovered = ray &&& (rAttacks ||| bAttacks) &&& occupiedNoK

                if discovered <> 0UL && ((ray &&& noRaysContaining) <> noRaysContaining) then
                    let threatenedSq = lsb discovered
                    this.AppendDirtyThreat(not putPiece, slider, board.[threatenedSq], sliderSq, threatenedSq)

            if addDirectAttacks then
                this.AppendDirtyThreat(putPiece, slider, pc, sliderSq, sq)

    member private this.UpdatePieceThreats(pc: Piece, putPiece: bool, sq: Square, computeRay: bool, noRaysContaining: Bitboard) =
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
                    this.ProcessSliders(sliders, putPiece, pc, sq, computeRay, noRaysContaining, occupiedNoK, rAttacks, bAttacks, false)
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
                    let whiteAttacks = this.PawnPushOrAttacks(White, sq)
                    let blackAttacks = this.PawnPushOrAttacks(Black, sq)
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
                    this.AppendDirtyThreat(putPiece, pc, board.[too], sq, too)

                if computeRay then
                    this.ProcessSliders(sliders, putPiece, pc, sq, computeRay, noRaysContaining, occupiedNoK, rAttacks, bAttacks, true)
                else
                    incoming <- incoming ||| sliders

                while incoming <> 0UL do
                    let from = popLsb &incoming
                    let src = board.[from]

                    if src <> NoPiece && pieceType src <> King then
                        this.AppendDirtyThreat(putPiece, src, pc, from, sq)

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
        if pt = Pawn then
            currentPawnKey <- currentPawnKey ^^^ zPiece pc sq
        else
            currentNonPawnKey <- currentNonPawnKey ^^^ zPiece pc sq
            if pt = Knight || pt = Bishop || pt = King then
                currentMinorKey <- currentMinorKey ^^^ zPiece pc sq
            elif pt = Rook || pt = Queen then
                currentMajorKey <- currentMajorKey ^^^ zPiece pc sq
        if active then
            dirtyPc.[dirtyN] <- pc; dirtySq.[dirtyN] <- sq; dirtySign.[dirtyN] <- 1; dirtyN <- dirtyN + 1
            this.UpdatePieceThreats(pc, true, sq, true, System.UInt64.MaxValue)

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member private this.RemovePiece (pc: Piece) (sq: Square) =
        Debug.Assert(pc <> NoPiece, "RemovePiece: NoPiece")
        if active then
            this.UpdatePieceThreats(pc, false, sq, true, System.UInt64.MaxValue)

        let b = 1UL <<< sq
        let pt = pieceType pc
        let c = pieceColor pc
        byTypeBB.[pt] <- byTypeBB.[pt] ^^^ b // bit known set -> XOR clears
        byTypeBB.[AllPieces] <- byTypeBB.[AllPieces] ^^^ b
        byColorBB.[c] <- byColorBB.[c] ^^^ b
        board.[sq] <- NoPiece
        currentKey <- currentKey ^^^ zPiece pc sq
        if pt = Pawn then
            currentPawnKey <- currentPawnKey ^^^ zPiece pc sq
        else
            currentNonPawnKey <- currentNonPawnKey ^^^ zPiece pc sq
            if pt = Knight || pt = Bishop || pt = King then
                currentMinorKey <- currentMinorKey ^^^ zPiece pc sq
            elif pt = Rook || pt = Queen then
                currentMajorKey <- currentMajorKey ^^^ zPiece pc sq
        if active then
            dirtyPc.[dirtyN] <- pc; dirtySq.[dirtyN] <- sq; dirtySign.[dirtyN] <- -1; dirtyN <- dirtyN + 1

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member private this.MovePiece (pc: Piece) (from: Square) (dst: Square) =
        Debug.Assert(pc <> NoPiece, "MovePiece: NoPiece")
        // PRE: dst is empty for pt and color c (captures call RemovePiece on dst first).
        let fromTo = (1UL <<< from) ^^^ (1UL <<< dst)
        if active then
            this.UpdatePieceThreats(pc, false, from, true, fromTo)

        let pt = pieceType pc
        let c = pieceColor pc
        byTypeBB.[pt] <- byTypeBB.[pt] ^^^ fromTo
        byTypeBB.[AllPieces] <- byTypeBB.[AllPieces] ^^^ fromTo
        byColorBB.[c] <- byColorBB.[c] ^^^ fromTo
        board.[from] <- NoPiece
        board.[dst] <- pc
        currentKey <- currentKey ^^^ zPiece pc from ^^^ zPiece pc dst
        if pt = Pawn then
            currentPawnKey <- currentPawnKey ^^^ zPiece pc from ^^^ zPiece pc dst
        else
            currentNonPawnKey <- currentNonPawnKey ^^^ zPiece pc from ^^^ zPiece pc dst
            if pt = Knight || pt = Bishop || pt = King then
                currentMinorKey <- currentMinorKey ^^^ zPiece pc from ^^^ zPiece pc dst
            elif pt = Rook || pt = Queen then
                currentMajorKey <- currentMajorKey ^^^ zPiece pc from ^^^ zPiece pc dst
        if active then
            dirtyPc.[dirtyN] <- pc; dirtySq.[dirtyN] <- from; dirtySign.[dirtyN] <- -1; dirtyN <- dirtyN + 1
            dirtyPc.[dirtyN] <- pc; dirtySq.[dirtyN] <- dst;  dirtySign.[dirtyN] <- 1;  dirtyN <- dirtyN + 1
            this.UpdatePieceThreats(pc, true, dst, true, fromTo)

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
        if pieceType oldPc = Pawn then
            currentPawnKey <- currentPawnKey ^^^ zPiece oldPc sq
        else
            currentNonPawnKey <- currentNonPawnKey ^^^ zPiece oldPc sq
            if pieceType oldPc = Knight || pieceType oldPc = Bishop || pieceType oldPc = King then
                currentMinorKey <- currentMinorKey ^^^ zPiece oldPc sq
            elif pieceType oldPc = Rook || pieceType oldPc = Queen then
                currentMajorKey <- currentMajorKey ^^^ zPiece oldPc sq
        if active then
            dirtyPc.[dirtyN] <- oldPc; dirtySq.[dirtyN] <- sq; dirtySign.[dirtyN] <- -1; dirtyN <- dirtyN + 1
            this.UpdatePieceThreats(oldPc, false, sq, false, 0UL)

        byTypeBB.[pieceType newPc] <- byTypeBB.[pieceType newPc] ||| b
        byTypeBB.[AllPieces] <- byTypeBB.[AllPieces] ||| b
        byColorBB.[pieceColor newPc] <- byColorBB.[pieceColor newPc] ||| b
        board.[sq] <- newPc
        currentKey <- currentKey ^^^ zPiece newPc sq
        if pieceType newPc = Pawn then
            currentPawnKey <- currentPawnKey ^^^ zPiece newPc sq
        else
            currentNonPawnKey <- currentNonPawnKey ^^^ zPiece newPc sq
            if pieceType newPc = Knight || pieceType newPc = Bishop || pieceType newPc = King then
                currentMinorKey <- currentMinorKey ^^^ zPiece newPc sq
            elif pieceType newPc = Rook || pieceType newPc = Queen then
                currentMajorKey <- currentMajorKey ^^^ zPiece newPc sq
        if active then
            dirtyPc.[dirtyN] <- newPc; dirtySq.[dirtyN] <- sq; dirtySign.[dirtyN] <- 1; dirtyN <- dirtyN + 1
            this.UpdatePieceThreats(newPc, true, sq, false, 0UL)

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

    // --- FullThreats NNUE accumulator: public read + lazy materialization ---
    member _.Active: bool = active

    member _.Top: int = top

    member private _.EnsureStorage() =
        if Array.isEmpty accW then
            // 64B-aligned + pinned: frame strides (2048B acc / 32B psqt) keep every frame's vector ops
            // split-free once the base is aligned (psqt's 32B stride lands frames at 0 or 32 mod 64 —
            // both fit a 32B load). Offsets fold into AccOff/PsqOff.
            let struct (aW, aWOff) = Accumulator.allocAligned64<int16> (AccMaxPly * Accumulator.L1)
            let struct (aB, aBOff) = Accumulator.allocAligned64<int16> (AccMaxPly * Accumulator.L1)
            let struct (pW, pWOff) = Accumulator.allocAligned64<int> (AccMaxPly * Accumulator.PsqtBuckets)
            let struct (pB, pBOff) = Accumulator.allocAligned64<int> (AccMaxPly * Accumulator.PsqtBuckets)
            accW <- aW
            accB <- aB
            psqW <- pW
            psqB <- pB
            accBaseW <- aWOff
            accBaseB <- aBOff
            psqBaseW <- pWOff
            psqBaseB <- pBOff
            computedW <- Array.zeroCreate AccMaxPly
            computedB <- Array.zeroCreate AccMaxPly
            // Delta payloads are PER-FRAME (offsets via DirtyOff/ThreatOff) — see the comment on those.
            frameDirtyPc <- Array.zeroCreate (AccMaxPly * Accumulator.MaxDirtyPieces)
            frameDirtySq <- Array.zeroCreate (AccMaxPly * Accumulator.MaxDirtyPieces)
            frameDirtySign <- Array.zeroCreate (AccMaxPly * Accumulator.MaxDirtyPieces)
            frameDirtyN <- Array.zeroCreate AccMaxPly
            frameThreats <- Array.zeroCreate (AccMaxPly * Accumulator.MaxDirtyThreats)
            frameThreatN <- Array.zeroCreate AccMaxPly
            frameChangedW <- Array.zeroCreate (AccMaxPly * Accumulator.MaxDirtyThreats)
            frameChangedB <- Array.zeroCreate (AccMaxPly * Accumulator.MaxDirtyThreats)
            frameChangedNW <- Array.zeroCreate AccMaxPly
            frameChangedNB <- Array.zeroCreate AccMaxPly
            frameChangedValid <- Array.zeroCreate AccMaxPly
            frameChangedKsqW <- Array.create AccMaxPly NoSquare
            frameChangedKsqB <- Array.create AccMaxPly NoSquare
            frameWhiteKingMoved <- Array.zeroCreate AccMaxPly
            frameBlackKingMoved <- Array.zeroCreate AccMaxPly
            frameThreatOverflow <- Array.zeroCreate AccMaxPly
            let struct (fA, fAOff) = Accumulator.allocAligned64<int16> (64 * 2 * Accumulator.L1)
            let struct (fP, fPOff) = Accumulator.allocAligned64<int> (64 * 2 * Accumulator.PsqtBuckets)
            finnyAcc <- fA
            finnyPsqt <- fP
            finnyAccBase <- fAOff
            finnyPsqBase <- fPOff
            // NoPiece <> 0: zeroCreate would claim a white pawn on every square (the Position.fs board-array
            // trap) — ResetFinny fills with NoPiece before first use; allocate only here.
            finnyBoard <- Array.zeroCreate (64 * 2 * 64)

    // Per-color: the aligned bases of accW/accB (independent pinned allocations) generally differ.
    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member private _.AccOff (pColor: Color) (frame: int) =
        (if pColor = White then accBaseW else accBaseB) + frame * Accumulator.L1

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member private _.PsqOff (pColor: Color) (frame: int) =
        (if pColor = White then psqBaseW else psqBaseB) + frame * Accumulator.PsqtBuckets

    // Per-frame offsets into the delta-payload arrays. These MUST be frame-multiplied: the lazy catch-up
    // walk (EnsureComputed/EnsureBothComputedCore) replays SEVERAL frames' payloads, so flattening these to a
    // shared single-frame buffer silently corrupts any >=2-frame walk (the 2026-07-01 audit bug — eager
    // materialization consumed each frame immediately and hid it). Guarded by the multi-frame walk test in
    // NNUETests.
    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member private _.DirtyOff(frame: int) = frame * Accumulator.MaxDirtyPieces

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member private _.ThreatOff(frame: int) = frame * Accumulator.MaxDirtyThreats

    member private this.EnumThreats(pColor: Color, buf: int[]) : int =
        match threatFn with
        | null -> 0
        | f ->
            if PosProf.Enabled then
                let profT0 = System.Diagnostics.Stopwatch.GetTimestamp()
                let n = f.Invoke(this, pColor, buf)
                PosProf.tEnumThreats <- PosProf.tEnumThreats + (System.Diagnostics.Stopwatch.GetTimestamp() - profT0)
                PosProf.nEnumThreats <- PosProf.nEnumThreats + 1L
                n
            else
                f.Invoke(this, pColor, buf)

    member private this.BuildHalf(pColor: Color, frame: int) =
        let acc = if pColor = White then accW else accB
        let psq = if pColor = White then psqW else psqB
        let accOff = this.AccOff pColor frame
        let psqOff = this.PsqOff pColor frame
        let ksq = this.KingSquare pColor

        for j in 0 .. Accumulator.L1 - 1 do
            acc.[accOff + j] <- biases.[j]

        System.Array.Clear(psq, psqOff, Accumulator.PsqtBuckets)

        for sq in 0 .. 63 do
            let pc = board.[sq]

            if pc <> NoPiece then
                Accumulator.addFeatureAt acc accOff psq psqOff halfWeights halfWBase halfPsqt (Accumulator.makeIndex pColor pc sq ksq) 1 Accumulator.UseAvx512 Accumulator.UseAvx2

    member private this.AddActiveThreats(pColor: Color, frame: int) =
        let acc = if pColor = White then accW else accB
        let psq = if pColor = White then psqW else psqB
        let accOff = this.AccOff pColor frame
        let psqOff = this.PsqOff pColor frame
        let n = this.EnumThreats(pColor, tmpW)

        for k in 0 .. n - 1 do
            Accumulator.addThreatAt acc accOff psq psqOff threatWeights threatWBase threatPsqt tmpW.[k] 1 Accumulator.UseAvx512 Accumulator.UseAvx2

    member private this.BuildFull(pColor: Color, frame: int) =
        let profT0 =
            if PosProf.Enabled then System.Diagnostics.Stopwatch.GetTimestamp() else 0L

        this.BuildHalf(pColor, frame)
        this.AddActiveThreats(pColor, frame)
        if pColor = White then computedW.[frame] <- true else computedB.[frame] <- true

        if PosProf.Enabled then
            PosProf.tBuild <- PosProf.tBuild + (System.Diagnostics.Stopwatch.GetTimestamp() - profT0)
            PosProf.nBuild <- PosProf.nBuild + 1L

    // Both perspectives' active threats from a SINGLE physical enumeration (slider rays walked once),
    // emitting per-perspective indices into tmpW/tmpB. Bit-exact vs two AddActiveThreats calls:
    // same indices, and int16 accumulation is order-independent (modular add).
    member private this.AddActiveThreatsBoth(frame: int) =
        match threatFnBoth with
        | null ->
            this.AddActiveThreats(White, frame)
            this.AddActiveThreats(Black, frame)
        | f ->
            let packed =
                if PosProf.Enabled then
                    let enumT0 = System.Diagnostics.Stopwatch.GetTimestamp()
                    let p = f.Invoke(this, tmpW, tmpB)
                    PosProf.tEnumThreats <- PosProf.tEnumThreats + (System.Diagnostics.Stopwatch.GetTimestamp() - enumT0)
                    PosProf.nEnumThreats <- PosProf.nEnumThreats + 1L
                    p
                else
                    f.Invoke(this, tmpW, tmpB)

            let nW = int (packed >>> 32)
            let nB = int (packed &&& 0xFFFFFFFFL)

            for k in 0 .. nW - 1 do
                Accumulator.addThreatAt accW (this.AccOff White frame) psqW (this.PsqOff White frame) threatWeights threatWBase threatPsqt tmpW.[k] 1 Accumulator.UseAvx512 Accumulator.UseAvx2

            for k in 0 .. nB - 1 do
                Accumulator.addThreatAt accB (this.AccOff Black frame) psqB (this.PsqOff Black frame) threatWeights threatWBase threatPsqt tmpB.[k] 1 Accumulator.UseAvx512 Accumulator.UseAvx2

    member private this.BuildFullBoth(frame: int) =
        let profT0 =
            if PosProf.Enabled then System.Diagnostics.Stopwatch.GetTimestamp() else 0L

        this.BuildHalf(White, frame)
        this.BuildHalf(Black, frame)
        this.AddActiveThreatsBoth(frame)
        computedW.[frame] <- true
        computedB.[frame] <- true

        if PosProf.Enabled then
            PosProf.tBuild <- PosProf.tBuild + (System.Diagnostics.Stopwatch.GetTimestamp() - profT0)
            PosProf.nBuild <- PosProf.nBuild + 1L

    member private this.BeginFrame() =
        Debug.Assert((top + 1 < AccMaxPly), "BeginFrame: stack overflow")
        top <- top + 1
        dirtyN <- 0
        dirtyThreatN <- 0
        dirtyThreatOverflow <- false
        frameDirtyN.[top] <- 0
        frameThreatN.[top] <- 0
        frameChangedNW.[top] <- 0
        frameChangedNB.[top] <- 0
        frameChangedValid.[top] <- false
        frameChangedKsqW.[top] <- NoSquare
        frameChangedKsqB.[top] <- NoSquare
        frameWhiteKingMoved.[top] <- false
        frameBlackKingMoved.[top] <- false
        frameThreatOverflow.[top] <- false
        computedW.[top] <- false
        computedB.[top] <- false

    member private this.CommitFrame() =
        let frame = top
        let dOff = this.DirtyOff frame
        let mutable whiteKingMoved = false
        let mutable blackKingMoved = false

        Debug.Assert((dirtyN <= Accumulator.MaxDirtyPieces), "CommitFrame: dirty piece overflow")

        for i in 0 .. dirtyN - 1 do
            let pc = dirtyPc.[i]
            frameDirtyPc.[dOff + i] <- pc
            frameDirtySq.[dOff + i] <- dirtySq.[i]
            frameDirtySign.[dOff + i] <- dirtySign.[i]

            if pieceType pc = King then
                if pieceColor pc = White then whiteKingMoved <- true else blackKingMoved <- true

        frameDirtyN.[frame] <- dirtyN
        frameWhiteKingMoved.[frame] <- whiteKingMoved
        frameBlackKingMoved.[frame] <- blackKingMoved

        if dirtyThreatOverflow then
            frameThreatOverflow.[frame] <- true
            frameThreatN.[frame] <- 0
            frameChangedNW.[frame] <- 0
            frameChangedNB.[frame] <- 0
            frameChangedValid.[frame] <- false
        else
            let threatN = min dirtyThreatN Accumulator.MaxDirtyThreats
            frameThreatN.[frame] <- threatN
            System.Array.Copy(dirtyThreats, 0, frameThreats, this.ThreatOff frame, threatN)

            if PosProf.Enabled && int64 threatN > PosProf.maxThreatN then
                PosProf.maxThreatN <- int64 threatN

            if threatN <> 0 then
                frameChangedValid.[frame] <- false
            else
                frameChangedNW.[frame] <- 0
                frameChangedNB.[frame] <- 0
                frameChangedValid.[frame] <- true

    member private _.FrameNeedsRefresh(pColor: Color, frame: int) : bool =
        frameThreatOverflow.[frame]
        || (pColor = White && frameWhiteKingMoved.[frame])
        || (pColor = Black && frameBlackKingMoved.[frame])

    /// Eagerly materialize the current frame's accumulator during Make: copy the parent (top-1) into top,
    /// then apply this frame's dirty piece + threat deltas immediately. After this, computedW/B.[top] are
    /// true and eval is O(1) — no lazy frame walk, no cache probe, no deferred delegate conversion at eval time.
    /// King moves or threat overflow force a full from-scratch rebuild (all HalfKA indices change).
    member private this.EagerUpdate() =
        let profT0 =
            if PosProf.Enabled then System.Diagnostics.Stopwatch.GetTimestamp() else 0L

        let frame = top

        if frameThreatOverflow.[frame] || frameWhiteKingMoved.[frame] || frameBlackKingMoved.[frame] then
            if UseFinny then
                this.RefreshFromFinnyBoth(frame)
            else
                this.BuildFullBoth(frame)
        else
            let ksqW = this.KingSquare White
            let ksqB = this.KingSquare Black
            this.EnsureChangedThreats(frame, ksqW, ksqB)
            this.ApplyFrameFusedTo(White, frame, frame - 1, frame)
            this.ApplyFrameFusedTo(Black, frame, frame - 1, frame)
            computedW.[frame] <- true
            computedB.[frame] <- true

        if PosProf.Enabled then
            PosProf.tEager <- PosProf.tEager + (System.Diagnostics.Stopwatch.GetTimestamp() - profT0)

    /// Ensure `frame`'s physical threat edges are converted into per-perspective SIGNED index lists
    /// (`frameChangedW/B` at ThreatOff frame; +(idx+1) = add, -(idx+1) = remove) valid for the CURRENT king
    /// squares. Cached per frame; re-run when either king square changed since caching (threat orientation
    /// depends on the own-king file mirror). No-op when the frame has no threat edges or no conversion
    /// delegate is bound (frameChangedValid stays false and the fused apply skips threats -- the pre-fused
    /// paths behaved identically).
    member private this.EnsureChangedThreats(frame: int, ksqW: Square, ksqB: Square) =
        let threatN = frameThreatN.[frame]

        if
            threatN <> 0
            && not (
                frameChangedValid.[frame]
                && frameChangedKsqW.[frame] = ksqW
                && frameChangedKsqB.[frame] = ksqB
            )
        then
            match threatFnChangedBoth with
            | null -> ()
            | f ->
                let profT0 =
                    if PosProf.Enabled then System.Diagnostics.Stopwatch.GetTimestamp() else 0L

                let off = this.ThreatOff frame
                let packed = f.Invoke(this, frameThreats, off, threatN, changedW, changedB)
                let nW = int (packed >>> 32)
                let nB = int (packed &&& 0xFFFFFFFFL)
                System.Array.Copy(changedW, 0, frameChangedW, off, nW)
                System.Array.Copy(changedB, 0, frameChangedB, off, nB)
                frameChangedNW.[frame] <- nW
                frameChangedNB.[frame] <- nB
                frameChangedKsqW.[frame] <- ksqW
                frameChangedKsqB.[frame] <- ksqB
                frameChangedValid.[frame] <- true

                if PosProf.Enabled then
                    PosProf.tGather <- PosProf.tGather + (System.Diagnostics.Stopwatch.GetTimestamp() - profT0)
                    PosProf.nGather <- PosProf.nGather + 1L

    /// Fused replay of ONE frame's deltas for one perspective: gather the frame's HalfKA dirty pieces and
    /// pre-converted signed threat rows (see EnsureChangedThreats -- callers run it first) into add/sub index
    /// lists, then apply them in a single register-tiled pass reading the accumulator at `srcFrame` and
    /// writing `dstFrame` (parent -> child materialization; the old sub/add pairing kernels are subsumed --
    /// the fused kernel folds every row into one accumulator sweep regardless of pairing).
    /// PRE for the HalfKA index math: no own-king move inside `frame` (walk callers stop at refresh barriers).
    member private this.ApplyFrameFusedTo(pColor: Color, frame: int, srcFrame: int, dstFrame: int) =
        let profT0 =
            if PosProf.Enabled then System.Diagnostics.Stopwatch.GetTimestamp() else 0L

        let acc = if pColor = White then accW else accB
        let psq = if pColor = White then psqW else psqB
        let ksq = this.KingSquare pColor
        let dOff = this.DirtyOff frame
        let dirtyN = frameDirtyN.[frame]
        let mutable nHalfAdd = 0
        let mutable nHalfSub = 0

        for i in 0 .. dirtyN - 1 do
            let idx =
                Accumulator.makeIndex pColor frameDirtyPc.[dOff + i] frameDirtySq.[dOff + i] ksq

            if frameDirtySign.[dOff + i] > 0 then
                fusedHalfAdd.[nHalfAdd] <- idx
                nHalfAdd <- nHalfAdd + 1
            else
                fusedHalfSub.[nHalfSub] <- idx
                nHalfSub <- nHalfSub + 1

        let mutable nThrAdd = 0
        let mutable nThrSub = 0

        if frameThreatN.[frame] <> 0 && frameChangedValid.[frame] then
            let off = this.ThreatOff frame
            let lst = if pColor = White then frameChangedW else frameChangedB
            let n = if pColor = White then frameChangedNW.[frame] else frameChangedNB.[frame]

            for i in 0 .. n - 1 do
                let v = lst.[off + i]

                if v > 0 then
                    fusedThrAdd.[nThrAdd] <- v - 1
                    nThrAdd <- nThrAdd + 1
                else
                    fusedThrSub.[nThrSub] <- -v - 1
                    nThrSub <- nThrSub + 1

        Accumulator.applyFused
            acc
            (this.AccOff pColor srcFrame)
            acc
            (this.AccOff pColor dstFrame)
            psq
            (this.PsqOff pColor srcFrame)
            psq
            (this.PsqOff pColor dstFrame)
            halfWeights
            halfWBase
            halfPsqt
            threatWeights
            threatWBase
            threatPsqt
            fusedHalfAdd
            nHalfAdd
            fusedHalfSub
            nHalfSub
            fusedThrAdd
            nThrAdd
            fusedThrSub
            nThrSub
            Accumulator.UseAvx512
            Accumulator.UseAvx2

        if PosProf.Enabled then
            PosProf.tApply <- PosProf.tApply + (System.Diagnostics.Stopwatch.GetTimestamp() - profT0)
            PosProf.nApply <- PosProf.nApply + 1L

    /// Re-seed every finny entry to the empty-board state (acc = biases, board = NoPiece). Called by
    /// EnableNNUE only — entries are otherwise self-correcting via the board diff and survive unmakes and
    /// whole searches (path independence: HalfKA add/sub are exact modular inverses).
    member private _.ResetFinny() =
        for e in 0 .. 127 do
            let aOff = finnyAccBase + e * Accumulator.L1

            for j in 0 .. Accumulator.L1 - 1 do
                finnyAcc.[aOff + j] <- biases.[j]

        System.Array.Clear(finnyPsqt, finnyPsqBase, 64 * 2 * Accumulator.PsqtBuckets)
        System.Array.Fill(finnyBoard, NoPiece)

    /// Finny pass 1: diff the entry for `pColor`'s CURRENT king square against the live board, fold the
    /// changed HalfKA rows into the ENTRY accumulator in place, and update the board snapshot.
    /// Returns the entry index.
    member private this.RefreshFinnyEntry(pColor: Color) : int =
        let ksq = this.KingSquare pColor
        let e = ksq * 2 + int pColor
        let eAcc = finnyAccBase + e * Accumulator.L1
        let ePsq = finnyPsqBase + e * Accumulator.PsqtBuckets
        let eB = e * 64
        let mutable nAdd = 0
        let mutable nSub = 0

        for sq in 0 .. 63 do
            let cur = board.[sq]
            let old = finnyBoard.[eB + sq]

            if cur <> old then
                if old <> NoPiece then
                    fusedHalfSub.[nSub] <- Accumulator.makeIndex pColor old sq ksq
                    nSub <- nSub + 1

                if cur <> NoPiece then
                    fusedHalfAdd.[nAdd] <- Accumulator.makeIndex pColor cur sq ksq
                    nAdd <- nAdd + 1

                finnyBoard.[eB + sq] <- cur

        Debug.Assert((nAdd <= 32 && nSub <= 32), "RefreshFinnyEntry: diff exceeds board piece bound")

        if nAdd + nSub > 0 then
            Accumulator.applyFused
                finnyAcc
                eAcc
                finnyAcc
                eAcc
                finnyPsqt
                ePsq
                finnyPsqt
                ePsq
                halfWeights
                halfWBase
                halfPsqt
                threatWeights
                threatWBase
                threatPsqt
                fusedHalfAdd
                nAdd
                fusedHalfSub
                nSub
                fusedThrAdd
                0
                fusedThrSub
                0
                Accumulator.UseAvx512
                Accumulator.UseAvx2

        e

    /// Finny pass 2: materialize `frame` for `pColor` as ENTRY + active threat rows in one fused pass
    /// (the entry->frame copy IS the first threat pass; nThr = 0 degenerates to a pure copy).
    member private this.FinnyFoldToFrame(pColor: Color, e: int, frame: int, thrList: int[], nThr: int) =
        let acc = if pColor = White then accW else accB
        let psq = if pColor = White then psqW else psqB

        Accumulator.applyFused
            finnyAcc
            (finnyAccBase + e * Accumulator.L1)
            acc
            (this.AccOff pColor frame)
            finnyPsqt
            (finnyPsqBase + e * Accumulator.PsqtBuckets)
            psq
            (this.PsqOff pColor frame)
            halfWeights
            halfWBase
            halfPsqt
            threatWeights
            threatWBase
            threatPsqt
            fusedHalfAdd
            0
            fusedHalfSub
            0
            thrList
            nThr
            fusedThrSub
            0
            Accumulator.UseAvx512
            Accumulator.UseAvx2

        if pColor = White then
            computedW.[frame] <- true
        else
            computedB.[frame] <- true

    /// Finny-table refresh: replaces the from-scratch BuildFull at refresh barriers (own-king move /
    /// threat overflow). Cost = 64-square diff + a few HalfKA rows into the entry + one entry->frame fold
    /// with the active threats, instead of bias-init + ~30 HalfKA rows + all threat rows from zero.
    member private this.RefreshFromFinny(pColor: Color, frame: int) =
        let profT0 =
            if PosProf.Enabled then System.Diagnostics.Stopwatch.GetTimestamp() else 0L

        let e = this.RefreshFinnyEntry pColor
        let buf = if pColor = White then tmpW else tmpB
        let nThr = this.EnumThreats(pColor, buf)
        this.FinnyFoldToFrame(pColor, e, frame, buf, nThr)

        if PosProf.Enabled then
            PosProf.tBuild <- PosProf.tBuild + (System.Diagnostics.Stopwatch.GetTimestamp() - profT0)
            PosProf.nBuild <- PosProf.nBuild + 1L

    /// Both-perspective finny refresh: one shared physical threat enumeration (like BuildFullBoth).
    member private this.RefreshFromFinnyBoth(frame: int) =
        match threatFnBoth with
        | null ->
            this.RefreshFromFinny(White, frame)
            this.RefreshFromFinny(Black, frame)
        | fb ->
            let profT0 =
                if PosProf.Enabled then System.Diagnostics.Stopwatch.GetTimestamp() else 0L

            let eW = this.RefreshFinnyEntry White
            let eB = this.RefreshFinnyEntry Black

            let packed =
                if PosProf.Enabled then
                    let enumT0 = System.Diagnostics.Stopwatch.GetTimestamp()
                    let p = fb.Invoke(this, tmpW, tmpB)
                    PosProf.tEnumThreats <- PosProf.tEnumThreats + (System.Diagnostics.Stopwatch.GetTimestamp() - enumT0)
                    PosProf.nEnumThreats <- PosProf.nEnumThreats + 1L
                    p
                else
                    fb.Invoke(this, tmpW, tmpB)

            let nW = int (packed >>> 32)
            let nB = int (packed &&& 0xFFFFFFFFL)
            this.FinnyFoldToFrame(White, eW, frame, tmpW, nW)
            this.FinnyFoldToFrame(Black, eB, frame, tmpB, nB)

            if PosProf.Enabled then
                PosProf.tBuild <- PosProf.tBuild + (System.Diagnostics.Stopwatch.GetTimestamp() - profT0)
                PosProf.nBuild <- PosProf.nBuild + 1L

    member private this.EnsureComputed(pColor: Color) =
        if active then
            let computed = if pColor = White then computedW else computedB

            if not computed.[top] then
                let mutable baseFrame = top
                let mutable blocked = false

                while baseFrame > 0 && not blocked && not computed.[baseFrame] do
                    if this.FrameNeedsRefresh(pColor, baseFrame) then
                        blocked <- true
                    else
                        baseFrame <- baseFrame - 1

                if blocked || not computed.[baseFrame] then
                    if UseFinny then
                        this.RefreshFromFinny(pColor, top)
                    else
                        this.BuildFull(pColor, top)
                else
                    // Fused walk: materialize EVERY frame from its parent -- same store traffic as the old
                    // apply-onto-top (one 2 KB store per walked frame), but sibling evals now reuse the
                    // materialized ancestors instead of re-walking them.
                    let ksqW = this.KingSquare White
                    let ksqB = this.KingSquare Black

                    for f in baseFrame + 1 .. top do
                        this.EnsureChangedThreats(f, ksqW, ksqB)
                        this.ApplyFrameFusedTo(pColor, f, f - 1, f)
                        computed.[f] <- true

    // Materialize BOTH perspectives at top in one frame walk (evalInternal always needs both). Takes the
    // merged path only when the two perspectives' back-walks agree (the common no-king-move/no-overflow
    // case); falls back to the per-perspective EnsureComputed otherwise (byte-identical to that path).
    member this.EnsureBothComputed() =
        if active && not (computedW.[top] && computedB.[top]) then
            let profT0 =
                if PosProf.Enabled then System.Diagnostics.Stopwatch.GetTimestamp() else 0L

            // Phase 1 fast-path: best-effort checkpoint cache. A validated hit pays an O(1) snapshot copy
            // instead of the O(distance) frame-delta walk below. Stored snapshots are bit-exact for any given
            // position regardless of the make/unmake path that reached it, so a hit is provably equivalent
            // to re-running the lazy walk.
            let accOffW = this.AccOff White top
            let accOffB = this.AccOff Black top
            let psqOffW = this.PsqOff White top
            let psqOffB = this.PsqOff Black top

            let cached =
                match checkpoint with
                | null -> false
                | cache ->
                    cache.TryProbe(this.Key, accW, accOffW, accB, accOffB, psqW, psqOffW, psqB, psqOffB)

            if cached then
                computedW.[top] <- true
                computedB.[top] <- true
            else
                this.EnsureBothComputedCore()

                // Best-effort populate. Checking both flags post-materialization guarantees we never cache a
                // partial snapshot, even on the mixed-rebuild branch + the per-perspective fallback path.
                if computedW.[top] && computedB.[top] then
                    match checkpoint with
                    | null -> ()
                    | cache ->
                        cache.Store(this.Key, accW, accOffW, accB, accOffB, psqW, psqOffW, psqB, psqOffB)

            if PosProf.Enabled then
                PosProf.tEnsure <- PosProf.tEnsure + (System.Diagnostics.Stopwatch.GetTimestamp() - profT0)
                PosProf.nEnsure <- PosProf.nEnsure + 1L

    /// EONEGO_PROF support: data-start address mod 64 for every hot accumulator/weight array (the 64B-
    /// alignment audit premise check). All targets are LOH-resident, so a momentary pin reads a stable
    /// address. Frame strides are 64B multiples (L1*2 = 2048B), so the base residue governs every frame.
    member _.ProfAlignReport() : string =
        // byteOff = the aligned base offset in BYTES (element offset * element size): the report shows the
        // residue of the EFFECTIVE data start, so post-alignment every entry should read 0.
        let m (name: string) (o: System.Array) (byteOff: int) =
            if isNull (box o) then
                name + "=?"
            else
                let h =
                    System.Runtime.InteropServices.GCHandle.Alloc(o, System.Runtime.InteropServices.GCHandleType.Pinned)

                let v = int ((h.AddrOfPinnedObject().ToInt64() + int64 byteOff) % 64L)
                h.Free()
                name + "=" + string v

        m "accW" accW (accBaseW * 2)
        + " " + m "accB" accB (accBaseB * 2)
        + " " + m "psqW" psqW (psqBaseW * 4)
        + " " + m "psqB" psqB (psqBaseB * 4)
        + " " + m "finnyAcc" finnyAcc (finnyAccBase * 2)
        + " " + m "finnyPsqt" finnyPsqt (finnyPsqBase * 4)
        + " " + m "halfW" halfWeights (halfWBase * 2)
        + " " + m "halfPsqt" halfPsqt 0
        + " " + m "thrW" threatWeights threatWBase
        + " " + m "thrPsqt" threatPsqt 0

    /// Bind the per-worker checkpoint cache. Pass `null` to disable (tests, from-scratch eval, etc.).
    /// `SearchControl` owns the table lifecycle; this Position merely holds a borrowed reference for the
    /// duration of a search.
    member _.BindCheckpoint(cache: AccCheckpointTable) : unit = checkpoint <- cache

    /// Detach the cache (no-op if already detached). Called by `SearchControl` once the search has joined to
    /// release the worker's borrowed reference; the position can continue to be reused by tests/tools.
    member _.UnbindCheckpoint() : unit = checkpoint <- null

    /// TEST HOOK: toggle eager materialization after `EnableNNUE` (which sets the production default).
    /// With `false`, Make records dirty frames only and evaluation pays the lazy multi-frame catch-up walk —
    /// the path the guardrail tests exercise. Production call sites never touch this.
    member internal _.SetEagerUpdates(v: bool) : unit = eagerUpdates <- v

    /// Unconditionally publish the current frame's computed accumulator snapshot to the bound checkpoint
    /// cache, if any. Used by `Worker.SetupRoot` to seed the root after `EnableNNUE` has already set the
    /// `computed` flags (so the early-return path inside `EnsureBothComputed` skips the populate).
    /// No-op when the accumulator is inactive, the current frame is not yet materialized, or no cache is bound.
    member this.SeedCheckpoint() : unit =
        if active && computedW.[top] && computedB.[top] then
            match checkpoint with
            | null -> ()
            | cache ->
                cache.Store(
                    this.Key,
                    accW,
                    this.AccOff White top,
                    accB,
                    this.AccOff Black top,
                    psqW,
                    this.PsqOff White top,
                    psqB,
                    this.PsqOff Black top
                )

    /// Phase 1 — the unchanged frame-walk materialization used when the checkpoint cache misses (or is
    /// null). Byte-for-byte identical to the pre-Phase-1 `EnsureBothComputed`; retained verbatim so that
    /// benchmarks + parity tests can isolate Phase 1's perf contribution empirically (toggle the UCI option
    /// `EnableAccCheckpoint` off — Phase 1 Step 5 — to route all calls through this core).
    member private this.EnsureBothComputedCore() =
        if active && not (computedW.[top] && computedB.[top]) then
            let mutable baseW = top
            let mutable blockedW = false

            while baseW > 0 && not blockedW && not computedW.[baseW] do
                if this.FrameNeedsRefresh(White, baseW) then blockedW <- true
                else baseW <- baseW - 1

            let mutable baseB = top
            let mutable blockedB = false

            while baseB > 0 && not blockedB && not computedB.[baseB] do
                if this.FrameNeedsRefresh(Black, baseB) then blockedB <- true
                else baseB <- baseB - 1

            let rebuildW = blockedW || not computedW.[baseW]
            let rebuildB = blockedB || not computedB.[baseB]

            if not rebuildW && not rebuildB && baseW = baseB then
                let ksqW = this.KingSquare White
                let ksqB = this.KingSquare Black

                for f in baseW + 1 .. top do
                    this.EnsureChangedThreats(f, ksqW, ksqB)
                    this.ApplyFrameFusedTo(White, f, f - 1, f)
                    this.ApplyFrameFusedTo(Black, f, f - 1, f)
                    computedW.[f] <- true
                    computedB.[f] <- true
            elif rebuildW && rebuildB then
                if UseFinny then
                    this.RefreshFromFinnyBoth(top)
                else
                    this.BuildFullBoth(top)
            else
                this.EnsureComputed White
                this.EnsureComputed Black

    /// Merged accumulator (biases + HalfKA + threats already summed) for a perspective, into caller spans.
    member this.ReadAccInto(pColor: Color, acc: System.Span<int16>, psqt: System.Span<int>) =
        this.EnsureComputed pColor
        let m = if pColor = White then accW else accB
        let mp = if pColor = White then psqW else psqB
        System.Span<int16>(m, this.AccOff pColor top, Accumulator.L1).CopyTo(acc)
        System.Span<int>(mp, this.PsqOff pColor top, Accumulator.PsqtBuckets).CopyTo(psqt)

    member this.AccSpan(pColor: Color) : System.Span<int16> =
        this.EnsureComputed pColor
        let m = if pColor = White then accW else accB
        System.Span<int16>(m, this.AccOff pColor top, Accumulator.L1)

    member this.PsqtSpan(pColor: Color) : System.Span<int> =
        this.EnsureComputed pColor
        let mp = if pColor = White then psqW else psqB
        System.Span<int>(mp, this.PsqOff pColor top, Accumulator.PsqtBuckets)

    member this.AccSpanComputed(pColor: Color) : System.Span<int16> =
        let m = if pColor = White then accW else accB
        System.Span<int16>(m, this.AccOff pColor top, Accumulator.L1)

    member this.PsqtSpanComputed(pColor: Color) : System.Span<int> =
        let mp = if pColor = White then psqW else psqB
        System.Span<int>(mp, this.PsqOff pColor top, Accumulator.PsqtBuckets)

    /// Compatibility escape hatch for tests/probes that still want an array. Prefer AccSpan in hot code.
    member this.AccArray(pColor: Color) : int16[] =
        this.EnsureComputed pColor
        let src = if pColor = White then accW else accB
        src.[this.AccOff pColor top .. this.AccOff pColor top + Accumulator.L1 - 1]

    member this.PsqtArray(pColor: Color) : int[] =
        this.EnsureComputed pColor
        let src = if pColor = White then psqW else psqB
        src.[this.PsqOff pColor top .. this.PsqOff pColor top + Accumulator.PsqtBuckets - 1]

    /// Bind weights + threat enumerators + materialize root. ROOT ONLY.
    member this.EnableNNUE
        (biasesIn: int16[])
        (halfWeightsIn: int16[])
        (halfWOffIn: int)
        (halfPsqtIn: int[])
        (threatWeightsIn: sbyte[])
        (threatWOffIn: int)
        (threatPsqtIn: int[])
        (threatFnIn: System.Func<Position, int, int[], int>)
        (threatFnBothIn: System.Func<Position, int[], int[], int64>)
        (threatFnChangedBothIn: System.Func<Position, int[], int, int, int[], int[], int64>)
        =
        // Callable at ANY stPly: it rebases the accumulator stack (top <- 0, frame 0 rebuilt from the
        // CURRENT board), which is exactly what Worker.SetupRoot needs after replaying a long game
        // history. The undo stack (states/stPly) is untouched — repetition detection keeps its keys.
        this.EnsureStorage()
        biases <- biasesIn
        halfWeights <- halfWeightsIn
        halfWBase <- halfWOffIn
        halfPsqt <- halfPsqtIn
        threatWeights <- threatWeightsIn
        threatWBase <- threatWOffIn
        threatPsqt <- threatPsqtIn
        threatFn <- threatFnIn
        threatFnBoth <- threatFnBothIn
        threatFnChangedBoth <- threatFnChangedBothIn
        top <- 0
        dirtyN <- 0
        dirtyThreatN <- 0
        dirtyThreatOverflow <- false
        System.Array.Clear(computedW, 0, computedW.Length)
        System.Array.Clear(computedB, 0, computedB.Length)
        frameDirtyN.[0] <- 0
        frameThreatN.[0] <- 0
        frameChangedNW.[0] <- 0
        frameChangedNB.[0] <- 0
        frameChangedValid.[0] <- false
        frameChangedKsqW.[0] <- NoSquare
        frameChangedKsqB.[0] <- NoSquare
        frameWhiteKingMoved.[0] <- false
        frameBlackKingMoved.[0] <- false
        frameThreatOverflow.[0] <- false
        active <- true
        // LAZY is the production default (audit 2026-07-01): 27.8% of makes are never evaluated, and the lazy
        // refresh is per-perspective — measured ~+10% nps over eager materialization on bit-identical trees.
        // Tests flip this via SetEagerUpdates to cover the eager machinery.
        eagerUpdates <- false
        this.ResetFinny()
        this.BuildFull(White, 0)
        this.BuildFull(Black, 0)

    /// Test/debug hook: collect the physical dirty FullThreats edges that `Make m` would record, without
    /// requiring NNUE weights. Returns -1 if the frame hit the overflow fallback.
    member this.DebugCollectDirtyThreats(m: Move, dst: int[]) : int =
        let wasActive = active
        let savedTop = top
        this.EnsureStorage()

        if not wasActive then
            active <- true
            top <- 0
            dirtyN <- 0

        this.Make m
        let frame = top
        let n = frameThreatN.[frame]
        let overflow = dirtyThreatOverflow

        if not overflow then
            System.Array.Copy(frameThreats, this.ThreatOff frame, dst, 0, min n dst.Length)

        this.Unmake m

        if not wasActive then
            active <- false
            top <- savedTop
            dirtyN <- 0

        if overflow then -1 else min n dst.Length

    // --- StateInfo scalar accessors (byref-local read avoids the ~120 B struct copy; the getter itself
    //     is trivial and JIT-inlined — F# forbids MethodImpl on a parameterless property) -------------
    member _.EpSquare: Square = let st = &states.[stPly] in st.EpSquare
    member _.CastlingRights: int = let st = &states.[stPly] in st.CastlingRights
    member _.Rule50: int = let st = &states.[stPly] in st.Rule50
    // Plies since the last null move (search-only; bounds the null-safe repetition walk). Mirrors Rule50.
    member _.PliesFromNull: int = let st = &states.[stPly] in st.PliesFromNull
    member _.Key: uint64 = let st = &states.[stPly] in st.Key
    member _.PawnKey: uint64 = let st = &states.[stPly] in st.PawnKey
    member _.MinorKey: uint64 = let st = &states.[stPly] in st.MinorKey
    member _.MajorKey: uint64 = let st = &states.[stPly] in st.MajorKey
    member _.NonPawnKey: uint64 = let st = &states.[stPly] in st.NonPawnKey
    member _.StPly: int = stPly

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

    // --- SEE: static-exchange evaluation >= threshold ----------
    // v1 simplifications (see plan D5): non-NORMAL m short-circuits to (0 >= threshold) — exactly the reference's
    // own early-out (promotion treated via that path); and the swap loop is NOT pin-aware (the
    // KING-terminate rule already covers the dominant illegal-recapture case; Pinners/BlockersForKing are
    // available for a v2 refinement). SEE drives only pruning/ordering, never legality. The control flow
    // (swap accumulator, res toggle, `swap < res` break, KING terminate, return bool(res)) mirrors the reference
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

    // --- slider blockers / pinners -----------
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

    // --- check-info: cache checkers / blockers / pinners / check squares --
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
        | _ -> 0UL // King never gives check (checkSquares[KING] = 0)

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

    /// From-scratch pawn-key oracle (piece-square terms of both colors' pawns only).
    member _.RecomputePawnKey() : uint64 =
        let mutable key = 0UL
        let mutable occ = byTypeBB.[Pawn]

        while occ <> 0UL do
            let sq = popLsb &occ
            key <- key ^^^ zPiece board.[sq] sq

        key

    /// From-scratch minor-key oracle (piece-square terms of both colors' knights, bishops and kings).
    member _.RecomputeMinorKey() : uint64 =
        let mutable key = 0UL
        let mutable occ = byTypeBB.[Knight] ||| byTypeBB.[Bishop] ||| byTypeBB.[King]

        while occ <> 0UL do
            let sq = popLsb &occ
            key <- key ^^^ zPiece board.[sq] sq

        key

    /// From-scratch major-key oracle (piece-square terms of both colors' rooks and queens).
    member _.RecomputeMajorKey() : uint64 =
        let mutable key = 0UL
        let mutable occ = byTypeBB.[Rook] ||| byTypeBB.[Queen]

        while occ <> 0UL do
            let sq = popLsb &occ
            key <- key ^^^ zPiece board.[sq] sq

        key

    /// From-scratch non-pawn-key oracle (piece-square terms of all non-pawn pieces, both colors).
    member _.RecomputeNonPawnKey() : uint64 =
        let mutable key = 0UL
        let mutable occ = byTypeBB.[AllPieces] &&& ~~~byTypeBB.[Pawn]

        while occ <> 0UL do
            let sq = popLsb &occ
            key <- key ^^^ zPiece board.[sq] sq

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
        currentPawnKey <- 0UL
        currentMinorKey <- 0UL
        currentMajorKey <- 0UL
        currentNonPawnKey <- 0UL
        stPly <- 0
        gamePly <- 0
        // A bulk board load invalidates the incremental accumulator: disable it so the piece-placement
        // below records NO deltas (32 PutPiece calls would overflow the small dirty buffer). The caller
        // re-enables via EnableNNUE, which rebuilds both perspectives from scratch (Refresh).
        active <- false
        top <- 0
        dirtyN <- 0
        dirtyThreatN <- 0
        dirtyThreatOverflow <- false
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
        // 4. en-passant target (kept only if a real capturer exists — the gate)
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
        st.PawnKey <- currentPawnKey
        st.MinorKey <- currentMinorKey
        st.MajorKey <- currentMajorKey
        st.NonPawnKey <- currentNonPawnKey
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
        Debug.Assert((this.RecomputePawnKey() = currentPawnKey), "LoadFen: incremental pawn key != from-scratch")
        Debug.Assert((this.RecomputeMinorKey() = currentMinorKey), "LoadFen: incremental minor key != from-scratch")
        Debug.Assert((this.RecomputeMajorKey() = currentMajorKey), "LoadFen: incremental major key != from-scratch")
        Debug.Assert((this.RecomputeNonPawnKey() = currentNonPawnKey), "LoadFen: incremental non-pawn key != from-scratch")

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
        let profT0 =
            if PosProf.Enabled then System.Diagnostics.Stopwatch.GetTimestamp() else 0L

        Debug.Assert((stPly + 1 < MaxPly), "Make: stack overflow")
        let us = sideToMove
        let them = flipColor us
        let from = fromSq m
        let dst = toSq m
        let pc = board.[from]
        // 1. seed key from the parent frame, clear stale ep term, advance the stack
        let prev = &states.[stPly]
        currentKey <- prev.Key ^^^ Side
        currentPawnKey <- prev.PawnKey // no side/ep/castle terms; the choke points do the rest
        currentMinorKey <- prev.MinorKey
        currentMajorKey <- prev.MajorKey
        currentNonPawnKey <- prev.NonPawnKey

        if prev.EpSquare <> NoSquare then
            currentKey <- currentKey ^^^ zEp (fileOf prev.EpSquare)

        let pCastling = prev.CastlingRights
        let pRule50 = prev.Rule50
        let pPlies = prev.PliesFromNull
        // NNUE: push a lazy dirty frame for real moves only. Null moves intentionally do not affect top.
        if active then
            this.BeginFrame()
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
        st.PawnKey <- currentPawnKey
        st.MinorKey <- currentMinorKey
        st.MajorKey <- currentMajorKey
        st.NonPawnKey <- currentNonPawnKey
        this.SetCheckInfo()
        // NNUE: persist dirty pieces + physical FullThreats deltas, then eagerly materialize the child
        // accumulator (copy parent + apply deltas). Eval becomes O(1) — no lazy frame walk at eval time.
        if active then
            this.CommitFrame()
            if eagerUpdates then this.EagerUpdate()

        if PosProf.Enabled then
            PosProf.tMake <- PosProf.tMake + (System.Diagnostics.Stopwatch.GetTimestamp() - profT0)
            PosProf.nMake <- PosProf.nMake + 1L

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
        currentPawnKey <- (let p = &states.[stPly] in p.PawnKey)
        currentMinorKey <- (let p = &states.[stPly] in p.MinorKey)
        currentMajorKey <- (let p = &states.[stPly] in p.MajorKey)
        currentNonPawnKey <- (let p = &states.[stPly] in p.NonPawnKey)
        // NNUE: pop the lazy frame. Parent materialization, if needed, is computed on demand.
        if active then
            Debug.Assert((top > 0), "Unmake: top underflow")
            top <- top - 1

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
        // NNUE: a null move moves no pieces ⇒ the accumulator is identical ⇒ no snapshot/update needed.
        if active then dirtyN <- 0
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
        st.PawnKey <- currentPawnKey // null move touches no pawns
        st.MinorKey <- currentMinorKey // ... nor any minors/kings
        st.MajorKey <- currentMajorKey
        st.NonPawnKey <- currentNonPawnKey
        this.SetCheckInfo()

    member this.UnmakeNull() : unit =
        Debug.Assert((stPly > 0), "UnmakeNull: stack underflow")
        stPly <- stPly - 1
        gamePly <- gamePly - 1
        sideToMove <- flipColor sideToMove
        currentKey <- (let p = &states.[stPly] in p.Key)
        currentPawnKey <- (let p = &states.[stPly] in p.PawnKey)
        currentMinorKey <- (let p = &states.[stPly] in p.MinorKey)
        currentMajorKey <- (let p = &states.[stPly] in p.MajorKey)
        currentNonPawnKey <- (let p = &states.[stPly] in p.NonPawnKey)
        // NNUE: a null move left the accumulator unchanged ⇒ nothing to restore.

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
            | _ -> // EnPassant: direct check from the landing square, plus the occupancy recompute.
                // AttackersTo's leaper terms read the LIVE bitboards (the mover is still on `from`
                // there; occ' only affects sliders), so the pawn's own check from `dst` is invisible
                // to it — test it explicitly, same CheckSquares idiom as the non-special arm. The
                // stale `from` square cannot false-positive (a pawn already attacking the enemy king
                // on our move would be an illegal position), and the captured pawn is filtered by
                // the &&& byColorBB.[us].
                let capSq = dst + (if us = White then -8 else 8)
                let occ' = (occ ^^^ (1UL <<< from) ^^^ (1UL <<< capSq)) ||| (1UL <<< dst)

                testBit (this.CheckSquares Pawn) dst
                || (this.AttackersTo theirKing occ' &&& byColorBB.[us]) <> 0UL

    // --- in-check resolution shared by IsPseudoLegal's NORMAL/Promotion arms -----------------------
    // king=true  -> destination must be safe with the king REMOVED from occ (X-ray through the vacated
    //               square) — this is the one piece of legality IsPseudoLegal does (king-into-check test).
    // king=false -> under single check the move must land on between(ksq,checker)|checkers (capture/interpose);
    //               under double check a non-king move is illegal. Pin legality for non-king pieces is left
    //               to isLegal (exactly as the reference). PRE: the move already passed its shape check.
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

    // --- IsPseudoLegal: is `m` playable in THIS position by sideToMove? --
    // Used by the MovePick to emit a TT/killer/counter move WITHOUT generating it (Make assumes
    // legality). Fully INLINE — it MUST NOT call MoveGeneration.generate (Position compiles first); the
    // three special-move validators are hand-rolled from Position's own accessors (mirroring tryCastle /
    // genPawnMoves / the evasion target so that IsPseudoLegal == membership in the generated set). Pin
    // legality for non-king pieces is deferred to isLegal, exactly as the reference.
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

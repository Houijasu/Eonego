/// Eonego — staged, allocation-free move generation for the chess engine.
///
/// Phase 1 delivers the PERFT GATE: a Stockfish-style pseudo-legal generator plus a cheap legality
/// filter (`isLegal`) that consumes the check/pin state Position already caches in `set_check_info`
/// (Checkers, BlockersForKing, Pinners, CheckSquares). Node counts are validated against the published
/// reference values for the standard test positions before any search is layered on.
///
/// LazySMP / lockless / atomic by construction: there is NO `let mutable` at module scope. Every per-call
/// move buffer is a caller-owned `stackalloc Span<Move>` living on the calling thread's stack; the only
/// shared state is Bitboard.fs's attack tables, immutable after static init (safe lock-free reads). A
/// module-level scratch `Move[]` "to avoid stackalloc" is forbidden — it would force a lock.
///
/// Buffer contract: generators write a bare `Span<Move>` (4-byte ints) and thread the write index as a
/// `byref<int>`. `Span<Move>`/`byref<int>` are by-ref-like — never capture them in a closure (heap
/// capture fails to compile), and a function may never RETURN a Span over its own stackalloc (the stack
/// memory dies on return) — the buffer is created in the frame that consumes it.
///
/// GenType: Captures/Quiets/Evasions/NonEvasions feed `generate`; `generateLegal` is the Phase-1 entry
/// (Evasions when in check, else NonEvasions, then compacted by `isLegal`). QuietChecks is reserved for
/// the Phase-2 MovePick; the Captures/Quiets split of pawn promotions is likewise a Phase-2 refinement
/// (perft only exercises NonEvasions/Evasions, which emit all four promotion pieces).
module Eonego.MoveGeneration

#nowarn "9" // NativePtr.stackalloc — AllowUnsafeBlocks is already set in the .fsproj

open System
open System.Diagnostics
open System.Runtime.CompilerServices
open Microsoft.FSharp.NativeInterop
open Eonego.Bitboard
open Eonego.Move
open Eonego.Position

// ---------------------------------------------------------------------------
// (1) GenType taxonomy + buffer size + color-relative pawn rank masks.
//     [<Literal>] ints (not a DU) so call sites pass a compile-time constant.
// ---------------------------------------------------------------------------
[<Literal>]
let Captures = 0

[<Literal>]
let Quiets = 1

[<Literal>]
let QuietChecks = 2 // reserved; NOT implemented in Phase 1 (search-only)

[<Literal>]
let Evasions = 3

[<Literal>]
let NonEvasions = 4

[<Literal>]
let Legal = 5 // never routed through `generate`; served by `generateLegal`

[<Literal>]
let MaxMoves = 256 // SF MAX_MOVES; max legal moves in any position ≈ 218

// Color-relative pawn rank masks (Bitboard.fs ships only Rank1/Rank8). Plain hex, 16 digits each.
[<Literal>]
let Rank2: Bitboard = 0x000000000000FF00UL // rank index 1

[<Literal>]
let Rank3: Bitboard = 0x0000000000FF0000UL // rank index 2

[<Literal>]
let Rank6: Bitboard = 0x0000FF0000000000UL // rank index 5

[<Literal>]
let Rank7: Bitboard = 0x00FF000000000000UL // rank index 6

// ---------------------------------------------------------------------------
// (2) Emit helpers — the only AggressiveInlining-attributed functions here.
//     Span passed by value (free copy, same buffer); index threaded by byref.
// ---------------------------------------------------------------------------
[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let addMove (moves: Span<Move>) (n: byref<int>) (m: Move) : unit =
    moves.[n] <- m
    n <- n + 1

/// Fan one pawn (from -> dst) into all four promotions. Emitted in EVERY path regardless of genType so an
/// under-promotion (incl. a checking knight-promotion) is never dropped. Q,R,B,N order is fixed for
/// reproducible perftDivide output; it does not affect node counts.
[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let addPromotions (moves: Span<Move>) (n: byref<int>) (from: Square) (dst: Square) : unit =
    addMove moves &n (mkPromotion from dst Queen)
    addMove moves &n (mkPromotion from dst Rook)
    addMove moves &n (mkPromotion from dst Bishop)
    addMove moves &n (mkPromotion from dst Knight)

/// Per-genType promotion emission (Stockfish make_promotions split, Phase-2 refinement). The QUEEN
/// promotion counts as a CAPTURE-class move (so a captures-only qsearch sees queen push-promotions); the
/// three under-promotions split capture vs quiet. `isCapture` = the promotion lands on an enemy square.
///   Captures  : Queen (push or capture) + Rook/Bishop/Knight only when capturing.
///   Quiets    : Rook/Bishop/Knight only when NOT capturing (queen lives in Captures).
///   Evasions / NonEvasions : all four (perft-preserving — these are the only types perft exercises).
/// Q,R,B,N order is kept (reproducible perftDivide). For all=true this is identical to addPromotions.
[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let addPromotionsGen
    (moves: Span<Move>)
    (n: byref<int>)
    (from: Square)
    (dst: Square)
    (genType: int)
    (isCapture: bool)
    : unit =
    let all = genType = Evasions || genType = NonEvasions

    if genType = Captures || all then
        addMove moves &n (mkPromotion from dst Queen)

    if (genType = Captures && isCapture) || (genType = Quiets && not isCapture) || all then
        addMove moves &n (mkPromotion from dst Rook)
        addMove moves &n (mkPromotion from dst Bishop)
        addMove moves &n (mkPromotion from dst Knight)

// ---------------------------------------------------------------------------
// (3) Per-piece generators (plain `let` — large bodies; they call Bitboard.fs's
//     already-inlined attack funcs).
// ---------------------------------------------------------------------------

/// All pawn moves for `us`, intersected with `target` (pushes vs empty, captures vs enemy — pawns compute
/// their own base masks so a generic target never emits a diagonal move onto an empty square). `target` is
/// the evasion restriction (~0UL for non-evasion gen types). EP is independent of `target`.
let genPawnMoves
    (pos: Position)
    (us: Color)
    (genType: int)
    (target: Bitboard)
    (moves: Span<Move>)
    (n: byref<int>)
    : unit =
    let them = flipColor us
    let occ = pos.Occupied
    let empty = ~~~occ
    let enemies = pos.ColorBB them
    let pawns = pos.PiecesCT us Pawn
    let promoMask = if us = White then Rank7 else Rank2
    let promo = pawns &&& promoMask
    let nonPromo = pawns &&& ~~~promoMask

    // --- pushes (quiet) + push-promotions : skipped for Captures ---
    if genType <> Captures then
        let single = (if us = White then shiftN nonPromo else shiftS nonPromo) &&& empty
        let mutable b = single &&& target

        while b <> 0UL do
            let dst = popLsb &b
            addMove moves &n (mkMove (if us = White then dst - 8 else dst + 8) dst)

        let dbl =
            (if us = White then
                 shiftN (single &&& Rank3)
             else
                 shiftS (single &&& Rank6))
            &&& empty
            &&& target

        let mutable d = dbl

        while d <> 0UL do
            let dst = popLsb &d
            addMove moves &n (mkMove (if us = White then dst - 16 else dst + 16) dst)

    // --- push-promotions (promo pawns advancing to an EMPTY back-rank square) ---
    // Computed OUTSIDE the `<>Captures` guard: a QUEEN push-promotion is a Captures-class move (so
    // qsearch sees it). The destination is empty, so it is masked by `empty` (not the capture target);
    // only Evasions further restricts to the check-block mask. addPromotionsGen does the per-type split.
    let mutable pp = (if us = White then shiftN promo else shiftS promo) &&& empty

    if genType = Evasions then
        pp <- pp &&& target

    while pp <> 0UL do
        let dst = popLsb &pp
        addPromotionsGen moves &n (if us = White then dst - 8 else dst + 8) dst genType false

    // --- captures + capture-promotions + en passant : skipped for Quiets ---
    if genType <> Quiets then
        // NE/NW for white, SE/SW for black; from = dst - shift-delta (NE+9, NW+7, SE-7, SW-9).
        let capE =
            (if us = White then shiftNE nonPromo else shiftSE nonPromo)
            &&& enemies
            &&& target

        let capW =
            (if us = White then shiftNW nonPromo else shiftSW nonPromo)
            &&& enemies
            &&& target

        let mutable e = capE

        while e <> 0UL do
            let dst = popLsb &e
            addMove moves &n (mkMove (if us = White then dst - 9 else dst + 7) dst)

        let mutable w = capW

        while w <> 0UL do
            let dst = popLsb &w
            addMove moves &n (mkMove (if us = White then dst - 7 else dst + 9) dst)
        // capture-promotions
        let cpE =
            (if us = White then shiftNE promo else shiftSE promo) &&& enemies &&& target

        let cpW =
            (if us = White then shiftNW promo else shiftSW promo) &&& enemies &&& target

        let mutable pe = cpE

        while pe <> 0UL do
            let dst = popLsb &pe
            addPromotionsGen moves &n (if us = White then dst - 9 else dst + 7) dst genType true

        let mutable pw = cpW

        while pw <> 0UL do
            let dst = popLsb &pw
            addPromotionsGen moves &n (if us = White then dst - 7 else dst + 9) dst genType true
        // en passant — independent of `target`; for Evasions only when the captured pawn is the checker.
        if pos.EpSquare <> NoSquare then
            let epSq = pos.EpSquare
            let capSq = if us = White then epSq - 8 else epSq + 8

            if genType <> Evasions || testBit pos.Checkers capSq then
                let mutable ep = pawnAttacks them epSq &&& pawns

                while ep <> 0UL do
                    let from = popLsb &ep
                    addMove moves &n (mkEnPassant from epSq)

/// Knight/Bishop/Rook/Queen moves for `us` under `target` (called once per piece type with a literal pt).
let genPieceMoves
    (pos: Position)
    (us: Color)
    (pt: PieceType)
    (target: Bitboard)
    (moves: Span<Move>)
    (n: byref<int>)
    : unit =
    let occ = pos.Occupied
    let mutable bb = pos.PiecesCT us pt

    while bb <> 0UL do
        let sq = popLsb &bb
        let mutable att = attacksFrom pt us sq occ &&& target

        while att <> 0UL do
            addMove moves &n (mkMove sq (popLsb &att))

/// Pseudo-legal king steps under `target`; self-check culling is `isLegal`'s job.
let genKingMoves (pos: Position) (us: Color) (target: Bitboard) (moves: Span<Move>) (n: byref<int>) : unit =
    let ksq = pos.KingSquare us
    let mutable att = kingAttacks ksq &&& target

    while att <> 0UL do
        addMove moves &n (mkMove ksq (popLsb &att))

/// One castling attempt for a single right bit (module-level so it never captures the Span/byref).
let private tryCastle
    (pos: Position)
    (us: Color)
    (them: Color)
    (ksq: Square)
    (occ: Bitboard)
    (right: int)
    (moves: Span<Move>)
    (n: byref<int>)
    : unit =
    if
        pos.CanCastle right
        && (pos.CastleEmptyPath right &&& occ) = 0UL
        && pos.PieceOn(pos.CastleRookSquare right) = makePiece us Rook
    then // rook-presence guard
        let mutable path = pos.CastleKingPath right
        let mutable safe = true

        while path <> 0UL && safe do
            if pos.AttackedBy them (popLsb &path) then
                safe <- false

        if safe then
            addMove moves &n (mkCastling ksq (pos.CastleKingDest right))

/// O-O / O-O-O. PRE: caller guarantees `not pos.InCheck` (castling is never generated in check).
let genCastling (pos: Position) (us: Color) (moves: Span<Move>) (n: byref<int>) : unit =
    let them = flipColor us
    let ksq = pos.KingSquare us
    let occ = pos.Occupied

    if us = White then
        tryCastle pos us them ksq occ WK moves &n
        tryCastle pos us them ksq occ WQ moves &n
    else
        tryCastle pos us them ksq occ BK moves &n
        tryCastle pos us them ksq occ BQ moves &n

// ---------------------------------------------------------------------------
// (4) Bulk dispatcher. Pseudo-legal for every genType; Evasions resolves the
//     check-mask. Returns the move count.
// ---------------------------------------------------------------------------
let generate (pos: Position) (moves: Span<Move>) (genType: int) : int =
    let us = pos.SideToMove
    let mutable n = 0

    if genType = Evasions then
        let ksq = pos.KingSquare us
        let checkers = pos.Checkers
        // King moves are always available (any non-own square); legality culls into-check squares.
        genKingMoves pos us (~~~(pos.ColorBB us)) moves &n

        if not (moreThanOne checkers) then // single check: capture the checker or interpose
            let checker = lsb checkers
            let evTarget = (between ksq checker) ||| checkers
            genPawnMoves pos us Evasions evTarget moves &n
            genPieceMoves pos us Knight evTarget moves &n
            genPieceMoves pos us Bishop evTarget moves &n
            genPieceMoves pos us Rook evTarget moves &n
            genPieceMoves pos us Queen evTarget moves &n
        // double check (moreThanOne) -> king moves only, already generated
        n
    else
        let target =
            match genType with
            | Captures -> pos.ColorBB(flipColor us)
            | Quiets -> ~~~(pos.Occupied)
            | NonEvasions -> ~~~(pos.ColorBB us)
            | _ ->
                Debug.Assert(false, "generate: unsupported genType (Legal/QuietChecks not routed here)")
                0UL

        genPawnMoves pos us genType target moves &n
        genPieceMoves pos us Knight target moves &n
        genPieceMoves pos us Bishop target moves &n
        genPieceMoves pos us Rook target moves &n
        genPieceMoves pos us Queen target moves &n
        genKingMoves pos us target moves &n

        if (genType = Quiets || genType = NonEvasions) && not pos.InCheck then
            genCastling pos us moves &n

        n

// ---------------------------------------------------------------------------
// (5) Legality filter (SF Position::legal). Module function — Position stays frozen.
// ---------------------------------------------------------------------------
let isLegal (pos: Position) (m: Move) : bool =
    let us = pos.SideToMove
    let them = flipColor us
    let ksq = pos.KingSquare us
    let from = fromSq m
    let dst = toSq m
    let occ = pos.Occupied

    if isEnPassant m then
        // Rebuild occupancy with BOTH the capturing and captured pawn gone; the king must not be exposed
        // to a rook/queen on the rank OR a bishop/queen on the diagonal (the discovered-check trap).
        let capSq = if us = White then dst - 8 else dst + 8
        let occ' = (occ ^^^ (bit from) ^^^ (bit capSq)) ||| (bit dst)
        let theirRQ = (pos.Pieces Rook ||| pos.Pieces Queen) &&& pos.ColorBB them
        let theirBQ = (pos.Pieces Bishop ||| pos.Pieces Queen) &&& pos.ColorBB them

        (rookAttacks ksq occ' &&& theirRQ) = 0UL
        && (bishopAttacks ksq occ' &&& theirBQ) = 0UL
    elif isCastling m then
        // Path + king-not-in-check already vetted in genCastling; and removing the king cannot reveal an
        // attacker on the landing square that was not already attacking the king's start (else it was
        // already in check, so genCastling would not have run).
        true
    elif from = ksq then
        // King move: destination must be unattacked with the king REMOVED from occ (X-ray through the
        // vacated square).
        (pos.AttackersTo dst (occ ^^^ (bit ksq)) &&& pos.ColorBB them) = 0UL
    else
        // Any other piece: legal iff not pinned, or moving along the pin ray.
        not (testBit (pos.BlockersForKing us) from) || aligned from dst ksq

// ---------------------------------------------------------------------------
// (6) Fully-legal generation. Pseudo-legal then in-place 0-alloc compaction with
//     SF's fast-path: only EP, king moves, and pinned pieces need the isLegal test.
// ---------------------------------------------------------------------------
let generateLegal (pos: Position) (moves: Span<Move>) : int =
    let us = pos.SideToMove
    let ksq = pos.KingSquare us
    let blockers = pos.BlockersForKing us
    let n = generate pos moves (if pos.InCheck then Evasions else NonEvasions)
    let mutable w = 0

    for i in 0 .. n - 1 do
        let m = moves.[i]
        let needsCheck = isEnPassant m || fromSq m = ksq || testBit blockers (fromSq m)

        if (not needsCheck) || isLegal pos m then
            moves.[w] <- m
            w <- w + 1

    w

// ---------------------------------------------------------------------------
// (7) Perft driver (the gate). Recurses fully to depth 0 so every move type's
//     Make/Unmake is exercised at the leaf. NOT tail-recursive by design.
// ---------------------------------------------------------------------------
let rec perft (pos: Position) (depth: int) : uint64 =
    if depth = 0 then
        1UL
    else
        let p = NativePtr.stackalloc<Move> MaxMoves
        let buf = Span<Move>(NativePtr.toVoidPtr p, MaxMoves)
        let n = generateLegal pos buf
        let mutable acc = 0UL

        for i in 0 .. n - 1 do
            let m = buf.[i]
            pos.Make m
            acc <- acc + perft pos (depth - 1)
            pos.Unmake m

        acc

/// Per-root-move node counts (cold/diagnostic path; allocates) to localize a perft mismatch.
let perftDivide (pos: Position) (depth: int) : (Move * uint64)[] =
    let p = NativePtr.stackalloc<Move> MaxMoves
    let buf = Span<Move>(NativePtr.toVoidPtr p, MaxMoves)
    let n = generateLegal pos buf
    let results = ResizeArray<Move * uint64>(n)

    for i in 0 .. n - 1 do
        let m = buf.[i]
        pos.Make m
        let cnt = if depth <= 1 then 1UL else perft pos (depth - 1)
        pos.Unmake m
        results.Add((m, cnt))

    results.ToArray()

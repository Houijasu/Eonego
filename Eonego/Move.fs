/// Eonego — ultra-fast packed 16-bit move representation for the chess engine.
///
/// Encoding (16 bits, carried in a 32-bit `int` working register):
///
///   bit: 15 14 | 13 12 | 11 10 09 08 07 06 | 05 04 03 02 01 00
///         flag  | promo |        from        |        to
///
///   to    bits 0-5    destination square 0..63 (LERF)
///   from  bits 6-11   origin square 0..63 (LERF)
///   promo bits 12-13  promotion piece as (pt - Knight): 0=N 1=B 2=R 3=Q  (ONLY meaningful when flag = Promotion)
///   flag  bits 14-15  0=Normal 1=Promotion 2=EnPassant 3=Castling
///
/// Two keys, two jobs:
///   fromTo       = m &&& 0xFFF    (12-bit; flag/promo-independent)  — butterfly history index.
///   moveMatchKey = m &&& 0x3FFF   (14-bit; from+to+promo)           — UCI legal-list restamp key.
///
/// StateInfo contract: a 16-bit move carries NO board state. The moving piece, captured piece, prior
/// castling rights, en-passant square, halfmove clock, and Zobrist key are recovered by Position's
/// StateInfo undo stack — never stored in the move.
///
/// PRE (NoSquare): never pass NoSquare (64) to a constructor — it sets bit 6/12 and corrupts the flag field.
/// MoveGen zero-fill: a zeroed `stackalloc Span<Move>` reads as MoveNone (a1->a1) — this is MoveGen's stackalloc
/// contract, not a guarantee made here.
/// make() consumer contract: Position.make should test `isSpecial m` first (one masked test, no shift); only the
/// rare special move pays the `moveFlag` dispatch.
///
/// Chess960 is out of v1 scope: castling uses standard king-to-square encoding (e1g1/e1c1/e8g8/e8c8).
module Eonego.Move

open System.Runtime.CompilerServices
open Eonego.Bitboard

// ---------------------------------------------------------------------------
// Working type — raw alias (no struct wrapper); the 16-bit payload rides in the
// low bits of a 32-bit int so every field use stays in an int context (zero
// conv ops). Storage width is decoupled via packed16/ofPacked.
// ---------------------------------------------------------------------------
type Move = int

// ---------------------------------------------------------------------------
// Move-type flag values (bits 14-15). [<Literal>] each on its own line.
// ---------------------------------------------------------------------------
[<Literal>]
let FlagNormal = 0

[<Literal>]
let FlagPromotion = 1

[<Literal>]
let FlagEnPassant = 2

[<Literal>]
let FlagCastling = 3

// ---------------------------------------------------------------------------
// Sentinels. Both have from == to (impossible for a legal move) so `isOk`
// rejects both. MoveNull = 0x41 = b1->b1 is Stockfish's Move(65).
// ---------------------------------------------------------------------------
[<Literal>]
let MoveNone = 0

[<Literal>]
let MoveNull = 0x41

// ---------------------------------------------------------------------------
// Constructors — let inline (touch only params + [<Literal>] consts + BCL, so
// cross-assembly source-inline is correct). PRE: from,dst in 0..63.
// ---------------------------------------------------------------------------

/// Normal (quiet or capture) move. PRE: from,dst in 0..63.
let inline mkMove (from: Square) (dst: Square) : Move = (from <<< 6) ||| dst

/// Promotion move. PRE: from,dst in 0..63 ; promo in {Knight,Bishop,Rook,Queen}.
/// Out-of-range promo corrupts the encoding — caller must honor the precondition.
let inline mkPromotion (from: Square) (dst: Square) (promo: PieceType) : Move =
    System.Diagnostics.Debug.Assert((promo >= Knight && promo <= Queen), "mkPromotion: promo must be Knight..Queen")
    (FlagPromotion <<< 14) ||| ((promo - Knight) <<< 12) ||| (from <<< 6) ||| dst

/// En-passant capture (dst = the square the capturing pawn lands on). PRE: from,dst in 0..63.
let inline mkEnPassant (from: Square) (dst: Square) : Move =
    (FlagEnPassant <<< 14) ||| (from <<< 6) ||| dst

/// Castling, standard king-to-square (e1g1/e1c1/e8g8/e8c8). PRE: from,dst in 0..63.
let inline mkCastling (from: Square) (dst: Square) : Move =
    (FlagCastling <<< 14) ||| (from <<< 6) ||| dst

// ---------------------------------------------------------------------------
// Accessors — let inline, pure shift+mask (branchless).
// ---------------------------------------------------------------------------

/// Destination square (0..63).
let inline toSq (m: Move) : Square = m &&& 0x3F

/// Origin square (0..63).
let inline fromSq (m: Move) : Square = (m >>> 6) &&& 0x3F

/// Move-type flag value 0..3 (the 0..3 form, e.g. for a Position.make dispatch).
let inline moveFlag (m: Move) : int = (m >>> 14) &&& 0x3

/// Promotion piece (Knight..Queen). PRE: isPromotion m (else decodes to Knight).
let inline promoType (m: Move) : PieceType = ((m >>> 12) &&& 0x3) + Knight

/// 12-bit from+to key (flag/promo-independent) — butterfly history index.
let inline fromTo (m: Move) : int = m &&& 0xFFF

/// 14-bit from+to+promo key — UCI legal-list restamp key (disambiguates under-promotions).
let inline moveMatchKey (m: Move) : int = m &&& 0x3FFF

// ---------------------------------------------------------------------------
// Predicates — let inline, DIRECT masked-compare on the in-place flag field
// (one and + one cmp, no shift). `Flag <<< 14` folds to a compile-time constant.
// ---------------------------------------------------------------------------

let inline isNormal (m: Move) : bool = (m &&& 0xC000) = (FlagNormal <<< 14)
let inline isPromotion (m: Move) : bool = (m &&& 0xC000) = (FlagPromotion <<< 14)
let inline isEnPassant (m: Move) : bool = (m &&& 0xC000) = (FlagEnPassant <<< 14)
let inline isCastling (m: Move) : bool = (m &&& 0xC000) = (FlagCastling <<< 14)

/// True for any non-Normal move (single masked test; the common-case fast path).
let inline isSpecial (m: Move) : bool = (m &&& 0xC000) <> 0

/// True iff m is the MoveNone sentinel.
let inline isNone (m: Move) : bool = m = MoveNone

/// True iff m is the MoveNull (null-move) sentinel.
let inline isNullMove (m: Move) : bool = m = MoveNull

/// Move-shape guard: rejects both sentinels and any from==to corruption.
/// (16-bit encoding integrity above bit 15 is packed16's job, not this.)
let inline isOk (m: Move) : bool = fromSq m <> toSq m

// ---------------------------------------------------------------------------
// TT compaction — storage width decoupled from the int working type.
// ---------------------------------------------------------------------------

/// Pack to 16 bits for transposition-table storage. PRE (debug-checked): high bits clear.
let inline packed16 (m: Move) : uint16 =
    System.Diagnostics.Debug.Assert((m &&& ~~~0xFFFF) = 0, "packed16: move has stray bits above 15")
    uint16 m

/// Unpack a 16-bit TT move back to the working type.
let inline ofPacked (p: uint16) : Move = int p

// ---------------------------------------------------------------------------
// Move-ordering carrier. Read copy-free via `let sm = &span.[i]` (IsReadOnly).
// Bare `Move` buffers stay 4-byte ints; ScoredMove (8 bytes) is used only after
// generation, never in the movegen hot buffer or the TT.
// ---------------------------------------------------------------------------
[<Struct; IsReadOnly>]
type ScoredMove = { Move: Move; Score: int }

/// Pair a move with an ordering score.
let inline mkScored (m: Move) (s: int) : ScoredMove = { Move = m; Score = s }

// ---------------------------------------------------------------------------
// UCI cold path — plain `let` (allocates strings; never hot).
// ---------------------------------------------------------------------------

let private fileChar (f: int) : char = char (int 'a' + f)
let private rankChar (r: int) : char = char (int '1' + r)

let private promoChar (pt: PieceType) : char =
    match pt with
    | Knight -> 'n'
    | Bishop -> 'b'
    | Rook -> 'r'
    | _ -> 'q'

/// Render a move as a UCI string ("e2e4" / "e7e8q" / "e1g1"). MoveNone AND MoveNull both render "0000".
let toUci (m: Move) : string =
    if m = MoveNone || m = MoveNull then
        "0000"
    else
        let f = fromSq m
        let t = toSq m

        let core =
            System.String(
                [| fileChar (fileOf f)
                   rankChar (rankOf f)
                   fileChar (fileOf t)
                   rankChar (rankOf t) |]
            )

        if isPromotion m then
            core + string (promoChar (promoType m))
        else
            core

/// Parse a UCI move string (context-free). Returns Normal/Promotion only — the Position layer re-stamps
/// EnPassant/Castling against the legal move list (keyed on moveMatchKey). Strict: length must be exactly
/// 4 or 5, lowercase only. "0000", any malformed input, and any from==dst all yield MoveNone.
let parseUci (s: string) : Move =
    if System.String.IsNullOrEmpty s then
        MoveNone
    elif s = "0000" then
        MoveNone
    elif s.Length <> 4 && s.Length <> 5 then
        MoveNone
    else
        let ff = int s.[0] - int 'a'
        let fr = int s.[1] - int '1'
        let tf = int s.[2] - int 'a'
        let tr = int s.[3] - int '1'

        if ff < 0 || ff > 7 || fr < 0 || fr > 7 || tf < 0 || tf > 7 || tr < 0 || tr > 7 then
            MoveNone
        else
            let from = mkSquare ff fr
            let dst = mkSquare tf tr

            if from = dst then
                MoveNone
            elif s.Length = 5 then
                match s.[4] with
                | 'n' -> mkPromotion from dst Knight
                | 'b' -> mkPromotion from dst Bishop
                | 'r' -> mkPromotion from dst Rook
                | 'q' -> mkPromotion from dst Queen
                | _ -> MoveNone
            else
                mkMove from dst

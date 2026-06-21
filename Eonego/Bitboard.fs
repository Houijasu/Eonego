/// Eonego — ultra-fast bitboard foundation for the chess engine.
///
/// Square layout: LERF (Little-Endian Rank-File). Bit i corresponds to square i,
/// with a1 = 0, b1 = 1, ..., h1 = 7, a2 = 8, ..., h8 = 63.
///   rank = sq >>> 3   (0..7, rank 1 = 0)
///   file = sq &&& 7   (0..7, file a = 0)
/// This pairs naturally with `north = bb <<< 8`.
///
/// Geometric lookup tables (Between/Line) are flat 64*64 arrays indexed as [a*64 + b].
///
/// Built incrementally per the implementation plan; sections are filled task-by-task.
module Eonego.Bitboard

open System
open System.Numerics
open System.Runtime.CompilerServices
open System.Runtime.Intrinsics.X86

// ---------------------------------------------------------------------------
// (2) Type aliases
// ---------------------------------------------------------------------------
// Raw aliases (no struct wrapper): the JIT/AOT lowers these to register operands.
type Bitboard = uint64 // LERF: bit i = square i (a1 = 0 .. h8 = 63)
type Square = int // 0..63
type Color = int // White = 0, Black = 1
type PieceType = int // 0=Pawn 1=Knight 2=Bishop 3=Rook 4=Queen 5=King
type Piece = int // 0..11 = color*6 + pieceType ; 12 = NoPiece

// ---------------------------------------------------------------------------
// (3) Constants
// ---------------------------------------------------------------------------
[<Literal>]
let White = 0

[<Literal>]
let Black = 1

[<Literal>]
let Pawn = 0

[<Literal>]
let Knight = 1

[<Literal>]
let Bishop = 2

[<Literal>]
let Rook = 3

[<Literal>]
let Queen = 4

[<Literal>]
let King = 5

[<Literal>]
let NoPiece = 12

[<Literal>]
let NoSquare = 64

// Castling-rights bits.
[<Literal>]
let WK = 1

[<Literal>]
let WQ = 2

[<Literal>]
let BK = 4

[<Literal>]
let BQ = 8

// ---------------------------------------------------------------------------
// (4) LERF scalar helpers
// ---------------------------------------------------------------------------
let inline fileOf (sq: Square) : int = sq &&& 7
let inline rankOf (sq: Square) : int = sq >>> 3
let inline mkSquare (file: int) (rank: int) : Square = (rank <<< 3) ||| file

let inline flipColor (c: Color) : Color = c ^^^ 1
let inline makePiece (c: Color) (pt: PieceType) : Piece = c * 6 + pt
let inline pieceColor (p: Piece) : Color = p / 6 // valid only for 0..11
let inline pieceType (p: Piece) : PieceType = p % 6 // gate NoPiece before calling

// ---------------------------------------------------------------------------
// (5) Hardware-capability snapshot (read ONCE; cold path only — never hot)
// ---------------------------------------------------------------------------
[<Struct>]
type Caps =
    { Popcnt: bool
      Bmi1: bool
      Bmi2: bool
      Lzcnt: bool
      Avx2: bool
      X64: bool }

let caps: Caps =
    { Popcnt = Popcnt.X64.IsSupported
      Bmi1 = Bmi1.X64.IsSupported
      Bmi2 = Bmi2.X64.IsSupported
      Lzcnt = Lzcnt.X64.IsSupported
      Avx2 = Avx2.IsSupported
      X64 = Environment.Is64BitProcess }

// ---------------------------------------------------------------------------
// (6) Leaf bit operations — SINGLE source of truth.
//     All lowered by the JIT/AOT to POPCNT / TZCNT / LZCNT / BLSR on x64,
//     with a correct software fallback baked into BitOperations.
// ---------------------------------------------------------------------------

/// Number of set bits.
let inline popCount (b: Bitboard) : int = BitOperations.PopCount b

/// Index (0..63) of the least-significant set bit. PRECONDITION: b <> 0 (returns 64 on 0).
let inline lsb (b: Bitboard) : Square = BitOperations.TrailingZeroCount b

/// Index (0..63) of the most-significant set bit. PRECONDITION: b <> 0 (returns -1 on 0).
let inline msb (b: Bitboard) : Square = 63 - BitOperations.LeadingZeroCount b

/// Index of the LSB; clears it from `b` in place. Canonical set-bit iterator step.
/// Emits BLSR (`b & (b-1)`) under BMI1. PRECONDITION: b <> 0.
let inline popLsb (b: byref<Bitboard>) : Square =
    let s = BitOperations.TrailingZeroCount b
    b <- b &&& (b - 1UL)
    s

/// Single-bit mask for a square.
let inline bit (sq: Square) : Bitboard = 1UL <<< sq

let inline testBit (b: Bitboard) (sq: Square) : bool = (b &&& (1UL <<< sq)) <> 0UL
let inline setBit (b: Bitboard) (sq: Square) : Bitboard = b ||| (1UL <<< sq)
let inline clearBit (b: Bitboard) (sq: Square) : Bitboard = b &&& ~~~(1UL <<< sq)
let inline toggleBit (b: Bitboard) (sq: Square) : Bitboard = b ^^^ (1UL <<< sq)

/// a AND (NOT b) — may map to the ANDN instruction.
let inline andNot (a: Bitboard) (b: Bitboard) : Bitboard = a &&& ~~~b

/// True when at least two bits are set.
let inline moreThanOne (b: Bitboard) : bool = (b &&& (b - 1UL)) <> 0UL

/// True when exactly one bit is set.
let inline hasSingleBit (b: Bitboard) : bool = b <> 0UL && (b &&& (b - 1UL)) = 0UL

// ---------------------------------------------------------------------------
// (7) File / rank / edge masks  ([<Literal>] requires PLAIN HEX — no computed ~~~)
// ---------------------------------------------------------------------------
[<Literal>]
let FileA: Bitboard = 0x0101010101010101UL

[<Literal>]
let FileH: Bitboard = 0x8080808080808080UL

[<Literal>]
let Rank1: Bitboard = 0x00000000000000ffUL

[<Literal>]
let Rank8: Bitboard = 0xff00000000000000UL

[<Literal>]
let NotAFile: Bitboard = 0xfefefefefefefefeUL // ~FileA

[<Literal>]
let NotHFile: Bitboard = 0x7f7f7f7f7f7f7f7fUL // ~FileH

[<Literal>]
let NotABFile: Bitboard = 0xfcfcfcfcfcfcfcfcUL // ~(FileA | FileB) — for knight moves

[<Literal>]
let NotGHFile: Bitboard = 0x3f3f3f3f3f3f3f3fUL // ~(FileG | FileH) — for knight moves

/// File mask for file 0..7.
let inline fileBB (f: int) : Bitboard = FileA <<< f
/// Rank mask for rank 0..7.
let inline rankBB (r: int) : Bitboard = Rank1 <<< (r * 8)

// ---------------------------------------------------------------------------
// (8) Wrap-safe directional shifts (LERF). Mask the off-board file BEFORE the
//     shift so an h-file bit never wraps onto the a-file of the next rank.
// ---------------------------------------------------------------------------
let inline shiftN (b: Bitboard) : Bitboard = b <<< 8
let inline shiftS (b: Bitboard) : Bitboard = b >>> 8
let inline shiftE (b: Bitboard) : Bitboard = (b &&& NotHFile) <<< 1
let inline shiftW (b: Bitboard) : Bitboard = (b &&& NotAFile) >>> 1
let inline shiftNE (b: Bitboard) : Bitboard = (b &&& NotHFile) <<< 9
let inline shiftNW (b: Bitboard) : Bitboard = (b &&& NotAFile) <<< 7
let inline shiftSE (b: Bitboard) : Bitboard = (b &&& NotHFile) >>> 7
let inline shiftSW (b: Bitboard) : Bitboard = (b &&& NotAFile) >>> 9

// Kogge-Stone fills (occupancy-independent ray smearing; used by later table init).
let inline northFill (b: Bitboard) : Bitboard =
    let b = b ||| (b <<< 8)
    let b = b ||| (b <<< 16)
    b ||| (b <<< 32)

let inline southFill (b: Bitboard) : Bitboard =
    let b = b ||| (b >>> 8)
    let b = b ||| (b >>> 16)
    b ||| (b >>> 32)

let inline fileFill (b: Bitboard) : Bitboard = northFill b ||| southFill b

// ---------------------------------------------------------------------------
// (9) Geometry tables: diagonals, Between[] (strictly-between), Line[] (full
//     ray through both squares), and forward-rank masks. Flat [a*64 + b].
//     Accessors use the AggressiveInlining ATTRIBUTE (not F# `inline`) so they
//     may read the private backing arrays yet still inline within the assembly.
// ---------------------------------------------------------------------------
let private Diag: Bitboard[] = Array.zeroCreate 64
let private AntiDiag: Bitboard[] = Array.zeroCreate 64
let private Between: Bitboard[] = Array.zeroCreate (64 * 64)
let private Line: Bitboard[] = Array.zeroCreate (64 * 64)
let private ForwardRanks: Bitboard[] = Array.zeroCreate (2 * 8)

/// Single-square step (LERF index delta) from a toward b when the two are
/// aligned; 0 when they are not.
let private stepDir (a: int) (b: int) : int =
    let df = fileOf b - fileOf a
    let dr = rankOf b - rankOf a

    if dr = 0 && df <> 0 then (if df > 0 then 1 else -1) // rank
    elif df = 0 && dr <> 0 then (if dr > 0 then 8 else -8) // file
    elif df = dr then (if dr > 0 then 9 else -9) // a1-h8 diagonal
    elif df = -dr then (if dr > 0 then 7 else -7) // anti-diagonal
    else 0

let private lineThrough (a: int) (b: int) : Bitboard =
    if rankOf a = rankOf b then
        rankBB (rankOf a)
    elif fileOf a = fileOf b then
        fileBB (fileOf a)
    elif fileOf a - rankOf a = fileOf b - rankOf b then
        Diag.[a]
    elif fileOf a + rankOf a = fileOf b + rankOf b then
        AntiDiag.[a]
    else
        0UL

let private initGeometry () =
    for sq in 0..63 do
        let mutable d = 0UL
        let mutable a = 0UL

        for t in 0..63 do
            if fileOf t - rankOf t = fileOf sq - rankOf sq then
                d <- d ||| bit t

            if fileOf t + rankOf t = fileOf sq + rankOf sq then
                a <- a ||| bit t

        Diag.[sq] <- d
        AntiDiag.[sq] <- a

    for a in 0..63 do
        for b in 0..63 do
            if a <> b then
                let ln = lineThrough a b

                if ln <> 0UL then
                    Line.[(a <<< 6) + b] <- ln
                    let step = stepDir a b
                    let mutable s = a + step
                    let mutable acc = 0UL

                    while s <> b do
                        acc <- acc ||| bit s
                        s <- s + step

                    Between.[(a <<< 6) + b] <- acc

    for r in 0..7 do
        let mutable w = 0UL

        for rr in r + 1 .. 7 do
            w <- w ||| rankBB rr

        ForwardRanks.[(White <<< 3) + r] <- w
        let mutable bl = 0UL

        for rr in 0 .. r - 1 do
            bl <- bl ||| rankBB rr

        ForwardRanks.[(Black <<< 3) + r] <- bl

do initGeometry ()

/// Squares strictly between a and b along their shared line; 0 if not aligned.
[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let between (a: Square) (b: Square) : Bitboard = Between.[(a <<< 6) + b]

/// Full ray through a and b across the whole board; 0 if not aligned.
[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let line (a: Square) (b: Square) : Bitboard = Line.[(a <<< 6) + b]

/// True if c lies on the line through a and b.
[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let aligned (a: Square) (b: Square) (c: Square) : bool =
    (Line.[(a <<< 6) + b] &&& (1UL <<< c)) <> 0UL

/// All ranks strictly ahead of `r` from `c`'s perspective (White advances up).
[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let forwardRanks (c: Color) (r: int) : Bitboard = ForwardRanks.[(c <<< 3) + r]

// ---------------------------------------------------------------------------
// (10) Classical ray-cast sliding attacks (obstruction difference).
//      Dual role: the ORACLE that validates the magic/PEXT tables, and the
//      BUILDER used to fill them at init.
//      RayAttacks.[dir*64 + sq] = the full open ray from sq in `dir`.
// ---------------------------------------------------------------------------
[<Literal>]
let private DN = 0 // north      (+8)  positive

[<Literal>]
let private DE = 1 // east       (+1)  positive

[<Literal>]
let private DNE = 2 // north-east (+9)  positive

[<Literal>]
let private DNW = 3 // north-west (+7)  positive

[<Literal>]
let private DS = 4 // south      (-8)  negative

[<Literal>]
let private DW = 5 // west       (-1)  negative

[<Literal>]
let private DSE = 6 // south-east (-7)  negative

[<Literal>]
let private DSW = 7 // south-west (-9)  negative

let private RayAttacks: Bitboard[] = Array.zeroCreate (8 * 64)

let private buildRay (shift: Bitboard -> Bitboard) (sq: int) : Bitboard =
    let mutable acc = 0UL
    let mutable x = shift (bit sq)

    while x <> 0UL do
        acc <- acc ||| x
        x <- shift x

    acc

let private initRays () =
    for sq in 0..63 do
        RayAttacks.[(DN <<< 6) + sq] <- buildRay shiftN sq
        RayAttacks.[(DE <<< 6) + sq] <- buildRay shiftE sq
        RayAttacks.[(DNE <<< 6) + sq] <- buildRay shiftNE sq
        RayAttacks.[(DNW <<< 6) + sq] <- buildRay shiftNW sq
        RayAttacks.[(DS <<< 6) + sq] <- buildRay shiftS sq
        RayAttacks.[(DW <<< 6) + sq] <- buildRay shiftW sq
        RayAttacks.[(DSE <<< 6) + sq] <- buildRay shiftSE sq
        RayAttacks.[(DSW <<< 6) + sq] <- buildRay shiftSW sq

do initRays ()

/// Ray toward higher squares: stop at (and include) the nearest blocker (lsb).
[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let private posRay (dir: int) (sq: int) (occ: Bitboard) : Bitboard =
    let attacks = RayAttacks.[(dir <<< 6) + sq]
    let blockers = attacks &&& occ

    if blockers = 0UL then
        attacks
    else
        attacks ^^^ RayAttacks.[(dir <<< 6) + BitOperations.TrailingZeroCount blockers]

/// Ray toward lower squares: stop at (and include) the nearest blocker (msb).
[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let private negRay (dir: int) (sq: int) (occ: Bitboard) : Bitboard =
    let attacks = RayAttacks.[(dir <<< 6) + sq]
    let blockers = attacks &&& occ

    if blockers = 0UL then
        attacks
    else
        attacks
        ^^^ RayAttacks.[(dir <<< 6) + (63 - BitOperations.LeadingZeroCount blockers)]

[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let rookAttacksClassical (sq: Square) (occ: Bitboard) : Bitboard =
    posRay DN sq occ ||| posRay DE sq occ ||| negRay DS sq occ ||| negRay DW sq occ

[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let bishopAttacksClassical (sq: Square) (occ: Bitboard) : Bitboard =
    posRay DNE sq occ
    ||| posRay DNW sq occ
    ||| negRay DSE sq occ
    ||| negRay DSW sq occ

[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let queenAttacksClassical (sq: Square) (occ: Bitboard) : Bitboard =
    rookAttacksClassical sq occ ||| bishopAttacksClassical sq occ

// ---------------------------------------------------------------------------
// (11) Leaper attack tables (occupancy-independent): pawn / knight / king.
//      Pawn table is indexed [(color <<< 6) + sq].
// ---------------------------------------------------------------------------
let private PawnAttacks: Bitboard[] = Array.zeroCreate 128
let private KnightAttacks: Bitboard[] = Array.zeroCreate 64
let private KingAttacks: Bitboard[] = Array.zeroCreate 64

let private kingFrom (b: Bitboard) : Bitboard =
    shiftN b
    ||| shiftS b
    ||| shiftE b
    ||| shiftW b
    ||| shiftNE b
    ||| shiftNW b
    ||| shiftSE b
    ||| shiftSW b

let private knightFrom (b: Bitboard) : Bitboard =
    let e1 = (b <<< 1) &&& NotAFile // east 1 file
    let e2 = (b <<< 2) &&& NotABFile // east 2 files
    let w1 = (b >>> 1) &&& NotHFile // west 1 file
    let w2 = (b >>> 2) &&& NotGHFile // west 2 files
    let h1 = e1 ||| w1 // one-file moves -> +/- 2 ranks
    let h2 = e2 ||| w2 // two-file moves -> +/- 1 rank
    (h1 <<< 16) ||| (h1 >>> 16) ||| (h2 <<< 8) ||| (h2 >>> 8)

let private initLeapers () =
    for sq in 0..63 do
        let b = bit sq
        KnightAttacks.[sq] <- knightFrom b
        KingAttacks.[sq] <- kingFrom b
        PawnAttacks.[(White <<< 6) + sq] <- shiftNE b ||| shiftNW b
        PawnAttacks.[(Black <<< 6) + sq] <- shiftSE b ||| shiftSW b

do initLeapers ()

[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let pawnAttacks (c: Color) (sq: Square) : Bitboard = PawnAttacks.[(c <<< 6) + sq]

[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let knightAttacks (sq: Square) : Bitboard = KnightAttacks.[sq]

[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let kingAttacks (sq: Square) : Bitboard = KingAttacks.[sq]

// ---------------------------------------------------------------------------
// (12) Slider relevant-occupancy masks + carry-rippler subset enumeration.
//      Masks exclude the board-edge square of each ray (a blocker there cannot
//      shadow anything) and the slider's own square.
// ---------------------------------------------------------------------------

/// Rook relevant-occupancy mask (each ray walked to the penultimate square).
let rookMask (sq: Square) : Bitboard =
    let f = fileOf sq
    let r = rankOf sq
    let mutable m = 0UL

    for rr in r + 1 .. 6 do
        m <- m ||| bit (mkSquare f rr) // north (stop before rank 8)

    for rr in r - 1 .. -1 .. 1 do
        m <- m ||| bit (mkSquare f rr) // south (stop before rank 1)

    for ff in f + 1 .. 6 do
        m <- m ||| bit (mkSquare ff r) // east  (stop before file h)

    for ff in f - 1 .. -1 .. 1 do
        m <- m ||| bit (mkSquare ff r) // west (stop before file a)

    m

/// Bishop relevant-occupancy mask (each diagonal ray walked to the penultimate square).
let bishopMask (sq: Square) : Bitboard =
    let f = fileOf sq
    let r = rankOf sq
    let mutable m = 0UL

    for (df, dr) in [ (1, 1); (1, -1); (-1, 1); (-1, -1) ] do
        let mutable ff = f + df
        let mutable rr = r + dr

        while ff >= 1 && ff <= 6 && rr >= 1 && rr <= 6 do
            m <- m ||| bit (mkSquare ff rr)
            ff <- ff + df
            rr <- rr + dr

    m

/// All 2^popcount(mask) subsets of `mask`, in carry-rippler order (index 0 = empty).
let subsetsOf (mask: Bitboard) : Bitboard[] =
    let result = Array.zeroCreate (1 <<< popCount mask)
    let mutable occ = 0UL
    let mutable i = 0
    let mutable go = true

    while go do
        result.[i] <- occ
        i <- i + 1
        occ <- (occ - mask) &&& mask

        if occ = 0UL then
            go <- false

    result

let private RookMasks: Bitboard[] = Array.init 64 rookMask
let private BishopMasks: Bitboard[] = Array.init 64 bishopMask

// ---------------------------------------------------------------------------
// (13) Fancy magic bitboards.
//      Magics are GENERATED deterministically at init (fixed-seed sparse search,
//      validated against the classical oracle) rather than embedded as 128 hex
//      constants — provably correct, deterministic, AOT-clean. Tables are flat
//      with per-square offsets for cache locality (rook 102400, bishop 5248).
// ---------------------------------------------------------------------------
[<Struct; IsReadOnly>]
type private Magic =
    { Mask: Bitboard
      Magic: uint64
      Shift: int
      Offset: int }

let private RookMagic: Magic[] = Array.zeroCreate 64
let private BishopMagic: Magic[] = Array.zeroCreate 64
let private RookTable: Bitboard[] = Array.zeroCreate 102400
let private BishopTable: Bitboard[] = Array.zeroCreate 5248

// xorshift64 PRNG with a fixed seed -> reproducible magic search.
let mutable private magicSeed = 0x2545F4914F6CDD1DUL

let private nextRand () : uint64 =
    let mutable s = magicSeed
    s <- s ^^^ (s <<< 13)
    s <- s ^^^ (s >>> 7)
    s <- s ^^^ (s <<< 17)
    magicSeed <- s
    s

let private sparseRand () : uint64 =
    nextRand () &&& nextRand () &&& nextRand ()

/// Search for a collision-free magic for `sq` (constructive collisions on equal
/// attack sets are allowed, as in standard fancy magics).
let private findMagic (sq: int) (mask: Bitboard) (classical: int -> Bitboard -> Bitboard) : uint64 =
    let n = popCount mask
    let shift = 64 - n
    let subs = subsetsOf mask
    let size = subs.Length
    let reference = Array.init size (fun i -> classical sq subs.[i])
    let used = Array.zeroCreate size: Bitboard[]
    let seen = Array.zeroCreate size: bool[]
    let mutable result = 0UL

    while result = 0UL do
        let magic = sparseRand ()
        // cheap filter: the top byte of mask*magic must be well populated.
        if popCount ((mask * magic) &&& 0xFF00000000000000UL) >= 6 then
            Array.fill seen 0 size false
            let mutable collision = false
            let mutable i = 0

            while i < size && not collision do
                let idx = int ((subs.[i] * magic) >>> shift)

                if not seen.[idx] then
                    seen.[idx] <- true
                    used.[idx] <- reference.[i]
                elif used.[idx] <> reference.[i] then
                    collision <- true

                i <- i + 1

            if not collision then
                result <- magic

    result

let private buildMagic
    (masks: Bitboard[])
    (magicArr: Magic[])
    (table: Bitboard[])
    (classical: int -> Bitboard -> Bitboard)
    =
    let mutable offset = 0

    for sq in 0..63 do
        let mask = masks.[sq]
        let n = popCount mask
        let shift = 64 - n
        let magic = findMagic sq mask classical

        magicArr.[sq] <-
            { Mask = mask
              Magic = magic
              Shift = shift
              Offset = offset }

        for occ in subsetsOf mask do
            let idx = offset + int ((occ * magic) >>> shift)
            let attack = classical sq occ
            System.Diagnostics.Debug.Assert((table.[idx] = 0UL || table.[idx] = attack), "magic table collision")
            table.[idx] <- attack

        offset <- offset + (1 <<< n)

do buildMagic RookMasks RookMagic RookTable rookAttacksClassical
do buildMagic BishopMasks BishopMagic BishopTable bishopAttacksClassical

[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let rookAttacksMagic (sq: Square) (occ: Bitboard) : Bitboard =
    let m = &RookMagic.[sq]
    RookTable.[m.Offset + int (((occ &&& m.Mask) * m.Magic) >>> m.Shift)]

[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let bishopAttacksMagic (sq: Square) (occ: Bitboard) : Bitboard =
    let m = &BishopMagic.[sq]
    BishopTable.[m.Offset + int (((occ &&& m.Mask) * m.Magic) >>> m.Shift)]

// ---------------------------------------------------------------------------
// (14) BMI2 PEXT slider tables. PEXT compresses the masked occupancy bits into
//      a dense contiguous index, so the same flat-with-offsets layout applies.
//      Built only when BMI2 is present (ParallelBitExtract is x64+BMI2-only).
// ---------------------------------------------------------------------------
[<Struct; IsReadOnly>]
type private Pext = { Mask: Bitboard; Offset: int }

let private RookPextIdx: Pext[] = Array.zeroCreate 64
let private BishopPextIdx: Pext[] = Array.zeroCreate 64
let private RookPext: Bitboard[] = Array.zeroCreate 102400
let private BishopPext: Bitboard[] = Array.zeroCreate 5248

let private buildPext (masks: Bitboard[]) (idx: Pext[]) (table: Bitboard[]) (classical: int -> Bitboard -> Bitboard) =
    let mutable offset = 0

    for sq in 0..63 do
        let mask = masks.[sq]
        idx.[sq] <- { Mask = mask; Offset = offset }

        for occ in subsetsOf mask do
            table.[offset + int (Bmi2.X64.ParallelBitExtract(occ, mask))] <- classical sq occ

        offset <- offset + (1 <<< popCount mask)

do
    if Bmi2.X64.IsSupported then
        buildPext RookMasks RookPextIdx RookPext rookAttacksClassical
        buildPext BishopMasks BishopPextIdx BishopPext bishopAttacksClassical

[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let rookAttacksPext (sq: Square) (occ: Bitboard) : Bitboard =
    let p = &RookPextIdx.[sq]
    RookPext.[p.Offset + int (Bmi2.X64.ParallelBitExtract(occ, p.Mask))]

[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let bishopAttacksPext (sq: Square) (occ: Bitboard) : Bitboard =
    let p = &BishopPextIdx.[sq]
    BishopPext.[p.Offset + int (Bmi2.X64.ParallelBitExtract(occ, p.Mask))]

// ---------------------------------------------------------------------------
// (15) Unified slider dispatch + attacksFrom.
//      `Bmi2.X64.IsSupported` is a JIT/AOT intrinsic that folds to a constant,
//      so the unused branch is eliminated — zero-cost selection, no runtime flag.
// ---------------------------------------------------------------------------

/// True when the BMI2 PEXT slider path is active (else magic fallback).
let usesPext: bool = Bmi2.X64.IsSupported

[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let rookAttacks (sq: Square) (occ: Bitboard) : Bitboard =
    if Bmi2.X64.IsSupported then
        rookAttacksPext sq occ
    else
        rookAttacksMagic sq occ

[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let bishopAttacks (sq: Square) (occ: Bitboard) : Bitboard =
    if Bmi2.X64.IsSupported then
        bishopAttacksPext sq occ
    else
        bishopAttacksMagic sq occ

[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let queenAttacks (sq: Square) (occ: Bitboard) : Bitboard =
    rookAttacks sq occ ||| bishopAttacks sq occ

/// Attack set of `pt` (PieceType 0..5) of colour `c` on `sq` given `occ`.
let attacksFrom (pt: PieceType) (c: Color) (sq: Square) (occ: Bitboard) : Bitboard =
    match pt with
    | Pawn -> pawnAttacks c sq
    | Knight -> knightAttacks sq
    | Bishop -> bishopAttacks sq occ
    | Rook -> rookAttacks sq occ
    | Queen -> queenAttacks sq occ
    | _ -> kingAttacks sq

/// Forces table initialization (built at module load anyway; this is a safe
/// explicit warmup point) and returns a one-line capability banner.
let init () : string =
    "Eonego.Bitboard: sliders="
    + (if usesPext then "PEXT" else "magic")
    + " popcnt="
    + string caps.Popcnt
    + " bmi1="
    + string caps.Bmi1
    + " bmi2="
    + string caps.Bmi2
    + " avx2="
    + string caps.Avx2
    + " x64="
    + string caps.X64

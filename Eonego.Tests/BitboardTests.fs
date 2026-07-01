module Eonego.Tests.BitboardTests

open Xunit
open Eonego.Bitboard

// ---------------------------------------------------------------------------
// Task 2 — LERF mapping + scalar helpers
// ---------------------------------------------------------------------------

[<Fact>]
let ``LERF: a1 = 0, h8 = 63`` () =
    Assert.Equal(0, mkSquare 0 0) // a1
    Assert.Equal(7, mkSquare 7 0) // h1
    Assert.Equal(56, mkSquare 0 7) // a8
    Assert.Equal(63, mkSquare 7 7) // h8

[<Fact>]
let ``file/rank round-trip for every square`` () =
    for sq in 0..63 do
        Assert.Equal(sq, mkSquare (fileOf sq) (rankOf sq))

[<Fact>]
let ``fileOf/rankOf decompose e4 (=28)`` () =
    Assert.Equal(28, mkSquare 4 3)
    Assert.Equal(4, fileOf 28)
    Assert.Equal(3, rankOf 28)

[<Fact>]
let ``piece encode/decode`` () =
    Assert.Equal(makePiece White Knight, 1)
    Assert.Equal(makePiece Black Pawn, 6)
    Assert.Equal(White, pieceColor (makePiece White Queen))
    Assert.Equal(Black, pieceColor (makePiece Black Rook))
    Assert.Equal(Queen, pieceType (makePiece Black Queen))
    Assert.Equal(Black, flipColor White)

// ---------------------------------------------------------------------------
// Task 3 — leaf bit operations
// ---------------------------------------------------------------------------

let private naivePopcount (b: uint64) =
    let mutable n = 0
    let mutable x = b

    while x <> 0UL do
        n <- n + 1
        x <- x &&& (x - 1UL)

    n

let private naiveLsb (b: uint64) =
    let mutable i = 0

    while i < 64 && (b >>> i) &&& 1UL = 0UL do
        i <- i + 1

    i

let private naiveMsb (b: uint64) =
    let mutable i = 63

    while i >= 0 && (b >>> i) &&& 1UL = 0UL do
        i <- i - 1

    i

/// Full-width 64-bit random (NextInt64 covers bits 0..62; OR in a random bit 63).
let private nextU64 (rng: System.Random) : uint64 =
    uint64 (rng.NextInt64()) ||| ((uint64 (rng.Next(0, 2))) <<< 63)

[<Fact>]
let ``popCount edges`` () =
    Assert.Equal(0, popCount 0UL)
    Assert.Equal(64, popCount 0xFFFFFFFFFFFFFFFFUL)
    Assert.Equal(4, popCount 0b1011010UL)

[<Fact>]
let ``lsb and msb`` () =
    Assert.Equal(0, lsb 1UL)
    Assert.Equal(7, lsb 0x80UL)
    Assert.Equal(63, msb 0x8000000000000000UL)
    Assert.Equal(0, msb 1UL)

[<Fact>]
let ``popLsb returns index and clears lowest bit`` () =
    let mutable b = 0b1010UL
    Assert.Equal(1, popLsb &b)
    Assert.Equal(0b1000UL, b)
    Assert.Equal(3, popLsb &b)
    Assert.Equal(0UL, b)

[<Fact>]
let ``set/clear/toggle/test bit`` () =
    let b = setBit 0UL 4
    Assert.True(testBit b 4)
    Assert.False(testBit b 5)
    Assert.Equal(0UL, clearBit b 4)
    Assert.Equal(b, toggleBit 0UL 4)
    Assert.Equal(0UL, toggleBit b 4)

[<Fact>]
let ``moreThanOne and hasSingleBit`` () =
    Assert.False(moreThanOne 0UL)
    Assert.False(moreThanOne (bit 9))
    Assert.True(moreThanOne (setBit (bit 9) 40))
    Assert.True(hasSingleBit (bit 31))
    Assert.False(hasSingleBit 0UL)
    Assert.False(hasSingleBit (setBit (bit 9) 40))

[<Fact>]
let ``bit ops parity vs naive over 10k random values`` () =
    let rng = System.Random(0xC0FFEE)

    for _ in 0..9999 do
        let v = nextU64 rng
        Assert.Equal(naivePopcount v, popCount v)

        if v <> 0UL then
            Assert.Equal(naiveLsb v, lsb v)
            Assert.Equal(naiveMsb v, msb v)
            // popLsb clears exactly the lsb
            let mutable m = v
            let s = popLsb &m
            Assert.Equal(naiveLsb v, s)
            Assert.Equal(v &&& (v - 1UL), m)

// ---------------------------------------------------------------------------
// Task 4 — edge masks + wrap-safe shifts
// ---------------------------------------------------------------------------

[<Fact>]
let ``file and rank masks`` () =
    Assert.Equal(0x0101010101010101UL, FileA)
    Assert.Equal(0x8080808080808080UL, FileH)
    Assert.Equal(0xFFUL, Rank1)
    Assert.Equal(0xFF00000000000000UL, Rank8)
    Assert.Equal(8, popCount FileA)
    Assert.Equal(8, popCount Rank1)
    Assert.Equal(FileA, fileBB 0)
    Assert.Equal(FileH, fileBB 7)
    Assert.Equal(Rank8, rankBB 7)

[<Fact>]
let ``orthogonal shifts move one square`` () =
    Assert.Equal(bit (mkSquare 0 1), shiftN (bit (mkSquare 0 0))) // a1 -> a2
    Assert.Equal(bit (mkSquare 0 0), shiftS (bit (mkSquare 0 1))) // a2 -> a1
    Assert.Equal(bit (mkSquare 1 0), shiftE (bit (mkSquare 0 0))) // a1 -> b1
    Assert.Equal(bit (mkSquare 0 0), shiftW (bit (mkSquare 1 0))) // b1 -> a1

[<Fact>]
let ``horizontal shifts do not wrap across the board edge`` () =
    Assert.Equal(0UL, shiftE (bit (mkSquare 7 0))) // h1 east -> off board
    Assert.Equal(0UL, shiftW (bit (mkSquare 0 0))) // a1 west -> off board
    Assert.Equal(0UL, shiftNE (bit (mkSquare 7 3))) // h4 NE -> off board
    Assert.Equal(0UL, shiftNW (bit (mkSquare 0 3))) // a4 NW -> off board
    Assert.Equal(0UL, shiftSE (bit (mkSquare 7 3))) // h4 SE -> off board
    Assert.Equal(0UL, shiftSW (bit (mkSquare 0 3))) // a4 SW -> off board

[<Fact>]
let ``diagonal shifts land on the right square`` () =
    Assert.Equal(bit (mkSquare 1 1), shiftNE (bit (mkSquare 0 0))) // a1 -> b2
    Assert.Equal(bit (mkSquare 0 1), shiftNW (bit (mkSquare 1 0))) // b1 -> a2
    Assert.Equal(bit (mkSquare 1 0), shiftSE (bit (mkSquare 0 1))) // a2 -> b1
    Assert.Equal(bit (mkSquare 0 0), shiftSW (bit (mkSquare 1 1))) // b2 -> a1

[<Fact>]
let ``shiftN then shiftS is identity on interior squares`` () =
    for sq in 0..63 do
        if rankOf sq < 7 then
            Assert.Equal(bit sq, shiftS (shiftN (bit sq)))

        if fileOf sq < 7 then
            Assert.Equal(bit sq, shiftW (shiftE (bit sq)))

[<Fact>]
let ``fileFill smears a single bit across its file`` () =
    Assert.Equal(FileA, fileFill (bit (mkSquare 0 3))) // any a-file bit fills file a
    Assert.Equal(FileH, fileFill (bit (mkSquare 7 5)))

// ---------------------------------------------------------------------------
// Task 5 — geometry: Between / Line / aligned / forwardRanks
// (validated against an independent brute-force ray-walk oracle)
// ---------------------------------------------------------------------------

let private oracleBetween (a: int) (b: int) : uint64 =
    let dirs = [ (1, 0); (-1, 0); (0, 1); (0, -1); (1, 1); (1, -1); (-1, 1); (-1, -1) ]
    let fa, ra = fileOf a, rankOf a
    let mutable result = 0UL

    for (df, dr) in dirs do
        let mutable f = fa + df
        let mutable r = ra + dr
        let mutable acc = 0UL
        let mutable hit = false
        let mutable stop = false

        while not stop && f >= 0 && f <= 7 && r >= 0 && r <= 7 do
            let s = mkSquare f r

            if s = b then
                hit <- true
                stop <- true
            else
                acc <- acc ||| bit s
                f <- f + df
                r <- r + dr

        if hit then
            result <- acc

    result

let private oracleLine (a: int) (b: int) : uint64 =
    if a = b then
        0UL
    else
        let axes = [ (1, 0); (0, 1); (1, 1); (1, -1) ]
        let fa, ra = fileOf a, rankOf a
        let mutable result = 0UL

        for (df, dr) in axes do
            let mutable lineBits = bit a
            let mutable onLine = false

            for sign in [ 1; -1 ] do
                let mutable f = fa + df * sign
                let mutable r = ra + dr * sign

                while f >= 0 && f <= 7 && r >= 0 && r <= 7 do
                    let s = mkSquare f r
                    lineBits <- lineBits ||| bit s

                    if s = b then
                        onLine <- true

                    f <- f + df * sign
                    r <- r + dr * sign

            if onLine then
                result <- lineBits

        result

[<Fact>]
let ``Between and Line match brute-force oracle over all 4096 pairs`` () =
    for a in 0..63 do
        for b in 0..63 do
            Assert.Equal(oracleBetween a b, between a b)
            Assert.Equal(oracleLine a b, line a b)

[<Fact>]
let ``Between is exclusive, symmetric, and zero when not aligned`` () =
    let a1, h8 = mkSquare 0 0, mkSquare 7 7
    Assert.Equal(between a1 h8, between h8 a1) // symmetric
    Assert.Equal(6, popCount (between a1 h8)) // 6 squares strictly between
    Assert.False(testBit (between a1 h8) a1) // excludes endpoints
    Assert.False(testBit (between a1 h8) h8)
    Assert.Equal(0UL, between a1 (mkSquare 1 2)) // knight offset: not aligned
    Assert.Equal(0UL, line a1 (mkSquare 1 2))
    Assert.Equal(0UL, between (mkSquare 0 0) (mkSquare 1 0)) // adjacent: nothing between

[<Fact>]
let ``aligned predicate`` () =
    let a1, h8 = mkSquare 0 0, mkSquare 7 7
    Assert.True(aligned a1 h8 (mkSquare 3 3)) // d4 on long diagonal
    Assert.False(aligned a1 h8 (mkSquare 3 4)) // d5 off it

[<Fact>]
let ``forwardRanks`` () =
    let above1 =
        rankBB 2 ||| rankBB 3 ||| rankBB 4 ||| rankBB 5 ||| rankBB 6 ||| rankBB 7

    Assert.Equal(above1, forwardRanks White 1)
    Assert.Equal(rankBB 0, forwardRanks Black 1)
    Assert.Equal(0UL, forwardRanks White 7)
    Assert.Equal(0UL, forwardRanks Black 0)

// ---------------------------------------------------------------------------
// Task 6 — classical ray-cast sliders (oracle/builder)
// (validated against an independent brute-force occupancy ray-walk)
// ---------------------------------------------------------------------------

let private slowSlide (sq: int) (occ: uint64) (dirs: (int * int) list) : uint64 =
    let fa, ra = fileOf sq, rankOf sq
    let mutable acc = 0UL

    for (df, dr) in dirs do
        let mutable f = fa + df
        let mutable r = ra + dr
        let mutable blocked = false

        while not blocked && f >= 0 && f <= 7 && r >= 0 && r <= 7 do
            let s = mkSquare f r
            acc <- acc ||| bit s

            if (occ >>> s) &&& 1UL <> 0UL then
                blocked <- true

            f <- f + df
            r <- r + dr

    acc

let private rookDirs = [ (1, 0); (-1, 0); (0, 1); (0, -1) ]
let private bishopDirs = [ (1, 1); (1, -1); (-1, 1); (-1, -1) ]

[<Fact>]
let ``rook classical: a1 on empty board`` () =
    let a1 = mkSquare 0 0
    let exp = (FileA ||| Rank1) &&& ~~~(bit a1)
    Assert.Equal(exp, rookAttacksClassical a1 0UL)
    Assert.Equal(14, popCount (rookAttacksClassical a1 0UL))

[<Fact>]
let ``bishop classical: a1 on empty board is the long diagonal`` () =
    let a1 = mkSquare 0 0
    Assert.Equal(7, popCount (bishopAttacksClassical a1 0UL))
    Assert.True(testBit (bishopAttacksClassical a1 0UL) (mkSquare 7 7)) // reaches h8

[<Fact>]
let ``rook classical stops at and includes the first blocker`` () =
    let a1 = mkSquare 0 0
    let d1 = mkSquare 3 0
    let att = rookAttacksClassical a1 (bit d1)
    Assert.True(testBit att d1) // capture square included
    Assert.False(testBit att (mkSquare 4 0)) // e1 is shadowed

[<Fact>]
let ``classical sliders match brute-force oracle over random occupancies`` () =
    let rng = System.Random(777)

    for sq in 0..63 do
        for _ in 0..199 do
            let occ = nextU64 rng
            let r = slowSlide sq occ rookDirs
            let b = slowSlide sq occ bishopDirs
            Assert.Equal(r, rookAttacksClassical sq occ)
            Assert.Equal(b, bishopAttacksClassical sq occ)
            Assert.Equal(r ||| b, queenAttacksClassical sq occ)

// ---------------------------------------------------------------------------
// Task 7 — leaper tables (pawn / knight / king)
// ---------------------------------------------------------------------------

let private leaperOracle (sq: int) (deltas: (int * int) list) : uint64 =
    let fa, ra = fileOf sq, rankOf sq
    let mutable acc = 0UL

    for (df, dr) in deltas do
        let f, r = fa + df, ra + dr

        if f >= 0 && f <= 7 && r >= 0 && r <= 7 then
            acc <- acc ||| bit (mkSquare f r)

    acc

let private knightDeltas =
    [ (-2, -1); (-2, 1); (-1, -2); (-1, 2); (1, -2); (1, 2); (2, -1); (2, 1) ]

let private kingDeltas =
    [ (-1, -1); (-1, 0); (-1, 1); (0, -1); (0, 1); (1, -1); (1, 0); (1, 1) ]

let private pawnOracle (c: int) (sq: int) : uint64 =
    let fa, ra = fileOf sq, rankOf sq
    let dr = if c = White then 1 else -1
    let mutable acc = 0UL

    for df in [ -1; 1 ] do
        let f, r = fa + df, ra + dr

        if f >= 0 && f <= 7 && r >= 0 && r <= 7 then
            acc <- acc ||| bit (mkSquare f r)

    acc

[<Fact>]
let ``knight a1 hits b3 and c2 only`` () =
    let a = knightAttacks (mkSquare 0 0)
    Assert.True(testBit a (mkSquare 1 2)) // b3
    Assert.True(testBit a (mkSquare 2 1)) // c2
    Assert.Equal(2, popCount a)

[<Fact>]
let ``knight in centre has 8 moves`` () =
    Assert.Equal(8, popCount (knightAttacks (mkSquare 4 3))) // e4

[<Fact>]
let ``king e1 has 5 moves`` () =
    Assert.Equal(5, popCount (kingAttacks (mkSquare 4 0)))

[<Fact>]
let ``white and black pawn captures from e4`` () =
    let e4 = mkSquare 4 3
    let w = pawnAttacks White e4
    Assert.True(testBit w (mkSquare 3 4)) // d5
    Assert.True(testBit w (mkSquare 5 4)) // f5
    Assert.Equal(2, popCount w)
    let b = pawnAttacks Black e4
    Assert.True(testBit b (mkSquare 3 2)) // d3
    Assert.True(testBit b (mkSquare 5 2)) // f3
    Assert.Equal(2, popCount b)

[<Fact>]
let ``pawn captures do not wrap off the a/h files`` () =
    Assert.Equal(1, popCount (pawnAttacks White (mkSquare 0 1))) // a2 -> b3 only
    Assert.Equal(1, popCount (pawnAttacks White (mkSquare 7 1))) // h2 -> g3 only

[<Fact>]
let ``leaper tables match brute-force oracle for every square`` () =
    for sq in 0..63 do
        Assert.Equal(leaperOracle sq knightDeltas, knightAttacks sq)
        Assert.Equal(leaperOracle sq kingDeltas, kingAttacks sq)
        Assert.Equal(pawnOracle White sq, pawnAttacks White sq)
        Assert.Equal(pawnOracle Black sq, pawnAttacks Black sq)

// ---------------------------------------------------------------------------
// Task 8 — slider masks + carry-rippler subsets
// ---------------------------------------------------------------------------

[<Fact>]
let ``rook mask bit counts: corner 12, edge 11, interior 10`` () =
    Assert.Equal(12, popCount (rookMask (mkSquare 0 0))) // a1 corner
    Assert.Equal(11, popCount (rookMask (mkSquare 4 0))) // e1 edge
    Assert.Equal(10, popCount (rookMask (mkSquare 4 3))) // e4 interior

[<Fact>]
let ``bishop mask bit counts: corner 6, interior up to 9`` () =
    Assert.Equal(6, popCount (bishopMask (mkSquare 0 0))) // a1
    Assert.Equal(9, popCount (bishopMask (mkSquare 3 3))) // d4
    Assert.Equal(5, popCount (bishopMask (mkSquare 4 0))) // e1

[<Fact>]
let ``slider masks exclude the slider square and the board edges they terminate on`` () =
    let a1 = mkSquare 0 0
    Assert.False(testBit (rookMask a1) a1)
    Assert.False(testBit (rookMask a1) (mkSquare 0 7)) // a8 edge excluded
    Assert.False(testBit (rookMask a1) (mkSquare 7 0)) // h1 edge excluded
    Assert.False(testBit (bishopMask a1) (mkSquare 7 7)) // h8 edge excluded

[<Fact>]
let ``total slider table sizes equal canonical 102400 and 5248`` () =
    let mutable rookTotal = 0
    let mutable bishopTotal = 0

    for sq in 0..63 do
        rookTotal <- rookTotal + (1 <<< popCount (rookMask sq))
        bishopTotal <- bishopTotal + (1 <<< popCount (bishopMask sq))

    Assert.Equal(102400, rookTotal)
    Assert.Equal(5248, bishopTotal)

[<Fact>]
let ``carry-rippler enumerates exactly the subsets of a mask`` () =
    let subs = subsetsOf 0b101UL |> Array.sort
    Assert.Equal<uint64[]>([| 0UL; 1UL; 4UL; 5UL |], subs)
    // count and subset property for a real mask
    let mask = bishopMask (mkSquare 3 3)
    let all = subsetsOf mask
    Assert.Equal(1 <<< popCount mask, all.Length)
    Assert.True(all |> Array.forall (fun s -> (s &&& mask) = s)) // each is a subset
    Assert.Equal(all.Length, (all |> Array.distinct |> Array.length)) // all distinct

// ---------------------------------------------------------------------------
// Task 9 — magic bitboards equal the classical oracle
// ---------------------------------------------------------------------------

[<Fact>]
let ``magic sliders equal classical over random occupancies`` () =
    ensureMagicBuilt () // magic tables are lazily skipped on PEXT CPUs; force-build for the parity check
    let rng = System.Random(0xBEEF)

    for sq in 0..63 do
        for _ in 0..199 do
            let occ = nextU64 rng
            Assert.Equal(rookAttacksClassical sq occ, rookAttacksMagic sq occ)
            Assert.Equal(bishopAttacksClassical sq occ, bishopAttacksMagic sq occ)

[<Fact>]
let ``magic equals classical for every relevant subset on sample squares`` () =
    ensureMagicBuilt () // magic tables are lazily skipped on PEXT CPUs; force-build for the parity check
    for sq in [ mkSquare 0 0; mkSquare 3 3; mkSquare 7 7; mkSquare 4 0; mkSquare 7 3 ] do
        for occ in subsetsOf (rookMask sq) do
            Assert.Equal(rookAttacksClassical sq occ, rookAttacksMagic sq occ)

        for occ in subsetsOf (bishopMask sq) do
            Assert.Equal(bishopAttacksClassical sq occ, bishopAttacksMagic sq occ)

// ---------------------------------------------------------------------------
// Task 10 — BMI2 PEXT sliders equal the classical oracle (skipped without BMI2)
// ---------------------------------------------------------------------------

[<Fact>]
let ``pext sliders equal classical over random occupancies (if BMI2)`` () =
    if System.Runtime.Intrinsics.X86.Bmi2.X64.IsSupported then
        let rng = System.Random(0xFEED)

        for sq in 0..63 do
            for _ in 0..199 do
                let occ = nextU64 rng
                Assert.Equal(rookAttacksClassical sq occ, rookAttacksPext sq occ)
                Assert.Equal(bishopAttacksClassical sq occ, bishopAttacksPext sq occ)

[<Fact>]
let ``pext equals classical for every relevant subset on sample squares (if BMI2)`` () =
    if System.Runtime.Intrinsics.X86.Bmi2.X64.IsSupported then
        for sq in [ mkSquare 0 0; mkSquare 3 3; mkSquare 7 7; mkSquare 4 0 ] do
            for occ in subsetsOf (rookMask sq) do
                Assert.Equal(rookAttacksClassical sq occ, rookAttacksPext sq occ)

            for occ in subsetsOf (bishopMask sq) do
                Assert.Equal(bishopAttacksClassical sq occ, bishopAttacksPext sq occ)

// ---------------------------------------------------------------------------
// Task 11 — unified attack API + attacksFrom
// ---------------------------------------------------------------------------

[<Fact>]
let ``unified slider API equals classical over random occupancies`` () =
    let rng = System.Random(0x1234)

    for sq in 0..63 do
        for _ in 0..99 do
            let occ = nextU64 rng
            Assert.Equal(rookAttacksClassical sq occ, rookAttacks sq occ)
            Assert.Equal(bishopAttacksClassical sq occ, bishopAttacks sq occ)
            Assert.Equal(queenAttacksClassical sq occ, queenAttacks sq occ)

[<Fact>]
let ``attacksFrom dispatches by piece type`` () =
    let occ = 0xFF00UL
    let sq = mkSquare 4 3
    Assert.Equal(pawnAttacks White sq, attacksFrom Pawn White sq occ)
    Assert.Equal(knightAttacks sq, attacksFrom Knight White sq occ)
    Assert.Equal(bishopAttacks sq occ, attacksFrom Bishop White sq occ)
    Assert.Equal(rookAttacks sq occ, attacksFrom Rook White sq occ)
    Assert.Equal(queenAttacks sq occ, attacksFrom Queen White sq occ)
    Assert.Equal(kingAttacks sq, attacksFrom King White sq occ)

[<Fact>]
let ``init reports the active slider backend`` () =
    let banner = init ()
    Assert.Contains("Eonego.Bitboard", banner)
    Assert.True(banner.Contains("PEXT") || banner.Contains("magic"))

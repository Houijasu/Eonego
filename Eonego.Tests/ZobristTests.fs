module Eonego.Tests.ZobristTests

open System.Collections.Generic
open Xunit
open Eonego.Bitboard
open Eonego.Zobrist

// ---------------------------------------------------------------------------
// CastlingKey is COMPOSED from four base right-keys, so the empty mask is 0.
// ---------------------------------------------------------------------------

[<Fact>]
let ``zCastle 0 is 0 (no rights -> no key term)`` () = Assert.Equal(0UL, zCastle 0)

// ---------------------------------------------------------------------------
// Linearity / telescoping: zCastle is linear over the mask bits, so
// zCastle a ^^^ zCastle b = zCastle (a ^^^ b). This is exactly what make relies
// on when it does `key ^^^ zCastle old ^^^ zCastle new`.
// ---------------------------------------------------------------------------

[<Fact>]
let ``zCastle is linear over rights bits (telescopes for any subset change)`` () =
    for a in 0..15 do
        for b in 0..15 do
            Assert.Equal(zCastle (a ^^^ b), zCastle a ^^^ zCastle b)

// ---------------------------------------------------------------------------
// Every key-table entry is nonzero and globally distinct (Psq 768 + EpFile 8 +
// Side 1 + CastlingKey[1..15] 15 = 792 values). Catches a zeroed/duplicated slot.
// ---------------------------------------------------------------------------

[<Fact>]
let ``all key entries are nonzero and pairwise distinct`` () =
    let seen = HashSet<uint64>()

    let add (v: uint64) =
        Assert.NotEqual(0UL, v)
        Assert.True(seen.Add v, "duplicate Zobrist key detected")

    add Side

    for pc in 0 .. NoPiece - 1 do
        for sq in 0..63 do
            add (zPiece pc sq)

    for f in 0..7 do
        add (zEp f)

    for m in 1..15 do
        add (zCastle m)

    Assert.Equal(1 + 12 * 64 + 8 + 15, seen.Count)

// ---------------------------------------------------------------------------
// Pinned literals: a fixed-seed PRNG with a fixed DRAW ORDER yields these exact
// values. Reordering or inserting a draw shifts the sequence and fails here.
// (Computed once from seed 0x9E3779B97F4A7C15UL with the documented draw order.)
// ---------------------------------------------------------------------------

[<Fact>]
let ``pinned literal keys catch a PRNG draw-order change`` () =
    Assert.Equal(0xDC1B77AE0BF34DADUL, Side)
    Assert.Equal(0x64F0EEB9026E6076UL, zPiece (makePiece White Pawn) (mkSquare 0 0)) // Psq[0]
    Assert.Equal(0xA79E285C96C7F8CBUL, zEp 0) // EpFile[0]
    Assert.Equal(0x338F8D2901A903ABUL, zCastle WK) // base wk key

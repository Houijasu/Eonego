module Eonego.Tests.NnueInputSymmetryTests

open Xunit
open Eonego.Bitboard
open Eonego.Position
open Eonego.NnueNetwork
open Eonego.NnueRegions
open Eonego.Nnue
open Eonego.Tests.TestFixtures

// ---------------------------------------------------------------------------
// Tests the NNUE FEATURE ASSEMBLER (assembleInput): the 2448 region copy, the STM bit, the 128 king
// one-hot, and the zero padding. NOTE: eval-antisymmetry (eval(p) == -eval(mirror p)) is NOT an invariant
// of this single-accumulator design (no us/them perspectives), so we test the assembler's structure +
// genuine mirror relationships instead, which is the code this PR actually ships.
// ---------------------------------------------------------------------------

let private stmIndex = AccSize // 2448
let private wKingBase = stmIndex + 1 // 2449
let private bKingBase = wKingBase + 64 // 2513

let private enabled (fen: string) : Position =
    let p = Position()
    p.LoadFen fen
    p.EnableNnue true
    p

[<Fact>]
let ``assembled input lays out accumulator, STM, king one-hots, and zero padding`` () =
    for fen in perftFens do
        let p = enabled fen
        let inp = assembleInput p
        Assert.Equal(PaddedL1, inp.Length)
        // region part == accumulator (byte cast)
        let acc = p.NnueAccumulator
        for i in 0 .. AccSize - 1 do
            Assert.Equal(byte acc.[i], inp.[i])
        // STM bit
        Assert.Equal(byte p.SideToMove, inp.[stmIndex])
        // each king block holds exactly one 1, at the king's square
        Assert.Equal(1uy, inp.[wKingBase + p.KingSquare White])
        Assert.Equal(1, Array.sumBy int inp.[wKingBase .. wKingBase + 63])
        Assert.Equal(1uy, inp.[bKingBase + p.KingSquare Black])
        Assert.Equal(1, Array.sumBy int inp.[bKingBase .. bKingBase + 63])
        // padding zeroed
        for i in InputSize .. PaddedL1 - 1 do
            Assert.Equal(0uy, inp.[i])

[<Fact>]
let ``feature assembly is consistent under the board mirror`` () =
    for fen in perftFens do
        let p = enabled fen
        let m = enabled (mirrorFen (p.ToFen()))
        let ip = assembleInput p
        let im = assembleInput m
        // total region count is mirror-invariant (same pieces; mirrored squares have equal region coverage)
        let sumRegions (a: byte[]) = Array.sumBy int a.[0 .. AccSize - 1]
        Assert.Equal(sumRegions ip, sumRegions im)
        // mirrorFen keeps the side to move
        Assert.Equal(ip.[stmIndex], im.[stmIndex])
        // kings swap color and flip vertically: white king of mirror == (black king of p) ^ 56, and vice versa
        Assert.Equal(1uy, im.[wKingBase + ((p.KingSquare Black) ^^^ 56)])
        Assert.Equal(1uy, im.[bKingBase + ((p.KingSquare White) ^^^ 56)])

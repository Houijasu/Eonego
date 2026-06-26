/// FullThreats feature-enumeration invariants. Without a real-Stockfish reference we can't bit-exact-verify
/// the threat indices, but we pin the structural invariants: every active index is in [0, Dimensions), the
/// active count never exceeds SF's MaxActiveDimensions (128), and a normal middlegame has a non-empty,
/// perspective-dependent threat set. (The material-accurate evals in NnueTests are the broader evidence.)
module Eonego.Tests.ThreatsTests

open Xunit
open Eonego.Bitboard
open Eonego.Position
open Eonego.Threats

let private fens =
    [ "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" // startpos
      "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1" // kiwipete
      "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1" // endgame
      "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3" ] // en-passant available

[<Fact>]
let ``active threat indices are in range and bounded by 128`` () =
    let buf = Array.zeroCreate 256

    for fen in fens do
        let pos = Position.OfFen fen

        for persp in [ White; Black ] do
            let n = appendActiveThreats persp pos buf
            Assert.True(n <= 128, sprintf "%s persp=%d: %d active features > 128" fen persp n)

            for i in 0 .. n - 1 do
                Assert.True(buf.[i] >= 0 && buf.[i] < Dimensions, sprintf "index %d out of [0,%d)" buf.[i] Dimensions)

[<Fact>]
let ``startpos has a non-empty threat set for both sides`` () =
    let buf = Array.zeroCreate 256
    let pos = Position.OfFen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    // Minor/major pieces defend their own pawns (e.g. Nb1->d2, Ra1->a2, Qd1->{c2,d2,e2}) ⇒ several threats.
    Assert.True(appendActiveThreats White pos buf > 4, "white startpos threat set too small")
    Assert.True(appendActiveThreats Black pos buf > 4, "black startpos threat set too small")

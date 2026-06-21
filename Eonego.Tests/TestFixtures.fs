/// Shared test fixtures for the Eonego suite — compiled first so every test module can reuse them.
/// (The originals in PositionTests are `private`; this module exposes public equivalents plus a few
/// movegen-specific helpers without touching the verified PositionTests file.)
module Eonego.Tests.TestFixtures

#nowarn "9" // NativePtr.stackalloc in collectLegal

open System
open Microsoft.FSharp.NativeInterop
open Xunit
open Eonego.Bitboard
open Eonego.Move
open Eonego.Position
open Eonego.MoveGeneration

// ---------------------------------------------------------------------------
// Canonical perft FEN set (CPW positions 1-6). Same strings as PositionTests.perftFens.
// ---------------------------------------------------------------------------
let perftFens =
    [| "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
       "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -"
       "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -"
       "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq -"
       "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ -"
       "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - -" |]

// ---------------------------------------------------------------------------
// Full-state snapshot (public copy of PositionTests' Snap) for Make/Unmake round-trip checks.
// ---------------------------------------------------------------------------
type Snap =
    { Types: uint64 list
      Colors: uint64 list
      Board: int list
      Occ: uint64
      Key: uint64
      Castle: int
      Ep: int
      Rule50: int
      Stm: int }

let snap (p: Position) : Snap =
    { Types = [ for pt in 0..5 -> p.Pieces pt ]
      Colors = [ p.ColorBB White; p.ColorBB Black ]
      Board = [ for sq in 0..63 -> p.PieceOn sq ]
      Occ = p.Occupied
      Key = p.Key
      Castle = p.CastlingRights
      Ep = p.EpSquare
      Rule50 = p.Rule50
      Stm = p.SideToMove }

/// Make then Unmake must restore every byte; and after Make incremental key == from-scratch.
let assertRoundTrips (p: Position) (m: Move) =
    let before = snap p
    p.Make m
    Assert.Equal(p.RecomputeKey(), p.Key)
    p.Unmake m
    Assert.True((before = snap p), "Make/Unmake did not restore full state for " + toUci m)

// ---------------------------------------------------------------------------
// Movegen helper — collect the legal move list into an array (cold path; allocates).
// Uses a per-call stackalloc Span<Move> (never escapes), then copies out.
// ---------------------------------------------------------------------------
let collectLegal (pos: Position) : Move[] =
    let p = NativePtr.stackalloc<Move> MaxMoves
    let buf = Span<Move>(NativePtr.toVoidPtr p, MaxMoves)
    let n = generateLegal pos buf
    let r = ResizeArray<Move>(n)

    for i in 0 .. n - 1 do
        r.Add buf.[i]

    r.ToArray()

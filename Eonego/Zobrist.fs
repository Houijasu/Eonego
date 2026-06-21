/// Eonego — Zobrist hash key tables for the chess engine.
///
/// Position.make / unmake maintain the 64-bit position key INCREMENTALLY, so the per-(piece,square),
/// castling-rights, en-passant-file, and side-to-move key tables must exist before Position compiles.
/// This module owns them. It mirrors Bitboard.fs's rhythm: private `uint64[]` tables filled once at
/// module load via `do`, with `[<MethodImpl(AggressiveInlining)>]` ATTRIBUTE accessors over the private
/// arrays (NOT F# `let inline` — a source-inlined read of a private binding fails at cross-assembly
/// callers; the attribute compiles a normal method the JIT still inlines in-assembly).
///
/// KEY CONVENTION (the single contract all three producers — Position.LoadFen, Position.make, and
/// Position.RecomputeKey — MUST match bit-for-bit):
///   key  =  XOR over occupied squares of  zPiece board.[sq] sq
///        ^^^ zCastle rights                              (ALWAYS; CastlingKey.[0] = 0UL)
///        ^^^ (if EpSquare <> NoSquare then zEp (fileOf EpSquare) else 0UL)   (en-passant FILE only)
///        ^^^ (if sideToMove = Black then Side else 0UL)                      (side term IFF Black)
///
/// CastlingKey is COMPOSED from four independent right-keys (wk/wq/bk/bq), so any rights change is a
/// 2-XOR `key ^^^ zCastle old ^^^ zCastle new` that telescopes for any subset (and CastlingKey.[0] = 0).
/// EpFile is keyed by FILE (transposition equality depends only on the file, never the rank).
///
/// Determinism: a fixed-seed xorshift64 PRNG (a DIFFERENT seed than Bitboard's magic search) makes the
/// whole table reproducible. The DRAW ORDER below is load-bearing — reordering or inserting a draw shifts
/// every subsequent key. ZobristTests pins specific entries to literal values so an accidental reorder
/// fails the build.
module Eonego.Zobrist

open System.Diagnostics
open System.Runtime.CompilerServices
open Eonego.Bitboard

// xorshift64 with a fixed seed (distinct from Bitboard's magicSeed = 0x2545F4914F6CDD1D so the two
// generators are independent). Mutable module state, used ONLY during one-shot init.
let mutable private seed = 0x9E3779B97F4A7C15UL

let private nextRand () : uint64 =
    let mutable s = seed
    s <- s ^^^ (s <<< 13)
    s <- s ^^^ (s >>> 7)
    s <- s ^^^ (s <<< 17)
    seed <- s
    s

let private Psq: uint64[] = Array.zeroCreate (12 * 64) // [(p <<< 6) + sq], p in 0..11
let private CastlingKey: uint64[] = Array.zeroCreate 16 // by 4-bit rights mask; [0] = 0UL
let private EpFile: uint64[] = Array.zeroCreate 8 // by file 0..7

// PRNG DRAW-ORDER CONTRACT (do NOT reorder; pinned by ZobristTests):
//   draw #1 -> Side ; then Psq.[0..767] ; then EpFile.[0..7] ; then wk, wq, bk, bq (compose CastlingKey).
let Side: uint64 = nextRand () // draw #1

let private initZobrist () =
    for i in 0 .. Psq.Length - 1 do
        Psq.[i] <- nextRand ()

    for f in 0..7 do
        EpFile.[f] <- nextRand ()

    let wk = nextRand ()
    let wq = nextRand ()
    let bk = nextRand ()
    let bq = nextRand ()

    for m in 0..15 do
        let mutable k = 0UL

        if m &&& WK <> 0 then
            k <- k ^^^ wk

        if m &&& WQ <> 0 then
            k <- k ^^^ wq

        if m &&& BK <> 0 then
            k <- k ^^^ bk

        if m &&& BQ <> 0 then
            k <- k ^^^ bq

        CastlingKey.[m] <- k

do initZobrist ()

/// Piece-square key. PRE: pc in 0..11 (a real piece). NoPiece (12) would index past Psq — guard before
/// calling (matters once the hot path switches to Unsafe.Add, where this becomes silent corruption).
[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let zPiece (pc: Piece) (sq: Square) : uint64 =
    Debug.Assert((pc >= 0 && pc < NoPiece), "zPiece: NoPiece/out-of-range piece indexes past Psq")
    Psq.[(pc <<< 6) + sq]

/// Castling-rights key for a 4-bit rights mask (0..15). zCastle 0 = 0UL.
[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let zCastle (rightsMask: int) : uint64 = CastlingKey.[rightsMask]

/// En-passant key for a file 0..7. PRE: guard EpSquare <> NoSquare before calling (fileOf 64 = 0 would
/// silently key file a).
[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let zEp (file: int) : uint64 =
    Debug.Assert((file >= 0 && file < 8), "zEp: file out of range (guard EpSquare <> NoSquare before calling)")
    EpFile.[file]

// zSide is the literal `Side`.

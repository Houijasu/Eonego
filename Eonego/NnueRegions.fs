/// Eonego — NNUE region-feature table + incremental accumulator primitives.
///
/// This module owns the *structural* (Position-independent) half of the NNUE feature set: the fixed map
/// from a board square to the set of "regions" that contain it, and the activate/deactivate/move primitives
/// that maintain the 2448-entry int8 region accumulator. It MUST compile BEFORE Position.fs, because the
/// three board-mutation choke points (PutPiece/RemovePiece/MovePiece — the EVAL-HOOK sites) call these
/// primitives; the network + forward pass (which need Position) live in later modules. This is the same
/// compile-order discipline Zobrist.fs uses for the incremental hash key.
///
/// FEATURE DEFINITION:
///   A "region" is a k×k axis-aligned sub-square of the 8×8 board (k = 1..8). There are Σ (9-k)² = 204 of
///   them. Each region carries 12 channels (one per Piece value, since Piece = color*6 + pieceType already
///   ranges 0..11), holding the COUNT of that piece-type-and-color inside the region. So the accumulator is
///   204 × 12 = 2448 int8 counts. A piece on a square contributes +1 to its own channel in every region that
///   contains the square (≈60 regions for a central square, 8 for a corner).
///
/// LAYOUT: regions are numbered grouped by window size k, then row-major by top-left (rank, file):
///   regionIndex k topRank topFile = regionSizeOffset.[k] + topRank*(9-k) + topFile
/// where regionSizeOffset.[k] = Σ_{j<k} (9-j)² (a prefix sum; offset.[9] = 204 sentinel). The flat
/// accumulator index is regionIndex*12 + piece.
///
/// THREAD-SAFETY: `regionsForSquare` is filled once in a module-level `do` and read-only thereafter, exactly
/// like Zobrist's tables and Evaluation's PST — any number of search threads may call the primitives on
/// DISTINCT accumulator arrays with zero shared writable state.
module Eonego.NnueRegions

open System.Diagnostics
open System.Runtime.CompilerServices
open Eonego.Bitboard

[<Literal>]
let RegionCount = 204 // Σ_{k=1}^{8} (9-k)²

[<Literal>]
let Channels = 12 // 6 white piece-types + 6 black, indexed directly by Piece (0..11)

[<Literal>]
let AccSize = 2448 // RegionCount * Channels

// regionSizeOffset.[k] = number of regions with window size < k (k in 1..8); [0] unused, [9] = 204 sentinel.
// Built by prefix-summing (9-j)² so a hand-transcription slip can't desync it from regionIndex.
let private regionSizeOffset: int[] = Array.zeroCreate 10

// regionsForSquare.[sq] = the region indices whose k×k window contains square sq (jagged, piece-independent).
let private regionsForSquare: int[][] = Array.zeroCreate 64

/// Region index 0..203 for the k×k window (1<=k<=8) whose top-left corner is (topRank, topFile),
/// with 0 <= topRank, topFile <= 8-k.
[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let regionIndex (k: int) (topRank: int) (topFile: int) : int =
    regionSizeOffset.[k] + topRank * (9 - k) + topFile

/// Flat accumulator index for (region, piece). `pc` IS color*6+pieceType, so it is the channel directly.
[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let channelIndex (region: int) (pc: Piece) : int = region * Channels + pc

let private initRegions () =
    // 1. prefix sums: offset.[k] = Σ_{j=1}^{k-1} (9-j)².  offset.[1]=0 ... offset.[8]=203, offset.[9]=204.
    regionSizeOffset.[1] <- 0
    for k in 2..9 do
        let side = 9 - (k - 1)
        regionSizeOffset.[k] <- regionSizeOffset.[k - 1] + side * side
    Debug.Assert((regionSizeOffset.[9] = RegionCount), "NnueRegions: region count != 204")
    // 2. per-square coverage lists.
    for sq in 0..63 do
        let r = rankOf sq
        let f = fileOf sq
        let acc = ResizeArray<int>()
        for k in 1..8 do
            let trLo = max 0 (r - k + 1)
            let trHi = min r (8 - k)
            let tfLo = max 0 (f - k + 1)
            let tfHi = min f (8 - k)
            for tr in trLo..trHi do
                for tf in tfLo..tfHi do
                    acc.Add(regionIndex k tr tf)
        regionsForSquare.[sq] <- acc.ToArray()

do initRegions ()

/// The region indices that contain `sq` (read-only — DO NOT mutate the returned array).
[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let regionsAt (sq: Square) : int[] = regionsForSquare.[sq]

/// Add 1 to `pc`'s channel in every region containing `sq`.
[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let activate (acc: sbyte[]) (pc: Piece) (sq: Square) : unit =
    Debug.Assert((pc >= 0 && pc < Channels && sq >= 0 && sq < 64), "NnueRegions.activate: bad pc/sq")
    let regions = regionsForSquare.[sq]

    for i in 0 .. regions.Length - 1 do
        let idx = channelIndex regions.[i] pc
        acc.[idx] <- acc.[idx] + 1y

/// Subtract 1 from `pc`'s channel in every region containing `sq`.
[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let deactivate (acc: sbyte[]) (pc: Piece) (sq: Square) : unit =
    Debug.Assert((pc >= 0 && pc < Channels && sq >= 0 && sq < 64), "NnueRegions.deactivate: bad pc/sq")
    let regions = regionsForSquare.[sq]

    for i in 0 .. regions.Length - 1 do
        let idx = channelIndex regions.[i] pc
        acc.[idx] <- acc.[idx] - 1y

/// Move `pc` from `from` to `dst`: deactivate(from) + activate(dst) on the same channel.
[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let move (acc: sbyte[]) (pc: Piece) (from: Square) (dst: Square) : unit =
    deactivate acc pc from
    activate acc pc dst

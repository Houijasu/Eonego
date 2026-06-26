/// HalfKAv2_hm feature indexing — the single source of truth for the SF NNUE accumulator math.
/// Compiles BEFORE Position.fs (Position maintains the incremental accumulator and calls these); NNUE.fs
/// (after Position) reuses them in the from-scratch `buildAcc` oracle. PURE: no Position, no SfNetwork.
/// Eonego squares are LERF (a1=0) == Stockfish; Color White=0/Black=1; Piece 0..11.
module Eonego.Accumulator

open System.Runtime.CompilerServices
open System.Runtime.Intrinsics
open System.Runtime.Intrinsics.X86
open Eonego.Bitboard

[<Literal>]
let L1 = 1024 // FullThreats accumulator dim (was 1536 for SF16); HalfKA weights are now [feature][1024]

[<Literal>]
let PsqtBuckets = 8

/// AVX2 is used when the CPU supports it and EONEGO_FORCE_SCALAR isn't set (read once). Scalar path always
/// present (correctness oracle + non-AVX2 CPUs). Kernels take an explicit `useAvx2` param so tests can
/// compare both branches in one process; production call sites pass this value.
let UseAvx2 =
    Avx2.IsSupported && System.Environment.GetEnvironmentVariable("EONEGO_FORCE_SCALAR") <> "1"

let private psWhite = [| 0; 128; 256; 384; 512; 640; 64; 192; 320; 448; 576; 640 |]
let private psBlack = [| 64; 192; 320; 448; 576; 640; 0; 128; 256; 384; 512; 640 |]

let private whiteKingBuckets =
    [| 28; 29; 30; 31; 31; 30; 29; 28
       24; 25; 26; 27; 27; 26; 25; 24
       20; 21; 22; 23; 23; 22; 21; 20
       16; 17; 18; 19; 19; 18; 17; 16
       12; 13; 14; 15; 15; 14; 13; 12
       8; 9; 10; 11; 11; 10; 9; 8
       4; 5; 6; 7; 7; 6; 5; 4
       0; 1; 2; 3; 3; 2; 1; 0 |]
    |> Array.map (fun v -> v * 704)

/// Feature index for piece `pc` on `sq`, from `pColor`'s perspective whose king is on `ksq`.
[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let makeIndex (pColor: Color) (pc: int) (sq: int) (ksq: int) : int =
    let orient = (if (ksq % 8) < 4 then 7 else 0) ^^^ (if pColor = Black then 56 else 0)
    let kbBase = whiteKingBuckets.[if pColor = White then ksq else ksq ^^^ 56]
    let ps = if pColor = White then psWhite else psBlack
    (sq ^^^ orient) + ps.[pc] + kbBase

/// acc[j] += sign*ftWeights[idx*L1+j] (all j); psqt[b] += sign*ftPsqt[idx*PsqtBuckets+b]. AVX2 widens the
/// int16 weights to int32 (vpmovsxwd) and adds/subtracts 8 lanes at a time; bit-exact int32 arithmetic.
[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let addFeature (acc: int[]) (psqt: int[]) (ftWeights: int16[]) (ftPsqt: int[]) (idx: int) (sign: int) (useAvx2: bool) =
    let wb = idx * L1

    if useAvx2 && Avx2.IsSupported then
        let mutable j = 0
        if sign > 0 then
            while j < L1 do
                let w = Avx2.ConvertToVector256Int32((Vector128.LoadUnsafe(&ftWeights.[wb + j]) : Vector128<int16>))
                Vector256.StoreUnsafe(Avx2.Add(Vector256.LoadUnsafe(&acc.[j]), w), &acc.[j])
                j <- j + 8
        else
            while j < L1 do
                let w = Avx2.ConvertToVector256Int32((Vector128.LoadUnsafe(&ftWeights.[wb + j]) : Vector128<int16>))
                Vector256.StoreUnsafe(Avx2.Subtract(Vector256.LoadUnsafe(&acc.[j]), w), &acc.[j])
                j <- j + 8
    else
        for j in 0 .. L1 - 1 do
            acc.[j] <- acc.[j] + sign * int ftWeights.[wb + j]

    let pb = idx * PsqtBuckets
    for b in 0 .. PsqtBuckets - 1 do
        psqt.[b] <- psqt.[b] + sign * ftPsqt.[pb + b]

/// As addFeature but for a FullThreats feature (int8 weights, int32 psqt). Updates a threat-only accumulator.
[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let addThreat (acc: int[]) (psqt: int[]) (threatWeights: sbyte[]) (threatPsqt: int[]) (idx: int) (sign: int) =
    let wb = idx * L1

    if UseAvx2 && Avx2.IsSupported then
        // sign-extend int8 weights (vpmovsxbd) and add/subtract 8 int32 lanes at a time.
        let mutable j = 0

        if sign > 0 then
            while j < L1 do
                let w = Avx2.ConvertToVector256Int32((Vector128.LoadUnsafe(&threatWeights.[wb + j]): Vector128<sbyte>))
                Vector256.StoreUnsafe(Avx2.Add(Vector256.LoadUnsafe(&acc.[j]), w), &acc.[j])
                j <- j + 8
        else
            while j < L1 do
                let w = Avx2.ConvertToVector256Int32((Vector128.LoadUnsafe(&threatWeights.[wb + j]): Vector128<sbyte>))
                Vector256.StoreUnsafe(Avx2.Subtract(Vector256.LoadUnsafe(&acc.[j]), w), &acc.[j])
                j <- j + 8
    else
        for j in 0 .. L1 - 1 do
            acc.[j] <- acc.[j] + sign * int threatWeights.[wb + j]

    let pb = idx * PsqtBuckets

    for b in 0 .. PsqtBuckets - 1 do
        psqt.[b] <- psqt.[b] + sign * threatPsqt.[pb + b]

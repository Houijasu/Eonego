/// HalfKAv2_hm feature indexing — the single source of truth for the NNUE accumulator math.
/// Compiles BEFORE Position.fs (Position maintains the incremental accumulator and calls these); NNUE.fs
/// (after Position) reuses them in the from-scratch `buildAcc` oracle. PURE: no Position, no Network.
/// Eonego squares are LERF (a1=0); Color White=0/Black=1; Piece 0..11.
module Eonego.Accumulator

open System.Runtime.CompilerServices
open System.Runtime.InteropServices
open System.Runtime.Intrinsics
open System.Runtime.Intrinsics.X86
open Eonego.Bitboard

[<Literal>]
let L1 = 1024 // FullThreats accumulator dim (was 1536 for the prior net); HalfKA weights are now [feature][1024]

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

// Shared dirty-frame payloads for Position's lazy NNUE stack. Dirty threats are physical edges in Eonego
// piece encoding; Threats.fs converts them to perspective-dependent feature indices at apply time.
[<Literal>]
let MaxDirtyPieces = 8

[<Literal>]
let MaxDirtyThreats = 512

[<Literal>]
let private DirtyThreatSignBit = 1 <<< 20

[<Literal>]
let private DirtyThreatEdgeMask = DirtyThreatSignBit - 1

[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let packDirtyThreatEdge (attacker: int) (from: int) (too: int) (attacked: int) : int =
    attacker ||| (attacked <<< 4) ||| (from <<< 8) ||| (too <<< 14)

[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let packSignedDirtyThreat (edge: int) (sign: int) : int =
    (edge &&& DirtyThreatEdgeMask) ||| (if sign > 0 then DirtyThreatSignBit else 0)

[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let dirtyThreatEdge (signedEdge: int) : int = signedEdge &&& DirtyThreatEdgeMask

[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let dirtyThreatSign (signedEdge: int) : int = if (signedEdge &&& DirtyThreatSignBit) <> 0 then 1 else -1

[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let dirtyThreatAttacker (edge: int) : int = edge &&& 0xF

[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let dirtyThreatAttacked (edge: int) : int = (edge >>> 4) &&& 0xF

[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let dirtyThreatFrom (edge: int) : int = (edge >>> 8) &&& 0x3F

[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let dirtyThreatTo (edge: int) : int = (edge >>> 14) &&& 0x3F

/// acc[accOff+j] += sign*ftWeights[idx*L1+j] (all j); psqt[psqtOff+b] += sign*ftPsqt[idx*PsqtBuckets+b].
/// int16 accumulator: HalfKA weights are already int16, so AVX2 adds/subtracts 16 lanes directly (no widen).
/// PSQT stays int32. The int16 sum wraps on two's-complement overflow identically in scalar and SIMD, but the
/// trained net keeps it in range (gated by the incremental==int32-oracle overflow test).
[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let addFeatureAt
    (acc: int16[])
    (accOff: int)
    (psqt: int[])
    (psqtOff: int)
    (ftWeights: int16[])
    (ftPsqt: int[])
    (idx: int)
    (sign: int)
    (useAvx2: bool)
    =
    let wb = idx * L1

    if useAvx2 then
        // Base-ref + element-offset loads/stores skip the per-iteration array bounds checks the JIT can't elide
        // when accOff/wb are runtime values. 16 int16 lanes/iter (vpaddw/vpsubw); bit-identical to scalar.
        let accBase = &MemoryMarshal.GetArrayDataReference acc
        let wBase = &MemoryMarshal.GetArrayDataReference ftWeights
        let mutable j = 0
        if sign > 0 then
            while j < L1 do
                let w = Vector256.LoadUnsafe(&wBase, unativeint (wb + j))
                let cur = Vector256.LoadUnsafe(&accBase, unativeint (accOff + j))
                Vector256.StoreUnsafe(Avx2.Add(cur, w), &accBase, unativeint (accOff + j))
                j <- j + 16
        else
            while j < L1 do
                let w = Vector256.LoadUnsafe(&wBase, unativeint (wb + j))
                let cur = Vector256.LoadUnsafe(&accBase, unativeint (accOff + j))
                Vector256.StoreUnsafe(Avx2.Subtract(cur, w), &accBase, unativeint (accOff + j))
                j <- j + 16
    else
        for j in 0 .. L1 - 1 do
            let k = accOff + j
            acc.[k] <- acc.[k] + int16 (sign * int ftWeights.[wb + j])

    let pb = idx * PsqtBuckets
    for b in 0 .. PsqtBuckets - 1 do
        let k = psqtOff + b
        psqt.[k] <- psqt.[k] + sign * ftPsqt.[pb + b]

/// acc += ftWeights[addIdx] - ftWeights[subIdx]. Common real-move case: same piece moves from one square to
/// another, so applying the remove+add as one vector pass halves accumulator load/store traffic.
[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let addFeaturePairAt
    (acc: int16[])
    (accOff: int)
    (psqt: int[])
    (psqtOff: int)
    (ftWeights: int16[])
    (ftPsqt: int[])
    (subIdx: int)
    (addIdx: int)
    (useAvx2: bool)
    =
    let subBase = subIdx * L1
    let addBase = addIdx * L1

    if useAvx2 then
        let accBase = &MemoryMarshal.GetArrayDataReference acc
        let wBase = &MemoryMarshal.GetArrayDataReference ftWeights
        let mutable j = 0

        while j < L1 do
            let cur = Vector256.LoadUnsafe(&accBase, unativeint (accOff + j))
            let addv = Vector256.LoadUnsafe(&wBase, unativeint (addBase + j))
            let subv = Vector256.LoadUnsafe(&wBase, unativeint (subBase + j))
            Vector256.StoreUnsafe(Avx2.Subtract(Avx2.Add(cur, addv), subv), &accBase, unativeint (accOff + j))
            j <- j + 16
    else
        for j in 0 .. L1 - 1 do
            let k = accOff + j
            acc.[k] <- acc.[k] + ftWeights.[addBase + j] - ftWeights.[subBase + j]

    let subPsq = subIdx * PsqtBuckets
    let addPsq = addIdx * PsqtBuckets

    for b in 0 .. PsqtBuckets - 1 do
        let k = psqtOff + b
        psqt.[k] <- psqt.[k] + ftPsqt.[addPsq + b] - ftPsqt.[subPsq + b]

/// Zero-offset compatibility wrapper.
[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let addFeature (acc: int16[]) (psqt: int[]) (ftWeights: int16[]) (ftPsqt: int[]) (idx: int) (sign: int) (useAvx2: bool) =
    addFeatureAt acc 0 psqt 0 ftWeights ftPsqt idx sign useAvx2

/// As addFeatureAt but for a FullThreats feature (int8 weights -> int16 accumulator, int32 psqt).
[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let addThreatAt
    (acc: int16[])
    (accOff: int)
    (psqt: int[])
    (psqtOff: int)
    (threatWeights: sbyte[])
    (threatPsqt: int[])
    (idx: int)
    (sign: int)
    (useAvx2: bool)
    =
    let wb = idx * L1

    if useAvx2 then
        // sign-extend int8 weights to int16 (vpmovsxbw) and add/subtract 16 int16 lanes at a time; base-ref +
        // offset loads/stores skip the per-iteration bounds checks (bit-identical to the scalar form).
        let accBase = &MemoryMarshal.GetArrayDataReference acc
        let wBase = &MemoryMarshal.GetArrayDataReference threatWeights
        let mutable j = 0

        if sign > 0 then
            while j < L1 do
                let w = Avx2.ConvertToVector256Int16(Vector128.LoadUnsafe(&wBase, unativeint (wb + j)))
                let cur = Vector256.LoadUnsafe(&accBase, unativeint (accOff + j))
                Vector256.StoreUnsafe(Avx2.Add(cur, w), &accBase, unativeint (accOff + j))
                j <- j + 16
        else
            while j < L1 do
                let w = Avx2.ConvertToVector256Int16(Vector128.LoadUnsafe(&wBase, unativeint (wb + j)))
                let cur = Vector256.LoadUnsafe(&accBase, unativeint (accOff + j))
                Vector256.StoreUnsafe(Avx2.Subtract(cur, w), &accBase, unativeint (accOff + j))
                j <- j + 16
    else
        for j in 0 .. L1 - 1 do
            let k = accOff + j
            acc.[k] <- acc.[k] + int16 (sign * int threatWeights.[wb + j])

    let pb = idx * PsqtBuckets

    for b in 0 .. PsqtBuckets - 1 do
        let k = psqtOff + b
        psqt.[k] <- psqt.[k] + sign * threatPsqt.[pb + b]

/// acc += threatWeights[addIdx] - threatWeights[subIdx]. Pairs one removed and one added FullThreats row
/// into a single accumulator pass, avoiding a second load/store of the destination accumulator.
[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let addThreatPairAt
    (acc: int16[])
    (accOff: int)
    (psqt: int[])
    (psqtOff: int)
    (threatWeights: sbyte[])
    (threatPsqt: int[])
    (subIdx: int)
    (addIdx: int)
    (useAvx2: bool)
    =
    let subBase = subIdx * L1
    let addBase = addIdx * L1

    if useAvx2 then
        let accBase = &MemoryMarshal.GetArrayDataReference acc
        let wBase = &MemoryMarshal.GetArrayDataReference threatWeights
        let mutable j = 0

        while j < L1 do
            let cur = Vector256.LoadUnsafe(&accBase, unativeint (accOff + j))
            let addv = Avx2.ConvertToVector256Int16(Vector128.LoadUnsafe(&wBase, unativeint (addBase + j)))
            let subv = Avx2.ConvertToVector256Int16(Vector128.LoadUnsafe(&wBase, unativeint (subBase + j)))
            Vector256.StoreUnsafe(Avx2.Subtract(Avx2.Add(cur, addv), subv), &accBase, unativeint (accOff + j))
            j <- j + 16
    else
        for j in 0 .. L1 - 1 do
            let k = accOff + j
            acc.[k] <- acc.[k] + int16 (int threatWeights.[addBase + j] - int threatWeights.[subBase + j])

    let subPsq = subIdx * PsqtBuckets
    let addPsq = addIdx * PsqtBuckets

    for b in 0 .. PsqtBuckets - 1 do
        let k = psqtOff + b
        psqt.[k] <- psqt.[k] + threatPsqt.[addPsq + b] - threatPsqt.[subPsq + b]

/// acc += (add0 - sub0) + (add1 - sub1) for two FullThreats row pairs. This keeps the destination
/// accumulator hot for one pass when a frame has multiple balanced threat changes.
[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let addThreatPair2At
    (acc: int16[])
    (accOff: int)
    (psqt: int[])
    (psqtOff: int)
    (threatWeights: sbyte[])
    (threatPsqt: int[])
    (subIdx0: int)
    (addIdx0: int)
    (subIdx1: int)
    (addIdx1: int)
    (useAvx2: bool)
    =
    let subBase0 = subIdx0 * L1
    let addBase0 = addIdx0 * L1
    let subBase1 = subIdx1 * L1
    let addBase1 = addIdx1 * L1

    if useAvx2 then
        let accBase = &MemoryMarshal.GetArrayDataReference acc
        let wBase = &MemoryMarshal.GetArrayDataReference threatWeights
        let mutable j = 0

        while j < L1 do
            let cur = Vector256.LoadUnsafe(&accBase, unativeint (accOff + j))
            let add0 = Avx2.ConvertToVector256Int16(Vector128.LoadUnsafe(&wBase, unativeint (addBase0 + j)))
            let sub0 = Avx2.ConvertToVector256Int16(Vector128.LoadUnsafe(&wBase, unativeint (subBase0 + j)))
            let add1 = Avx2.ConvertToVector256Int16(Vector128.LoadUnsafe(&wBase, unativeint (addBase1 + j)))
            let sub1 = Avx2.ConvertToVector256Int16(Vector128.LoadUnsafe(&wBase, unativeint (subBase1 + j)))
            let delta = Avx2.Subtract(Avx2.Add(add0, add1), Avx2.Add(sub0, sub1))
            Vector256.StoreUnsafe(Avx2.Add(cur, delta), &accBase, unativeint (accOff + j))
            j <- j + 16
    else
        for j in 0 .. L1 - 1 do
            let k = accOff + j
            acc.[k] <-
                acc.[k]
                + int16 (
                    int threatWeights.[addBase0 + j]
                    - int threatWeights.[subBase0 + j]
                    + int threatWeights.[addBase1 + j]
                    - int threatWeights.[subBase1 + j]
                )

    let subPsq0 = subIdx0 * PsqtBuckets
    let addPsq0 = addIdx0 * PsqtBuckets
    let subPsq1 = subIdx1 * PsqtBuckets
    let addPsq1 = addIdx1 * PsqtBuckets

    for b in 0 .. PsqtBuckets - 1 do
        let k = psqtOff + b

        psqt.[k] <-
            psqt.[k]
            + threatPsqt.[addPsq0 + b]
            - threatPsqt.[subPsq0 + b]
            + threatPsqt.[addPsq1 + b]
            - threatPsqt.[subPsq1 + b]

/// Zero-offset compatibility wrapper.
[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let addThreat (acc: int16[]) (psqt: int[]) (threatWeights: sbyte[]) (threatPsqt: int[]) (idx: int) (sign: int) =
    addThreatAt acc 0 psqt 0 threatWeights threatPsqt idx sign UseAvx2

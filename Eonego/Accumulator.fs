/// HalfKAv2_hm feature indexing — the single source of truth for the NNUE accumulator math.
/// Compiles BEFORE Position.fs (Position maintains the incremental accumulator and calls these); NNUE.fs
/// (after Position) reuses them in the from-scratch `buildAcc` oracle. PURE: no Position, no Network.
/// Eonego squares are LERF (a1=0); Color White=0/Black=1; Piece 0..11.
module Eonego.Accumulator

#nowarn "9" // NativePtr in the weight-row prefetch (address-of only; prefetch never faults)

open System.Runtime.CompilerServices
open System.Runtime.InteropServices
open System.Runtime.Intrinsics
open System.Runtime.Intrinsics.X86
open Microsoft.FSharp.NativeInterop
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

/// AVX-512 is used when the CPU supports it, scalar isn't forced, and the AVX-512 opt-out isn't set.
let UseAvx512 =
    Avx512F.IsSupported
    && Avx512BW.IsSupported
    && System.Environment.GetEnvironmentVariable("EONEGO_FORCE_SCALAR") <> "1"
    && System.Environment.GetEnvironmentVariable("EONEGO_FORCE_NOAVX512") <> "1"

/// Weight-row software prefetch: each row's leading 256 B into L1 before the tile sweep (mode 1 from
/// the old A/B knob). Semantics-free — node counts byte-identical with prefetch off.

/// Pinned (POH) allocation whose usable region starts 64-byte aligned: returns the array plus the element
/// offset of the aligned region. Managed array bases land at arbitrary 8B residues (measured 16/24/32/40/56
/// mod 64 on this runtime), which makes ~50% of the 32B accumulator/weight-row vector ops split cache lines;
/// pinning makes the address — and therefore the alignment — stable for the array's lifetime.
let allocAligned64<'T when 'T: unmanaged> (len: int) : struct ('T[] * int) =
    let arr = System.GC.AllocateArray<'T>(len + 64 / sizeof<'T>, true)

    let h =
        System.Runtime.InteropServices.GCHandle.Alloc(arr, System.Runtime.InteropServices.GCHandleType.Pinned)

    let addr = h.AddrOfPinnedObject().ToInt64()
    h.Free()
    let offBytes = int ((64L - addr % 64L) % 64L)
    struct (arr, offBytes / sizeof<'T>)

/// Copy `src` into a fresh 64B-aligned pinned array (see allocAligned64); returns (array, element offset).
let copyAligned64<'T when 'T: unmanaged> (src: 'T[]) : struct ('T[] * int) =
    let struct (arr, off) = allocAligned64<'T> src.Length
    System.Array.Copy(src, 0, arr, off, src.Length)
    struct (arr, off)

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

// Right-sized 512 -> 128 (2026-07-02): measured high-water mark is 30 physical edges/move over 950k makes
// (midgame depth-15 + kiwipete depth-14, EONEGO_PROF maxThreatN); the reference engine bounds its equivalent
// list at 96. Overflow degrades gracefully (dirtyThreatOverflow -> full-refresh fallback in CommitFrame).
// Per-frame payload cost is 3 * AccMaxPly * MaxDirtyThreats * 4 B, so this is 384 KB/Position instead of 1.5 MB.
[<Literal>]
let MaxDirtyThreats = 128

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
    (ftWOff: int)
    (ftPsqt: int[])
    (idx: int)
    (sign: int)
    (useAvx512: bool)
    (useAvx2: bool)
    =
    let wb = ftWOff + idx * L1

    if useAvx512 then
        let accBase = &MemoryMarshal.GetArrayDataReference acc
        let wBase = &MemoryMarshal.GetArrayDataReference ftWeights
        let mutable j = 0

        if sign > 0 then
            while j < L1 do
                let w0 = Vector512.LoadUnsafe(&wBase, unativeint (wb + j))
                let w1 = Vector512.LoadUnsafe(&wBase, unativeint (wb + j + 32))
                let w2 = Vector512.LoadUnsafe(&wBase, unativeint (wb + j + 64))
                let w3 = Vector512.LoadUnsafe(&wBase, unativeint (wb + j + 96))
                let cur0 = Vector512.LoadUnsafe(&accBase, unativeint (accOff + j))
                let cur1 = Vector512.LoadUnsafe(&accBase, unativeint (accOff + j + 32))
                let cur2 = Vector512.LoadUnsafe(&accBase, unativeint (accOff + j + 64))
                let cur3 = Vector512.LoadUnsafe(&accBase, unativeint (accOff + j + 96))
                Vector512.StoreUnsafe(cur0 + w0, &accBase, unativeint (accOff + j))
                Vector512.StoreUnsafe(cur1 + w1, &accBase, unativeint (accOff + j + 32))
                Vector512.StoreUnsafe(cur2 + w2, &accBase, unativeint (accOff + j + 64))
                Vector512.StoreUnsafe(cur3 + w3, &accBase, unativeint (accOff + j + 96))
                j <- j + 128
        else
            while j < L1 do
                let w0 = Vector512.LoadUnsafe(&wBase, unativeint (wb + j))
                let w1 = Vector512.LoadUnsafe(&wBase, unativeint (wb + j + 32))
                let w2 = Vector512.LoadUnsafe(&wBase, unativeint (wb + j + 64))
                let w3 = Vector512.LoadUnsafe(&wBase, unativeint (wb + j + 96))
                let cur0 = Vector512.LoadUnsafe(&accBase, unativeint (accOff + j))
                let cur1 = Vector512.LoadUnsafe(&accBase, unativeint (accOff + j + 32))
                let cur2 = Vector512.LoadUnsafe(&accBase, unativeint (accOff + j + 64))
                let cur3 = Vector512.LoadUnsafe(&accBase, unativeint (accOff + j + 96))
                Vector512.StoreUnsafe(cur0 - w0, &accBase, unativeint (accOff + j))
                Vector512.StoreUnsafe(cur1 - w1, &accBase, unativeint (accOff + j + 32))
                Vector512.StoreUnsafe(cur2 - w2, &accBase, unativeint (accOff + j + 64))
                Vector512.StoreUnsafe(cur3 - w3, &accBase, unativeint (accOff + j + 96))
                j <- j + 128
    elif useAvx2 then
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
    (useAvx512: bool)
    (useAvx2: bool)
    =
    let subBase = subIdx * L1
    let addBase = addIdx * L1

    if useAvx512 then
        let accBase = &MemoryMarshal.GetArrayDataReference acc
        let wBase = &MemoryMarshal.GetArrayDataReference ftWeights
        let mutable j = 0

        while j < L1 do
            let cur0 = Vector512.LoadUnsafe(&accBase, unativeint (accOff + j))
            let cur1 = Vector512.LoadUnsafe(&accBase, unativeint (accOff + j + 32))
            let cur2 = Vector512.LoadUnsafe(&accBase, unativeint (accOff + j + 64))
            let cur3 = Vector512.LoadUnsafe(&accBase, unativeint (accOff + j + 96))
            let add0 = Vector512.LoadUnsafe(&wBase, unativeint (addBase + j))
            let add1 = Vector512.LoadUnsafe(&wBase, unativeint (addBase + j + 32))
            let add2 = Vector512.LoadUnsafe(&wBase, unativeint (addBase + j + 64))
            let add3 = Vector512.LoadUnsafe(&wBase, unativeint (addBase + j + 96))
            let sub0 = Vector512.LoadUnsafe(&wBase, unativeint (subBase + j))
            let sub1 = Vector512.LoadUnsafe(&wBase, unativeint (subBase + j + 32))
            let sub2 = Vector512.LoadUnsafe(&wBase, unativeint (subBase + j + 64))
            let sub3 = Vector512.LoadUnsafe(&wBase, unativeint (subBase + j + 96))
            Vector512.StoreUnsafe(cur0 + add0 - sub0, &accBase, unativeint (accOff + j))
            Vector512.StoreUnsafe(cur1 + add1 - sub1, &accBase, unativeint (accOff + j + 32))
            Vector512.StoreUnsafe(cur2 + add2 - sub2, &accBase, unativeint (accOff + j + 64))
            Vector512.StoreUnsafe(cur3 + add3 - sub3, &accBase, unativeint (accOff + j + 96))
            j <- j + 128
    elif useAvx2 then
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
let addFeature (acc: int16[]) (psqt: int[]) (ftWeights: int16[]) (ftPsqt: int[]) (idx: int) (sign: int) (useAvx512: bool) (useAvx2: bool) =
    addFeatureAt acc 0 psqt 0 ftWeights 0 ftPsqt idx sign useAvx512 useAvx2

/// As addFeatureAt but for a FullThreats feature (int8 weights -> int16 accumulator, int32 psqt).
[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let addThreatAt
    (acc: int16[])
    (accOff: int)
    (psqt: int[])
    (psqtOff: int)
    (threatWeights: sbyte[])
    (thrWOff: int)
    (threatPsqt: int[])
    (idx: int)
    (sign: int)
    (useAvx512: bool)
    (useAvx2: bool)
    =
    let wb = thrWOff + idx * L1

    if useAvx512 then
        let accBase = &MemoryMarshal.GetArrayDataReference acc
        let wBase = &MemoryMarshal.GetArrayDataReference threatWeights
        let mutable j = 0

        if sign > 0 then
            while j < L1 do
                let w0 = Avx512BW.ConvertToVector512Int16(Vector256.LoadUnsafe(&wBase, unativeint (wb + j)))
                let w1 = Avx512BW.ConvertToVector512Int16(Vector256.LoadUnsafe(&wBase, unativeint (wb + j + 32)))
                let w2 = Avx512BW.ConvertToVector512Int16(Vector256.LoadUnsafe(&wBase, unativeint (wb + j + 64)))
                let w3 = Avx512BW.ConvertToVector512Int16(Vector256.LoadUnsafe(&wBase, unativeint (wb + j + 96)))
                let cur0 = Vector512.LoadUnsafe(&accBase, unativeint (accOff + j))
                let cur1 = Vector512.LoadUnsafe(&accBase, unativeint (accOff + j + 32))
                let cur2 = Vector512.LoadUnsafe(&accBase, unativeint (accOff + j + 64))
                let cur3 = Vector512.LoadUnsafe(&accBase, unativeint (accOff + j + 96))
                Vector512.StoreUnsafe(cur0 + w0, &accBase, unativeint (accOff + j))
                Vector512.StoreUnsafe(cur1 + w1, &accBase, unativeint (accOff + j + 32))
                Vector512.StoreUnsafe(cur2 + w2, &accBase, unativeint (accOff + j + 64))
                Vector512.StoreUnsafe(cur3 + w3, &accBase, unativeint (accOff + j + 96))
                j <- j + 128
        else
            while j < L1 do
                let w0 = Avx512BW.ConvertToVector512Int16(Vector256.LoadUnsafe(&wBase, unativeint (wb + j)))
                let w1 = Avx512BW.ConvertToVector512Int16(Vector256.LoadUnsafe(&wBase, unativeint (wb + j + 32)))
                let w2 = Avx512BW.ConvertToVector512Int16(Vector256.LoadUnsafe(&wBase, unativeint (wb + j + 64)))
                let w3 = Avx512BW.ConvertToVector512Int16(Vector256.LoadUnsafe(&wBase, unativeint (wb + j + 96)))
                let cur0 = Vector512.LoadUnsafe(&accBase, unativeint (accOff + j))
                let cur1 = Vector512.LoadUnsafe(&accBase, unativeint (accOff + j + 32))
                let cur2 = Vector512.LoadUnsafe(&accBase, unativeint (accOff + j + 64))
                let cur3 = Vector512.LoadUnsafe(&accBase, unativeint (accOff + j + 96))
                Vector512.StoreUnsafe(cur0 - w0, &accBase, unativeint (accOff + j))
                Vector512.StoreUnsafe(cur1 - w1, &accBase, unativeint (accOff + j + 32))
                Vector512.StoreUnsafe(cur2 - w2, &accBase, unativeint (accOff + j + 64))
                Vector512.StoreUnsafe(cur3 - w3, &accBase, unativeint (accOff + j + 96))
                j <- j + 128
    elif useAvx2 then
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
    (useAvx512: bool)
    (useAvx2: bool)
    =
    let subBase = subIdx * L1
    let addBase = addIdx * L1

    if useAvx512 then
        let accBase = &MemoryMarshal.GetArrayDataReference acc
        let wBase = &MemoryMarshal.GetArrayDataReference threatWeights
        let mutable j = 0

        while j < L1 do
            let cur0 = Vector512.LoadUnsafe(&accBase, unativeint (accOff + j))
            let cur1 = Vector512.LoadUnsafe(&accBase, unativeint (accOff + j + 32))
            let cur2 = Vector512.LoadUnsafe(&accBase, unativeint (accOff + j + 64))
            let cur3 = Vector512.LoadUnsafe(&accBase, unativeint (accOff + j + 96))
            let add0 = Avx512BW.ConvertToVector512Int16(Vector256.LoadUnsafe(&wBase, unativeint (addBase + j)))
            let add1 = Avx512BW.ConvertToVector512Int16(Vector256.LoadUnsafe(&wBase, unativeint (addBase + j + 32)))
            let add2 = Avx512BW.ConvertToVector512Int16(Vector256.LoadUnsafe(&wBase, unativeint (addBase + j + 64)))
            let add3 = Avx512BW.ConvertToVector512Int16(Vector256.LoadUnsafe(&wBase, unativeint (addBase + j + 96)))
            let sub0 = Avx512BW.ConvertToVector512Int16(Vector256.LoadUnsafe(&wBase, unativeint (subBase + j)))
            let sub1 = Avx512BW.ConvertToVector512Int16(Vector256.LoadUnsafe(&wBase, unativeint (subBase + j + 32)))
            let sub2 = Avx512BW.ConvertToVector512Int16(Vector256.LoadUnsafe(&wBase, unativeint (subBase + j + 64)))
            let sub3 = Avx512BW.ConvertToVector512Int16(Vector256.LoadUnsafe(&wBase, unativeint (subBase + j + 96)))
            Vector512.StoreUnsafe(cur0 + add0 - sub0, &accBase, unativeint (accOff + j))
            Vector512.StoreUnsafe(cur1 + add1 - sub1, &accBase, unativeint (accOff + j + 32))
            Vector512.StoreUnsafe(cur2 + add2 - sub2, &accBase, unativeint (accOff + j + 64))
            Vector512.StoreUnsafe(cur3 + add3 - sub3, &accBase, unativeint (accOff + j + 96))
            j <- j + 128
    elif useAvx2 then
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
    (useAvx512: bool)
    (useAvx2: bool)
    =
    let subBase0 = subIdx0 * L1
    let addBase0 = addIdx0 * L1
    let subBase1 = subIdx1 * L1
    let addBase1 = addIdx1 * L1

    if useAvx512 then
        let accBase = &MemoryMarshal.GetArrayDataReference acc
        let wBase = &MemoryMarshal.GetArrayDataReference threatWeights
        let mutable j = 0

        while j < L1 do
            let cur0 = Vector512.LoadUnsafe(&accBase, unativeint (accOff + j))
            let cur1 = Vector512.LoadUnsafe(&accBase, unativeint (accOff + j + 32))
            let cur2 = Vector512.LoadUnsafe(&accBase, unativeint (accOff + j + 64))
            let cur3 = Vector512.LoadUnsafe(&accBase, unativeint (accOff + j + 96))
            let add00 = Avx512BW.ConvertToVector512Int16(Vector256.LoadUnsafe(&wBase, unativeint (addBase0 + j)))
            let sub00 = Avx512BW.ConvertToVector512Int16(Vector256.LoadUnsafe(&wBase, unativeint (subBase0 + j)))
            let add10 = Avx512BW.ConvertToVector512Int16(Vector256.LoadUnsafe(&wBase, unativeint (addBase1 + j)))
            let sub10 = Avx512BW.ConvertToVector512Int16(Vector256.LoadUnsafe(&wBase, unativeint (subBase1 + j)))
            let add01 = Avx512BW.ConvertToVector512Int16(Vector256.LoadUnsafe(&wBase, unativeint (addBase0 + j + 32)))
            let sub01 = Avx512BW.ConvertToVector512Int16(Vector256.LoadUnsafe(&wBase, unativeint (subBase0 + j + 32)))
            let add11 = Avx512BW.ConvertToVector512Int16(Vector256.LoadUnsafe(&wBase, unativeint (addBase1 + j + 32)))
            let sub11 = Avx512BW.ConvertToVector512Int16(Vector256.LoadUnsafe(&wBase, unativeint (subBase1 + j + 32)))
            let add02 = Avx512BW.ConvertToVector512Int16(Vector256.LoadUnsafe(&wBase, unativeint (addBase0 + j + 64)))
            let sub02 = Avx512BW.ConvertToVector512Int16(Vector256.LoadUnsafe(&wBase, unativeint (subBase0 + j + 64)))
            let add12 = Avx512BW.ConvertToVector512Int16(Vector256.LoadUnsafe(&wBase, unativeint (addBase1 + j + 64)))
            let sub12 = Avx512BW.ConvertToVector512Int16(Vector256.LoadUnsafe(&wBase, unativeint (subBase1 + j + 64)))
            let add03 = Avx512BW.ConvertToVector512Int16(Vector256.LoadUnsafe(&wBase, unativeint (addBase0 + j + 96)))
            let sub03 = Avx512BW.ConvertToVector512Int16(Vector256.LoadUnsafe(&wBase, unativeint (subBase0 + j + 96)))
            let add13 = Avx512BW.ConvertToVector512Int16(Vector256.LoadUnsafe(&wBase, unativeint (addBase1 + j + 96)))
            let sub13 = Avx512BW.ConvertToVector512Int16(Vector256.LoadUnsafe(&wBase, unativeint (subBase1 + j + 96)))
            let delta0 = (add00 + add10) - (sub00 + sub10)
            let delta1 = (add01 + add11) - (sub01 + sub11)
            let delta2 = (add02 + add12) - (sub02 + sub12)
            let delta3 = (add03 + add13) - (sub03 + sub13)
            Vector512.StoreUnsafe(cur0 + delta0, &accBase, unativeint (accOff + j))
            Vector512.StoreUnsafe(cur1 + delta1, &accBase, unativeint (accOff + j + 32))
            Vector512.StoreUnsafe(cur2 + delta2, &accBase, unativeint (accOff + j + 64))
            Vector512.StoreUnsafe(cur3 + delta3, &accBase, unativeint (accOff + j + 96))
            j <- j + 128
    elif useAvx2 then
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
    addThreatAt acc 0 psqt 0 threatWeights 0 threatPsqt idx sign UseAvx512 UseAvx2

// ---------------------------------------------------------------------------
// Fused apply: ONE register-tiled pass over the accumulator applying ALL of a
// frame's row deltas (HalfKA int16 rows + FullThreats int8 rows, adds and
// subs), reading from a source slot and writing to a destination slot.
// Replaces the feature-outer kernels above on the hot walk path: a move with
// ~30 threat rows used to re-stream the full 2 KB accumulator ~8x; this
// streams it once (src -> registers -> dst) while weight rows stream through.
// src may equal dst exactly (in-place) or be a disjoint slot (parent->child
// frame materialization, finny-entry -> frame fold).
// ---------------------------------------------------------------------------

// NOTE (2026-07-02): a threat weight-row prefetch at index-gather time (Sse.Prefetch1 of the row head,
// issued in EnsureChangedThreats for both perspectives) was implemented and A/B-measured NEUTRAL at
// depth-14/15 benches (A 431-436k vs B 414-429k nps means, inside the thermal noise band) — the hot rows
// of the ~60 MB threat table fit L3 in that regime. Reverted; retry only with long-TC evidence.

/// Max weight rows folded into one accumulator pass. Larger row sets are split into multiple passes
/// (first pass src->dst, remainder in-place on dst): L2 hardware prefetchers track only ~16-32 streams,
/// and each pass adds just 4 KB of L1-resident accumulator traffic. Benchmark knob.
[<Literal>]
let FusedMaxRowsPerPass = 32

/// One fused pass over a row-list SLICE: dst[j] = src[j] + Σ halfW[halfAdd[k]] − Σ halfW[halfSub[k]]
/// + Σ thrW[thrAdd[k]] − Σ thrW[thrSub[k]] (j in [0,L1); psqt likewise over PsqtBuckets).
/// PRE: src and dst ranges are identical or fully disjoint (asserted by the caller-facing applyFused).
[<MethodImpl(MethodImplOptions.NoInlining)>]
let private applyFusedPass
    (srcAcc: int16[])
    (srcOff: int)
    (dstAcc: int16[])
    (dstOff: int)
    (srcPsq: int[])
    (srcPsqOff: int)
    (dstPsq: int[])
    (dstPsqOff: int)
    (halfW: int16[])
    (hwOff: int)
    (halfPsqt: int[])
    (thrW: sbyte[])
    (twOff: int)
    (thrPsqt: int[])
    (halfAdd: int[])
    (haOff: int)
    (haN: int)
    (halfSub: int[])
    (hsOff: int)
    (hsN: int)
    (thrAdd: int[])
    (taOff: int)
    (taN: int)
    (thrSub: int[])
    (tsOff: int)
    (tsN: int)
    (useAvx512: bool)
    (useAvx2: bool)
    =
    if useAvx512 then
        let srcBase = &MemoryMarshal.GetArrayDataReference srcAcc
        let dstBase = &MemoryMarshal.GetArrayDataReference dstAcc
        let hwBase = &MemoryMarshal.GetArrayDataReference halfW
        let twBase = &MemoryMarshal.GetArrayDataReference thrW

        let mutable t = 0

        while t < L1 do
            let mutable a0 = Vector512.LoadUnsafe(&srcBase, unativeint (srcOff + t))
            let mutable a1 = Vector512.LoadUnsafe(&srcBase, unativeint (srcOff + t + 32))
            let mutable a2 = Vector512.LoadUnsafe(&srcBase, unativeint (srcOff + t + 64))
            let mutable a3 = Vector512.LoadUnsafe(&srcBase, unativeint (srcOff + t + 96))

            let mutable k = 0

            while k < haN do
                let rb = hwOff + halfAdd.[haOff + k] * L1 + t
                a0 <- a0 + Vector512.LoadUnsafe(&hwBase, unativeint rb)
                a1 <- a1 + Vector512.LoadUnsafe(&hwBase, unativeint (rb + 32))
                a2 <- a2 + Vector512.LoadUnsafe(&hwBase, unativeint (rb + 64))
                a3 <- a3 + Vector512.LoadUnsafe(&hwBase, unativeint (rb + 96))
                k <- k + 1

            k <- 0

            while k < hsN do
                let rb = hwOff + halfSub.[hsOff + k] * L1 + t
                a0 <- a0 - Vector512.LoadUnsafe(&hwBase, unativeint rb)
                a1 <- a1 - Vector512.LoadUnsafe(&hwBase, unativeint (rb + 32))
                a2 <- a2 - Vector512.LoadUnsafe(&hwBase, unativeint (rb + 64))
                a3 <- a3 - Vector512.LoadUnsafe(&hwBase, unativeint (rb + 96))
                k <- k + 1

            k <- 0

            while k < taN do
                let rb = twOff + thrAdd.[taOff + k] * L1 + t
                a0 <- a0 + Avx512BW.ConvertToVector512Int16(Vector256.LoadUnsafe(&twBase, unativeint rb))
                a1 <- a1 + Avx512BW.ConvertToVector512Int16(Vector256.LoadUnsafe(&twBase, unativeint (rb + 32)))
                a2 <- a2 + Avx512BW.ConvertToVector512Int16(Vector256.LoadUnsafe(&twBase, unativeint (rb + 64)))
                a3 <- a3 + Avx512BW.ConvertToVector512Int16(Vector256.LoadUnsafe(&twBase, unativeint (rb + 96)))
                k <- k + 1

            k <- 0

            while k < tsN do
                let rb = twOff + thrSub.[tsOff + k] * L1 + t
                a0 <- a0 - Avx512BW.ConvertToVector512Int16(Vector256.LoadUnsafe(&twBase, unativeint rb))
                a1 <- a1 - Avx512BW.ConvertToVector512Int16(Vector256.LoadUnsafe(&twBase, unativeint (rb + 32)))
                a2 <- a2 - Avx512BW.ConvertToVector512Int16(Vector256.LoadUnsafe(&twBase, unativeint (rb + 64)))
                a3 <- a3 - Avx512BW.ConvertToVector512Int16(Vector256.LoadUnsafe(&twBase, unativeint (rb + 96)))
                k <- k + 1

            Vector512.StoreUnsafe(a0, &dstBase, unativeint (dstOff + t))
            Vector512.StoreUnsafe(a1, &dstBase, unativeint (dstOff + t + 32))
            Vector512.StoreUnsafe(a2, &dstBase, unativeint (dstOff + t + 64))
            Vector512.StoreUnsafe(a3, &dstBase, unativeint (dstOff + t + 96))
            t <- t + 128
    elif useAvx2 then
        // 4-ymm tile (64 int16) x 16 tiles; the tile lives in registers while every row's chunk streams
        // through (proven-safe register budget — mirrors the fc0 GEMV 4-accumulator blocking in NNUE.fs).
        // Base-ref + element-offset loads/stores skip the bounds checks the JIT can't elide on runtime
        // offsets; the small index lists use ordinary bounds-checked access (n <= 128, negligible).
        let srcBase = &MemoryMarshal.GetArrayDataReference srcAcc
        let dstBase = &MemoryMarshal.GetArrayDataReference dstAcc
        let hwBase = &MemoryMarshal.GetArrayDataReference halfW
        let twBase = &MemoryMarshal.GetArrayDataReference thrW
        let mutable t = 0

        while t < L1 do
            let mutable a0 = Vector256.LoadUnsafe(&srcBase, unativeint (srcOff + t))
            let mutable a1 = Vector256.LoadUnsafe(&srcBase, unativeint (srcOff + t + 16))
            let mutable a2 = Vector256.LoadUnsafe(&srcBase, unativeint (srcOff + t + 32))
            let mutable a3 = Vector256.LoadUnsafe(&srcBase, unativeint (srcOff + t + 48))

            let mutable k = 0

            while k < haN do
                let rb = hwOff + halfAdd.[haOff + k] * L1 + t
                a0 <- Avx2.Add(a0, Vector256.LoadUnsafe(&hwBase, unativeint rb))
                a1 <- Avx2.Add(a1, Vector256.LoadUnsafe(&hwBase, unativeint (rb + 16)))
                a2 <- Avx2.Add(a2, Vector256.LoadUnsafe(&hwBase, unativeint (rb + 32)))
                a3 <- Avx2.Add(a3, Vector256.LoadUnsafe(&hwBase, unativeint (rb + 48)))
                k <- k + 1

            k <- 0

            while k < hsN do
                let rb = hwOff + halfSub.[hsOff + k] * L1 + t
                a0 <- Avx2.Subtract(a0, Vector256.LoadUnsafe(&hwBase, unativeint rb))
                a1 <- Avx2.Subtract(a1, Vector256.LoadUnsafe(&hwBase, unativeint (rb + 16)))
                a2 <- Avx2.Subtract(a2, Vector256.LoadUnsafe(&hwBase, unativeint (rb + 32)))
                a3 <- Avx2.Subtract(a3, Vector256.LoadUnsafe(&hwBase, unativeint (rb + 48)))
                k <- k + 1

            k <- 0

            while k < taN do
                let rb = twOff + thrAdd.[taOff + k] * L1 + t
                a0 <- Avx2.Add(a0, Avx2.ConvertToVector256Int16(Vector128.LoadUnsafe(&twBase, unativeint rb)))
                a1 <- Avx2.Add(a1, Avx2.ConvertToVector256Int16(Vector128.LoadUnsafe(&twBase, unativeint (rb + 16))))
                a2 <- Avx2.Add(a2, Avx2.ConvertToVector256Int16(Vector128.LoadUnsafe(&twBase, unativeint (rb + 32))))
                a3 <- Avx2.Add(a3, Avx2.ConvertToVector256Int16(Vector128.LoadUnsafe(&twBase, unativeint (rb + 48))))
                k <- k + 1

            k <- 0

            while k < tsN do
                let rb = twOff + thrSub.[tsOff + k] * L1 + t
                a0 <- Avx2.Subtract(a0, Avx2.ConvertToVector256Int16(Vector128.LoadUnsafe(&twBase, unativeint rb)))
                a1 <- Avx2.Subtract(a1, Avx2.ConvertToVector256Int16(Vector128.LoadUnsafe(&twBase, unativeint (rb + 16))))
                a2 <- Avx2.Subtract(a2, Avx2.ConvertToVector256Int16(Vector128.LoadUnsafe(&twBase, unativeint (rb + 32))))
                a3 <- Avx2.Subtract(a3, Avx2.ConvertToVector256Int16(Vector128.LoadUnsafe(&twBase, unativeint (rb + 48))))
                k <- k + 1

            Vector256.StoreUnsafe(a0, &dstBase, unativeint (dstOff + t))
            Vector256.StoreUnsafe(a1, &dstBase, unativeint (dstOff + t + 16))
            Vector256.StoreUnsafe(a2, &dstBase, unativeint (dstOff + t + 32))
            Vector256.StoreUnsafe(a3, &dstBase, unativeint (dstOff + t + 48))
            t <- t + 64
    else
        // Scalar reference: int32 accumulate then truncate to int16 — bit-identical to sequential int16
        // wrapping adds (truncation mod 2^16 is a homomorphism over addition; |sum| << 2^31).
        for j in 0 .. L1 - 1 do
            let mutable s = int srcAcc.[srcOff + j]

            for k in 0 .. haN - 1 do
                s <- s + int halfW.[hwOff + halfAdd.[haOff + k] * L1 + j]

            for k in 0 .. hsN - 1 do
                s <- s - int halfW.[hwOff + halfSub.[hsOff + k] * L1 + j]

            for k in 0 .. taN - 1 do
                s <- s + int thrW.[twOff + thrAdd.[taOff + k] * L1 + j]

            for k in 0 .. tsN - 1 do
                s <- s - int thrW.[twOff + thrSub.[tsOff + k] * L1 + j]

            dstAcc.[dstOff + j] <- int16 s

    // PSQT: 8 int32 buckets = one pass (scalar; trivially cheap next to the L1 body).
    for b in 0 .. PsqtBuckets - 1 do
        let mutable s = srcPsq.[srcPsqOff + b]

        for k in 0 .. haN - 1 do
            s <- s + halfPsqt.[halfAdd.[haOff + k] * PsqtBuckets + b]

        for k in 0 .. hsN - 1 do
            s <- s - halfPsqt.[halfSub.[hsOff + k] * PsqtBuckets + b]

        for k in 0 .. taN - 1 do
            s <- s + thrPsqt.[thrAdd.[taOff + k] * PsqtBuckets + b]

        for k in 0 .. tsN - 1 do
            s <- s - thrPsqt.[thrSub.[tsOff + k] * PsqtBuckets + b]

        dstPsq.[dstPsqOff + b] <- s

/// Fused frame apply (see the section comment). Row lists larger than `FusedMaxRowsPerPass` in total are
/// split: the first pass reads src and writes dst, the remaining passes run in place on dst. Order-of-
/// application is irrelevant for the result (int16/int32 wrapping adds commute), so chunking is bit-exact.
let applyFused
    (srcAcc: int16[])
    (srcOff: int)
    (dstAcc: int16[])
    (dstOff: int)
    (srcPsq: int[])
    (srcPsqOff: int)
    (dstPsq: int[])
    (dstPsqOff: int)
    (halfW: int16[])
    (hwOff: int)
    (halfPsqt: int[])
    (thrW: sbyte[])
    (twOff: int)
    (thrPsqt: int[])
    (halfAdd: int[])
    (nHalfAdd: int)
    (halfSub: int[])
    (nHalfSub: int)
    (thrAdd: int[])
    (nThrAdd: int)
    (thrSub: int[])
    (nThrSub: int)
    (useAvx512: bool)
    (useAvx2: bool)
    =
    System.Diagnostics.Debug.Assert(
        (not (System.Object.ReferenceEquals(srcAcc, dstAcc)))
        || srcOff = dstOff
        || abs (srcOff - dstOff) >= L1,
        "applyFused: partially overlapping src/dst ranges"
    )

    // Weight-row prefetch: fire every row's leading lines before the tile sweep. Weight arrays are
    // POH-pinned (allocAligned64), so raw addresses are stable; prefetch never faults. HalfKA rows
    // are 2 KB int16, threat rows 1 KB int8 — leading 256 B covers the first 2 (half) / 4 (threat) tiles.
    if useAvx2 && Sse.IsSupported then
        let inline rowPf (basePtr: nativeint) (_rowBytes: int) =
            Sse.Prefetch0(NativePtr.toVoidPtr (NativePtr.ofNativeInt<byte> basePtr))
            Sse.Prefetch0(NativePtr.toVoidPtr (NativePtr.ofNativeInt<byte> (basePtr + 64n)))
            Sse.Prefetch0(NativePtr.toVoidPtr (NativePtr.ofNativeInt<byte> (basePtr + 128n)))
            Sse.Prefetch0(NativePtr.toVoidPtr (NativePtr.ofNativeInt<byte> (basePtr + 192n)))

        if nHalfAdd + nHalfSub > 0 then
            let hwRef = &MemoryMarshal.GetArrayDataReference halfW
            let hwPtr = NativePtr.toNativeInt (NativePtr.ofVoidPtr<byte> (Unsafe.AsPointer(&hwRef)))

            for k in 0 .. nHalfAdd - 1 do
                rowPf (hwPtr + nativeint ((hwOff + halfAdd.[k] * L1) * 2)) (L1 * 2)

            for k in 0 .. nHalfSub - 1 do
                rowPf (hwPtr + nativeint ((hwOff + halfSub.[k] * L1) * 2)) (L1 * 2)

        if nThrAdd + nThrSub > 0 then
            let twRef = &MemoryMarshal.GetArrayDataReference thrW
            let twPtr = NativePtr.toNativeInt (NativePtr.ofVoidPtr<byte> (Unsafe.AsPointer(&twRef)))

            for k in 0 .. nThrAdd - 1 do
                rowPf (twPtr + nativeint (twOff + thrAdd.[k] * L1)) L1

            for k in 0 .. nThrSub - 1 do
                rowPf (twPtr + nativeint (twOff + thrSub.[k] * L1)) L1

    let total = nHalfAdd + nHalfSub + nThrAdd + nThrSub

    if total <= FusedMaxRowsPerPass then
        applyFusedPass
            srcAcc srcOff dstAcc dstOff srcPsq srcPsqOff dstPsq dstPsqOff
            halfW hwOff halfPsqt thrW twOff thrPsqt
            halfAdd 0 nHalfAdd halfSub 0 nHalfSub thrAdd 0 nThrAdd thrSub 0 nThrSub
            useAvx512
            useAvx2
    else
        // Slice the virtual concatenation (halfAdd ++ halfSub ++ thrAdd ++ thrSub) into passes.
        let mutable consumed = 0
        let mutable first = true

        while consumed < total do
            let take = min FusedMaxRowsPerPass (total - consumed)
            let hi = consumed + take

            let inline sliceOf (gBase: int) (n: int) =
                let s = max 0 (min n (consumed - gBase))
                let e = max 0 (min n (hi - gBase))
                struct (s, e - s)

            let struct (haO, haN) = sliceOf 0 nHalfAdd
            let struct (hsO, hsN) = sliceOf nHalfAdd nHalfSub
            let struct (taO, taN) = sliceOf (nHalfAdd + nHalfSub) nThrAdd
            let struct (tsO, tsN) = sliceOf (nHalfAdd + nHalfSub + nThrAdd) nThrSub

            if first then
                applyFusedPass
                    srcAcc srcOff dstAcc dstOff srcPsq srcPsqOff dstPsq dstPsqOff
                    halfW hwOff halfPsqt thrW twOff thrPsqt
                    halfAdd haO haN halfSub hsO hsN thrAdd taO taN thrSub tsO tsN
                    useAvx512
                    useAvx2
            else
                applyFusedPass
                    dstAcc dstOff dstAcc dstOff dstPsq dstPsqOff dstPsq dstPsqOff
                    halfW hwOff halfPsqt thrW twOff thrPsqt
                    halfAdd haO haN halfSub hsO hsN thrAdd taO taN thrSub tsO tsN
                    useAvx512
                    useAvx2

            first <- false
            consumed <- hi

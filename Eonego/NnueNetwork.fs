/// Eonego — NNUE network container, fail-soft weight loader, and quantized forward-pass kernels.
///
/// Position-INDEPENDENT (it consumes an already-assembled feature vector, never a Position), so it sits
/// after Evaluation.fs and is reachable by both Nnue.fs (the Position-aware seam) and Uci.fs (the loader).
///
/// QUANTIZATION CONTRACT (the single source of truth shared by the kernel and the integer reference):
///   * inputs are unsigned bytes in [0,127]; layer weights are signed int8 (L1..L4) / int16 (L5).
///   * each int8 GEMV accumulates in int32: out[o] = bias[o] + Σ input[i]*weight[o][i].
///   * L1..L4 outputs pass through ShiftRoundClip(acc, shift) -> byte[0,127], keeping the next layer's
///     vpmaddubsw unsigned operand <=127 so the int16 intermediate (127*127*2 = 32258 < 32767) NEVER
///     saturates.
///   * L5 (int16 weights, 16->1) is scalar; the raw int32 is divided by quantScale -> WHITE-RELATIVE
///     centipawns. The negamax sign flip + clamp are applied by the caller (Nnue.evaluate), not here.
///
/// LAYOUT: weights are stored row-major in the KERNEL'S PADDED layout (`w.[out*paddedIn + col]`): L1 rows
/// padded 2577->2592, L4 rows padded 16->32 (pad columns zero-filled by the loader). The on-wire file is
/// UNPADDED; `load`/`loadBytes` transpose into this layout. AVX2 (vpmaddubsw+vpmaddwd) and a bit-identical
/// scalar path are both provided; EONEGO_FORCE_SCALAR=1 forces scalar for differential testing.
module Eonego.NnueNetwork

#nowarn "9" // NativePtr.stackalloc / fixed — AllowUnsafeBlocks is set in the .fsproj

open System
open System.IO
open System.Runtime.Intrinsics
open System.Runtime.Intrinsics.X86
open Microsoft.FSharp.NativeInterop
open Eonego.NnueRegions

// --- architecture literals (the validation oracle) -----------------------------------------------------
[<Literal>]
let Magic = "EONGNNUE"

[<Literal>]
let Version = 2u

[<Literal>]
let InputSize = 2577 // 2448 region counts + 1 STM + 128 king one-hot

[<Literal>]
let PaddedL1 = 2592 // InputSize rounded up to 81*32

[<Literal>]
let L1Size = 64

[<Literal>]
let L2Size = 32

[<Literal>]
let L3Size = 16

[<Literal>]
let L4Size = 16

[<Literal>]
let L4PaddedIn = 32 // L4's 16 inputs padded up to a 32-multiple

[<Literal>]
let OutputSize = 1

[<Literal>]
let ClipMax = 127

[<Literal>]
let PieceColSumSize = 64 * Channels * L1Size // per-(sq,pc) folded L1 column

[<Literal>]
let AuxColSize = (InputSize - AccSize) * L1Size // STM + 128 king one-hot columns

// --- L1 incremental tables (precomputed once at load) -------------------------------------------------
/// Fold every region column touching (sq,pc) into one 64-wide int32 vector so piece ops are a single add.
let private precomputePieceColSum (l1w: sbyte[]) : int[] =
    let pcs = Array.zeroCreate PieceColSumSize

    for sq in 0 .. 63 do
        let regions = regionsAt sq

        for pc in 0 .. Channels - 1 do
            let pBase = (sq * Channels + pc) * L1Size

            for o in 0 .. L1Size - 1 do
                let wRow = o * PaddedL1
                let mutable sum = 0

                for ri in 0 .. regions.Length - 1 do
                    sum <- sum + int l1w.[wRow + channelIndex regions.[ri] pc]

                pcs.[pBase + o] <- sum

    pcs

/// Direct L1 weight columns for STM (idx 2448) and the 128 king one-hots (2449..2576).
let private precomputeAuxCol (l1w: sbyte[]) : int[] =
    let aux = Array.zeroCreate AuxColSize

    for idx in AccSize .. InputSize - 1 do
        let aBase = (idx - AccSize) * L1Size

        for o in 0 .. L1Size - 1 do
            aux.[aBase + o] <- int l1w.[o * PaddedL1 + idx]

    aux

// --- immutable network container ----------------------------------------------------------------------
/// Read-only BY CONTRACT: the getters expose the backing arrays directly (zero-copy for the kernel's `fixed`
/// pin), so callers MUST NOT mutate them. Under that contract the net is safe to share across all LazySMP
/// search threads, exactly like the PeSTO tables. Weights are in padded kernel layout.
[<Sealed>]
type Network
    (
        version: uint32,
        quantScale: int,
        shift1: int,
        shift2: int,
        shift3: int,
        shift4: int,
        l1w: sbyte[],
        l1b: int[],
        l2w: sbyte[],
        l2b: int[],
        l3w: sbyte[],
        l3b: int[],
        l4w: sbyte[],
        l4b: int[],
        l5w: int16[],
        l5b0: int,
        pieceColSum: int[],
        auxCol: int[]
    ) =
    member _.Version = version
    member _.QuantScale = quantScale
    member _.Shift1 = shift1
    member _.Shift2 = shift2
    member _.Shift3 = shift3
    member _.Shift4 = shift4
    member _.L1W = l1w
    member _.L1B = l1b
    member _.L2W = l2w
    member _.L2B = l2b
    member _.L3W = l3w
    member _.L3B = l3b
    member _.L4W = l4w
    member _.L4B = l4b
    member _.L5W = l5w
    member _.L5B0 = l5b0
    /// Precomputed Σ_{r∋sq} L1W[o, r*12+pc] — read-only, thread-shared.
    member _.PieceColSum = pieceColSum
    /// L1W columns for inputs [AccSize, InputSize) — read-only, thread-shared.
    member _.AuxCol = auxCol

type NnueLoadResult =
    | Loaded of Network
    | Failed of reason: string

// --- loader (never throws; degrades to Failed on any I/O / decode / mismatch) --------------------------
let private readInt8Padded (br: BinaryReader) (rows: int) (inDim: int) (paddedIn: int) : sbyte[] =
    let w = Array.zeroCreate (rows * paddedIn) // pad columns stay 0

    for o in 0 .. rows - 1 do
        for i in 0 .. inDim - 1 do
            w.[o * paddedIn + i] <- br.ReadSByte()

    w

let private readInt32 (br: BinaryReader) (n: int) : int[] =
    let a = Array.zeroCreate n

    for i in 0 .. n - 1 do
        a.[i] <- br.ReadInt32()

    a

let private readInt16 (br: BinaryReader) (n: int) : int16[] =
    let a = Array.zeroCreate n

    for i in 0 .. n - 1 do
        a.[i] <- br.ReadInt16()

    a

/// Total on-wire payload (bytes after the 56-byte v2 header), computed from the architecture literals.
let private payloadBytes =
    (L1Size * InputSize + L2Size * L1Size + L3Size * L2Size + L4Size * L3Size) // int8 weights
    + (L1Size + L2Size + L3Size + L4Size) * 4 // int32 biases L1..L4
    + (OutputSize * L4Size) * 2 // int16 L5 weights
    + OutputSize * 4 // int32 L5 bias

/// Parse + validate a little-endian EONGNNUE net from a byte buffer. NEVER throws.
let loadBytes (bytes: byte[]) : NnueLoadResult =
    try
        use ms = new MemoryStream(bytes, false)
        use br = new BinaryReader(ms)
        let magic = Text.Encoding.ASCII.GetString(br.ReadBytes 8)

        if magic <> Magic then
            Failed(sprintf "bad magic %A" magic)
        else
            let version = br.ReadUInt32()
            let inp = br.ReadUInt32()
            let l1 = br.ReadUInt32()
            let l2 = br.ReadUInt32()
            let l3 = br.ReadUInt32()
            let l4 = br.ReadUInt32()
            let out = br.ReadUInt32()
            let qs = br.ReadInt32()
            let shift1 = br.ReadInt32()
            let shift2 = br.ReadInt32()
            let shift3 = br.ReadInt32()
            let shift4 = br.ReadInt32()

            if version <> Version then
                Failed(sprintf "unsupported version %d (expected %d)" version Version)
            elif inp <> uint InputSize then
                Failed(sprintf "bad inputSize %d (expected %d)" inp InputSize)
            elif l1 <> uint L1Size then
                Failed(sprintf "bad l1Size %d (expected %d)" l1 L1Size)
            elif l2 <> uint L2Size then
                Failed(sprintf "bad l2Size %d (expected %d)" l2 L2Size)
            elif l3 <> uint L3Size then
                Failed(sprintf "bad l3Size %d (expected %d)" l3 L3Size)
            elif l4 <> uint L4Size then
                Failed(sprintf "bad l4Size %d (expected %d)" l4 L4Size)
            elif out <> uint OutputSize then
                Failed(sprintf "bad outputSize %d (expected %d)" out OutputSize)
            elif qs <= 0 then
                Failed(sprintf "bad quantScale %d (must be > 0)" qs)
            elif shift1 < 0 || shift1 > 31 then
                Failed(sprintf "bad shift1 %d (expected 0..31)" shift1)
            elif shift2 < 0 || shift2 > 31 then
                Failed(sprintf "bad shift2 %d (expected 0..31)" shift2)
            elif shift3 < 0 || shift3 > 31 then
                Failed(sprintf "bad shift3 %d (expected 0..31)" shift3)
            elif shift4 < 0 || shift4 > 31 then
                Failed(sprintf "bad shift4 %d (expected 0..31)" shift4)
            elif int64 (ms.Length - ms.Position) < int64 payloadBytes then
                Failed "file truncated: weight payload shorter than architecture"
            else
                let l1w = readInt8Padded br L1Size InputSize PaddedL1
                let l1b = readInt32 br L1Size
                let l2w = readInt8Padded br L2Size L1Size L1Size
                let l2b = readInt32 br L2Size
                let l3w = readInt8Padded br L3Size L2Size L2Size
                let l3b = readInt32 br L3Size
                let l4w = readInt8Padded br L4Size L3Size L4PaddedIn
                let l4b = readInt32 br L4Size
                let l5w = readInt16 br (OutputSize * L4Size)
                let l5b = readInt32 br OutputSize
                let pieceColSum = precomputePieceColSum l1w
                let auxCol = precomputeAuxCol l1w
                Loaded(Network(version, qs, shift1, shift2, shift3, shift4, l1w, l1b, l2w, l2b, l3w, l3b, l4w, l4b, l5w, l5b.[0], pieceColSum, auxCol))
    with ex ->
        Failed(sprintf "decode exception: %s" ex.Message)

/// Read + validate a net file. NEVER throws (missing/unreadable -> Failed).
let load (path: string) : NnueLoadResult =
    try
        if not (File.Exists path) then
            Failed(sprintf "file not found: %s" path)
        else
            loadBytes (File.ReadAllBytes path)
    with ex ->
        Failed(sprintf "load exception: %s" ex.Message)

// --- forward-pass kernels -----------------------------------------------------------------------------
let forceScalar: bool =
    match Environment.GetEnvironmentVariable "EONEGO_FORCE_SCALAR" with
    | null
    | ""
    | "0" -> false
    | _ -> true

/// AVX2 unless forced scalar or the CPU lacks AVX2. Read once.
let useAvx2Default: bool = Avx2.IsSupported && not forceScalar

let inline private clamp (x: int) : int =
    if x < 0 then 0
    elif x > ClipMax then ClipMax
    else x

let inline private hsum256 (v: Vector256<int>) : int =
    let s1 = Sse2.Add(v.GetLower(), v.GetUpper())
    let s2 = Sse2.Add(s1, Sse2.Shuffle(s1, 0b01_00_11_10uy))
    let s3 = Sse2.Add(s2, Sse2.Shuffle(s2, 0b00_00_00_01uy))
    s3.ToScalar()

/// int8 affine GEMV (no activation). out[o] = bias[o] + Σ_{i<paddedIn} input[i]*weights[o*paddedIn+i].
/// `paddedIn` is a multiple of 32; input bytes in [0,127]; the scalar path mirrors vpmaddubsw's adjacent
/// int16 pairing (with the never-reached saturation encoded) so AVX2 and scalar are bit-identical.
let private affine
    (useAvx2: bool)
    (input: nativeptr<byte>)
    (weights: nativeptr<sbyte>)
    (bias: nativeptr<int>)
    (out: nativeptr<int>)
    (paddedIn: int)
    (outSize: int)
    : unit =
    if useAvx2 then
        let ones = Vector256.Create(1s)
        let chunks = paddedIn >>> 5

        for o in 0 .. outSize - 1 do
            let wRow = NativePtr.add weights (o * paddedIn)
            let mutable acc = Vector256<int>.Zero
            let mutable c = 0

            while c < chunks do
                let off = c <<< 5
                let u = Avx.LoadVector256(NativePtr.add input off)
                let w = Avx.LoadVector256(NativePtr.add wRow off)
                let p16 = Avx2.MultiplyAddAdjacent(u, w)
                let p32 = Avx2.MultiplyAddAdjacent(p16, ones)
                acc <- Avx2.Add(acc, p32)
                c <- c + 1

            NativePtr.set out o (hsum256 acc + NativePtr.get bias o)
    else
        for o in 0 .. outSize - 1 do
            let wRow = NativePtr.add weights (o * paddedIn)
            let mutable acc = 0
            let mutable i = 0

            while i < paddedIn do
                let p =
                    int (NativePtr.get input i) * int (NativePtr.get wRow i)
                    + int (NativePtr.get input (i + 1)) * int (NativePtr.get wRow (i + 1))

                let p16 =
                    if p > 32767 then 32767
                    elif p < -32768 then -32768
                    else p

                acc <- acc + p16
                i <- i + 2

            NativePtr.set out o (acc + NativePtr.get bias o)

let private clippedReLU (shift: int) (acc: nativeptr<int>) (out: nativeptr<byte>) (n: int) : unit =
    let round = if shift > 0 then 1 <<< (shift - 1) else 0

    for i in 0 .. n - 1 do
        NativePtr.set out i (byte (clamp ((NativePtr.get acc i + round) >>> shift)))

let private finalLayer (input: nativeptr<byte>) (w16: nativeptr<int16>) (bias0: int) : int =
    let mutable acc = bias0

    for i in 0 .. L4Size - 1 do
        acc <- acc + int (NativePtr.get input i) * int (NativePtr.get w16 i)

    acc

/// Full forward pass over an assembled, padded input (length PaddedL1; tail [InputSize,PaddedL1) zeroed).
/// Returns WHITE-RELATIVE centipawns (caller applies clamp + negamax sign). 0 B/op (stack intermediates).
/// `useAvx2=false` forces the bit-identical scalar path.
let forwardWith (useAvx2: bool) (net: Network) (input: nativeptr<byte>) : int =
    let l1acc = NativePtr.stackalloc<int> L1Size
    let l1out = NativePtr.stackalloc<byte> L1Size
    let l2acc = NativePtr.stackalloc<int> L2Size
    let l2out = NativePtr.stackalloc<byte> L2Size
    let l3acc = NativePtr.stackalloc<int> L3Size
    let l3out = NativePtr.stackalloc<byte> L4PaddedIn // 32: [L3Size,L4PaddedIn) zero-padded for L4
    let l4acc = NativePtr.stackalloc<int> L4Size
    let l4out = NativePtr.stackalloc<byte> L4Size

    for k in L3Size .. L4PaddedIn - 1 do
        NativePtr.set l3out k 0uy

    use l1w = fixed net.L1W
    use l1b = fixed net.L1B
    use l2w = fixed net.L2W
    use l2b = fixed net.L2B
    use l3w = fixed net.L3W
    use l3b = fixed net.L3B
    use l4w = fixed net.L4W
    use l4b = fixed net.L4B
    use l5w = fixed net.L5W

    affine useAvx2 input l1w l1b l1acc PaddedL1 L1Size
    clippedReLU net.Shift1 l1acc l1out L1Size
    affine useAvx2 l1out l2w l2b l2acc L1Size L2Size
    clippedReLU net.Shift2 l2acc l2out L2Size
    affine useAvx2 l2out l3w l3b l3acc L2Size L3Size
    clippedReLU net.Shift3 l3acc l3out L3Size
    affine useAvx2 l3out l4w l4b l4acc L4PaddedIn L4Size
    clippedReLU net.Shift4 l4acc l4out L4Size
    (finalLayer l4out l5w net.L5B0) / net.QuantScale

/// Default-dispatch forward (AVX2 unless EONEGO_FORCE_SCALAR / no AVX2). Hot-path entry for Nnue.evaluate.
let forward (net: Network) (input: nativeptr<byte>) : int = forwardWith useAvx2Default net input

/// Forward pass from a maintained L1 pre-activation (skips the L1 affine). Bit-identical to forwardWith's
/// L2..L5 given the same l1acc the affine would produce.
let forwardFromL1 (useAvx2: bool) (net: Network) (l1acc: int[]) : int =
    let l1out = NativePtr.stackalloc<byte> L1Size
    let l2acc = NativePtr.stackalloc<int> L2Size
    let l2out = NativePtr.stackalloc<byte> L2Size
    let l3acc = NativePtr.stackalloc<int> L3Size
    let l3out = NativePtr.stackalloc<byte> L4PaddedIn
    let l4acc = NativePtr.stackalloc<int> L4Size
    let l4out = NativePtr.stackalloc<byte> L4Size

    for k in L3Size .. L4PaddedIn - 1 do
        NativePtr.set l3out k 0uy

    use l1accPin = fixed l1acc
    use l2w = fixed net.L2W
    use l2b = fixed net.L2B
    use l3w = fixed net.L3W
    use l3b = fixed net.L3B
    use l4w = fixed net.L4W
    use l4b = fixed net.L4B
    use l5w = fixed net.L5W

    clippedReLU net.Shift1 l1accPin l1out L1Size
    affine useAvx2 l1out l2w l2b l2acc L1Size L2Size
    clippedReLU net.Shift2 l2acc l2out L2Size
    affine useAvx2 l2out l3w l3b l3acc L2Size L3Size
    clippedReLU net.Shift3 l3acc l3out L3Size
    affine useAvx2 l3out l4w l4b l4acc L4PaddedIn L4Size
    clippedReLU net.Shift4 l4acc l4out L4Size
    (finalLayer l4out l5w net.L5B0) / net.QuantScale

/// Default-dispatch forwardFromL1 (hot path when the Position-side L1 accumulator is bound).
let forwardFromL1Default (net: Network) (l1acc: int[]) : int = forwardFromL1 useAvx2Default net l1acc

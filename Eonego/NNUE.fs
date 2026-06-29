/// Stockfish-master NNUE ("FullThreats" net, version 0x6A448AFA) loader + evaluator for Eonego.
/// Clean-room from the published nnue-pytorch format + the SF master inference spec. Dual-input feature
/// transformer: the L1=1024 accumulator (and 8 PSQT buckets) is built from BOTH HalfKAv2_hm features
/// (int16 `weights`) AND FullThreats features (int8 `threatWeights`); HalfKA indexing is reused from
/// `Accumulator.makeIndex`, threats from `Threats`. Bound positions use Position's lazy incremental
/// accumulator; unbound positions keep the from-scratch path as an oracle.
///
/// Licensing: Stockfish `.nnue` FILES are CC0; this loader is a clean-room implementation of the file
/// format (not copyrightable) and the integer inference, not a copy of SF's GPL C++. See THIRD-PARTY-NOTICES.
///
/// AOT/F#: pure byte parsing; no printf; fail-soft; forward is stackalloc'd (0 heap alloc on the hot path).
#nowarn "9"
module Eonego.Nnue

open System
open System.Runtime.CompilerServices
open System.Runtime.Intrinsics
open System.Runtime.Intrinsics.X86
open Microsoft.FSharp.NativeInterop
open Eonego.Bitboard
open Eonego.Position

// ---------------------------------------------------------------------------
// Architecture constants (SF master FullThreats net; verified against src/nnue/*)
// ---------------------------------------------------------------------------
[<Literal>]
let SfVersion = 0x6A448AFA

[<Literal>]
let HalfKaDims = 22528 // HalfKAv2_hm feature dimension (704 planes * 32 king buckets)

[<Literal>]
let ThreatDims = 60720 // FullThreats feature dimension

[<Literal>]
let L1 = 1024 // TransformedFeatureDimensions (accumulator output per perspective)

[<Literal>]
let Half = 512 // L1 / 2

[<Literal>]
let PsqtBuckets = 8

[<Literal>]
let LayerStacks = 8

[<Literal>]
let Fc0Out = 32 // FC_0_OUTPUTS (31) + 1 forwarded skip term

[<Literal>]
let Fc1In = 64 // FC_0_OUTPUTS*2 = 62, padded to a multiple of 32

[<Literal>]
let Fc1Out = 32

[<Literal>]
let Fc2In = 32

[<Literal>]
let FtMaxVal = 255

[<Literal>]
let EvalMax = 10000

/// SF-Value that equals one pawn (100 cp). SF derives this per-position via a WDL model; we use a fixed
/// approximation (calibrate so startpos reads ~+25 cp). TUNABLE.
[<Literal>]
let NormalizeToPawnValue = 356

let UseAvx2 =
    Avx2.IsSupported && Environment.GetEnvironmentVariable("EONEGO_FORCE_SCALAR") <> "1"

/// AVX-VNNI (vpdpbusd) fuses the two-op vpmaddubsw+vpmaddwd GEMV into a single instruction. Bit-exact with the
/// AVX2 emulation whenever no i16 saturation occurs (proven for these kernels: max |ft|*|w|*2 = 32004 < 32767).
/// Implies AVX2 (the FT product and accumulator stay on the AVX2 path). EONEGO_FORCE_NOVNNI disables it so the
/// scalar/AVX2 parity tests can force the non-VNNI path. Read once.
let UseVnni =
    AvxVnni.IsSupported
    && Environment.GetEnvironmentVariable("EONEGO_FORCE_SCALAR") <> "1"
    && Environment.GetEnvironmentVariable("EONEGO_FORCE_NOVNNI") <> "1"

/// One per-bucket fc stack: fc0 (L1->32), fc1 (62->32), fc2 (32->1). Weights int8 (natural [out][padded_in]),
/// biases int32.
type SfLayerStack =
    { Fc0W: sbyte[] // Fc0Out * L1
      Fc0B: int[] // Fc0Out
      Fc1W: sbyte[] // Fc1Out * Fc1In
      Fc1B: int[] // Fc1Out
      Fc2W: sbyte[] // 1 * Fc2In
      Fc2B: int[] } // 1

type SfNetwork =
    { Version: int
      Hash: uint32
      Desc: string
      FtHash: uint32
      FtBiases: int16[] // L1
      Weights: int16[] // HalfKaDims * L1   (HalfKA -> accumulator, row-major [feature][L1])
      ThreatWeights: sbyte[] // ThreatDims * L1   (Threats -> accumulator, row-major [feature][L1])
      PsqtWeights: int[] // HalfKaDims * PsqtBuckets
      ThreatPsqtWeights: int[] // ThreatDims * PsqtBuckets
      Stacks: SfLayerStack[] }

type SfLoadResult =
    | Loaded of SfNetwork
    | Failed of string

// ---------------------------------------------------------------------------
// Byte cursor + COMPRESSED_LEB128 auto-detection (FT arrays are leb128; threat FT + layer params are raw).
// ---------------------------------------------------------------------------
let private Leb128Magic = "COMPRESSED_LEB128"B

[<Sealed>]
type private Cursor(buf: byte[]) =
    let mutable pos = 0
    member _.Pos = pos
    member _.Remaining = buf.Length - pos
    member _.AtEnd = pos >= buf.Length

    member _.U8() : int =
        let v = int buf.[pos]
        pos <- pos + 1
        v

    member _.U32() : uint32 =
        let v =
            uint32 buf.[pos]
            ||| (uint32 buf.[pos + 1] <<< 8)
            ||| (uint32 buf.[pos + 2] <<< 16)
            ||| (uint32 buf.[pos + 3] <<< 24)

        pos <- pos + 4
        v

    member this.I32() : int = int (this.U32())

    member _.Str(n: int) : string =
        let s = Text.Encoding.UTF8.GetString(buf, pos, n)
        pos <- pos + n
        s

    member _.PeekLeb128() : bool =
        if buf.Length - pos < Leb128Magic.Length then
            false
        else
            let mutable ok = true
            let mutable i = 0

            while ok && i < Leb128Magic.Length do
                if buf.[pos + i] <> Leb128Magic.[i] then
                    ok <- false

                i <- i + 1

            ok

    member _.SLeb() : int64 =
        let mutable result = 0L
        let mutable shift = 0
        let mutable more = true
        let mutable last = 0uy

        while more do
            let b = buf.[pos]
            pos <- pos + 1
            last <- b
            result <- result ||| (int64 (b &&& 0x7Fuy) <<< shift)
            shift <- shift + 7
            more <- (b &&& 0x80uy) <> 0uy

        if shift < 64 && (last &&& 0x40uy) <> 0uy then
            result <- result ||| (-1L <<< shift)

        result

    member this.OpenLeb128() : int =
        pos <- pos + Leb128Magic.Length
        this.I32()

// Array readers: leb128-if-magic-present, else raw little-endian.
let private readI16 (c: Cursor) (n: int) : int16[] =
    let out = Array.zeroCreate<int16> n

    if c.PeekLeb128() then
        c.OpenLeb128() |> ignore
        for i in 0 .. n - 1 do
            out.[i] <- int16 (c.SLeb())
    else
        for i in 0 .. n - 1 do
            out.[i] <- int16 (c.U8() ||| (c.U8() <<< 8))

    out

let private readI32 (c: Cursor) (n: int) : int[] =
    let out = Array.zeroCreate<int> n

    if c.PeekLeb128() then
        c.OpenLeb128() |> ignore
        for i in 0 .. n - 1 do
            out.[i] <- int (c.SLeb())
    else
        for i in 0 .. n - 1 do
            out.[i] <- c.I32()

    out

let private readI8 (c: Cursor) (n: int) : sbyte[] =
    let out = Array.zeroCreate<sbyte> n

    if c.PeekLeb128() then
        c.OpenLeb128() |> ignore
        for i in 0 .. n - 1 do
            out.[i] <- sbyte (c.SLeb())
    else
        for i in 0 .. n - 1 do
            out.[i] <- sbyte (c.U8())

    out

// One fc stack: arch hash, then fc_0 (bias+weights), fc_1 (bias+weights), fc_2 (bias+weights). All raw.
let private readStack (c: Cursor) : SfLayerStack =
    c.U32() |> ignore // per-stack architecture hash
    let fc0b = readI32 c Fc0Out
    let fc0w = readI8 c (Fc0Out * L1)
    let fc1b = readI32 c Fc1Out
    let fc1w = readI8 c (Fc1Out * Fc1In)
    let fc2b = readI32 c 1
    let fc2w = readI8 c (1 * Fc2In)

    { Fc0W = fc0w
      Fc0B = fc0b
      Fc1W = fc1w
      Fc1B = fc1b
      Fc2W = fc2w
      Fc2B = fc2b }

let loadBytes (buf: byte[]) : SfLoadResult =
    try
        let c = Cursor(buf)
        let version = c.I32()

        if version <> SfVersion then
            Failed("unexpected NNUE version 0x" + version.ToString("X8") + " (expected 0x" + SfVersion.ToString("X8") + ")")
        else
            let hash = c.U32()
            let descLen = c.I32()

            if descLen < 0 || descLen > c.Remaining then
                Failed("bad description length " + string descLen)
            else
                let desc = c.Str descLen
                let ftHash = c.U32()
                let ftBiases = readI16 c L1
                let threatWeights = readI8 c (ThreatDims * L1) // RAW (~62 MB)
                let threatPsqt = readI32 c (ThreatDims * PsqtBuckets)
                let weights = readI16 c (HalfKaDims * L1)
                let psqtWeights = readI32 c (HalfKaDims * PsqtBuckets)
                let stacks = Array.init LayerStacks (fun _ -> readStack c)

                if not c.AtEnd then
                    Failed("trailing " + string c.Remaining + " bytes after parse (layout mismatch)")
                else
                    Loaded
                        { Version = version
                          Hash = hash
                          Desc = desc
                          FtHash = ftHash
                          FtBiases = ftBiases
                          Weights = weights
                          ThreatWeights = threatWeights
                          PsqtWeights = psqtWeights
                          ThreatPsqtWeights = threatPsqt
                          Stacks = stacks }
    with ex ->
        Failed("exception during NNUE parse: " + ex.Message)

let load (path: string) : SfLoadResult =
    if not (IO.File.Exists path) then
        Failed("file not found: " + path)
    else
        try
            loadBytes (IO.File.ReadAllBytes path)
        with ex ->
            Failed("could not read file: " + ex.Message)

// ---------------------------------------------------------------------------
// From-scratch accumulator + forward
// ---------------------------------------------------------------------------

/// Per-thread reusable active-threat index buffer (LazySMP-safe; keeps the eval 0-alloc on the hot path).
type private ThreatBuf() =
    [<ThreadStatic; DefaultValue>]
    static val mutable private buf: int[] | null

    static member Get() : int[] =
        match ThreatBuf.buf with
        | null ->
            let b = Array.zeroCreate Threats.MaxActive
            ThreatBuf.buf <- b
            b
        | b -> b

/// Build one perspective's L1 accumulator + PSQT from scratch: bias + HalfKA features (int16 weights) +
/// FullThreats features (int8 weights).
let private buildAcc (net: SfNetwork) (pos: Position) (pColor: Color) (acc: Span<int>) (psqt: Span<int>) =
    let ksq = pos.KingSquare pColor

    for j in 0 .. L1 - 1 do
        acc.[j] <- int net.FtBiases.[j]

    for b in 0 .. PsqtBuckets - 1 do
        psqt.[b] <- 0

    // HalfKAv2_hm: one feature per piece on the board.
    for sq in 0 .. 63 do
        let pc = pos.PieceOn sq

        if pc <> NoPiece then
            let idx = Accumulator.makeIndex pColor pc sq ksq
            let wb = idx * L1

            for j in 0 .. L1 - 1 do
                acc.[j] <- acc.[j] + int net.Weights.[wb + j]

            let pb = idx * PsqtBuckets

            for b in 0 .. PsqtBuckets - 1 do
                psqt.[b] <- psqt.[b] + net.PsqtWeights.[pb + b]

    // FullThreats: active threat features.
    let tbuf = ThreatBuf.Get()
    let n = Threats.appendActiveThreats pColor pos tbuf

    for k in 0 .. n - 1 do
        let idx = tbuf.[k]
        let wb = idx * L1

        for j in 0 .. L1 - 1 do
            acc.[j] <- acc.[j] + int net.ThreatWeights.[wb + j]

        let pb = idx * PsqtBuckets

        for b in 0 .. PsqtBuckets - 1 do
            psqt.[b] <- psqt.[b] + net.ThreatPsqtWeights.[pb + b]

/// Public oracle for tests/diagnostics: build one perspective's raw int32 accumulator from the board. Kept as
/// int32 (the mathematically exact "true value") so tests can assert the int16 incremental path == int16(this)
/// AND that this never exceeds int16 range (the overflow gate).
let buildAccOracle (net: SfNetwork) (pos: Position) (pColor: Color) (acc: Span<int>) (psqt: Span<int>) =
    buildAcc net pos pColor acc psqt

/// Production from-scratch build into an int16 accumulator (unbound positions / eval else-branch): build the
/// exact int32 value, then narrow to int16. Narrowing == the incremental int16 wrap (int16 of a sum-in-int32
/// is order-independent, and the int32 oracle never overflows int32). PSQT stays int32.
[<System.Runtime.CompilerServices.SkipLocalsInit>]
let private buildAccProd (net: SfNetwork) (pos: Position) (pColor: Color) (acc: Span<int16>) (psqt: Span<int>) =
    let tmpP = NativePtr.stackalloc<int> L1
    let tmp = Span<int>(NativePtr.toVoidPtr tmpP, L1)
    buildAcc net pos pColor tmp psqt

    for j in 0 .. L1 - 1 do
        acc.[j] <- int16 tmp.[j]

let inline private clampFt (v: int) = if v < 0 then 0 elif v > FtMaxVal then FtMaxVal else v

// Hoisted SIMD constants: avoid rebuilding per call (VOnes16 is used by the fc GEMVs; V255s by the FT product).
// Bit-identical to the inline `Vector256.Create(...)`; AOT may already fold them, so this is cheap insurance.
let private VOnes16 = Vector256.Create(1s)
let private V255s = Vector256.Create(255s)

/// 16 lanes of clamp(acc[k],0,255)*clamp(acc[k+Half],0,255) >> 9 (the FT pairwise product), result in [0,127].
[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let private ftClampProd16 (acc: Span<int16>) (k: int) : Vector256<int16> =
    // Stay in int16: clamp to [0,255] (vpmaxsw/vpminsw), vpmullw. The clamped product <= 65025 fits the low
    // 16 bits exactly, and vpsrlw is an UNSIGNED shift, so >>9 reads it as unsigned (65025 -> 127). Bit-exact.
    let a = Avx2.Min(Avx2.Max(Vector256.LoadUnsafe(&acc.[k]), Vector256<int16>.Zero), V255s)
    let b = Avx2.Min(Avx2.Max(Vector256.LoadUnsafe(&acc.[k + Half]), Vector256<int16>.Zero), V255s)
    Avx2.ShiftRightLogical(Avx2.MultiplyLow(a, b), 9uy)

/// Narrow 32 FT products from `acc` (offset `off`) to 32 linear-order bytes at `ft[dst]`. One vpackuswb packs
/// the two int16 halves (bytes [a0-7,b0-7,a8-15,b8-15]); vpermq 0xD8 (qwords 0,2,1,3) restores linear order.
[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let private ftNarrow32 (ft: Span<byte>) (acc: Span<int16>) (off: int) (dst: int) =
    let packed = Avx2.PackUnsignedSaturate(ftClampProd16 acc off, ftClampProd16 acc (off + 16))
    Vector256.StoreUnsafe(Avx2.Permute4x64(packed.AsInt64(), 0xD8uy).AsByte(), &ft.[dst])

/// Feature-transformer pairwise product into a u8 buffer: ft[j] = clamp(a,0,255)*clamp(b,0,255)/512.
/// AVX2: clamp via Min/Max, vpmulld, >>9, narrow to byte. Bit-exact with scalar (product <= 65025, /512).
let private ftProductInto (accUs: Span<int16>) (accThem: Span<int16>) (ft: Span<byte>) (useAvx2: bool) =
    if useAvx2 then
        // 32 u8 outputs/iter via ftNarrow32 (vpackusdw x2 -> vpackuswb -> vpermd). Half (512) is a multiple of 32.
        let mutable j = 0

        while j < Half do
            ftNarrow32 ft accUs j j
            ftNarrow32 ft accThem j (Half + j)
            j <- j + 32
    else
        for j in 0 .. Half - 1 do
            ft.[j] <- byte ((clampFt (int accUs.[j]) * clampFt (int accUs.[j + Half])) / 512)
            ft.[Half + j] <- byte ((clampFt (int accThem.[j]) * clampFt (int accThem.[j + Half])) / 512)

/// fc_0 GEMV (1024 -> 32): vpmaddubsw(u8 ft, i8 w) -> i16 pairs, vpmaddwd(_, ones) -> i32. Bit-exact:
/// max |ft|*|w|*2 = 126*127*2 = 32004 < 32767 (no i16 saturation); int32 accumulation order-independent.
let private fc0Gemv (ft: Span<byte>) (fc0w: sbyte[]) (fc0b: int[]) (fc0: Span<int>) (useAvx2: bool) (useVnni: bool) =
    if useVnni then
        // Register-block 4 outputs: 4 independent vpdpbusd accumulators break the per-output dependency
        // chain (vpdpbusd ~5c latency) so the loop runs throughput-bound, not latency-bound; ft reused 4x.
        // Bit-exact: each output's accumulation order is unchanged (Fc0Out=32 is divisible by 4).
        let mutable o = 0

        while o < Fc0Out do
            let wb0 = o * L1
            let wb1 = wb0 + L1
            let wb2 = wb1 + L1
            let wb3 = wb2 + L1
            let mutable a0 = Vector256<int>.Zero
            let mutable a1 = Vector256<int>.Zero
            let mutable a2 = Vector256<int>.Zero
            let mutable a3 = Vector256<int>.Zero
            let mutable i = 0

            while i < L1 do
                let u = (Vector256.LoadUnsafe(&ft.[i]): Vector256<byte>)
                a0 <- AvxVnni.MultiplyWideningAndAdd(a0, u, (Vector256.LoadUnsafe(&fc0w.[wb0 + i]): Vector256<sbyte>))
                a1 <- AvxVnni.MultiplyWideningAndAdd(a1, u, (Vector256.LoadUnsafe(&fc0w.[wb1 + i]): Vector256<sbyte>))
                a2 <- AvxVnni.MultiplyWideningAndAdd(a2, u, (Vector256.LoadUnsafe(&fc0w.[wb2 + i]): Vector256<sbyte>))
                a3 <- AvxVnni.MultiplyWideningAndAdd(a3, u, (Vector256.LoadUnsafe(&fc0w.[wb3 + i]): Vector256<sbyte>))
                i <- i + 32

            fc0.[o] <- fc0b.[o] + Vector256.Sum(a0)
            fc0.[o + 1] <- fc0b.[o + 1] + Vector256.Sum(a1)
            fc0.[o + 2] <- fc0b.[o + 2] + Vector256.Sum(a2)
            fc0.[o + 3] <- fc0b.[o + 3] + Vector256.Sum(a3)
            o <- o + 4
    elif useAvx2 then
        let mutable o = 0

        while o < Fc0Out do
            let wb0 = o * L1
            let wb1 = wb0 + L1
            let wb2 = wb1 + L1
            let wb3 = wb2 + L1
            let mutable a0 = Vector256<int>.Zero
            let mutable a1 = Vector256<int>.Zero
            let mutable a2 = Vector256<int>.Zero
            let mutable a3 = Vector256<int>.Zero
            let mutable i = 0

            while i < L1 do
                let u = (Vector256.LoadUnsafe(&ft.[i]): Vector256<byte>)
                a0 <- Avx2.Add(a0, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(u, (Vector256.LoadUnsafe(&fc0w.[wb0 + i]): Vector256<sbyte>)), VOnes16))
                a1 <- Avx2.Add(a1, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(u, (Vector256.LoadUnsafe(&fc0w.[wb1 + i]): Vector256<sbyte>)), VOnes16))
                a2 <- Avx2.Add(a2, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(u, (Vector256.LoadUnsafe(&fc0w.[wb2 + i]): Vector256<sbyte>)), VOnes16))
                a3 <- Avx2.Add(a3, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(u, (Vector256.LoadUnsafe(&fc0w.[wb3 + i]): Vector256<sbyte>)), VOnes16))
                i <- i + 32

            fc0.[o] <- fc0b.[o] + Vector256.Sum(a0)
            fc0.[o + 1] <- fc0b.[o + 1] + Vector256.Sum(a1)
            fc0.[o + 2] <- fc0b.[o + 2] + Vector256.Sum(a2)
            fc0.[o + 3] <- fc0b.[o + 3] + Vector256.Sum(a3)
            o <- o + 4
    else
        for o in 0 .. Fc0Out - 1 do
            let mutable s = fc0b.[o]
            let wb = o * L1

            for j in 0 .. L1 - 1 do
                s <- s + int fc0w.[wb + j] * int ft.[j]

            fc0.[o] <- s

/// fc_1 GEMV (64 -> 32) over u8 activations: same vpmaddubsw/vpmaddwd pattern as fc0Gemv. Bit-exact: conc in
/// [0,127], w in [-128,127] => |pair| <= 127*128*2 = 32512 < 32767 (no i16 saturation); int32 sum is
/// order-independent (|total| <= 64*16256 << 2^31). conc/wb are byte-indexed; Fc1In = 64.
let private fc1Gemv (conc: Span<byte>) (fc1w: sbyte[]) (fc1b: int[]) (fc1: Span<int>) (useAvx2: bool) (useVnni: bool) =
    if useVnni then
        for o in 0 .. Fc1Out - 1 do
            let wb = o * Fc1In
            let mutable acc = Vector256<int>.Zero
            let mutable i = 0

            while i < Fc1In do
                let u = (Vector256.LoadUnsafe(&conc.[i]): Vector256<byte>)
                let w = (Vector256.LoadUnsafe(&fc1w.[wb + i]): Vector256<sbyte>)
                acc <- AvxVnni.MultiplyWideningAndAdd(acc, u, w)
                i <- i + 32

            fc1.[o] <- fc1b.[o] + Vector256.Sum(acc)
    elif useAvx2 then
        for o in 0 .. Fc1Out - 1 do
            let wb = o * Fc1In
            let mutable acc = Vector256<int>.Zero
            let mutable i = 0

            while i < Fc1In do
                let u = (Vector256.LoadUnsafe(&conc.[i]): Vector256<byte>)
                let w = (Vector256.LoadUnsafe(&fc1w.[wb + i]): Vector256<sbyte>)
                acc <- Avx2.Add(acc, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(u, w), VOnes16))
                i <- i + 32

            fc1.[o] <- fc1b.[o] + Vector256.Sum(acc)
    else
        for o in 0 .. Fc1Out - 1 do
            let mutable s = fc1b.[o]
            let wb = o * Fc1In

            for j in 0 .. Fc1In - 1 do
                s <- s + int fc1w.[wb + j] * int conc.[j]

            fc1.[o] <- s

/// fc_2 (32 -> 1) over u8 activations: a single vpmaddubsw/vpmaddwd dot product. Bit-exact (same bounds).
let private fc2Dot (a1: Span<byte>) (fc2w: sbyte[]) (fc2b: int) (useAvx2: bool) (useVnni: bool) : int =
    if useVnni then
        let u = (Vector256.LoadUnsafe(&a1.[0]): Vector256<byte>)
        let w = (Vector256.LoadUnsafe(&fc2w.[0]): Vector256<sbyte>)
        fc2b + Vector256.Sum(AvxVnni.MultiplyWideningAndAdd(Vector256<int>.Zero, u, w))
    elif useAvx2 then
        let ones = VOnes16
        let u = (Vector256.LoadUnsafe(&a1.[0]): Vector256<byte>)
        let w = (Vector256.LoadUnsafe(&fc2w.[0]): Vector256<sbyte>)
        fc2b + Vector256.Sum(Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(u, w), ones))
    else
        let mutable s = fc2b

        for j in 0 .. Fc2In - 1 do
            s <- s + int fc2w.[j] * int a1.[j]

        s

/// SF-Value (the network output, ~NormalizeToPawnValue per pawn): (125*psqt + 131*positional)/128.
/// `useAvx2` selects the SIMD vs scalar forward path (production passes the module `UseAvx2`; tests pass both
/// to assert bit-exactness). SkipLocalsInit drops stackalloc zeroing; every stack buffer is fully written
/// before read, and the two padded conc lanes are set explicitly.
[<System.Runtime.CompilerServices.SkipLocalsInit>]
let private evalFromAcc
    (net: SfNetwork)
    (pos: Position)
    (accW: Span<int16>)
    (accB: Span<int16>)
    (psqW: Span<int>)
    (psqB: Span<int>)
    (useAvx2: bool)
    (useVnni: bool)
    : int =
    let stm = pos.SideToMove
    let accUs = if stm = White then accW else accB
    let accThem = if stm = White then accB else accW
    let psqtUs = if stm = White then psqW else psqB
    let psqtThem = if stm = White then psqB else psqW

    let bucket = (popCount pos.Occupied - 1) / 4
    let stack = net.Stacks.[bucket]

    // Stackallocs at method scope. conc/a1 are u8 activations (all values clamped to [0,127]) so the fc1/fc2
    // GEMVs can use the same vpmaddubsw/vpmaddwd kernel as fc0.
    let ftPtr = NativePtr.stackalloc<byte> L1
    let ft = Span<byte>(NativePtr.toVoidPtr ftPtr, L1)
    let fc0Ptr = NativePtr.stackalloc<int> Fc0Out
    let fc0 = Span<int>(NativePtr.toVoidPtr fc0Ptr, Fc0Out)
    let concPtr = NativePtr.stackalloc<byte> Fc1In
    let conc = Span<byte>(NativePtr.toVoidPtr concPtr, Fc1In)
    let fc1Ptr = NativePtr.stackalloc<int> Fc1Out
    let fc1 = Span<int>(NativePtr.toVoidPtr fc1Ptr, Fc1Out)
    let a1Ptr = NativePtr.stackalloc<byte> Fc2In
    let a1 = Span<byte>(NativePtr.toVoidPtr a1Ptr, Fc2In)

    // Feature transformer (u8 ft) + fc_0 GEMV (1024 -> 32).
    ftProductInto accUs accThem ft useAvx2
    fc0Gemv ft stack.Fc0W stack.Fc0B fc0 useAvx2 useVnni

    // conc: sqr(fc0[0..30]) in [0..30], lin(fc0[0..30]) in [31..61], [62..63]=0. fc0[31] is skip-only.
    for o in 0 .. 30 do
        let x = fc0.[o]
        conc.[o] <- byte (min 127L ((int64 x * int64 x) >>> 21)) // SqrClippedReLU, shift 2*7+7
        conc.[31 + o] <- byte (max 0 (min 127 (x >>> 7))) // ClippedReLU, shift 7

    conc.[62] <- 0uy
    conc.[63] <- 0uy

    // fc_1: 64 -> 32 (AVX2 GEMV)
    fc1Gemv conc stack.Fc1W stack.Fc1B fc1 useAvx2 useVnni

    // ac_1: clipped relu, shift 6 -> u8
    for o in 0 .. Fc2In - 1 do
        a1.[o] <- byte (max 0 (min 127 (fc1.[o] >>> 6)))

    // fc_2: 32 -> 1
    let fc2 = fc2Dot a1 stack.Fc2W stack.Fc2B.[0] useAvx2 useVnni

    // output: fwdOut = fc2 + skip; outputValue = fwdOut * (600*16) / (HiddenOneVal*64*2 = 16384)
    let fwdOut = fc2 + fc0.[Fc0Out - 1]
    let outputValue = int (int64 fwdOut * 9600L / 16384L)
    let psqtInternal = (psqtUs.[bucket] - psqtThem.[bucket]) / 2

    // network.evaluate returns (psqt/16, positional/16); eval blend nnue = (125*psqt + 131*positional)/128.
    let psqtV = psqtInternal / 16
    let posV = outputValue / 16
    (125 * psqtV + 131 * posV) / 128

[<System.Runtime.CompilerServices.SkipLocalsInit>]
let evalInternal (net: SfNetwork) (pos: Position) (useAvx2: bool) (useVnni: bool) : int =
    if pos.SfActive then
        // Incremental hot path: materialize BOTH perspectives in one frame walk, then read them in place.
        pos.SfEnsureBothComputed()
        let accW = pos.SfAccSpan White
        let accB = pos.SfAccSpan Black
        let psqW = pos.SfPsqtSpan White
        let psqB = pos.SfPsqtSpan Black
        evalFromAcc net pos accW accB psqW psqB useAvx2 useVnni
    else
        // From-scratch path (tests / unbound positions): int16 accumulator (PSQT int32), built via buildAccProd.
        let accWP = NativePtr.stackalloc<int16> L1
        let accW = Span<int16>(NativePtr.toVoidPtr accWP, L1)
        let accBP = NativePtr.stackalloc<int16> L1
        let accB = Span<int16>(NativePtr.toVoidPtr accBP, L1)
        let psqWP = NativePtr.stackalloc<int> PsqtBuckets
        let psqW = Span<int>(NativePtr.toVoidPtr psqWP, PsqtBuckets)
        let psqBP = NativePtr.stackalloc<int> PsqtBuckets
        let psqB = Span<int>(NativePtr.toVoidPtr psqBP, PsqtBuckets)

        buildAccProd net pos White accW psqW
        buildAccProd net pos Black accB psqB
        evalFromAcc net pos accW accB psqW psqB useAvx2 useVnni

/// Side-to-move-relative centipawns, clamped to +/-EvalMax.
let evalCp (net: SfNetwork) (pos: Position) : int =
    let cp = int (int64 (evalInternal net pos UseAvx2 UseVnni) * 100L / int64 NormalizeToPawnValue)
    max -EvalMax (min EvalMax cp)

/// Bind the net into the Position's incremental accumulator (root only). The threat enumerator is passed as
/// a delegate so Position needn't depend on Threats. After binding, `evalInternal` reads the maintained
/// accumulator; unbound, it falls back to the from-scratch oracle (used by tests).
let bindNnue (net: SfNetwork) (pos: Position) : unit =
    pos.EnableNnue
        net.FtBiases
        net.Weights
        net.PsqtWeights
        net.ThreatWeights
        net.ThreatPsqtWeights
        (System.Func<Position, int, int[], int>(fun p persp buf -> Threats.appendActiveThreats persp p buf))
        (System.Func<Position, int[], int[], int64>(fun p bw bb -> Threats.appendActiveThreatsBoth p bw bb))
        (System.Func<Position, int[], int, int, int[], int[], int64>(fun p dirty off n bw bb ->
            Threats.appendChangedThreatsBothAt p dirty off n bw bb))

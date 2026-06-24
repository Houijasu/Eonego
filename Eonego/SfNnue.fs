/// Stockfish NNUE (HalfKAv2_hm) value-network LOADER — clean-room from the published nnue-pytorch
/// `serialize.py` format and the SF16-era inference spec. This file parses a real Stockfish `.nnue`
/// file (e.g. `nn-5af11540bbfe.nnue`, version 0x7AF32F20, L1=1536) into immutable weight arrays.
///
/// Licensing: Stockfish `.nnue` network FILES are distributed under CC0-1.0 (public domain) by the
/// official-stockfish/networks project — no copyleft, no attribution required. This loader is a
/// clean-room implementation of the *file format* (which is not copyrightable); it does NOT copy
/// Stockfish's GPL C++ source. See THIRD-PARTY-NOTICES.md.
///
/// Scope of THIS file: the loader + format only. The HalfKAv2_hm feature indexing, the integer
/// inference (accumulator -> pairwise u8 -> 8 layer-stacks -> centipawns), and the bit-exact parity
/// gate against real Stockfish `eval` are the next steps (separate files).
///
/// AOT/F#: pure byte parsing over a `byte[]`; no printf; fail-soft (never throws on a bad file).
module Eonego.SfNnue

open System
open System.Runtime.Intrinsics
open System.Runtime.Intrinsics.X86
open Eonego.Bitboard
open Eonego.Position

// ---------------------------------------------------------------------------
// Architecture constants (SF16-era pure HalfKAv2_hm net; verified against the real header bytes)
// ---------------------------------------------------------------------------
[<Literal>]
let SfVersion = 0x7AF32F20 // classic SF16 NNUE version tag

[<Literal>]
let NumInputs = 22528 // HalfKAv2_hm feature dimension = 704 planes * 32 king buckets

[<Literal>]
let L1 = 1536 // TransformedFeatureDimensions (feature-transformer output per perspective)

[<Literal>]
let PsqtBuckets = 8

[<Literal>]
let LayerStacks = 8 // one fc-stack per piece-count bucket

[<Literal>]
let Fc0Out = 16 // FC_0_OUTPUTS (15) + 1 forwarded skip term

[<Literal>]
let Fc1In = 32 // FC_0_OUTPUTS*2 = 30, padded to a multiple of 32

[<Literal>]
let Fc1Out = 32

[<Literal>]
let Fc2In = 32

/// One per-bucket fully-connected stack: fc0 (L1->16), fc1 (32->32), fc2 (32->1). Weights int8,
/// biases int32, stored row-major [out][padded_in] exactly as nnue-pytorch's `write_fc_layer` emits.
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
      FtWeights: int16[] // NumInputs * L1  (row-major [feature][L1])
      FtPsqt: int[] // NumInputs * PsqtBuckets (int32)
      Stacks: SfLayerStack[] } // LayerStacks

type SfLoadResult =
    | Loaded of SfNetwork
    | Failed of string

// ---------------------------------------------------------------------------
// Byte cursor over the file image, with COMPRESSED_LEB128 auto-detection per array (the SF reader
// detects the magic string at each array position; FT arrays are leb128, FC arrays may be raw).
// ---------------------------------------------------------------------------
let private Leb128Magic = "COMPRESSED_LEB128"B // 17 bytes

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

    member this.U32() : uint32 =
        let v =
            uint32 buf.[pos]
            ||| (uint32 buf.[pos + 1] <<< 8)
            ||| (uint32 buf.[pos + 2] <<< 16)
            ||| (uint32 buf.[pos + 3] <<< 24)

        pos <- pos + 4
        v

    member this.I32() : int = int (this.U32())

    /// UTF-8 string of `n` bytes.
    member _.Str(n: int) : string =
        let s = Text.Encoding.UTF8.GetString(buf, pos, n)
        pos <- pos + n
        s

    /// Does the magic string sit at the cursor? (no advance)
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

    /// Read one signed-LEB128 varint (advances the cursor).
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

    /// Skip the COMPRESSED_LEB128 magic + read the u32 byte-count; returns the declared byte count.
    member this.OpenLeb128() : int =
        pos <- pos + Leb128Magic.Length
        this.I32()

// ---------------------------------------------------------------------------
// Array readers: auto-detect leb128-vs-raw, fill `n` elements.
// ---------------------------------------------------------------------------
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

let private readStack (c: Cursor) : SfLayerStack =
    c.U32() |> ignore // fc_hash (per-stack)
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

// ---------------------------------------------------------------------------
// Public load (fail-soft)
// ---------------------------------------------------------------------------
let loadBytes (buf: byte[]) : SfLoadResult =
    try
        let c = Cursor(buf)
        let version = c.I32()

        if version <> SfVersion then
            Failed(sprintf "unexpected NNUE version 0x%08X (expected SF16 0x%08X)" version SfVersion)
        else
            let hash = c.U32()
            let descLen = c.I32()

            if descLen < 0 || descLen > c.Remaining then
                Failed(sprintf "bad description length %d" descLen)
            else
                let desc = c.Str descLen
                let ftHash = c.U32()
                let ftBiases = readI16 c L1
                let ftWeights = readI16 c (NumInputs * L1)
                let ftPsqt = readI32 c (NumInputs * PsqtBuckets)
                let stacks = Array.init LayerStacks (fun _ -> readStack c)

                if not c.AtEnd then
                    Failed(sprintf "trailing %d bytes after parse (layout mismatch)" c.Remaining)
                else
                    Loaded
                        { Version = version
                          Hash = hash
                          Desc = desc
                          FtHash = ftHash
                          FtBiases = ftBiases
                          FtWeights = ftWeights
                          FtPsqt = ftPsqt
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
// HalfKAv2_hm feature indexing (exact reproduction of sf_16 half_ka_v2_hm.cpp make_index).
//   index = (sq ^ orient) + PieceSquareIndex[persp][pc] + KingBuckets[persp][ksq]
// Eonego squares are LERF (a1=0, rank-major) == Stockfish's. Eonego piece = color*6 + pieceType
// (WP=0..WK=5, BP=6..BK=11); we map straight into SF's PieceSquareIndex (PS_*) blocks.
// ---------------------------------------------------------------------------

/// Build one perspective's int16 accumulator (1536) and int32 PSQT accumulation (8) from scratch.
let private buildAcc (net: SfNetwork) (pos: Position) (pColor: Color) : int[] * int[] =
    let ksq = pos.KingSquare pColor
    let acc = Array.zeroCreate<int> SfAccumulator.L1

    for j in 0 .. SfAccumulator.L1 - 1 do
        acc.[j] <- int net.FtBiases.[j]

    let psqt = Array.zeroCreate<int> SfAccumulator.PsqtBuckets

    for sq in 0 .. 63 do
        let pc = pos.PieceOn sq

        if pc <> NoPiece then
            let idx = SfAccumulator.makeIndex pColor pc sq ksq
            SfAccumulator.addFeature acc psqt net.FtWeights net.FtPsqt idx 1 SfAccumulator.UseAvx2

    (acc, psqt)

/// Public oracle seam: the from-scratch accumulator for a perspective (used by tests + the no-enable path).
let accumulatorOf (net: SfNetwork) (pos: Position) (pColor: Color) : int[] * int[] = buildAcc net pos pColor

/// fc_0 GEMV (1536 -> 16). AVX2: vpmaddubsw(ft_u8, w_i8) -> int16 pairs, vpmaddwd(_, ones) -> int32,
/// accumulate over 48 chunks of 32, horizontal-sum, add bias. Bit-exact: no vpmaddubsw saturation
/// (127*127*2 = 32258 < 32767), int32 accumulation order-independent.
let fc0Gemv (useAvx2: bool) (ft: byte[]) (fc0w: sbyte[]) (fc0b: int[]) (fc0: int[]) =
    if useAvx2 && Avx2.IsSupported then
        let ones = Vector256.Create(1s)
        for o in 0 .. Fc0Out - 1 do
            let wb = o * L1
            let mutable acc = Vector256<int>.Zero
            let mutable i = 0
            while i < L1 do
                let u = (Vector256.LoadUnsafe(&ft.[i]) : Vector256<byte>)
                let w = (Vector256.LoadUnsafe(&fc0w.[wb + i]) : Vector256<sbyte>)
                acc <- Avx2.Add(acc, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(u, w), ones))
                i <- i + 32
            fc0.[o] <- fc0b.[o] + Vector256.Sum(acc)
    else
        for o in 0 .. Fc0Out - 1 do
            let mutable s = fc0b.[o]
            let wb = o * L1
            for i in 0 .. L1 - 1 do
                s <- s + int fc0w.[wb + i] * int ft.[i]
            fc0.[o] <- s

/// FT pairwise clamped product -> uint8: ft[j] = clamp(acc[j],0,127)*clamp(acc[j+768],0,127)/128
/// (us into [0..767], them into [768..1535]). AVX2 vectorizes the clamp(Min/Max)*MultiplyLow*>>7
/// (values non-negative so >>7 == /128, product <= 16129 fits int32), then narrows the 8 int32
/// to bytes scalar (no lane-crossing pack). Bit-exact with scalar.
let ftProduct (useAvx2: bool) (accUs: int[]) (accThem: int[]) (ft: byte[]) =
    let half = L1 / 2

    if useAvx2 && Avx2.IsSupported then
        let zero = Vector256<int>.Zero
        let c127 = Vector256.Create(127)
        let tmp = Array.zeroCreate<int> 8

        // us side: accUs[j] paired with accUs[j+half] -> ft[j]
        let mutable j = 0
        while j < half do
            let a = Avx2.Min(Avx2.Max((Vector256.LoadUnsafe(&accUs.[j]) : Vector256<int>), zero), c127)
            let b = Avx2.Min(Avx2.Max((Vector256.LoadUnsafe(&accUs.[j + half]) : Vector256<int>), zero), c127)
            Vector256.StoreUnsafe(Avx2.ShiftRightLogical(Avx2.MultiplyLow(a, b), 7uy), &tmp.[0])
            for k in 0 .. 7 do
                ft.[j + k] <- byte tmp.[k]
            j <- j + 8

        // them side: accThem[j] paired with accThem[j+half] -> ft[half+j]
        j <- 0
        while j < half do
            let a = Avx2.Min(Avx2.Max((Vector256.LoadUnsafe(&accThem.[j]) : Vector256<int>), zero), c127)
            let b = Avx2.Min(Avx2.Max((Vector256.LoadUnsafe(&accThem.[j + half]) : Vector256<int>), zero), c127)
            Vector256.StoreUnsafe(Avx2.ShiftRightLogical(Avx2.MultiplyLow(a, b), 7uy), &tmp.[0])
            for k in 0 .. 7 do
                ft.[half + j + k] <- byte tmp.[k]
            j <- j + 8
    else
        for j in 0 .. half - 1 do
            let u0 = max 0 (min 127 accUs.[j])
            let u1 = max 0 (min 127 accUs.[j + half])
            ft.[j] <- byte ((u0 * u1) / 128)
            let t0 = max 0 (min 127 accThem.[j])
            let t1 = max 0 (min 127 accThem.[j + half])
            ft.[half + j] <- byte ((t0 * t1) / 128)

/// The raw side-to-move-relative network value (psqt + positional), in SF internal units (a pawn ~
/// 16*328). This is exactly `NNUE::evaluate(pos, false) * OutputScale` before the /16 truncation.
let evalInternal (net: SfNetwork) (pos: Position) : int =
    let stm = pos.SideToMove
    let them = 1 - stm

    let accUs, psqtUs =
        if pos.SfActive then
            (if stm = White then pos.SfWhiteAcc, pos.SfWhitePsqt else pos.SfBlackAcc, pos.SfBlackPsqt)
        else
            buildAcc net pos stm

    let accThem, psqtThem =
        if pos.SfActive then
            (if them = White then pos.SfWhiteAcc, pos.SfWhitePsqt else pos.SfBlackAcc, pos.SfBlackPsqt)
        else
            buildAcc net pos them

    let mutable pieceCount = 0

    for sq in 0 .. 63 do
        if pos.PieceOn sq <> NoPiece then
            pieceCount <- pieceCount + 1

    let bucket = (pieceCount - 1) / 4
    let half = L1 / 2

    // Feature transformer: pairwise clamped product -> uint8 (us in [0..767], them in [768..1535]).
    let ft = Array.zeroCreate<byte> L1
    ftProduct SfAccumulator.UseAvx2 accUs accThem ft

    let stack = net.Stacks.[bucket]

    // fc_0: 1536 -> 16 (int32)
    let fc0 = Array.zeroCreate<int> Fc0Out
    fc0Gemv SfAccumulator.UseAvx2 ft stack.Fc0W stack.Fc0B fc0

    // ac_sqr_0[0..14] then ac_0[0..14] -> 30 values (padded to 32 with zeros for fc_1).
    let conc = Array.zeroCreate<int> Fc1In

    for o in 0 .. 14 do // FC_0_OUTPUTS = 15
        let x = fc0.[o]
        let sq = (int64 x * int64 x) >>> 12 // 2*WeightScaleBits
        conc.[o] <- int (min 127L (sq / 128L))
        conc.[15 + o] <- max 0 (min 127 (x >>> 6))

    // fc_1: 32 -> 32 (the two padding inputs are 0, so they contribute nothing).
    let fc1 = Array.zeroCreate<int> Fc1Out

    for o in 0 .. Fc1Out - 1 do
        let mutable s = stack.Fc1B.[o]
        let wb = o * Fc1In

        for i in 0 .. Fc1In - 1 do
            s <- s + int stack.Fc1W.[wb + i] * conc.[i]

        fc1.[o] <- s

    // ac_1: clipped relu (32)
    let a1 = Array.zeroCreate<int> Fc2In

    for o in 0 .. Fc2In - 1 do
        a1.[o] <- max 0 (min 127 (fc1.[o] >>> 6))

    // fc_2: 32 -> 1
    let mutable fc2 = stack.Fc2B.[0]

    for i in 0 .. Fc2In - 1 do
        fc2 <- fc2 + int stack.Fc2W.[i] * a1.[i]

    let fwdOut = fc0.[Fc0Out - 1] * 9600 / 8128 // fc_0_out[15] * (600*16) / (127*64)
    let positional = fc2 + fwdOut
    let psqt = (psqtUs.[bucket] - psqtThem.[bucket]) / 2
    psqt + positional

/// Eonego-centipawn clamp bound (was in the now-deleted Nnue.fs). Stays < MATE_IN_MAX_PLY and inside
/// the TT int16 eval field.
[<Literal>]
let EvalMax = 10000

/// SF-internal -> Eonego centipawns. evalInternal returns psqt+positional in SF internal units; SF's
/// value is /16 (OutputScale) and pawns = that /328 (NormalizeToPawnValue). So cp = internal*100/5248
/// (100 cp == 1 pawn). Side-to-move-relative (negamax), matching the old eval; clamped to +/-EvalMax.
let evalCp (net: SfNetwork) (pos: Position) : int =
    // int64 intermediate: guards the *100 against int32 overflow for a partially-trained / adversarial
    // net whose raw evalInternal is huge (real SF nets stay well inside int32, but evalInternal has no
    // internal clamp — the training pipeline can produce out-of-range nets).
    let cp = int (int64 (evalInternal net pos) * 100L / 5248L)
    max -EvalMax (min EvalMax cp)

/// Bind the net's feature-transformer weights into the position's incremental accumulator (root only).
let bindSfNnue (net: SfNetwork) (pos: Position) : unit =
    pos.EnableSfNnue net.FtWeights net.FtPsqt net.FtBiases

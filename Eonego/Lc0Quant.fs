/// Lc0 int8 quantization kernels: saturation-safe u8xi8 GEMM via vpmaddubsw/vpmaddwd, mirroring the proven
/// SfNnue.fc0Gemv pattern. Standalone + parity-tested in isolation BEFORE any production-forward wiring.
///
/// Scheme (matches the FakeQuantActs accuracy gate in Lc0NetTests):
///   - activations  -> [0,127] per-tensor (scale = max/127); post-ReLU values are >= 0 (negatives clamp to 0).
///   - weights      -> [-127,127] per output row (scale = rowmax/127).
///   The 127*127*2 = 32258 < 32767 bound keeps vpmaddubsw from saturating, so AVX2 == scalar BIT-EXACTLY and
///   int32 accumulation is order-independent. A row's float result is rawInt32 * actScale * rowScale + bias.
/// AOT-safe: no Printf, no reflection, pure arithmetic + intrinsics.
module Eonego.Lc0Quant

open System
open System.Runtime.Intrinsics
open System.Runtime.Intrinsics.X86

/// Per-output-row symmetric int8 quant of a [rows][k] weight matrix. Returns packed sbyte weights and the
/// per-row dequant scales (rowmax/127). A zero row gets scale 1 (its quantized weights are all zero anyway).
let quantizeRowsI8 (w: float32[]) (rows: int) (k: int) : struct (sbyte[] * float32[]) =
    let q = Array.zeroCreate<sbyte> (rows * k)
    let scales = Array.zeroCreate<float32> rows

    for r in 0 .. rows - 1 do
        let b = r * k
        let mutable mx = 0.0f

        for c in 0 .. k - 1 do
            let a = abs w.[b + c]
            if a > mx then mx <- a

        let scale = if mx > 0.0f then mx / 127.0f else 1.0f
        let inv = if mx > 0.0f then 127.0f / mx else 0.0f
        scales.[r] <- scale

        for c in 0 .. k - 1 do
            let v = MathF.Round(w.[b + c] * inv)
            q.[b + c] <- sbyte (max -127.0f (min 127.0f v))

    struct (q, scales)

/// Per-tensor activation quant to [0,127] (post-ReLU values >= 0; negatives clamp to 0). Fills `dst`
/// (length >= n) with bytes and returns the dequant scale (max/127). dst[i] = round(clamp(a,0)/scale).
let quantizeActsU8 (a: float32[]) (off: int) (n: int) (dst: byte[]) : float32 =
    let mutable mx = 0.0f

    for i in off .. off + n - 1 do
        if a.[i] > mx then mx <- a.[i]

    if mx <= 0.0f then
        Array.Clear(dst, 0, n)
        1.0f
    else
        let inv = 127.0f / mx

        for i in 0 .. n - 1 do
            let v = a.[off + i]
            let q = if v <= 0.0f then 0.0f else MathF.Round(v * inv)
            dst.[i] <- byte (min 127.0f q)

        mx / 127.0f

/// raw int32 dot of a u8 activation slice and an i8 weight slice, each `k` contiguous bytes. AVX2:
/// vpmaddubsw(u8,i8)->i16 pairs, vpmaddwd(_,ones)->i32 over 32-byte chunks + a scalar tail (k need not be a
/// multiple of 32). Bit-exact AVX2 == scalar (saturation-free per the 32258 bound; int32 sum order-free).
let inline i8Dot (useAvx2: bool) (act: byte[]) (aOff: int) (w: sbyte[]) (wOff: int) (k: int) : int =
    if useAvx2 && Avx2.IsSupported then
        let ones = Vector256.Create(1s)
        let k32 = k &&& ~~~31 // largest multiple of 32 <= k
        let mutable acc = Vector256<int>.Zero
        let mutable i = 0

        while i < k32 do
            let u = (Vector256.LoadUnsafe(&act.[aOff + i]) : Vector256<byte>)
            let wv = (Vector256.LoadUnsafe(&w.[wOff + i]) : Vector256<sbyte>)
            acc <- Avx2.Add(acc, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(u, wv), ones))
            i <- i + 32

        let mutable s = Vector256.Sum(acc)

        while i < k do
            s <- s + int w.[wOff + i] * int act.[aOff + i]
            i <- i + 1

        s
    else
        let mutable s = 0

        for j in 0 .. k - 1 do
            s <- s + int w.[wOff + j] * int act.[aOff + j]

        s

/// out[r] = sum_{j<k} act[j] * w[r*k + j] (int32). Per-row dot of one shared activation vector. See i8Dot.
let i8DotRows (useAvx2: bool) (act: byte[]) (w: sbyte[]) (rows: int) (k: int) (out: int[]) : unit =
    for r in 0 .. rows - 1 do
        out.[r] <- i8Dot useAvx2 act 0 w (r * k) k

/// Full int8 matvec: quantize the float activations, dot against pre-quantized weights, dequant per row.
/// out[r] = rawInt32 * actScale * rowScale[r] + bias[r]. `actBuf` (>= k bytes) and `rawBuf` (>= rows ints)
/// are caller-owned scratch (reused across calls — no allocation here).
let i8Matvec
    (useAvx2: bool)
    (actF: float32[])
    (off: int)
    (k: int)
    (wI8: sbyte[])
    (rowScale: float32[])
    (bias: float32[])
    (rows: int)
    (actBuf: byte[])
    (rawBuf: int[])
    (out: float32[])
    : unit =
    let aScale = quantizeActsU8 actF off k actBuf
    i8DotRows useAvx2 actBuf wI8 rows k rawBuf

    for r in 0 .. rows - 1 do
        out.[r] <- float32 rawBuf.[r] * aScale * rowScale.[r] + bias.[r]

/// int8 conv (the tower prize). NCHW channel-major tensor input[inC * sp] (sp = nPos*64), weights
/// wI8[outC][inC*k*k] i8 with per-output-channel wScale, k in {1,3}, same padding per 8x8 board. Output is
/// fp32 out[outC * sp] (dequantized) so the rest of the fp32 pipeline — ReLU/SE/gate — is unchanged.
///
/// vpmaddubsw needs the contraction dim (kk = inC*k*k) CONTIGUOUS in bytes for both operands, but the
/// fp32 broadcast-weight kernel keeps activations in [j][s] (s-strided) layout. So we quantize the input to
/// u8 per-tensor, then im2col-TRANSPOSE into colT[s][j] (j contiguous), and per (o,s) do a u8xi8 dot.
///   out[o*sp + s] = bias[o] + actScale * wScale[o] * rawInt32.
/// Padding cells are 0 (a valid u8 that contributes nothing). `inU8` (>= inC*sp) and `colT` (>= sp*kk) are
/// caller-owned scratch (reused across calls — no allocation here).
let i8Conv
    (useAvx2: bool)
    (input: float32[])
    (inC: int)
    (outC: int)
    (k: int)
    (nPos: int)
    (wI8: sbyte[])
    (wScale: float32[])
    (bias: float32[])
    (inU8: byte[])
    (colT: byte[])
    (out: float32[])
    : unit =
    let sp = nPos * 64
    let kk = inC * k * k

    // Per-tensor u8 quant of the whole conv input (post-ReLU >= 0; negatives clamp to 0).
    let aScale = quantizeActsU8 input 0 (inC * sp) inU8

    // im2col-transpose: colT[s*kk + (c*k*k + ky*k + kx)] = inU8[c][gather]  (s-major so j is contiguous).
    if k = 1 then
        for s in 0 .. sp - 1 do
            let ob = s * kk

            for c in 0 .. inC - 1 do
                colT.[ob + c] <- inU8.[c * sp + s]
    else
        // 3x3, same padding within each 8x8 board (board b occupies columns [b*64, b*64+64)).
        for b in 0 .. nPos - 1 do
            let boardBase = b * 64

            for sy in 0..7 do
                for sx in 0..7 do
                    let s = boardBase + sy * 8 + sx
                    let ob = s * kk

                    for c in 0 .. inC - 1 do
                        let cbase = c * sp + boardBase
                        let jc = c * 9

                        for ky in 0..2 do
                            let iy = sy + ky - 1

                            for kx in 0..2 do
                                let ix = sx + kx - 1

                                colT.[ob + jc + ky * 3 + kx] <-
                                    if iy >= 0 && iy < 8 && ix >= 0 && ix < 8 then
                                        inU8.[cbase + iy * 8 + ix]
                                    else
                                        0uy

    // GEMM: out[o][s] = bias[o] + actScale*wScale[o] * dot(W_i8[o], colT[s]).
    for o in 0 .. outC - 1 do
        let ob = o * sp
        let wb = o * kk
        let deq = aScale * wScale.[o]
        let bo = bias.[o]

        for s in 0 .. sp - 1 do
            let raw = i8Dot useAvx2 colT (s * kk) wI8 wb kk
            out.[ob + s] <- bo + deq * float32 raw

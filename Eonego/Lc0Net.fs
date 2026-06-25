/// Lc0 SE-residual CNN forward pass (float32, scalar + AVX2/FMA). Produces 1858 policy logits + scalar
/// value from the 112-plane encoding. NCHW layout: a tensor is float32[C*64], channel-major (channel c at
/// [c*64 .. c*64+63], spatial s = rank*8+file). Conv = im2col + broadcast-weight GEMM; SE = pool/FC/FC/gate.
/// Standard ML cross-correlation convention (weights [outC][inC][kh][kw], kh=rank). See memory lc0-net-spec.
module Eonego.Lc0Net

open System
open System.Runtime.Intrinsics
open System.Runtime.Intrinsics.X86
open Eonego.Move
open Eonego.Bitboard
open Eonego.Position
open Eonego.Lc0Proto

// ---------------------------------------------------------------------------
// Kernels (gated by `useAvx2`; scalar path is the reference).
// ---------------------------------------------------------------------------

/// TEST-ONLY activation fake-quant toggle. When true, every reluInPlace snaps its output slice to a
/// per-tensor symmetric int8 grid (scale = max/127), modelling int8 activation precision as it compounds
/// through the 21-block tower — WITHOUT writing any int8 kernel. Post-ReLU activations feed every conv/FC
/// GEMM, so this covers the inter-block stream that error would accumulate in. It is a *pessimistic* proxy
/// for a real u8-activation kernel (127 positive levels vs u8's 255). Default false => the production
/// forward is byte-identical (one predictable-false branch per relu). NOT thread-safe; tests are single-threaded.
let mutable FakeQuantActs = false

let private reluInPlace (useAvx2: bool) (a: float32[]) (off: int) (n: int) : unit =
    let endi = off + n

    if useAvx2 && Avx.IsSupported then
        let z = Vector256<float32>.Zero
        let mutable i = off

        while i + 8 <= endi do
            Vector256.StoreUnsafe(Avx.Max(Vector256.LoadUnsafe(&a.[i]), z), &a.[i])
            i <- i + 8

        while i < endi do
            (if a.[i] < 0.0f then a.[i] <- 0.0f)
            i <- i + 1
    else
        for i in off .. endi - 1 do
            if a.[i] < 0.0f then a.[i] <- 0.0f

    if FakeQuantActs then
        // Per-tensor symmetric int8 of the post-ReLU slice (values >= 0): scale = max/127, round, rescale.
        let mutable mx = 0.0f

        for i in off .. endi - 1 do
            if a.[i] > mx then mx <- a.[i]

        if mx > 0.0f then
            let inv = 127.0f / mx
            let scale = mx / 127.0f

            for i in off .. endi - 1 do
                a.[i] <- MathF.Round(a.[i] * inv) * scale

/// out[outC * nPos*64] = conv(W[outC*inC*k*k], input[inC * nPos*64]) + bias ; k in {1,3}, same padding.
/// Batched over `nPos` independent 8x8 boards: a tensor is channel-major float32[C * nPos*64], a channel's
/// nPos boards laid out back-to-back ([ch*sp + b*64 + cell]). The broadcast weight w[o][j] is reused over all
/// nPos*64 columns, so weight RAM traffic is amortized nPos-fold (the point of batching). nPos=1 is the
/// single-position path, byte-identical to the pre-batch kernel. `col` scratch length >= inC*k*k * nPos*64.
let private conv
    (useAvx2: bool)
    (w: float32[])
    (bias: float32[])
    (inC: int)
    (outC: int)
    (k: int)
    (nPos: int)
    (input: float32[])
    (col: float32[])
    (out: float32[])
    : unit =
    let kk = inC * k * k
    let sp = nPos * 64

    // im2col (k=1 is a straight copy). For k=3 the 3x3 gather is per-board (8x8 padding within each board).
    if k = 1 then
        Array.blit input 0 col 0 (inC * sp)
    else
        for c in 0 .. inC - 1 do
            for ky in 0..2 do
                let dy = ky - 1

                for kx in 0..2 do
                    let dx = kx - 1
                    let cbase = (c * 9 + ky * 3 + kx) * sp

                    for b in 0 .. nPos - 1 do
                        let ibase = c * sp + b * 64
                        let obase = cbase + b * 64

                        for sy in 0..7 do
                            let iy = sy + dy

                            for sx in 0..7 do
                                let ix = sx + dx

                                col.[obase + sy * 8 + sx] <-
                                    if iy >= 0 && iy < 8 && ix >= 0 && ix < 8 then
                                        input.[ibase + iy * 8 + ix]
                                    else
                                        0.0f

    // GEMM out[o][col] = sum_kk W[o][kk]*col[kk][col] + b[o], broadcast-weight over all sp spatial columns.
    if useAvx2 && Fma.IsSupported then
        for o in 0 .. outC - 1 do
            let obase = o * sp
            let bv = Vector256.Create(bias.[o])
            let mutable s = 0

            while s < sp do
                Vector256.StoreUnsafe(bv, &out.[obase + s])
                s <- s + 8

            let wbase = o * kk

            for j in 0 .. kk - 1 do
                let wv = Vector256.Create(w.[wbase + j])
                let cbase = j * sp
                let mutable s2 = 0

                while s2 < sp do
                    let cv = Vector256.LoadUnsafe(&col.[cbase + s2])
                    let acc = Vector256.LoadUnsafe(&out.[obase + s2])
                    Vector256.StoreUnsafe(Fma.MultiplyAdd(wv, cv, acc), &out.[obase + s2])
                    s2 <- s2 + 8
    else
        for o in 0 .. outC - 1 do
            let obase = o * sp
            let b = bias.[o]

            for s in 0 .. sp - 1 do
                out.[obase + s] <- b

            let wbase = o * kk

            for j in 0 .. kk - 1 do
                let wv = w.[wbase + j]
                let cbase = j * sp

                for s in 0 .. sp - 1 do
                    out.[obase + s] <- out.[obase + s] + wv * col.[cbase + s]

/// out[outOff + m] = W[m*K] . flat[flatOff ..] + bias[m]  (dense matrix-vector, K a multiple of 8). The
/// offsets let a batched caller run one board's slice (flat[b*K..], out[b*outN..]) without copying.
let private fcMatvec
    (useAvx2: bool)
    (w: float32[])
    (bias: float32[])
    (outN: int)
    (kdim: int)
    (flat: float32[])
    (flatOff: int)
    (out: float32[])
    (outOff: int)
    : unit =
    if useAvx2 && Fma.IsSupported then
        for m in 0 .. outN - 1 do
            let wbase = m * kdim
            let mutable acc = Vector256<float32>.Zero
            let mutable i = 0

            while i + 8 <= kdim do
                acc <- Fma.MultiplyAdd(Vector256.LoadUnsafe(&w.[wbase + i]), Vector256.LoadUnsafe(&flat.[flatOff + i]), acc)
                i <- i + 8

            let mutable s = Vector256.Sum acc

            while i < kdim do
                s <- s + w.[wbase + i] * flat.[flatOff + i]
                i <- i + 1

            out.[outOff + m] <- s + bias.[m]
    else
        for m in 0 .. outN - 1 do
            let wbase = m * kdim
            let mutable s = bias.[m]

            for i in 0 .. kdim - 1 do
                s <- s + w.[wbase + i] * flat.[flatOff + i]

            out.[outOff + m] <- s

/// Squeeze-excitation: pool x -> FC1(+relu) -> FC2 -> gate `relu(sigmoid(scale)*x + bias + blockInput)`.
/// Writes the block output back into `x`.
let private applySE
    (useAvx2: bool)
    (se: Lc0SE)
    (nPos: int)
    (x: float32[])
    (blockInput: float32[])
    (z: float32[])
    (s1: float32[])
    (s2: float32[])
    : unit =
    let c = se.C
    let seCh = se.SeCh
    let sp = nPos * 64

    // Per-board global-average pool: z[b*c + ch] = mean over that board's 64 cells.
    for b in 0 .. nPos - 1 do
        for ch in 0 .. c - 1 do
            let cbase = ch * sp + b * 64
            let mutable sum = 0.0f

            for s in 0..63 do
                sum <- sum + x.[cbase + s]

            z.[b * c + ch] <- sum / 64.0f

    // Per-board excitation FCs (z -> seCh, relu, -> 2c).
    for b in 0 .. nPos - 1 do
        fcMatvec useAvx2 se.W1 se.B1 seCh c z (b * c) s1 (b * seCh)
        reluInPlace useAvx2 s1 (b * seCh) seCh
        fcMatvec useAvx2 se.W2 se.B2 (2 * c) seCh s1 (b * seCh) s2 (b * 2 * c)

    // Per-board gate: relu(sigmoid(scale)*x + bias + blockInput).
    for b in 0 .. nPos - 1 do
        let s2base = b * 2 * c

        for ch in 0 .. c - 1 do
            let scale = 1.0f / (1.0f + MathF.Exp(-s2.[s2base + ch]))
            let bias = s2.[s2base + c + ch]
            let cbase = ch * sp + b * 64

            for s in 0..63 do
                let v = scale * x.[cbase + s] + bias + blockInput.[cbase + s]
                x.[cbase + s] <- if v < 0.0f then 0.0f else v

// ---------------------------------------------------------------------------
// Forward pass.
// ---------------------------------------------------------------------------
/// Per-Worker reusable forward-pass scratch: ONE allocation, reused across every expansion on that worker —
/// kills the ~870 KB/call churn that dominated GC at high thread counts. Sized once from the net's dims. All
/// kernels OVERWRITE their outputs (conv/fc write bias-first then accumulate; SE rewrites z/s1/s2; im2col
/// writes exactly the region the GEMM reads), so no per-call zeroing is needed. NOTE `forward` returns
/// `Logits` (an alias of this buffer), so it must be consumed before the next `forward` on the SAME scratch —
/// the sequential per-worker expansion path does exactly that; parallel callers need one scratch each.
[<Sealed>]
type Lc0Scratch(net: Lc0Net) =
    let c = net.Channels
    member val Col: float32[] = Array.zeroCreate (max (c * 9 * 64) (Lc0Encoder.Planes * 9 * 64))
    member val Cur: float32[] = Array.zeroCreate (c * 64)
    member val Tmp: float32[] = Array.zeroCreate (c * 64)
    member val ResSave: float32[] = Array.zeroCreate (c * 64)
    member val Z: float32[] = Array.zeroCreate c
    member val S1: float32[] = Array.zeroCreate 64
    member val S2: float32[] = Array.zeroCreate (2 * c)
    member val PolConv: float32[] = Array.zeroCreate (net.PolicyConv.OutC * 64)
    member val Logits: float32[] = Array.zeroCreate Lc0PolicyMap.NumPolicy
    member val ValConv: float32[] = Array.zeroCreate (net.ValueConv.OutC * 64)
    member val Vh: float32[] = Array.zeroCreate net.Value1B.Length
    member val Vout: float32[] = Array.zeroCreate 1

/// Run the net on a 112-plane input (float32[112*64]); returns (policy logits[1858], scalar value in [-1,1]).
/// Uses caller-owned `scratch` (reused across calls); the returned logits ALIAS `scratch.Logits` — consume
/// before the next call on the same scratch (or give concurrent callers their own scratch).
let forward (useAvx2: bool) (net: Lc0Net) (scratch: Lc0Scratch) (input: float32[]) : struct (float32[] * float32) =
    let c = net.Channels
    let col = scratch.Col
    let cur = scratch.Cur
    let tmp = scratch.Tmp
    let resSave = scratch.ResSave
    let z = scratch.Z
    let s1 = scratch.S1
    let s2 = scratch.S2

    // Input convolution 112 -> 256 (3x3) + ReLU.
    conv useAvx2 net.Input.W net.Input.B net.Input.InC net.Input.OutC net.Input.K 1 input col cur
    reluInPlace useAvx2 cur 0 (c * 64)

    // SE-residual tower.
    for blk in net.Tower do
        Array.blit cur 0 resSave 0 (c * 64)
        conv useAvx2 blk.Conv1.W blk.Conv1.B blk.Conv1.InC blk.Conv1.OutC blk.Conv1.K 1 cur col tmp
        reluInPlace useAvx2 tmp 0 (c * 64)
        conv useAvx2 blk.Conv2.W blk.Conv2.B blk.Conv2.InC blk.Conv2.OutC blk.Conv2.K 1 tmp col cur
        applySE useAvx2 blk.Se 1 cur resSave z s1 s2

    // Policy head: conv 256->256 (1x1) + ReLU -> FC -> 1858 logits.
    let polC = net.PolicyConv.OutC
    let polConv = scratch.PolConv
    conv useAvx2 net.PolicyConv.W net.PolicyConv.B net.PolicyConv.InC polC 1 1 cur col polConv
    reluInPlace useAvx2 polConv 0 (polC * 64)
    let logits = scratch.Logits
    fcMatvec useAvx2 net.PolicyW net.PolicyB Lc0PolicyMap.NumPolicy (polC * 64) polConv 0 logits 0

    // Value head: conv 256->64 (1x1) + ReLU -> FC 4096->128 + ReLU -> FC 128->1 -> tanh.
    let valC = net.ValueConv.OutC
    let valConv = scratch.ValConv
    conv useAvx2 net.ValueConv.W net.ValueConv.B net.ValueConv.InC valC 1 1 cur col valConv
    reluInPlace useAvx2 valConv 0 (valC * 64)
    let vh = scratch.Vh
    fcMatvec useAvx2 net.Value1W net.Value1B net.Value1B.Length (valC * 64) valConv 0 vh 0
    reluInPlace useAvx2 vh 0 net.Value1B.Length
    let vout = scratch.Vout
    fcMatvec useAvx2 net.Value2W net.Value2B 1 net.Value1B.Length vh 0 vout 0

    struct (logits, MathF.Tanh vout.[0])

// ---------------------------------------------------------------------------
// int8 quantized forward. The conv TOWER (~1.6G MACs, the real bottleneck) plus the policy/value convs and
// FCs run through the saturation-safe u8xi8 kernels (Lc0Quant); the input conv stays fp32 (its raw planes
// mix piece 0/1 with rule50 up to ~100, so a single per-tensor act scale would crush the piece planes), and
// SE stays fp32 (negligible compute, non-post-ReLU inputs). Accuracy is green-lit by the FakeQuantActs gate
// in Lc0NetTests (this path is strictly more accurate — it quantizes fewer layers than that gate).
// ---------------------------------------------------------------------------
type Lc0ConvI8 =
    { W: sbyte[] // [outC][inC*k*k] i8
      Scale: float32[] // per-output-channel dequant scale
      B: float32[] // fp32 BN-folded bias
      InC: int
      OutC: int
      K: int }

type Lc0Int8 =
    { Tower: struct (Lc0ConvI8 * Lc0ConvI8)[] // (conv1, conv2) per block; SE stays fp32 in the source net
      PolicyConv: Lc0ConvI8
      PolicyW: sbyte[]
      PolicyScale: float32[]
      ValueConv: Lc0ConvI8
      Value1W: sbyte[]
      Value1Scale: float32[]
      Value2W: sbyte[]
      Value2Scale: float32[] }

let private quantizeConv (cb: Lc0ConvBlock) : Lc0ConvI8 =
    let struct (w, sc) = Lc0Quant.quantizeRowsI8 cb.W cb.OutC (cb.W.Length / cb.OutC)
    { W = w; Scale = sc; B = cb.B; InC = cb.InC; OutC = cb.OutC; K = cb.K }

/// Build the int8 companion of a net (call ONCE at load — weights become ~4x smaller and feed the i8 kernels).
let quantize (net: Lc0Net) : Lc0Int8 =
    let struct (pw, ps) =
        Lc0Quant.quantizeRowsI8 net.PolicyW Lc0PolicyMap.NumPolicy (net.PolicyW.Length / Lc0PolicyMap.NumPolicy)

    let struct (v1, v1s) =
        Lc0Quant.quantizeRowsI8 net.Value1W net.Value1B.Length (net.Value1W.Length / net.Value1B.Length)

    let struct (v2, v2s) = Lc0Quant.quantizeRowsI8 net.Value2W 1 net.Value2W.Length

    { Tower = net.Tower |> Array.map (fun r -> struct (quantizeConv r.Conv1, quantizeConv r.Conv2))
      PolicyConv = quantizeConv net.PolicyConv
      PolicyW = pw
      PolicyScale = ps
      ValueConv = quantizeConv net.ValueConv
      Value1W = v1
      Value1Scale = v1s
      Value2W = v2
      Value2Scale = v2s }

/// Per-Worker int8 scratch (byte im2col buffers + FC act/raw scratch), sized once from the net's dims.
[<Sealed>]
type Lc0Int8Scratch(net: Lc0Net) =
    let c = net.Channels
    member val InU8: byte[] = Array.zeroCreate (c * 64) // >= max conv inC * 64
    member val ColT: byte[] = Array.zeroCreate (64 * (c * 9)) // >= 64 * max kk (tower 3x3)
    member val ActBuf: byte[] = Array.zeroCreate (net.PolicyConv.OutC * 64) // >= max FC k
    member val RawBuf: int[] = Array.zeroCreate Lc0PolicyMap.NumPolicy // >= max FC rows

/// int8 forward: fp32 input conv + ReLU, then the SE-residual tower and policy/value heads through the int8
/// conv/FC kernels (SE gate stays fp32). Returns (logits aliasing scratch.Logits, value). Mirrors `forward`
/// buffer-for-buffer; `qs` byte buffers are reused across every conv/FC (each call fully overwrites its region).
let forwardI8
    (useAvx2: bool)
    (net: Lc0Net)
    (q: Lc0Int8)
    (scratch: Lc0Scratch)
    (qs: Lc0Int8Scratch)
    (input: float32[])
    : struct (float32[] * float32) =
    let c = net.Channels
    let col = scratch.Col
    let cur = scratch.Cur
    let tmp = scratch.Tmp
    let resSave = scratch.ResSave
    let z = scratch.Z
    let s1 = scratch.S1
    let s2 = scratch.S2
    let inU8 = qs.InU8
    let colT = qs.ColT

    // Input convolution 112 -> 256 (3x3) + ReLU — fp32 (raw planes mix piece 0/1 with rule50 up to ~100).
    conv useAvx2 net.Input.W net.Input.B net.Input.InC net.Input.OutC net.Input.K 1 input col cur
    reluInPlace useAvx2 cur 0 (c * 64)

    // SE-residual tower (int8 convs, fp32 SE gate).
    for i in 0 .. net.Tower.Length - 1 do
        let struct (c1, c2) = q.Tower.[i]
        Array.blit cur 0 resSave 0 (c * 64)
        Lc0Quant.i8Conv useAvx2 cur c1.InC c1.OutC c1.K 1 c1.W c1.Scale c1.B inU8 colT tmp
        reluInPlace useAvx2 tmp 0 (c * 64)
        Lc0Quant.i8Conv useAvx2 tmp c2.InC c2.OutC c2.K 1 c2.W c2.Scale c2.B inU8 colT cur
        applySE useAvx2 net.Tower.[i].Se 1 cur resSave z s1 s2

    // Policy head: int8 conv 256->256 (1x1) + ReLU -> int8 FC -> 1858 logits.
    let polC = net.PolicyConv.OutC
    let polConv = scratch.PolConv
    Lc0Quant.i8Conv useAvx2 cur q.PolicyConv.InC polC q.PolicyConv.K 1 q.PolicyConv.W q.PolicyConv.Scale q.PolicyConv.B inU8 colT polConv
    reluInPlace useAvx2 polConv 0 (polC * 64)
    let logits = scratch.Logits
    Lc0Quant.i8Matvec useAvx2 polConv 0 (polC * 64) q.PolicyW q.PolicyScale net.PolicyB Lc0PolicyMap.NumPolicy qs.ActBuf qs.RawBuf logits

    // Value head: int8 conv 256->64 (1x1) + ReLU -> int8 FC 4096->128 + ReLU -> int8 FC 128->1 -> tanh.
    let valC = net.ValueConv.OutC
    let valConv = scratch.ValConv
    Lc0Quant.i8Conv useAvx2 cur q.ValueConv.InC valC q.ValueConv.K 1 q.ValueConv.W q.ValueConv.Scale q.ValueConv.B inU8 colT valConv
    reluInPlace useAvx2 valConv 0 (valC * 64)
    let vh = scratch.Vh
    Lc0Quant.i8Matvec useAvx2 valConv 0 (valC * 64) q.Value1W q.Value1Scale net.Value1B net.Value1B.Length qs.ActBuf qs.RawBuf vh
    reluInPlace useAvx2 vh 0 net.Value1B.Length
    let vout = scratch.Vout
    Lc0Quant.i8Matvec useAvx2 vh 0 net.Value1B.Length q.Value2W q.Value2Scale net.Value2B 1 qs.ActBuf qs.RawBuf vout

    struct (logits, MathF.Tanh vout.[0])

/// Per-Worker BATCHED forward scratch, sized for up to `maxBatch` positions (maxBatch*64 spatial columns).
/// Same role as Lc0Scratch but batched; `HeadTmp` repacks one board's strided conv output into the
/// contiguous [channel*64+cell] vector the head FCs expect.
[<Sealed>]
type Lc0BatchScratch(net: Lc0Net, maxBatch: int) =
    let c = net.Channels
    let sp = maxBatch * 64
    member val MaxBatch = maxBatch
    member val Col: float32[] = Array.zeroCreate ((max (c * 9) (Lc0Encoder.Planes * 9)) * sp)
    member val Cur: float32[] = Array.zeroCreate (c * sp)
    member val Tmp: float32[] = Array.zeroCreate (c * sp)
    member val ResSave: float32[] = Array.zeroCreate (c * sp)
    member val Z: float32[] = Array.zeroCreate (maxBatch * c)
    member val S1: float32[] = Array.zeroCreate (maxBatch * 64)
    member val S2: float32[] = Array.zeroCreate (maxBatch * 2 * c)
    member val PolConv: float32[] = Array.zeroCreate (net.PolicyConv.OutC * sp)
    member val ValConv: float32[] = Array.zeroCreate (net.ValueConv.OutC * sp)
    member val Vh: float32[] = Array.zeroCreate (maxBatch * net.Value1B.Length)
    member val Vout: float32[] = Array.zeroCreate maxBatch
    member val HeadTmp: float32[] = Array.zeroCreate ((max net.PolicyConv.OutC net.ValueConv.OutC) * 64)

/// Batched forward over `nPos` packed positions. `input` is the batched encoding float32[Planes * nPos*64]
/// (channel-major: input[ch*(nPos*64) + b*64 + cell]). Writes board b's 1858 logits to outLogits[b*1858 ..]
/// and its scalar value to outValues[b]. One weight stream for all nPos positions — the batching win.
let forwardBatch
    (useAvx2: bool)
    (net: Lc0Net)
    (scratch: Lc0BatchScratch)
    (nPos: int)
    (input: float32[])
    (outLogits: float32[])
    (outValues: float32[])
    : unit =
    let c = net.Channels
    let sp = nPos * 64
    let col = scratch.Col
    let cur = scratch.Cur
    let tmp = scratch.Tmp
    let resSave = scratch.ResSave
    let z = scratch.Z
    let s1 = scratch.S1
    let s2 = scratch.S2
    let htmp = scratch.HeadTmp

    // Input convolution + ReLU, then the SE-residual tower (all batched over nPos boards).
    conv useAvx2 net.Input.W net.Input.B net.Input.InC net.Input.OutC net.Input.K nPos input col cur
    reluInPlace useAvx2 cur 0 (c * sp)

    for blk in net.Tower do
        Array.blit cur 0 resSave 0 (c * sp)
        conv useAvx2 blk.Conv1.W blk.Conv1.B blk.Conv1.InC blk.Conv1.OutC blk.Conv1.K nPos cur col tmp
        reluInPlace useAvx2 tmp 0 (c * sp)
        conv useAvx2 blk.Conv2.W blk.Conv2.B blk.Conv2.InC blk.Conv2.OutC blk.Conv2.K nPos tmp col cur
        applySE useAvx2 blk.Se nPos cur resSave z s1 s2

    // Policy head: 1x1 conv + ReLU, then per-board FC to 1858 logits (repack the board's strided slice first).
    let polC = net.PolicyConv.OutC
    let polConv = scratch.PolConv
    conv useAvx2 net.PolicyConv.W net.PolicyConv.B net.PolicyConv.InC polC 1 nPos cur col polConv
    reluInPlace useAvx2 polConv 0 (polC * sp)
    let pk = polC * 64

    for b in 0 .. nPos - 1 do
        for o in 0 .. polC - 1 do
            Array.blit polConv (o * sp + b * 64) htmp (o * 64) 64

        fcMatvec useAvx2 net.PolicyW net.PolicyB Lc0PolicyMap.NumPolicy pk htmp 0 outLogits (b * Lc0PolicyMap.NumPolicy)

    // Value head: 1x1 conv + ReLU, then per-board FC1(+ReLU) -> FC2 -> tanh.
    let valC = net.ValueConv.OutC
    let valConv = scratch.ValConv
    conv useAvx2 net.ValueConv.W net.ValueConv.B net.ValueConv.InC valC 1 nPos cur col valConv
    reluInPlace useAvx2 valConv 0 (valC * sp)
    let vh = scratch.Vh
    let vk = valC * 64
    let v1n = net.Value1B.Length

    for b in 0 .. nPos - 1 do
        for o in 0 .. valC - 1 do
            Array.blit valConv (o * sp + b * 64) htmp (o * 64) 64

        fcMatvec useAvx2 net.Value1W net.Value1B v1n vk htmp 0 vh (b * v1n)
        reluInPlace useAvx2 vh (b * v1n) v1n
        fcMatvec useAvx2 net.Value2W net.Value2B 1 v1n vh (b * v1n) scratch.Vout b
        outValues.[b] <- MathF.Tanh scratch.Vout.[b]

/// Softmax the legal moves' policy logits (from a forward/forwardBatch output; board b's logits start at
/// `logitsOff`) into outPriors[0..count-1]. Shared by the single (lc0PriorsInto) and batched (runBatch) paths.
let lc0PriorsFromLogits
    (logits: float32[])
    (logitsOff: int)
    (pos: Position)
    (moves: Move[])
    (count: int)
    (outPriors: float32[])
    : unit =
    if count = 1 then
        outPriors.[0] <- 1.0f
    else
        let stmIsBlack = (pos.SideToMove = Black)
        let mutable mx = System.Single.NegativeInfinity

        for i in 0 .. count - 1 do
            let idx = Lc0PolicyMap.moveToNNIndex stmIsBlack moves.[i]
            let l = if idx >= 0 then logits.[logitsOff + idx] else -1e9f
            outPriors.[i] <- l
            if l > mx then mx <- l

        let mutable sum = 0.0f

        for i in 0 .. count - 1 do
            let e = MathF.Exp(outPriors.[i] - mx)
            outPriors.[i] <- e
            sum <- sum + e

        let inv = if sum > 0.0f then 1.0f / sum else 1.0f

        for i in 0 .. count - 1 do
            outPriors.[i] <- outPriors.[i] * inv

/// Encode `pos`, run the net, gather the legal moves' policy logits, softmax into `outPriors[0..count-1]`.
/// Returns the value as a win-probability q in [0,1] (STM-relative). `inBuf` is a caller-owned 112*64 buffer.
let lc0PriorsInto
    (useAvx2: bool)
    (net: Lc0Net)
    (pos: Position)
    (moves: Move[])
    (count: int)
    (inBuf: float32[])
    (scratch: Lc0Scratch)
    (outPriors: float32[])
    : float32 =
    Lc0Encoder.encodeInto pos inBuf
    let struct (logits, value) = forward useAvx2 net scratch inBuf
    lc0PriorsFromLogits logits 0 pos moves count outPriors
    (value + 1.0f) * 0.5f

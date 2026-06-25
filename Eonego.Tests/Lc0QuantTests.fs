/// Parity gates for the standalone Lc0 int8 GEMM core (Lc0Quant), validated in isolation BEFORE it is wired
/// into the forward pass. Two properties:
///   (1) the integer kernel is BIT-EXACT AVX2 == scalar (saturation-free u8xi8, order-independent int32 sum),
///       including k that is NOT a multiple of 32 (exercises the scalar tail);
///   (2) the full quantized matvec tracks an fp32 reference within int8 tolerance on realistic magnitudes.
module Eonego.Tests.Lc0QuantTests

open Xunit
open Eonego.Lc0Quant

// Deterministic LCG so the gate is reproducible (no wall-clock seed).
let private mkRng (seed: int) =
    let mutable s = uint32 seed ||| 1u

    fun () ->
        s <- s * 1664525u + 1013904223u
        s

[<Fact>]
let ``Lc0Quant i8DotRows is bit-exact AVX2 == scalar (incl. non-32-multiple k)`` () =
    // Mix of k values: exact multiples of 32 and ones with a tail (1008 = 112*9 input-conv K, 2304 = 256*9).
    for struct (rows, k) in [ struct (7, 64); struct (5, 96); struct (9, 1008); struct (3, 2304); struct (4, 130) ] do
        let rng = mkRng (rows * 1000 + k)
        let act = Array.init k (fun _ -> byte (rng () % 128u)) // [0,127]
        let w = Array.init (rows * k) (fun _ -> sbyte (int (rng () % 255u) - 127)) // [-127,127]

        let outA = Array.zeroCreate<int> rows
        let outS = Array.zeroCreate<int> rows
        i8DotRows true act w rows k outA // AVX2 path (falls back to scalar if unsupported)
        i8DotRows false act w rows k outS // scalar reference

        for r in 0 .. rows - 1 do
            Assert.Equal(outS.[r], outA.[r])

        // Cross-check against an independent int accumulation (guards the kernel itself, not just A==S).
        for r in 0 .. rows - 1 do
            let mutable s = 0
            for j in 0 .. k - 1 do
                s <- s + int w.[r * k + j] * int act.[j]
            Assert.Equal(s, outS.[r])

[<Fact>]
let ``Lc0Quant i8Matvec tracks fp32 reference within int8 tolerance`` () =
    let rows = 64
    let k = 2304 // 256*9, a real tower-conv contraction length
    let rng = mkRng 12345

    // Realistic magnitudes: weights ~N-ish in [-0.5,0.5], post-ReLU activations in [0, 4].
    let unit () = float32 (rng () % 100000u) / 100000.0f // [0,1)
    let wF = Array.init (rows * k) (fun _ -> (unit () - 0.5f))
    let bias = Array.init rows (fun _ -> unit () - 0.5f)
    let actF = Array.init k (fun _ -> unit () * 4.0f) // >= 0 (post-ReLU)

    // fp32 reference.
    let refOut = Array.zeroCreate<float32> rows

    for r in 0 .. rows - 1 do
        let mutable s = bias.[r]
        for j in 0 .. k - 1 do
            s <- s + wF.[r * k + j] * actF.[j]
        refOut.[r] <- s

    // int8 path.
    let struct (wI8, rowScale) = quantizeRowsI8 wF rows k
    let actBuf = Array.zeroCreate<byte> k
    let rawBuf = Array.zeroCreate<int> rows
    let i8Out = Array.zeroCreate<float32> rows
    i8Matvec true actF 0 k wI8 rowScale bias rows actBuf rawBuf i8Out

    // Per-output absolute error should be small relative to the output magnitude. With k=2304 random terms
    // the reference is O(sqrt(k)*scale) ~ a few units; int8 (per-row weight + per-tensor act) error is well
    // under a tenth of that. Use a fixed, generous-but-meaningful bound.
    let mutable maxd = 0.0f

    for r in 0 .. rows - 1 do
        maxd <- max maxd (abs (refOut.[r] - i8Out.[r]))

    Assert.True(maxd < 0.5f, sprintf "i8Matvec max abs error %f vs fp32" maxd)

[<Fact>]
let ``Lc0Quant i8Conv is bit-exact AVX2==scalar and tracks fp32`` () =
    // (inC, outC, k, nPos): exercises k=1 & k=3, single & batched boards, and a non-32-multiple kk tail (inC=10 -> kk=90).
    for struct (inC, outC, k, nPos) in
        [ struct (32, 16, 3, 1)
          struct (10, 8, 3, 2)
          struct (64, 8, 1, 1)
          struct (48, 12, 1, 2) ] do
        let sp = nPos * 64
        let kk = inC * k * k
        let half = k / 2
        let rng = mkRng (inC * 100 + outC * 7 + k * 3 + nPos)
        let unit () = float32 (rng () % 100000u) / 100000.0f

        let input = Array.init (inC * sp) (fun _ -> unit () * 3.0f) // post-ReLU >= 0
        let wF = Array.init (outC * kk) (fun _ -> unit () - 0.5f)
        let bias = Array.init outC (fun _ -> unit () - 0.5f)

        let struct (wI8, wScale) = quantizeRowsI8 wF outC kk

        let outA = Array.zeroCreate<float32> (outC * sp)
        let outS = Array.zeroCreate<float32> (outC * sp)
        i8Conv true input inC outC k nPos wI8 wScale bias (Array.zeroCreate<byte> (inC * sp)) (Array.zeroCreate<byte> (sp * kk)) outA
        i8Conv false input inC outC k nPos wI8 wScale bias (Array.zeroCreate<byte> (inC * sp)) (Array.zeroCreate<byte> (sp * kk)) outS

        // (1) AVX2 == scalar bit-exact (identical int dot + identical float dequant).
        for i in 0 .. outC * sp - 1 do
            Assert.Equal(outS.[i], outA.[i])

        // (2) fp32 broadcast-weight reference conv (same padding), within int8 tolerance.
        let mutable maxd = 0.0f

        for o in 0 .. outC - 1 do
            for b in 0 .. nPos - 1 do
                let boardBase = b * 64

                for sy in 0..7 do
                    for sx in 0..7 do
                        let s = boardBase + sy * 8 + sx
                        let mutable acc = bias.[o]

                        for c in 0 .. inC - 1 do
                            for ky in 0 .. k - 1 do
                                let iy = sy + ky - half

                                for kx in 0 .. k - 1 do
                                    let ix = sx + kx - half

                                    if iy >= 0 && iy < 8 && ix >= 0 && ix < 8 then
                                        acc <-
                                            acc
                                            + wF.[o * kk + c * k * k + ky * k + kx]
                                              * input.[c * sp + boardBase + iy * 8 + ix]

                        maxd <- max maxd (abs (acc - outA.[o * sp + s]))

        Assert.True(maxd < 0.6f, sprintf "i8Conv(inC=%d,k=%d,nPos=%d) max abs error %f vs fp32" inC k nPos maxd)

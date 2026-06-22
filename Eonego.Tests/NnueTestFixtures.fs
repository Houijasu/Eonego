/// Shared NNUE test fixtures: a deterministic raw (on-wire) net, the binary serializer (inverse of the
/// loader), a readable INTEGER reference forward pass (the oracle the AVX2/scalar kernel is checked against,
/// 0 tolerance), and helpers to run the kernel over a byte[] input.
module Eonego.Tests.NnueTestFixtures

#nowarn "9" // fixed pin in runKernel

open System
open System.IO
open Microsoft.FSharp.NativeInterop
open Eonego.NnueNetwork

/// Raw on-wire net (UNPADDED weights, row-major out*in+col), the form the file format stores and the
/// reference forward pass consumes. The loader pads these into the kernel's Network layout.
type RawNet =
    { QuantScale: int
      Shift1: int
      Shift2: int
      Shift3: int
      Shift4: int
      L1W: sbyte[] // L1Size * InputSize
      L1B: int[] // L1Size
      L2W: sbyte[] // L2Size * L1Size
      L2B: int[]
      L3W: sbyte[] // L3Size * L2Size
      L3B: int[]
      L4W: sbyte[] // L4Size * L3Size
      L4B: int[]
      L5W: int16[] // OutputSize * L4Size
      L5B: int }

/// All-zero net with a given quantScale (used to pin bias/scale plumbing).
let zeroNet (qs: int) : RawNet =
    { QuantScale = qs
      Shift1 = 0
      Shift2 = 0
      Shift3 = 0
      Shift4 = 0
      L1W = Array.zeroCreate (L1Size * InputSize)
      L1B = Array.zeroCreate L1Size
      L2W = Array.zeroCreate (L2Size * L1Size)
      L2B = Array.zeroCreate L2Size
      L3W = Array.zeroCreate (L3Size * L2Size)
      L3B = Array.zeroCreate L3Size
      L4W = Array.zeroCreate (L4Size * L3Size)
      L4B = Array.zeroCreate L4Size
      L5W = Array.zeroCreate L4Size
      L5B = 0 }

/// Deterministic seeded net. int8 weights span the FULL [-128,127] range (incl. extremes) to stress the
/// no-saturation invariant; inputs stay <=127 so pair products never exceed int16.
let buildSeededRefNet (seed: int) : RawNet =
    let r = Random seed
    let i8 n = Array.init n (fun _ -> sbyte (r.Next(-128, 128)))
    let i16 n = Array.init n (fun _ -> int16 (r.Next(-128, 128)))
    let i32 n = Array.init n (fun _ -> r.Next(-200, 200))

    { QuantScale = 64
      Shift1 = 8
      Shift2 = 7
      Shift3 = 6
      Shift4 = 5
      L1W = i8 (L1Size * InputSize)
      L1B = i32 L1Size
      L2W = i8 (L2Size * L1Size)
      L2B = i32 L2Size
      L3W = i8 (L3Size * L2Size)
      L3B = i32 L3Size
      L4W = i8 (L4Size * L3Size)
      L4B = i32 L4Size
      L5W = i16 L4Size
      L5B = r.Next(-1000, 1000) }

/// Serialize a raw net to the EONGNNUE little-endian byte layout (the inverse of NnueNetwork.loadBytes).
let serialize (raw: RawNet) : byte[] =
    use ms = new MemoryStream()
    use bw = new BinaryWriter(ms)
    bw.Write(Text.Encoding.ASCII.GetBytes Magic) // 8 raw bytes (NOT length-prefixed)
    bw.Write(Version)
    bw.Write(uint32 InputSize)
    bw.Write(uint32 L1Size)
    bw.Write(uint32 L2Size)
    bw.Write(uint32 L3Size)
    bw.Write(uint32 L4Size)
    bw.Write(uint32 OutputSize)
    bw.Write(raw.QuantScale)
    bw.Write(raw.Shift1)
    bw.Write(raw.Shift2)
    bw.Write(raw.Shift3)
    bw.Write(raw.Shift4)
    for v in raw.L1W do bw.Write(v)
    for v in raw.L1B do bw.Write(v)
    for v in raw.L2W do bw.Write(v)
    for v in raw.L2B do bw.Write(v)
    for v in raw.L3W do bw.Write(v)
    for v in raw.L3B do bw.Write(v)
    for v in raw.L4W do bw.Write(v)
    for v in raw.L4B do bw.Write(v)
    for v in raw.L5W do bw.Write(v)
    bw.Write(raw.L5B)
    bw.Flush()
    ms.ToArray()

let loadOrFail (raw: RawNet) : Network =
    match loadBytes (serialize raw) with
    | Loaded net -> net
    | Failed r -> failwithf "loadBytes failed unexpectedly: %s" r

let inline private clip (x: int) =
    if x < 0 then 0
    elif x > ClipMax then ClipMax
    else x

let inline private shiftClip (shift: int) (acc: int) =
    let round = if shift > 0 then 1 <<< (shift - 1) else 0
    clip ((acc + round) >>> shift)

/// Readable integer reference forward pass (the oracle). Same arithmetic + ClippedReLU(127) the kernel
/// uses, on the UNPADDED weights; no saturation occurs for inputs <=127, so it must match the kernel exactly.
let refForward (raw: RawNet) (input: byte[]) : int =
    let l1 = Array.zeroCreate L1Size

    for o in 0 .. L1Size - 1 do
        let mutable acc = raw.L1B.[o]
        for i in 0 .. InputSize - 1 do
            acc <- acc + int input.[i] * int raw.L1W.[o * InputSize + i]
        l1.[o] <- shiftClip raw.Shift1 acc

    let l2 = Array.zeroCreate L2Size

    for o in 0 .. L2Size - 1 do
        let mutable acc = raw.L2B.[o]
        for i in 0 .. L1Size - 1 do
            acc <- acc + l1.[i] * int raw.L2W.[o * L1Size + i]
        l2.[o] <- shiftClip raw.Shift2 acc

    let l3 = Array.zeroCreate L3Size

    for o in 0 .. L3Size - 1 do
        let mutable acc = raw.L3B.[o]
        for i in 0 .. L2Size - 1 do
            acc <- acc + l2.[i] * int raw.L3W.[o * L2Size + i]
        l3.[o] <- shiftClip raw.Shift3 acc

    let l4 = Array.zeroCreate L4Size

    for o in 0 .. L4Size - 1 do
        let mutable acc = raw.L4B.[o]
        for i in 0 .. L3Size - 1 do
            acc <- acc + l3.[i] * int raw.L4W.[o * L3Size + i]
        l4.[o] <- shiftClip raw.Shift4 acc

    let mutable acc = raw.L5B

    for i in 0 .. L4Size - 1 do
        acc <- acc + l4.[i] * int raw.L5W.[i]

    acc / raw.QuantScale

/// A random valid kernel input of length PaddedL1 (values in [0,127] on [0,InputSize), tail zeroed).
let makeInput (seed: int) : byte[] =
    let r = Random seed
    let a = Array.zeroCreate PaddedL1

    for i in 0 .. InputSize - 1 do
        a.[i] <- byte (r.Next(0, 128))

    a

let zeroInput () : byte[] = Array.zeroCreate PaddedL1

/// Run the kernel over a byte[] input (length PaddedL1). useAvx2=false forces the scalar path.
let runKernel (useAvx2: bool) (net: Network) (input: byte[]) : int =
    use p = fixed input
    forwardWith useAvx2 net p

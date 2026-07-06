/// Eonego — policy + WDL sidecar heads sharing the FullThreats NNUE trunk (plan: policy-net campaign).
///
/// The policy head consumes the SAME 1024-wide u8 FT pairwise-product buffer the value stack feeds fc_0
/// (`NNUE.ftInto`), so the expensive part — the incremental accumulator — is already paid for. Head v1 is
/// hidden-decomposed from->to: pfc0 (1024 -> 128, int8, CReLU) -> pfrom/pto (128 -> 64 each) -> per-position
/// logit arrays; per-move score = from[relSq stm (fromSq m)] + to[relSq stm (toSq m)] — an O(1) lookup.
/// Squares are STM-relative (Black vertically flipped, s ^^^ 56 — the accumulator perspective convention;
/// MUST match trainer/move_encoder.py). The WDL head (3 outputs per bucket off the value stack's a1
/// activation) lives in the same sidecar and is evaluated on demand only (root/PV), never per node.
///
/// EONPOL01 sidecar format (little-endian; separate file — NNUE.loadBytes hard-fails on trailing bytes):
///   magic    "EONPOL01" (8 bytes)
///   version  u32 = 1
///   ftHash   u32   (must equal the .nnue Network.FtHash — refuses a head trained on a foreign trunk)
///   hidden   u32   (must be 128 in v1)
///   shift0   u32   (pfc0 CReLU shift)
///   wdlShift u32   (WDL softmax temperature shift)
///   flags    u32   (bit 0: WDL section present)
///   pfc0B    i32[128]        pfc0W  i8[128*1024]   (row-major [out][in])
///   pfromB   i32[64]         pfromW i8[64*128]
///   ptoB     i32[64]         ptoW   i8[64*128]
///   [flags&1] wdlB i32[LayerStacks*3]; wdlW i8[LayerStacks*3*Fc2In]
///   (must end exactly at EOF)
///
/// LazySMP / lockless: PolicyNetwork is immutable after load; per-node state (logit arrays, Zobrist
/// staleness keys) is Worker-owned. Kernel bit-exactness contract mirrors NNUE's fc1Gemv: inputs are u8
/// in [0,127], weights i8 => |vpmaddubsw pair| <= 127*128*2 = 32512 < 32767 (no i16 saturation) and the
/// int32 accumulation is order-independent, so VNNI == AVX2 == scalar exactly.
module Eonego.Policy

#nowarn "9"

open System
open System.Runtime.CompilerServices
open System.Runtime.InteropServices
open System.Runtime.Intrinsics
open System.Runtime.Intrinsics.X86
open Microsoft.FSharp.NativeInterop
open Eonego.Bitboard
open Eonego.Position

[<Literal>]
let Hidden = 128 // pfc0 output width (v1 fixed; the loader rejects anything else)

[<Literal>]
let SqOut = 64 // pfrom/pto output width (one logit per STM-relative square)

[<AllowNullLiteral; Sealed>]
type PolicyNetwork
    (
        ftHash: uint32,
        shift0: int,
        wdlShift: int,
        pfc0B: int[],
        pfc0W: sbyte[],
        pfromB: int[],
        pfromW: sbyte[],
        ptoB: int[],
        ptoW: sbyte[],
        wdlB: int[],
        wdlW: sbyte[]
    ) =
    member _.FtHash = ftHash
    member _.Shift0 = shift0
    member _.WdlShift = wdlShift
    member _.Pfc0B = pfc0B
    member _.Pfc0W = pfc0W
    member _.PfromB = pfromB
    member _.PfromW = pfromW
    member _.PtoB = ptoB
    member _.PtoW = ptoW
    member _.WdlB = wdlB // LayerStacks*3, empty when no WDL section
    member _.WdlW = wdlW // LayerStacks*3*Fc2In
    member _.HasWdl = wdlB.Length > 0

type PolicyLoadResult =
    | PolicyLoaded of PolicyNetwork
    | PolicyFailed of string

/// STM-relative square: White reads the board as-is, Black vertically flipped (rank mirror) — the same
/// perspective convention as the accumulator. The trainer's move_encoder.py MUST apply the same map.
[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let relSq (stm: int) (s: int) : int = if stm = White then s else s ^^^ 56

// ---------------------------------------------------------------------------
// Loader (plain little-endian, no leb128; must consume the buffer exactly)
// ---------------------------------------------------------------------------
let private readI32At (buf: byte[]) (pos: int) : int =
    int buf.[pos]
    ||| (int buf.[pos + 1] <<< 8)
    ||| (int buf.[pos + 2] <<< 16)
    ||| (int buf.[pos + 3] <<< 24)

let loadBytes (buf: byte[]) (expectedFtHash: uint32) : PolicyLoadResult =
    let magic = "EONPOL01"B

    if buf.Length < 32 then
        PolicyFailed "truncated header"
    elif Array.sub buf 0 8 <> magic then
        PolicyFailed "bad magic (want EONPOL01)"
    else
        let version = readI32At buf 8
        let ftHash = uint32 (readI32At buf 12)
        let hidden = readI32At buf 16
        let shift0 = readI32At buf 20
        let wdlShift = readI32At buf 24
        let flags = readI32At buf 28

        if version <> 1 then
            PolicyFailed ("unsupported version " + string version)
        elif ftHash <> expectedFtHash then
            PolicyFailed "ftHash mismatch (policy head trained on a different trunk)"
        elif hidden <> Hidden then
            PolicyFailed ("unsupported hidden width " + string hidden)
        elif shift0 < 0 || shift0 > 31 || wdlShift < 0 || wdlShift > 31 then
            PolicyFailed "shift out of range"
        else
            let hasWdl = flags &&& 1 <> 0
            let wdlN = NNUE.LayerStacks * 3

            let expectedLen =
                32
                + Hidden * 4 + Hidden * NNUE.L1
                + SqOut * 4 + SqOut * Hidden
                + SqOut * 4 + SqOut * Hidden
                + (if hasWdl then wdlN * 4 + wdlN * NNUE.Fc2In else 0)

            if buf.Length <> expectedLen then
                PolicyFailed ("bad length " + string buf.Length + " (want " + string expectedLen + ")")
            else
                let mutable p = 32

                let readI32Arr (n: int) =
                    let out = Array.init n (fun i -> readI32At buf (p + 4 * i))
                    p <- p + 4 * n
                    out

                let readI8Arr (n: int) =
                    let out = Array.zeroCreate<sbyte> n
                    Buffer.BlockCopy(buf, p, out, 0, n)
                    p <- p + n
                    out

                let pfc0B = readI32Arr Hidden
                let pfc0W = readI8Arr (Hidden * NNUE.L1)
                let pfromB = readI32Arr SqOut
                let pfromW = readI8Arr (SqOut * Hidden)
                let ptoB = readI32Arr SqOut
                let ptoW = readI8Arr (SqOut * Hidden)
                let wdlB = if hasWdl then readI32Arr wdlN else Array.empty
                let wdlW = if hasWdl then readI8Arr (wdlN * NNUE.Fc2In) else Array.empty
                PolicyLoaded(PolicyNetwork(ftHash, shift0, wdlShift, pfc0B, pfc0W, pfromB, pfromW, ptoB, ptoW, wdlB, wdlW))

let load (path: string) (expectedFtHash: uint32) : PolicyLoadResult =
    try
        loadBytes (IO.File.ReadAllBytes path) expectedFtHash
    with ex ->
        PolicyFailed ex.Message

// ---------------------------------------------------------------------------
// Kernels: u8 x i8 GEMV, rows x cols with cols a multiple of 32. Same instruction pattern as NNUE's
// fc1Gemv (vpdpbusd / vpmaddubsw+vpmaddwd+vpaddd / scalar), one accumulator per row.
// ---------------------------------------------------------------------------
let private VOnes16 = Vector256.Create(1s)

let private gemvU8I8
    (input: Span<byte>)
    (w: sbyte[])
    (b: int[])
    (out: Span<int>)
    (rows: int)
    (cols: int)
    (useAvx2: bool)
    (useVnni: bool)
    : unit =
    if useVnni then
        let inBase = &MemoryMarshal.GetReference input
        let wBase = &MemoryMarshal.GetArrayDataReference w

        for o in 0 .. rows - 1 do
            let wb = o * cols
            let mutable acc = Vector256<int>.Zero
            let mutable i = 0

            while i < cols do
                let u = (Vector256.LoadUnsafe(&inBase, unativeint i): Vector256<byte>)
                let wv = (Vector256.LoadUnsafe(&wBase, unativeint (wb + i)): Vector256<sbyte>)
                acc <- AvxVnni.MultiplyWideningAndAdd(acc, u, wv)
                i <- i + 32

            out.[o] <- b.[o] + Vector256.Sum(acc)
    elif useAvx2 then
        let inBase = &MemoryMarshal.GetReference input
        let wBase = &MemoryMarshal.GetArrayDataReference w

        for o in 0 .. rows - 1 do
            let wb = o * cols
            let mutable acc = Vector256<int>.Zero
            let mutable i = 0

            while i < cols do
                let u = (Vector256.LoadUnsafe(&inBase, unativeint i): Vector256<byte>)
                let wv = (Vector256.LoadUnsafe(&wBase, unativeint (wb + i)): Vector256<sbyte>)
                acc <- Avx2.Add(acc, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(u, wv), VOnes16))
                i <- i + 32

            out.[o] <- b.[o] + Vector256.Sum(acc)
    else
        for o in 0 .. rows - 1 do
            let mutable s = b.[o]
            let wb = o * cols

            for j in 0 .. cols - 1 do
                s <- s + int w.[wb + j] * int input.[j]

            out.[o] <- s

/// Head forward from a caller-provided ft buffer (the parity/test surface — kernel paths explicit).
/// Writes the 64 from-logits and 64 to-logits (raw i32, STM-relative square index).
[<System.Runtime.CompilerServices.SkipLocalsInit>]
let forwardFromFt
    (pnet: PolicyNetwork)
    (ft: Span<byte>)
    (fromOut: Span<int>)
    (toOut: Span<int>)
    (useAvx2: bool)
    (useVnni: bool)
    : unit =
    let accPtr = NativePtr.stackalloc<int> Hidden
    let acc = Span<int>(NativePtr.toVoidPtr accPtr, Hidden)
    let hidPtr = NativePtr.stackalloc<byte> Hidden
    let hid = Span<byte>(NativePtr.toVoidPtr hidPtr, Hidden)

    gemvU8I8 ft pnet.Pfc0W pnet.Pfc0B acc Hidden NNUE.L1 useAvx2 useVnni

    let shift0 = pnet.Shift0

    for o in 0 .. Hidden - 1 do
        hid.[o] <- byte (max 0 (min 127 (acc.[o] >>> shift0)))

    gemvU8I8 hid pnet.PfromW pnet.PfromB fromOut SqOut Hidden useAvx2 useVnni
    gemvU8I8 hid pnet.PtoW pnet.PtoB toOut SqOut Hidden useAvx2 useVnni

/// Production fill: materialize the FT product for the CURRENT position and run the head. Called
/// lazily from MovePick's StgQuietInit (once per node, Zobrist-guarded by the caller).
[<System.Runtime.CompilerServices.SkipLocalsInit>]
let fillLogits (net: NNUE.Network) (pnet: PolicyNetwork) (pos: Position) (fromOut: Span<int>) (toOut: Span<int>) : unit =
    let ftPtr = NativePtr.stackalloc<byte> NNUE.L1
    let ft = Span<byte>(NativePtr.toVoidPtr ftPtr, NNUE.L1)
    let nnzPtr = NativePtr.stackalloc<byte> (NNUE.L1 / 32)
    let nnz = Span<byte>(NativePtr.toVoidPtr nnzPtr, NNUE.L1 / 32)
    NNUE.ftInto net pos ft nnz
    forwardFromFt pnet ft fromOut toOut NNUE.UseAvx2 NNUE.UseVnni

/// WDL head: per-mille struct (win, draw, loss) for the CURRENT position, side-to-move relative.
/// On-demand only (root/PV reporting, MCTS Q) — a 3x32 scalar dot off the value stack's a1 activation.
/// PRE: pnet.HasWdl.
[<System.Runtime.CompilerServices.SkipLocalsInit>]
let evalWDL (net: NNUE.Network) (pnet: PolicyNetwork) (pos: Position) : struct (int * int * int) =
    let ftPtr = NativePtr.stackalloc<byte> NNUE.L1
    let ft = Span<byte>(NativePtr.toVoidPtr ftPtr, NNUE.L1)
    let nnzPtr = NativePtr.stackalloc<byte> (NNUE.L1 / 32)
    let nnz = Span<byte>(NativePtr.toVoidPtr nnzPtr, NNUE.L1 / 32)
    NNUE.ftInto net pos ft nnz
    let a1Ptr = NativePtr.stackalloc<byte> NNUE.Fc2In
    let a1 = Span<byte>(NativePtr.toVoidPtr a1Ptr, NNUE.Fc2In)
    let bucket = NNUE.a1FromFt net pos ft nnz a1

    // No inner function here: a1 is a byref-like Span and closures cannot capture it (FS0406).
    let logitsPtr = NativePtr.stackalloc<int> 3
    let logits = Span<int>(NativePtr.toVoidPtr logitsPtr, 3)

    for k in 0 .. 2 do
        let idx = bucket * 3 + k
        let wb = idx * NNUE.Fc2In
        let mutable s = pnet.WdlB.[idx]

        for j in 0 .. NNUE.Fc2In - 1 do
            s <- s + int pnet.WdlW.[wb + j] * int a1.[j]

        logits.[k] <- s

    let scale = float (1 <<< pnet.WdlShift)
    let lw = float logits.[0] / scale
    let ld = float logits.[1] / scale
    let ll = float logits.[2] / scale
    let m = max lw (max ld ll)
    let ew = Math.Exp(lw - m)
    let ed = Math.Exp(ld - m)
    let el = Math.Exp(ll - m)
    let sum = ew + ed + el
    let w = int (Math.Round(1000.0 * ew / sum))
    let d = int (Math.Round(1000.0 * ed / sum))
    struct (w, d, max 0 (1000 - w - d))

/// Policy sidecar (Policy.fs) tests: EONPOL02 loader validation, kernel bit-exactness
/// (scalar == AVX2 == VNNI), forward determinism, WDL sanity, and the two search gates —
/// OFF byte-identity (a loaded sidecar with UsePolicy=false must not move a single node)
/// and the random-weights smoke (production wiring produces a legal move, deterministically).
module Eonego.Tests.PolicyTests

open System
open System.IO
open Xunit
open Eonego.Bitboard
open Eonego.Move
open Eonego.Policy
open Eonego.Tests.TestFixtures

// ---------------------------------------------------------------------------
// EONPOL02 byte-image builder (seeded random weights, parameterized hidden width)
// ---------------------------------------------------------------------------
let private buildSidecarH (hidden: int) (seed: int) (ftHash: uint32) (withWdl: bool) : byte[] =
    let rng = Random(seed)
    use ms = new MemoryStream()
    use w = new BinaryWriter(ms)
    w.Write(Text.Encoding.ASCII.GetBytes "EONPOL02")
    w.Write(2) // version
    w.Write(int ftHash)
    w.Write(hidden)
    w.Write(6) // shift0
    w.Write(6) // wdlShift
    w.Write(if withWdl then 1 else 0)

    let writeI32s (n: int) (bound: int) =
        for _ in 1..n do
            w.Write(rng.Next(-bound, bound + 1))

    let writeI8s (n: int) =
        for _ in 1..n do
            w.Write(sbyte (rng.Next(-127, 128)))

    writeI32s hidden 5000
    writeI8s (hidden * Eonego.NNUE.L1)
    writeI32s HeadOut 5000
    writeI8s (HeadOut * hidden)
    writeI32s HeadOut 5000
    writeI8s (HeadOut * hidden)

    if withWdl then
        writeI32s (Eonego.NNUE.LayerStacks * 3) 5000
        writeI8s (Eonego.NNUE.LayerStacks * 3 * Eonego.NNUE.Fc2In)

    w.Flush()
    ms.ToArray()

let private buildSidecar (seed: int) (ftHash: uint32) (withWdl: bool) : byte[] =
    buildSidecarH 128 seed ftHash withWdl

let private loadOk (seed: int) (ftHash: uint32) (withWdl: bool) : PolicyNetwork =
    match loadBytes (buildSidecar seed ftHash withWdl) ftHash with
    | PolicyLoaded p -> p
    | PolicyFailed why -> failwith ("expected load, got: " + why)

// ---------------------------------------------------------------------------
// Loader validation
// ---------------------------------------------------------------------------
[<Fact>]
let ``loader accepts a well-formed image (with and without WDL)`` () =
    let p = loadOk 1 0xABCD1234u false
    Assert.False p.HasWdl
    let p2 = loadOk 2 0xABCD1234u true
    Assert.True p2.HasWdl

[<Fact>]
let ``loader accepts wider hidden and rejects a non-multiple-of-32 width`` () =
    match loadBytes (buildSidecarH 512 3 0xABCD1234u false) 0xABCD1234u with
    | PolicyLoaded p -> Assert.Equal(512, p.Hidden)
    | PolicyFailed why -> failwith ("expected 512-hidden load, got: " + why)

    match loadBytes (buildSidecarH 100 3 0xABCD1234u false) 0xABCD1234u with
    | PolicyFailed why -> Assert.Contains("hidden", why)
    | PolicyLoaded _ -> failwith "loaded a hidden=100 image"

[<Fact>]
let ``loader rejects bad magic`` () =
    let buf = buildSidecar 1 0xABCD1234u false
    buf.[0] <- byte 'X'

    match loadBytes buf 0xABCD1234u with
    | PolicyFailed why -> Assert.Contains("magic", why)
    | PolicyLoaded _ -> failwith "loaded a bad-magic image"

[<Fact>]
let ``loader rejects a foreign trunk (ftHash mismatch)`` () =
    let buf = buildSidecar 1 0xABCD1234u false

    match loadBytes buf 0xDEADBEEFu with
    | PolicyFailed why -> Assert.Contains("ftHash", why)
    | PolicyLoaded _ -> failwith "loaded a foreign-trunk image"

[<Fact>]
let ``loader rejects truncated and oversized images`` () =
    let buf = buildSidecar 1 0xABCD1234u false

    match loadBytes buf.[.. buf.Length - 2] 0xABCD1234u with
    | PolicyFailed _ -> ()
    | PolicyLoaded _ -> failwith "loaded a truncated image"

    match loadBytes (Array.append buf [| 0uy |]) 0xABCD1234u with
    | PolicyFailed _ -> ()
    | PolicyLoaded _ -> failwith "loaded an oversized image"

// ---------------------------------------------------------------------------
// Kernel bit-exactness: scalar == AVX2 == VNNI on random ft inputs
// ---------------------------------------------------------------------------
[<Fact>]
let ``forwardFromFt is bit-exact across kernel paths`` () =
    let pnet = loadOk 7 0u false
    let rng = Random(42)
    let ft = Array.zeroCreate<byte> Eonego.NNUE.L1

    for trial in 1..8 do
        // ft values live in [0,127] (the FT pairwise-product range the saturation proof assumes);
        // sprinkle zero chunks so the input resembles the real sparse activation.
        for i in 0 .. ft.Length - 1 do
            ft.[i] <- if rng.Next(3) = 0 then 0uy else byte (rng.Next(128))

        let fScalar = Array.zeroCreate<int> HeadOut
        let tScalar = Array.zeroCreate<int> HeadOut
        forwardFromFt pnet (ft.AsSpan()) (fScalar.AsSpan()) (tScalar.AsSpan()) false false

        if System.Runtime.Intrinsics.X86.Avx2.IsSupported then
            let fAvx = Array.zeroCreate<int> HeadOut
            let tAvx = Array.zeroCreate<int> HeadOut
            forwardFromFt pnet (ft.AsSpan()) (fAvx.AsSpan()) (tAvx.AsSpan()) true false
            Assert.Equal<int[]>(fScalar, fAvx)
            Assert.Equal<int[]>(tScalar, tAvx)

        if System.Runtime.Intrinsics.X86.AvxVnni.IsSupported then
            let fVnni = Array.zeroCreate<int> HeadOut
            let tVnni = Array.zeroCreate<int> HeadOut
            forwardFromFt pnet (ft.AsSpan()) (fVnni.AsSpan()) (tVnni.AsSpan()) true true
            Assert.Equal<int[]>(fScalar, fVnni)
            Assert.Equal<int[]>(tScalar, tVnni)

        ignore trial

[<Fact>]
let ``relSq is identity for White and a vertical flip for Black`` () =
    Assert.Equal(0, relSq White 0)
    Assert.Equal(63, relSq White 63)
    Assert.Equal(56, relSq Black 0) // a1 -> a8
    Assert.Equal(7, relSq Black 63) // h8 -> h1
    Assert.Equal(28, relSq Black 36) // e5 -> e4

// ---------------------------------------------------------------------------
// Net-dependent tests (soft-skip when nets/main.nnue is absent)
// ---------------------------------------------------------------------------
[<Fact>]
let ``fillLogits is deterministic and position-dependent`` () =
    match tryLoadNet () with
    | None -> ()
    | Some net ->
        let pnet = loadOk 11 net.FtHash false
        let pos1 = Eonego.Position.Position.OfFen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        let pos2 = Eonego.Position.Position.OfFen "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
        let fA = Array.zeroCreate<int> HeadOut
        let tA = Array.zeroCreate<int> HeadOut
        let fB = Array.zeroCreate<int> HeadOut
        let tB = Array.zeroCreate<int> HeadOut
        fillLogits net pnet pos1 (fA.AsSpan()) (tA.AsSpan())
        fillLogits net pnet pos1 (fB.AsSpan()) (tB.AsSpan())
        Assert.Equal<int[]>(fA, fB)
        Assert.Equal<int[]>(tA, tB)
        fillLogits net pnet pos2 (fB.AsSpan()) (tB.AsSpan())
        Assert.NotEqual<int[]>(fA, fB)

[<Fact>]
let ``evalWDL returns a per-mille distribution`` () =
    match tryLoadNet () with
    | None -> ()
    | Some net ->
        let pnet = loadOk 13 net.FtHash true
        let pos = Eonego.Position.Position.OfFen "r1bq1rk1/pp2bppp/2n1pn2/3p4/3P4/2NBPN2/PP3PPP/R1BQ1RK1 w - - 0 9"
        let struct (w, d, l) = evalWDL net pnet pos
        Assert.InRange(w, 0, 1000)
        Assert.InRange(d, 0, 1000)
        Assert.InRange(l, 0, 1000)
        Assert.Equal(1000, w + d + l)

[<Fact>]
let ``a loaded sidecar with UsePolicy=false is byte-identical to no sidecar`` () =
    match tryLoadNet () with
    | None -> ()
    | Some net ->
        let pnet = loadOk 17 net.FtHash false
        let cfg = { Eonego.Search.defaultConfig with HashMb = 16 }
        let fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
        let struct (s0, n0, m0) = Eonego.Search.searchToDepthNet fen [||] 9 cfg (Some net)
        let struct (s1, n1, m1) = Eonego.Search.searchToDepthPolicy fen [||] 9 cfg (Some net) (Some pnet)
        Assert.Equal(s0, s1)
        Assert.Equal(n0, n1)
        Assert.Equal(m0, m1)

[<Fact>]
let ``the policy LMR term actually fires (parent-key guard regression)`` () =
    // Regression for the 2026-07-06 wiring bug: the LMR site runs AFTER pos.Make, so guarding on
    // pos.Key there compared the CHILD's key against the parent-filled slot and the term was dead
    // (tree byte-identical to OFF while the fill provably ran). With random logits and the default
    // PolLmrThresh = 0, roughly half the quiets draw the extra reduction — the tree MUST differ.
    match tryLoadNet () with
    | None -> ()
    | Some net ->
        let pnet = loadOk 23 net.FtHash false
        let cfg = { Eonego.Search.defaultConfig with HashMb = 16 }
        let fen = "r1bq1rk1/pp2bppp/2n1pn2/3p4/3P4/2NBPN2/PP3PPP/R1BQ1RK1 w - - 0 9"
        let struct (_, nOff, _) = Eonego.Search.searchToDepthNet fen [||] 10 cfg (Some net)

        let struct (_, nOn, _) =
            Eonego.Search.searchToDepthPolicy fen [||] 10 { cfg with UsePolicy = true } (Some net) (Some pnet)

        Assert.NotEqual(nOff, nOn)

[<Fact>]
let ``random-weights policy search is legal and deterministic`` () =
    match tryLoadNet () with
    | None -> ()
    | Some net ->
        let pnet = loadOk 19 net.FtHash false
        let cfg = { Eonego.Search.defaultConfig with HashMb = 16; UsePolicy = true }
        let fen = "r1bq1rk1/pp2bppp/2n1pn2/3p4/3P4/2NBPN2/PP3PPP/R1BQ1RK1 w - - 0 9"
        let struct (_, n0, m0) = Eonego.Search.searchToDepthPolicy fen [||] 9 cfg (Some net) (Some pnet)
        let struct (_, n1, m1) = Eonego.Search.searchToDepthPolicy fen [||] 9 cfg (Some net) (Some pnet)
        Assert.Equal(n0, n1)
        Assert.Equal(m0, m1)
        Assert.NotEqual(MoveNone, m0)
        let pos = Eonego.Position.Position.OfFen fen
        let legal = collectLegal pos
        Assert.Contains(m0, legal)

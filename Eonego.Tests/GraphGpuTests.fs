module Eonego.Tests.GraphGpuTests

open System
open System.IO
open Xunit
open Eonego.GraphGpu
open Eonego.GraphFeatures
open Eonego.Search
open Eonego.Tests.TestFixtures
open Eonego.Position

let private writeModel (path: string) (schema: uint32) =
    use bw = new BinaryWriter(File.Create path)
    bw.Write(ModelMagic)
    bw.Write(ModelVersion)
    bw.Write(int schema)
    bw.Write(128)
    bw.Write(3)
    bw.Write(3)
    bw.Write(600)
    bw.Write(0)
    bw.Flush()

[<Fact>]
let ``EONGR01 loader accepts a valid header`` () =
    let path = Path.Combine(Path.GetTempPath(), "eonego-valid.eongr01")
    writeModel path SchemaHash

    match load path with
    | GraphLoaded n ->
        Assert.Equal(path, n.Path)
        Assert.Equal(128, n.DModel)
        Assert.Equal(3, n.Layers)
        Assert.True(n.HasPolicy)
        Assert.True(n.HasWdl)
    | GraphFailed why -> failwith ("expected graph load, got " + why)

[<Fact>]
let ``EONGR01 loader rejects a foreign feature schema`` () =
    let path = Path.Combine(Path.GetTempPath(), "eonego-bad-schema.eongr01")
    writeModel path 0x12345678u

    match load path with
    | GraphFailed why -> Assert.Contains("schema", why)
    | GraphLoaded _ -> failwith "loaded a model with the wrong feature schema"

let private fallbackGraph () =
    let path = Path.Combine(Path.GetTempPath(), "eonego-fallback.eongr01")
    writeModel path SchemaHash
    let n =
        match load path with
        | GraphLoaded n -> n
        | GraphFailed why -> failwith why
    new GraphBatcher(n, GraphMode.Leaf, 0, 16, 0)

[<Fact>]
let ``GraphBatcher reports fallback instead of fake inference`` () =
    use b = fallbackGraph ()
    let pos = Position.OfFen "4k3/8/8/8/4N3/8/8/4K3 w - - 0 1"
    Assert.Equal(ValueNone, b.TryEvaluate(pos, false))
    Assert.Equal(1L, b.FallbackCount)
    Assert.False(b.SupportsInference)

[<Fact>]
let ``GraphBatcher empty batch succeeds without native inference`` () =
    use b = fallbackGraph ()

    match b.TryEvaluateBatch([||], true) with
    | ValueSome xs -> Assert.Empty xs
    | ValueNone -> failwith "empty graph batch should not require native inference"

    Assert.Equal(0L, b.FallbackCount)

[<Fact>]
let ``GraphBatcher batch fallback counts every requested position`` () =
    use b = fallbackGraph ()
    let positions =
        [| Position.OfFen "4k3/8/8/8/4N3/8/8/4K3 w - - 0 1"
           Position.OfFen "4k3/8/8/3q4/8/8/8/4K3 w - - 0 1"
           Position.OfFen "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1" |]

    Assert.Equal(ValueNone, b.TryEvaluateBatch(positions, true))
    Assert.Equal(int64 positions.Length, b.FallbackCount)
    Assert.False(b.SupportsInference)

[<Fact>]
let ``graph fallback search is byte-identical to NNUE search`` () =
    match tryLoadNet () with
    | None -> ()
    | Some net ->
        let path = Path.Combine(Path.GetTempPath(), "eonego-search-fallback.eongr01")
        writeModel path SchemaHash
        let gnet =
            match load path with
            | GraphLoaded n -> n
            | GraphFailed why -> failwith why
        use graph = new GraphBatcher(gnet, GraphMode.Leaf, 0, 16, 0)
        let fen = "r1bq1rk1/pp2bppp/2n1pn2/3p4/3P4/2NBPN2/PP3PPP/R1BQ1RK1 w - - 0 9"
        let cfg = { defaultConfig with HashMb = 16; UseGraph = true; GraphMode = GraphMode.Leaf }
        let struct (s0, n0, m0) = searchToDepthNet fen [||] 8 { cfg with UseGraph = false } (Some net)
        let tt = Eonego.Transposition.TranspositionTable(16)
        let control =
            SearchControl(cfg, { defaultLimits with Depth = 8 }, tt, fen, [||], ?net = Some net, ?graph = Some graph)
        control.Reset()
        control.NewSearch()
        let w = Worker(0, true, control)
        w.SetupRoot()
        control.StartClock 0L 0L
        let s1 = negamax w w.Pos (-INF) INF 8 0 true false
        Assert.Equal(s0, s1)
        Assert.Equal(n0, w.Nodes)
        Assert.Equal(m0, w.Pv.[0])
        Assert.False(graph.SupportsInference)

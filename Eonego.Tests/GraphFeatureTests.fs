module Eonego.Tests.GraphFeatureTests

open System
open System.IO
open Xunit
open Eonego.Bitboard
open Eonego.GraphFeatures
open Eonego.Position

let private nodeField (r: GraphRecord) (i: int) (f: int) =
    int r.Nodes.[i * NodeFeatureCount + f]

let private edgeField (r: GraphRecord) (i: int) (j: int) (f: int) =
    int r.Edges.[((i * MaxGraphNodes) + j) * EdgeFeatureCount + f]

let private findNode (r: GraphRecord) (pt: int) (absSq: int) =
    let mutable found = -1

    for i in 0 .. r.NodeCount - 1 do
        if nodeField r i 3 = pt && nodeField r i 4 = absSq then
            found <- i

    Assert.True(found >= 0, "node not found")
    found

[<Fact>]
let ``graph extraction is deterministic and fixed width`` () =
    let fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
    let a = extract (Position.OfFen fen)
    let b = extract (Position.OfFen fen)
    Assert.Equal(a.NodeCount, b.NodeCount)
    Assert.Equal<byte[]>(a.Nodes, b.Nodes)
    Assert.Equal<byte[]>(a.Edges, b.Edges)
    Assert.Equal<byte[]>(a.Globals, b.Globals)
    Assert.Equal(MaxGraphNodes * NodeFeatureCount, a.Nodes.Length)
    Assert.Equal(MaxGraphNodes * MaxGraphNodes * EdgeFeatureCount, a.Edges.Length)
    Assert.Equal(GlobalFeatureCount, a.Globals.Length)

[<Fact>]
let ``start position uses all thirty two piece nodes`` () =
    let r = extract (Position.OfFen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    Assert.Equal(32, r.NodeCount)
    Assert.Equal(32, int r.Globals.[6])

[<Fact>]
let ``relative square follows side to move perspective`` () =
    let w = extract (Position.OfFen "4k3/8/8/8/4N3/8/8/4K3 w - - 0 1")
    let b = extract (Position.OfFen "4k3/8/8/8/4N3/8/8/4K3 b - - 0 1")
    let e4 = 28
    let wi = findNode w Knight e4
    let bi = findNode b Knight e4
    Assert.Equal(e4, nodeField w wi 5)
    Assert.Equal(e4 ^^^ 56, nodeField b bi 5)

[<Fact>]
let ``edge features mark direct piece attacks`` () =
    let r = extract (Position.OfFen "4k3/8/5p2/8/4N3/8/8/4K3 w - - 0 1")
    let n = findNode r Knight 28 // e4
    let p = findNode r Pawn 45   // f6
    Assert.Equal(1, edgeField r n p 2)
    Assert.Equal(1, edgeField r p n 3)

[<Fact>]
let ``binary record writer emits the documented fixed size`` () =
    let r = extract (Position.OfFen "4k3/8/8/8/4N3/8/8/4K3 w - - 0 1")
    use ms = new MemoryStream()
    use bw = new BinaryWriter(ms)
    writeRecord bw r
    bw.Flush()
    Assert.Equal(int64 RecordBytes, ms.Length)
    Assert.Equal<byte[]>(ms.ToArray(), toBytes r)

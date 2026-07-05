/// FullThreats feature-enumeration invariants. Without a reference implementation we can't bit-exact-verify
/// the threat indices, but we pin the structural invariants: every active index is in [0, Dimensions), the
/// active count never exceeds the active-feature cap (128), and a normal middlegame has a non-empty,
/// perspective-dependent threat set. (The material-accurate evals in NNUETests are the broader evidence.)
module Eonego.Tests.ThreatsTests

open System.Collections.Generic
open Xunit
open Eonego.Bitboard
open Eonego.Accumulator
open Eonego.Move
open Eonego.Position
open Eonego.Threats
open Eonego.Tests.TestFixtures

let private fens =
    [ "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" // startpos
      "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1" // kiwipete
      "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1" // endgame
      "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3" ] // en-passant available

[<Fact>]
let ``active threat indices are in range and bounded by 128`` () =
    let buf = Array.zeroCreate 256

    for fen in fens do
        let pos = Position.OfFen fen

        for persp in [ White; Black ] do
            let n = appendActiveThreats persp pos buf
            Assert.True(n <= 128, sprintf "%s persp=%d: %d active features > 128" fen persp n)

            for i in 0 .. n - 1 do
                Assert.True(buf.[i] >= 0 && buf.[i] < Dimensions, sprintf "index %d out of [0,%d)" buf.[i] Dimensions)

[<Fact>]
let ``startpos has a non-empty threat set for both sides`` () =
    let buf = Array.zeroCreate 256
    let pos = Position.OfFen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    // Minor/major pieces defend their own pawns (e.g. Nb1->d2, Ra1->a2, Qd1->{c2,d2,e2}) ⇒ several threats.
    Assert.True(appendActiveThreats White pos buf > 4, "white startpos threat set too small")
    Assert.True(appendActiveThreats Black pos buf > 4, "black startpos threat set too small")

let private addCount (d: Dictionary<int, int>) (idx: int) (delta: int) =
    let mutable old = 0

    if d.TryGetValue(idx, &old) then
        let v = old + delta

        if v = 0 then
            d.Remove(idx) |> ignore
        else
            d.[idx] <- v
    else
        d.[idx] <- delta

let private activeCounts (persp: int) (pos: Position) =
    let d = Dictionary<int, int>()
    let buf = Array.zeroCreate MaxActive
    let n = appendActiveThreats persp pos buf

    for i in 0 .. n - 1 do
        addCount d buf.[i] 1

    d

let private addSignedDeltas (d: Dictionary<int, int>) (buf: int[]) (n: int) =
    for i in 0 .. n - 1 do
        let v = buf.[i]
        let sign, idx = if v > 0 then 1, v - 1 else -1, -v - 1
        addCount d idx sign

let private asSortedPairs (d: Dictionary<int, int>) =
    d
    |> Seq.map (fun kv -> kv.Key, kv.Value)
    |> Seq.sortBy fst
    |> Seq.toArray

let private assertCountsEqual label expected actual =
    let e = asSortedPairs expected
    let a = asSortedPairs actual
    Assert.True((e = a), sprintf "%s expected=%A actual=%A" label e a)

let private checkDirtyMove (pos: Position) (m: Move) =
    let fen = pos.ToFen()
    let beforeW = activeCounts White pos
    let beforeB = activeCounts Black pos
    let oldKsqW = pos.KingSquare White
    let oldKsqB = pos.KingSquare Black
    let dirty = Array.zeroCreate MaxDirtyThreats
    let dirtyN = pos.DebugCollectDirtyThreats(m, dirty)
    Assert.True(dirtyN >= 0, sprintf "dirty threat overflow for %s in %s" (toUCI m) (pos.ToFen()))

    pos.Make m
    let changedW = Array.zeroCreate MaxDirtyThreats
    let changedB = Array.zeroCreate MaxDirtyThreats
    let packed = appendChangedThreatsBoth pos dirty dirtyN changedW changedB
    let nW = int (packed >>> 32)
    let nB = int (packed &&& 0xFFFFFFFFL)

    if oldKsqW = pos.KingSquare White then
        addSignedDeltas beforeW changedW nW
        assertCountsEqual ("white " + toUCI m + " in " + fen) beforeW (activeCounts White pos)

    if oldKsqB = pos.KingSquare Black then
        addSignedDeltas beforeB changedB nB
        assertCountsEqual ("black " + toUCI m + " in " + fen) beforeB (activeCounts Black pos)

    pos.Unmake m

[<Fact>]
let ``appendActiveThreatsBoth equals separate perspective enumeration`` () =
    let bufW = Array.zeroCreate MaxActive
    let bufB = Array.zeroCreate MaxActive
    let sep = Array.zeroCreate MaxActive

    for fen in fens do
        let pos = Position.OfFen fen
        let packed = appendActiveThreatsBoth pos bufW bufB
        let nW = int (packed >>> 32)
        let nB = int (packed &&& 0xFFFFFFFFL)
        let sepW = appendActiveThreats White pos sep
        Assert.Equal<int[]>(Array.sort bufW.[0 .. nW - 1], Array.sort sep.[0 .. sepW - 1])
        let sepB = appendActiveThreats Black pos sep
        Assert.Equal<int[]>(Array.sort bufB.[0 .. nB - 1], Array.sort sep.[0 .. sepB - 1])

[<Fact>]
let ``dirty FullThreats deltas match full active threat diff`` () =
    let rec walk (pos: Position) depth =
        if depth > 0 then
            for m in collectLegal pos do
                checkDirtyMove pos m
                pos.Make m
                walk pos (depth - 1)
                pos.Unmake m

    let cases =
        [ "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
          "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"
          "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1"
          "4r2k/8/8/8/8/8/4N3/4K3 w - - 0 1"
          "8/8/8/R2pP2k/8/8/8/K7 w - d6 0 1" ]

    for fen in cases do
        walk (Position.OfFen fen) 2

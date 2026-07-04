module Eonego.Tests.RetrogradeTests

open System
open System.Diagnostics
open Xunit
open Eonego.Bitboard
open Eonego.Move
open Eonego.Position
open Eonego.Search
open Eonego.Retrograde
open Eonego.Tests.TestFixtures

let private proveAndTime (pce: Piece) =
    let sw = Stopwatch.StartNew()
    let res = verifySolved pce
    sw.Stop()
    Assert.True(res.IsNone, defaultArg res (sprintf "verifySolved failed for %d" pce))
    Assert.InRange(sw.Elapsed.TotalSeconds, 0.0, 30.0)
    Console.WriteLine(sprintf "verifySolved %d: %.3f ms" pce sw.Elapsed.TotalMilliseconds)
    sw.Elapsed

[<Fact>]
let ``Retrograde index bijection round-trips sampled indices`` () =
    let samples =
        seq {
            yield 0
            yield RetroSize - 1
            for idx in 0 .. 1009 .. RetroSize - 1 do
                yield idx
        }
        |> Seq.distinct

    for idx in samples do
        let stm = idxStm idx
        let wk = idxWk idx
        let bk = idxBk idx
        let pc = idxPc idx

        Assert.InRange(stm, 0, 1)
        Assert.InRange(wk, 0, 63)
        Assert.InRange(bk, 0, 63)
        Assert.InRange(pc, 0, 63)
        Assert.Equal(idx, idxOf stm wk bk pc)

[<Fact>]
let ``Retrograde verifySolved white knight proves the table`` () =
    let _ = proveAndTime (makePiece White Knight)
    ()

[<Fact>]
let ``Retrograde verifySolved white bishop proves the table`` () =
    let _ = proveAndTime (makePiece White Bishop)
    ()

[<Fact>]
let ``Retrograde verifySolved white rook proves the table`` () =
    let _ = proveAndTime (makePiece White Rook)
    ()

[<Fact>]
let ``Retrograde verifySolved white queen proves the table`` () =
    let _ = proveAndTime (makePiece White Queen)
    ()

[<Fact>]
let ``Retrograde verifySolved white pawn proves the table`` () =
    let _ = proveAndTime (makePiece White Pawn)
    ()

[<Fact>]
let ``Retrograde stats invariants hold after solving`` () =
    let check pce expectNoMates =
        verifySolved pce |> ignore
        let struct (legal, wins, losses, maxWin, maxLoss) = statsOf (solvedTable pce)

        Assert.True(legal > 0, sprintf "expected legal positions for %d" pce)

        if expectNoMates then
            Assert.Equal(0, wins)
            Assert.Equal(0, losses)
        else
            Assert.True(wins > 0, sprintf "expected wins for %d" pce)
            Assert.True(losses > 0, sprintf "expected losses for %d" pce)
            Assert.True(maxWin > 0, sprintf "expected maxWin for %d" pce)
            Assert.True(maxLoss > 0, sprintf "expected maxLoss for %d" pce)

        Console.WriteLine(sprintf "stats %d: legal=%d wins=%d losses=%d maxWin=%d maxLoss=%d" pce legal wins losses maxWin maxLoss)
        legal, wins, losses, maxWin, maxLoss

    let _ = check (makePiece White Knight) true
    let _ = check (makePiece White Bishop) true
    let _ = check (makePiece White Rook) false
    let _ = check (makePiece White Queen) false
    let _ = check (makePiece White Pawn) false
    ()

[<Fact>]
let ``Retrograde probe maps terminal KQK positions correctly`` () =
    let mateFen = "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1"
    let matePos = Position.OfFen mateFen
    Assert.True(matePos.InCheck)
    Assert.Empty(collectLegal matePos)
    ensureSolved (makePiece White Queen)
    Assert.Equal(ValueSome -1y, probe matePos)

    let staleFen = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"
    let stalePos = Position.OfFen staleFen
    Assert.False(stalePos.InCheck)
    Assert.Empty(collectLegal stalePos)
    Assert.Equal(ValueSome 0y, probe stalePos)

[<Fact>]
let ``Retrograde probe sees a mate in one for White`` () =
    let fen = "8/8/8/8/8/8/8/k1KQ4 w - - 0 1"
    let pos = Position.OfFen fen
    ensureSolved (makePiece White Queen)
    match probe pos with
    | ValueSome v ->
        Assert.True(v > 0y, sprintf "expected winning probe value, got %d" (int v))
        Assert.Equal(1, retroDtm v)
    | ValueNone -> Assert.Fail("expected retrograde probe hit")

[<Fact>]
let ``Retrograde probe rejects non-3-man and castling-right positions`` () =
    ensureSolved (makePiece White Queen)
    Assert.Equal(ValueNone, probe (Position.OfFen "4k3/8/8/8/8/8/8/3QK2R w K - 0 1"))
    Assert.Equal(ValueNone, probe (Position.OfFen "4k3/8/8/8/8/8/8/R3K3 w Q - 0 1"))

[<Fact>]
let ``Retrograde end-to-end search soft-skips without a net`` () =
    match tryLoadNet () with
    | None -> ()
    | Some net ->
        ensureSolved (makePiece White Queen)
        let fen = "8/8/8/8/8/8/8/k1KQ4 w - - 0 1"
        let struct (score, _, _) = searchToDepthNet fen [||] 4 defaultConfig (Some net)
        Assert.True(score >= MATE_IN_MAX_PLY, sprintf "expected mate score, got %d" score)

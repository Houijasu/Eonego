/// MovePick stage-machine verification: the picker yields exactly the legal move set (cross-checked vs
/// generateLegal), the TT move is first-and-once, good captures precede quiets and bad captures trail them,
/// killers/counter occupy the refutation slot, the evasion/qsearch/probcut chains behave, and skipQuiets is
/// structurally lazy (quiets are never generated when skipped).
module Eonego.Tests.MovePickTests

#nowarn "9" // NativePtr.stackalloc

open System
open System.Collections.Generic
open Microsoft.FSharp.NativeInterop
open Xunit
open Eonego.Bitboard
open Eonego.Move
open Eonego.Position
open Eonego.MoveGeneration
open Eonego.History
open Eonego.MovePick
open Eonego.Tests.TestFixtures

let private sq (f: int) (r: int) : Square = mkSquare f r
let private startFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

let private drainMain
    (pos: Position)
    (tables: Tables)
    (tt: Move)
    (k1: Move)
    (k2: Move)
    (cm: Move)
    (depth: int)
    (skipQ: bool)
    : Move[] =
    let pm = NativePtr.stackalloc<Move> MaxMoves
    let moves = Span<Move>(NativePtr.toVoidPtr pm, MaxMoves)
    let ps = NativePtr.stackalloc<int> MaxMoves
    let scores = Span<int>(NativePtr.toVoidPtr ps, MaxMoves)
    let mutable mp = mkMain pos tables tt k1 k2 cm depth -1 -1 -1 -1 moves scores
    let r = ResizeArray<Move>()
    let mutable m = nextMove &mp skipQ

    while m <> MoveNone do
        r.Add m
        m <- nextMove &mp skipQ

    r.ToArray()

let private drainQSearch (pos: Position) (tables: Tables) (tt: Move) : Move[] =
    let pm = NativePtr.stackalloc<Move> MaxMoves
    let moves = Span<Move>(NativePtr.toVoidPtr pm, MaxMoves)
    let ps = NativePtr.stackalloc<int> MaxMoves
    let scores = Span<int>(NativePtr.toVoidPtr ps, MaxMoves)
    let mutable mp = mkQSearch pos tables tt moves scores
    let r = ResizeArray<Move>()
    let mutable m = nextMove &mp false

    while m <> MoveNone do
        r.Add m
        m <- nextMove &mp false

    r.ToArray()

let private drainProbCut (pos: Position) (tables: Tables) (tt: Move) (threshold: int) : Move[] =
    let pm = NativePtr.stackalloc<Move> MaxMoves
    let moves = Span<Move>(NativePtr.toVoidPtr pm, MaxMoves)
    let ps = NativePtr.stackalloc<int> MaxMoves
    let scores = Span<int>(NativePtr.toVoidPtr ps, MaxMoves)
    let mutable mp = mkProbCut pos tables tt threshold moves scores
    let r = ResizeArray<Move>()
    let mutable m = nextMove &mp false

    while m <> MoveNone do
        r.Add m
        m <- nextMove &mp false

    r.ToArray()

let private mpFens =
    [ yield! perftFens
      "4r2k/8/8/8/8/8/8/4K3 w - - 0 1" // in check (Re8)
      "k7/8/8/3pP3/4K3/8/8/8 w - d6 0 1" // ep as a check evasion
      "5rk1/8/8/8/8/8/8/4K2R w K - 0 1" // O-O present
      "7k/4P3/8/8/8/8/8/4K3 w - - 0 1" ] // promotion present

// (1) the drain, filtered by isLegal, is EXACTLY the generateLegal set, with no duplicates.
[<Fact>]
let ``main picker yields exactly the legal move set`` () =
    let tables = Tables()

    for fen in mpFens do
        let p = Position.OfFen fen
        let drained = drainMain p tables MoveNone MoveNone MoveNone MoveNone 8 false
        Assert.Equal(drained.Length, (HashSet<int>(drained)).Count) // no duplicates
        let pickerLegal = drained |> Array.filter (isLegal p) |> Array.sort
        let genLegal = collectLegal p |> Array.sort
        Assert.Equal<int[]>(genLegal, pickerLegal)

// (2) TT move first and exactly once, set still complete. Cover quiet / capture / promotion TT moves.
[<Fact>]
let ``TT move is emitted first and exactly once`` () =
    let tables = Tables()

    let check (fen: string) (tt: Move) =
        let p = Position.OfFen fen
        let drained = drainMain p tables tt MoveNone MoveNone MoveNone 8 false
        Assert.Equal(tt, drained.[0])
        Assert.Equal(1, drained |> Array.filter ((=) tt) |> Array.length)
        Assert.Equal((collectLegal p).Length, drained.Length)

    check startFen (mkMove (sq 4 1) (sq 4 3)) // quiet TT (e2e4)
    check "4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1" (mkMove (sq 4 3) (sq 3 4)) // capture TT (e4xd5)
    check "7k/4P3/8/8/8/8/8/4K3 w - - 0 1" (mkPromotion (sq 4 6) (sq 4 7) Queen) // promotion TT

// (3) good captures precede every quiet; bad captures trail every quiet.
[<Fact>]
let ``good captures lead, bad captures trail the quiets`` () =
    let tables = Tables()

    let classifyOrder (fen: string) =
        let p = Position.OfFen fen
        let drained = drainMain p tables MoveNone MoveNone MoveNone MoveNone 4 false
        let mutable lastGood = -1
        let mutable firstQuiet = Int32.MaxValue
        let mutable lastQuiet = -1
        let mutable firstBad = Int32.MaxValue

        drained
        |> Array.iteri (fun i m ->
            if isLegal p m then
                let cap = p.PieceOn(toSq m) <> NoPiece || isEnPassant m

                if cap && p.SeeGe m 0 then
                    lastGood <- max lastGood i
                elif cap then
                    firstBad <- min firstBad i
                else
                    (firstQuiet <- min firstQuiet i
                     lastQuiet <- max lastQuiet i))

        lastGood, firstQuiet, lastQuiet, firstBad
    // Kiwipete: rich in good captures + quiets
    let (lastGood, firstQuiet, _, _) = classifyOrder perftFens.[1]

    if lastGood >= 0 && firstQuiet < Int32.MaxValue then
        Assert.True(lastGood < firstQuiet, "a good capture appeared after a quiet")
    // bad-capture position: Nxd5 is the only capture (defended by Bc6) -> must trail the quiets
    let (_, _, lastQuiet, firstBad) =
        classifyOrder "4k3/8/2b5/3p4/8/4N3/8/4K3 w - - 0 1"

    Assert.True(firstBad < Int32.MaxValue && lastQuiet >= 0, "fixture must have a quiet and a bad capture")
    Assert.True(lastQuiet < firstBad, "a bad capture appeared before a quiet")

// (4) killers/counter occupy the refutation slot (after good captures, before the quiet mass), once each;
//     a capture supplied as a killer is rejected by the refutation gate but still appears as a capture.
[<Fact>]
let ``killers and counter fill the refutation slot`` () =
    let tables = Tables()
    // quiet-only position: Pe2 + Ke1
    let p = Position.OfFen "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"
    let k1 = mkMove (sq 4 1) (sq 4 3) // e2e4 (killer1)
    let cm = mkMove (sq 4 1) (sq 4 2) // e2e3 (countermove)
    let drained = drainMain p tables MoveNone k1 MoveNone cm 2 false
    let idx m = Array.findIndex ((=) m) drained
    Assert.Equal(1, drained |> Array.filter ((=) k1) |> Array.length)
    Assert.Equal(1, drained |> Array.filter ((=) cm) |> Array.length)
    Assert.True(idx k1 < idx cm, "killer1 should precede the countermove")
    let firstKing = drained |> Array.findIndex (fun m -> fromSq m = sq 4 0)
    Assert.True(idx cm < firstKing, "refutations should precede the general quiets")
    // a capture offered as a killer is NOT taken from the refutation slot (it is a capture)
    let pc = Position.OfFen "4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1"
    let capKiller = mkMove (sq 4 3) (sq 3 4) // e4xd5 is a capture
    let drained2 = drainMain pc tables MoveNone capKiller MoveNone MoveNone 2 false
    Assert.Equal(1, drained2 |> Array.filter ((=) capKiller) |> Array.length)
    Assert.True(pc.PieceOn(sq 3 4) <> NoPiece) // d5 really holds an enemy pawn
    Assert.Equal(0, drained2 |> Array.findIndex ((=) capKiller)) // emitted first (good capture)

// (5) evasion chain: capture-of-checker is ordered first; the legal-filtered drain == legal evasions.
[<Fact>]
let ``evasion picker orders the checker capture first`` () =
    let tables = Tables()
    let p = Position.OfFen "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1" // Re2 checks Ke1
    Assert.True(p.InCheck)
    let drained = drainMain p tables MoveNone MoveNone MoveNone MoveNone 4 false
    Assert.Equal(mkMove (sq 4 0) (sq 4 1), drained.[0]) // Kxe2 (captures the checker) first
    let pickerLegal = drained |> Array.filter (isLegal p) |> Array.sort
    Assert.Equal<int[]>(collectLegal p |> Array.sort, pickerLegal)

// (6) qsearch reaches captures only and INCLUDES the queen push-promotion (the promotion-split fix).
[<Fact>]
let ``qsearch includes queen push-promotion and no quiet king moves`` () =
    let tables = Tables()
    let p = Position.OfFen "7k/4P3/8/8/8/8/8/4K3 w - - 0 1" // Pe7 push-promotes; king quiets exist
    let drained = drainQSearch p tables MoveNone
    Assert.Contains(mkPromotion (sq 4 6) (sq 4 7) Queen, drained) // e7e8=Q present
    Assert.DoesNotContain(mkMove (sq 4 0) (sq 3 0), drained) // Kd1 (a quiet) absent

    Assert.True(
        drained
        |> Array.forall (fun m -> p.PieceOn(toSq m) <> NoPiece || isEnPassant m || isPromotion m),
        "qsearch yielded a plain quiet move"
    )

// (7) probcut: only captures passing SEE >= threshold are emitted.
[<Fact>]
let ``probcut gates captures by the SEE threshold`` () =
    let tables = Tables()
    let p = Position.OfFen "4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1" // e4xd5 wins the queen, SEE +900
    let e4d5 = mkMove (sq 4 3) (sq 3 4)
    Assert.Contains(e4d5, drainProbCut p tables MoveNone 500) // 900 >= 500 -> emitted
    Assert.DoesNotContain(e4d5, drainProbCut p tables MoveNone 1000) // 900 < 1000 -> rejected

// (8) skipQuiets emits only captures, and is structurally lazy (quiets never generated when skipped).
[<Fact>]
let ``skipQuiets yields only captures`` () =
    let tables = Tables()
    let p = Position.OfFen perftFens.[1] // Kiwipete (captures + quiets)
    let drained = drainMain p tables MoveNone MoveNone MoveNone MoveNone 4 true
    Assert.NotEmpty(drained)

    Assert.True(
        drained |> Array.forall (fun m -> p.PieceOn(toSq m) <> NoPiece || isEnPassant m),
        "skipQuiets emitted a non-capture move"
    )

[<Fact>]
let ``skipQuiets never generates quiets (structural laziness)`` () =
    let tables = Tables()
    let p = Position.OfFen startFen // 0 captures, 20 quiets
    let pm = NativePtr.stackalloc<Move> MaxMoves
    let moves = Span<Move>(NativePtr.toVoidPtr pm, MaxMoves)
    let ps = NativePtr.stackalloc<int> MaxMoves
    let scores = Span<int>(NativePtr.toVoidPtr ps, MaxMoves)
    let mutable mp = mkMain p tables MoveNone MoveNone MoveNone MoveNone 8 -1 -1 -1 -1 moves scores
    let first = nextMove &mp true
    Assert.Equal(MoveNone, first) // nothing to emit (quiets skipped)
    Assert.Equal(0, mp.EndMoves) // generate(Quiets) NEVER ran (would set ~20)

/// History.Tables: SF "gravity" updates must saturate within int16 (never overflow) and converge toward
/// the divisor D; counter-moves and killers must round-trip per (piece,to) / per ply.
module Eonego.Tests.HistoryTests

open Xunit
open Eonego.Bitboard
open Eonego.Move
open Eonego.History

let private sq (f: int) (r: int) : Square = mkSquare f r

[<Fact>]
let ``main history saturates at +D and -D without int16 overflow`` () =
    let m = mkMove (sq 4 1) (sq 4 3) // e2e4
    let tp = Tables()

    for _ in 1..1000 do
        tp.UpdateMain White m MainHistD

    Assert.Equal(MainHistD, tp.MainHistory White (fromTo m)) // pins exactly at +D
    let tn = Tables()

    for _ in 1..1000 do
        tn.UpdateMain White m (-MainHistD)

    Assert.Equal(-MainHistD, tn.MainHistory White (fromTo m)) // pins exactly at -D

[<Fact>]
let ``main history gravity converges from a moderate bonus and stays bounded`` () =
    let m = mkMove (sq 6 0) (sq 5 2) // g1f3
    let t = Tables()

    for _ in 1..2000 do
        t.UpdateMain White m 200 // small repeated bonus

    let v = t.MainHistory White (fromTo m)
    Assert.True(v > 0 && v <= MainHistD, sprintf "value %d not in (0, %d]" v MainHistD)
    Assert.True(v > MainHistD / 2, "gravity should converge well above half the cap")

[<Fact>]
let ``capture history saturates within int16`` () =
    let pc = makePiece White Knight
    let dst = sq 3 4
    let t = Tables()

    for _ in 1..1000 do
        t.UpdateCapture pc dst Pawn CaptureHistD

    Assert.Equal(CaptureHistD, t.CaptureHistory pc dst Pawn)
    Assert.True(t.CaptureHistory pc dst Pawn <= 32767) // fits int16
    Assert.Equal(0, t.CaptureHistory pc dst Bishop) // a different capturedPT slot is untouched

[<Fact>]
let ``counter-moves round-trip per (prevPiece, prevTo)`` () =
    let t = Tables()
    let prevPc = makePiece Black Knight
    let prevTo = sq 5 2
    let cm = mkMove (sq 4 1) (sq 4 3)
    t.SetCounter prevPc prevTo cm
    Assert.Equal(cm, t.CounterMove prevPc prevTo)
    Assert.Equal(MoveNone, t.CounterMove prevPc (sq 0 0)) // untouched slot

[<Fact>]
let ``killers slide and de-duplicate per ply`` () =
    let t = Tables()
    let k1 = mkMove (sq 1 0) (sq 2 2) // b1c3
    let k2 = mkMove (sq 6 0) (sq 5 2) // g1f3
    t.SetKiller 5 k1
    Assert.Equal(k1, t.Killer 5 0)
    t.SetKiller 5 k2
    Assert.Equal(k2, t.Killer 5 0)
    Assert.Equal(k1, t.Killer 5 1) // k1 slid to slot 1
    t.SetKiller 5 k2 // duplicate -> no change
    Assert.Equal(k2, t.Killer 5 0)
    Assert.Equal(k1, t.Killer 5 1)
    Assert.Equal(MoveNone, t.Killer 6 0) // a different ply is untouched

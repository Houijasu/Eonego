/// History.Tables: "gravity" updates must saturate within int16 (never overflow) and converge toward
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
let ``continuation history saturates within int16 and disables on the -1 sentinel`` () =
    let prevPc = makePiece Black Knight
    let prevTo = sq 5 2
    let pc = makePiece White Pawn
    let dst = sq 4 3
    let t = Tables()

    for _ in 1..1000 do
        t.UpdateCont1 prevPc prevTo pc dst ContHistD
        t.UpdateCont2 prevPc prevTo pc dst (-ContHistD)

    Assert.Equal(ContHistD, t.ContHistory1 prevPc prevTo pc dst) // pins at +D
    Assert.Equal(-ContHistD, t.ContHistory2 prevPc prevTo pc dst) // pins at -D
    Assert.True(t.ContHistory1 prevPc prevTo pc dst <= 32767) // fits int16
    Assert.Equal(0, t.ContHistory1 prevPc prevTo pc (sq 0 0)) // a different slot is untouched
    Assert.Equal(0, t.ContHistory1 -1 prevTo pc dst) // -1 prevPc sentinel reads 0

[<Fact>]
let ``continuation history -1 sentinel makes updates a no-op`` () =
    let pc = makePiece White Pawn
    let dst = sq 4 3
    let t = Tables()

    for _ in 1..1000 do
        t.UpdateCont1 -1 -1 pc dst ContHistD // disabled context: must not write anywhere

    // cont1/cont2 stay zero everywhere (spot-check a real context slot)
    Assert.Equal(0, t.ContHistory1 (makePiece Black Knight) (sq 5 2) pc dst)

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

[<Fact>]
let ``continuation correction history indexes every (side, piece, square) without overflow`` () =
    // Regression (crash: IndexOutOfRangeException "when black to move"): corrCont was sized 2*768 but
    // indexed (c <<< 10) ||| (prevPc*64 + prevTo) — stride 1024, not 768. For Black (c = 1) the base
    // is 1024, so a previous King/Queen move (piece code >= 8: WQueen = 8, WKing = 10) gives an index
    // >= 1536 and the checked read throws. Exercise the WHOLE valid input space; must never throw.
    let t = Tables()

    for c in [ White; Black ] do
        for pt in [ Pawn; Knight; Bishop; Rook; Queen; King ] do
            for owner in [ White; Black ] do
                let prevPc = makePiece owner pt

                for prevTo in 0..63 do
                    t.UpdateCorrCont c prevPc prevTo 128 // write: index must be in range
                    t.CorrHistCont c prevPc prevTo |> ignore // read: index must be in range

    // the specific case that used to crash (Black to move, previous move by the White king)
    Assert.NotEqual(0, t.CorrHistCont Black (makePiece White King) 60)

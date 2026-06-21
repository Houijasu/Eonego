/// Verifies the Phase-2 promotion split in generate(): the QUEEN promotion is a CAPTURES-class move (so a
/// captures-only qsearch sees queen push-promotions), under-promotions split capture vs quiet, and
/// Evasions/NonEvasions still emit all four (perft-preserving). Independent of the MovePick.
module Eonego.Tests.PromotionSplitTests

#nowarn "9" // NativePtr.stackalloc

open System
open Microsoft.FSharp.NativeInterop
open Xunit
open Eonego.Bitboard
open Eonego.Move
open Eonego.Position
open Eonego.MoveGeneration

let private sq (f: int) (r: int) : Square = mkSquare f r

let private gen (fen: string) (genType: int) : Move[] =
    let p = Position.OfFen fen
    let pm = NativePtr.stackalloc<Move> MaxMoves
    let buf = Span<Move>(NativePtr.toVoidPtr pm, MaxMoves)
    let n = generate p buf genType
    let r = Array.zeroCreate n

    for i in 0 .. n - 1 do
        r.[i] <- buf.[i]

    r

let private promoKinds (ms: Move[]) (from: Square) (dst: Square) : int[] =
    ms
    |> Array.filter (fun m -> isPromotion m && fromSq m = from && toSq m = dst)
    |> Array.map promoType
    |> Array.sort

[<Fact>]
let ``queen push-promotion is a capture-class move; under-promos are quiet`` () =
    let fen = "7k/4P3/8/8/8/8/8/4K3 w - - 0 1" // Pe7 push-promotes onto the empty e8
    let e7, e8 = sq 4 6, sq 4 7
    Assert.Equal<int[]>([| Queen |], promoKinds (gen fen Captures) e7 e8) // Captures: Q only
    Assert.Equal<int[]>([| Knight; Bishop; Rook |], promoKinds (gen fen Quiets) e7 e8) // Quiets: under-promos only
    Assert.Equal<int[]>([| Knight; Bishop; Rook; Queen |], promoKinds (gen fen NonEvasions) e7 e8) // all four

[<Fact>]
let ``capture-promotions all land in Captures, none in Quiets`` () =
    let fen = "3rk3/4P3/8/8/8/8/8/4K3 w - - 0 1" // Pe7 captures-promotes on d8 (black rook)
    let e7, d8 = sq 4 6, sq 3 7
    Assert.Equal<int[]>([| Knight; Bishop; Rook; Queen |], promoKinds (gen fen Captures) e7 d8)
    Assert.Empty(promoKinds (gen fen Quiets) e7 d8)

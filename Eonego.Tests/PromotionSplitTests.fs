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
open Eonego.History
open Eonego.Transposition
open Eonego.Search

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

[<Fact>]
let ``a push-promotion cutoff does not teach the captured-pawn history slot`` () =
    // Queen push-promotions are deliberately staged with captures, but no piece is captured on e8.
    // The search's history classification must therefore not turn pieceType(NoPiece) into Pawn and
    // contaminate the capture-history slot for an imaginary pawn victim.
    let fen = "8/4P3/7k/8/8/8/8/4K3 w - - 0 1"

    let cfg =
        { defaultConfig with
            UseTt = false
            UseRetro = false
            UseSyzygy = false
            UseCorrHist = false }

    let control = SearchControl(cfg, { defaultLimits with Depth = 1 }, TranspositionTable(1), fen, [||])
    control.Reset()
    control.NewSearch()
    let w = Worker(0, true, control)
    w.SetupRoot()
    control.StartClock 0L 0L

    let e8 = sq 4 7
    let whitePawn = makePiece White Pawn
    Assert.Equal(0, w.Tables.CaptureHistory whitePawn e8 Pawn)

    // Neutral eval plus beta=0 makes the first (queen) promotion fail high and exercise the
    // cutoff-history update deterministically.
    let score = negamax w w.Pos -1 0 1 0 false false
    Assert.True(score >= 0)
    Assert.Equal(0, w.Tables.CaptureHistory whitePawn e8 Pawn)
    Assert.True(w.Tables.CaptureHistory whitePawn e8 NoPieceType > 0)

[<Fact>]
let ``late quiet pruning still searches a noncapturing underpromotion`` () =
    let fen = "8/4P3/7k/8/8/8/8/4K3 w - - 0 1"
    let tt = TranspositionTable(1)

    let cfg =
        { defaultConfig with
            UseTt = true
            UsePruning = true
            UseRazoring = false
            UseRetro = false
            UseSyzygy = false
            UseCorrHist = false
            UsePolicy = false
            UsePawnHist = false }

    let control = SearchControl(cfg, defaultLimits, tt, fen, [||])
    control.Reset()
    control.NewSearch()
    let w = Worker(0, true, control)
    w.SetupRoot()
    control.NodeSum <- (fun () -> w.Nodes)
    control.StartClock 0L 0L

    let e1, e7, e8 = sq 4 0, sq 4 6, sq 4 7
    w.Tables.UpdateMain White (mkPromotion e7 e8 Knight) (-MainHistD)

    for dst in [ sq 3 0; sq 5 0; sq 3 1; sq 4 1; sq 5 1 ] do
        w.Tables.UpdateMain White (mkMove e1 dst) MainHistD

    // A high-history king move reaches the futility gate first and sets global skipQuiets. The picker
    // must still surface the later R/B/N promotions; the rook child's qsearch TT entry proves it did.
    negamax w w.Pos 1000 1001 1 0 false false |> ignore
    let rookChild = Position.OfFen fen
    rookChild.Make(mkPromotion e7 e8 Rook)
    let struct (hit, _, _, _, _, _, _) = tt.Probe(ttKey rookChild)
    Assert.True(hit, "futility skipQuiets swallowed the later rook underpromotion")

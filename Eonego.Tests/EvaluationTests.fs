module Eonego.Tests.EvaluationTests

open System
open Xunit
open Eonego.Bitboard
open Eonego.Move
open Eonego.Position
open Eonego.MoveGeneration
open Eonego.Evaluation
open Eonego.Tests.TestFixtures

// ---------------------------------------------------------------------------
// Mirror test helper (pure FEN transform). Realizes M(p): vertical flip (sq ^^^ 56) + color swap, with
// side-to-move KEPT, so that eval(p) == -eval(M p). Reverse the ORDER of the 8 rank substrings ONLY —
// reversing characters within a rank would be a 180° rotation (s^63), the classic bug.
// VALID ONLY because PST-only eval ignores castling/ep (blanked to "-"); revisit if eval grows new terms.
// ---------------------------------------------------------------------------
let private boardField (fen: string) = fen.Split(' ').[0]

let mirrorFen (fen: string) : string =
    let parts = fen.Split(' ')
    let flipped = parts.[0].Split('/') |> Array.rev // vertical flip (rank r <-> 7-r)

    let swapCase (s: string) =
        String(
            s.ToCharArray()
            |> Array.map (fun ch ->
                if Char.IsUpper ch then Char.ToLower ch
                elif Char.IsLower ch then Char.ToUpper ch
                else ch)
        )

    let board = flipped |> Array.map swapCase |> String.concat "/"
    sprintf "%s %s - - 0 1" board parts.[1] // keep side-to-move; castling/ep inert

let private assertSymmetric (p: Position) =
    let mirrored = Position.OfFen(mirrorFen (p.ToFen())) // fresh instance — no Make aliasing
    Assert.Equal(eval p, -(eval mirrored))

// ---------------------------------------------------------------------------
// Helper sanity — board-string level, independent of eval.
// ---------------------------------------------------------------------------
[<Fact>]
let ``mirrorFen flips vertically + swaps colors (startpos is a fixed point, Kiwipete is not)`` () =
    Assert.Equal(boardField StartPosFen, boardField (mirrorFen StartPosFen)) // catches 180° (s^63) bug
    Assert.True(boardField perftFens.[1] <> boardField (mirrorFen perftFens.[1]))

// ---------------------------------------------------------------------------
// 1. Symmetry gate (most important): eval(p) == -eval(mirror p), EXACT, over canonical FENs + a seeded
//    random self-play walk (exercises both side-to-move signs).
// ---------------------------------------------------------------------------
[<Fact>]
let ``eval is exactly anti-symmetric under the board mirror`` () =
    for fen in perftFens do
        assertSymmetric (Position.OfFen fen)

    let rng = Random(20260621) // pinned seed
    let mutable p = Position.OfFen StartPosFen
    let mutable ply = 0

    for _ in 1..1000 do
        assertSymmetric p
        let moves = collectLegal p

        if moves.Length = 0 || ply >= 80 then // terminal or ply cap -> restart
            p <- Position.OfFen StartPosFen
            ply <- 0
        else
            p.Make moves.[rng.Next moves.Length]
            ply <- ply + 1

// ---------------------------------------------------------------------------
// 2. Startpos eval == 0 (no tempo), full midgame phase.
// ---------------------------------------------------------------------------
[<Fact>]
let ``startpos evaluates to 0 at phase 24`` () =
    let p = Position.OfFen StartPosFen
    Assert.Equal(0, eval p)
    let struct (mg, eg, phase) = evalTrace p
    Assert.Equal(struct (0, 0, 24), struct (mg, eg, phase))

// ---------------------------------------------------------------------------
// 3. Material dominance: up a queen is strongly positive for the side to move.
// ---------------------------------------------------------------------------
[<Fact>]
let ``up a queen is strongly positive for the side to move`` () =
    let p = Position.OfFen "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    Assert.True(eval p > 600, sprintf "expected > 600, got %d" (eval p))

// ---------------------------------------------------------------------------
// 4. Full-table transcription guard.
//    (a) mirror identity for every (pt, sq) — guards the init arithmetic over all 768 entries.
//    (b) pinned checksum + sum (regression guard). Sums cross-check the material formula:
//        MGSUM = 128*Σ mgValue + 2*Σ rawMg ; EGSUM = 128*Σ egValue + 2*Σ rawEg.
// ---------------------------------------------------------------------------
[<Fact>]
let ``PST mirror identity holds for every piece type and square`` () =
    for pt in 0..5 do
        for s in 0..63 do
            Assert.Equal(mgScoreOf (makePiece White pt) s, mgScoreOf (makePiece Black pt) (s ^^^ 56))
            Assert.Equal(egScoreOf (makePiece White pt) s, egScoreOf (makePiece Black pt) (s ^^^ 56))

[<Fact>]
let ``combined tables reproduce the pinned PeSTO checksum`` () =
    let mutable hm, he, sm, se = 17, 17, 0, 0

    for pc in 0..11 do
        for sq in 0..63 do
            let m = mgScoreOf pc sq
            let e = egScoreOf pc sq
            hm <- hm * 31 + m
            he <- he * 31 + e
            sm <- sm + m
            se <- se + e

    Assert.Equal(1617154431, hm)
    Assert.Equal(1643598729, he)
    Assert.Equal(293994, sm)
    Assert.Equal(274460, se)

// ---------------------------------------------------------------------------
// 5. Pinned exact micro-positions — hand-computed from the published tables (independent value guard).
//
//    `7k/8/8/8/8/8/8/QK6 w - -`: white Q@a1(0), white K@b1(1), black K@h8(63). phase = 4 (one queen).
//      mg = (1025 + mgQ[56]=-1) + (mgK[57]=36) - (mgK[63]=14)            = 1024 + 36 - 14 = 1046
//      eg = (936  + egQ[56]=-33) + (egK[57]=-34) - (egK[63]=-43)         = 903  - 34 + 43 = 912
//      score = (1046*4 + 912*20)/24 = 22424/24 = 934 ; white to move -> eval = 934.
//
//    `8/8/8/3k4/8/3K4/4P3/8 w - -`: white P@e2(12), white K@d3(19), black K@d5(35). phase = 0.
//      eg = (94 + egP[52]=13) + (egK[43]=21) - (egK[35]=24)              = 107 + 21 - 24 = 104
//      score = (mg*0 + 104*24)/24 = 104 ; white to move -> eval = 104.
// ---------------------------------------------------------------------------
[<Fact>]
let ``pinned exact totals: KQ vs K`` () =
    let p = Position.OfFen "7k/8/8/8/8/8/8/QK6 w - -"
    Assert.Equal(struct (1046, 912, 4), evalTrace p)
    Assert.Equal(934, eval p)

[<Fact>]
let ``pinned exact totals: pawn endgame (phase 0)`` () =
    let p = Position.OfFen "8/8/8/3k4/8/3K4/4P3/8 w - -"
    Assert.Equal(struct (60, 104, 0), evalTrace p)
    Assert.Equal(104, eval p)

// ---------------------------------------------------------------------------
// 6. Phase extremes + interpolation wiring.
// ---------------------------------------------------------------------------
[<Fact>]
let ``phase 0: eval is the EG score only`` () =
    let p = Position.OfFen "8/8/8/3k4/8/3K4/4P3/8 w - -"
    let struct (_, eg, phase) = evalTrace p
    Assert.Equal(0, phase)
    Assert.Equal(eg, eval p) // white to move -> eval = white-rel eg

[<Fact>]
let ``phase 24: eval is the MG score only (asymmetric full material)`` () =
    let p = Position.OfFen perftFens.[1] // Kiwipete: full material, asymmetric, w to move
    let struct (mg, _, phase) = evalTrace p
    Assert.Equal(24, phase)
    Assert.Equal(mg, eval p) // white to move -> eval = white-rel mg

[<Fact>]
let ``midgame: eval equals the signed taper recomputed from evalTrace`` () =
    let p = Position.OfFen perftFens.[2] // 4-rook-ish endgame, phase strictly in (0,24)
    let struct (mg, eg, phase) = evalTrace p
    Assert.True(phase > 0 && phase < 24, sprintf "phase = %d" phase)
    let score = (mg * phase + eg * (24 - phase)) / 24
    let expected = if p.SideToMove = White then score else -score
    Assert.Equal(expected, eval p)

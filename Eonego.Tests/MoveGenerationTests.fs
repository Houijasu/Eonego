module Eonego.Tests.MoveGenerationTests

open Xunit
open Eonego.Bitboard
open Eonego.Move
open Eonego.Position
open Eonego.MoveGeneration
open Eonego.Tests.TestFixtures

// Named squares (LERF).
let private sq f r = mkSquare f r
let private a1 = sq 0 0
let private c1 = sq 2 0
let private d1 = sq 3 0
let private e1 = sq 4 0
let private f1 = sq 5 0
let private g1 = sq 6 0
let private a2 = sq 0 1
let private e2 = sq 4 1
let private e3 = sq 4 2
let private e4 = sq 4 3
let private e5 = sq 4 4
let private e6 = sq 4 5
let private d6 = sq 3 5

let private hasMove (ms: Move[]) (from: Square) (dst: Square) =
    ms |> Array.exists (fun m -> fromSq m = from && toSq m = dst)

// ---------------------------------------------------------------------------
// Perft gate — default tier (d1..d4 for all six canonical positions).
// ---------------------------------------------------------------------------
[<Theory>]
[<InlineData("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 1, 20L)>]
[<InlineData("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 2, 400L)>]
[<InlineData("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 3, 8902L)>]
[<InlineData("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 4, 197281L)>]
[<InlineData("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -", 1, 48L)>]
[<InlineData("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -", 2, 2039L)>]
[<InlineData("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -", 3, 97862L)>]
[<InlineData("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -", 4, 4085603L)>]
[<InlineData("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -", 1, 14L)>]
[<InlineData("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -", 2, 191L)>]
[<InlineData("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -", 3, 2812L)>]
[<InlineData("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -", 4, 43238L)>]
[<InlineData("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq -", 1, 6L)>]
[<InlineData("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq -", 2, 264L)>]
[<InlineData("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq -", 3, 9467L)>]
[<InlineData("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq -", 4, 422333L)>]
[<InlineData("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ -", 1, 44L)>]
[<InlineData("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ -", 2, 1486L)>]
[<InlineData("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ -", 3, 62379L)>]
[<InlineData("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ -", 4, 2103487L)>]
[<InlineData("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - -", 1, 46L)>]
[<InlineData("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - -", 2, 2079L)>]
[<InlineData("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - -", 3, 89890L)>]
[<InlineData("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - -", 4, 3894594L)>]
let ``perft matches reference (d1-d4)`` (fen: string) (depth: int) (expected: int64) =
    Assert.Equal(uint64 expected, perft (Position.OfFen fen) depth)

// ---------------------------------------------------------------------------
// Perft gate — slow tier (d5 + a couple of deep d6). Opt-in via the trait.
// ---------------------------------------------------------------------------
[<Theory>]
[<Trait("Category", "Slow")>]
[<InlineData("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 5, 4865609L)>]
[<InlineData("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 6, 119060324L)>]
[<InlineData("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -", 5, 193690690L)>]
[<InlineData("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -", 5, 674624L)>]
[<InlineData("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -", 6, 11030083L)>]
[<InlineData("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq -", 5, 15833292L)>]
[<InlineData("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ -", 5, 89941194L)>]
[<InlineData("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - -", 5, 164075551L)>]
let ``perft matches reference (slow d5-d6)`` (fen: string) (depth: int) (expected: int64) =
    Assert.Equal(uint64 expected, perft (Position.OfFen fen) depth)

// ---------------------------------------------------------------------------
// Targeted fixtures the perft node-count table does NOT isolate.
// ---------------------------------------------------------------------------

[<Fact>]
let ``rank masks have the right bits`` () =
    Assert.Equal(8, popCount Rank2)
    Assert.Equal(8, popCount Rank3)
    Assert.Equal(8, popCount Rank6)
    Assert.Equal(8, popCount Rank7)
    Assert.Equal(mkSquare 0 1, lsb Rank2) // a2
    Assert.Equal(mkSquare 0 2, lsb Rank3) // a3
    Assert.Equal(mkSquare 0 5, lsb Rank6) // a6
    Assert.Equal(mkSquare 0 6, lsb Rank7) // a7

[<Fact>]
let ``en passant illegal under a diagonal pin (bishop test)`` () =
    // White Kc3, Pe5; black Bg7 pins e5 along a1-h8; black pawn d5 just played d7-d5 (ep d6).
    // exd6 e.p. would vacate e5 and expose Kc3 to Bg7 -> the EP move must be filtered out.
    let p = Position.OfFen "k7/6b1/8/3pP3/8/2K5/8/8 w - d6 0 1"
    Assert.Equal(d6, p.EpSquare) // ep was kept (a capturer exists)
    let ms = collectLegal p
    Assert.False(ms |> Array.exists isEnPassant, "illegal diagonal-pin EP leaked into the legal list")
    // e5 is itself pinned on the a1-h8 diagonal, so it has no legal move at all; the king still does.
    Assert.False(ms |> Array.exists (fun m -> fromSq m = e5), "the diagonally-pinned pawn should be frozen")
    Assert.NotEmpty(ms)

[<Fact>]
let ``en passant is generated as a check evasion`` () =
    // Black pawn d5 (just double-pushed) checks white Ke4; exd6 e.p. removes the checker.
    let p = Position.OfFen "k7/8/8/3pP3/4K3/8/8/8 w - d6 0 1"
    Assert.True(p.InCheck)
    let ms = collectLegal p

    Assert.True(
        ms |> Array.exists (fun m -> isEnPassant m && toSq m = d6),
        "EP capture of the checking pawn was dropped from the evasion list"
    )

[<Fact>]
let ``king may not flee along the checking ray`` () =
    // Black Re8 checks Ke1 down the e-file; Ke2 stays on the ray and must be illegal.
    let p = Position.OfFen "4r2k/8/8/8/8/8/8/4K3 w - - 0 1"
    let ms = collectLegal p
    Assert.False(hasMove ms e1 e2, "Ke2 along the checking file should be illegal")
    Assert.True(hasMove ms e1 d1)
    Assert.True(hasMove ms e1 f1)
    Assert.Equal(4, ms.Length) // d1, f1, d2, f2 only

[<Fact>]
let ``double check yields king moves only`` () =
    // Re8 (file) + Bh4 (diagonal) both check Ke1.
    let p = Position.OfFen "k3r3/8/8/8/7b/8/8/4K3 w - - 0 1"
    Assert.Equal(2, popCount p.Checkers)
    let ms = collectLegal p
    Assert.NotEmpty(ms)
    Assert.True(ms |> Array.forall (fun m -> fromSq m = e1), "a non-king move resolved a double check")

[<Fact>]
let ``O-O is blocked when the king crosses an attacked square`` () =
    // Black Rf8 attacks f1; white O-O (e1g1) passes f1 -> illegal.
    let p = Position.OfFen "5rk1/8/8/8/8/8/8/4K2R w K - 0 1"
    let ms = collectLegal p
    Assert.False(ms |> Array.exists isCastling, "O-O through the attacked f1 should be illegal")

[<Fact>]
let ``O-O-O is allowed when only the empty b-file square is attacked`` () =
    // Black Rb8 attacks b1, but b1 is only required to be EMPTY (king never transits it) -> O-O-O legal.
    let p = Position.OfFen "1r5k/8/8/8/8/8/8/R3K3 w Q - 0 1"
    let ms = collectLegal p
    Assert.True(hasMove ms e1 c1, "O-O-O should be legal with b1 attacked-but-empty")
    Assert.True(ms |> Array.exists isCastling)

[<Fact>]
let ``castling is suppressed while in check`` () =
    // Re8 checks Ke1; castling must never be generated in check even with the right held.
    let p = Position.OfFen "4r2k/8/8/8/8/8/8/4K2R w K - 0 1"
    Assert.True(p.InCheck)
    let ms = collectLegal p
    Assert.False(ms |> Array.exists isCastling)

[<Fact>]
let ``a blocked pawn does not double-push`` () =
    // White Pe2 with a black pawn on e4: e2e3 legal, e2e4 blocked.
    let p = Position.OfFen "4k3/8/8/8/4p3/8/4P3/4K3 w - - 0 1"
    let ms = collectLegal p
    Assert.True(hasMove ms e2 e3)
    Assert.False(hasMove ms e2 e4, "double push over/into a blocker must not be generated")

[<Fact>]
let ``black promotion emits all four under-promotions`` () =
    // Black Pa2 to move pushes a2a1 and promotes.
    let p = Position.OfFen "4k3/8/8/8/8/8/p7/4K3 b - - 0 1"
    let ms = collectLegal p

    let promos =
        ms |> Array.filter (fun m -> isPromotion m && fromSq m = a2 && toSq m = a1)

    Assert.Equal(4, promos.Length)
    let kinds = promos |> Array.map promoType |> Array.sort
    Assert.Equal<int[]>([| Knight; Bishop; Rook; Queen |], kinds)

[<Fact>]
let ``a pinned knight has no moves`` () =
    // White Ne2 pinned to Ke1 by Re8 — a knight can never move along the pin.
    let p = Position.OfFen "4r2k/8/8/8/8/8/4N3/4K3 w - - 0 1"
    let ms = collectLegal p
    Assert.False(ms |> Array.exists (fun m -> fromSq m = e2), "the pinned knight should have no legal move")

[<Fact>]
let ``a pinned rook still slides along the pin`` () =
    // White Re2 pinned to Ke1 by Re8 may move only along the e-file.
    let p = Position.OfFen "4r2k/8/8/8/8/8/4R3/4K3 w - - 0 1"
    let rookMoves = collectLegal p |> Array.filter (fun m -> fromSq m = e2)
    Assert.NotEmpty(rookMoves)
    Assert.True(rookMoves |> Array.forall (fun m -> fileOf (toSq m) = 4), "the pinned rook left the pin file")

// ---------------------------------------------------------------------------
// Make/Unmake tie-in: every legal move from every canonical position must round-trip
// and keep the incremental key consistent (binds MoveGen to the verified Make/Unmake).
// ---------------------------------------------------------------------------
[<Fact>]
let ``every legal move round-trips Make/Unmake on each canonical position`` () =
    for fen in perftFens do
        let p = Position.OfFen fen

        for m in collectLegal p do
            assertRoundTrips p m

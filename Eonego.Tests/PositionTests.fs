module Eonego.Tests.PositionTests

open Xunit
open Eonego.Bitboard
open Eonego.Move
open Eonego.Position

// Named squares (LERF).
let private a1 = mkSquare 0 0
let private b1 = mkSquare 1 0
let private c1 = mkSquare 2 0
let private d1 = mkSquare 3 0
let private e1 = mkSquare 4 0
let private f1 = mkSquare 5 0
let private g1 = mkSquare 6 0
let private h1 = mkSquare 7 0
let private e2 = mkSquare 4 1
let private e3 = mkSquare 4 2
let private e4 = mkSquare 4 3
let private a7 = mkSquare 0 6
let private a8 = mkSquare 0 7
let private c8 = mkSquare 2 7
let private e8 = mkSquare 4 7
let private g8 = mkSquare 6 7

// Full-state snapshot built from PUBLIC accessors (lists -> F# structural equality compares contents).
type private Snap =
    { Types: uint64 list
      Colors: uint64 list
      Board: int list
      Occ: uint64
      Key: uint64
      Castle: int
      Ep: int
      Rule50: int
      Stm: int }

let private snap (p: Position) : Snap =
    { Types = [ for pt in 0..5 -> p.Pieces pt ]
      Colors = [ p.ColorBB White; p.ColorBB Black ]
      Board = [ for sq in 0..63 -> p.PieceOn sq ]
      Occ = p.Occupied
      Key = p.Key
      Castle = p.CastlingRights
      Ep = p.EpSquare
      Rule50 = p.Rule50
      Stm = p.SideToMove }

// Make then Unmake must restore every byte; and after Make incremental key == from-scratch.
let private roundTrips (fen: string) (m: Move) =
    let p = Position.OfFen fen
    let before = snap p
    p.Make m
    Assert.Equal(p.RecomputeKey(), p.Key)
    p.Unmake m
    Assert.True((before = snap p), "Make/Unmake did not restore full state")

// ---------------------------------------------------------------------------
// Task 2 — skeleton populated via the public SetStartPos, asserted via public API.
// ---------------------------------------------------------------------------

[<Fact>]
let ``SetStartPos lays out the standard position`` () =
    let p = Position()
    p.SetStartPos()
    Assert.Equal(32, popCount p.Occupied)
    Assert.Equal(16, popCount (p.ColorBB White))
    Assert.Equal(16, popCount (p.ColorBB Black))
    Assert.Equal(16, popCount (p.Pieces Pawn)) // byTypeBB.[Pawn] = both colors
    Assert.Equal(8, popCount (p.PiecesCT White Pawn)) // white pawns only
    Assert.Equal(makePiece White King, p.PieceOn e1)
    Assert.Equal(makePiece Black King, p.PieceOn e8)
    Assert.Equal(makePiece White Rook, p.PieceOn a1)
    Assert.Equal(e1, p.KingSquare White)
    Assert.Equal(e8, p.KingSquare Black)
    Assert.Equal(White, p.SideToMove)
    Assert.True(p.IsEmpty(mkSquare 4 3)) // e4 empty

[<Fact>]
let ``start position castling rights and ep`` () =
    let p = Position()
    p.SetStartPos()
    Assert.Equal(WK ||| WQ ||| BK ||| BQ, p.CastlingRights)
    Assert.True(p.CanCastle WK)
    Assert.True(p.CanCastle BQ)
    Assert.Equal(NoSquare, p.EpSquare)
    Assert.Equal(0, p.Rule50)

[<Fact>]
let ``castling-path tables are correct`` () =
    let p = Position()
    Assert.Equal(bit f1 ||| bit g1, p.CastleKingPath WK)
    Assert.Equal(bit f1 ||| bit g1, p.CastleEmptyPath WK)
    Assert.Equal(bit d1 ||| bit c1, p.CastleKingPath WQ)
    Assert.Equal(bit d1 ||| bit c1 ||| bit b1, p.CastleEmptyPath WQ) // O-O-O includes b-file
    Assert.Equal(h1, p.CastleRookSquare WK)
    Assert.Equal(a1, p.CastleRookSquare WQ)
    Assert.Equal(g1, p.CastleKingDest WK)
    Assert.Equal(c1, p.CastleKingDest WQ)

[<Fact>]
let ``start position incremental key matches from-scratch RecomputeKey`` () =
    let p = Position()
    p.SetStartPos()
    Assert.Equal(p.RecomputeKey(), p.Key)
    Assert.NotEqual(0UL, p.Key)

// ---------------------------------------------------------------------------
// Task 3 — FEN load / export. The canonical perft set (carried from the Bitboard plan).
// Note: only startpos is a full 6-field FEN; the others are the abbreviated CPW forms.
// ---------------------------------------------------------------------------

let private perftFens =
    [| "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
       "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -"
       "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -"
       "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq -"
       "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ -"
       "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - -" |]

[<Fact>]
let ``OfFen startpos round-trips to the canonical 6-field string`` () =
    let p = Position.OfFen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    Assert.Equal("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", p.ToFen())

[<Fact>]
let ``ToFen . OfFen is idempotent and key-preserving for every perft FEN`` () =
    for s in perftFens do
        let p1 = Position.OfFen s
        let f1 = p1.ToFen()
        let p2 = Position.OfFen f1
        Assert.Equal(p1.Key, p2.Key) // load is deterministic on its own output
        Assert.Equal(f1, p2.ToFen()) // ToFen . OfFen is idempotent (normalized form is fixed)
        Assert.Equal(p1.RecomputeKey(), p1.Key) // incremental == from-scratch on load

[<Fact>]
let ``abbreviated FENs parse with tolerant clocks`` () =
    let p = Position.OfFen "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -" // Position 3, 4-field
    Assert.Equal(0, p.Rule50)
    Assert.Equal(White, p.SideToMove)
    Assert.Equal(0, p.CastlingRights)
    Assert.Equal(NoSquare, p.EpSquare)

// --- EP "real capturer exists" gate, three ways ----------------------------

[<Fact>]
let ``ep target with a real capturer is kept (square + key term)`` () =
    let withEp =
        Position.OfFen "rnbqkbnr/pp1ppppp/8/2pP4/8/8/PPP1PPPP/RNBQKBNR w KQkq c6 0 3"

    let c6 = mkSquare 2 5
    Assert.Equal(c6, withEp.EpSquare)
    Assert.Contains("c6", withEp.ToFen())
    // the ep term actually changed the key: same board+rights with no ep differs by exactly zEp(file c)
    let noEp =
        Position.OfFen "rnbqkbnr/pp1ppppp/8/2pP4/8/8/PPP1PPPP/RNBQKBNR w KQkq - 0 3"

    Assert.NotEqual(withEp.Key, noEp.Key)

[<Fact>]
let ``ep target without a capturer is dropped (square + key + export)`` () =
    let fab =
        Position.OfFen "rnbqkbnr/pp1ppppp/8/2p5/8/8/PPPPPPPP/RNBQKBNR w KQkq c6 0 3"

    Assert.Equal(NoSquare, fab.EpSquare) // no white pawn on b5/d5 -> dropped
    Assert.Contains(" - ", fab.ToFen()) // exported as '-'

    let same =
        Position.OfFen "rnbqkbnr/pp1ppppp/8/2p5/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 3"

    Assert.Equal(same.Key, fab.Key) // fabricated ep left no key term

// ---------------------------------------------------------------------------
// Task 4 — AttackersTo, SliderBlockers, SetCheckInfo.
// ---------------------------------------------------------------------------

[<Fact>]
let ``AttackersTo crosses pawn colors correctly`` () =
    let p = Position.OfFen "4k3/8/8/3p4/3P4/8/8/4K3 w - - 0 1" // white pawn d4, black pawn d5
    let occ = p.Occupied
    let d4 = mkSquare 3 3
    let d5 = mkSquare 3 4
    Assert.True(testBit (p.AttackersTo (mkSquare 2 4) occ) d4) // white d4 attacks c5
    Assert.True(testBit (p.AttackersTo (mkSquare 4 4) occ) d4) // white d4 attacks e5
    Assert.True(testBit (p.AttackersTo (mkSquare 2 3) occ) d5) // black d5 attacks c4
    Assert.True(testBit (p.AttackersTo (mkSquare 4 3) occ) d5) // black d5 attacks e4

[<Fact>]
let ``single and double check populate Checkers`` () =
    let single = Position.OfFen "k3r3/8/8/8/8/8/8/4K3 w - - 0 1" // rook e8 checks Ke1
    Assert.Equal(1, popCount single.Checkers)
    Assert.True(single.InCheck)
    Assert.True(testBit single.Checkers (mkSquare 4 7)) // e8
    let dbl = Position.OfFen "k3r3/8/8/8/7b/8/8/4K3 w - - 0 1" // rook e8 + bishop h4 both check Ke1
    Assert.Equal(2, popCount dbl.Checkers)

[<Fact>]
let ``pin shows up as blocker + pinner`` () =
    let p = Position.OfFen "k3r3/8/8/8/8/8/4B3/4K3 w - - 0 1" // Be2 pinned to Ke1 by re8
    Assert.True(testBit (p.BlockersForKing White) (mkSquare 4 1)) // e2 is a blocker
    Assert.True(testBit (p.Pinners White) (mkSquare 4 7)) // e8 is the pinner
    Assert.False(p.InCheck)

[<Fact>]
let ``stacked rooks: single nearest blocker, occ ^ snipers proven`` () =
    let p = Position.OfFen "r6k/8/r7/8/R7/8/8/K7 w - - 0 1" // Ka1, Ra4, ra6, ra8
    let a4 = mkSquare 0 3
    let a6 = mkSquare 0 5
    Assert.Equal(bit a4, p.BlockersForKing White) // EXACTLY {a4}; a6/a8 are not blockers
    Assert.True(testBit (p.Pinners White) a6) // nearest slider is a pinner (a8 unspecified)

[<Fact>]
let ``CheckSquares(Knight) is knight attacks of the enemy king`` () =
    let p = Position()
    p.SetStartPos() // white to move -> them = Black, king e8
    Assert.Equal(knightAttacks (mkSquare 4 7), p.CheckSquares Knight)

// ---------------------------------------------------------------------------
// Task 5 — Make/Unmake fast path + full-state-restore invariant.
// ---------------------------------------------------------------------------

[<Fact>]
let ``quiet move round-trips`` () =
    roundTrips "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" (mkMove b1 (mkSquare 2 2)) // Nb1-c3

[<Fact>]
let ``capture round-trips and resets rule50`` () =
    let fen = "4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1" // Pe4 x d5
    roundTrips fen (mkMove e4 (mkSquare 3 4))
    let p = Position.OfFen fen
    p.Make(mkMove e4 (mkSquare 3 4))
    Assert.Equal(0, p.Rule50) // pawn capture resets the clock
    Assert.Equal(makePiece White Pawn, p.PieceOn(mkSquare 3 4)) // white pawn now on d5

[<Fact>]
let ``capturing a rook on its home square revokes that side's right`` () =
    let fen = "r3k3/R7/8/8/8/8/8/4K3 w q - 0 1" // Ra7 x a8 (a8 = black queenside rook home)
    roundTrips fen (mkMove a7 a8)
    let p = Position.OfFen fen
    Assert.Equal(BQ, p.CastlingRights)
    p.Make(mkMove a7 a8)
    Assert.Equal(0, p.CastlingRights) // BQ revoked by capture on a8

[<Fact>]
let ``double push sets ep only when a real capturer exists`` () =
    // black pawn on d4 can take e3 ep -> ep set
    let withCapturer = "4k3/8/8/8/3p4/8/4P3/4K3 w - - 0 1"
    roundTrips withCapturer (mkMove e2 e4)
    let p = Position.OfFen withCapturer
    p.Make(mkMove e2 e4)
    Assert.Equal(e3, p.EpSquare)
    // no adjacent black pawn -> ep stays NoSquare
    let noCapturer = "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"
    roundTrips noCapturer (mkMove e2 e4)
    let q = Position.OfFen noCapturer
    q.Make(mkMove e2 e4)
    Assert.Equal(NoSquare, q.EpSquare)

[<Fact>]
let ``quiet king move revokes both of its side's rights`` () =
    let fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
    roundTrips fen (mkMove e1 e2)
    let p = Position.OfFen fen
    p.Make(mkMove e1 e2)
    Assert.Equal(BK ||| BQ, p.CastlingRights) // white K,Q cleared; black intact

// ---------------------------------------------------------------------------
// Task 6 — Make/Unmake special moves.
// ---------------------------------------------------------------------------

[<Fact>]
let ``white en passant round-trips and removes the victim on dst-8`` () =
    let fen = "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1" // Pe5 x d6 e.p. (black just played d7-d5)
    let d6 = mkSquare 3 5
    let d5 = mkSquare 3 4
    let e5 = mkSquare 4 4
    roundTrips fen (mkEnPassant e5 d6)
    let p = Position.OfFen fen
    p.Make(mkEnPassant e5 d6)
    Assert.Equal(makePiece White Pawn, p.PieceOn d6)
    Assert.True(p.IsEmpty d5) // victim removed on d5, not d6
    Assert.True(p.IsEmpty(mkSquare 4 4))

[<Fact>]
let ``black en passant round-trips`` () =
    let fen = "4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1" // d4 x e3 e.p. (white just played e2-e4)
    roundTrips fen (mkEnPassant (mkSquare 3 3) e3)
    let p = Position.OfFen fen
    p.Make(mkEnPassant (mkSquare 3 3) e3)
    Assert.Equal(makePiece Black Pawn, p.PieceOn e3)
    Assert.True(p.IsEmpty e4) // captured white pawn was on e4 (= e3 + 8)

[<Fact>]
let ``all four promotions round-trip and place the right piece`` () =
    let fen = "8/P6k/8/8/8/8/8/4K3 w - - 0 1" // a7-a8=?

    for promo in [ Knight; Bishop; Rook; Queen ] do
        roundTrips fen (mkPromotion a7 a8 promo)
        let p = Position.OfFen fen
        p.Make(mkPromotion a7 a8 promo)
        Assert.Equal(makePiece White promo, p.PieceOn a8)
        Assert.True(p.IsEmpty a7)

[<Fact>]
let ``promotion-capture restores rook and pawn, removes the queen, revokes the right`` () =
    let fen = "4k3/8/8/8/8/8/1p6/R3K3 b Q - 0 1" // ...b2 x a1=Q (captures the WQ rook)
    let b2 = mkSquare 1 1
    roundTrips fen (mkPromotion b2 a1 Queen)
    let p = Position.OfFen fen
    p.Make(mkPromotion b2 a1 Queen)
    Assert.Equal(makePiece Black Queen, p.PieceOn a1)
    Assert.Equal(0, p.CastlingRights) // a1 rook captured -> WQ gone

[<Fact>]
let ``all four castles round-trip; rights cleared for that color only, rule50 not reset`` () =
    let w = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 5 1"
    let b = "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 5 1"
    roundTrips w (mkCastling e1 g1)
    roundTrips w (mkCastling e1 c1)
    roundTrips b (mkCastling e8 g8)
    roundTrips b (mkCastling e8 c8)
    // white O-O specifics
    let p = Position.OfFen w
    p.Make(mkCastling e1 g1)
    Assert.Equal(makePiece White King, p.PieceOn g1)
    Assert.Equal(makePiece White Rook, p.PieceOn f1)
    Assert.True(p.IsEmpty e1 && p.IsEmpty h1)
    Assert.Equal(BK ||| BQ, p.CastlingRights) // white rights gone, black intact
    Assert.Equal(6, p.Rule50) // castling does NOT reset the clock

// ---------------------------------------------------------------------------
// Task 7 — MakeNull/UnmakeNull symmetry + GivesCheck.
// ---------------------------------------------------------------------------

[<Fact>]
let ``MakeNull then UnmakeNull restores full state and flips side`` () =
    let p = Position.OfFen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    let before = snap p
    p.MakeNull()
    Assert.Equal(Black, p.SideToMove)
    Assert.Equal(NoSquare, p.EpSquare)
    p.UnmakeNull()
    Assert.True((before = snap p), "MakeNull/UnmakeNull did not restore full state")

[<Fact>]
let ``GivesCheck: direct knight check vs quiet`` () =
    let p = Position.OfFen "4k3/8/8/3N4/8/8/8/4K3 w - - 0 1" // Nd5
    Assert.True(p.GivesCheck(mkMove (mkSquare 3 4) (mkSquare 5 5))) // Nf6+ (knight attacks e8)
    Assert.False(p.GivesCheck(mkMove (mkSquare 3 4) (mkSquare 2 2))) // Nc3 — no check
    let s = Position()
    s.SetStartPos()
    Assert.False(s.GivesCheck(mkMove b1 (mkSquare 2 2))) // Nb1-c3 in startpos: no check

[<Fact>]
let ``GivesCheck: discovered check`` () =
    let p = Position.OfFen "4k3/8/8/8/4N3/8/8/K3R3 w - - 0 1" // Re1 behind Ne4, black Ke8
    Assert.True(p.GivesCheck(mkMove (mkSquare 4 3) (mkSquare 2 4))) // Ne4-c5 uncovers Re1+

[<Fact>]
let ``GivesCheck: en passant exposes a rank check`` () =
    let p = Position.OfFen "8/8/8/R2pP2k/8/8/8/K7 w - d6 0 1" // Ra5 ... d5 P e5 ... k h5
    let e5 = mkSquare 4 4
    let d6 = mkSquare 3 5
    Assert.True(p.GivesCheck(mkEnPassant e5 d6)) // exd6 e.p. clears rank 5 -> Ra5+ on h5

[<Fact>]
let ``GivesCheck: promotion down a shadowed file (occ ^ from recompute)`` () =
    let p = Position.OfFen "K7/4P3/8/8/8/8/8/4k3 w - - 0 1" // Pe7, black Ke1 down the e-file
    Assert.True(p.GivesCheck(mkPromotion (mkSquare 4 6) (mkSquare 4 7) Queen)) // e8=Q+ (pawn shadowed e-file)

[<Fact>]
let ``GivesCheck: under-promotion to knight uses promoType`` () =
    let p = Position.OfFen "8/2P1k3/8/8/8/8/8/7K w - - 0 1" // Pc7, black Ke7
    Assert.True(p.GivesCheck(mkPromotion (mkSquare 2 6) (mkSquare 2 7) Knight)) // c8=N+ (a queen would NOT check e7)

[<Fact>]
let ``GivesCheck: castling where the rook lands giving check`` () =
    let p = Position.OfFen "5k2/8/8/8/8/8/8/4K2R w K - 0 1" // O-O lands rook on f1, black Kf8
    Assert.True(p.GivesCheck(mkCastling e1 g1)) // rook f1 checks f8

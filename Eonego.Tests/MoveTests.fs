module Eonego.Tests.MoveTests

open Xunit
open Eonego.Bitboard
open Eonego.Move

// Named squares for readable UCI assertions.
let private a1 = mkSquare 0 0
let private b1 = mkSquare 1 0
let private c1 = mkSquare 2 0
let private e1 = mkSquare 4 0
let private g1 = mkSquare 6 0
let private e2 = mkSquare 4 1
let private d5 = mkSquare 3 4
let private e4 = mkSquare 4 3
let private e7 = mkSquare 4 6
let private e8 = mkSquare 4 7
let private h8 = mkSquare 7 7

let private promos = [ Knight; Bishop; Rook; Queen ]

// ---------------------------------------------------------------------------
// Task 1 — type, literals, sentinels
// ---------------------------------------------------------------------------

[<Fact>]
let ``sentinels: MoveNone is a1a1, MoveNull is b1b1, both distinct`` () =
    Assert.Equal(0, MoveNone)
    Assert.Equal(0x41, MoveNull)
    Assert.NotEqual(MoveNone, MoveNull)
    Assert.Equal(a1, fromSq MoveNone)
    Assert.Equal(a1, toSq MoveNone)
    Assert.Equal(b1, fromSq MoveNull)
    Assert.Equal(b1, toSq MoveNull)

[<Fact>]
let ``sentinels: isOk rejects both, isNone/isNullMove discriminate`` () =
    Assert.False(isOk MoveNone)
    Assert.False(isOk MoveNull)
    Assert.True(isNone MoveNone)
    Assert.False(isNone MoveNull)
    Assert.True(isNullMove MoveNull)
    Assert.False(isNullMove MoveNone)

// ---------------------------------------------------------------------------
// Task 2 — mkMove + from/to round-trip (exhaustive over all 4096 pairs)
// ---------------------------------------------------------------------------

[<Fact>]
let ``mkMove round-trips from/to over all 4096 pairs, flag Normal`` () =
    for from in 0..63 do
        for dst in 0..63 do
            let m = mkMove from dst
            Assert.Equal(from, fromSq m)
            Assert.Equal(dst, toSq m)
            Assert.Equal(FlagNormal, moveFlag m)
            Assert.True(isNormal m)
            Assert.False(isSpecial m)

// ---------------------------------------------------------------------------
// Task 3 — move kinds: flags + predicate exclusivity
// ---------------------------------------------------------------------------

[<Fact>]
let ``en-passant and castling carry their flag and survive from/to`` () =
    let ep = mkEnPassant e4 d5
    Assert.Equal(FlagEnPassant, moveFlag ep)
    Assert.True(isEnPassant ep)
    Assert.True(isSpecial ep)
    Assert.Equal(e4, fromSq ep)
    Assert.Equal(d5, toSq ep)

    let castle = mkCastling e1 g1
    Assert.Equal(FlagCastling, moveFlag castle)
    Assert.True(isCastling castle)
    Assert.True(isSpecial castle)
    Assert.Equal(e1, fromSq castle)
    Assert.Equal(g1, toSq castle)

[<Fact>]
let ``the four kind-predicates are mutually exclusive and exhaustive`` () =
    let samples =
        [ mkMove e2 e4; mkPromotion e7 e8 Queen; mkEnPassant e4 d5; mkCastling e1 g1 ]

    for m in samples do
        let hits =
            [ isNormal m; isPromotion m; isEnPassant m; isCastling m ]
            |> List.filter id
            |> List.length

        Assert.Equal(1, hits)
        Assert.Equal(isSpecial m, not (isNormal m))

// ---------------------------------------------------------------------------
// Task 4 — promotion encode/decode + the documented out-of-contract corruption
// ---------------------------------------------------------------------------

[<Fact>]
let ``mkPromotion round-trips Knight..Queen over all 4096 pairs`` () =
    for from in 0..63 do
        for dst in 0..63 do
            for promo in promos do
                let m = mkPromotion from dst promo
                Assert.Equal(from, fromSq m)
                Assert.Equal(dst, toSq m)
                Assert.Equal(FlagPromotion, moveFlag m)
                Assert.True(isPromotion m)
                Assert.Equal(promo, promoType m)

[<Fact>]
let ``non-promotion moves report isPromotion = false`` () =
    Assert.False(isPromotion (mkMove e2 e4))
    Assert.False(isPromotion (mkEnPassant e4 d5))
    Assert.False(isPromotion (mkCastling e1 g1))

// Release-observable negative test: we build the RAW encoding by hand (NOT via mkPromotion,
// whose Debug.Assert would fire in a Debug test run) to pin WHY callers must keep promo in
// Knight..Queen — the contract violation silently corrupts the move.
[<Fact>]
let ``out-of-contract promo corrupts the encoding (documents the PRE)`` () =
    // King (5): promo bits = (5-Knight)=4 truncate to 0 -> reads as a Knight promotion.
    let rawKing =
        (FlagPromotion <<< 14) ||| ((King - Knight) <<< 12) ||| (e7 <<< 6) ||| e8

    Assert.True(isPromotion rawKing)
    Assert.Equal(Knight, promoType rawKing)
    // Pawn (0): (0-Knight) = -1 scribbles the high word -> flag is no longer Promotion.
    let rawPawn =
        (FlagPromotion <<< 14) ||| ((Pawn - Knight) <<< 12) ||| (e7 <<< 6) ||| e8

    Assert.NotEqual(FlagPromotion, moveFlag rawPawn)

// ---------------------------------------------------------------------------
// Task 5 — fromTo / moveMatchKey / isOk / bit hygiene (all constructors)
// ---------------------------------------------------------------------------

[<Fact>]
let ``fromTo is the 12-bit from+to key, moveMatchKey adds promo`` () =
    let nm = mkMove e2 e4
    Assert.Equal((e2 <<< 6) ||| e4, fromTo nm)
    Assert.Equal(fromTo nm, moveMatchKey nm) // no promo/flag bits -> equal
    let pm = mkPromotion e7 e8 Rook
    Assert.Equal((e7 <<< 6) ||| e8, fromTo pm) // fromTo ignores promo + flag
    Assert.Equal(pm &&& 0x3FFF, moveMatchKey pm) // moveMatchKey keeps promo, drops flag
    Assert.NotEqual(moveMatchKey (mkPromotion e7 e8 Knight), moveMatchKey pm) // under-promo disambiguated

[<Fact>]
let ``isOk rejects every from==to and accepts real moves`` () =
    for sq in 0..63 do
        Assert.False(isOk (mkMove sq sq))

    Assert.True(isOk (mkMove e2 e4))
    Assert.True(isOk (mkPromotion e7 e8 Queen))

[<Fact>]
let ``no constructor sets bits above 15; non-promo constructors leave promo bits clear`` () =
    for from in 0..63 do
        for dst in 0..63 do
            let normal = mkMove from dst
            let ep = mkEnPassant from dst
            let castle = mkCastling from dst

            for m in [ normal; ep; castle ] do
                Assert.Equal(0, m &&& ~~~0xFFFF) // 16-bit fit
                Assert.Equal(0, (m >>> 12) &&& 0x3) // promo bits clear for non-promotions

            for promo in promos do
                Assert.Equal(0, (mkPromotion from dst promo) &&& ~~~0xFFFF)

// ---------------------------------------------------------------------------
// Task 6 — packed16 round-trip (16-bit-fit invariant for TT storage)
// ---------------------------------------------------------------------------

[<Fact>]
let ``packed16 round-trips and proves the 16-bit fit`` () =
    let rng = System.Random(0x4242)

    for _ in 0..9999 do
        let from = rng.Next(64)
        let dst = rng.Next(64)

        let m =
            match rng.Next(4) with
            | 0 -> mkMove from dst
            | 1 -> mkPromotion from dst (promos.[rng.Next(4)])
            | 2 -> mkEnPassant from dst
            | _ -> mkCastling from dst

        Assert.Equal(m, m &&& 0xFFFF) // fits in 16 bits
        Assert.Equal(m, ofPacked (packed16 m)) // lossless TT round-trip

// ---------------------------------------------------------------------------
// Task 7 — ScoredMove carrier
// ---------------------------------------------------------------------------

[<Fact>]
let ``ScoredMove round-trips, is 8 bytes, and reads copy-free via byref`` () =
    let m = mkMove e2 e4
    let sm = mkScored m 1234
    Assert.Equal(m, sm.Move)
    Assert.Equal(1234, sm.Score)
    Assert.Equal(8, sizeof<ScoredMove>)
    let arr = [| mkScored (mkMove e2 e4) 7; mkScored (mkPromotion e7 e8 Queen) -9 |]
    let r = &arr.[1] // byref read (copy-free)
    Assert.Equal(mkPromotion e7 e8 Queen, r.Move)
    Assert.Equal(-9, r.Score)

// ---------------------------------------------------------------------------
// Task 8 — UCI (strict)
// ---------------------------------------------------------------------------

[<Fact>]
let ``toUci formats normal, promotion, castling, and both sentinels`` () =
    Assert.Equal("e2e4", toUci (mkMove e2 e4))
    Assert.Equal("e7e8q", toUci (mkPromotion e7 e8 Queen))
    Assert.Equal("e7e8n", toUci (mkPromotion e7 e8 Knight))
    Assert.Equal("e1g1", toUci (mkCastling e1 g1))
    Assert.Equal("0000", toUci MoveNone)
    Assert.Equal("0000", toUci MoveNull) // MoveNull must NOT surface as "b1b1"

[<Fact>]
let ``parseUci is context-free: castling string parses to a Normal move`` () =
    Assert.Equal(mkMove e1 g1, parseUci "e1g1")
    Assert.True(isNormal (parseUci "e1g1"))
    Assert.Equal(mkPromotion e7 e8 Knight, parseUci "e7e8n")

[<Fact>]
let ``parseUci then toUci is identity over normal and promotion moves`` () =
    for from in 0..63 do
        for dst in 0..63 do
            if from <> dst then
                let nm = mkMove from dst
                Assert.Equal(nm, parseUci (toUci nm))

                for promo in promos do
                    let pm = mkPromotion from dst promo
                    Assert.Equal(pm, parseUci (toUci pm))

[<Fact>]
let ``parseUci rejects malformed input and from==dst with MoveNone`` () =
    let bad =
        [ ""
          "0000"
          "e2"
          "e2e"
          "e2e4qq"
          "e7e8queen" // length / null
          "i2e4"
          "e9e4"
          "e2i4"
          "e2e9" // out-of-range file/rank
          "E2E4"
          "e2E4" // uppercase
          "e7e8x"
          "e7e8Q"
          "e7e81" // bogus promo char
          "a1a1"
          "b1b1"
          "h8h8" ] // from==dst (must not leak sentinels)

    for s in bad do
        Assert.Equal(MoveNone, parseUci s)

// ---------------------------------------------------------------------------
// Task 9 — full encode/decode oracle over (from, dst, kind, promo)
// ---------------------------------------------------------------------------

[<Fact>]
let ``oracle: every constructor's fields decode back to their inputs`` () =
    for from in 0..63 do
        for dst in 0..63 do
            // normal / en-passant / castling
            let cases =
                [ FlagNormal, mkMove from dst
                  FlagEnPassant, mkEnPassant from dst
                  FlagCastling, mkCastling from dst ]

            for (flag, m) in cases do
                Assert.Equal(from, fromSq m)
                Assert.Equal(dst, toSq m)
                Assert.Equal(flag, moveFlag m)
            // promotions
            for promo in promos do
                let m = mkPromotion from dst promo
                Assert.Equal(from, fromSq m)
                Assert.Equal(dst, toSq m)
                Assert.Equal(FlagPromotion, moveFlag m)
                Assert.Equal(promo, promoType m)

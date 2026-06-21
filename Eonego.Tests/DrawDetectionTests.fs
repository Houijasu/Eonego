/// Draw detection (the search's job): repetition, the 50-move rule with the mate-on-100th exception,
/// insufficient material, and stalemate scoring exactly 0 (not -MATE).
module Eonego.Tests.DrawDetectionTests

open Xunit
open Eonego.Move
open Eonego.Position
open Eonego.Search
open Eonego.Tests.TestFixtures

let private play (p: Position) (uci: string) =
    let t = parseUci uci
    let mutable chosen = MoveNone

    for m in collectLegal p do
        if
            fromSq m = fromSq t
            && toSq m = toSq t
            && (not (isPromotion t) || promoType m = promoType t)
        then
            chosen <- m

    p.Make chosen

[<Fact>]
let ``a repetition inside the tree is detected as a draw`` () =
    // 1.Nf3 Nf6 2.Ng1 Ng8 returns to the start position (twofold = draw in search).
    let p = Position.OfFen StartPosFen
    Assert.False(isRepetition p)

    for uci in [ "g1f3"; "g8f6"; "f3g1"; "f6g8" ] do
        play p uci

    Assert.True(isRepetition p)
    Assert.True(isImmediateDraw p)

[<Fact>]
let ``fifty-move rule is a draw when a legal move exists`` () =
    Assert.True(isImmediateDraw (Position.OfFen "8/8/4k3/8/8/4K3/8/7R w - - 100 1"))

[<Fact>]
let ``checkmate on the 100th ply is mate, not a fifty-move draw`` () =
    // Back-rank mate at halfmove 100: black Kg8 is mated by Re8 (f8/h8 covered with the king removed,
    // f7/g7/h7 are its own pawns), so the 50-move rule must NOT override the checkmate.
    let fen = "4R1k1/5ppp/8/8/8/8/8/6K1 b - - 100 1"
    Assert.False(isImmediateDraw (Position.OfFen fen))
    let struct (score, _, _) = searchToDepth fen [||] 2 defaultConfig
    Assert.True(score <= -MATE_IN_MAX_PLY)

[<Theory>]
[<InlineData("8/8/4k3/8/8/4K3/8/8 w - - 0 1")>] // KvK
[<InlineData("8/8/4k3/8/8/4K3/8/6N1 w - - 0 1")>] // KNvK
[<InlineData("8/8/4k3/8/8/4K3/8/6B1 w - - 0 1")>] // KBvK
let ``insufficient material is a draw`` (fen: string) =
    Assert.True(insufficientMaterial (Position.OfFen fen))

[<Fact>]
let ``KB vs KB same-coloured bishops is a draw`` () =
    // Bf2 (dark) and Bc1 (dark) -> same colour -> draw.
    Assert.True(insufficientMaterial (Position.OfFen "8/8/4k3/8/8/4K3/5B2/2b5 w - - 0 1"))

[<Fact>]
let ``KB vs KB opposite-coloured bishops is not insufficient`` () =
    // Bf2 (dark) and Bd1 (light) -> a real game, not a forced draw.
    Assert.False(insufficientMaterial (Position.OfFen "8/8/4k3/8/8/4K3/5B2/3b4 w - - 0 1"))

[<Fact>]
let ``stalemate scores zero, not mate`` () =
    let fen = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"
    let p = Position.OfFen fen
    Assert.False(p.InCheck)
    Assert.Empty(collectLegal p)
    let struct (score, _, _) = searchToDepth fen [||] 2 defaultConfig
    Assert.Equal(0, score)

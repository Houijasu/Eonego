/// Monty policy port — step 1 tests: the geometric move-index tables.
/// Validates each pseudo-attack generator against well-known popcount totals, pins the derived
/// index-space size (OFFSETS[5][64] computed, not assumed), and — when the real net file is present
/// — cross-checks its byte length against the derived layout (the definitive size arbiter).
module Eonego.Tests.MontyPolicyTests

open System
open System.IO
open System.Numerics
open Xunit
open Eonego.Bitboard
open Eonego.Move
open Eonego.Position
open Eonego.MoveGeneration
open Eonego.MontyPolicy
open Eonego.Tests.TestFixtures

let private pieceTotal pc =
    let mutable s = 0

    for sq in 0..63 do
        s <- s + BitOperations.PopCount(destinations.[sq, pc])

    s

[<Fact>]
let ``geometric generators match known popcounts`` () =
    Assert.Equal(2, BitOperations.PopCount(destinations.[0, 1])) // knight a1
    Assert.Equal(8, BitOperations.PopCount(destinations.[27, 1])) // knight d4
    Assert.Equal(3, BitOperations.PopCount(destinations.[0, 5])) // king a1
    Assert.Equal(8, BitOperations.PopCount(destinations.[27, 5])) // king d4
    Assert.Equal(14, BitOperations.PopCount(destinations.[27, 3])) // rook d4 (always 14)
    Assert.Equal(13, BitOperations.PopCount(destinations.[27, 2])) // bishop d4
    Assert.Equal(7, BitOperations.PopCount(destinations.[0, 2])) // bishop a1
    Assert.Equal(27, BitOperations.PopCount(destinations.[27, 4])) // queen d4
    // Pawn forward-fan: 3 in the centre, 2 on the a/h files, 0 on rank 8 (shifts off-board).
    Assert.Equal(3, BitOperations.PopCount(destinations.[4, 0])) // e1
    Assert.Equal(2, BitOperations.PopCount(destinations.[0, 0])) // a1
    Assert.Equal(0, BitOperations.PopCount(destinations.[60, 0])) // e8

[<Fact>]
let ``per-piece geometric totals are exact`` () =
    Assert.Equal(154, pieceTotal 0) // pawn fan
    Assert.Equal(336, pieceTotal 1) // knight
    Assert.Equal(560, pieceTotal 2) // bishop
    Assert.Equal(896, pieceTotal 3) // rook (14 * 64)
    Assert.Equal(1456, pieceTotal 4) // queen
    Assert.Equal(420, pieceTotal 5) // king

[<Fact>]
let ``move-index space size is derived from the tables`` () =
    // Grand geometric total across ALL six piece types (NOT the queen-only 1456).
    Assert.Equal(3822, OffsetsBase)
    Assert.Equal(3920, FromTo) // 3822 + 88 + 2 + 8
    Assert.Equal(7840, NumMovesIndices) // 2 * FROM_TO
    Assert.Equal(57_294_496L, ExpectedNetBytes) // Jackal p8008192009q.network

/// Locate the (gitignored) Monty policy net relative to the repo root.
let private netPath () : string option =
    let mutable dir = DirectoryInfo(AppContext.BaseDirectory)
    let mutable root = None

    while root.IsNone && not (isNull dir) do
        if File.Exists(Path.Combine(dir.FullName, "Eonego.slnx")) then
            root <- Some dir.FullName

        dir <- dir.Parent

    match root with
    | Some r ->
        let p = Path.Combine(r, "nets", "jackal-policy.network")
        if File.Exists p then Some p else None
    | None -> None

[<Fact>]
let ``real net file length matches the derived layout (if present)`` () =
    match netPath () with
    | None -> () // soft-skip: net not downloaded
    | Some p -> Assert.Equal(ExpectedNetBytes, (FileInfo p).Length)

// ---------------------------------------------------------------------------
// Move-index assembly: every legal move must map to a DISTINCT index in [0, FromTo) (no good-SEE
// doubling yet). Exercises double-push, castling, promotions, and en passant.
// ---------------------------------------------------------------------------
let private indexFens =
    [ "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" // startpos (double pushes; king on e => hm)
      "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1" // kiwipete (both-side castling)
      "8/4P3/8/8/8/8/8/4K1k1 w - - 0 1" // e8 promotions Q/R/B/N
      "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3" // en passant exf6
      "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1" ] // endgame, king on a-file (hm = 0)

[<Fact>]
let ``every legal move maps to a distinct in-range base index`` () =
    for fen in indexFens do
        let pos = Position.OfFen fen
        let moves = collectLegal pos
        Assert.True(moves.Length > 0, "no legal moves for " + fen)
        let seen = System.Collections.Generic.HashSet<int>()

        for m in moves do
            let idx = moveToIndexBase pos m
            Assert.True(idx >= 0 && idx < FromTo, sprintf "base idx %d out of [0,%d) in %s" idx FromTo fen)
            Assert.True(seen.Add idx, sprintf "duplicate base idx %d in %s" idx fen)

[<Fact>]
let ``loader parses the Jackal policy net into the right-sized i8 arrays`` () =
    match netPath () with
    | None -> () // soft-skip
    | Some p ->
        match load p with
        | Failed reason -> Assert.Fail("policy net failed to load: " + reason)
        | Loaded net ->
            Assert.Equal(PolicyInputSize * PolicyHl, net.L0W.Length)
            Assert.Equal(PolicyHl, net.L0B.Length)
            Assert.Equal(NumMovesIndices * PolicyHidden, net.L1W.Length)
            Assert.Equal(NumMovesIndices, net.L1B.Length)
            // non-trivial weights (loader didn't zero-fill / misalign)
            Assert.Contains(net.L0W, (fun w -> w <> 0y))
            Assert.Contains(net.L1W, (fun w -> w <> 0y))

// ---------------------------------------------------------------------------
// SEE + mapMoveToIndex tests.
// ---------------------------------------------------------------------------

[<Fact>]
let ``mapMoveToIndex is in range and see passes for winning capture`` () =
    // All moves in these positions must map to a valid full index [0, NumMovesIndices).
    let fens =
        [ "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
          "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
          "4k3/8/8/8/8/8/4P3/4K2R w K - 0 1" ]
    for fen in fens do
        let pos = Position.OfFen fen
        let moves = collectLegal pos
        for m in moves do
            let idx = mapMoveToIndex pos m
            Assert.True(
                idx >= 0 && idx < NumMovesIndices,
                sprintf "idx %d out of [0,%d) for move %s in %s" idx NumMovesIndices (toUci m) fen)
    // e4xd5: white pawn captures a free rook — winning capture, must pass SEE threshold -108.
    let p = Position.OfFen "4k3/8/8/3r4/4P3/8/8/4K3 w - - 0 1"
    let moves = collectLegal p
    let exd5 = moves |> Array.find (fun m -> toUci m = "e4d5")
    Assert.True(see p exd5 -108, "winning capture e4d5 (pawn takes free rook) should pass SEE")
    // Qxd5: queen captures free pawn — winning, should pass SEE.
    let p2 = Position.OfFen "4k3/8/8/3p4/8/8/8/3QK3 w - - 0 1"
    let moves2 = collectLegal p2
    let qxd5 = moves2 |> Array.find (fun m -> toUci m = "d1d5")
    Assert.True(see p2 qxd5 -108, "winning capture Qxd5 (queen takes free pawn) should pass SEE")
    // White queen takes pawn on d5 defended by black rook on d8: Qxd5 Rxd5, net loss for white.
    // SEE trace: moveValue=100(pawn), balance=100-(-108)=208>=0; balance-=1250(Q)=-1042<0.
    // Loop: Black rook recaptures: balance=-(-1042)-1-650=391>=0; side flips to White; break.
    // Return White(0) != White(0) = false. So SEE correctly returns false.
    let p3 = Position.OfFen "3rk3/8/8/3p4/8/8/8/3QK3 w - - 0 1"
    let moves3 = collectLegal p3
    let qxd5 = moves3 |> Array.find (fun m -> toUci m = "d1d5")
    Assert.False(see p3 qxd5 -108, "losing Qxd5 defended by rook should fail SEE")

// ---------------------------------------------------------------------------
// mapFeatures (Task 2): sparse 3072-element policy input.
// ---------------------------------------------------------------------------

[<Fact>]
let ``mapFeatures: 32 distinct in-range features at startpos`` () =
    let pos = Position.OfFen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    let feats = mapFeatures pos
    Assert.Equal(32, feats.Length)
    for f in feats do Assert.True(f >= 0 && f < 3072, sprintf "feat %d out of range" f)
    Assert.Equal(feats.Length, (Set.ofArray feats).Count)

[<Fact>]
let ``mapFeatures: feature count equals piece count on several FENs`` () =
    for fen in [ "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
                 "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 b - - 0 1"
                 "4k3/8/8/8/8/8/4P3/4K3 b - - 0 1" ] do
        let pos = Position.OfFen fen
        let feats = mapFeatures pos
        let pieceCount = System.Numerics.BitOperations.PopCount(pos.Occupied)
        Assert.Equal(pieceCount, feats.Length)
        for f in feats do Assert.True(f >= 0 && f < 3072)

[<Fact>]
let ``mapMoveToIndex full index is good-SEE-doubled base index`` () =
    // For a clearly winning capture the full index should be in the upper half [FromTo, 2*FromTo).
    let p = Position.OfFen "4k3/8/8/3r4/4P3/8/8/4K3 w - - 0 1"
    let moves = collectLegal p
    let exd5 = moves |> Array.find (fun m -> toUci m = "e4d5")
    let baseIdx = moveToIndexBase p exd5
    let fullIdx = mapMoveToIndex p exd5
    Assert.Equal(FromTo + baseIdx, fullIdx) // good SEE -> upper half
    // For a quiet move the full index should equal the base index (lower half [0, FromTo)).
    let pq = Position.OfFen "4k3/8/8/8/4P3/8/8/4K3 w - - 0 1"
    let movesq = collectLegal pq
    // e4e5 is quiet
    let e4e5 = movesq |> Array.find (fun m -> toUci m = "e4e5")
    let baseQ = moveToIndexBase pq e4e5
    let fullQ = mapMoveToIndex pq e4e5
    // Quiet pawn move e4e5: moveValue=0, threshold=-108 -> balance=108>=0; balance-=100(P)=8>=0 -> SEE true.
    // So quiet pawn moves ALSO pass SEE(-108) and land in the upper half [FromTo, 2*FromTo).
    Assert.Equal(FromTo + baseQ, fullQ)
    Assert.True(fullQ >= 0 && fullQ < NumMovesIndices)

// ---------------------------------------------------------------------------
// Task 3 — parity test: our int8 inference must match the Jackal oracle BASE net
// within ±1.5 percentage points for the top moves of both test positions.
//
// Oracle BASE net values captured from:
//   terminal.exe < "position startpos\npolicy\nposition fen <kiwipete>\npolicy\nquit"
//
//   startpos BASE:  d2d4 23.71%, c2c4 22.01%, g1f3 13.59%, e2e4 13.26%
//   kiwipete BASE:  e2a6 38.87%, d5e6 10.36%, c3b5 7.60%,  d5d6 5.29%
// ---------------------------------------------------------------------------

[<Fact>]
let ``policy priors match the Jackal oracle BASE net`` () =
    match netPath () with
    | None -> () // soft-skip: net not present
    | Some p ->
        match load p with
        | Failed r -> Assert.Fail r
        | Loaded net ->
            let check fen (expected: (string * float32) list) =
                let pos = Position.OfFen fen
                let buf = Array.zeroCreate<Move> 256
                let span = System.Span<Move>(buf, 0, 256)
                let n = generateLegal pos span
                let moves = Array.sub buf 0 n
                let priors : float32[] = policyPriors net pos moves n
                for (uci, exp) in expected do
                    let i = moves |> Array.findIndex (fun m -> toUci m = uci)
                    let pct = priors.[i] * 100.0f
                    Assert.True(
                        abs (pct - exp) < 1.5f,
                        sprintf "%s: mine %.2f%% vs oracle %.2f%%" uci pct exp)

            // startpos — BASE net top 4
            check "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
                [ "d2d4", 23.71f
                  "c2c4", 22.01f
                  "g1f3", 13.59f
                  "e2e4", 13.26f ]

            // kiwipete — BASE net top 4
            check "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
                [ "e2a6", 38.87f
                  "d5e6", 10.36f
                  "c3b5",  7.60f
                  "d5d6",  5.29f ]

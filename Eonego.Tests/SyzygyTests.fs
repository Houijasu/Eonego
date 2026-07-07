/// Syzygy probing (Syzygy.fs): gate/no-table behavior (always run) plus real-table probes and a
/// Retrograde cross-check — the ~1600-line Fathom port previously had ZERO test coverage.
///
/// The real-table tests are OPT-IN via EONEGO_TEST_TB=1: Syzygy module state is GLOBAL (init/free),
/// xUnit runs test classes in parallel, and defaultConfig.UseSyzygy = true — a loaded table would
/// flip Syzygy.Largest under every concurrently-running search test and perturb their pinned node
/// counts. Run them isolated:
///   EONEGO_TEST_TB=1 dotnet test --filter "FullyQualifiedName~SyzygyTests"
/// Tables default to C:\Syzygy\Syzygy345WDL;C:\Syzygy\Syzygy345DTZ (EONEGO_TEST_TB_PATH overrides);
/// absent tables soft-skip, like tryLoadNet.
module Eonego.Tests.SyzygyTests

open System
open System.Text
open Xunit
open Eonego
open Eonego.Bitboard
open Eonego.Move
open Eonego.Position
open Eonego.Tests.TestFixtures

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// KQvK, White to move: trivially won for White.
let private KQvKWhite = "4k3/8/4K3/8/8/8/8/4Q3 w - - 0 1"
/// Same position, Black (the bare king) to move: lost for the side to move.
let private KQvKBlack = "4k3/8/4K3/8/8/8/8/4Q3 b - - 0 1"
/// Wrong-corner KPvK (h-pawn, defender on h8): a book draw.
let private KPvKDraw = "7k/8/8/8/8/8/7P/7K w - - 0 1"
/// KQvK with exactly one mate in 1 (Qg1-g7#, guarded by Kf6).
let private KQvKMate1 = "7k/8/5K2/8/8/8/8/6Q1 w - - 0 1"
/// KRvK with a live castling right — the probeRoot gate must refuse it.
let private KRvKCastle = "4k3/8/8/8/8/8/8/4K2R w K - 0 1"

let private tbPath =
    match Environment.GetEnvironmentVariable "EONEGO_TEST_TB_PATH" with
    | null
    | "" -> @"C:\Syzygy\Syzygy345WDL;C:\Syzygy\Syzygy345DTZ"
    | p -> p

let private tbEnabled =
    Environment.GetEnvironmentVariable "EONEGO_TEST_TB" = "1"
    && (tbPath.Split(';') |> Array.forall IO.Directory.Exists)

/// Run `f` with real tables loaded, then ALWAYS free them so Largest is back to 0 and no state
/// leaks into later tests in this (sequential) collection. Soft-skips when opt-in is absent.
let private withTables (f: unit -> unit) =
    if tbEnabled then
        try
            Assert.True(Syzygy.init tbPath, "Syzygy.init failed for " + tbPath)
            Assert.True(Syzygy.Largest >= 3, "no tables found under " + tbPath)
            f ()
        finally
            Syzygy.free ()

// ---------------------------------------------------------------------------
// No tables: every probe must fail cleanly and the root filter must impose nothing.
// ---------------------------------------------------------------------------

[<Fact>]
let ``probes are unavailable without tables`` () =
    Syzygy.free ()
    Assert.Equal(0, Syzygy.Largest)

    // Direct probes only fail deterministically in a process that never loaded tables: free()
    // zeroes Largest (which inerts every SEARCH gate — the engine-facing contract) but keeps the
    // lazy tbHash/paths, so a post-free direct probe may still re-read files. Under the opt-in
    // table run the withTables tests may have executed first, so scope these to the default run.
    if not tbEnabled then
        let pos = Position.OfFen KQvKWhite
        Assert.Equal(Int32.MinValue, Syzygy.probeWDL pos)
        Assert.Equal(Int32.MinValue, Syzygy.probeDTZ pos)

[<Fact>]
let ``probeRoot imposes no restriction without tables`` () =
    Syzygy.free ()
    Assert.Empty(Syzygy.probeRoot (Position.OfFen KQvKWhite))

[<Fact>]
let ``init with an empty path succeeds and clears`` () =
    Assert.True(Syzygy.init "")
    Assert.Equal(0, Syzygy.Largest)

// ---------------------------------------------------------------------------
// Real tables (opt-in): WDL/DTZ classes on known positions.
// ---------------------------------------------------------------------------

[<Fact>]
let ``wdl is side-to-move relative on the KQvK classes`` () =
    withTables (fun () ->
        Assert.Equal(2, Syzygy.probeWDL (Position.OfFen KQvKWhite))
        Assert.Equal(-2, Syzygy.probeWDL (Position.OfFen KQvKBlack))
        Assert.Equal(0, Syzygy.probeWDL (Position.OfFen KPvKDraw)))

[<Fact>]
let ``dtz sign and 3-man magnitude are sane`` () =
    withTables (fun () ->
        let win = Syzygy.probeDTZ (Position.OfFen KQvKWhite)
        Assert.True(win > 0 && win <= 40, "KQvK dtz out of range: " + string win)
        let loss = Syzygy.probeDTZ (Position.OfFen KQvKBlack)
        Assert.True(loss < 0 && loss >= -40, "KvKQ dtz out of range: " + string loss)
        Assert.Equal(0, Syzygy.probeDTZ (Position.OfFen KPvKDraw)))

// ---------------------------------------------------------------------------
// Real tables (opt-in): probeRoot filtering semantics.
// ---------------------------------------------------------------------------

[<Fact>]
let ``probeRoot keeps a strict win-preserving subset in a won position`` () =
    withTables (fun () ->
        let pos = Position.OfFen KQvKWhite
        let all = collectLegal pos
        let kept = Syzygy.probeRoot pos
        Assert.True(kept.Length > 0, "filter returned nothing in a won position")
        Assert.True(kept.Length < all.Length, "filter kept every move — no restriction happened")

        // Every kept move must preserve the win: the child (opponent to move) reads as a loss.
        for m in kept do
            Assert.Contains(m, all)
            pos.Make m
            Assert.Equal(-2, Syzygy.probeWDL pos)
            pos.Unmake m)

[<Fact>]
let ``probeRoot keeps exactly the mating moves when mate is on the board`` () =
    withTables (fun () ->
        let pos = Position.OfFen KQvKMate1
        let kept = Syzygy.probeRoot pos
        Assert.True(kept.Length > 0, "filter returned nothing with a mate in 1 available")

        // Mate scores v = 1, the minimal DTZ basis; quiet winning moves score >= 2 — so the
        // minimal-tie set must consist of checkmates only.
        for m in kept do
            pos.Make m
            Assert.True(pos.InCheck, "kept non-checking move " + toUCI m)
            Assert.Empty(collectLegal pos)
            pos.Unmake m)

[<Fact>]
let ``probeRoot imposes nothing for the losing side`` () =
    withTables (fun () ->
        // Max-resistance filtering is deliberately NOT applied: the search's practical judgement
        // stands when everything loses.
        Assert.Empty(Syzygy.probeRoot (Position.OfFen KQvKBlack)))

[<Fact>]
let ``probeRoot refuses positions with live castling rights`` () =
    withTables (fun () -> Assert.Empty(Syzygy.probeRoot (Position.OfFen KRvKCastle)))

// ---------------------------------------------------------------------------
// Real tables (opt-in): cross-check WDL (and DTZ signs) against the retrograde solver — two
// fully independent implementations must agree on every sampled KQvK position.
// ---------------------------------------------------------------------------

/// Build a KQvK FEN from raw squares (LERF 0..63) and side to move.
let private fenOf (wk: int) (bk: int) (q: int) (stm: string) : string =
    let board = Array.create 64 '.'
    board.[wk] <- 'K'
    board.[bk] <- 'k'
    board.[q] <- 'Q'
    let sb = StringBuilder(32)

    for rank in 7 .. -1 .. 0 do
        let mutable run = 0

        for file in 0 .. 7 do
            let c = board.[rank * 8 + file]

            if c = '.' then
                run <- run + 1
            else
                if run > 0 then sb.Append(run) |> ignore
                run <- 0
                sb.Append(c) |> ignore

        if run > 0 then sb.Append(run) |> ignore
        if rank > 0 then sb.Append('/') |> ignore

    sb.Append(' ').Append(stm).Append(" - - 0 1") |> ignore
    sb.ToString()

let private chebyshev (a: int) (b: int) =
    max (abs (fileOf a - fileOf b)) (abs (rankOf a - rankOf b))

[<Fact>]
let ``wdl agrees with the retrograde solver across sampled KQvK positions`` () =
    withTables (fun () ->
        Retrograde.ensureSolved (makePiece White Queen)
        let mutable nChecked = 0

        for wk in 0 .. 3 .. 63 do
            for bk in 0 .. 5 .. 63 do
                for q in 0 .. 7 .. 63 do
                    if wk <> bk && wk <> q && bk <> q && chebyshev wk bk > 1 then
                        for stm in [| "w"; "b" |] do
                            // Legal iff the side NOT to move is not in check (king-capturable
                            // positions are illegal and crash Fathom's genCaptures by design).
                            let flipped =
                                Position.OfFen(fenOf wk bk q (if stm = "w" then "b" else "w"))

                            if not flipped.InCheck then
                                let pos = Position.OfFen(fenOf wk bk q stm)

                                match Retrograde.probe pos with
                                | ValueNone -> ()
                                | ValueSome v ->
                                    let wdl = Syzygy.probeWDL pos
                                    // KQvK has no cursed wins (max DTM ~10 moves), so the retro
                                    // sign maps to the exact WDL class.
                                    let expected = if v > 0y then 2 elif v < 0y then -2 else 0

                                    Assert.True(
                                        (expected = wdl),
                                        "wdl mismatch at "
                                        + fenOf wk bk q stm
                                        + ": retro="
                                        + string v
                                        + " wdl="
                                        + string wdl
                                    )

                                    // DTZ agreement on the sign, sampled (pricier probe).
                                    if nChecked % 8 = 0 then
                                        let dtz = Syzygy.probeDTZ pos
                                        Assert.True(
                                            ((sign dtz) = (sign expected)),
                                            "dtz sign mismatch at " + fenOf wk bk q stm
                                        )

                                    nChecked <- nChecked + 1

        Assert.True(nChecked > 500, "too few positions cross-checked: " + string nChecked))

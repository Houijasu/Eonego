/// df-pn mate oracle: proof/disproof units on hand-verified checks-only positions, the
/// verification-pass certification property (every published proof replays to a legal all-check
/// mate), rule-50 / repetition / abort edges, determinism, and the strided cross-check against the
/// retrograde ground truth. Every test builds its OWN Solver — the engine singleton
/// (DFPN.shared()) is never touched, so xUnit parallelism cannot create cross-class order
/// dependence (the RetrogradeTests probe lesson).
module Eonego.Tests.DFPNTests

open System
open System.Text
open Xunit
open Eonego.Bitboard
open Eonego.Move
open Eonego.Position
open Eonego.MoveGeneration
open Eonego.Retrograde
open Eonego.DFPN
open Eonego.Transposition
open Eonego.Search
open Eonego.Tests.TestFixtures

let private neverStop: unit -> bool = fun () -> false

/// Small private solver (4 MiB table is plenty for unit positions).
let private solve (fen: string) : SolveResult =
    Solver(4).SolveWith(fen, [||], neverStop, 5_000_000L, 2_000_000L)

/// The certification property: a Proven result replays from the root as a legal line where every
/// attacker move gives check and the final position is checkmate, with |PV| = MatePlies (odd).
let private assertCertifiedMate (fen: string) (r: SolveResult) =
    Assert.True(r.Proven, "expected a proven checks-only mate for " + fen)
    Assert.Equal(r.MatePlies, r.PV.Length)
    Assert.Equal(1, r.MatePlies % 2) // all-check mates end on an attacker move
    Assert.Equal(r.Move, r.PV.[0])

    let pos = Position.OfFen fen

    for i in 0 .. r.PV.Length - 1 do
        let m = r.PV.[i]
        Assert.Contains(m, collectLegal pos)

        if i % 2 = 0 then
            Assert.True(pos.GivesCheck m, "attacker PV move must check: " + toUCI m)

        pos.Make m

    Assert.True(pos.InCheck)
    Assert.Empty(collectLegal pos) // checkmate

// ---------------------------------------------------------------------------
// Proofs
// ---------------------------------------------------------------------------

[<Fact>]
let ``proves mate in one`` () =
    // wKb6, Rh1, bKa8: Rh8#. Same FEN SearchTests pins to MATE - 1 / h1h8.
    let r = solve "k7/8/1K6/8/8/8/8/7R w - - 0 1"
    assertCertifiedMate "k7/8/1K6/8/8/8/8/7R w - - 0 1" r
    Assert.Equal(1, r.MatePlies)
    Assert.Equal("h1h8", toUCI r.Move)

[<Fact>]
let ``proves an all-check mate in three plies`` () =
    // wKb1, Rh6, Rg1, bKd7: 1.Rg7+! (Rh6 keeps rank 6 sealed; the king is too far to capture
    // either rook) Kc8/Kd8/Ke8 2.Rh8#. 1.Rh7+? instead opens rank 6 (Kd6 runs) — so the proving
    // first move is uniquely g1g7. No mate in one exists.
    let fen = "8/3k4/7R/8/8/8/8/1K4R1 w - - 0 1"
    let r = solve fen
    assertCertifiedMate fen r
    Assert.Equal(3, r.MatePlies)
    Assert.Equal("g1g7", toUCI r.Move)

[<Fact>]
let ``mate on the hundredth rule50 ply is still a proof`` () =
    // rule50 = 99: the mating move lands exactly on 100 — mate wins (the rule-50 arm runs after
    // the mate terminal, matching the search's isImmediateDraw convention).
    let fen = "k7/8/1K6/8/8/8/8/7R w - - 99 80"
    let r = solve fen
    Assert.True r.Proven
    Assert.Equal(1, r.MatePlies)

[<Fact>]
let ``rule50 exhaustion blocks a deeper proof`` () =
    // Same two-rook mate-in-3, but rule50 = 99: after 1.Rg7+ the defender node sits at 100 with
    // legal moves — draw claim, so no checks-only mate is provable.
    let r = solve "8/3k4/7R/8/8/8/8/1K4R1 w - - 99 80"
    Assert.False r.Proven

// ---------------------------------------------------------------------------
// Disproofs
// ---------------------------------------------------------------------------

[<Fact>]
let ``disproves the start position immediately`` () =
    // No checking move at the root: OR node with zero children.
    let r = solve "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    Assert.True r.Disproved
    Assert.False r.Proven

[<Fact>]
let ``disproves a mate whose key move is quiet`` () =
    // SearchTests' mate-in-2 (7k/8/5K2/... 1.Kg6! then Ra8#): the key move is QUIET, so the
    // checks-only oracle must disprove — the alpha-beta search owns this mate, not the oracle.
    let r = solve "7k/8/5K2/8/8/8/8/R7 w - - 0 1"
    Assert.True r.Disproved
    Assert.False r.Proven

[<Fact>]
let ``disproves check sequences that can never mate a bare king`` () =
    // KR vs bare K with the attacking king far away: a lone rook cannot mate by checks (every
    // mating net needs quiet king approaches). Check shuttles repeat positions, so this exercises
    // the path-repetition disproof arm as well as plain check exhaustion.
    let r = solve "7k/8/8/8/8/8/8/RK6 w - - 0 1"
    Assert.True r.Disproved
    Assert.False r.Proven

// ---------------------------------------------------------------------------
// Control: abort, determinism, publication slot
// ---------------------------------------------------------------------------

[<Fact>]
let ``an aborted solve publishes nothing`` () =
    // A 5-node cap aborts any non-trivial solve (the cap is checked on every MID entry). A tiny
    // position can legitimately finish before the 2048-node stop poll — that is correct behaviour,
    // so the abort test uses the deterministic node cap, not the stop callback.
    let r =
        Solver(4).SolveWith("8/3k4/7R/8/8/8/8/1K4R1 w - - 0 1", [||], neverStop, 5L, 2_000_000L)

    Assert.False r.Proven
    Assert.False r.Disproved

[<Fact>]
let ``same solver, same position, identical node counts`` () =
    let s = Solver(4)
    let fen = "8/3k4/R7/8/8/8/8/1R5K w - - 0 1"
    let struct (pn1, dn1) = s.RawSolve(fen, [||], neverStop, 0L)
    let n1 = s.Nodes
    let struct (pn2, dn2) = s.RawSolve(fen, [||], neverStop, 0L)
    Assert.Equal(pn1, pn2)
    Assert.Equal(dn1, dn2)
    Assert.Equal(n1, s.Nodes) // per-solve generation bump => byte-identical re-solve

[<Fact>]
let ``oracle result slot round-trips`` () =
    let o = OracleResult()
    let struct (has0, _, _) = o.TryGet()
    Assert.False has0
    o.Publish(parseUCI "b1b7", 3, [| parseUCI "b1b7" |])
    let struct (has, m, plies) = o.TryGet()
    Assert.True has
    Assert.Equal("b1b7", toUCI m)
    Assert.Equal(3, plies)
    Assert.Single o.PV |> ignore

// ---------------------------------------------------------------------------
// goCore consumption: the bestmove override reads the control's Oracle slot after the search.
// Pre-publishing a "proof" (UseDFPN=false — no oracle thread) makes these fully deterministic.
// ---------------------------------------------------------------------------

let private goWithOracle (fen: string) (depth: int) (publish: OracleResult -> unit) =
    let cfg = { defaultConfig with Threads = 1 }

    let limits =
        { defaultLimits with Depth = depth }

    let control = SearchControl(cfg, limits, TranspositionTable(16), fen, [||])
    publish control.Oracle
    Eonego.Search.go control |> ignore
    control

[<Fact>]
let ``a published proof overrides the searched bestmove and score`` () =
    // Startpos, fake proven move a2a3 (a move the search would never choose) with matePlies = 5:
    // the override must both switch the move and report MATE - 5.
    let control =
        goWithOracle "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" 2 (fun o ->
            o.Publish(parseUCI "a2a3", 5, [| parseUCI "a2a3" |]))

    Assert.Equal("a2a3", toUCI control.LastBest)
    Assert.Equal(MATE - 5, control.LastScore)

[<Fact>]
let ``a shorter search-proven mate beats the oracle`` () =
    // Mate-in-1 FEN at depth 4: the search proves MATE - 1 itself; a published 9-ply proof must
    // NOT displace the shorter mate.
    let control =
        goWithOracle "k7/8/1K6/8/8/8/8/7R w - - 0 1" 4 (fun o ->
            o.Publish(parseUCI "h1h2", 9, [| parseUCI "h1h2" |]))

    Assert.Equal("h1h8", toUCI control.LastBest)
    Assert.Equal(MATE - 1, control.LastScore)

// ---------------------------------------------------------------------------
// Tactics-suite certification sweep: every row the oracle DOES prove must certify. The suite is
// teacher-verified mates, but most need quiet key moves — no minimum proof count is asserted.
// ---------------------------------------------------------------------------

let private repoRoot () =
    let mutable dir = IO.DirectoryInfo(AppContext.BaseDirectory)
    let mutable root = None

    while root.IsNone && not (isNull dir) do
        if IO.File.Exists(IO.Path.Combine(dir.FullName, "Eonego.slnx")) then
            root <- Some dir.FullName

        dir <- dir.Parent

    root

[<Fact>]
let ``every proof over the tactics mate rows certifies`` () =
    match repoRoot () with
    | None -> () // soft-skip outside a full checkout
    | Some root ->
        let path = IO.Path.Combine(root, "trainer", "suites", "kga_tactics.tsv")

        if IO.File.Exists path then
            let s = Solver(16)
            let mutable rows = 0

            for line in IO.File.ReadAllLines path do
                let parts = line.Split '\t'

                if parts.Length >= 3 && parts.[2].Trim() = "mate" then
                    rows <- rows + 1
                    let fen = parts.[0]
                    let r = s.SolveWith(fen, [||], neverStop, 1_000_000L, 1_000_000L)

                    if r.Proven then
                        assertCertifiedMate fen r

            Assert.True(rows >= 50, "expected the 56 mate-tagged rows, saw " + string rows)

// ---------------------------------------------------------------------------
// Retrograde cross-check: DFPN proof => the exact table says the side to move wins, and the
// checks-only distance can only be >= the DTM optimum. One-directional by design — retro wins via
// quiet key moves are expected DFPN disproofs. Uses a PRIVATE solveSignature values array (no
// publication), so no shared solver state leaks to other test classes.
// ---------------------------------------------------------------------------

let private wqValues = lazy (solveSignature (makePiece White Queen) Array.empty)

[<Fact>]
[<Trait("Category", "Slow")>]
let ``DFPN proofs agree with the retrograde queen table on a stride sample`` () =
    let values = wqValues.Force()
    let s = Solver(16)
    let sb = StringBuilder(80)
    let pce = makePiece White Queen
    let mutable idx = 0
    let mutable checked' = 0

    while idx < RetroSize do
        if values.[idx] <> RetroIllegal then
            let fen = fenOf sb pce (idxStm idx) (idxWk idx) (idxBk idx) (idxPc idx)
            let r = s.SolveWith(fen, [||], neverStop, 200_000L, 200_000L)

            if r.Proven then
                checked' <- checked' + 1
                let v = values.[idx]
                Assert.True(v > 0y, "DFPN proved a mate the exact table calls " + string (int v) + ": " + fen)
                Assert.True(r.MatePlies >= retroDtm v)
                assertCertifiedMate fen r

        idx <- idx + 4093 // odd stride, coprime with RetroSize — ~128 samples

    Assert.True(checked' > 0, "stride sample never hit a provable checks-only mate")

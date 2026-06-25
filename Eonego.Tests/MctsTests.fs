/// MCTS-root / negamax-leaf hybrid tests.
///
/// Two layers: (1) pure-helper unit tests pin the arena, PUCT, value scaling, and the MCTS-Solver in
/// isolation; (2) full-hybrid tests via the deterministic `mctsToIterations(Tree)` oracle assert the
/// search commits to forced mates (solver) and wins hanging material (the αβ leaf), runs EXACTLY the
/// requested iteration count (root.N == iters — the backup off-by-one guard), and is deterministic.
module Eonego.Tests.MctsTests

open Xunit
open Eonego.Bitboard
open Eonego.Move
open Eonego.Position
open Eonego.Transposition
open Eonego.Search
open Eonego.Mcts
open Eonego.Tests.TestFixtures

let private approx (eps: float32) (a: float32) (b: float32) = abs (a - b) < eps

// ---------------------------------------------------------------------------
// Value scaling: cp <-> winprob
// ---------------------------------------------------------------------------
[<Fact>]
let ``cpToWinProb is 0.5 at cp=0, monotone, and clamps mates`` () =
    let k = 200.0f
    Assert.True(approx 1e-4f (cpToWinProb 0 k) 0.5f)
    Assert.True(cpToWinProb 100 k > cpToWinProb 0 k)
    Assert.True(cpToWinProb 0 k > cpToWinProb -100 k)
    Assert.Equal(1.0f, cpToWinProb MATE k)
    Assert.Equal(0.0f, cpToWinProb -MATE k)
    Assert.Equal(1.0f, cpToWinProb MATE_IN_MAX_PLY k)
    Assert.Equal(0.0f, cpToWinProb -MATE_IN_MAX_PLY k)

[<Fact>]
let ``winProbToCp round-trips cpToWinProb within rounding`` () =
    let k = 200.0f

    for cp in [ -300; -150; -1; 0; 1; 150; 300 ] do
        let rt = winProbToCp (cpToWinProb cp k) k
        Assert.True(abs (rt - cp) <= 2, "cp=" + string cp + " round-tripped to " + string rt)

// ---------------------------------------------------------------------------
// PUCT
// ---------------------------------------------------------------------------
[<Fact>]
let ``puct uses FPU for an unvisited child and pure exploit at cpuct=0`` () =
    // Unvisited child, zero prior => exploitation collapses to FPU, exploration term is 0.
    Assert.True(approx 1e-5f (puctScoreOf 0 0.0f false 0.0f 1.0f 1.5f 0.5f) 0.5f)
    // Visited child with W/N = 1/4 (child POV) => parent-POV exploit = 1 - 0.25 = 0.75; cpuct=0 kills explore.
    Assert.True(approx 1e-5f (puctScoreOf 4 1.0f true 0.5f 10.0f 0.0f 0.5f) 0.75f)

[<Fact>]
let ``puct exploration rewards higher prior and parent visits`` () =
    // Same (unvisited) child, larger prior => larger U.
    let lo = puctScoreOf 0 0.0f true 0.1f 10.0f 1.5f 0.5f
    let hi = puctScoreOf 0 0.0f true 0.9f 10.0f 1.5f 0.5f
    Assert.True(hi > lo)

// ---------------------------------------------------------------------------
// Arena
// ---------------------------------------------------------------------------
[<Fact>]
let ``arena allocates sequentially and grows by doubling preserving contents`` () =
    let tree = MctsTree()
    let n0 = tree.AllocNode 0
    Assert.Equal(0, n0)
    tree.AddVisit(n0, 0.5f)
    tree.AddVisit(n0, 0.5f)
    tree.AddVisit(n0, 0.5f)
    // Force several grows past the initial 4096 capacity.
    for _ in 1..5000 do
        tree.AllocNode 1 |> ignore

    Assert.Equal(5001, tree.NodeCount)
    // Node 0's state survived the reallocations.
    Assert.Equal(3, tree.NodeN n0)
    Assert.True(approx 1e-5f (tree.NodeW n0) 1.5f)

// ---------------------------------------------------------------------------
// MCTS-Solver propagation (hand-built sub-trees)
// ---------------------------------------------------------------------------
/// Build a root with `childProofs.Length` children, each a leaf whose Proven tag is set as given.
let private mkSolverTree (childProofs: int list) : MctsTree * int =
    let tree = MctsTree()
    let root = tree.AllocNode 0
    let cnt = List.length childProofs
    let eb = tree.AllocEdges cnt
    childProofs
    |> List.iteri (fun i pr ->
        let c = tree.AllocNode 1
        tree.SetEdge(eb + i, MoveNone, 1.0f / float32 cnt)
        tree.SetEdgeChild(eb + i, c)
        tree.SetProven(c, pr))
    tree.SetNodeEdges(root, eb, cnt)
    (tree, root)

[<Fact>]
let ``solver proves a win when any child is a proven loss`` () =
    let (tree, root) = mkSolverTree [ PrUnknown; PrLoss; PrWin ]
    solverUpdate tree root
    Assert.Equal(PrWin, tree.NodeProven root)

[<Fact>]
let ``solver proves a loss only when all children are proven wins`` () =
    let (tree, root) = mkSolverTree [ PrWin; PrWin; PrWin ]
    solverUpdate tree root
    Assert.Equal(PrLoss, tree.NodeProven root)

[<Fact>]
let ``solver proves a draw when best forced outcome is a draw`` () =
    let (tree, root) = mkSolverTree [ PrWin; PrDraw; PrWin ]
    solverUpdate tree root
    Assert.Equal(PrDraw, tree.NodeProven root)

[<Fact>]
let ``solver stays unknown while a child is unproven (no winning move)`` () =
    let (tree, root) = mkSolverTree [ PrWin; PrUnknown; PrDraw ]
    solverUpdate tree root
    Assert.Equal(PrUnknown, tree.NodeProven root)

[<Fact>]
let ``solver never vacuously proves a zero-child node`` () =
    let tree = MctsTree()
    let n = tree.AllocNode 0 // no edges
    solverUpdate tree n
    Assert.Equal(PrUnknown, tree.NodeProven n)

// ---------------------------------------------------------------------------
// MCTS-Solver mate distance (shortest win / longest loss)
// ---------------------------------------------------------------------------
/// mkSolverTree variant carrying (provenTag, proofPly) per child via SetProvenPly.
let private mkSolverTreePly (children: (int * int) list) : MctsTree * int =
    let tree = MctsTree()
    let root = tree.AllocNode 0
    let cnt = List.length children
    let eb = tree.AllocEdges cnt

    children
    |> List.iteri (fun i (pr, ply) ->
        let c = tree.AllocNode 1
        tree.SetEdge(eb + i, MoveNone, 1.0f / float32 cnt)
        tree.SetEdgeChild(eb + i, c)
        tree.SetProvenPly(c, pr, ply))

    tree.SetNodeEdges(root, eb, cnt)
    (tree, root)

[<Fact>]
let ``solver proves the win with the shortest mate distance`` () =
    let (tree, root) = mkSolverTreePly [ (PrLoss, 3); (PrLoss, 1); (PrWin, 2) ]
    solverUpdate tree root
    Assert.Equal(PrWin, tree.NodeProven root)
    Assert.Equal(2, tree.NodeProofPly root) // min(3, 1) + 1

[<Fact>]
let ``solver proves the loss with the longest defence`` () =
    let (tree, root) = mkSolverTreePly [ (PrWin, 2); (PrWin, 5); (PrWin, 3) ]
    solverUpdate tree root
    Assert.Equal(PrLoss, tree.NodeProven root)
    Assert.Equal(6, tree.NodeProofPly root) // max(2, 5, 3) + 1

[<Fact>]
let ``bestRootMove prefers the shorter mate among winning moves`` () =
    let (tree, root) = mkSolverTreePly [ (PrLoss, 5); (PrLoss, 2) ]
    let (_, ei) = bestRootMove tree root
    Assert.Equal(1, ei) // edge 1 = the PrLoss (win-for-us) child with the shorter proof ply

// ---------------------------------------------------------------------------
// Full hybrid (deterministic oracle)
// ---------------------------------------------------------------------------
[<Fact>]
let ``runs exactly the requested number of iterations (root.N == iters)`` () =
    let iters = 300
    let (tree, root, _) = mctsToIterationsTree StartPosFen [||] iters defaultConfig None
    Assert.Equal(iters, tree.NodeN root)

[<Fact>]
let ``solver propagates a forced mate to the root`` () =
    // White: Kg6, Ra1; Black: Kh8 — a forced mate (Ra8 is mate-in-1; KR-vs-K is a forced win generally).
    // The solver tracks only win/loss/draw (no mate DISTANCE — a documented v1 limit), so we assert that a
    // forced mate reached the root, not that the SHORTEST mate move is chosen (several moves all force mate).
    let fen = "7k/8/6K1/8/8/8/8/R7 w - - 0 1"
    let struct (score, _, _) = mctsToIterations fen [||] 200 defaultConfig None
    Assert.True(score >= MATE_IN_MAX_PLY, "expected a mate score, got " + string score)

[<Fact>]
let ``leaf search makes the hybrid win a hanging queen`` () =
    // White Pe4 can take the Black queen on d5; the αβ leaf supplies the decisive value.
    match tryLoadSfNet () with
    | None -> () // soft-skip: SF net absent
    | Some net ->
        let fen = "4k3/8/8/3q4/4P3/8/8/3RK3 w - - 0 1"
        let struct (score, _, m) = mctsToIterations fen [||] 400 defaultConfig (Some net)
        Assert.Equal(mkSquare 3 4, toSq m) // exd5: destination d5
        Assert.True(score > 300, "expected a decisive material win, got " + string score)

[<Fact>]
let ``single-thread fixed-iteration MCTS is deterministic`` () =
    let fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1" // Kiwipete
    let struct (s1, _, m1) = mctsToIterations fen [||] 250 defaultConfig None
    let struct (s2, _, m2) = mctsToIterations fen [||] 250 defaultConfig None
    Assert.Equal(toUci m1, toUci m2)
    Assert.Equal(s1, s2)

[<Fact>]
let ``mctsToIterationsTree runs exactly the requested iterations (Worker.Iters)`` () =
    let (_, _, w) = mctsToIterationsTree StartPosFen [||] 50 defaultConfig None
    Assert.Equal(50L, w.Iters)

// ---------------------------------------------------------------------------
// Phase 1: mid-leaf-stop accounting, safe-envelope fallback, merged decision
// ---------------------------------------------------------------------------
[<Fact>]
let ``a mid-leaf node-limit stop never over-counts iterations (Iters == backed-up root visits)`` () =
    match tryLoadSfNet () with
    | None -> () // soft-skip: needs the SF leaf eval to generate leaf nodes
    | Some net ->
        // A tiny node budget trips the leaf negamax's own CheckTime mid-playout; the aborted iteration must
        // bump neither root visits nor the iteration counter, so the two stay exactly equal.
        let lim = { defaultLimits with Nodes = 500L }
        let (tree, root, w) = mctsToIterationsTreeLim StartPosFen [||] 1000 lim defaultConfig (Some net) None
        Assert.Equal(int64 (tree.NodeN root), w.Iters)

/// Build n plies of reversible knight shuffles (g1f3/f3g1, g8f6/f6g8) from startpos as a legal move list —
/// used to push the root past the MCTS safe envelope (safe = MaxPly - rootMoves - MaxSearchPly - 1 < 1).
let private knightShuffle (n: int) : Move[] =
    let pos = Position()
    pos.LoadFen StartPosFen
    let g1, f3 = mkSquare 6 0, mkSquare 5 2
    let g8, f6 = mkSquare 6 7, mkSquare 5 5
    let result = ResizeArray<Move>(n)

    for i in 0 .. n - 1 do
        let (frm, dst) =
            if i % 2 = 0 then (if (i / 2) % 2 = 0 then (g1, f3) else (f3, g1))
            else (if (i / 2) % 2 = 0 then (g8, f6) else (f6, g8))

        let m = collectLegal pos |> Array.find (fun mv -> fromSq mv = frm && toSq mv = dst)
        result.Add m
        pos.Make m

    result.ToArray()

[<Fact>]
let ``a root past the safe envelope falls back to alpha-beta (same move as Search.go)`` () =
    match tryLoadSfNet () with
    | None -> () // soft-skip: the fallback runs Search.go, which needs the SF eval
    | Some net ->
        let moves = knightShuffle 780 // safe = 777 - 780 < 1 => MCTS hands the whole search to alpha-beta
        let cfg = { defaultConfig with Threads = 1 }
        let lim = { defaultLimits with Depth = 4 }

        let runWith (search: SearchControl -> Move) =
            let tt = TranspositionTable(max 1 cfg.HashMb)
            search (SearchControl(cfg, lim, tt, StartPosFen, moves, ?net = Some net))

        Assert.Equal(toUci (runWith Eonego.Search.go), toUci (runWith (fun c -> mctsSearch c (MctsReuse()))))

[<Fact>]
let ``parallel mctsSearch returns a legal, coherent merged decision`` () =
    match tryLoadSfNet () with
    | None -> () // soft-skip: needs the SF leaf eval to see the won queen
    | Some net ->
        let fen = "4k3/8/8/3q4/4P3/8/8/3RK3 w - - 0 1" // exd5 wins the queen
        let cfg = { defaultConfig with Threads = 4 }
        let lim = { defaultLimits with Depth = 1 } // 1000 iters/worker; the αβ leaf supplies the value
        let tt = TranspositionTable(max 1 cfg.HashMb)
        let control = SearchControl(cfg, lim, tt, fen, [||], ?net = Some net)
        let m = mctsSearch control (MctsReuse())
        let p = Position()
        p.LoadFen fen
        let legal = collectLegal p |> Array.map toUci
        Assert.Contains(toUci m, legal) // the merged bestmove is legal
        Assert.Equal(toUci control.LastBest, toUci m) // bestmove == the published merged decision
        Assert.Equal(mkSquare 3 4, toSq m) // exd5: destination d5
        Assert.True(control.LastScore > 300, "merged score should reflect the won queen, got " + string control.LastScore)

[<Fact>]
let ``go depth iteration budget is global across threads, not multiplied per worker`` () =
    match tryLoadSfNet () with
    | None -> () // soft-skip: the leaf eval needs the SF net
    | Some net ->
        // `go depth 2` => 2000 total iterations. With a GLOBAL budget that total holds regardless of thread
        // count (Threads splits it), instead of the old per-worker budget where Threads=4 did 4×2000=8000.
        let run threads =
            let cfg = { defaultConfig with Threads = threads; MctsLeafDepth = 0 } // qsearch leaf => fast
            let lim = { defaultLimits with Depth = 2 }
            let tt = TranspositionTable(max 1 cfg.HashMb)
            let control = SearchControl(cfg, lim, tt, StartPosFen, [||], ?net = Some net)
            mctsSearch control (MctsReuse()) |> ignore
            control.IterSum()

        Assert.Equal(2000L, run 1) // single worker: exactly the budget
        let t4 = run 4
        // ~2000 (a few extra from workers passing the check before the last increments land), NOT 4×2000.
        Assert.True(t4 >= 2000L && t4 < 4000L, "Threads=4 total iters should be ~2000 (global), got " + string t4)

[<Fact>]
let ``batched Lc0 MCTS gather produces a legal, principled move with consistent counts`` () =
    match tryLoadSfNet (), tryLoadLc0 () with
    | Some sf, Some lc0 ->
        // UseLc0 + MctsBatchSize>1 routes runWorker through the virtual-loss batched gather (runBatch):
        // gather B leaves -> ONE forwardBatch for priors -> per-leaf negamax value + backup (revert VL).
        let cfg = { defaultConfig with UseLc0 = true; MctsBatchSize = 4 }
        let (tree, root, w) = mctsToIterationsTreeLim StartPosFen [||] 100 defaultLimits cfg (Some sf) (Some lc0)
        // root visits == iteration count (both count backed-up leaves); budget met (batch may overshoot by <B).
        Assert.Equal(int64 (tree.NodeN root), w.Iters)
        Assert.True(tree.NodeN root >= 100, "root visits " + string (tree.NodeN root))
        // the Lc0 priors + gather + backup must still concentrate visits on a principled opening.
        let (mv, _) = bestRootMove tree root
        let best = toUci mv
        let principled = [ "e2e4"; "d2d4"; "g1f3"; "c2c4"; "g2g3"; "b1c3" ]
        Assert.True(List.contains best principled, "batched startpos best = " + best)
    | _ -> () // soft-skip: needs both the SF and Lc0 nets

// ---------------------------------------------------------------------------
// WAC-style tactical regression suite (deterministic oracle + SF leaf). A guard against search regressions
// from future tuning; each position is a short, unambiguous tactic the hybrid must find.
// ---------------------------------------------------------------------------
[<Theory>]
[<InlineData("6k1/5ppp/8/8/8/8/8/R6K w - - 0 1", "a1a8")>] // mate in 1
[<InlineData("4k3/8/8/3q4/4P3/8/8/3RK3 w - - 0 1", "e4d5")>] // win the hanging queen
[<InlineData("3r2k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1", "d1d8")>] // win the back-rank rook
[<InlineData("3qk3/8/8/8/8/8/8/3RK3 w - - 0 1", "d1d8")>] // win the pinned queen
let ``WAC: hybrid finds the winning tactic`` (fen: string) (expected: string) =
    match tryLoadSfNet () with
    | None -> () // soft-skip: tactics need the SF leaf eval
    | Some net ->
        let struct (_, _, m) = mctsToIterations fen [||] 400 defaultConfig (Some net)
        Assert.Equal(expected, toUci m)

// ---------------------------------------------------------------------------
// Tree reuse: lazy root promotion to a played move's subtree
// ---------------------------------------------------------------------------
[<Fact>]
let ``PromoteToChild returns the expanded child subtree for a played move`` () =
    let pos = Position()
    pos.LoadFen StartPosFen
    let legal = collectLegal pos
    let mv0 = legal.[0]
    let mv1 = legal.[1]
    let tree = MctsTree()
    let root = tree.AllocNode 0
    let eb = tree.AllocEdges 2
    tree.SetEdge(eb, mv0, 0.6f)
    tree.SetEdge(eb + 1, mv1, 0.4f)
    tree.SetNodeEdges(root, eb, 2)
    // mv0 gets an expanded child carrying accumulated stats; mv1 stays unexpanded.
    let child0 = tree.AllocNode 1
    tree.SetEdgeChild(eb, child0)
    tree.AddVisit(child0, 0.7f)
    Assert.Equal(child0, tree.PromoteToChild(root, mv0)) // reuse the visited subtree
    Assert.Equal(-1, tree.PromoteToChild(root, mv1)) // edge exists but child unexpanded => not reusable
    Assert.Equal(-1, tree.PromoteToChild(root, MoveNone)) // no matching edge
    // the promoted child keeps its prior search stats (the point of reuse).
    Assert.Equal(1, tree.NodeN child0)
    Assert.True(abs (tree.NodeW child0 - 0.7f) < 1e-6f)

[<Fact>]
let ``mctsSearch populates the reuse state for the next search`` () =
    match tryLoadSfNet () with
    | None -> () // soft-skip: needs the SF leaf
    | Some net ->
        let cfg = { defaultConfig with Threads = 1 }
        let lim = { defaultLimits with Depth = 1 }
        let tt = TranspositionTable(max 1 cfg.HashMb)
        let control = SearchControl(cfg, lim, tt, StartPosFen, [||], ?net = Some net)
        let reuse = MctsReuse()
        mctsSearch control reuse |> ignore
        Assert.False(obj.ReferenceEquals(reuse.Tree, null)) // worker 0's tree was saved
        Assert.Equal(StartPosFen, reuse.Fen)
        Assert.Equal(0, reuse.Moves.Length)
        Assert.True(reuse.Root >= 0)
        Assert.True(reuse.Tree.NodeN reuse.Root > 0) // the saved root carries accumulated visits

[<Fact>]
let ``mctsSearch reuses the same tree object for a played move`` () =
    match tryLoadSfNet () with
    | None -> () // soft-skip: needs the SF leaf
    | Some net ->
        let cfg = { defaultConfig with Threads = 1 }
        let lim = { defaultLimits with Depth = 2 } // enough iterations to expand e2e4's child

        let mk fen moves =
            let tt = TranspositionTable(max 1 cfg.HashMb)
            SearchControl(cfg, lim, tt, fen, moves, ?net = Some net)

        let p = Position()
        p.LoadFen StartPosFen
        let e2e4 = collectLegal p |> Array.find (fun m -> toUci m = "e2e4")

        let reuse = MctsReuse()
        mctsSearch (mk StartPosFen [||]) reuse |> ignore
        let treeAfter1 = reuse.Tree
        // search 2 extends by e2e4: worker 0 should lazily promote that subtree, i.e. keep the SAME tree object.
        mctsSearch (mk StartPosFen [| e2e4 |]) reuse |> ignore
        Assert.True(obj.ReferenceEquals(reuse.Tree, treeAfter1), "expected the e2e4 subtree to be reused")

[<Fact>]
let ``Compact preserves the live subtree and drops garbage`` () =
    let tree = MctsTree()
    let root = tree.AllocNode 0
    let pos = Position()
    pos.LoadFen StartPosFen
    let legal = collectLegal pos
    let eb = tree.AllocEdges 2
    tree.SetEdge(eb, legal.[0], 0.5f)
    tree.SetEdge(eb + 1, legal.[1], 0.5f)
    tree.SetNodeEdges(root, eb, 2)
    // child0 with a grandchild — the live subtree to keep.
    let c0 = tree.AllocNode 1
    tree.SetEdgeChild(eb, c0)
    tree.AddVisit(c0, 0.7f)
    let geb = tree.AllocEdges 1
    tree.SetEdge(geb, legal.[2], 1.0f)
    let gc = tree.AllocNode 0
    tree.SetEdgeChild(geb, gc)
    tree.AddVisit(gc, 0.3f)
    tree.SetNodeEdges(c0, geb, 1)
    // child1 — becomes unreachable garbage when we compact around c0.
    let c1 = tree.AllocNode 1
    tree.SetEdgeChild(eb + 1, c1)
    tree.AddVisit(c1, 0.4f)
    Assert.Equal(4, tree.NodeCount) // root, c0, gc, c1
    // compact the subtree rooted at c0: keeps c0 + gc (2 nodes), drops root + c1.
    let (compacted, newRoot) = tree.Compact c0
    Assert.Equal(2, compacted.NodeCount)
    Assert.Equal(1, compacted.NodeN newRoot)
    Assert.True(abs (compacted.NodeW newRoot - 0.7f) < 1e-6f)
    Assert.Equal(1, compacted.NodeNumEdges newRoot)
    let ngc = compacted.EdgeChild(compacted.NodeFirstEdge newRoot)
    Assert.True(ngc >= 0) // the grandchild survived with its stats
    Assert.Equal(1, compacted.NodeN ngc)
    Assert.True(abs (compacted.NodeW ngc - 0.3f) < 1e-6f)

// ---------------------------------------------------------------------------
// Ponder clock: search unbounded until ponderhit arms the remembered budget (race-safe both orders)
// ---------------------------------------------------------------------------
[<Fact>]
let ``ponder clock stays unbounded until ponderhit arms the budget`` () =
    let mkControl ponder =
        let lim = { defaultLimits with Ponder = ponder; WTime = 3000; BTime = 3000 }
        SearchControl(defaultConfig, lim, TranspositionTable(1), StartPosFen, [||])

    // ponder: the budget is stored but the clock runs unbounded (soft = 0) until ponderhit.
    let c1 = mkControl true
    c1.StartClockPonder 100L 400L true
    Assert.Equal(0L, c1.BaseSoftMs)
    Assert.False(c1.SoftTimeUp) // soft = 0 is never "up"
    c1.PonderHit()
    Assert.Equal(100L, c1.BaseSoftMs) // ponderhit armed the remembered soft budget

    // race: ponderhit arrives BEFORE the search stored the budget -> StartClockPonder arms it when it runs.
    let c2 = mkControl true
    c2.PonderHit()
    Assert.Equal(0L, c2.BaseSoftMs) // nothing to arm yet
    c2.StartClockPonder 100L 400L true
    Assert.Equal(100L, c2.BaseSoftMs)

    // a non-ponder search: ponderhit is a no-op (cannot disturb the real clock).
    let c3 = SearchControl(defaultConfig, { defaultLimits with WTime = 3000 }, TranspositionTable(1), StartPosFen, [||])
    c3.StartClockPonder 100L 400L false
    Assert.Equal(100L, c3.BaseSoftMs)
    c3.PonderHit()
    Assert.Equal(100L, c3.BaseSoftMs)

[<Fact>]
let ``go nodes is an MCTS iteration budget, not a leaf-node limit`` () =
    match tryLoadSfNet () with
    | None -> () // soft-skip: the leaf eval generates the nodes that the old semantics tripped on
    | Some net ->
        let cfg = { defaultConfig with Threads = 1 }
        let lim = { defaultLimits with Nodes = 300L }
        let tt = TranspositionTable(max 1 cfg.HashMb)
        let control = SearchControl(cfg, lim, tt, StartPosFen, [||], ?net = Some net)
        mctsSearch control (MctsReuse()) |> ignore
        // ~300 backed-up iterations (the old leaf-node semantics stopped at ~0 because the first depth-6 leaf
        // already exceeds 300 nodes, returning an unsearched move).
        Assert.True(control.IterSum() >= 300L, "iters " + string (control.IterSum()))
        Assert.True(control.IterSum() <= 302L, "iters " + string (control.IterSum()))

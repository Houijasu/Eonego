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

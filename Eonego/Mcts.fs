/// Eonego — MCTS-root / negamax-leaf hybrid (Baier–Winands MCTS-MB). An MCTS tree manages the
/// root/strategic search; each leaf is evaluated by a fixed-depth full-window negamax (the existing
/// engine, reused UNCHANGED as the oracle: NNUE + qsearch + the full pruning stack). MCTS-Solver
/// propagates proven win/loss/draw so the averaging backup commits to forced lines the αβ leaf finds.
///
/// ROOT-PARALLEL: `mctsSearch` fans out N independent trees (one per Worker), sharing only the lock-free
/// TT and the Volatile stop flag, and decides from their MERGED root statistics. The board is real: the
/// selection path is walked with Position.Make/Unmake and the leaf negamax runs on that exact Position, so
/// repetition/50-move/insufficient-material detection works via the engine's own history. Each tree is a
/// flat index-based arena (no GC pointers), allocated once per `go` and grown by doubling. Priors come from
/// the Lc0 policy CNN when one is loaded (EONEGO_LC0), else a softmax over move-ordering history scores.
/// This is the UCI production search (Uci.fs hardwires UseMcts on); αβ (Search.go) remains the leaf oracle
/// and the deterministic test path, and also takes over when a root is too deep for a fixed-depth leaf.
///
/// AOT/F#: no printfn (Console.Out only); arena = struct records in raw arrays (same in-place mutation
/// idiom as Search.StackEntry[]); never hold a byref/array-ref across an Alloc (a realloc invalidates).
module Eonego.Mcts

open System
open System.Runtime.CompilerServices
open System.Threading
open Eonego.Bitboard
open Eonego.Move
open Eonego.Position
open Eonego.MoveGeneration
open Eonego.History
open Eonego.SfNnue
open Eonego.Transposition
open Eonego.Search

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
[<Literal>]
let PrUnknown = 0

[<Literal>]
let PrWin = 1 // side-to-move at THIS node has a proven win

[<Literal>]
let PrLoss = 2 // side-to-move at THIS node is proven lost

[<Literal>]
let PrDraw = 3

[<Literal>]
let MctsIterPerDepth = 1000 // `go depth N` (no time stop) => N * this many iterations

let private Fpu = 0.5f // first-play-urgency for unvisited children (neutral)
let private MctsPriorTemp = 1000.0f // softmax temperature over move-ordering scores
// Periodic UCI info cadence: WALL-CLOCK, not iteration count. With Lc0 priors the driver runs a few it/s, so
// an iteration-count trigger (e.g. every 256) would emit nothing for a minute+ and analysis looks frozen.
let private MctsReportIntervalMs = 500L

// ---------------------------------------------------------------------------
// Flat arena (struct records in raw arrays — mirrors Search.StackEntry)
// ---------------------------------------------------------------------------
[<Struct>]
type MctsEdge =
    { mutable Move: Move
      mutable Prior: float32 // P(s,a), normalized over siblings
      mutable Child: int } // index into nodes[]; -1 = not yet expanded

[<Struct>]
type MctsNode =
    { mutable N: int // visit count
      mutable W: float32 // sum of leaf win-probs, from THIS node's side-to-move POV
      mutable FirstEdge: int // index into edges[]; -1 = unexpanded
      mutable NumEdges: int
      mutable Proven: int // PrUnknown | PrWin | PrLoss | PrDraw
      mutable Stm: int } // side-to-move at this node (debug/aux)

[<Sealed>]
type MctsTree(nodeCap: int, edgeCap: int) =
    let mutable nodes: MctsNode[] = Array.zeroCreate nodeCap
    let mutable edges: MctsEdge[] = Array.zeroCreate edgeCap
    let mutable nodeCount = 0
    let mutable edgeCount = 0
    member _.NodeCount = nodeCount
    member _.EdgeCount = edgeCount
    // Read-only inspection for tests; valid only until the next AllocNode/AllocEdges (realloc).
    member _.Nodes = nodes
    member _.Edges = edges

    new() = MctsTree(4096, 16384)

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member _.AllocNode(stm: int) : int =
        if nodeCount >= nodes.Length then
            let bigger = Array.zeroCreate (nodes.Length * 2)
            System.Array.Copy(nodes, bigger, nodeCount)
            nodes <- bigger

        let i = nodeCount
        nodeCount <- nodeCount + 1
        nodes.[i].N <- 0
        nodes.[i].W <- 0.0f
        nodes.[i].FirstEdge <- -1
        nodes.[i].NumEdges <- 0
        nodes.[i].Proven <- PrUnknown
        nodes.[i].Stm <- stm
        i

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member _.AllocEdges(count: int) : int =
        if edgeCount + count > edges.Length then
            let mutable cap = edges.Length * 2

            while cap < edgeCount + count do
                cap <- cap * 2

            let bigger = Array.zeroCreate cap
            System.Array.Copy(edges, bigger, edgeCount)
            edges <- bigger

        let b = edgeCount
        edgeCount <- edgeCount + count
        b

    // Reads (copy a field out of the current array)
    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member _.NodeN(i) = nodes.[i].N

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member _.NodeW(i) = nodes.[i].W

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member _.NodeProven(i) = nodes.[i].Proven

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member _.NodeFirstEdge(i) = nodes.[i].FirstEdge

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member _.NodeNumEdges(i) = nodes.[i].NumEdges

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member _.NodeStm(i) = nodes.[i].Stm

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member _.EdgeMove(i) = edges.[i].Move

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member _.EdgePrior(i) = edges.[i].Prior

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member _.EdgeChild(i) = edges.[i].Child

    // Mutations (always operate on the CURRENT arrays — no stale-ref foot-gun)
    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member _.SetProven(i, p) = nodes.[i].Proven <- p

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member _.SetNodeEdges(i, fe, ne) =
        nodes.[i].FirstEdge <- fe
        nodes.[i].NumEdges <- ne

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member _.AddVisit(i, q) =
        let mutable n = &nodes.[i]
        n.N <- n.N + 1
        n.W <- n.W + q

    // Virtual loss for batched gather: a virtual child-WIN (W += vl) is a LOSS for the parent, so a node
    // pulled into the current batch looks more-visited and worse, steering the next gather elsewhere.
    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member _.AddVirtualLoss(i, vl: int) =
        let mutable n = &nodes.[i]
        n.N <- n.N + vl
        n.W <- n.W + float32 vl

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member _.UndoVirtualLoss(i, vl: int) =
        let mutable n = &nodes.[i]
        n.N <- n.N - vl
        n.W <- n.W - float32 vl

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member _.SetEdge(i, m, prior) =
        edges.[i].Move <- m
        edges.[i].Prior <- prior
        edges.[i].Child <- -1

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member _.SetEdgePrior(i, prior) = edges.[i].Prior <- prior

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member _.SetEdgeChild(i, c) = edges.[i].Child <- c

// ---------------------------------------------------------------------------
// Value scaling (single source of truth). cp is side-to-move-relative (negamax convention).
// ---------------------------------------------------------------------------
[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let cpToWinProb (cp: int) (k: float32) : float32 =
    if cp >= MATE_IN_MAX_PLY then 1.0f
    elif cp <= -MATE_IN_MAX_PLY then 0.0f
    else 1.0f / (1.0f + MathF.Exp(-(float32 cp) / k))

[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let winProbToCp (q: float32) (k: float32) : int =
    let qc = min 0.999f (max 0.001f q)
    int (k * MathF.Log(qc / (1.0f - qc)))

// ---------------------------------------------------------------------------
// PUCT
// ---------------------------------------------------------------------------
[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let puctScoreOf
    (childN: int)
    (childW: float32)
    (childExists: bool)
    (prior: float32)
    (sqrtParentN: float32)
    (cpuct: float32)
    (fpu: float32)
    : float32 =
    // exploitation is from the PARENT's POV: a child's Q is its own (opponent) POV, so flip with 1 - Q.
    let exploit =
        if (not childExists) || childN = 0 then
            fpu
        else
            1.0f - (childW / float32 childN)

    let np = if childExists then childN else 0
    exploit + cpuct * prior * sqrtParentN / (1.0f + float32 np)

[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let private provenQ (tree: MctsTree) (idx: int) : float32 =
    match tree.NodeProven idx with
    | PrWin -> 1.0f
    | PrLoss -> 0.0f
    | _ -> 0.5f // PrDraw

/// Pick the edge maximizing PUCT, skipping children proven won for the child's mover (= lost for us).
/// Strict `>` keeps the lowest edge index on ties => deterministic. Returns -1 if no edge is selectable.
let selectEdge (tree: MctsTree) (nodeIdx: int) (cpuct: float32) : int =
    let fe = tree.NodeFirstEdge nodeIdx
    let ne = tree.NodeNumEdges nodeIdx
    let parentN = tree.NodeN nodeIdx
    let sqrtParentN = MathF.Sqrt(float32 (max 1 parentN))
    let mutable best = -1
    let mutable bestScore = System.Single.NegativeInfinity

    for k in 0 .. ne - 1 do
        let ei = fe + k
        let ci = tree.EdgeChild ei
        let isWinForChild = ci >= 0 && tree.NodeProven ci = PrWin

        if not isWinForChild then
            let cn = if ci >= 0 then tree.NodeN ci else 0
            let cw = if ci >= 0 then tree.NodeW ci else 0.0f
            let s = puctScoreOf cn cw (ci >= 0) (tree.EdgePrior ei) sqrtParentN cpuct Fpu

            if s > bestScore then
                bestScore <- s
                best <- ei

    best

// ---------------------------------------------------------------------------
// MCTS-Solver. Tags are from the node's own side-to-move POV; a Child<0 edge counts as PrUnknown.
// ---------------------------------------------------------------------------
let solverUpdate (tree: MctsTree) (p: int) : unit =
    if tree.NodeProven p = PrUnknown then
        let ne = tree.NodeNumEdges p

        if ne > 0 then // never vacuously prove a zero-child node
            let fe = tree.NodeFirstEdge p
            let mutable anyLoss = false
            let mutable allProven = true
            let mutable anyDraw = false
            let mutable allWin = true

            for k in 0 .. ne - 1 do
                let ci = tree.EdgeChild (fe + k)
                let cp = if ci >= 0 then tree.NodeProven ci else PrUnknown

                if cp = PrLoss then anyLoss <- true
                elif cp = PrWin then ()
                elif cp = PrDraw then
                    anyDraw <- true
                    allWin <- false
                else
                    allProven <- false
                    allWin <- false

            if anyLoss then tree.SetProven(p, PrWin) // a child is lost for them => we win
            elif allProven && allWin then tree.SetProven(p, PrLoss) // every reply wins for them => we lose
            elif allProven && anyDraw then tree.SetProven(p, PrDraw) // best we force is a draw

// ---------------------------------------------------------------------------
// Priors: softmax over a move-ordering score (replicated from MovePick's private scorers, over the
// LEGAL set — the picker emits pseudo-legal moves). No cont-hist context at an MCTS node => main hist.
// ---------------------------------------------------------------------------
[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let private scoreMove (pos: Position) (tables: Tables) (stm: Color) (m: Move) : int =
    let toS = toSq m
    let captured = pos.PieceOn toS

    if captured <> NoPiece || isEnPassant m then
        let capturedPT = if isEnPassant m then Pawn else pieceType captured
        let pc = pos.PieceOn(fromSq m)
        7 * pieceValueOf capturedPT + tables.CaptureHistory pc toS capturedPT
    else
        tables.MainHistory stm (fromTo m)

/// Expand a (pre-allocated) node at the current w.Pos: terminal/draw check, else legal moves + priors.
/// allowDraw=false at the search root (the engine never treats the root as an immediate draw).
let private expand (tree: MctsTree) (w: Worker) (nodeIdx: int) (allowDraw: bool) : unit =
    let pos = w.Pos

    if allowDraw && isImmediateDraw pos then
        tree.SetProven(nodeIdx, PrDraw)
    else
        let moves = w.MoveBuf
        let span = System.Span<Move>(moves, 0, MaxMoves)
        let cnt = generateLegal pos span

        if cnt = 0 then
            tree.SetProven(nodeIdx, (if pos.InCheck then PrLoss else PrDraw))
        else
            let tables = w.Tables
            let stm = pos.SideToMove

            if w.Control.Config.UseLc0 && w.Control.Lc0Net.IsSome then
                // Lc0 CNN priors (true hybrid: Lc0 policy + SF-NNUE alpha-beta leaf). v1 ignores the Lc0
                // value head; the leaf eval stays the negamax+SF path below. Runs once per node-expansion.
                let net = w.Control.Lc0Net.Value
                Lc0Net.lc0PriorsInto SfAccumulator.UseAvx2 net pos moves cnt w.Lc0InBuf w.Lc0Scratch w.Lc0Priors
                |> ignore

                let priors = w.Lc0Priors
                let edgeBase = tree.AllocEdges cnt

                for i in 0 .. cnt - 1 do
                    tree.SetEdge(edgeBase + i, moves.[i], priors.[i])

                tree.SetNodeEdges(nodeIdx, edgeBase, cnt)
            else
                // History fallback: softmax over move-ordering scores.
                let scores = w.ScoreBuf
                let mutable maxScore = System.Int32.MinValue

                for i in 0 .. cnt - 1 do
                    let s = scoreMove pos tables stm moves.[i]
                    scores.[i] <- s

                    if s > maxScore then
                        maxScore <- s

                let edgeBase = tree.AllocEdges cnt
                let mutable sum = 0.0f

                for i in 0 .. cnt - 1 do
                    let wgt = MathF.Exp(float32 (scores.[i] - maxScore) / MctsPriorTemp)
                    tree.SetEdge(edgeBase + i, moves.[i], wgt) // store the raw weight; normalize below
                    sum <- sum + wgt

                let inv = if sum > 0.0f then 1.0f / sum else 1.0f

                for i in 0 .. cnt - 1 do
                    tree.SetEdgePrior(edgeBase + i, tree.EdgePrior(edgeBase + i) * inv)

                tree.SetNodeEdges(nodeIdx, edgeBase, cnt)

/// Fixed-depth full-window negamax leaf, on w.Pos at the leaf. Stack prefix cleared for fresh-worker
/// parity (matches searchToDepthNet); kills stale-ExcludedMove singular bugs. Mate scores => proven.
let private evalLeaf (tree: MctsTree) (w: Worker) (leafDepth: int) (k: float32) (leafIdx: int) : float32 =
    System.Array.Clear(w.Stack, 0, StackOffset + 2)
    w.NmpMinPly <- 0
    let cp = negamax w w.Pos (-INF) INF leafDepth 0 false false

    if cp >= MATE_IN_MAX_PLY then tree.SetProven(leafIdx, PrWin)
    elif cp <= -MATE_IN_MAX_PLY then tree.SetProven(leafIdx, PrLoss)

    cpToWinProb cp k

// ---------------------------------------------------------------------------
// One MCTS iteration: SELECT (Make down) -> EXPAND/EVAL leaf -> BACKUP (flip up) -> Unmake back.
// Returns struct(tree depth reached, backedUp). backedUp is false when a stop aborted the playout
// mid-leaf, so the caller can avoid counting an iteration that committed no value (root.N stays in
// lockstep with the reported iteration count). pathNodes/pathEdges are caller-owned scratch (reused).
// ---------------------------------------------------------------------------
let private runIteration
    (tree: MctsTree)
    (w: Worker)
    (cpuct: float32)
    (leafDepth: int)
    (k: float32)
    (maxTreeDepth: int)
    (pathNodes: int[])
    (pathEdges: int[])
    (rootIdx: int)
    : struct (int * bool) =
    let pos = w.Pos
    let mutable node = rootIdx
    let mutable pathLen = 0
    let mutable leaf = -1
    let mutable q = 0.0f
    let mutable looping = true

    while looping do
        pathNodes.[pathLen] <- node

        if tree.NodeProven node <> PrUnknown then
            leaf <- node
            q <- provenQ tree node
            looping <- false
        elif pathLen >= maxTreeDepth then
            leaf <- node
            q <- evalLeaf tree w leafDepth k node
            looping <- false
        else
            let e = selectEdge tree node cpuct

            if e < 0 then
                leaf <- node // every child is a proven win for them => we are lost
                q <- 0.0f
                looping <- false
            else
                pos.Make(tree.EdgeMove e)
                pathEdges.[pathLen] <- e
                pathLen <- pathLen + 1
                let ci = tree.EdgeChild e

                if ci < 0 then
                    let c = tree.AllocNode(int pos.SideToMove)
                    tree.SetEdgeChild(e, c)
                    expand tree w c true

                    q <-
                        if tree.NodeProven c <> PrUnknown then
                            provenQ tree c
                        else
                            evalLeaf tree w leafDepth k c

                    leaf <- c
                    looping <- false
                else
                    node <- ci

    // BACKUP (skip if a stop hit mid-leaf — a partial value). leaf gets q (own POV); ancestors flip.
    // Read the stop flag once so the backup decision and the reported `backedUp` are consistent.
    let backedUp = not w.Control.Stopped

    if backedUp then
        let mutable qq = q
        tree.AddVisit(leaf, qq)
        solverUpdate tree leaf

        for d in pathLen - 1 .. -1 .. 0 do
            qq <- 1.0f - qq
            tree.AddVisit(pathNodes.[d], qq)

            if tree.NodeProven pathNodes.[d] = PrUnknown then
                solverUpdate tree pathNodes.[d]

    // ALWAYS unmake the whole path, stop or not, so w.Pos is restored to the root.
    for d in pathLen - 1 .. -1 .. 0 do
        pos.Unmake(tree.EdgeMove(pathEdges.[d]))

    struct (pathLen, backedUp)

// ---------------------------------------------------------------------------
// Batched Lc0 evaluation (per-worker, virtual-loss gather). Gather B leaves, run ONE Lc0 forwardBatch for
// all B prior-sets (one weight stream => RAM bandwidth amortized B-fold), then per leaf set priors + negamax
// value + backup (reverting virtual loss). Used only when UseLc0 && MctsBatchSize>1 (history/single paths
// are untouched). The leaf VALUE is still the SF negamax (the Lc0 value head is ignored, as elsewhere).
// ---------------------------------------------------------------------------
[<Sealed>]
type private MctsBatch(net: Lc0Proto.Lc0Net, maxBatch: int, maxTreeDepth: int) =
    let planes = Lc0Encoder.Planes
    let stride = maxTreeDepth + 1
    member val Scratch = Lc0Net.Lc0BatchScratch(net, maxBatch)
    member val LeafInputs: float32[] = Array.zeroCreate (maxBatch * planes * 64) // per-leaf single-board encoding
    member val BatchInput: float32[] = Array.zeroCreate (maxBatch * planes * 64) // packed channel-major batched layout
    member val Logits: float32[] = Array.zeroCreate (maxBatch * Lc0PolicyMap.NumPolicy)
    member val Values: float32[] = Array.zeroCreate maxBatch
    member val Priors: float32[] = Array.zeroCreate MaxMoves
    member val LeafNodes: int[] = Array.zeroCreate maxBatch
    member val PathLens: int[] = Array.zeroCreate maxBatch
    member val Paths: int[] = Array.zeroCreate (maxBatch * stride) // leaf i's edges at i*stride
    member _.Stride = stride

[<Literal>]
let private VirtualLoss = 1

/// One batch: gather up to maxBatch leaves (virtual loss; terminals/proven backed up inline; stop on collision
/// or stop), ONE forwardBatch for all priors, then per leaf set priors + negamax value + backup (revert VL).
/// Returns struct(maxDepthReached, completedBackups); completedBackups drives the iteration/budget counters.
let private runBatch
    (tree: MctsTree)
    (w: Worker)
    (cpuct: float32)
    (leafDepth: int)
    (k: float32)
    (maxTreeDepth: int)
    (rootIdx: int)
    (batch: MctsBatch)
    (maxBatch: int)
    : struct (int * int) =
    let pos = w.Pos
    let net = w.Control.Lc0Net.Value
    let planes = Lc0Encoder.Planes
    let stride = batch.Stride
    let mutable nLeaves = 0
    let mutable completed = 0
    let mutable maxDepthReached = 0

    // ---- GATHER (at most maxBatch attempts: bounds the loop even if every selection hits a terminal) ----
    let mutable attempts = 0
    let mutable gathering = true

    while gathering && attempts < maxBatch && not w.Control.Stopped do
        attempts <- attempts + 1
        let pathBase = nLeaves * stride
        let mutable node = rootIdx
        let mutable pathLen = 0
        let mutable outcome = 0 // 1 = batch leaf, 2 = terminal (inline backup), 3 = collision
        let mutable termQ = 0.0f
        let mutable leafNode = -1
        let mutable walking = true

        while walking do
            if tree.NodeProven node <> PrUnknown then
                outcome <- 2
                termQ <- provenQ tree node
                leafNode <- node
                walking <- false
            elif tree.NodeNumEdges node = 0 then
                outcome <- 3 // unexpanded node reached by selection = a pending batch leaf => collision
                walking <- false
            elif pathLen >= maxTreeDepth then
                outcome <- 2
                termQ <- evalLeaf tree w leafDepth k node
                leafNode <- node
                walking <- false
            else
                let e = selectEdge tree node cpuct

                if e < 0 then
                    outcome <- 2 // every child is a proven win for them => we are lost
                    termQ <- 0.0f
                    leafNode <- node
                    walking <- false
                else
                    pos.Make(tree.EdgeMove e)
                    batch.Paths.[pathBase + pathLen] <- e
                    pathLen <- pathLen + 1
                    let ci = tree.EdgeChild e

                    if ci < 0 then
                        let c = tree.AllocNode(int pos.SideToMove)
                        tree.SetEdgeChild(e, c)
                        let moves = w.MoveBuf
                        let span = System.Span<Move>(moves, 0, MaxMoves)
                        let cnt = generateLegal pos span

                        if cnt = 0 then
                            tree.SetProven(c, (if pos.InCheck then PrLoss else PrDraw))
                            outcome <- 2
                            termQ <- provenQ tree c
                            leafNode <- c
                        elif isImmediateDraw pos then
                            tree.SetProven(c, PrDraw)
                            outcome <- 2
                            termQ <- provenQ tree c
                            leafNode <- c
                        else
                            // batch leaf: encode the position now (board is at the leaf), priors come later.
                            Lc0Encoder.encodeInto pos w.Lc0InBuf
                            Array.blit w.Lc0InBuf 0 batch.LeafInputs (nLeaves * planes * 64) (planes * 64)
                            outcome <- 1
                            leafNode <- c

                        walking <- false
                    else
                        node <- ci

        if pathLen > maxDepthReached then
            maxDepthReached <- pathLen

        if outcome = 1 then
            // batch leaf: record path/leaf, apply virtual loss to every stepped child (incl. the leaf), unmake.
            batch.LeafNodes.[nLeaves] <- leafNode
            batch.PathLens.[nLeaves] <- pathLen

            for d in 0 .. pathLen - 1 do
                let ci = tree.EdgeChild(batch.Paths.[pathBase + d])
                if ci >= 0 then tree.AddVirtualLoss(ci, VirtualLoss)

            nLeaves <- nLeaves + 1

            for d in pathLen - 1 .. -1 .. 0 do
                pos.Unmake(tree.EdgeMove(batch.Paths.[pathBase + d]))
        elif outcome = 2 then
            // terminal/proven: real backup inline (no virtual loss applied on this path), unmake.
            tree.AddVisit(leafNode, termQ)
            solverUpdate tree leafNode
            let mutable qq = termQ

            for d in pathLen - 1 .. -1 .. 0 do
                qq <- 1.0f - qq
                let nodeD = if d = 0 then rootIdx else tree.EdgeChild(batch.Paths.[pathBase + d - 1])
                tree.AddVisit(nodeD, qq)

                if tree.NodeProven nodeD = PrUnknown then
                    solverUpdate tree nodeD

            completed <- completed + 1

            for d in pathLen - 1 .. -1 .. 0 do
                pos.Unmake(tree.EdgeMove(batch.Paths.[pathBase + d]))
        else
            // collision: discard this partial gather and stop (process what we have).
            for d in pathLen - 1 .. -1 .. 0 do
                pos.Unmake(tree.EdgeMove(batch.Paths.[pathBase + d]))

            gathering <- false

    // ---- EVALUATE: one batched forward over all gathered leaves ----
    if nLeaves > 0 then
        // pack per-leaf single-board encodings into the channel-major batched layout (ch, then board).
        for ch in 0 .. planes - 1 do
            for b in 0 .. nLeaves - 1 do
                Array.blit batch.LeafInputs (b * planes * 64 + ch * 64) batch.BatchInput (ch * nLeaves * 64 + b * 64) 64

        Lc0Net.forwardBatch SfAccumulator.UseAvx2 net batch.Scratch nLeaves batch.BatchInput batch.Logits batch.Values

        // ---- PROCESS each leaf: re-walk, set priors, negamax value, revert VL, real backup, unmake ----
        for i in 0 .. nLeaves - 1 do
            let pb = i * stride
            let plen = batch.PathLens.[i]
            let leaf = batch.LeafNodes.[i]

            if w.Control.Stopped then
                // bail: still revert this leaf's virtual loss (tree-only) so the tree isn't left distorted.
                for d in 0 .. plen - 1 do
                    let ci = tree.EdgeChild(batch.Paths.[pb + d])
                    if ci >= 0 then tree.UndoVirtualLoss(ci, VirtualLoss)
            else
                for d in 0 .. plen - 1 do
                    pos.Make(tree.EdgeMove(batch.Paths.[pb + d]))

                // priors from the batched logits (softmax over the leaf's legal moves).
                let moves = w.MoveBuf
                let span = System.Span<Move>(moves, 0, MaxMoves)
                let cnt = generateLegal pos span
                Lc0Net.lc0PriorsFromLogits batch.Logits (i * Lc0PolicyMap.NumPolicy) pos moves cnt batch.Priors
                let edgeBase = tree.AllocEdges cnt

                for j in 0 .. cnt - 1 do
                    tree.SetEdge(edgeBase + j, moves.[j], batch.Priors.[j])

                tree.SetNodeEdges(leaf, edgeBase, cnt)

                // value = the SF negamax leaf (sets Proven on a mate, like the single path).
                let q = evalLeaf tree w leafDepth k leaf

                for d in 0 .. plen - 1 do
                    let ci = tree.EdgeChild(batch.Paths.[pb + d])
                    if ci >= 0 then tree.UndoVirtualLoss(ci, VirtualLoss)

                tree.AddVisit(leaf, q)
                solverUpdate tree leaf
                let mutable qq = q

                for d in plen - 1 .. -1 .. 0 do
                    qq <- 1.0f - qq
                    let nodeD = if d = 0 then rootIdx else tree.EdgeChild(batch.Paths.[pb + d - 1])
                    tree.AddVisit(nodeD, qq)

                    if tree.NodeProven nodeD = PrUnknown then
                        solverUpdate tree nodeD

                completed <- completed + 1

                for d in plen - 1 .. -1 .. 0 do
                    pos.Unmake(tree.EdgeMove(batch.Paths.[pb + d]))

    struct (maxDepthReached, completed)

// ---------------------------------------------------------------------------
// Root selection / reporting
// ---------------------------------------------------------------------------
/// Best root move: prefer a proven-win-for-us child (PrLoss child); avoid proven-loss (PrWin child)
/// unless all moves lose; otherwise max visit count, tie-break by parent-POV Q. Returns (move, edge).
let bestRootMove (tree: MctsTree) (rootIdx: int) : Move * int =
    let fe = tree.NodeFirstEdge rootIdx
    let ne = tree.NodeNumEdges rootIdx

    if ne <= 0 then
        (MoveNone, -1)
    else
        let mutable bestEi = -1
        let mutable bestCat = -1
        let mutable bestN = System.Int32.MinValue
        let mutable bestQ = System.Single.NegativeInfinity

        for k in 0 .. ne - 1 do
            let ei = fe + k
            let ci = tree.EdgeChild ei
            let proven = if ci >= 0 then tree.NodeProven ci else PrUnknown
            // category: PrLoss child (win for us)=2 best; unknown/draw=1; PrWin child (loss for us)=0
            let cat =
                if proven = PrLoss then 2
                elif proven = PrWin then 0
                else 1

            let n = if ci >= 0 then tree.NodeN ci else 0

            let qv =
                if ci >= 0 && tree.NodeN ci > 0 then
                    1.0f - tree.NodeW ci / float32 (tree.NodeN ci)
                else
                    0.0f

            if cat > bestCat || (cat = bestCat && (n > bestN || (n = bestN && qv > bestQ))) then
                bestCat <- cat
                bestN <- n
                bestQ <- qv
                bestEi <- ei

        (tree.EdgeMove bestEi, bestEi)

let private rootScoreCp (tree: MctsTree) (rootIdx: int) (bestEi: int) (k: float32) : int =
    match tree.NodeProven rootIdx with
    | PrWin -> MATE - 1
    | PrLoss -> -(MATE - 1)
    | PrDraw -> 0
    | _ ->
        if bestEi < 0 then
            0
        else
            let ci = tree.EdgeChild bestEi

            if ci >= 0 && tree.NodeProven ci = PrLoss then MATE - 1 // proven win for us
            elif ci >= 0 && tree.NodeProven ci = PrWin then -(MATE - 1)
            elif ci < 0 || tree.NodeN ci = 0 then 0
            else winProbToCp (1.0f - tree.NodeW ci / float32 (tree.NodeN ci)) k

// ---------------------------------------------------------------------------
// Root merge (root-parallel: combine N independent trees into one decision)
// ---------------------------------------------------------------------------
/// One root move's summed statistics across all workers' trees. Tags are child-POV, like MctsNode.Proven.
[<Struct>]
type private MergedMove =
    { mutable Move: Move
      mutable N: int // summed child visit counts across trees
      mutable W: float32 // summed child W (child POV) across trees
      mutable Proven: int } // unioned proven tag (see combineProven)

/// Union proven tags across trees. Order-independent / idempotent. A single worker's αβ-leaf mate proof is
/// exact, so any PrLoss child (proven win for us) wins; any PrWin child (proven loss for us) is next; any
/// PrDraw is advisory (same category as unknown for selection). Mirrors bestRootMove's per-tree categories.
let private combineProven (acc: int) (child: int) : int =
    if acc = PrLoss || child = PrLoss then PrLoss
    elif acc = PrWin || child = PrWin then PrWin
    elif acc = PrDraw || child = PrDraw then PrDraw
    else PrUnknown

/// Index-aligned merge of N root trees. Every tree expands the same root with the same deterministic
/// generateLegal + scoreMove, so root edges share identical move order — sum child N/W per edge index.
/// R1: a tree whose root layout differs (e.g. an unexpanded `safe < 1` root, or any future move-ordering
/// change) is SKIPPED whole rather than misaligned. Uses tree 0 as the canonical layout.
let private mergeRoots (trees: MctsTree[]) (roots: int[]) : MergedMove[] =
    let t0 = trees.[0]
    let r0 = roots.[0]
    let fe0 = t0.NodeFirstEdge r0
    let ne0 = t0.NodeNumEdges r0

    if ne0 <= 0 then
        [||]
    else
        let acc =
            Array.init ne0 (fun k ->
                { Move = t0.EdgeMove(fe0 + k)
                  N = 0
                  W = 0.0f
                  Proven = PrUnknown })

        for ti in 0 .. trees.Length - 1 do
            let t = trees.[ti]

            if not (obj.ReferenceEquals(t, null)) then
                let r = roots.[ti]
                let fe = t.NodeFirstEdge r
                let ne = t.NodeNumEdges r
                // Verify this tree's root layout matches tree 0 before adding (R1).
                let mutable aligned = ne = ne0
                let mutable kk = 0

                while aligned && kk < ne0 do
                    if t.EdgeMove(fe + kk) <> acc.[kk].Move then
                        aligned <- false

                    kk <- kk + 1

                if aligned then
                    for k in 0 .. ne0 - 1 do
                        let ci = t.EdgeChild(fe + k)

                        if ci >= 0 then
                            acc.[k].N <- acc.[k].N + t.NodeN ci
                            acc.[k].W <- acc.[k].W + t.NodeW ci
                            acc.[k].Proven <- combineProven acc.[k].Proven (t.NodeProven ci)

        acc

/// Pick the final move from merged stats with bestRootMove's tie-break extended over summed values:
/// (1) proven category (win-for-us > unknown/draw > loss-for-us), (2) summed visits N, (3) summed-stat
/// parent-POV Q. Strict `>` over an ascending scan keeps the lowest edge index on exact ties (deterministic).
let private bestMergedMove (acc: MergedMove[]) : Move * int =
    if acc.Length = 0 then
        (MoveNone, -1)
    else
        let mutable best = -1
        let mutable bestCat = -1
        let mutable bestN = System.Int32.MinValue
        let mutable bestQ = System.Single.NegativeInfinity

        for i in 0 .. acc.Length - 1 do
            let a = acc.[i]

            let cat =
                if a.Proven = PrLoss then 2
                elif a.Proven = PrWin then 0
                else 1

            let qv = if a.N > 0 then 1.0f - a.W / float32 a.N else 0.0f

            if cat > bestCat || (cat = bestCat && (a.N > bestN || (a.N = bestN && qv > bestQ))) then
                bestCat <- cat
                bestN <- a.N
                bestQ <- qv
                best <- i

        (acc.[best].Move, best)

/// Score (cp) for the chosen merged move, mirroring rootScoreCp's child branch so the reported score matches
/// the reported move: proven win/loss -> mate, proven draw / no visits -> 0, else sigmoid of parent-POV Q.
let private mergedScoreCp (a: MergedMove) (k: float32) : int =
    match a.Proven with
    | PrLoss -> MATE - 1
    | PrWin -> -(MATE - 1)
    | PrDraw -> 0
    | _ -> if a.N > 0 then winProbToCp (1.0f - a.W / float32 a.N) k else 0

let private scoreStr (score: int) : string =
    if score >= MATE_IN_MAX_PLY then "mate " + string ((MATE - score + 1) / 2)
    elif score <= -MATE_IN_MAX_PLY then "mate " + string (-((MATE + score + 1) / 2))
    else "cp " + string score

let private writeLine (s: string) = System.Console.Out.WriteLine(s)

/// Greedy max-N descent from the root (visited children only); appends " <uci>" tokens to sb.
let private buildPv (tree: MctsTree) (rootIdx: int) (sb: System.Text.StringBuilder) =
    let mutable node = rootIdx
    let mutable cont = true
    let mutable depth = 0

    while cont && depth < 128 do
        let fe = tree.NodeFirstEdge node
        let ne = tree.NodeNumEdges node

        if ne <= 0 then
            cont <- false
        else
            let mutable bestEi = -1
            let mutable bestN = 0

            for k in 0 .. ne - 1 do
                let ei = fe + k
                let ci = tree.EdgeChild ei
                let n = if ci >= 0 then tree.NodeN ci else 0

                if n > bestN then
                    bestN <- n
                    bestEi <- ei

            if bestEi < 0 then
                cont <- false
            else
                sb.Append(' ').Append(toUci (tree.EdgeMove bestEi)) |> ignore
                let ci = tree.EdgeChild bestEi
                if ci < 0 then cont <- false else node <- ci
                depth <- depth + 1

/// Merged PV for the chosen root move across all workers' trees: head = the merged best move; tail
/// descends the worker subtree that visited that move most (greedy max-N, reusing buildPv on the child).
/// Reads live trees (safe: root edges are immutable after expand, child indices only grow, int/float
/// reads are atomic — an approximate-but-crash-free PV is fine for an info line).
let private buildMergedPv (trees: MctsTree[]) (roots: int[]) (mergedMove: Move) (sb: System.Text.StringBuilder) =
    sb.Append(' ').Append(toUci mergedMove) |> ignore
    let mutable bestTree = -1
    let mutable bestChild = -1
    let mutable bestN = -1

    for ti in 0 .. trees.Length - 1 do
        let t = trees.[ti]

        if not (obj.ReferenceEquals(t, null)) then
            let r = roots.[ti]
            let fe = t.NodeFirstEdge r
            let ne = t.NodeNumEdges r
            let mutable k = 0

            while k < ne do
                if t.EdgeMove(fe + k) = mergedMove then
                    let ci = t.EdgeChild(fe + k)
                    let n = if ci >= 0 then t.NodeN ci else 0

                    if n > bestN then
                        bestN <- n
                        bestChild <- ci
                        bestTree <- ti

                    k <- ne // found this tree's edge for the move; stop scanning it
                else
                    k <- k + 1

    if bestTree >= 0 && bestChild >= 0 then
        buildPv trees.[bestTree] bestChild sb

/// Emit the two MCTS info lines for an explicit merged decision (move + score), so the periodic and final
/// reports share one code path and the PV head always equals the decided move.
///   leaf  = negamax leaf nodes (NodeSum) — comparable to alpha-beta nps and consistent with `go nodes N`.
///   iters = MCTS root iterations / playouts (IterSum) — the Lc0-style tree-growth metric (secondary line).
let private emitMctsInfo
    (control: SearchControl)
    (depth: int)
    (score: int)
    (mergedMove: Move)
    (trees: MctsTree[])
    (roots: int[])
    =
    let ms = control.ElapsedMs
    let leaf = control.NodeSum()
    let leafNps = if ms > 0L then leaf * 1000L / ms else leaf
    let iters = control.IterSum()
    let iterNps = if ms > 0L then iters * 1000L / ms else iters
    let sb = System.Text.StringBuilder(192)

    sb
        .Append("info depth ")
        .Append(depth)
        .Append(" score ")
        .Append(scoreStr score)
        .Append(" nodes ")
        .Append(leaf)
        .Append(" nps ")
        .Append(leafNps)
        .Append(" time ")
        .Append(ms)
        .Append(" pv")
    |> ignore

    if mergedMove <> MoveNone then
        buildMergedPv trees roots mergedMove sb

    writeLine (sb.ToString())

    // Secondary line: the MCTS-native root/iteration throughput (leaf repeated for an at-a-glance pairing).
    let sb2 = System.Text.StringBuilder(96)

    sb2
        .Append("info string root ")
        .Append(iters)
        .Append(" iters ")
        .Append(iterNps)
        .Append(" it/s | leaf ")
        .Append(leaf)
        .Append(" nodes ")
        .Append(leafNps)
        .Append(" nps")
    |> ignore

    writeLine (sb2.ToString())

/// Periodic driver report: merge the live per-worker trees into one decision and emit it. The non-driver
/// trees are still being mutated; mergeRoots/buildMergedPv tolerate that (see buildMergedPv's note).
let private reportMctsMerged (control: SearchControl) (trees: MctsTree[]) (roots: int[]) (k: float32) (depth: int) =
    let acc = mergeRoots trees roots
    let (mv, bestIdx) = bestMergedMove acc
    let score = if bestIdx >= 0 then mergedScoreCp acc.[bestIdx] k else 0
    emitMctsInfo control depth score mv trees roots

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------
/// One worker's full MCTS life-cycle on its OWN tree (the per-thread body, unchanged from the v1
/// single-thread loop). `checkSoft`/`stopOnSolved`/`report` are driver-only knobs: the parallel
/// orchestrator passes them true for worker 0 and false for the rest, so non-driver workers exit only on
/// the shared stop flag (or the shared global iteration budget) and never report. The test entries call this
/// directly with a single worker, preserving the deterministic exactly-`maxIters` path.
let private runWorker
    (w: Worker)
    (control: SearchControl)
    (maxIters: int)
    (checkSoft: bool)
    (stopOnSolved: bool)
    (report: bool)
    (trees: MctsTree[])
    (roots: int[])
    (globalIters: int64[])
    : struct (MctsTree * int * int) =
    let cfg = control.Config
    let cpuct = float32 cfg.MctsCpuct / 100.0f
    let kf = float32 (max 1 cfg.MctsK)
    let leafDepth = max 0 cfg.MctsLeafDepth
    // Overflow guard: the leaf negamax runs on w.Pos already R+D deep; keep R+D+MaxSearchPly < MaxPly.
    let safe = MaxPly - control.RootMoves.Length - MaxSearchPly - 1
    // GLOBAL iteration budget: maxIters is the TOTAL across all workers, not per-worker (Threads splits the
    // depth budget instead of each worker redoing all of it). Pre-size this worker's arena for its ~share to
    // avoid mid-search reallocations; the arena still grows if the share runs over. (~40 edges/node.)
    let threads = max 1 cfg.Threads
    let perWorker = if maxIters > 0 then maxIters / threads + 256 else 0

    let tree =
        if perWorker > 0 then
            MctsTree(perWorker + 1, perWorker * 40 + 1024)
        else
            MctsTree()

    let rootIdx = tree.AllocNode(int w.Pos.SideToMove)
    // Publish this worker's tree/root so the driver's periodic report can merge in-progress stats across all
    // workers (runParallel's final merge reads the same slots). roots first (plain int), then the tree via
    // Volatile (release) so any reader that observes the tree also observes its root index.
    roots.[w.Id] <- rootIdx
    System.Threading.Volatile.Write(&trees.[w.Id], tree)

    if safe < 1 then
        struct (tree, rootIdx, 0) // position already at/over the base engine's envelope: caller falls back
    else
        let maxTreeDepth = min 256 safe
        let pathNodes = Array.zeroCreate (maxTreeDepth + 1)
        let pathEdges = Array.zeroCreate (maxTreeDepth + 1)
        expand tree w rootIdx false // root expanded once; allowDraw=false
        // Batched Lc0 eval: gather B leaves through ONE forwardBatch when Lc0 is active (amortizes weight RAM
        // bandwidth). Off (single-leaf runIteration) for history priors and batch size 1 — all current paths.
        let useBatch = cfg.UseLc0 && cfg.MctsBatchSize > 1 && control.Lc0Net.IsSome

        let batch =
            if useBatch then
                MctsBatch(control.Lc0Net.Value, cfg.MctsBatchSize, maxTreeDepth)
            else
                Unchecked.defaultof<MctsBatch>

        let mutable maxDepthReached = 0
        let mutable lastReportMs = 0L
        let mutable looping = true

        while looping do
            // Driver only: turn a hard-time / node-limit overrun into the shared stop flag. The leaf
            // negamax also drives CheckTime (w.IsMain, every 2047 nodes), but this keeps the MCTS loop
            // itself responsive when leaves are shallow/cheap. Non-driver workers never write the flag.
            if checkSoft then
                control.CheckTime(control.NodeSum())

            if maxIters > 0 && Volatile.Read(&globalIters.[0]) >= int64 maxIters then
                looping <- false // shared global budget exhausted across all workers
            elif control.Stopped then
                looping <- false
            elif checkSoft && control.SoftTimeUp then
                looping <- false
            elif stopOnSolved && tree.NodeProven rootIdx <> PrUnknown then
                looping <- false
            elif useBatch then
                let struct (d, completed) =
                    runBatch tree w cpuct leafDepth kf maxTreeDepth rootIdx batch cfg.MctsBatchSize

                if d > maxDepthReached then
                    maxDepthReached <- d

                if completed > 0 then
                    for _ in 1 .. completed do
                        w.IncIters() // each backed-up leaf is one completed iteration

                    Interlocked.Add(&globalIters.[0], int64 completed) |> ignore // charge the global budget

                if report && control.ElapsedMs - lastReportMs >= MctsReportIntervalMs then
                    reportMctsMerged control trees roots kf maxDepthReached
                    lastReportMs <- control.ElapsedMs
            else
                let struct (d, backedUp) =
                    runIteration tree w cpuct leafDepth kf maxTreeDepth pathNodes pathEdges rootIdx

                if d > maxDepthReached then
                    maxDepthReached <- d

                if backedUp then
                    w.IncIters() // count only backed-up iterations; a mid-leaf stop aborts the playout
                    Interlocked.Increment(&globalIters.[0]) |> ignore // charge the shared global budget

                if report && control.ElapsedMs - lastReportMs >= MctsReportIntervalMs then
                    reportMctsMerged control trees roots kf maxDepthReached
                    lastReportMs <- control.ElapsedMs

        struct (tree, rootIdx, maxDepthReached)

/// Root-parallel orchestrator (mirrors Search.go's LazySMP fan-out): N independent trees, one Worker each,
/// sharing ONLY the lock-free TT (cross-warms leaf negamax) and the Volatile stop flag — no shared tree, no
/// virtual loss, no new locking. Worker 0 (driver) owns soft-time / solved / reporting; when it returns,
/// control.Stop() ends the cohort and we Join the rest. Non-driver workers run on 16 MB stacks (the leaf
/// negamax recurses deep with singular extensions, exactly like `go`). Returns each worker's tree + root
/// index plus the driver's maxDepthReached (for the final info line).
let private runParallel
    (workers: Worker[])
    (control: SearchControl)
    (maxIters: int)
    : struct (MctsTree[] * int[] * int) =
    let n = workers.Length
    let trees: MctsTree[] = Array.zeroCreate n
    let roots: int[] = Array.zeroCreate n
    let globalIters = [| 0L |] // shared GLOBAL iteration budget — every worker charges it; budget splits ~N×

    // Each worker publishes its own trees.[Id]/roots.[Id] slot inside runWorker, so the driver can merge
    // in-progress stats and the slots are already settled here after the join.
    let threads =
        [| for i in 1 .. n - 1 ->
               let wi = workers.[i]

               let t =
                   Thread(
                       ThreadStart(fun () ->
                           runWorker wi control maxIters false false false trees roots globalIters
                           |> ignore),
                       16 * 1024 * 1024
                   )

               t.IsBackground <- true
               t.Start()
               t |]

    // Driver on the calling thread: owns soft-time / stop-on-solved / reporting.
    let struct (_, _, d0) = runWorker workers.[0] control maxIters true true true trees roots globalIters
    control.Stop() // tell the non-driver workers to exit

    for t in threads do
        t.Join()

    struct (trees, roots, d0)

/// Root-parallel MCTS body (mirrors Search.go): N workers, periodic info from the driver, exactly one
/// bestmove from the MERGED root statistics. Parallel results are timing-nondeterministic (same as LazySMP);
/// the deterministic test path goes through `runWorker` directly (Threads is irrelevant there).
let private runMctsParallel (control: SearchControl) : Move =
    control.Reset()
    control.Tt.NewSearch()
    let n = max 1 control.Config.Threads
    let workers = Array.init n (fun i -> Worker(i, (i = 0), control))

    for wk in workers do
        wk.SetupRoot()

    // Aggregate live node count across all workers (relaxed reads — reporting only, exactly like Search.go).
    control.NodeSum <-
        (fun () ->
            let mutable s = 0L

            for wk in workers do
                s <- s + wk.Nodes

            s)
    // Aggregate completed MCTS iterations across all workers (for UCI `info nodes`/`nps` reporting).
    control.IterSum <-
        (fun () ->
            let mutable s = 0L

            for wk in workers do
                s <- s + wk.Iters

            s)

    let stm = workers.[0].Pos.SideToMove
    let (soft, hard) = computeTimes control.Config.MoveOverhead control.Limits stm
    control.StartClock soft hard

    // GLOBAL budget = maxIters across ALL workers (Threads splits a `go depth N` budget ~N×, so wall-clock
    // latency drops with thread count instead of each worker redoing the whole N*MctsIterPerDepth).
    let maxIters =
        if control.Limits.Depth > 0 then
            control.Limits.Depth * MctsIterPerDepth
        else
            0 // unbounded: governed by time / nodes / stop

    let struct (trees, roots, depthReached) = runParallel workers control maxIters
    let kf = float32 (max 1 control.Config.MctsK)
    let acc = mergeRoots trees roots
    let (mv, bestIdx) = bestMergedMove acc

    let best =
        if isLegalRoot workers.[0].Pos mv then
            mv
        else
            firstLegalMove workers.[0].Pos

    control.LastBest <- best
    control.LastScore <- (if bestIdx >= 0 then mergedScoreCp acc.[bestIdx] kf else 0)
    // Final info from the MERGED decision: PV head = the emitted bestmove, score = LastScore, so the info
    // line, the score, and the bestmove all reference the same merged root decision.
    emitMctsInfo control depthReached control.LastScore best trees roots
    writeLine ("bestmove " + toUci best)
    best

/// Production entry (mirrors Search.go). A position so deep that a fixed-depth leaf negamax would overrun
/// MaxPly has no room for an MCTS leaf — hand the whole search to alpha-beta (Search.go emits its own single
/// bestmove, so we must not emit another). Otherwise run the root-parallel MCTS body.
let mctsSearch (control: SearchControl) : Move =
    let safe = MaxPly - control.RootMoves.Length - MaxSearchPly - 1

    if safe < 1 then
        Search.go control
    else
        runMctsParallel control

/// White-box test entry with explicit limits: run up to `iters` iterations on a single worker and return the
/// tree + root index + worker. With defaultLimits this is the deterministic exactly-`iters` path; with a node
/// budget it lets a test drive a mid-leaf stop (the leaf negamax's own CheckTime fires) to exercise backup
/// accounting (root.N stays equal to w.Iters — an aborted playout bumps neither).
let mctsToIterationsTreeLim
    (fen: string)
    (rootMoves: Move[])
    (iters: int)
    (lim: SearchLimits)
    (cfg: SearchConfig)
    (net: SfNetwork option)
    (lc0: Lc0Proto.Lc0Net option)
    : MctsTree * int * Worker =
    let tt = TranspositionTable(max 1 cfg.HashMb)

    let control =
        SearchControl(cfg, lim, tt, fen, rootMoves, ?net = net, ?lc0Net = lc0)

    let w = Worker(0, true, control)
    w.SetupRoot()
    control.Reset()
    control.Tt.NewSearch()
    control.NodeSum <- (fun () -> w.Nodes)
    control.IterSum <- (fun () -> w.Iters)
    control.StartClock 0L 0L
    // Single worker, no orchestrator, no merge: Threads-independent. Slot arrays are length 1 (Id = 0); the
    // global budget with one worker is exactly `iters`, so root.N / w.Iters land on exactly `iters`.
    let trees: MctsTree[] = Array.zeroCreate 1
    let roots: int[] = Array.zeroCreate 1
    let globalIters = [| 0L |]
    let struct (tree, rootIdx, _) = runWorker w control iters false false false trees roots globalIters
    (tree, rootIdx, w)

/// White-box test entry: run EXACTLY `iters` iterations (no time/stop/solver early-out) and return the
/// tree + root index + worker, so tests can assert root.N == iters, the visit distribution, and w.Iters.
let mctsToIterationsTree
    (fen: string)
    (rootMoves: Move[])
    (iters: int)
    (cfg: SearchConfig)
    (net: SfNetwork option)
    : MctsTree * int * Worker =
    mctsToIterationsTreeLim fen rootMoves iters defaultLimits cfg net None

/// Deterministic oracle (runs MCTS regardless of cfg.UseMcts): struct(rootScoreCp, nodes, bestMove).
let mctsToIterations
    (fen: string)
    (rootMoves: Move[])
    (iters: int)
    (cfg: SearchConfig)
    (net: SfNetwork option)
    : struct (int * int64 * Move) =
    let (tree, rootIdx, w) = mctsToIterationsTree fen rootMoves iters cfg net
    let (mv, bestEi) = bestRootMove tree rootIdx

    let best =
        if isLegalRoot w.Pos mv then mv else firstLegalMove w.Pos

    let score = rootScoreCp tree rootIdx bestEi (float32 (max 1 cfg.MctsK))
    struct (score, w.Nodes, best)

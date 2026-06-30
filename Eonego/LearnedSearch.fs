/// Eonego — LearnedSearch: a third search paradigm (Phase A: mechanics + hand-coded priority).
/// A learned BEST-FIRST MINIMAX over an explicit tree of concrete boards — neither alpha-beta nor MCTS.
/// Hand-written: expand (generate legal children + NNUE-eval) and EXACT minimax backup. The "learned"
/// parts (priority net, stop head) arrive in Phase B/C; Phase A uses a fixed hand-coded priority that
/// recovers classic best-first-minimax (expand the principal-variation leaf).
///
/// Default-OFF: dispatched from UCI only when the `UseLearnedSearch` option is true. Leaf eval is the
/// SF FullThreats NNUE (control.Net); the priority is hand-coded (no learned net needed yet).
///
/// NNUE wrinkle (the economic crux): the accumulator is path-local/incremental, but best-first jumps.
/// We keep ONE Position per search, navigate cursor→leaf by unmake-to-LCA + make-down, and MATERIALIZE
/// the leaf (SfEnsureBothComputed) before evaluating its children — otherwise every child re-walks the
/// accumulator to the root (O(b*depth)). See the plan for the cost model.
///
/// AOT/F#: no printf/sprintf (Console.Out only); stackalloc move buffers; the leaf-eval is an injected
/// Position->int so tests can run net-free with a deterministic stub.
module Eonego.LearnedSearch

#nowarn "9" // NativePtr.stackalloc for move buffers

open System
open System.Text
open Microsoft.FSharp.NativeInterop
open Eonego.Bitboard
open Eonego.Move
open Eonego.Position
open Eonego.Nnue
open Eonego.MoveGeneration
open Eonego.Transposition
open Eonego.Search

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
[<Literal>]
let DefaultNodesCap = 1_048_576 // 1M nodes (~40 MB core arrays); override via EONEGO_LS_NODES_CAP

[<Literal>]
let FlagExpanded = 1uy

[<Literal>]
let FlagTerminal = 2uy

[<Literal>]
let MaxBeam = 256 // upper bound on beam width per round (sizes the selection scratch)

// Priority-net feature vector (quantised small ints). The cached layout — and the lstrace dump order —
// MUST match exactly: 0 depth, 1 moveIndex, 2 seeBucket, 3 givesCheck, 4 isCapture, 5 isPromotion,
// 6 leafEvalQ, 7 deltaEvalQ, 8 inCheck, 9 siblingSpread, 10 onPv (the only DYNAMIC slot).
[<Literal>]
let NF = 11

[<Literal>]
let FOnPv = 10 // onPv slot — recomputed per scoring round, never cached

[<Literal>]
let MaxHidden = 64 // upper bound on EONGLS hidden width (sizes inference scratch)

let private writeLine (s: string) = Console.Out.WriteLine s

let inline private clampI (v: int) (lo: int) (hi: int) : int =
    if v < lo then lo elif v > hi then hi else v

let inline private rsh (acc: int) (s: int) : int =
    if s > 0 then (acc + (1 <<< (s - 1))) >>> s else acc

/// Read the arena node cap (env override, AOT-safe — no sprintf).
let nodesCap () : int =
    match Environment.GetEnvironmentVariable "EONEGO_LS_NODES_CAP" with
    | null
    | "" -> DefaultNodesCap
    | s ->
        match Int32.TryParse s with
        | true, v when v >= 64 -> v
        | _ -> DefaultNodesCap

/// Beam width (env override). Default 1 = pure best-first minimax (expand the PV leaf); >1 widens.
let beamWidth () : int =
    match Environment.GetEnvironmentVariable "EONEGO_LS_BEAM" with
    | null
    | "" -> 1
    | s ->
        match Int32.TryParse s with
        | true, v when v >= 1 -> min v MaxBeam
        | _ -> 1

// ---------------------------------------------------------------------------
// EONGLS priority net: a tiny int8 MLP NF→H1→H2→1 over the quantised feature vector. When loaded it
// drives the expansion priority; when absent, the hand-coded fallback priority is used.
// ---------------------------------------------------------------------------
type LsNetwork =
    { NF: int
      H1: int
      H2: int
      OutDim: int
      Shift1: int
      Shift2: int
      OutShift: int
      W1: sbyte[]
      B1: int[]
      W2: sbyte[]
      B2: int[]
      W3: sbyte[]
      B3: int[] }

/// Integer forward pass: a1=clip((x·W1+B1+round)>>S1, 0,127); a2 likewise; out=(a2·W3+B3+round)>>Sout
/// (signed, no final clip). `h1`/`h2` are caller scratch (length ≥ H1/H2). Mirrored bit-for-bit by
/// trainer/ls_intref.py — this is the ONLY engine↔Python parity surface (features are engine-dumped).
let private forwardInto (net: LsNetwork) (x: int[]) (h1: int[]) (h2: int[]) : int =
    for j in 0 .. net.H1 - 1 do
        let mutable acc = net.B1.[j]
        let b = j * net.NF

        for i in 0 .. net.NF - 1 do
            acc <- acc + x.[i] * int net.W1.[b + i]

        h1.[j] <- clampI (rsh acc net.Shift1) 0 127

    for j in 0 .. net.H2 - 1 do
        let mutable acc = net.B2.[j]
        let b = j * net.H1

        for i in 0 .. net.H1 - 1 do
            acc <- acc + h1.[i] * int net.W2.[b + i]

        h2.[j] <- clampI (rsh acc net.Shift2) 0 127

    let mutable acc = net.B3.[0]

    for i in 0 .. net.H2 - 1 do
        acc <- acc + h2.[i] * int net.W3.[i]

    rsh acc net.OutShift

/// Run the priority net on a single feature row (the `lsforward` parity command). Allocates scratch —
/// NOT a hot path; the search hot path uses per-LsTree scratch instead.
let inferRow (net: LsNetwork) (x: int[]) : int =
    forwardInto net x (Array.zeroCreate net.H1) (Array.zeroCreate net.H2)

/// Parse EONGLS bytes → an LsNetwork (None + a logged reason on failure). NO global state — the caller
/// owns the returned net and threads it into the search (this is the fix for the AOT net-load miscompile).
let loadBytes (bytes: byte[]) : LsNetwork option =
    let fail (msg: string) : LsNetwork option =
        writeLine ("info string LearnedFile ignored: " + msg)
        None

    try
        if bytes.Length < 40 then
            fail ("EONGLS too short (" + string bytes.Length + " bytes)")
        else
            let magic = System.Text.Encoding.ASCII.GetString(bytes, 0, 8)

            if magic <> "EONGLS01" then
                fail ("bad magic '" + magic + "'")
            else
                let mutable o = 8

                let i32 () =
                    let v =
                        int bytes.[o]
                        ||| (int bytes.[o + 1] <<< 8)
                        ||| (int bytes.[o + 2] <<< 16)
                        ||| (int bytes.[o + 3] <<< 24)

                    o <- o + 4
                    v

                let version = i32 ()
                let nf = i32 ()
                let h1 = i32 ()
                let h2 = i32 ()
                let outDim = i32 ()
                let s1 = i32 ()
                let s2 = i32 ()
                let so = i32 ()

                if version <> 1 then fail ("bad version " + string version)
                elif nf <> NF then fail ("NF mismatch: net " + string nf + " engine " + string NF)
                elif h1 < 1 || h1 > MaxHidden || h2 < 1 || h2 > MaxHidden then
                    fail ("bad hidden dims " + string h1 + "/" + string h2)
                elif outDim <> 1 then
                    fail ("OutDim must be 1, got " + string outDim)
                else
                    let readI8 (n: int) : sbyte[] =
                        let a = Array.zeroCreate<sbyte> n
                        for k in 0 .. n - 1 do a.[k] <- sbyte bytes.[o + k]
                        o <- o + n
                        a

                    let readI32 (n: int) : int[] =
                        let a = Array.zeroCreate<int> n

                        for k in 0 .. n - 1 do
                            let p = o + k * 4
                            a.[k] <- int bytes.[p] ||| (int bytes.[p + 1] <<< 8) ||| (int bytes.[p + 2] <<< 16) ||| (int bytes.[p + 3] <<< 24)

                        o <- o + n * 4
                        a

                    let w1 = readI8 (nf * h1)
                    let b1 = readI32 h1
                    let w2 = readI8 (h1 * h2)
                    let b2 = readI32 h2
                    let w3 = readI8 (h2 * outDim)
                    let b3 = readI32 outDim

                    if o <> bytes.Length then
                        fail ("trailing " + string (bytes.Length - o) + " bytes")
                    else
                        writeLine "info string LearnedFile loaded"

                        Some
                            { NF = nf
                              H1 = h1
                              H2 = h2
                              OutDim = outDim
                              Shift1 = s1
                              Shift2 = s2
                              OutShift = so
                              W1 = w1
                              B1 = b1
                              W2 = w2
                              B2 = b2
                              W3 = w3
                              B3 = b3 }
    with ex ->
        fail ("exception parsing EONGLS: " + ex.Message)

let loadFile (path: string) : LsNetwork option =
    loadBytes (try System.IO.File.ReadAllBytes path with _ -> [||])

// ---------------------------------------------------------------------------
// Flat struct-of-arrays arena + per-search tree state. Single-threaded (Phase A ignores UCI Threads).
// ---------------------------------------------------------------------------
[<Sealed>]
type LsTree(cap: int) =
    // SoA node columns (indexed by node id).
    let parent: int[] = Array.zeroCreate cap
    let moveFromParent: int[] = Array.zeroCreate cap // Move is a packed int
    let firstChild: int[] = Array.zeroCreate cap
    let childCount: int[] = Array.zeroCreate cap
    let leafEval: int[] = Array.zeroCreate cap
    let value: int[] = Array.zeroCreate cap
    let bestChild: int[] = Array.zeroCreate cap
    let depth: int[] = Array.zeroCreate cap
    let flags: byte[] = Array.zeroCreate cap
    // Cached static priority features per node (NF int16 each; slot FOnPv is filled dynamically).
    let feat: int16[] = Array.zeroCreate (cap * NF)
    // Cached net-priority per node — computed ONCE at creation (the features are static), so frontier
    // selection is an O(frontier) int scan, NOT an O(frontier × NN-forward) rescan every round.
    let priority: int[] = Array.zeroCreate cap
    // Frontier = non-terminal, unexpanded leaves (ids). Worst case ~cap (distinct node ids).
    let frontier: int[] = Array.zeroCreate cap
    let mutable frontierCount = 0
    // Per-round beam selection scratch (node ids of the chosen leaves).
    let beamSel: int[] = Array.zeroCreate MaxBeam
    // Navigation scratch: moves on the path LCA→target (bounded by tree depth).
    let pathBuf: int[] = Array.zeroCreate (MaxSearchPly + 8)
    // Priority-net inference scratch (0-alloc hot path).
    let featBuf: int[] = Array.zeroCreate NF
    let h1Buf: int[] = Array.zeroCreate MaxHidden
    let h2Buf: int[] = Array.zeroCreate MaxHidden
    let mutable nextFree = 0
    let mutable cursor = 0 // node id whose root→node move path the Position currently reflects
    let mutable maxDepthSeen = 0
    let mutable lsNet: LsNetwork option = None // priority net for THIS search (per-instance; no global static)

    /// The priority net driving this search's expansion order (None ⇒ hand-coded fallback priority).
    member _.LsNet
        with get () = lsNet
        and set v = lsNet <- v

    member _.Count = nextFree
    member _.Cap = cap
    member _.Value(i) = value.[i]
    member _.BestChild(i) = bestChild.[i]
    member _.MoveFromParent(i) = moveFromParent.[i]
    member _.FirstChild(i) = firstChild.[i]
    member _.ChildCount(i) = childCount.[i]
    member _.Depth(i) = depth.[i]
    member _.IsTerminal(i) = flags.[i] &&& FlagTerminal <> 0uy
    member _.MaxDepthSeen = maxDepthSeen

    member private _.NewNode(p: int, mv: int, d: int) : int =
        let i = nextFree
        nextFree <- nextFree + 1
        parent.[i] <- p
        moveFromParent.[i] <- mv
        firstChild.[i] <- -1
        childCount.[i] <- 0
        depth.[i] <- d
        flags.[i] <- 0uy
        value.[i] <- 0
        leafEval.[i] <- 0
        bestChild.[i] <- -1
        if d > maxDepthSeen then maxDepthSeen <- d
        i

    /// Reset to a single unevaluated root; caller evaluates the root and seeds the frontier.
    member this.Reset() : int =
        nextFree <- 0
        frontierCount <- 0
        cursor <- 0
        maxDepthSeen <- 0
        this.NewNode(-1, int MoveNone, 0)

    /// Lowest-common-ancestor navigation: unmake cursor up to the LCA, make down to `target`.
    /// The incremental NNUE accumulator follows for free (Make pushes, Unmake pops frames).
    member _.Navigate(pos: Position, target: int) =
        if target <> cursor then
            let mutable a = cursor
            let mutable b = target
            // Lift the deeper of {a,b} to equal depth (unmake `a`; `b` only walks pointers).
            while depth.[a] > depth.[b] do
                pos.Unmake moveFromParent.[a]
                a <- parent.[a]

            while depth.[b] > depth.[a] do
                b <- parent.[b]
            // Walk both up to the LCA.
            while a <> b do
                pos.Unmake moveFromParent.[a]
                a <- parent.[a]
                b <- parent.[b]
            // a == b == LCA. Collect the path target→LCA, then make it in reverse.
            let lca = a
            let mutable t = target
            let mutable np = 0

            while t <> lca do
                pathBuf.[np] <- moveFromParent.[t]
                np <- np + 1
                t <- parent.[t]

            for k in np - 1 .. -1 .. 0 do
                pos.Make pathBuf.[k]

            cursor <- target

    /// Expand `leaf` (PRE: cursor == leaf, leaf unexpanded & non-terminal).
    /// Returns the number of children attached (0 ⇒ the node turned out terminal: mate/stalemate).
    /// -1 ⇒ could not fit the child block (caller should stop).
    member this.Expand(pos: Position, leaf: int, leafEval': Position -> int, materialize: bool) : int =
        // MATERIALIZE the leaf so each 1-ply child eval is O(1) off it (not an O(depth) re-walk to root).
        // `materialize=false` exists ONLY for the cost-model benchmark; production always materializes.
        if materialize && pos.SfActive then pos.SfEnsureBothComputed()

        let pp = NativePtr.stackalloc<Move> MaxMoves
        let buf = Span<Move>(NativePtr.toVoidPtr pp, MaxMoves)
        let n = generateLegal pos buf

        if n = 0 then
            // Terminal at the node being expanded: checkmate or stalemate.
            value.[leaf] <- (if pos.InCheck then -MATE + depth.[leaf] else 0)
            flags.[leaf] <- flags.[leaf] ||| FlagTerminal ||| FlagExpanded
            0
        elif nextFree + n > cap then
            -1
        else
            let d = depth.[leaf] + 1
            let first = nextFree
            let parentEval = leafEval.[leaf]
            // Second buffer for child terminal-detection (separate from the parent's `buf`).
            let cp = NativePtr.stackalloc<Move> MaxMoves
            let cbuf = Span<Move>(NativePtr.toVoidPtr cp, MaxMoves)
            let mutable minE = Int32.MaxValue
            let mutable maxE = Int32.MinValue

            for k in 0 .. n - 1 do
                let m = buf.[k]
                let child = this.NewNode(leaf, int m, d)
                // Move features computed on the PARENT position, BEFORE Make.
                let isCap = isEnPassant m || not (pos.IsEmpty(toSq m))
                let givesChk = pos.GivesCheck m

                let seeB =
                    if not isCap then 0
                    elif pos.SeeGe m 900 then 127
                    elif pos.SeeGe m 300 then 96
                    elif pos.SeeGe m 0 then 64
                    else 32

                pos.Make m
                // Draw at ply>0 (matches Search: isImmediateDraw applied below root only).
                if isImmediateDraw pos then
                    value.[child] <- 0
                    leafEval.[child] <- 0
                    flags.[child] <- FlagTerminal
                else
                    let cn = generateLegal pos cbuf

                    if cn = 0 then
                        // Mate (side-to-move at the child is mated) or stalemate. Score is node-relative.
                        value.[child] <- (if pos.InCheck then -MATE + d else 0)
                        leafEval.[child] <- value.[child]
                        flags.[child] <- FlagTerminal
                    else
                        let ev = leafEval' pos
                        leafEval.[child] <- ev
                        value.[child] <- ev
                        // Cache the quantised static features (slot FOnPv left 0 — filled at scoring time).
                        // This order MUST match the lstrace dump and ls_features.py column order.
                        let fo = child * NF
                        feat.[fo + 0] <- int16 (min d 127)
                        feat.[fo + 1] <- int16 (min k 127)
                        feat.[fo + 2] <- int16 seeB
                        feat.[fo + 3] <- int16 (if givesChk then 64 else 0)
                        feat.[fo + 4] <- int16 (if isCap then 64 else 0)
                        feat.[fo + 5] <- int16 (if isPromotion m then 64 else 0)
                        feat.[fo + 6] <- int16 (clampI (ev / 80) (-127) 127)
                        feat.[fo + 7] <- int16 (clampI ((ev + parentEval) / 80) (-127) 127)
                        feat.[fo + 8] <- int16 (if pos.InCheck then 64 else 0)

                        if ev < minE then minE <- ev
                        if ev > maxE then maxE <- ev

                pos.Unmake m

            // Sibling-value spread over the non-terminal children's leaf evals, cached into each.
            let spreadQ = if maxE >= minE then clampI ((maxE - minE) / 80) 0 127 else 0

            for k in 0 .. n - 1 do
                let child = first + k

                if flags.[child] &&& FlagTerminal = 0uy then
                    feat.[child * NF + 9] <- int16 spreadQ
                    // Cache the net priority now (onPv=false; onPv carries no learned signal — it was a
                    // constant in the bootstrap data). Avoids re-running the net per frontier leaf per round.
                    match lsNet with
                    | Some net ->
                        this.FillFeatures(child, false)
                        priority.[child] <- forwardInto net featBuf h1Buf h2Buf
                    | None -> ()

            firstChild.[leaf] <- first
            childCount.[leaf] <- n
            flags.[leaf] <- flags.[leaf] ||| FlagExpanded
            n

    /// Negamax backup from `node` to the root: value = max over children of (−child.value),
    /// updating bestChild. Short-circuits when neither value nor bestChild changes.
    member _.Backup(node: int) =
        let mutable cur = node
        let mutable go = true

        while go && cur >= 0 do
            let fc = firstChild.[cur]

            if fc < 0 then
                // Terminal/leaf — value already set; just continue to parent.
                cur <- parent.[cur]
            else
                let cc = childCount.[cur]
                let mutable best = Int32.MinValue
                let mutable bc = -1

                for k in 0 .. cc - 1 do
                    let c = fc + k
                    let v = -value.[c]

                    if v > best then
                        best <- v
                        bc <- c

                let oldV = value.[cur]
                let oldB = bestChild.[cur]
                value.[cur] <- best
                bestChild.[cur] <- bc

                if oldV = best && oldB = bc then go <- false
                cur <- parent.[cur]

    /// The current principal-variation leaf: walk root→bestChild until an unexpanded node.
    member private _.PvLeaf() : int =
        let mutable n = 0

        while firstChild.[n] >= 0 && bestChild.[n] >= 0 do
            n <- bestChild.[n]

        n

    member private _.AddFrontier(node: int) =
        frontier.[frontierCount] <- node
        frontierCount <- frontierCount + 1

    /// Assemble the NF feature row for `node` into `featBuf`, filling the dynamic onPv slot.
    member private _.FillFeatures(node: int, onPv: bool) =
        let fo = node * NF

        for i in 0 .. NF - 1 do
            featBuf.[i] <- int feat.[fo + i]

        featBuf.[FOnPv] <- (if onPv then 64 else 0)

    /// Expansion priority. With an EONGLS net loaded, π = the net's expansion-utility output (CACHED at
    /// node creation — see Expand); otherwise the hand-coded fallback (expand the PV leaf — classic
    /// best-first minimax).
    member private _.Priority(node: int, pvLeaf: int) : int =
        match lsNet with
        | Some _ -> priority.[node]
        | None ->
            if node = pvLeaf then 100_000
            else depth.[node] * 1_000 + abs leafEval.[node] / 10

    /// Is `node` already chosen in beamSel[0..count-1]?
    member private _.PickedAlready(node: int, count: int) : bool =
        let mutable found = false
        let mutable i = 0

        while not found && i < count do
            if beamSel.[i] = node then found <- true
            i <- i + 1

        found

    /// Drop expanded/terminal nodes from the frontier, keeping only unexpanded non-terminal leaves
    /// (stable order). Called once per round after expansions have appended the new children.
    member private _.CompactFrontier() =
        let mutable j = 0

        for i in 0 .. frontierCount - 1 do
            let n = frontier.[i]

            if firstChild.[n] < 0 && flags.[n] &&& FlagTerminal = 0uy then
                frontier.[j] <- n
                j <- j + 1

        frontierCount <- j

    /// One beam round: pick the top-`beamW` frontier leaves by (priority desc, node-id asc — the
    /// determinism tiebreak), expand them, back up, compact. `useProximity` reorders the chosen leaves by
    /// node id before expanding; it does NOT change results (within-round expansion is commutative), only
    /// navigation cost — and LsNavBench found it ~24% SLOWER (id is a poor tree-locality proxy), so it is
    /// OFF by default. Returns the number of leaves expanded (0 ⇒ frontier empty or capacity hit).
    member this.Round(pos: Position, leafEval': Position -> int, beamW: int, useProximity: bool, materialize: bool) : int =
        if frontierCount = 0 then
            0
        else
            let pv = this.PvLeaf()
            let w = min beamW (min frontierCount MaxBeam)
            // Select w distinct frontier entries with the highest priority (tie → lowest node id).
            for s in 0 .. w - 1 do
                let mutable bn = Int32.MaxValue
                let mutable bp = Int32.MinValue

                for i in 0 .. frontierCount - 1 do
                    let node = frontier.[i]

                    if not (this.PickedAlready(node, s)) then
                        let p = this.Priority(node, pv)

                        if p > bp || (p = bp && node < bn) then
                            bp <- p
                            bn <- node

                beamSel.[s] <- bn

            // Proximity order: ascending node id ⇒ sibling/cousin clusters expanded back-to-back.
            if useProximity then
                for a in 1 .. w - 1 do
                    let v = beamSel.[a]
                    let mutable b = a - 1

                    while b >= 0 && beamSel.[b] > v do
                        beamSel.[b + 1] <- beamSel.[b]
                        b <- b - 1

                    beamSel.[b + 1] <- v

            let mutable expanded = 0
            let mutable s = 0
            let mutable hitCap = false

            while s < w && not hitCap do
                let leaf = beamSel.[s]
                this.Navigate(pos, leaf)
                let nc = this.Expand(pos, leaf, leafEval', materialize)

                if nc < 0 then
                    hitCap <- true
                else
                    if nc > 0 then
                        let fc = firstChild.[leaf]

                        for k in 0 .. nc - 1 do
                            let c = fc + k
                            // Cap tree depth at MaxSearchPly-1: leaves at the horizon stay as leaf-eval
                            // leaves (never expanded) — keeps the accumulator frame stack (Position.MaxPly
                            // =1024) safe and mate scores inside the ±MATE_IN_MAX_PLY band.
                            if flags.[c] &&& FlagTerminal = 0uy && depth.[c] < MaxSearchPly - 1 then
                                this.AddFrontier c

                    this.Backup(leaf)
                    expanded <- expanded + 1

                s <- s + 1

            this.CompactFrontier()
            expanded

    /// Seed: evaluate the root and put it on the frontier.
    member this.SeedRoot(pos: Position, root: int, leafEval': Position -> int, materialize: bool) =
        if materialize && pos.SfActive then pos.SfEnsureBothComputed()
        let ev = leafEval' pos
        leafEval.[root] <- ev
        value.[root] <- ev
        this.AddFrontier root

    /// EXHAUSTIVE expansion to a uniform depth (the correctness oracle driver — NOT best-first).
    member this.ExpandFull(pos: Position, node: int, maxDepth: int, leafEval': Position -> int) =
        if depth.[node] < maxDepth && flags.[node] &&& FlagTerminal = 0uy then
            this.Navigate(pos, node)
            let nc = this.Expand(pos, node, leafEval', true)

            if nc > 0 then
                let fc = firstChild.[node]

                for k in 0 .. nc - 1 do
                    this.ExpandFull(pos, fc + k, maxDepth, leafEval')

                this.Backup(node)

    member this.PvString() : string =
        let sb = StringBuilder()
        let mutable n = 0

        while firstChild.[n] >= 0 && bestChild.[n] >= 0 do
            let bc = bestChild.[n]
            sb.Append(toUci (moveFromParent.[bc])).Append(' ') |> ignore
            n <- bc

        sb.ToString().TrimEnd()

    /// Length of the principal variation (root→bestChild chain) — the meaningful "depth" to report.
    member _.PvLen() : int =
        let mutable n = 0
        let mutable d = 0

        while firstChild.[n] >= 0 && bestChild.[n] >= 0 do
            n <- bestChild.[n]
            d <- d + 1

        d

    /// The move to play: the root's best child (negamax argmax). MoveNone if the root has no children.
    member _.RootMove() : Move =
        let bc = bestChild.[0]
        if bc >= 0 then moveFromParent.[bc] else MoveNone

    /// UCI move path root→node (space-separated) — for trace labelling / reproducing a node's position.
    member _.MovePath(node: int) : string =
        let mutable n = node
        let mutable np = 0

        while n > 0 do
            pathBuf.[np] <- moveFromParent.[n]
            np <- np + 1
            n <- parent.[n]

        let sb = StringBuilder()

        for k in np - 1 .. -1 .. 0 do
            if sb.Length > 0 then sb.Append(' ') |> ignore
            sb.Append(toUci pathBuf.[k]) |> ignore

        sb.ToString()

    /// Supervised trace generator (Phase B bootstrap): W=1 best-first to `budget`; per expansion, dump the
    /// selected node's feature row + leafEval + ROOT-IMPACT (signed Δ of the backed-up root value the
    /// expansion caused — the self-supervised expansion-utility target). One TSV row per expansion:
    /// rootFen \t movepath \t f0..f{NF-1} \t leafEval \t rootImpact.
    member this.TraceTo(pos: Position, leafEval': Position -> int, budget: int64, writer: System.IO.TextWriter, rootFen: string) =
        let root = this.Reset()
        this.SeedRoot(pos, root, leafEval', true)
        let mutable running = true

        while running do
            if frontierCount = 0 || (budget > 0L && int64 this.Count >= budget) then
                running <- false
            else
                let pv = this.PvLeaf()
                let mutable bn = Int32.MaxValue
                let mutable bp = Int32.MinValue

                for i in 0 .. frontierCount - 1 do
                    let node = frontier.[i]
                    let p = this.Priority(node, pv)

                    if p > bp || (p = bp && node < bn) then
                        bp <- p
                        bn <- node

                let leaf = bn
                // The root (id 0) has no move-from-parent ⇒ no cached features; never dump its row.
                let dump = leaf <> 0
                // Capture the decision's feature row + label inputs BEFORE expanding.
                if dump then this.FillFeatures(leaf, (leaf = pv))
                let path = if dump then this.MovePath leaf else ""
                let leafEv = leafEval.[leaf]
                let rootBefore = value.[0]
                this.Navigate(pos, leaf)
                let nc = this.Expand(pos, leaf, leafEval', true)

                if nc < 0 then
                    running <- false
                else
                    if nc > 0 then
                        let fc = firstChild.[leaf]

                        for k in 0 .. nc - 1 do
                            let c = fc + k

                            if flags.[c] &&& FlagTerminal = 0uy && depth.[c] < MaxSearchPly - 1 then
                                this.AddFrontier c

                    this.Backup(leaf)
                    this.CompactFrontier()

                    if dump then
                        let rootImpact = value.[0] - rootBefore
                        let sb = StringBuilder()
                        sb.Append(rootFen).Append('\t').Append(path).Append('\t') |> ignore

                        for i in 0 .. NF - 1 do
                            sb.Append(featBuf.[i]).Append('\t') |> ignore

                        sb.Append(leafEv).Append('\t').Append(rootImpact) |> ignore
                        writer.WriteLine(sb.ToString())

// ---------------------------------------------------------------------------
// Setup + reporting helpers
// ---------------------------------------------------------------------------
let private setupPos (fen: string) (rootMoves: Move[]) (net: SfNetwork option) : Position =
    let pos = Position()
    pos.LoadFen fen

    match net with
    | Some n -> Nnue.bindNnue n pos
    | None -> ()

    for m in rootMoves do
        pos.Make m

    pos

let private scoreStr (v: int) : string =
    if v >= MATE_IN_MAX_PLY then "mate " + string ((MATE - v + 1) / 2)
    elif v <= -MATE_IN_MAX_PLY then "mate " + string (-((MATE + v + 1) / 2))
    else "cp " + string v

// ---------------------------------------------------------------------------
// Best-first driver (budget-limited) + the deterministic test oracles.
// ---------------------------------------------------------------------------

/// Run best-first to a node budget (0 = until the tree is fully resolved or the arena fills).
/// `report` emits UCI `info` lines (main/UCI use); the oracles pass false.
let private runBestFirst
    (tree: LsTree)
    (pos: Position)
    (control: SearchControl option)
    (leafEval: Position -> int)
    (maxNodes: int64)
    (report: bool)
    (beamW: int)
    (useProximity: bool)
    (materialize: bool)
    (lsNet: LsNetwork option)
    : struct (int * int64 * Move) =
    let root = tree.Reset()
    tree.LsNet <- lsNet // priority net for this search (None ⇒ hand-coded fallback)
    tree.SeedRoot(pos, root, leafEval, materialize)
    let mutable running = true
    let mutable rounds = 0L
    let sw = System.Diagnostics.Stopwatch.StartNew()

    while running do
        let stopped =
            match control with
            | Some c -> c.Stopped || c.SoftTimeUp
            | None -> false

        if stopped then
            running <- false
        elif maxNodes > 0L && int64 tree.Count >= maxNodes then
            running <- false
        else
            let expanded = tree.Round(pos, leafEval, beamW, useProximity, materialize)

            if expanded = 0 then
                running <- false
            else
                rounds <- rounds + 1L

                match control with
                | Some c ->
                    if (rounds &&& 255L) = 0L then c.CheckTime(int64 tree.Count)
                | None -> ()

                if report && (rounds &&& 255L) = 0L then
                    let ms = max 1L sw.ElapsedMilliseconds
                    let nodes = int64 tree.Count

                    writeLine (
                        "info depth " + string (tree.PvLen ())
                        + " seldepth " + string tree.MaxDepthSeen
                        + " score " + scoreStr (tree.Value 0)
                        + " nodes " + string nodes
                        + " nps " + string (nodes * 1000L / ms)
                        + " time " + string ms
                        + " pv " + tree.PvString()
                    )

    struct (tree.Value 0, int64 tree.Count, tree.RootMove())

// --- Public test/tooling oracles (self-contained; mirror Search.searchToDepthNet/searchToNodesNet) ---

/// FULL-EXPANSION oracle: build the complete tree to `depth` with NNUE-or-stub leaves, return the
/// backed-up root value/move. `leafEval` is injected so tests can run net-free with a deterministic stub.
let searchToDepthEval
    (fen: string)
    (rootMoves: Move[])
    (depth: int)
    (cap: int)
    (net: SfNetwork option)
    (leafEval: Position -> int)
    : struct (int * int64 * Move) =
    let pos = setupPos fen rootMoves net
    let tree = LsTree(cap)
    let root = tree.Reset()
    tree.SeedRoot(pos, root, leafEval, true)
    tree.ExpandFull(pos, root, depth, leafEval)
    struct (tree.Value 0, int64 tree.Count, tree.RootMove())

/// Best-first to a node budget (determinism / nps checks). `leafEval` injected.
let searchToNodesEval
    (fen: string)
    (rootMoves: Move[])
    (nodes: int64)
    (cap: int)
    (net: SfNetwork option)
    (leafEval: Position -> int)
    : struct (int * int64 * Move) =
    let pos = setupPos fen rootMoves net
    let tree = LsTree(cap)
    // Proximity OFF: the node-id-sort proximity heuristic benchmarked ~24% SLOWER than priority order
    // (id is a poor tree-locality proxy). Materialize-X stays ON (1.53× tax without it). See LsNavBench.
    runBestFirst tree pos None leafEval nodes false (beamWidth ()) false true None

/// Tuned best-first entry for the cost-model micro-benchmark: explicit beam width / proximity /
/// materialize-X toggles (the last is benchmark-only — production always materializes). Not used in play.
let searchToNodesTuned
    (fen: string)
    (rootMoves: Move[])
    (nodes: int64)
    (cap: int)
    (net: SfNetwork option)
    (leafEval: Position -> int)
    (beamW: int)
    (useProximity: bool)
    (materialize: bool)
    : struct (int * int64 * Move) =
    let pos = setupPos fen rootMoves net
    let tree = LsTree(cap)
    runBestFirst tree pos None leafEval nodes false beamW useProximity materialize None

let private netLeafEval (net: SfNetwork option) : Position -> int =
    match net with
    | Some n -> (fun (p: Position) -> Nnue.evalCp n p)
    | None -> (fun _ -> 0)

let searchToDepth
    (fen: string)
    (rootMoves: Move[])
    (depth: int)
    (cfg: SearchConfig)
    (net: SfNetwork option)
    : struct (int * int64 * Move) =
    let pos = setupPos fen rootMoves net
    let tree = LsTree(nodesCap ())
    let root = tree.Reset()
    let leafEval = netLeafEval net
    tree.SeedRoot(pos, root, leafEval, true)
    tree.ExpandFull(pos, root, depth, leafEval)
    struct (tree.Value 0, int64 tree.Count, tree.RootMove())

let searchToNodes
    (fen: string)
    (rootMoves: Move[])
    (nodes: int64)
    (cfg: SearchConfig)
    (net: SfNetwork option)
    : struct (int * int64 * Move) =
    let pos = setupPos fen rootMoves net
    let tree = LsTree(nodesCap ())
    runBestFirst tree pos None (netLeafEval net) nodes false (beamWidth ()) false true None

// ---------------------------------------------------------------------------
// UCI dispatch entry — mirrors Search.go side effects (clock, info, LastBest, prints bestmove).
// ---------------------------------------------------------------------------
let private computeTimes (control: SearchControl) (pos: Position) : struct (int64 * int64) =
    let lim = control.Limits
    let oh = int64 control.Config.MoveOverhead

    if lim.MoveTime > 0 then
        let t = max 1L (int64 lim.MoveTime - oh)
        struct (t, t)
    elif lim.Infinite || (lim.WTime <= 0 && lim.BTime <= 0) then
        struct (0L, 0L) // depth/nodes/infinite — no wall-clock budget
    else
        let struct (t, inc) =
            if pos.SideToMove = White then struct (lim.WTime, lim.WInc) else struct (lim.BTime, lim.BInc)

        let mtg = if lim.MovesToGo > 0 then lim.MovesToGo else 30
        let soft = max 1L (int64 (t / mtg + inc) - oh)
        let hard = max 1L (min (int64 t - oh) (int64 t / 2L))
        struct (soft, hard)

/// UCI dispatch entry. `lsNet` (the priority net, threaded per-search — NOT a global static) drives the
/// expansion order when Some; None ⇒ hand-coded fallback priority.
let go (control: SearchControl) (lsNet: LsNetwork option) : Move =
    control.Reset()
    control.Tt.NewSearch()
    let pos = setupPos control.RootFen control.RootMoves control.Net
    let leafEval = netLeafEval control.Net
    let struct (soft, hard) = computeTimes control pos
    control.StartClockPonder soft hard control.Limits.Ponder
    let tree = LsTree(nodesCap ())

    let struct (score, _, move) =
        runBestFirst tree pos (Some control) leafEval control.Limits.Nodes true (beamWidth ()) false true lsNet

    let best =
        if move <> MoveNone then move else firstLegalMove pos

    control.LastBest <- best
    control.LastScore <- score
    writeLine ("bestmove " + toUci best)
    best

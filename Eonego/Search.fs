/// Eonego - alpha-beta / PVS search (Phase 3). LazySMP: N independent iterative-deepening searches, each
/// over its OWN Position + History.Tables + stack/PV/buffers/node counter. Shared writable search state is
/// centralized in SearchControl: stop/timing, the TT, and the optional accumulator checkpoint + DAG tables;
/// everything else is per-worker, rebuilt from the immutable root. Fail-soft. With UsePruning=false the search is plain full-window alpha-beta
/// (PVS/LMR/extensions/null/mate-distance/qsearch-SEE/history-writes all disabled) — the correctness oracle.
///
/// AOT/F#: no printfn (Console.Out.WriteLine only); the byref-like MovePick is built per node and never
/// escapes; per-ply move/score buffers are preallocated per worker (no per-frame stackalloc); stop is a
/// Volatile flag, never an exception.
module Eonego.Search

#nowarn "9" // NativePtr.stackalloc in the rare draw/root legality helpers

open System
open System.Text
open System.Threading
open Microsoft.FSharp.NativeInterop
open Eonego.Bitboard
open Eonego.Move
open Eonego.Position
open Eonego.Nnue
open Eonego.MoveGeneration
open Eonego.History
open Eonego.MovePick
open Eonego.Transposition
open Eonego.AccCheckpoint
open Eonego.DagNode
open Eonego.DagWorkQueue

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
[<Literal>]
let MaxSearchPly = 246 // < Position.MaxPly (1024); deeper plies hit a hard guard

[<Literal>]
let MATE = 32000

[<Literal>]
let MATE_IN_MAX_PLY = 31754 // MATE - MaxSearchPly

[<Literal>]
let INF = 32001

[<Literal>]
let VALUE_NONE = 32002 // sentinel; never a real score, never cutoff-compared

[<Literal>]
let StackOffset = 4 // so (ss-1)/(ss-2) are valid at the root

// Mate-ply correction: store relative to the node, read back relative to the root (the #1 search bug).
let inline valueToTt (v: int) (ply: int) : int =
    if v >= MATE_IN_MAX_PLY then v + ply
    elif v <= -MATE_IN_MAX_PLY then v - ply
    else v

let inline valueFromTt (v: int) (ply: int) : int =
    if v = VALUE_NONE then VALUE_NONE
    elif v >= MATE_IN_MAX_PLY then v - ply
    elif v <= -MATE_IN_MAX_PLY then v + ply
    else v

/// LMR reduction table r[depth][moveCount], built once at init (read-only ⇒ LazySMP-safe, like the eval
/// tables). r grows with both depth and move number: late moves at high depth are searched much shallower.
// Flat 64*64 (row d*64 + m), NOT int[,]: a single-dimension array's element access elides bounds checks the
// JIT cannot for multidim T[,], and the `< 64` guard below already bounds both indices.
let private Reductions: int[] =
    let t = Array.zeroCreate (64 * 64)

    for d in 1..63 do
        for m in 1..63 do
            t.[d * 64 + m] <- int (0.5 + log (float d) * log (float m) / 2.2)

    t

let inline private reduction (depth: int) (moveCount: int) : int =
    if depth < 64 && moveCount < 64 then
        Reductions.[depth * 64 + moveCount]
    else
        6

// ---------------------------------------------------------------------------
// Config / limits (immutable; read-once, never per-node mutable shared state)
// ---------------------------------------------------------------------------
type SearchConfig =
    { Threads: int
      HashMb: int
      UseTt: bool
      UsePruning: bool
      UseProbCut: bool
      UseIir: bool
      UseRazoring: bool
      UseHistoryPruning: bool
      UseDeltaPruning: bool
      UseContHist: bool
      UseSingular: bool
      UseNmpVerify: bool
      UseLmrTweaks: bool
      UseAspTweaks: bool
      // Qsearch TT protocol: non-PV bound cutoffs at qsearch entry + a BoundLower store on stand-pat
      // fail-highs. The store is tree-neutral (a revisit reaches the identical stand-pat value) and only
      // skips repeated NNUE evals; the cutoff changes tree shape (suite-total nodes -3..-9% at d13-d15,
      // per-position volatile) — keep it toggleable for SPRT.
      UseQsTt: bool
      // A bound-consistent TT score replaces the raw static eval as the WORKING eval that RFP/razoring/
      // NMP (and the qsearch stand-pat) prune on — a real search result is tighter than a heuristic eval.
      // Stack.StaticEval, `improving`, move-loop futility and every TT store keep the RAW eval (SF's
      // `unadjustedStaticEval` contract; correction history depends on it staying raw).
      UseTtEvalAdjust: bool
      MoveOverhead: int
      // Phase 1: NNUE accumulator checkpoint cache. Set to 0 to disable; ~4 MiB is the recommended default
      // (1024 slots, ~4.1 KiB/slot). Cleared per search by `SearchControl.NewSearch` (alongside the TT gen
      // bump). Probes are best-effort lock-free (matching the TT); populates on each successful
      // `Position.EnsureBothComputed` materialization.
      AccCheckpointMb: int
      // Phase 2: DAG node table. Set to 0 to disable; ~2 MiB is the default (the table carries in-flight
      // partial-result + alpha/beta-window metadata the TT cannot encode). Worker search routes through the DAG
      // claim/probe/complete lifecycle for nodes within the split-ply window; fallback to plain recursion
      // outside that window or when the table is full.
      DagHashMb: int
      // Phase 3: root-move work distribution via the Vyukov MPMC work queue. OFF (default) = classic LazySMP
      // (every worker independently searches the full tree, sharing only via TT/DAG). ON = root parallelism:
      // the main worker searches the best root move (full PVS window) and pushes the rest to a shared
      // DagWorkQueue; all workers pop root moves and search them with null windows. The TT carries the
      // results back to the main worker's collection phase. Falls back to LazySMP when Threads <= 1.
      UseWorkQueue: bool
      // Number of principal variations to report (UCI MultiPV). 1 = classic single-PV search. >1 makes the
      // main worker search each root line with the previous lines' best moves excluded at the root; helpers
      // always run single-PV. Forces the classic LazySMP path (no root-move work queue).
      MultiPv: int }

type SearchLimits =
    { MoveTime: int
      WTime: int
      WInc: int
      BTime: int
      BInc: int
      MovesToGo: int
      Depth: int
      Nodes: int64
      Infinite: bool
      Mate: int
      Ponder: bool }

let defaultConfig =
    { Threads = 1
      HashMb = 16
      UseTt = true
      UsePruning = true
      UseProbCut = true
      UseIir = true
      UseRazoring = true
      UseHistoryPruning = true
      UseDeltaPruning = true
      UseContHist = true
      UseSingular = true
      UseNmpVerify = true
      UseLmrTweaks = true
      UseAspTweaks = true
      UseQsTt = true
      UseTtEvalAdjust = true
      MoveOverhead = 10
      AccCheckpointMb = 0
      DagHashMb = 0
      UseWorkQueue = false
      MultiPv = 1 }

let defaultLimits =
    { MoveTime = 0
      WTime = 0
      WInc = 0
      BTime = 0
      BInc = 0
      MovesToGo = 0
      Depth = 0
      Nodes = 0L
      Infinite = false
      Mate = 0
      Ponder = false }

[<Struct>]
type StackEntry =
    { mutable StaticEval: int
      mutable CurrentMove: Move
      mutable MovedPiece: Piece
      mutable ExcludedMove: Move } // singular-extension exclusion: the move banned in THIS node's search

// ---------------------------------------------------------------------------
// Draw / material helpers (the search owns draw detection; eval deferred it)
// ---------------------------------------------------------------------------
let inline hasNonPawnMaterial (pos: Position) : bool =
    let us = pos.SideToMove

    (pos.PiecesCT us Knight
     ||| pos.PiecesCT us Bishop
     ||| pos.PiecesCT us Rook
     ||| pos.PiecesCT us Queen)
    <> 0UL

/// KvK, KNvK, KBvK, KB-vs-KB-same-colour. All from public bitboard accessors.
let insufficientMaterial (pos: Position) : bool =
    if pos.Pieces Pawn <> 0UL || pos.Pieces Rook <> 0UL || pos.Pieces Queen <> 0UL then
        false
    else
        let knights = popCount (pos.Pieces Knight)
        let bishops = popCount (pos.Pieces Bishop)
        let minors = knights + bishops

        if minors <= 1 then
            true
        elif knights = 0 && bishops = 2 then
            let wB = pos.PiecesCT White Bishop
            let bB = pos.PiecesCT Black Bishop

            if popCount wB = 1 && popCount bB = 1 then
                let a = lsb wB
                let b = lsb bB
                ((fileOf a + rankOf a) &&& 1) = ((fileOf b + rankOf b) &&& 1)
            else
                false
        else
            false

/// Twofold-in-tree = draw (sound search optimisation). Walk the Position's own key history, bounded by the
/// reversible window AND the last null move so the scan never crosses a null. Earliest repeat is 4 plies.
let isRepetition (pos: Position) : bool =
    let endN = min pos.Rule50 pos.PliesFromNull

    if endN < 4 then
        false
    else
        let cur = pos.Key
        let mutable i = 4
        let mutable found = false

        while not found && i <= endN do
            if pos.KeyAt i = cur then
                found <- true

            i <- i + 2

        found

let private hasAnyLegalMove (pos: Position) : bool =
    let p = NativePtr.stackalloc<Move> MaxMoves
    let buf = Span<Move>(NativePtr.toVoidPtr p, MaxMoves)
    generateLegal pos buf > 0

/// 50-move rule applies (Rule50 >= 100) EXCEPT when it is checkmate on the 100th ply: if a legal move
/// exists the rule holds; if none exists the node's move loop yields mate/stalemate, preserving the exception.
let isImmediateDraw (pos: Position) : bool =
    let rule50 = pos.Rule50

    if rule50 < 4 && pos.Pieces Pawn <> 0UL then
        false
    else
        isRepetition pos
        || insufficientMaterial pos
        || (rule50 >= 100 && hasAnyLegalMove pos)

let firstLegalMove (pos: Position) : Move =
    let p = NativePtr.stackalloc<Move> MaxMoves
    let buf = Span<Move>(NativePtr.toVoidPtr p, MaxMoves)
    let n = generateLegal pos buf
    if n > 0 then buf.[0] else MoveNone

let private countLegalMoves (pos: Position) : int =
    let p = NativePtr.stackalloc<Move> MaxMoves
    let buf = Span<Move>(NativePtr.toVoidPtr p, MaxMoves)
    generateLegal pos buf

let isLegalRoot (pos: Position) (m: Move) : bool =
    if m = MoveNone then
        false
    else
        let p = NativePtr.stackalloc<Move> MaxMoves
        let buf = Span<Move>(NativePtr.toVoidPtr p, MaxMoves)
        let n = generateLegal pos buf
        let mutable found = false

        for i in 0 .. n - 1 do
            if buf.[i] = m then
                found <- true

        found

// ---------------------------------------------------------------------------
// Shared control: atomically-published stop/timing plus shared best-effort caches.
// ---------------------------------------------------------------------------
[<Literal>]
let private PonderIdle = 0

[<Literal>]
let private PonderBudgetStored = 1

[<Literal>]
let private PonderHitPending = 2

[<Literal>]
let private PonderArmed = 3

[<Sealed>]
type SearchControl
    (config: SearchConfig, limits: SearchLimits, tt: TranspositionTable, rootFen: string, rootMoves: Move[], ?net: Network) =
    let mutable stopFlag = 0
    let mutable startTick = System.Diagnostics.Stopwatch.GetTimestamp()
    let mutable softMs = 0L
    let mutable hardMs = 0L
    let mutable baseSoftMs = 0L // the un-scaled optimum; the per-iteration manager rescales softMs from it
    let mutable ponderSoft = 0L // real budget remembered during a ponder search; armed by PonderHit
    let mutable ponderHard = 0L
    let mutable ponderState = PonderIdle

    let elapsedMs () =
        let start = Volatile.Read(&startTick)
        let delta = System.Diagnostics.Stopwatch.GetTimestamp() - start
        (delta * 1000L) / System.Diagnostics.Stopwatch.Frequency

    // Phase 1: per-search NNUE accumulator checkpoint cache. `null` when the config disables it (zero MiB).
    // Owned here, bound to each worker's `Position` via `BindCheckpoint` in `Worker.SetupRoot`; cleared in
    // `NewSearch` alongside `tt.newSearch`.
    let accCheckpoint: AccCheckpointTable =
        if config.AccCheckpointMb <= 0 then null else AccCheckpointTable(config.AccCheckpointMb)
    // Phase 2: per-search DAG node table. `null` when disabled in config. Carries in-flight partial-result
    // + alpha/beta-window metadata the TT cannot encode; the worker search's claim->expand->complete lifecycle
    // routes through it (Phase 2+). Cleared per search alongside the TT and the accumulator cache.
    let dagTable: DagNodeTable =
        if config.DagHashMb <= 0 then null else DagNodeTable(config.DagHashMb)
    // Phase 3: root-move work-distribution queue (Vyukov MPMC). `null` when UseWorkQueue is off or Threads <= 1.
    // Recreated per search in `NewSearch` (drained + fresh). The main worker pushes unsearched root-move keys
    // each iteration; all workers pop and search the corresponding subtree with a null PVS window.
    let mutable workQueue: DagWorkQueue = null
    // Shared alpha published by the main worker after its first-move PVS search, so helpers know the null-window
    // bound. Volatile — read by helpers, written by main.
    let mutable sharedRootAlpha = 0
    member _.Config = config
    member _.Limits = limits
    member _.Tt = tt
    member _.RootFen = rootFen
    member _.RootMoves = rootMoves
    member _.Net: Network option = net
    /// Borrowed reference to the per-search accumulator checkpoint cache; `null` when disabled in config.
    member _.AccCheckpoint: AccCheckpointTable = accCheckpoint
    /// Borrowed reference to the per-search DAG node table; `null` when disabled in config.
    member _.DagTable: DagNodeTable = dagTable
    /// Borrowed reference to the per-search work-distribution queue; `null` when UseWorkQueue is off.
    member _.WorkQueue: DagWorkQueue = workQueue
    /// Shared root alpha for the null-window helper searches (published by the main worker).
    member _.SharedRootAlpha: int = Volatile.Read(&sharedRootAlpha)
    member _.SetSharedRootAlpha(v: int) = Volatile.Write(&sharedRootAlpha, v)
    member val LastBest: Move = MoveNone with get, set // result of the most recent go()
    member val LastScore: int = 0 with get, set
    /// Aggregate live node count across all workers (relaxed reads — reporting only, set by go()).
    member val NodeSum: unit -> int64 = (fun () -> 0L) with get, set
    member _.Stopped: bool = Volatile.Read(&stopFlag) <> 0
    member _.Stop() = Volatile.Write(&stopFlag, 1)
    member _.Reset() = Volatile.Write(&stopFlag, 0)
    /// Per-search reset shared between `tt.NewSearch` and the checkpoint cache; safe to call only between
    /// searches (no live Workers probing). `go` invokes this once before spawning workers.
    member this.NewSearch() : unit =
        tt.NewSearch()
        match accCheckpoint with
        | null -> ()
        | cache -> cache.Clear()
        match dagTable with
        | null -> ()
        | dag -> dag.NewSearch()
        // Phase 3: recreate the work queue fresh for each search (guarantees empty + resets Vyukov seqs).
        if config.UseWorkQueue && config.Threads > 1 then
            workQueue <- DagWorkQueue(256)
        else
            workQueue <- null
    member _.ElapsedMs: int64 = elapsedMs ()

    member _.SoftTimeUp: bool =
        let soft = Volatile.Read(&softMs)
        soft > 0L && elapsedMs () >= soft

    member _.BaseSoftMs: int64 = Volatile.Read(&baseSoftMs)
    member _.SetSoft(v: int64) = Volatile.Write(&softMs, v)

    member _.StartClock (soft: int64) (hard: int64) =
        Volatile.Write(&startTick, System.Diagnostics.Stopwatch.GetTimestamp())
        Volatile.Write(&baseSoftMs, soft)
        Volatile.Write(&hardMs, hard)
        Volatile.Write(&softMs, soft)

    /// Ponder: search unbounded now and remember the real budget; PonderHit arms it (the clock starts then,
    /// so the time used while pondering on the opponent's clock is free). If a ponderhit already raced ahead
    /// (arrived before this stored the budget), arm immediately here.
    member this.StartClockPonder (soft: int64) (hard: int64) (ponder: bool) =
        if ponder then
            Volatile.Write(&ponderSoft, soft)
            Volatile.Write(&ponderHard, hard)
            // Start unbounded before publishing the stored-budget state; this prevents a racing PonderHit
            // from arming the real clock only to be overwritten back to unbounded.
            this.StartClock 0L 0L

            let prev = Interlocked.CompareExchange(&ponderState, PonderBudgetStored, PonderIdle)

            if prev = PonderHitPending then
                if Interlocked.CompareExchange(&ponderState, PonderArmed, PonderHitPending) = PonderHitPending then
                    this.StartClock soft hard
        else
            this.StartClock soft hard

    /// The opponent played the predicted move: arm the remembered ponder budget. No-op for a non-ponder
    /// search (spurious ponderhit); if the budget is not stored yet, StartClockPonder arms it when it runs.
    member this.PonderHit() =
        if limits.Ponder then
            let mutable done' = false

            while not done' do
                let state = Volatile.Read(&ponderState)

                if state = PonderIdle then
                    done' <- Interlocked.CompareExchange(&ponderState, PonderHitPending, PonderIdle) = PonderIdle
                elif state = PonderBudgetStored then
                    if Interlocked.CompareExchange(&ponderState, PonderArmed, PonderBudgetStored) = PonderBudgetStored then
                        let soft = Volatile.Read(&ponderSoft)
                        let hard = Volatile.Read(&ponderHard)
                        this.StartClock soft hard
                        done' <- true
                else
                    done' <- true

    /// Main worker only: convert a time/node budget overrun into the shared stop flag.
    member _.CheckTime(nodes: int64) =
        let hard = Volatile.Read(&hardMs)

        if (hard > 0L && elapsedMs () >= hard) || (limits.Nodes > 0L && nodes >= limits.Nodes) then
            Volatile.Write(&stopFlag, 1)

// ---------------------------------------------------------------------------
// Per-worker state: own Position, Tables, search stack, triangular PV, move/score/quiet buffers, counters.
// ---------------------------------------------------------------------------
[<Sealed>]
type Worker(id: int, isMain: bool, control: SearchControl) =
    let pos = Position()
    let tables = Tables()
    let stack: StackEntry[] = Array.zeroCreate (MaxSearchPly + StackOffset + 4)
    let moveBuf: Move[] = Array.zeroCreate (MaxSearchPly * MaxMoves)
    let scoreBuf: int[] = Array.zeroCreate (MaxSearchPly * MaxMoves)
    let quietsBuf: Move[] = Array.zeroCreate (MaxSearchPly * MaxMoves)
    // Dedicated per-ply buffers for the singular-extension exclusion search (which re-enters negamax at the
    // SAME ply). Must be per-ply, NOT a single set: SE at ply P recurses into a normal child at P+1 that can
    // trigger its OWN exclusion search, so two exclusion searches at different plies are live on the call
    // stack simultaneously and must not share a buffer. (Same-ply SE still cannot nest — the ExcludedMove
    // guard blocks that — so one slot per ply suffices.)
    let exclMoveBuf: Move[] = Array.zeroCreate (MaxSearchPly * MaxMoves)
    let exclScoreBuf: int[] = Array.zeroCreate (MaxSearchPly * MaxMoves)
    let exclQuietsBuf: Move[] = Array.zeroCreate (MaxSearchPly * MaxMoves)
    let pv: Move[] = Array.zeroCreate (MaxSearchPly * MaxSearchPly)
    let mutable nodes = 0L
    let mutable selDepth = 0
    let mutable rootBest = MoveNone
    let mutable rootScore = 0
    let mutable completedDepth = 0
    let mutable nmpMinPly = 0 // NMP allowed only at ply >= this; set during an NMP-verification region (Feature 7)
    let mutable stopSeen = false
    // MultiPV: best moves of the lines already searched this iteration, excluded at the root (ply 0) so the
    // next line finds the next-best move. Empty (count 0) in single-PV play — the ply-0 check is then free.
    let rootExcluded: Move[] = Array.zeroCreate 256
    let mutable nRootExcluded = 0
    member _.Id = id
    member _.IsMain = isMain
    member _.Control = control
    member _.Pos = pos
    member _.Tables = tables
    member _.Stack = stack
    member _.MoveBuf = moveBuf
    member _.ScoreBuf = scoreBuf
    member _.QuietsBuf = quietsBuf
    member _.ExclMoveBuf = exclMoveBuf
    member _.ExclScoreBuf = exclScoreBuf
    member _.ExclQuietsBuf = exclQuietsBuf
    member _.Pv = pv

    member _.Nodes
        with get () = nodes
        and set v = nodes <- v

    member _.SelDepth
        with get () = selDepth
        and set v = selDepth <- v

    member _.RootBest
        with get () = rootBest
        and set v = rootBest <- v

    member _.RootScore
        with get () = rootScore
        and set v = rootScore <- v

    member _.CompletedDepth
        with get () = completedDepth
        and set v = completedDepth <- v

    member _.NmpMinPly
        with get () = nmpMinPly
        and set v = nmpMinPly <- v

    member _.StopSeen
        with get () = stopSeen
        and set v = stopSeen <- v

    /// MultiPV: 0-based index of the line currently being searched (reporting: currmovenumber offset).
    member val RootPvIdx = 0 with get, set

    member _.ClearRootExclusions() = nRootExcluded <- 0

    member _.AddRootExclusion(m: Move) =
        if nRootExcluded < rootExcluded.Length then
            rootExcluded.[nRootExcluded] <- m
            nRootExcluded <- nRootExcluded + 1

    member _.IsRootExcluded(m: Move) : bool =
        let mutable i = 0
        let mutable found = false

        while not found && i < nRootExcluded do
            if rootExcluded.[i] = m then
                found <- true

            i <- i + 1

        found

    /// Rebuild this worker's Position from the immutable root (FEN + replayed moves) and reset per-search state.
    member _.SetupRoot() =
        pos.LoadFen control.RootFen

        match control.Net with
        | Some net -> Nnue.bindNnue net pos
        | None -> ()

        for m in control.RootMoves do
            pos.Make m

        // Phase 1: borrow the per-search checkpoint cache so `EnsureBothComputed` can probe/store snapshots
        // during the upcoming search. `null` disables the fast-path entirely. Seed the root snapshot now —
        // `EnableNnue` just materialized frame 0 and set the computed flags, so the early-return path inside
        // `EnsureBothComputed` would otherwise skip the populate on the first eval at the root.
        pos.BindCheckpoint control.AccCheckpoint
        pos.SeedCheckpoint()

        tables.Clear()
        nodes <- 0L
        selDepth <- 0
        rootBest <- MoveNone
        rootScore <- 0
        completedDepth <- 0
        stopSeen <- false
        nRootExcluded <- 0

let inline private pollStop (w: Worker) : bool =
    if w.StopSeen then
        true
    elif (w.Nodes &&& 8191L) = 0L then
        if w.IsMain then
            w.Control.CheckTime w.Nodes

        let stopped = w.Control.Stopped
        if stopped then
            w.StopSeen <- true
        stopped
    else
        false

let private writeLine (s: string) = System.Console.Out.WriteLine(s)

let evalPos (w: Worker) (pos: Position) : int =
    match w.Control.Net with
    | Some net ->
        if PosProf.Enabled then
            let profT0 = System.Diagnostics.Stopwatch.GetTimestamp()
            let v = Nnue.evalCp net pos
            PosProf.tEval <- PosProf.tEval + (System.Diagnostics.Stopwatch.GetTimestamp() - profT0)
            PosProf.nEval <- PosProf.nEval + 1L
            v
        else
            Nnue.evalCp net pos
    | None -> 0 // unreachable in play: UCI refuses to search with no net (see UCI.startSearch)

let private updatePv (w: Worker) (ply: int) (m: Move) =
    let pv = w.Pv
    let basep = ply * MaxSearchPly
    pv.[basep] <- m

    if ply + 1 < MaxSearchPly then
        let baseChild = (ply + 1) * MaxSearchPly
        let mutable i = 0
        let mutable cont = true

        while cont && i < MaxSearchPly - 1 do
            let cmv = pv.[baseChild + i]
            pv.[basep + 1 + i] <- cmv
            if cmv = MoveNone then cont <- false else i <- i + 1

        if cont then
            pv.[basep + MaxSearchPly - 1] <- MoveNone
    else
        pv.[basep + 1] <- MoveNone

// ---------------------------------------------------------------------------
// Quiescence (fail-soft leaves)
// ---------------------------------------------------------------------------
let rec qsearch (w: Worker) (pos: Position) (alphaIn: int) (betaIn: int) (ply: int) : int =
    w.Nodes <- w.Nodes + 1L

    let stopNow = pollStop w

    if ply > w.SelDepth then
        w.SelDepth <- ply

    if ply > 0 && stopNow then
        0
    elif ply > 0 && isImmediateDraw pos then
        0
    elif ply >= MaxSearchPly || pos.StPly >= Position.AccStackLimit then
        (if pos.InCheck then 0 else evalPos w pos)
    else
        let cfg = w.Control.Config
        let useTt = cfg.UseTt
        let usePruning = cfg.UsePruning
        let inCheck = pos.InCheck
        let isPvNode = betaIn - alphaIn > 1
        let mutable alpha = alphaIn
        let beta = betaIn
        let mutable ttHit = false
        let mutable ttMove = MoveNone
        let mutable ttEval = VALUE_NONE
        let mutable ttScore = 0
        let mutable ttBound = BoundNone

        if useTt then
            let struct (hit, m, sc, ev, _, bd, _) = w.Control.Tt.Probe pos.Key

            if hit then
                ttHit <- true
                ttMove <- m
                ttEval <- ev
                ttScore <- valueFromTt sc ply
                ttBound <- bd

        let mutable best = -INF
        let mutable bestMove = MoveNone
        let mutable rawEval = VALUE_NONE
        let mutable cutoff = false

        // TT cutoff (non-PV): every entry is depth-sufficient here (qsearch stores at depth 0, negamax
        // deeper), so only the bound has to agree with the window — the same gate as negamax step 2.
        if
            useTt
            && cfg.UseQsTt
            && ttHit
            && not isPvNode
            && (ttBound = BoundExact
                || (ttBound = BoundLower && ttScore >= beta)
                || (ttBound = BoundUpper && ttScore <= alpha))
        then
            best <- ttScore
            cutoff <- true

        if not inCheck && not cutoff then
            // Stand-pat: reuse the TT-stored static eval when present (it is the same deterministic
            // evalPos value qsearch would recompute and store), mirroring negamax. Bit-exact, skips the
            // NNUE forward on a TT hit.
            let rawSp = if ttEval <> VALUE_NONE then ttEval else evalPos w pos
            rawEval <- rawSp

            // TT score as a better stand-pat (same bound-consistency rule as negamax step 3b). rawEval —
            // what a fail-high store below and the post-move-loop store persist — stays the raw eval.
            let sp =
                if
                    usePruning
                    && cfg.UseTtEvalAdjust
                    && ttHit
                    && abs ttScore < MATE_IN_MAX_PLY
                    && (ttBound &&& (if ttScore > rawSp then BoundLower else BoundUpper)) <> 0
                then
                    ttScore
                else
                    rawSp

            best <- sp

            if sp >= beta then
                cutoff <- true
                // Stand-pat fail-high: without a store, a transposed re-visit repeats the eval (and its
                // lazy-accumulator catch-up walk). sp is this node's genuine fail-soft return value, so a
                // depth-0 BoundLower entry is sound; only written on a TT miss (never clobber deeper data).
                if useTt && cfg.UseQsTt && not ttHit then
                    w.Control.Tt.Store pos.Key 0 BoundLower (valueToTt sp ply) rawEval MoveNone false
            elif sp > alpha then
                alpha <- sp

        if cutoff then
            best
        else
            let us = pos.SideToMove
            let ksq = pos.KingSquare us
            // Loop-invariant: BlockersForKing(us) is restored across every Make/Unmake in this loop, but the
            // JIT can't prove it (Make/Unmake are opaque), so hoist it out of the per-move legality test.
            let usBlockers = pos.BlockersForKing us
            let moves = w.MoveBuf.AsSpan(ply * MaxMoves, MaxMoves)
            let scores = w.ScoreBuf.AsSpan(ply * MaxMoves, MaxMoves)
            let mutable mp = mkQSearch pos w.Tables ttMove moves scores
            let mutable movesPlayed = 0
            let mutable m = nextMove &mp false

            while m <> MoveNone && not cutoff do
                let needsCheck =
                    isEnPassant m || fromSq m = ksq || testBit usBlockers (fromSq m)

                let legal = (not needsCheck) || isLegal pos m

                let prune =
                    usePruning
                    && not inCheck
                    && not (isPromotion m)
                    && not (pos.GivesCheck m)
                    && (not (pos.SeeGe m 1) // existing SEE prune
                        // delta pruning: a capture that can't lift alpha even after winning the
                        // piece (captures only — guard pieceType against an empty destination).
                        || (cfg.UseDeltaPruning
                            && (let dst = pos.PieceOn(toSq m) in
                                (isEnPassant m || dst <> NoPiece)
                                && (let capturedValue =
                                        if isEnPassant m then pieceValueOf Pawn
                                        else pieceValueOf (pieceType dst)

                                    rawEval + 200 + capturedValue <= alpha))))

                if legal && not prune then
                    movesPlayed <- movesPlayed + 1
                    pos.Make m
                    w.Control.Tt.Prefetch pos.Key // child probes this exact key at entry
                    let v = -(qsearch w pos (-beta) (-alpha) (ply + 1))
                    pos.Unmake m

                    if pollStop w then
                        cutoff <- true
                    else
                        if v > best then
                            best <- v
                            bestMove <- m

                            if v > alpha then
                                alpha <- v

                        if alpha >= beta then
                            cutoff <- true

                m <- if cutoff then MoveNone else nextMove &mp false

            if inCheck && movesPlayed = 0 then
                -MATE + ply
            else
                if useTt && not (pollStop w) then
                    let bound =
                        if best <= alphaIn then BoundUpper
                        elif best >= beta then BoundLower
                        else BoundExact

                    w.Control.Tt.Store pos.Key 0 bound (valueToTt best ply) rawEval bestMove false

                best

// ---------------------------------------------------------------------------
// Negamax / PVS (fail-soft)
// ---------------------------------------------------------------------------
let rec negamax (w: Worker) (pos: Position) (alphaIn: int) (betaIn: int) (depthIn: int) (ply: int) (isPv: bool) (cutNode: bool) : int =
    w.Nodes <- w.Nodes + 1L

    let stopNow = pollStop w

    if ply > w.SelDepth then
        w.SelDepth <- ply
    // Terminate this PV row up-front so a parent's PV copy stops here when this node bottoms out into
    // qsearch / a draw / a TT cutoff without producing its own continuation (prevents stale PV tails).
    if isPv && ply < MaxSearchPly then
        w.Pv.[ply * MaxSearchPly] <- MoveNone

    if ply > 0 && stopNow then
        0
    elif ply > 0 && isImmediateDraw pos then
        0
    elif ply >= MaxSearchPly || pos.StPly >= Position.AccStackLimit then
        (if pos.InCheck then 0 else evalPos w pos)
    elif depthIn <= 0 then
        qsearch w pos alphaIn betaIn ply
    else
        let cfg = w.Control.Config
        let usePruning = cfg.UsePruning
        let useTt = cfg.UseTt
        let inCheck = pos.InCheck
        let ssCur = ply + StackOffset
        // Singular extension: this node's own excluded move (set by the parent for a singular search). Clear
        // the CHILD's slot so a normal recursion sees no exclusion; never clear our own (that erases it).
        let excludedMove = w.Stack.[ssCur].ExcludedMove
        w.Stack.[ssCur + 1].ExcludedMove <- MoveNone
        let mutable alpha = alphaIn
        let mutable beta = betaIn
        let mutable result = 0
        let mutable produced = false
        let mutable ttMove = MoveNone
        let mutable ttHit = false
        let mutable ttScore = 0
        let mutable ttEval = VALUE_NONE
        let mutable ttDepth = 0
        let mutable ttBound = BoundNone
        // Phase 2: claim->expand->complete lifecycle. `dagClaim` is the table slot token returned by
        // `TryClaim`; it must be passed back to Complete/Cancel so only the claimant can publish or release
        // the node. The claim window is captured before alpha mutates during the move loop.
        let mutable dagClaim = NoClaim
        let mutable dagClaimAlpha = 0
        let mutable dagClaimBeta = 0
        let mutable dagCompleted = false
        // ttPv: this node is (or was) on a PV. Sticky — start from isPv, OR in a probed former-PV mark.
        let mutable ttPv = isPv

        // 1. mate-distance pruning (gated -> oracle stays strictly full-window)
        if usePruning then
            if alpha < -MATE + ply then
                alpha <- -MATE + ply

            if beta > MATE - ply - 1 then
                beta <- MATE - ply - 1

            if alpha >= beta then
                result <- alpha
                produced <- true

        // 2. TT probe + (non-PV) cutoff
        if not produced && useTt then
            let struct (hit, m, sc, ev, dp, bd, pv) = w.Control.Tt.Probe pos.Key

            if hit then
                ttHit <- true
                ttMove <- m
                ttScore <- valueFromTt sc ply
                ttEval <- ev
                ttDepth <- dp
                ttBound <- bd
                ttPv <- ttPv || pv

                if not isPv && excludedMove = MoveNone && dp >= depthIn then
                    if
                        bd = BoundExact
                        || (bd = BoundLower && ttScore >= beta)
                        || (bd = BoundUpper && ttScore <= alpha)
                    then
                        result <- ttScore
                        produced <- true

        // 2b. DAG probe + claim (Phase 2). The DAG carries the alpha/beta-window context the TT cannot encode: a
        // hit with `status = Done` AND `windowContains(storedAlpha, storedBeta, alpha, beta)` permits a verbatim
        // cutoff (the original search was under a window that *contains* ours, so its bound applies here —
        // the alpha-beta transposition soundness theorem). The TT may have a looser or unrelated bound, so
        // the DAG can cutoff where the TT cannot. Claim is best-effort: on race/cluster-full the caller
        // silently proceeds to plain recursion (semantically identical), exactly like a TT miss. Both the
        // probe and the claim gate on `excludedMove = MoveNone` to avoid recording singular-extension nodes
        // whose key is identical to a normal search's.
        if not produced && useTt && excludedMove = MoveNone then
            match w.Control.DagTable with
            | null -> ()
            | dag ->
                let struct (st, dm, ds, da, db, dd) = dag.Probe pos.Key

                if st = StatusDone && dd >= depthIn && windowContains da db alpha beta then
                    // Mirror the TT cutoff gate: non-PV nodes only (PV nodes must re-examine the full PV
                    // to populate `w.Pv`. The TT cutoff applies the same `not isPv` rule).
                    if not isPv then
                        result <- valueFromTt ds ply
                        produced <- true
                elif st = StatusEmpty then
                    let claim = dag.TryClaim pos.Key alpha beta depthIn
                    if claim <> NoClaim then
                        dagClaim <- claim
                        dagClaimAlpha <- alpha
                        dagClaimBeta <- beta

        // 3. static eval (+ "improving": static eval higher than 2 plies ago — hoisted up so ProbCut/pruning
        //    can read it; prunes less when our position is improving)
        let mutable staticEval = VALUE_NONE
        let mutable improving = false

        if not produced then
            staticEval <-
                if inCheck then VALUE_NONE
                elif ttHit && ttEval <> VALUE_NONE then ttEval
                else evalPos w pos

            w.Stack.[ssCur].StaticEval <- staticEval

            improving <-
                (not inCheck)
                && (let prev2 = w.Stack.[ssCur - 2].StaticEval in
                    prev2 = VALUE_NONE || staticEval > prev2)

        // 3b. TT score as a better WORKING eval: a bound-consistent search score is tighter than the
        //     heuristic static eval, so RFP/razoring/NMP below prune against it. Only this local —
        //     Stack.StaticEval (already stored raw above), `improving`, the move-loop futility gate and
        //     the TT store all keep the RAW eval (SF's `unadjustedStaticEval` contract).
        let mutable workingEval = staticEval

        if
            not produced
            && usePruning
            && cfg.UseTtEvalAdjust
            && not inCheck
            && ttHit
            && abs ttScore < MATE_IN_MAX_PLY
            && (ttBound &&& (if ttScore > staticEval then BoundLower else BoundUpper)) <> 0
        then
            workingEval <- ttScore

        // 4. pruning block (gated, non-PV, not in check; disabled during a singular exclusion search)
        if not produced && usePruning && not isPv && not inCheck && excludedMove = MoveNone then
            if
                depthIn <= 6
                && workingEval - 120 * depthIn - (if ttPv then 20 else 0) >= beta
                && abs beta < MATE_IN_MAX_PLY
            then
                result <- workingEval
                produced <- true
            // razoring: at shallow depth a static eval far below alpha is verified by qsearch;
            // if even captures can't lift alpha, fail low. Mutually exclusive with RFP/null-move.
            elif
                cfg.UseRazoring
                && depthIn <= 3
                && abs alpha < MATE_IN_MAX_PLY
                && workingEval + (240 + 200 * depthIn) <= alpha
            then
                let v = qsearch w pos alpha (alpha + 1) ply

                if v <= alpha && not (pollStop w) then
                    result <- v
                    produced <- true
            elif
                depthIn >= 3
                && ply >= w.NmpMinPly
                && pos.PliesFromNull > 0
                && workingEval >= beta
                && hasNonPawnMaterial pos
            then
                let r = 3 + depthIn / 4 + (if workingEval - beta > 200 then 1 else 0)
                w.Stack.[ssCur].CurrentMove <- MoveNull
                w.Stack.[ssCur].MovedPiece <- NoPiece
                pos.MakeNull()
                w.Control.Tt.Prefetch pos.Key // child probes this exact key at entry
                let v = -(negamax w pos (-beta) (-beta + 1) (depthIn - r) (ply + 1) false false)
                pos.UnmakeNull()

                if v >= beta && not (pollStop w) then
                    // Zugzwang verification: at high depth (and not already inside a verification region),
                    // re-search the ORIGINAL position with NMP suppressed for the next few plies (NmpMinPly),
                    // and only trust the cutoff if it also fails high. NmpMinPly both bounds the verification
                    // cost (deeper plies resume NMP) and prevents re-verifying at every node.
                    let clampedV = if v >= MATE_IN_MAX_PLY then beta else v

                    if (not cfg.UseNmpVerify) || depthIn < 12 || w.NmpMinPly > 0 then
                        result <- clampedV
                        produced <- true
                    else
                        w.NmpMinPly <- ply + 3 * (depthIn - r) / 4
                        let vv = negamax w pos (beta - 1) beta (depthIn - r) ply false false
                        w.NmpMinPly <- 0

                        if vv >= beta && not (pollStop w) then
                            result <- clampedV
                            produced <- true

        // 4.5 ProbCut: a strong capture that HOLDS a reduced search above beta+margin is a cutoff.
        if
            not produced
            && usePruning
            && cfg.UseProbCut
            && not isPv
            && not inCheck
            && excludedMove = MoveNone
            && depthIn >= 5
            && abs beta < MATE_IN_MAX_PLY
        then
            let probCutBeta = beta + 200 - (if improving then 50 else 0)
            // Conservative skip heuristic (NOT a proof the node can't reach probCutBeta).
            let ttBlocks =
                ttHit && ttDepth >= depthIn - 3 && ttScore <> VALUE_NONE && ttScore < probCutBeta

            if not ttBlocks then
                let us = pos.SideToMove
                let ksq = pos.KingSquare us
                let usBlockers = pos.BlockersForKing us // loop-invariant across Make/Unmake (see qsearch note)
                let pcMoves = w.MoveBuf.AsSpan(ply * MaxMoves, MaxMoves)
                let pcScores = w.ScoreBuf.AsSpan(ply * MaxMoves, MaxMoves)
                let mutable pcMp = mkProbCut pos w.Tables ttMove (probCutBeta - staticEval) pcMoves pcScores
                let mutable pcMove = nextMove &pcMp false

                while pcMove <> MoveNone && not produced && not (pollStop w) do
                    let needsCheck =
                        isEnPassant pcMove
                        || fromSq pcMove = ksq
                        || testBit usBlockers (fromSq pcMove)

                    if (not needsCheck) || isLegal pos pcMove then
                        w.Stack.[ssCur].CurrentMove <- pcMove
                        w.Stack.[ssCur].MovedPiece <- pos.PieceOn(fromSq pcMove)
                        pos.Make pcMove
                        w.Control.Tt.Prefetch pos.Key // child probes this exact key at entry
                        let mutable v = -(qsearch w pos (-probCutBeta) (-probCutBeta + 1) (ply + 1))

                        if v >= probCutBeta then
                            v <- -(negamax w pos (-probCutBeta) (-probCutBeta + 1) (depthIn - 4) (ply + 1) false (not cutNode))

                        pos.Unmake pcMove

                        if not (pollStop w) && v >= probCutBeta then
                            // Trust a sufficient existing TT entry over clobbering it; else store + return v.
                            if ttHit && ttDepth >= depthIn - 3 && ttScore <> VALUE_NONE && ttScore >= probCutBeta then
                                result <- ttScore
                            else
                                if useTt then
                                    w.Control.Tt.Store pos.Key (depthIn - 3) BoundLower (valueToTt v ply) staticEval pcMove false

                                result <- v

                            produced <- true

                    pcMove <- if produced || pollStop w then MoveNone else nextMove &pcMp false

        // 5. move loop
        if not produced then
            // IIR: with no TT move to order on, reduce a ply and let the next iteration's TT
            // move improve ordering. `depth` below is the *searched* depth (may be depthIn - 1),
            // not the requested one. `ply > 0` keeps IIR off the fixed-depth root (test == game).
            let mutable depth = depthIn

            if usePruning && cfg.UseIir && ply > 0 && ttMove = MoveNone && depthIn >= 4 then
                depth <- depth - 1

            let prevMove =
                if ply > 0 then
                    w.Stack.[ssCur - 1].CurrentMove
                else
                    MoveNone

            let prevPiece = if ply > 0 then w.Stack.[ssCur - 1].MovedPiece else NoPiece

            let cm =
                if ply > 0 && prevMove <> MoveNone && prevMove <> MoveNull && prevPiece <> NoPiece then
                    w.Tables.CounterMove prevPiece (toSq prevMove)
                else
                    MoveNone

            let k1 = w.Tables.Killer ply 0
            let k2 = w.Tables.Killer ply 1

            // Continuation-history context: the 1-ply (ss-1) and 2-ply (ss-2) previous moves' piece/to.
            // `-1` disables a term (root / after null / NoPiece, or UseContHist off). Zero-init stack slots
            // below the root carry CurrentMove = MoveNone, so the MoveNone/MoveNull guard rejects them.
            let contOn = cfg.UseContHist

            let struct (prev1Pc, prev1To) =
                if contOn && prevMove <> MoveNone && prevMove <> MoveNull && prevPiece <> NoPiece then
                    struct (prevPiece, toSq prevMove)
                else
                    struct (-1, -1)

            let prev2Move = w.Stack.[ssCur - 2].CurrentMove
            let prev2Piece = w.Stack.[ssCur - 2].MovedPiece

            let struct (prev2Pc, prev2To) =
                if contOn && prev2Move <> MoveNone && prev2Move <> MoveNull && prev2Piece <> NoPiece then
                    struct (prev2Piece, toSq prev2Move)
                else
                    struct (-1, -1)

            let us = pos.SideToMove
            let ksq = pos.KingSquare us
            let usBlockers = pos.BlockersForKing us // loop-invariant across Make/Unmake (see qsearch note)
            // During a singular exclusion search (same ply) use dedicated per-ply buffers so we don't clobber
            // the OUTER node's in-flight move/score/quiet data at this ply (nor a concurrent exclusion search
            // at another ply up the call stack).
            let useExcl = excludedMove <> MoveNone
            let bufBase = ply * MaxMoves
            let moves = (if useExcl then w.ExclMoveBuf else w.MoveBuf).AsSpan(bufBase, MaxMoves)
            let scores = (if useExcl then w.ExclScoreBuf else w.ScoreBuf).AsSpan(bufBase, MaxMoves)
            let quietsBuf = if useExcl then w.ExclQuietsBuf else w.QuietsBuf

            let mutable mp =
                mkMain pos w.Tables ttMove k1 k2 cm depth prev1Pc prev1To prev2Pc prev2To moves scores

            if isPv then
                w.Pv.[ply * MaxSearchPly] <- MoveNone

            let quietsBase = bufBase
            let mutable best = -INF
            let mutable bestMove = MoveNone
            let mutable moveCount = 0
            let mutable nQuiets = 0
            let mutable skipQuiets = false
            let mutable cutoff = false
            // Multicut (set in the singular block): the node fails high on TWO moves at reduced depth, so
            // the whole move loop is abandoned and `best` returned WITHOUT a TT store.
            let mutable multicut = false
            let mutable m = nextMove &mp skipQuiets

            while m <> MoveNone && not cutoff do
                let needsCheck =
                    isEnPassant m || fromSq m = ksq || testBit usBlockers (fromSq m)

                // MultiPV: at the root, skip the best moves of the lines already searched this iteration.
                let rootSkip = ply = 0 && m <> MoveNone && w.IsRootExcluded m

                if m <> excludedMove && not rootSkip && ((not needsCheck) || isLegal pos m) then
                    moveCount <- moveCount + 1

                    // UCI currmove: only at the root of the main worker, and only once the search has run
                    // long enough that a GUI actually benefits (avoids flooding stdout in fast games).
                    // Not during a singular-exclusion re-search (same ply 0, reduced depth — not real progress).
                    if ply = 0 && w.IsMain && excludedMove = MoveNone && w.Control.ElapsedMs > 3000L then
                        writeLine (
                            "info depth "
                            + string depthIn
                            + " currmove "
                            + toUci m
                            + " currmovenumber "
                            + string (moveCount + w.RootPvIdx)
                        )

                    let isQuiet =
                        (pos.PieceOn(toSq m) = NoPiece) && not (isEnPassant m) && not (isPromotion m)

                    let givesCheck = pos.GivesCheck m
                    let mutable doMove = true
                    // perf: `reduction depth moveCount` is a pure table lookup; `depth`/`moveCount` are both
                    // unchanged for the rest of this move's processing, so the forward-pruning use below and
                    // the LMR base reduction further down (if both run) share one evaluation instead of two.
                    let mutable cachedReduction = -1
                    // perf: for a checking move, `pos.SeeGe m 0` (needed below by the check-extension decision)
                    // is reused by the quiet SEE-prune check just below it: SeeGe is monotone in its threshold,
                    // and the quiet-prune threshold is always <= 0, so SeeGe(m,0)=true already proves the laxer
                    // quiet-prune check would also pass, letting it skip its own exchange walk entirely.
                    let mutable see0Known = false
                    let mutable see0Val = false

                    // --- forward pruning: never on the first real move, never when we might be getting mated ---
                    if usePruning && not isPv && not inCheck && best > -MATE_IN_MAX_PLY then
                        if cachedReduction < 0 then
                            cachedReduction <- reduction depth moveCount

                        let lmrDepth = max 0 (depth - 1 - cachedReduction)

                        if isQuiet then
                            // late-move (move-count) pruning — stop trying quiets once deep into the list
                            if
                                depth <= 8
                                && moveCount >= (3 + depth * depth) / (if improving then 1 else 2)
                            then
                                skipQuiets <- true
                                doMove <- false
                            // history — a late quiet with very negative butterfly history. Per-move
                            // skip (NOT skipQuiets): the picker only partial-sorts quiets, so a worse
                            // quiet can precede a better one in the unsorted tail.
                            elif
                                cfg.UseHistoryPruning
                                && lmrDepth <= 6
                                && moveCount > 3
                                && w.Tables.MainHistory us (fromTo m) < -500 - 800 * lmrDepth
                            then
                                doMove <- false
                            // futility — a quiet that can't lift alpha at shallow (reduced) depth
                            elif (not givesCheck) && lmrDepth <= 6 && staticEval + 120 + 110 * lmrDepth <= alpha then
                                skipQuiets <- true
                                doMove <- false
                            // SEE — a quiet that walks into a losing exchange. For a checking move, probe the
                            // threshold-0 exchange first (cached for reuse by the extension decision below): if
                            // it already passes, the laxer threshold is guaranteed to pass too (SEE monotonicity,
                            // lax threshold <= 0), so the second walk at the lax threshold is skipped entirely.
                            elif
                                lmrDepth <= 7
                                && (if givesCheck then
                                        see0Known <- true
                                        see0Val <- pos.SeeGe m 0
                                        not see0Val && not (pos.SeeGe m (-25 * lmrDepth * lmrDepth))
                                    else
                                        not (pos.SeeGe m (-25 * lmrDepth * lmrDepth)))
                            then
                                doMove <- false
                        elif (not givesCheck) && depth <= 6 && not (pos.SeeGe m (-90 * depth)) then
                            // SEE — a capture that loses material by static exchange at shallow depth
                            doMove <- false

                    let mutable ext = 0

                    if doMove then
                        if usePruning && givesCheck then
                            let see0 = if see0Known then see0Val else pos.SeeGe m 0
                            ext <- if see0 then 1 else 0

                        // Singular / double extension: the TT move, at depth >= 8 with a depth-sufficient
                        // lower-bound TT score, is "singular" if every OTHER move fails low against a
                        // reduced exclusion window. Extend it (twice if it fails low by a clear margin).
                        // A fail-HIGH exclusion search is informative too (SF-master else-branches):
                        // multicut — some OTHER move also beats beta at reduced depth, so with the TT move's
                        // own lower bound this node is a proven cut-node: return sv without searching a
                        // single move (and without a TT store — the bound is only reduced-depth);
                        // negative extensions — the TT move is NOT singular here, so trim its depth when
                        // its TT score already beats beta, or at expected cut-nodes. All ply > 0 only
                        // (the root must always search and produce a move).
                        if
                            usePruning
                            && cfg.UseSingular
                            && excludedMove = MoveNone
                            && m = ttMove
                            && depth >= 8
                            && ttDepth >= depth - 3
                            && (ttBound &&& BoundLower) <> 0
                            && abs ttScore < MATE_IN_MAX_PLY
                        then
                            let singularBeta = ttScore - 2 * depth
                            let sDepth = (depth - 1) / 2
                            w.Stack.[ssCur].ExcludedMove <- m
                            let sv = negamax w pos (singularBeta - 1) singularBeta sDepth ply false cutNode
                            w.Stack.[ssCur].ExcludedMove <- MoveNone

                            if not (pollStop w) then
                                if sv < singularBeta then
                                    ext <- ext + (if (not isPv) && sv < singularBeta - 16 then 2 else 1)
                                elif ply > 0 && sv >= beta && abs sv < MATE_IN_MAX_PLY then
                                    multicut <- true
                                    best <- sv
                                    cutoff <- true
                                elif ply > 0 && ttScore >= beta then
                                    ext <- ext - 2
                                elif ply > 0 && cutNode then
                                    ext <- ext - 2

                    if doMove && not multicut then
                        // Cap so check + double extensions can't push newDepth past depthIn + 1.
                        let maxExt = depthIn + 1 - (depth - 1)

                        if ext > maxExt then
                            ext <- max 0 maxExt

                        let newDepth = depth - 1 + ext
                        w.Stack.[ssCur].CurrentMove <- m
                        w.Stack.[ssCur].MovedPiece <- pos.PieceOn(fromSq m)

                        if isQuiet && nQuiets < MaxMoves - 1 then
                            quietsBuf.[quietsBase + nQuiets] <- m
                            nQuiets <- nQuiets + 1

                        pos.Make m
                        w.Control.Tt.Prefetch pos.Key // child probes this exact key at entry
                        let mutable v = 0

                        if moveCount = 1 then
                            v <- -(negamax w pos (-beta) (-alpha) newDepth (ply + 1) isPv (if isPv then false else not cutNode))
                        elif not usePruning then
                            v <- -(negamax w pos (-beta) (-alpha) newDepth (ply + 1) false (not cutNode))
                        else
                            // LMR: table reduction, deeper for non-improving / quiet moves, lighter for captures
                            let mutable r =
                                if depth >= 3 && moveCount > 1 && not givesCheck then
                                    let mutable rr =
                                        if cachedReduction < 0 then
                                            reduction depth moveCount
                                        else
                                            cachedReduction

                                    if not improving then
                                        rr <- rr + 1

                                    if not isQuiet then
                                        rr <- rr - 1

                                    // Richer LMR (gated; OFF => exactly the legacy reduction above). Net-REDUCING
                                    // tweaks only: reduce MORE at expected cut-nodes and for strongly-negative
                                    // continuation/main history. The "reduce less" variants (ttPv, strong
                                    // history) were dropped — each r-1 deepens the scout and triggers cascading
                                    // full-depth re-searches, which roughly doubled the tree for no measured
                                    // gain. Good moves already sort first (continuation history) so they get a
                                    // small move-count reduction anyway.
                                    if cfg.UseLmrTweaks then
                                        if cutNode && m <> ttMove then
                                            rr <- rr + 2

                                        let pc = w.Stack.[ssCur].MovedPiece

                                        let hist =
                                            w.Tables.MainHistory us (fromTo m)
                                            + w.Tables.ContHistory1 prev1Pc prev1To pc (toSq m)
                                            + w.Tables.ContHistory2 prev2Pc prev2To pc (toSq m)

                                        if hist < -12000 then
                                            rr <- rr + 1

                                    rr
                                else
                                    0

                            if r > newDepth - 1 then
                                r <- newDepth - 1

                            if r < 0 then
                                r <- 0

                            v <- -(negamax w pos (-alpha - 1) (-alpha) (newDepth - r) (ply + 1) false true)

                            if v > alpha && r > 0 then
                                v <- -(negamax w pos (-alpha - 1) (-alpha) newDepth (ply + 1) false (not cutNode))

                            if v > alpha && v < beta then
                                v <- -(negamax w pos (-beta) (-alpha) newDepth (ply + 1) isPv false)

                        pos.Unmake m

                        if pollStop w then
                            cutoff <- true
                        else
                            if v > best then
                                best <- v
                                bestMove <- m

                                if v > alpha then
                                    alpha <- v

                                    if isPv then
                                        updatePv w ply m

                            if alpha >= beta then
                                cutoff <- true

                                if usePruning then
                                    let bonus = statBonus depth

                                    if isQuiet then
                                        let mPc = pos.PieceOn(fromSq m)
                                        w.Tables.UpdateMain us m bonus
                                        w.Tables.UpdateCont1 prev1Pc prev1To mPc (toSq m) bonus
                                        w.Tables.UpdateCont2 prev2Pc prev2To mPc (toSq m) bonus

                                        for qi in 0 .. nQuiets - 2 do
                                            let q = quietsBuf.[quietsBase + qi]
                                            let qPc = pos.PieceOn(fromSq q)
                                            w.Tables.UpdateMain us q (-bonus)
                                            w.Tables.UpdateCont1 prev1Pc prev1To qPc (toSq q) (-bonus)
                                            w.Tables.UpdateCont2 prev2Pc prev2To qPc (toSq q) (-bonus)

                                        w.Tables.SetKiller ply m

                                        if
                                            ply > 0
                                            && prevMove <> MoveNone
                                            && prevMove <> MoveNull
                                            && prevPiece <> NoPiece
                                        then
                                            w.Tables.SetCounter prevPiece (toSq prevMove) m
                                    else
                                        let capPt =
                                            if isEnPassant m then
                                                Pawn
                                            else
                                                pieceType (pos.PieceOn(toSq m))

                                        w.Tables.UpdateCapture (pos.PieceOn(fromSq m)) (toSq m) capPt bonus

                m <- if cutoff then MoveNone else nextMove &mp skipQuiets

            if moveCount = 0 then
                // No legal move searched. During an exclusion search this means the excluded move was the
                // only move -> report a fail-low so the caller treats it as singular (don't store).
                result <- (if excludedMove <> MoveNone then alpha elif inCheck then -MATE + ply else 0)
            else
                if useTt && not multicut && not (pollStop w) && excludedMove = MoveNone then
                    let bound =
                        if best <= alphaIn then BoundUpper
                        elif best >= beta then BoundLower
                        else BoundExact

                    w.Control.Tt.Store pos.Key depth bound (valueToTt best ply) staticEval bestMove ttPv

                    // Phase 2: publish the result under the original claimed window. `alpha` is mutated by
                    // searched moves, so using it here would make the reusable window narrower than the
                    // actual search window.
                    if dagClaim <> NoClaim then
                        match w.Control.DagTable with
                        | null -> () // claimed against a since-disabled table; ignore
                        | dag ->
                            dagCompleted <-
                                dag.Complete dagClaim pos.Key bestMove (valueToTt best ply) dagClaimAlpha dagClaimBeta depth

                result <- best

        if dagClaim <> NoClaim && not dagCompleted then
            match w.Control.DagTable with
            | null -> ()
            | dag -> dag.Cancel(dagClaim, pos.Key) |> ignore

        result

// ---------------------------------------------------------------------------
// Reporting (Console only — never printfn)
// ---------------------------------------------------------------------------
let private scoreString (score: int) : string =
    if score >= MATE_IN_MAX_PLY then
        "mate " + string ((MATE - score + 1) / 2)
    elif score <= -MATE_IN_MAX_PLY then
        "mate " + string (-((MATE + score + 1) / 2))
    else
        "cp " + string score

/// One `info ... multipv k ...` line. `bound` is "" or " lowerbound"/" upperbound" (aspiration fail
/// high/low). The PV is read from `pv.[pvBase..]` (MoveNone-terminated); `fallback` covers an empty PV.
let private reportLine (w: Worker) (depth: int) (pvNum: int) (score: int) (bound: string) (pv: Move[]) (pvBase: int) (fallback: Move) =
    let ms = w.Control.ElapsedMs
    let nodes = w.Control.NodeSum() // aggregate across all workers, not just the main one
    let nps = if ms > 0L then nodes * 1000L / ms else nodes
    let sb = StringBuilder(160)

    sb
        .Append("info depth ")
        .Append(depth)
        .Append(" seldepth ")
        .Append(w.SelDepth)
        .Append(" multipv ")
        .Append(pvNum)
        .Append(" score ")
        .Append(scoreString score)
        .Append(bound)
        .Append(" nodes ")
        .Append(nodes)
        .Append(" nps ")
        .Append(nps)
        .Append(" hashfull ")
        .Append(w.Control.Tt.Hashfull())
        .Append(" time ")
        .Append(ms)
        .Append(" pv")
    |> ignore

    let mutable i = 0
    let mutable cont = true

    while cont && i < MaxSearchPly do
        let mv = pv.[pvBase + i]

        if mv = MoveNone then
            cont <- false
        else
            sb.Append(' ').Append(toUci mv) |> ignore
            i <- i + 1

    if i = 0 && fallback <> MoveNone then
        sb.Append(' ').Append(toUci fallback) |> ignore

    writeLine (sb.ToString())

let private reportInfo (w: Worker) (depth: int) (score: int) =
    reportLine w depth 1 score "" w.Pv 0 w.RootBest

// ---------------------------------------------------------------------------
// Iterative deepening (aspiration). All workers run it; only the main reports + sets the time stop.
// ---------------------------------------------------------------------------
let iterativeDeepening (w: Worker) (maxDepth: int) : unit =
    let cfg = w.Control.Config
    // MultiPV: only the main worker searches extra lines (they exist purely for reporting); helpers stay
    // single-PV. Clamped to the number of legal root moves so every line has a distinct best move.
    let multiPv =
        if w.IsMain && cfg.MultiPv > 1 then
            max 1 (min cfg.MultiPv (countLegalMoves w.Pos))
        else
            1

    // Per-line state: previous score (aspiration centre), and this iteration's score/move/PV per line.
    let prevScores = Array.create multiPv (evalPos w w.Pos)
    let lineScores = Array.create multiPv 0
    let lineMoves = Array.create multiPv MoveNone
    let linePvs: Move[] = Array.zeroCreate (multiPv * MaxSearchPly)

    // Swap two lines across every parallel array (insertion sort below keeps GUI output score-ordered).
    let swapLines (i: int) (j: int) =
        let ts = lineScores.[i] in
        lineScores.[i] <- lineScores.[j]
        lineScores.[j] <- ts
        let tp = prevScores.[i] in
        prevScores.[i] <- prevScores.[j]
        prevScores.[j] <- tp
        let tm = lineMoves.[i] in
        lineMoves.[i] <- lineMoves.[j]
        lineMoves.[j] <- tm

        for k in 0 .. MaxSearchPly - 1 do
            let t = linePvs.[i * MaxSearchPly + k]
            linePvs.[i * MaxSearchPly + k] <- linePvs.[j * MaxSearchPly + k]
            linePvs.[j * MaxSearchPly + k] <- t

    let mutable depth = 1

    while depth <= maxDepth && not w.Control.Stopped do
        w.ClearRootExclusions()
        let mutable pvIdx = 0
        let mutable iterationOk = true

        while pvIdx < multiPv && iterationOk do
            w.RootPvIdx <- pvIdx
            let prev = prevScores.[pvIdx]
            // Tweaked: initial window scaled by score magnitude (tighter near equal scores). OFF => flat 16.
            let mutable delta =
                if cfg.UseAspTweaks then 10 + prev * prev / 15000 else 16

            let fullWindow = depth <= 4 || abs prev >= MATE_IN_MAX_PLY
            let mutable alpha = if fullWindow then -INF else max (-INF) (prev - delta)
            let mutable beta = if fullWindow then INF else min INF (prev + delta)
            let mutable score = 0
            let mutable searching = true
            // Tweaked: drop the re-search depth by up to 3 plies on consecutive fail-highs (reported depth
            // stays `depth`). OFF (or no fail-high) => attemptDepth = depth, i.e. the legacy behaviour.
            let mutable attemptDepth = depth

            while searching do
                score <- negamax w w.Pos alpha beta (max 1 attemptDepth) 0 true false

                if w.Control.Stopped then
                    searching <- false
                elif score <= alpha then
                    // Root fail-low: tell the GUI the score is only an upper bound (long searches only,
                    // matching the currmove threshold, so fast games don't flood stdout).
                    if w.IsMain && w.Control.ElapsedMs > 3000L then
                        reportLine w depth (pvIdx + 1) score " upperbound" w.Pv 0 w.RootBest

                    beta <- (alpha + beta) / 2
                    alpha <- max (-INF) (score - delta)
                    delta <- delta * 2
                    attemptDepth <- depth // a fail-low re-search runs at full depth again
                elif score >= beta then
                    if w.IsMain && w.Control.ElapsedMs > 3000L then
                        reportLine w depth (pvIdx + 1) score " lowerbound" w.Pv 0 w.RootBest

                    beta <- min INF (score + delta)
                    delta <- delta * 2

                    if cfg.UseAspTweaks && attemptDepth > depth - 3 && abs score < MATE_IN_MAX_PLY then
                        attemptDepth <- attemptDepth - 1
                else
                    searching <- false

            if w.Control.Stopped then
                iterationOk <- false
            else
                prevScores.[pvIdx] <- score
                lineScores.[pvIdx] <- score
                lineMoves.[pvIdx] <- w.Pv.[0]
                Array.blit w.Pv 0 linePvs (pvIdx * MaxSearchPly) MaxSearchPly
                // Exclude this line's best move at the root so the next line finds the next-best move.
                w.AddRootExclusion w.Pv.[0]
                pvIdx <- pvIdx + 1

        w.ClearRootExclusions()
        w.RootPvIdx <- 0

        if iterationOk then
            // Order lines by score (descending) so multipv 1 is always the strongest line and each line's
            // aspiration centre follows its move next iteration.
            for i in 1 .. multiPv - 1 do
                let mutable j = i

                while j > 0 && lineScores.[j - 1] < lineScores.[j] do
                    swapLines (j - 1) j
                    j <- j - 1

            // Keep w.Pv row 0 = the BEST line's PV (go()'s final report and RootBest read it).
            if multiPv > 1 then
                Array.blit linePvs 0 w.Pv 0 MaxSearchPly

            w.RootScore <- lineScores.[0]
            w.RootBest <- lineMoves.[0]
            w.CompletedDepth <- depth

            if w.IsMain then
                for k in 0 .. multiPv - 1 do
                    reportLine w depth (k + 1) lineScores.[k] "" linePvs (k * MaxSearchPly) lineMoves.[k]

            // Stop on a proven mate: a forced mate score cannot be improved by deeper iterations, so
            // deepening further only burns the clock (and, at extreme depth, pushes the search toward the
            // ply cap). Stop the whole search (shared flag) the moment the main worker confirms one.
            // MultiPV keeps deepening — the user is exploring alternatives, not just the mate line.
            if w.IsMain && multiPv = 1 && abs lineScores.[0] >= MATE_IN_MAX_PLY then
                w.Control.Stop()

        depth <- depth + 1

        // Between iterations the main worker stops once the soft (optimum) budget is spent; the hard budget
        // stops a running iteration mid-flight via CheckTime. (Best-move-stability / predictive scaling was
        // tried but over-conserved time without SPRT tuning — reverted; MoveOverhead is kept below.)
        if w.IsMain && w.Control.SoftTimeUp then
            w.Control.Stop()

// ---------------------------------------------------------------------------
// Root-move parallelism (Phase 3): the main worker searches the first (best) root
// move with a full PVS window and pushes the rest to a shared DagWorkQueue. All
// workers pop root-move keys and search the corresponding subtree with a null PVS
// window. The TT carries results back to the main worker's collection phase, which
// re-searches any fail-high moves with a full window. A Barrier synchronises the
// three phases per depth: (1) main pushes, (2) all search, (3) main collects.
// ---------------------------------------------------------------------------
let iterativeDeepeningRootPar (w: Worker) (maxDepth: int) (barrier: Threading.Barrier)
                              (rootMoves: Move[]) (rootKeys: uint64[]) =
    let cfg = w.Control.Config
    let queue = w.Control.WorkQueue
    let mutable prev = evalPos w w.Pos
    let mutable depth = 1

    while depth <= maxDepth && not w.Control.Stopped do
        let mutable bestMove = rootMoves.[0]
        let mutable bestScore = -INF
        let mutable alpha = -INF

        if w.IsMain then
            // Phase 1: search the first (best) root move with the aspiration window (same loop as ID).
            let mutable delta = if cfg.UseAspTweaks then 10 + prev * prev / 15000 else 16
            let fullWindow = depth <= 4 || abs prev >= MATE_IN_MAX_PLY
            let mutable aspAlpha = if fullWindow then -INF else max (-INF) (prev - delta)
            let mutable aspBeta = if fullWindow then INF else min INF (prev + delta)
            let mutable score = 0
            let mutable searching = true
            let mutable attemptDepth = depth

            while searching do
                w.Pos.Make rootMoves.[0]
                score <- -(negamax w w.Pos (-aspBeta) (-aspAlpha) (max 1 attemptDepth) 1 true false)
                w.Pos.Unmake rootMoves.[0]

                if w.Control.Stopped then
                    searching <- false
                elif score <= aspAlpha then
                    aspBeta <- (aspAlpha + aspBeta) / 2
                    aspAlpha <- max (-INF) (score - delta)
                    delta <- delta * 2
                    attemptDepth <- depth
                elif score >= aspBeta then
                    aspBeta <- min INF (score + delta)
                    delta <- delta * 2
                    if cfg.UseAspTweaks && attemptDepth > depth - 3 && abs score < MATE_IN_MAX_PLY then
                        attemptDepth <- attemptDepth - 1
                else
                    searching <- false

            if not w.Control.Stopped then
                bestScore <- score
                bestMove <- rootMoves.[0]
                alpha <- score
                prev <- score
                w.RootScore <- score
                // NOT w.Pv.[0]: root-par searches at ply 1, so PV row 0 is never written — reading it
                // left RootBest = MoveNone whenever the incumbent stayed best, and go() then fell back
                // to the first generated legal move (see the LazySmpTests regression test).
                w.RootBest <- rootMoves.[0]
                w.CompletedDepth <- depth
                w.Control.SetSharedRootAlpha alpha

                // Phase 2: push remaining root-move keys to the work queue for helpers (and self).
                for i in 1 .. rootMoves.Length - 1 do
                    queue.TryPush rootKeys.[i] |> ignore

        // Barrier 1: main has pushed (or stopped); all workers proceed to pop+search.
        barrier.SignalAndWait()

        // Phase 3: all workers pop root-move keys and search with a null PVS window.
        if not w.Control.Stopped then
            let mutable searching = true

            while searching && not w.Control.Stopped do
                match queue.TryPop() with
                | Some key ->
                    let mutable idx = -1
                    let mutable i = 1

                    while idx < 0 && i < rootMoves.Length do
                        if rootKeys.[i] = key then idx <- i
                        i <- i + 1

                    if idx > 0 then
                        let nullAlpha = w.Control.SharedRootAlpha
                        w.Pos.Make rootMoves.[idx]
                        let _ = negamax w w.Pos (-nullAlpha - 1) (-nullAlpha) (max 1 (depth - 1)) 1 false true
                        w.Pos.Unmake rootMoves.[idx]
                | None ->
                    searching <- false

        // Barrier 2: all workers done searching; main proceeds to collect.
        barrier.SignalAndWait()

        if w.IsMain then
            // Drain leftovers (a worker stopped mid-pop leaves items; safe to discard — TT has partial data).
            while (queue.TryPop() |> Option.isSome) do ()

            if not w.Control.Stopped then
                // Phase 4: collect results from TT, re-search fail-highs with full window.
                for i in 1 .. rootMoves.Length - 1 do
                    let struct (hit, _, sc, _, dp, bd, _) = w.Control.Tt.Probe rootKeys.[i]

                    if hit && dp >= depth - 1 then
                        // The entry belongs to the position AFTER rootMoves.[i] (opponent to move): its
                        // score is from the OPPONENT's perspective, so the root move is worth -score —
                        // and the root move failing HIGH is the child failing LOW (BoundUpper).
                        let score = -(valueFromTt sc 1)

                        if (bd = BoundUpper || bd = BoundExact) && score > alpha then
                            w.Pos.Make rootMoves.[i]
                            let v = -(negamax w w.Pos (-INF) (-alpha) (max 1 (depth - 1)) 1 true false)
                            w.Pos.Unmake rootMoves.[i]

                            if not w.Control.Stopped && v > alpha then
                                alpha <- v
                                bestScore <- v
                                bestMove <- rootMoves.[i]
                                w.RootScore <- v
                                w.RootBest <- rootMoves.[i]
                                w.Control.SetSharedRootAlpha alpha

                w.CompletedDepth <- depth
                reportInfo w depth bestScore

                if abs bestScore >= MATE_IN_MAX_PLY then
                    w.Control.Stop()

        // Barrier 3: ready for next depth (helpers wait while main collects + reorders).
        barrier.SignalAndWait()

        depth <- depth + 1

        // Reorder: move the best root move to [0] so the next iteration's PVS first-move is the strongest.
        if w.IsMain && bestMove <> rootMoves.[0] && bestMove <> MoveNone then
            let idx = Array.IndexOf(rootMoves, bestMove)

            if idx > 0 then
                let tmpM = rootMoves.[0]
                rootMoves.[0] <- rootMoves.[idx]
                rootMoves.[idx] <- tmpM
                let tmpK = rootKeys.[0]
                rootKeys.[0] <- rootKeys.[idx]
                rootKeys.[idx] <- tmpK

        if w.IsMain && w.Control.SoftTimeUp then
            w.Control.Stop()

// ---------------------------------------------------------------------------
// Time budget (v1): movetime, or wtime/winc(+movestogo); depth/nodes/mate/infinite => no time stop.
// ---------------------------------------------------------------------------
let computeTimes (moveOverhead: int) (l: SearchLimits) (stm: Color) : int64 * int64 =
    let overhead = int64 (max 0 moveOverhead)

    if l.MoveTime > 0 then
        // Fixed move time: pure HARD limit (soft = 0 disables the soft/stability/predictive early-stop) so
        // the search uses the whole budget and CheckTime stops it mid-iteration at the deadline.
        (0L, max 1L (int64 l.MoveTime - overhead))
    elif l.Infinite || l.Depth > 0 || l.Nodes > 0L || l.Mate > 0 then
        (0L, 0L)
    else
        let time = if stm = White then l.WTime else l.BTime
        let inc = if stm = White then l.WInc else l.BInc

        if time <= 0 then
            (0L, 0L)
        else
            // Proven baseline budget + MoveOverhead: soft ~ (clock - overhead)/movestogo + 3/4*inc; hard caps
            // one move at 40% of the clock (or 4x soft). The optimum/maximum-scaling experiment over the
            // soft limit was reverted as an un-tuned regression (it spent only ~1/3 of the budget).
            let mtg = if l.MovesToGo > 0 then l.MovesToGo else 30
            let avail = max 1L (int64 time - overhead)
            let soft = avail / int64 mtg + int64 inc * 3L / 4L
            let hard = min (int64 (float time * 0.4)) (soft * 4L)
            (max 1L soft, max 1L hard)

// ---------------------------------------------------------------------------
// LazySMP orchestration. Returns the chosen best move (and prints `bestmove`).
// ---------------------------------------------------------------------------
let go (control: SearchControl) : Move =
    control.Reset()
    control.NewSearch()

    if PosProf.Enabled then
        PosProf.reset ()
    let n = max 1 control.Config.Threads
    let workers = Array.init n (fun i -> Worker(i, (i = 0), control))

    for w in workers do
        w.SetupRoot()
    // Live nps/nodes report the AGGREGATE over all workers (relaxed cross-thread reads, reporting only).
    control.NodeSum <-
        (fun () ->
            let mutable s = 0L

            for wk in workers do
                s <- s + wk.Nodes

            s)

    let stm = workers.[0].Pos.SideToMove

    let struct (soft, hard) =
        (let (a, b) = computeTimes control.Config.MoveOverhead control.Limits stm in struct (a, b))

    control.StartClockPonder soft hard control.Limits.Ponder

    let maxDepth =
        if control.Limits.Depth > 0 then
            min control.Limits.Depth (MaxSearchPly - 1)
        else
            (MaxSearchPly - 1)

    // Branch: root-move parallelism (Phase 3) vs classic LazySMP. MultiPV needs per-line root exclusion,
    // which the root-par phases don't support — force the classic path.
    let useRootPar =
        control.Config.UseWorkQueue
        && n > 1
        && control.WorkQueue <> null
        && control.Config.MultiPv <= 1

    if useRootPar then
        // Generate root moves + their resulting position keys from the main worker's Position.
        let p = NativePtr.stackalloc<Move> MaxMoves
        let buf = Span<Move>(NativePtr.toVoidPtr p, MaxMoves)
        let nMoves = generateLegal workers.[0].Pos buf

        let rootMoves: Move[] =
            if nMoves = 0 then
                [| MoveNone |]
            else
                buf.Slice(0, nMoves).ToArray()

        let rootKeys: uint64[] = Array.zeroCreate rootMoves.Length

        for i in 0 .. rootMoves.Length - 1 do
            workers.[0].Pos.Make rootMoves.[i]
            rootKeys.[i] <- workers.[0].Pos.Key
            workers.[0].Pos.Unmake rootMoves.[i]

        let barrier = new Barrier(n)

        let threads =
            [| for i in 1 .. n - 1 ->
                   let w = workers.[i]

                   let t =
                       Thread(
                           ThreadStart(fun () -> iterativeDeepeningRootPar w maxDepth barrier rootMoves rootKeys),
                           16 * 1024 * 1024
                       )

                   t.IsBackground <- true
                   t.Start()
                   t |]

        iterativeDeepeningRootPar workers.[0] maxDepth barrier rootMoves rootKeys
        control.Stop()

        for t in threads do
            t.Join()

        barrier.Dispose()
    else
        let threads =
            [| for i in 1 .. n - 1 ->
                   let w = workers.[i]

                   let t =
                       Thread(ThreadStart(fun () -> iterativeDeepening w maxDepth), 16 * 1024 * 1024)

                   t.IsBackground <- true
                   t.Start()
                   t |]

        iterativeDeepening workers.[0] maxDepth
        control.Stop()

        for t in threads do
            t.Join()

    // EONEGO_PROF=1: one-line phase breakdown (Threads=1 semantics; see PosProf doc). String concat, not
    // sprintf — Printf formatters crash under NativeAOT (see fsharp-dotnet-aot-gotchas).
    if PosProf.Enabled then
        let ms (t: int64) =
            (double t * 1000.0 / double System.Diagnostics.Stopwatch.Frequency).ToString("F0")

        writeLine (
            "info string prof makeMs=" + ms PosProf.tMake
            + " eagerMs=" + ms PosProf.tEager
            + " ensureMs=" + ms PosProf.tEnsure
            + " buildMs=" + ms PosProf.tBuild
            + " evalMs=" + ms PosProf.tEval
            + " nMake=" + string PosProf.nMake
            + " nEnsure=" + string PosProf.nEnsure
            + " nBuild=" + string PosProf.nBuild
            + " nEval=" + string PosProf.nEval
            + " maxThreatN=" + string PosProf.maxThreatN
        )

        // Ensure-bucket decomposition (all ⊂ ensureMs; enumMs ⊂ buildMs): walk applies vs dirty-threat
        // gathers vs refresh threat enumeration, with call counts for per-op costs.
        writeLine (
            "info string prof2 applyMs=" + ms PosProf.tApply
            + " gatherMs=" + ms PosProf.tGather
            + " enumMs=" + ms PosProf.tEnumThreats
            + " nApply=" + string PosProf.nApply
            + " nGather=" + string PosProf.nGather
            + " nEnum=" + string PosProf.nEnumThreats
        )

        writeLine ("info string prof3 align " + workers.[0].Pos.ProfAlignReport())

    let rb = workers.[0].RootBest

    let best =
        if isLegalRoot workers.[0].Pos rb then
            rb
        else
            firstLegalMove workers.[0].Pos

    control.LastBest <- best
    control.LastScore <- workers.[0].RootScore

    if workers.[0].CompletedDepth > 0 then
        reportInfo workers.[0] workers.[0].CompletedDepth workers.[0].RootScore

    writeLine ("bestmove " + toUci best)
    best

// ---------------------------------------------------------------------------
// Test entry: a single fixed-depth, FULL-WINDOW negamax (bypasses aspiration/ID). The correctness oracle.
// ---------------------------------------------------------------------------
let searchToDepthNet (fen: string) (rootMoves: Move[]) (depth: int) (cfg: SearchConfig) (net: Network option) : struct (int * int64 * Move) =
    let tt = TranspositionTable(max 1 cfg.HashMb)

    let control =
        SearchControl(cfg, { defaultLimits with Depth = depth }, tt, fen, rootMoves, ?net = net)

    control.Reset()
    control.NewSearch()
    let w = Worker(0, true, control)
    w.SetupRoot()
    control.StartClock 0L 0L
    let score = negamax w w.Pos (-INF) INF depth 0 true false
    struct (score, w.Nodes, w.Pv.[0])

/// Single-thread iterative deepening until the node budget is exhausted (no UCI stdout). Test/tooling entry.
let searchToNodesNet (fen: string) (rootMoves: Move[]) (nodes: int64) (cfg: SearchConfig) (net: Network option) : struct (int * int64 * Move) =
    let tt = TranspositionTable(max 1 cfg.HashMb)
    let limits = { defaultLimits with Nodes = nodes }

    let control =
        SearchControl(cfg, limits, tt, fen, rootMoves, ?net = net)

    control.Reset()
    control.NewSearch()
    let w = Worker(0, true, control)
    w.SetupRoot()
    control.NodeSum <- (fun () -> w.Nodes)
    control.StartClock 0L 0L
    iterativeDeepening w (MaxSearchPly - 1)
    control.Stop()

    let best =
        if isLegalRoot w.Pos w.RootBest then w.RootBest else firstLegalMove w.Pos

    struct (w.RootScore, w.Nodes, best)

let searchToDepth (fen: string) (rootMoves: Move[]) (depth: int) (cfg: SearchConfig) : struct (int * int64 * Move) =
    searchToDepthNet fen rootMoves depth cfg None

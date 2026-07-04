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

    // Tunables is compiled first, so its statics are initialized before this table builds.
    let div = float Tunables.LmrDiv100 / 100.0
    let off = float Tunables.LmrOff100 / 100.0

    for d in 1..63 do
        for m in 1..63 do
            t.[d * 64 + m] <- int (off + log (float d) * log (float m) / div)

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
      // Stack.StaticEval, `improving`, move-loop futility and every TT store keep the RAW eval
      // (the unadjusted static-eval contract; correction history depends on it staying raw).
      UseTtEvalAdjust: bool
      // Extend every SEE>=0 checking move by one ply (legacy behaviour). The reference engine dropped its
      // generic check extension years ago — the singular machinery and LMR cover the useful cases, and the
      // unconditional +1 inflates every subtree containing a safe check. OFF by default (2026-07-02 A/B);
      // the flag preserves the legacy arm for SPRT.
      UseCheckExt: bool
      // In-check qsearch: once one evasion has produced a non-mated score, skip the remaining QUIET
      // evasions (captures still searched; the mate-detection path needs movesPlayed >= 1 and the cap
      // can only fire after a move was searched, so it never masks a mate). Default OFF: measured a
      // +35% suite-node REGRESSION at d14/d15 (2026-07-02) — the pessimistic leaf values it injects
      // cost more tree than the skipped evasions save. Kept behind the flag for a future SPRT only.
      UseQsEvasionCap: bool
      // Correction history: a per-(stm, pawn-structure) gravity table of persistent (bestValue −
      // staticEval) error corrects the WORKING static eval wherever it feeds pruning (negamax step 3 +
      // qsearch stand-pat). Every TT store keeps the RAW eval. Adjudicated by self-play match — node
      // counts cannot judge an eval-accuracy feature.
      UseCorrHist: bool
      // Minor-piece correction history rider (needs UseCorrHist): a second correction table keyed by
      // Position.MinorKey (knights+bishops+kings), summed with the pawn term before the /CorrApplyDiv
      // scale and taught the same clamped error at the same update gate.
      UseCorrMinor: bool
      // Capture futility: at shallow reduced depth, skip a non-checking capture when even banking the
      // captured piece's full value cannot lift the static eval to alpha (capture analogue of the
      // quiet futility gate; promotions excluded — their material swing isn't in capturedValue).
      UseCaptFut: bool
      // Partial-iteration commit (classic path, timed games): on a hard stop mid-iteration, adopt the
      // interrupted iteration's best fully-searched root move instead of discarding all its progress.
      UsePartialCommit: bool
      // Weighted ss-4 continuation history: a cont4 table taught alongside cont1/cont2 and read at
      // 1/Cont4Div weight in the LMR history term only (NOT move ordering — the naive full-weight
      // ordering variant measured +25-42% suite nodes and was reverted 2026-07-02).
      UseCont4: bool
      // Rule-50 shuffle damping (evaluate() wrapper term): static eval decays linearly with the
      // halfmove counter (eval -= eval*rule50/Rule50DampDiv), so fortresses/shuffling drift toward
      // the draw score instead of holding full value until the search's rule-50 horizon. Identity at
      // rule50=0; perturbs ALL search trees (deep nodes accumulate counter), hence config-gated.
      UseR50Damp: bool
      // Quiet CHECKING moves at the FIRST qsearch ply (DEPTH_QS_CHECKS): the captures-only
      // horizon is blind to mating attacks that start with a quiet check. SEE-losing checks skipped;
      // deeper qsearch plies stay captures-only.
      UseQsChecks: bool
      // Root effort ordering: between iterations, root moves are re-sorted by the node
      // count their subtrees consumed last iteration (best move stays first). A move producing
      // expensive near-misses rises in the order and sheds reduction — the escape hatch for a slow
      // win buried under its own failed-scout history (the b3-b4 fixture pathology). Classic
      // single-PV path only.
      UseRootEffort: bool
      // Root re-verification on stagnation: when the root score is flat (see Tunables.RootVerify*),
      // each iteration gives ONE rotating non-best root move a full-window unreduced PV search —
      // the fresh look that pierces stale TT bounds a null-window scout can never re-open. Classic
      // single-PV path only; free when the score is moving.
      UseRootVerify: bool
      // Retrograde search: probe on-demand-solved 3-man endings at negamax/qsearch entry for exact
      // DTM scores (Retrograde.fs; signatures solve in the background once a low-material root
      // appears). Exactness, not pruning — deliberately independent of UsePruning; the pruning-off
      // oracle configs must pin this false themselves.
      UseRetro: bool
      MoveOverhead: int
      // Phase 1: NNUE accumulator checkpoint cache. Set to 0 to disable; ~4 MiB is the recommended default
      // (1024 slots, ~4.1 KiB/slot). Cleared per search by `SearchControl.NewSearch` (alongside the TT gen
      // bump). Probes are best-effort lock-free (matching the TT); populates on each successful
      // `Position.EnsureBothComputed` materialization.
      AccCheckpointMb: int
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
      Ponder: bool
      // UCI `go searchmoves ...`: restrict the root to these (already legality-stamped) moves.
      // Empty = no restriction. Analysis feature — lets a GUI force full attention onto one candidate.
      SearchMoves: Move[] }

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
      UseCheckExt = false
      UseQsEvasionCap = false
      UseCorrHist = true
      UseCorrMinor = false
      UseCaptFut = false
      UsePartialCommit = false
      UseCont4 = false
      UseR50Damp = true
      UseQsChecks = false
      UseRootEffort = false
      UseRootVerify = false
      UseRetro = true
      MoveOverhead = 10
      AccCheckpointMb = 0
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
      Ponder = false
      SearchMoves = [||] }

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
    member _.Config = config
    member _.Limits = limits
    member _.Tt = tt
    member _.RootFen = rootFen
    member _.RootMoves = rootMoves
    member _.Net: Network option = net
    /// Borrowed reference to the per-search accumulator checkpoint cache; `null` when disabled in config.
    member _.AccCheckpoint: AccCheckpointTable = accCheckpoint
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
    // Rebindable (worker pool): goPooled points a reused worker at the new search's control between
    // searches (single-threaded there — the UCI driver stops+joins any running search first). Every
    // reader goes through this one mutable field, so nothing can cache a stale control.
    let mutable control = control
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
    // Root effort ordering (UseRootEffort): persistent root move list + per-move subtree node counts,
    // re-sorted by iterativeDeepening between iterations. Inactive (RootListActive=false) => the root
    // iterates the normal staged picker, byte-identical to legacy.
    let rootMv: Move[] = Array.zeroCreate MaxMoves
    let rootNodes: int64[] = Array.zeroCreate MaxMoves
    let mutable rootCnt = 0
    let mutable rootListActive = false
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

    /// Partial-iteration commit (EONEGO_PARTIAL=1): the best root move that raised alpha during the
    /// CURRENT (possibly interrupted) iteration, with its score. Reset to MoveNone at each iteration
    /// start; a hard-stopped iteration's progress is otherwise discarded wholesale (up to hard≈3×soft
    /// of clock). Written at ply 0 only, never during an exclusion re-search or a MultiPV side line.
    member val IterBest = MoveNone with get, set
    member val IterScore = 0 with get, set

    member _.RootMv = rootMv
    member _.RootNodes = rootNodes

    member _.RootCnt
        with get () = rootCnt
        and set v = rootCnt <- v

    member _.RootListActive
        with get () = rootListActive
        and set v = rootListActive <- v

    /// Root re-verification (UseRootVerify): the ONE root move this iteration searches full-window,
    /// unreduced and PV-flagged even when late in the order. MoveNone = inactive.
    member val RootVerifyMove = MoveNone with get, set

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

    /// Worker pool: point this worker at a new search's control. Call strictly between searches
    /// (no live search thread), then SetupRoot.
    member _.Rebind(c: SearchControl) = control <- c

    /// Rebuild this worker's Position from the immutable root (FEN + replayed moves) and reset per-search
    /// state. keepHistory=true (worker pool): killers + counter-moves are cleared but the gravity history
    /// tables (butterfly/capture/cont/corr) persist across searches — warm ordering from move 2 on.
    member this.SetupRoot(?keepHistory: bool) =
        pos.LoadFen control.RootFen

        // Replay the game history BEFORE binding NNUE: with the net bound, every Make pushes an
        // accumulator frame, and a `position ... moves` list longer than AccMaxPly (256) overflowed the
        // frame stack (real crash: long games in a GUI / the match harness). Binding after the replay
        // rebases the accumulator stack at the search root (EnableNnue materializes frame 0 from the
        // current board), so search depth alone bounds frame usage.
        for m in control.RootMoves do
            pos.Make m

        match control.Net with
        | Some net -> Nnue.bindNnue net pos
        | None -> ()

        // Phase 1: borrow the per-search checkpoint cache so `EnsureBothComputed` can probe/store snapshots
        // during the upcoming search. `null` disables the fast-path entirely. Seed the root snapshot now —
        // `EnableNnue` just materialized frame 0 and set the computed flags, so the early-return path inside
        // `EnsureBothComputed` would otherwise skip the populate on the first eval at the root.
        pos.BindCheckpoint control.AccCheckpoint
        pos.SeedCheckpoint()

        if defaultArg keepHistory false then
            tables.NewSearch()
        else
            tables.Clear()

        nodes <- 0L
        selDepth <- 0
        rootBest <- MoveNone
        rootScore <- 0
        completedDepth <- 0
        stopSeen <- false
        nRootExcluded <- 0
        rootCnt <- 0
        rootListActive <- false
        this.RootVerifyMove <- MoveNone

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
        let v =
            if PosProf.Enabled then
                let profT0 = System.Diagnostics.Stopwatch.GetTimestamp()
                let v = Nnue.evalCp net pos
                PosProf.tEval <- PosProf.tEval + (System.Diagnostics.Stopwatch.GetTimestamp() - profT0)
                PosProf.nEval <- PosProf.nEval + 1L
                v
            else
                Nnue.evalCp net pos
        // Rule-50 shuffle damping (see SearchConfig.UseR50Damp): identity at rule50 = 0, and damping
        // only shrinks magnitude, so the EvalMax clamp inside evalCp still bounds the result.
        if w.Control.Config.UseR50Damp then
            v - v * pos.Rule50 / Tunables.Rule50DampDiv
        else
            v
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
// Retrograde probe -> search score. VALUE_NONE = no usable hit (signature unsolved, guard failed).
// A win/loss is trusted only when the mate fits the rule-50 budget (Q/R optimal lines never reset
// the counter; conservative for pawn signatures — they fall through and the search copes) AND stays
// inside the mate band: ply + dtm <= MaxSearchPly keeps every returned score >= MATE_IN_MAX_PLY,
// preserving valueToTt/valueFromTt ply adjustment, `mate N` reporting, and the singular ttScore
// gate. Draws are unconditionally safe. Exact-score contract: stm winning with dtm d at ply p means
// the mate position scores -MATE + (p+d) with the loser to move — backed up here as MATE - (p+d),
// bit-compatible with what a deep search would return.
// ---------------------------------------------------------------------------
let retroScoreAt (pos: Position) (ply: int) : int =
    match Eonego.Retrograde.probe pos with
    | ValueNone -> VALUE_NONE
    | ValueSome v ->
        if v = 0y then
            0
        else
            let dtm = abs (int v) - 1

            if dtm > 100 - pos.Rule50 || ply + dtm > MaxSearchPly then VALUE_NONE
            elif v > 0y then MATE - (ply + dtm)
            else -MATE + (ply + dtm)

// ---------------------------------------------------------------------------
// Quiescence (fail-soft leaves). `qsDepth` counts plies BELOW the qsearch entry (0 at the negamax
// boundary, negative deeper) — it exists solely so UseQsChecks can restrict quiet checking moves to
// the first layer; captures/evasions ignore it.
// ---------------------------------------------------------------------------
let rec qsearch (w: Worker) (pos: Position) (alphaIn: int) (betaIn: int) (ply: int) (qsDepth: int) : int =
    w.Nodes <- w.Nodes + 1L

    let stopNow = pollStop w

    if ply > w.SelDepth then
        w.SelDepth <- ply

    // Retrograde search: exact DTM score once the background solver has published this signature.
    // ply > 0 (the root needs a move, and one test drives qsearch at ply 0); path draws (repetition/
    // rule-50, the arm above) outrank the retro value. qsearch has no excluded-move state to guard.
    let retroScore =
        if
            w.Control.Config.UseRetro
            && ply > 0
            && popCount pos.Occupied <= 3
            && pos.CastlingRights = 0
        then
            retroScoreAt pos ply
        else
            VALUE_NONE

    if ply > 0 && stopNow then
        0
    elif ply > 0 && isImmediateDraw pos then
        0
    elif retroScore <> VALUE_NONE then
        retroScore
    // Acc-frame guard on pos.Top (search frames since the root rebase), NOT StPly: StPly includes the
    // GAME history, so after ~255 game plies the old guard bailed to a raw eval at every node.
    elif ply >= MaxSearchPly || pos.Top >= Position.AccStackLimit then
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

            // Correction history: same corrected-working-eval rule as negamax step 3 (rawEval stays raw
            // for the TT stores below).
            let rawSp =
                if usePruning && cfg.UseCorrHist then
                    let corrSum =
                        w.Tables.CorrHist pos.SideToMove pos.PawnKey
                        + (if cfg.UseCorrMinor then
                               w.Tables.CorrHistMinor pos.SideToMove pos.MinorKey
                           else
                               0)

                    rawSp + corrSum / Tunables.CorrApplyDiv
                else
                    rawSp

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
                    && (if inCheck then
                            // Quiet-evasion cap: `best` only rises above the mated band after a searched
                            // evasion, so this can never leave a node move-less (see UseQsEvasionCap doc).
                            cfg.UseQsEvasionCap
                            && best > -MATE_IN_MAX_PLY
                            && pos.PieceOn(toSq m) = NoPiece
                            && not (isEnPassant m)
                            && not (isPromotion m)
                        else
                            not (isPromotion m)
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

                                            rawEval + Tunables.QsDeltaBase + capturedValue <= alpha)))))

                if legal && not prune then
                    movesPlayed <- movesPlayed + 1
                    pos.Make m
                    w.Control.Tt.Prefetch pos.Key // child probes this exact key at entry
                    let v = -(qsearch w pos (-beta) (-alpha) (ply + 1) (qsDepth - 1))
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

            // Quiet CHECKING moves at the first qsearch ply only (UseQsChecks, DEPTH_QS_CHECKS):
            // after the capture picker drains, try quiet moves that give check and don't lose material
            // by SEE (spite checks explode the tree for nothing). The child is an evasion node, so its
            // replies are searched exhaustively — one extra move class at exactly one layer. The 1 KB
            // stackalloc is bounded: at most one qsDepth=0 frame exists per call-stack branch.
            if usePruning && cfg.UseQsChecks && not cutoff && not inCheck && qsDepth = 0 then
                let qcPtr = NativePtr.stackalloc<Move> MaxMoves
                let qcBuf = Span<Move>(NativePtr.toVoidPtr qcPtr, MaxMoves)
                let nLegal = generateLegal pos qcBuf
                let mutable i = 0

                while i < nLegal && not cutoff do
                    let qm = qcBuf.[i]

                    let isQuiet =
                        (pos.PieceOn(toSq qm) = NoPiece)
                        && not (isEnPassant qm)
                        && not (isPromotion qm)

                    if isQuiet && pos.GivesCheck qm && pos.SeeGe qm 0 then
                        pos.Make qm
                        w.Control.Tt.Prefetch pos.Key
                        let v = -(qsearch w pos (-beta) (-alpha) (ply + 1) (qsDepth - 1))
                        pos.Unmake qm

                        if pollStop w then
                            cutoff <- true
                        else
                            if v > best then
                                best <- v
                                bestMove <- qm

                                if v > alpha then
                                    alpha <- v

                            if alpha >= beta then
                                cutoff <- true

                    i <- i + 1

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

    // Retrograde search: exact DTM score once the background solver has published this signature.
    // Guards: ply > 0 (the root needs a move — children return exact scores, so depth-1 iterative
    // deepening already plays DTM-optimally); no live castling rights (a 3-man position with O-O
    // available is real and un-modeled); no excluded move (a singular-verification node must be
    // searched WITHOUT the excluded move — a retro value assumes all moves available and would
    // corrupt the singularity test). Path draws (repetition/rule-50) outrank the retro value.
    let retroScore =
        if
            w.Control.Config.UseRetro
            && ply > 0
            && popCount pos.Occupied <= 3
            && pos.CastlingRights = 0
            && w.Stack.[ply + StackOffset].ExcludedMove = MoveNone
        then
            retroScoreAt pos ply
        else
            VALUE_NONE

    if ply > 0 && stopNow then
        0
    elif ply > 0 && isImmediateDraw pos then
        0
    elif retroScore <> VALUE_NONE then
        retroScore
    // Acc-frame guard on pos.Top (search frames since the root rebase), NOT StPly: StPly includes the
    // GAME history, so after ~255 game plies the old guard bailed to a raw eval at every node.
    elif ply >= MaxSearchPly || pos.Top >= Position.AccStackLimit then
        (if pos.InCheck then 0 else evalPos w pos)
    elif depthIn <= 0 then
        qsearch w pos alphaIn betaIn ply 0
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

        // 3. static eval (+ "improving": static eval higher than 2 plies ago — hoisted up so ProbCut/pruning
        //    can read it; prunes less when our position is improving). `staticEval` is the CORRECTED value
        //    (correction history) consumed by Stack/improving/pruning; `rawStaticEval` is what the TT store
        //    persists (the unadjusted contract — the ttEval-reuse path above depends on it staying raw).
        let mutable staticEval = VALUE_NONE
        let mutable rawStaticEval = VALUE_NONE
        let mutable improving = false

        if not produced then
            rawStaticEval <-
                if inCheck then VALUE_NONE
                elif ttHit && ttEval <> VALUE_NONE then ttEval
                else evalPos w pos

            staticEval <-
                if rawStaticEval <> VALUE_NONE && usePruning && cfg.UseCorrHist then
                    let corrSum =
                        w.Tables.CorrHist pos.SideToMove pos.PawnKey
                        + (if cfg.UseCorrMinor then
                               w.Tables.CorrHistMinor pos.SideToMove pos.MinorKey
                           else
                               0)

                    rawStaticEval + corrSum / Tunables.CorrApplyDiv
                else
                    rawStaticEval

            w.Stack.[ssCur].StaticEval <- staticEval

            improving <-
                (not inCheck)
                && (let prev2 = w.Stack.[ssCur - 2].StaticEval in
                    prev2 = VALUE_NONE || staticEval > prev2)

        // 3b. TT score as a better WORKING eval: a bound-consistent search score is tighter than the
        //     heuristic static eval, so RFP/razoring/NMP below prune against it. Only this local —
        //     Stack.StaticEval (already stored raw above), `improving`, the move-loop futility gate and
        //     the TT store all keep the RAW eval (the unadjusted static-eval contract).
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
                && workingEval - Tunables.RfpMargin * depthIn - (if ttPv then Tunables.RfpTtPvBonus else 0) >= beta
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
                && workingEval + (Tunables.RazorBase + Tunables.RazorSlope * depthIn) <= alpha
            then
                let v = qsearch w pos alpha (alpha + 1) ply 0

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
                let r =
                    Tunables.NmpBase
                    + depthIn / Tunables.NmpDepthDiv
                    + (if workingEval - beta > Tunables.NmpEvalMargin then 1 else 0)
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
            let probCutBeta =
                beta + Tunables.ProbCutMargin - (if improving then Tunables.ProbCutImproving else 0)
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
                        let mutable v = -(qsearch w pos (-probCutBeta) (-probCutBeta + 1) (ply + 1) 0)

                        if v >= probCutBeta then
                            v <- -(negamax w pos (-probCutBeta) (-probCutBeta + 1) (depthIn - 4) (ply + 1) false (not cutNode))

                        pos.Unmake pcMove

                        if not (pollStop w) && v >= probCutBeta then
                            // Trust a sufficient existing TT entry over clobbering it; else store + return v.
                            if ttHit && ttDepth >= depthIn - 3 && ttScore <> VALUE_NONE && ttScore >= probCutBeta then
                                result <- ttScore
                            else
                                if useTt then
                                    // rawStaticEval, NOT staticEval: every TT store persists the RAW eval (the
                                    // contract at step 3 — the ttEval-reuse path re-applies correction history,
                                    // so storing the corrected value here double-corrected on re-probe).
                                    w.Control.Tt.Store pos.Key (depthIn - 3) BoundLower (valueToTt v ply) rawStaticEval pcMove false

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

            // ss-4 continuation key (cont4 rider): ssCur - 4 = ply + StackOffset - 4 >= 0 always
            // (StackOffset = 4), and the zeroed below-root slots fail the MoveNone guard.
            let struct (prev4Pc, prev4To) =
                if contOn && cfg.UseCont4 then
                    let prev4Move = w.Stack.[ssCur - 4].CurrentMove
                    let prev4Piece = w.Stack.[ssCur - 4].MovedPiece

                    if prev4Move <> MoveNone && prev4Move <> MoveNull && prev4Piece <> NoPiece then
                        struct (prev4Piece, toSq prev4Move)
                    else
                        struct (-1, -1)
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
            // Root effort ordering: when active, the root iterates the worker's persistent effort-sorted
            // list instead of the staged picker (exclusion re-searches keep the picker — they must not
            // touch the effort accounting). `go searchmoves` restriction is read once here too.
            let useRootList = ply = 0 && w.RootListActive && excludedMove = MoveNone
            let rootAllowed = if ply = 0 then w.Control.Limits.SearchMoves else [||]
            let mutable rootIdx = 0

            let mutable m =
                if useRootList then
                    (if w.RootCnt > 0 then w.RootMv.[0] else MoveNone)
                else
                    nextMove &mp skipQuiets

            while m <> MoveNone && not cutoff do
                let needsCheck =
                    isEnPassant m || fromSq m = ksq || testBit usBlockers (fromSq m)

                // MultiPV: at the root, skip the best moves of the lines already searched this iteration.
                // `go searchmoves`: skip root moves outside the requested set (empty = no restriction).
                let rootSkip =
                    ply = 0
                    && m <> MoveNone
                    && (w.IsRootExcluded m
                        || (rootAllowed.Length > 0 && not (System.Array.IndexOf(rootAllowed, m) >= 0)))

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
                                && moveCount >= (Tunables.LmpBase + depth * depth) / (if improving then 1 else 2)
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
                                && w.Tables.MainHistory us (fromTo m) < -Tunables.HistPruneBase - Tunables.HistPruneSlope * lmrDepth
                            then
                                doMove <- false
                            // futility — a quiet that can't lift alpha at shallow (reduced) depth
                            elif
                                (not givesCheck)
                                && lmrDepth <= 6
                                && staticEval + Tunables.FutBase + Tunables.FutSlope * lmrDepth <= alpha
                            then
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
                                        not see0Val && not (pos.SeeGe m (-Tunables.SeeQuietMult * lmrDepth * lmrDepth))
                                    else
                                        not (pos.SeeGe m (-Tunables.SeeQuietMult * lmrDepth * lmrDepth)))
                            then
                                doMove <- false
                        // capture futility — even banking the captured piece's full value cannot lift the
                        // static eval to alpha at shallow (reduced) depth. Promotions excluded (their
                        // material swing is not in capturedValue). Runs before the SEE gate: a compare is
                        // cheaper than an exchange walk.
                        elif
                            cfg.UseCaptFut
                            && (not givesCheck)
                            && not (isPromotion m)
                            && lmrDepth <= 6
                            && (let dst = pos.PieceOn(toSq m)

                                let capturedValue =
                                    if isEnPassant m then pieceValueOf Pawn
                                    elif dst = NoPiece then 0
                                    else pieceValueOf (pieceType dst)

                                staticEval + Tunables.CaptFutBase + Tunables.CaptFutSlope * lmrDepth + capturedValue <= alpha)
                        then
                            doMove <- false
                        elif (not givesCheck) && depth <= 6 && not (pos.SeeGe m (-Tunables.SeeCaptMult * depth)) then
                            // SEE — a capture that loses material by static exchange at shallow depth
                            doMove <- false

                    let mutable ext = 0

                    if doMove then
                        if usePruning && cfg.UseCheckExt && givesCheck then
                            let see0 = if see0Known then see0Val else pos.SeeGe m 0
                            ext <- if see0 then 1 else 0

                        // Singular / double extension: the TT move, at depth >= 8 with a depth-sufficient
                        // lower-bound TT score, is "singular" if every OTHER move fails low against a
                        // reduced exclusion window. Extend it (twice if it fails low by a clear margin).
                        // A fail-HIGH exclusion search is informative too (reference else-branches):
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
                            let singularBeta = ttScore - Tunables.SingularMul16 * depth / 16
                            let sDepth = (depth - 1) / 2
                            w.Stack.[ssCur].ExcludedMove <- m
                            let sv = negamax w pos (singularBeta - 1) singularBeta sDepth ply false cutNode
                            w.Stack.[ssCur].ExcludedMove <- MoveNone

                            if not (pollStop w) then
                                if sv < singularBeta then
                                    ext <- ext + (if (not isPv) && sv < singularBeta - Tunables.DoubleExtMargin then 2 else 1)
                                elif ply > 0 && sv >= beta && abs sv < MATE_IN_MAX_PLY then
                                    multicut <- true
                                    best <- sv
                                    cutoff <- true
                                elif ply > 0 && ttScore >= beta then
                                    ext <- ext - 2
                                elif ply > 0 && cutNode then
                                    ext <- ext - 2

                    let mutable newDepth = 0

                    if doMove && not multicut then
                        // Cap so check + double extensions can't push newDepth past depthIn + 1.
                        let maxExt = depthIn + 1 - (depth - 1)

                        if ext > maxExt then
                            ext <- max 0 maxExt

                        newDepth <- depth - 1 + ext
                        w.Stack.[ssCur].CurrentMove <- m
                        w.Stack.[ssCur].MovedPiece <- pos.PieceOn(fromSq m)

                        // Root effort: nodes consumed by this root move's subtree (read back after Unmake).
                        let effortStart = if useRootList then w.Nodes else 0L

                        pos.Make m
                        w.Control.Tt.Prefetch pos.Key // child probes this exact key at entry

                        if isQuiet && nQuiets < MaxMoves - 1 then
                            quietsBuf.[quietsBase + nQuiets] <- m
                            nQuiets <- nQuiets + 1

                        let mutable v = 0

                        // Root re-verification: the designated move gets the first-move treatment —
                        // full window, no reduction, PV-flagged — so its subtree is searched WITHOUT
                        // TT cutoffs on the PV spine (the fresh look stale bounds otherwise prevent).
                        let rootVerify =
                            ply = 0 && excludedMove = MoveNone && m = w.RootVerifyMove

                        if moveCount = 1 || rootVerify then
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
                                            + (if cfg.UseCont4 then
                                                   w.Tables.ContHistory4 prev4Pc prev4To pc (toSq m)
                                                   / Tunables.Cont4Div
                                               else
                                                   0)

                                        if hist < Tunables.LmrHistThresh then
                                            rr <- rr + 1

                                    rr
                                else
                                    0

                            // Root LMR cap (99 = off): root moves shed reduction so a buried slow win
                            // still gets a deep enough scout to surface (b3-b4 fixture pathology).
                            if ply = 0 && r > Tunables.RootLmrCap then
                                r <- Tunables.RootLmrCap

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

                        if useRootList then
                            w.RootNodes.[rootIdx] <- w.RootNodes.[rootIdx] + (w.Nodes - effortStart)

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

                                    // Partial-iteration commit bookkeeping: remember the root's best
                                    // fully-searched alpha-raiser so a hard-stopped iteration can still
                                    // contribute its progress (adopted in iterativeDeepening on stop).
                                    if ply = 0 && excludedMove = MoveNone && w.RootPvIdx = 0 then
                                        w.IterBest <- m
                                        w.IterScore <- v

                            if alpha >= beta then
                                cutoff <- true

                                if usePruning then
                                    let bonus = statBonus depth

                                    if isQuiet then
                                        let mPc = pos.PieceOn(fromSq m)
                                        w.Tables.UpdateMain us m bonus
                                        w.Tables.UpdateCont1 prev1Pc prev1To mPc (toSq m) bonus
                                        w.Tables.UpdateCont2 prev2Pc prev2To mPc (toSq m) bonus

                                        if cfg.UseCont4 then
                                            w.Tables.UpdateCont4 prev4Pc prev4To mPc (toSq m) bonus

                                        for qi in 0 .. nQuiets - 2 do
                                            let q = quietsBuf.[quietsBase + qi]
                                            let qPc = pos.PieceOn(fromSq q)
                                            w.Tables.UpdateMain us q (-bonus)
                                            w.Tables.UpdateCont1 prev1Pc prev1To qPc (toSq q) (-bonus)
                                            w.Tables.UpdateCont2 prev2Pc prev2To qPc (toSq q) (-bonus)

                                            if cfg.UseCont4 then
                                                w.Tables.UpdateCont4 prev4Pc prev4To qPc (toSq q) (-bonus)

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

                m <-
                    if cutoff then
                        MoveNone
                    elif useRootList then
                        rootIdx <- rootIdx + 1
                        if rootIdx < w.RootCnt then w.RootMv.[rootIdx] else MoveNone
                    else
                        nextMove &mp skipQuiets

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

                    w.Control.Tt.Store pos.Key depth bound (valueToTt best ply) rawStaticEval bestMove ttPv

                // Correction history: teach the table this pawn structure's persistent eval error. Only
                // when the result is eval-like (not in check, best move not a capture) and the bound
                // direction doesn't contradict the sign (standard gates): a fail-high below staticEval or a
                // fail-low above it carries no usable signal.
                if
                    usePruning
                    && cfg.UseCorrHist
                    && not multicut
                    && not inCheck
                    && excludedMove = MoveNone
                    && not (pollStop w)
                    && rawStaticEval <> VALUE_NONE
                    && (bestMove = MoveNone
                        || (pos.PieceOn(toSq bestMove) = NoPiece && not (isEnPassant bestMove)))
                    && not (best >= beta && best <= staticEval)
                    && not (bestMove = MoveNone && best >= staticEval)
                then
                    let bonus =
                        max -Tunables.CorrClamp (min Tunables.CorrClamp ((best - staticEval) * depth / Tunables.CorrDepthDiv))
                    w.Tables.UpdateCorr pos.SideToMove pos.PawnKey bonus

                    if cfg.UseCorrMinor then
                        w.Tables.UpdateCorrMinor pos.SideToMove pos.MinorKey bonus

                result <- best

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

    // A mate PV can end well short of the announced distance: the search stops deepening once the
    // mate is proven (clock-saving), qsearch keeps no PV rows, and mate-distance pruning returns at
    // PV nodes before any move once the window is mate-bounded — while the TT still holds the whole
    // proof. Extend the REPORTED line by replaying the array PV on the worker's root position and
    // following legal TT moves from the tail, capped at the exact mate distance. Cold reporting path:
    // main worker only, between searches, make/unmake balanced; every step legality-checked so a
    // stale or replaced TT entry can never print an illegal line (the walk just stops short instead).
    if i > 0 && abs score >= MATE_IN_MAX_PLY then
        let matePlies = MATE - abs score
        let pos = w.Pos
        let made: Move[] = Array.zeroCreate MaxSearchPly
        let mutable nMade = 0
        let mutable ok = true

        while ok && nMade < i do
            let mv = pv.[pvBase + nMade]

            if isLegalRoot pos mv then
                pos.Make mv
                made.[nMade] <- mv
                nMade <- nMade + 1
            else
                ok <- false

        while ok && nMade < matePlies && nMade < MaxSearchPly - 1 do
            let struct (hit, m, _, _, _, _, _) = w.Control.Tt.Probe pos.Key

            if hit && isLegalRoot pos m then
                sb.Append(' ').Append(toUci m) |> ignore
                pos.Make m
                made.[nMade] <- m
                nMade <- nMade + 1
            else
                ok <- false

        for j in nMade - 1 .. -1 .. 0 do
            pos.Unmake made.[j]

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

    // Root effort ordering (UseRootEffort, classic single-PV only): seed the worker's persistent root
    // list from legal generation; negamax's ply-0 loop iterates it and accounts per-move subtree
    // nodes; the block at the end of each completed iteration re-sorts it (best first, then by effort).
    let useRootList = cfg.UseRootEffort && multiPv = 1
    // Root re-verification (UseRootVerify) needs the same list (as a rotation pool) but NOT the
    // list-driven iteration — the picker stays in charge unless UseRootEffort is also on.
    let useRootVerify = cfg.UseRootVerify && multiPv = 1

    if useRootList || useRootVerify then
        w.RootCnt <- generateLegal w.Pos (Span<Move>(w.RootMv))
        Array.Clear(w.RootNodes, 0, w.RootNodes.Length)

    w.RootListActive <- useRootList

    // Stagnation detector state: ring of the last 6 completed-iteration scores + rotation cursor.
    // Each candidate is verified for 3 CONSECUTIVE iterations before rotating on: one fresh PV look
    // only re-opens its spine, while consecutive looks compound (each stores fresh deep bounds the
    // next look searches past) — needed to rebuild a subtree soaked in stale <=alpha bounds.
    let verifyHist = Array.create 6 0
    let mutable verifyHistN = 0
    let mutable verifyIdx = 0
    let mutable verifyStreak = 0

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

    // SMP depth-skipping was tried here (odd-id helpers striding by 2) and MEASURED WORSE
    // 2026-07-04: 16T/10s midgame depth d26 -> d23 — the striding helpers burn time on deep
    // iterations they can't finish while their width contribution (which fed the main thread's
    // TT) is lost. The reference engine's thread-to-depth conversion comes from tree efficiency,
    // not scheduling. Do not re-add without a tree-efficiency step-change first.
    let mutable depth = 1

    while depth <= maxDepth && not w.Control.Stopped do
        w.ClearRootExclusions()
        // Partial-iteration commit: forget the previous iteration's root progress marker — only a
        // move fully searched WITHIN the iteration a hard stop interrupts may be adopted below.
        w.IterBest <- MoveNone
        let mutable pvIdx = 0
        let mutable iterationOk = true

        while pvIdx < multiPv && iterationOk do
            w.RootPvIdx <- pvIdx
            let prev = prevScores.[pvIdx]
            // Tweaked: initial window scaled by score magnitude (tighter near equal scores). OFF => flat 16.
            let mutable delta =
                if cfg.UseAspTweaks then
                    Tunables.AspInitDelta + prev * prev / Tunables.AspSqDiv
                else
                    16

            // A3 diversity rider (EONEGO_T_HELPER_ASP > 0): odd-id helpers search a wider initial
            // window, so LazySMP threads diverge by construction rather than only by TT races.
            if Tunables.HelperAspOffset > 0 && not w.IsMain && w.Id % 2 = 1 then
                delta <- delta + Tunables.HelperAspOffset

            let fullWindow = depth <= 4 || abs prev >= MATE_IN_MAX_PLY
            let mutable alpha = if fullWindow then -INF else max (-INF) (prev - delta)
            let mutable beta = if fullWindow then INF else min INF (prev + delta)
            let mutable score = 0
            let mutable searching = true
            // Tweaked: drop the re-search depth by up to 3 plies on consecutive fail-highs (reported depth
            // stays `depth`). OFF (or no fail-high) => attemptDepth = depth, i.e. the legacy behaviour.
            let mutable attemptDepth = depth
            // Fail-high arbitration (AspFailHighRed = 2): the pre-widen beta of the last fail-high, and
            // whether this iteration already spent its one full-depth arbitration re-search.
            let mutable lastFailHighBeta = System.Int32.MinValue
            let mutable arbitrated = false

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
                    lastFailHighBeta <- System.Int32.MinValue
                elif score >= beta then
                    if w.IsMain && w.Control.ElapsedMs > 3000L then
                        reportLine w depth (pvIdx + 1) score " lowerbound" w.Pv 0 w.RootBest

                    lastFailHighBeta <- beta
                    beta <- min INF (score + delta)
                    delta <- delta * 2

                    if
                        cfg.UseAspTweaks
                        && Tunables.AspFailHighRed >= 1
                        && attemptDepth > depth - 3
                        && abs score < MATE_IN_MAX_PLY
                    then
                        attemptDepth <- attemptDepth - 1
                elif
                    // Fail-high CONFIRMATION (mode 2): this iteration failed high and was re-searched
                    // at REDUCED depth; whatever in-window score that produced may have erased a deep
                    // fail-high signal (the b3-b4 suppression: the reduced look lands just above the
                    // old beta and the creep repeats forever). Confirm ONCE at full depth before
                    // accepting the iteration's result.
                    Tunables.AspFailHighRed = 2
                    && not arbitrated
                    && lastFailHighBeta <> System.Int32.MinValue
                    && attemptDepth < depth
                then
                    arbitrated <- true
                    attemptDepth <- depth
                    lastFailHighBeta <- System.Int32.MinValue
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

            // Root effort ordering: best move to the front, the rest by the nodes their subtrees
            // consumed this iteration (descending — expensive near-misses get searched earlier and
            // less reduced next iteration). Effort then resets so each iteration's sort reflects
            // fresh evidence at the new depth.
            if useRootList && w.RootCnt > 1 then
                let bi = System.Array.IndexOf(w.RootMv, w.RootBest, 0, w.RootCnt)

                if bi > 0 then
                    let tm = w.RootMv.[0] in w.RootMv.[0] <- w.RootMv.[bi]; w.RootMv.[bi] <- tm
                    let tn = w.RootNodes.[0] in w.RootNodes.[0] <- w.RootNodes.[bi]; w.RootNodes.[bi] <- tn

                // Insertion sort of [1..RootCnt-1] by effort desc (tiny N; stable).
                for i in 2 .. w.RootCnt - 1 do
                    let mv = w.RootMv.[i]
                    let nd = w.RootNodes.[i]
                    let mutable j = i - 1

                    while j >= 1 && w.RootNodes.[j] < nd do
                        w.RootMv.[j + 1] <- w.RootMv.[j]
                        w.RootNodes.[j + 1] <- w.RootNodes.[j]
                        j <- j - 1

                    w.RootMv.[j + 1] <- mv
                    w.RootNodes.[j + 1] <- nd

                Array.Clear(w.RootNodes, 0, w.RootCnt)

            // Root re-verification: once the score has been flat across the last 6 completed
            // iterations at sufficient depth, designate the next rotating non-best root move for a
            // full-window PV look next iteration. Score moving again => detector re-arms, no cost.
            if useRootVerify then
                verifyHist.[verifyHistN % 6] <- lineScores.[0]
                verifyHistN <- verifyHistN + 1
                w.RootVerifyMove <- MoveNone

                if verifyHistN >= 6 && depth >= Tunables.RootVerifyDepth && w.RootCnt > 1 then
                    let mutable lo = System.Int32.MaxValue
                    let mutable hi = System.Int32.MinValue

                    for s in verifyHist do
                        lo <- min lo s
                        hi <- max hi s

                    if hi - lo <= Tunables.RootVerifyBand && abs lineScores.[0] < MATE_IN_MAX_PLY then
                        // Advance the rotation only every 3rd stagnant iteration; skip the current
                        // best move (it already gets the PV treatment).
                        if verifyStreak >= 3 then
                            verifyIdx <- verifyIdx + 1
                            verifyStreak <- 0

                        let mutable tries = 0

                        while tries < w.RootCnt && w.RootMv.[verifyIdx % w.RootCnt] = lineMoves.[0] do
                            verifyIdx <- verifyIdx + 1
                            tries <- tries + 1

                        w.RootVerifyMove <- w.RootMv.[verifyIdx % w.RootCnt]
                        verifyStreak <- verifyStreak + 1

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

    // Partial-iteration commit: the classic path otherwise discards ALL root progress of a hard-stopped
    // iteration (up to hard≈3×soft of clock burned for nothing). Adopt its best fully-searched root
    // improvement — updatePv already wrote PV row 0 for it, so this also re-aligns the reported PV with
    // RootBest. After a COMPLETED iteration IterBest equals lineMoves[0], making the adoption a no-op.
    // Fixed-depth sweeps never hard-stop mid-iteration with progress, so node counts are untouched.
    if cfg.UsePartialCommit && multiPv = 1 && w.IterBest <> MoveNone then
        w.RootBest <- w.IterBest
        w.RootScore <- w.IterScore

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
            // Constants live in Tunables (EONEGO_T_TM_*, defaults = the exact v1 values) so game-clock
            // matches can tune them without a rebuild.
            let mtg = if l.MovesToGo > 0 then l.MovesToGo else Tunables.TmMtg
            let avail = max 1L (int64 time - overhead)
            let soft = avail / int64 mtg + int64 inc * int64 Tunables.TmIncFrac100 / 100L
            let hard = min (int64 time * int64 Tunables.TmHardClockPct / 100L) (soft * int64 Tunables.TmHardSoftMult)
            (max 1L soft, max 1L hard)

// ---------------------------------------------------------------------------
// LazySMP orchestration. Returns the chosen best move (and prints `bestmove`).
// ---------------------------------------------------------------------------
/// Thread vote over the workers' published root results (classic LazySMP only). Returns the
/// index of the worker whose (move, score, depth) the search should report. Voters are workers that
/// completed at least one iteration; each contributes (score − minScore + 40) × depth to its move's
/// tally. Deterministic: ties prefer the first-encountered move, then the deepest voter for the winning
/// move, then the higher score, then the lowest worker index. Mate override: a voter holding a proven
/// mate wins outright (deepest proof = highest score) — consensus must not outvote a proof.
/// Pure over parallel arrays so tests can drive it without workers (cold path; runs once per `go`).
let internal voteBest (moves: Move[]) (scores: int[]) (depths: int[]) (n: int) : int =
    let inline isVoter i = depths.[i] > 0 && moves.[i] <> MoveNone

    let mutable mate = -1

    for i in 0 .. n - 1 do
        if isVoter i && scores.[i] >= MATE_IN_MAX_PLY && (mate < 0 || scores.[i] > scores.[mate]) then
            mate <- i

    if mate >= 0 then
        mate
    else
        let mutable minScore = System.Int32.MaxValue
        let mutable nVoters = 0

        for i in 0 .. n - 1 do
            if isVoter i then
                nVoters <- nVoters + 1
                minScore <- min minScore scores.[i]

        if nVoters = 0 then
            0
        else
            let inline voteOf (m: Move) =
                let mutable v = 0L

                for j in 0 .. n - 1 do
                    if isVoter j && moves.[j] = m then
                        v <- v + int64 (scores.[j] - minScore + 40) * int64 depths.[j]

                v

            let mutable win = -1
            let mutable winVote = System.Int64.MinValue

            for i in 0 .. n - 1 do
                if isVoter i then
                    let v = voteOf moves.[i]

                    if v > winVote then
                        winVote <- v
                        win <- i

            // Among voters for the winning move: deepest, then highest score, then lowest index
            // (ascending scan with strict > keeps the lowest index on full ties).
            let wm = moves.[win]
            let mutable chosen = win

            for i in 0 .. n - 1 do
                if isVoter i && moves.[i] = wm then
                    if
                        depths.[i] > depths.[chosen]
                        || (depths.[i] = depths.[chosen] && scores.[i] > scores.[chosen])
                    then
                        chosen <- i

            chosen

/// Shared orchestration body: `workers` are already created (or pool-rebound) and SetupRoot-ed by the
/// caller; everything from the clock start through the bestmove print lives here.
let private goCore (control: SearchControl) (workers: Worker[]) : Move =
    let n = workers.Length
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

    // Thread vote (single-PV only): helpers' completed-iteration results outvote the main worker when
    // a deeper/better-scoring consensus exists. MultiPV's bestmove must match the main worker's
    // reported line 1 — it keeps the legacy worker-0 selection. n = 1 bypass keeps the single-thread
    // path structurally identical (LazySmpTests determinism).
    let chosen =
        if n = 1 || control.Config.MultiPv > 1 then
            workers.[0]
        else
            let vMoves = Array.init n (fun i -> workers.[i].RootBest)
            let vScores = Array.init n (fun i -> workers.[i].RootScore)
            let vDepths = Array.init n (fun i -> workers.[i].CompletedDepth)
            workers.[voteBest vMoves vScores vDepths n]

    let rb = chosen.RootBest

    let best =
        if isLegalRoot workers.[0].Pos rb then
            rb
        else
            firstLegalMove workers.[0].Pos

    control.LastBest <- best
    control.LastScore <- chosen.RootScore

    if chosen.CompletedDepth > 0 then
        reportInfo chosen chosen.CompletedDepth chosen.RootScore

    writeLine ("bestmove " + toUci best)
    best

/// Arm a control for one search: clear the stop flag/counters and bump the TT age. The UCI driver
/// calls this on ITS OWN thread before spawning the search thread — go/goPooled running Reset on the
/// spawned thread let a `stop`/`quit` that raced in between `Thread.Start` and `Reset` be silently
/// ERASED (the search then ran unbounded) or, on the other side of the race, abort depth 1 into a
/// first-legal bestmove. Direct callers (tests, goArmed-less paths) use go/goPooled which arm inline.
let arm (control: SearchControl) : unit =
    control.Reset()
    control.NewSearch()

/// Search body: the control must already be armed (see `arm`).
let goArmed (control: SearchControl) : Move =
    if PosProf.Enabled then
        PosProf.reset ()

    let n = max 1 control.Config.Threads
    let workers = Array.init n (fun i -> Worker(i, (i = 0), control))

    for w in workers do
        w.SetupRoot()

    goCore control workers

let go (control: SearchControl) : Move =
    arm control
    goArmed control

/// Worker pool entry (UCI EONEGO_POOL=1): reuse per-thread Workers across `go` calls. Skips the
/// per-move worker allocation (~3.5MB × threads of zeroed LOH — measured 26ms median per go at 16T)
/// and keeps the gravity history tables warm across moves (killers/counters are re-cleared per
/// search by SetupRoot keepHistory). The caller owns the pool lifetime: recreate on Threads change,
/// drop on ucinewgame. Workers must be rebound here — the control is new every search.
/// The control must already be armed (see `arm`).
let goPooledArmed (control: SearchControl) (workers: Worker[]) : Move =
    if PosProf.Enabled then
        PosProf.reset ()

    for w in workers do
        w.Rebind control
        w.SetupRoot(keepHistory = true)

    goCore control workers

let goPooled (control: SearchControl) (workers: Worker[]) : Move =
    arm control
    goPooledArmed control workers

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

/// Eonego — alpha-beta / PVS search (Phase 3). LazySMP: N independent iterative-deepening searches, each
/// over its OWN Position + History.Tables + stack/PV/buffers/node counter. The ONLY shared writable state
/// is the lock-free TT plus the atomic stop flag (SearchControl); everything else is per-worker, rebuilt
/// from the immutable root. Fail-soft. With UsePruning=false the search is plain full-window alpha-beta
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
open Eonego.SfNnue
open Eonego.MoveGeneration
open Eonego.History
open Eonego.MovePick
open Eonego.Transposition

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
let private Reductions: int[,] =
    let t = Array2D.zeroCreate 64 64

    for d in 1..63 do
        for m in 1..63 do
            t.[d, m] <- int (0.5 + log (float d) * log (float m) / 2.2)

    t

let inline private reduction (depth: int) (moveCount: int) : int =
    if depth < 64 && moveCount < 64 then
        Reductions.[depth, moveCount]
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
      MoveOverhead: int
      UseMcts: bool // route `go` through the MCTS-root / negamax-leaf hybrid (Mcts.fs) instead of alpha-beta
      MctsCpuct: int // PUCT exploration constant, fixed-point ×100 (150 => 1.5)
      MctsLeafDepth: int // fixed negamax depth at each MCTS leaf (0 => qsearch-only leaf)
      MctsK: int // logistic cp->winprob scale (centipawns); match the training K
      UseLc0: bool // use the Lc0 CNN (Lc0Net.fs) for MCTS priors; requires a loaded Lc0 net (else history fallback)
      MctsBatchSize: int } // >1: batch B leaves/worker through one Lc0 forward (virtual-loss gather); 1: single eval

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
      MoveOverhead = 10
      UseMcts = false
      MctsCpuct = 150
      MctsLeafDepth = 6
      MctsK = 200
      UseLc0 = false
      MctsBatchSize = 1 }

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
    isRepetition pos
    || insufficientMaterial pos
    || (pos.Rule50 >= 100 && hasAnyLegalMove pos)

let firstLegalMove (pos: Position) : Move =
    let p = NativePtr.stackalloc<Move> MaxMoves
    let buf = Span<Move>(NativePtr.toVoidPtr p, MaxMoves)
    let n = generateLegal pos buf
    if n > 0 then buf.[0] else MoveNone

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
// Shared control: the ONLY cross-thread writable state (atomic stop flag) + clock + immutable inputs.
// ---------------------------------------------------------------------------
[<Sealed>]
type SearchControl
    (config: SearchConfig, limits: SearchLimits, tt: TranspositionTable, rootFen: string, rootMoves: Move[], ?net: SfNetwork, ?lc0Net: Lc0Proto.Lc0Net) =
    let mutable stopFlag = 0
    let sw = System.Diagnostics.Stopwatch()
    let mutable softMs = 0L
    let mutable hardMs = 0L
    let mutable baseSoftMs = 0L // the un-scaled optimum; the per-iteration manager rescales softMs from it
    let mutable ponderSoft = 0L // real budget remembered during a ponder search; armed by PonderHit
    let mutable ponderHard = 0L
    let mutable ponderHitPending = false // ponderhit that arrived before the search stored its budget
    member _.Config = config
    member _.Limits = limits
    member _.Tt = tt
    member _.RootFen = rootFen
    member _.RootMoves = rootMoves
    member _.Net: SfNetwork option = net
    member _.Lc0Net: Lc0Proto.Lc0Net option = lc0Net
    member val LastBest: Move = MoveNone with get, set // result of the most recent go()
    member val LastScore: int = 0 with get, set
    /// Aggregate live node count across all workers (relaxed reads — reporting only, set by go()).
    member val NodeSum: unit -> int64 = (fun () -> 0L) with get, set
    /// Aggregate completed MCTS iteration count across all workers (reporting only, set by mctsSearch()).
    /// UCI `info nodes`/`nps` reports this for MCTS; `go nodes N` limits still use NodeSum (leaf nodes).
    member val IterSum: unit -> int64 = (fun () -> 0L) with get, set
    member _.Stopped: bool = Volatile.Read(&stopFlag) <> 0
    member _.Stop() = Volatile.Write(&stopFlag, 1)
    member _.Reset() = Volatile.Write(&stopFlag, 0)
    member _.ElapsedMs: int64 = sw.ElapsedMilliseconds
    member _.SoftTimeUp: bool = softMs > 0L && sw.ElapsedMilliseconds >= softMs
    member _.BaseSoftMs: int64 = baseSoftMs
    member _.SetSoft(v: int64) = softMs <- v

    member _.StartClock (soft: int64) (hard: int64) =
        softMs <- soft
        baseSoftMs <- soft
        hardMs <- hard
        sw.Restart()

    /// Ponder: search unbounded now and remember the real budget; PonderHit arms it (the clock starts then,
    /// so the time used while pondering on the opponent's clock is free). If a ponderhit already raced ahead
    /// (arrived before this stored the budget), arm immediately here.
    member this.StartClockPonder (soft: int64) (hard: int64) (ponder: bool) =
        if ponder then
            ponderSoft <- soft
            ponderHard <- hard

            if ponderHitPending then
                this.StartClock soft hard
            else
                this.StartClock 0L 0L
        else
            this.StartClock soft hard

    /// The opponent played the predicted move: arm the remembered ponder budget. No-op for a non-ponder
    /// search (spurious ponderhit); if the budget is not stored yet, StartClockPonder arms it when it runs.
    member this.PonderHit() =
        if limits.Ponder then
            ponderHitPending <- true

            if ponderSoft > 0L || ponderHard > 0L then
                this.StartClock ponderSoft ponderHard

    /// Main worker only: convert a time/node budget overrun into the shared stop flag.
    member _.CheckTime(nodes: int64) =
        if
            (hardMs > 0L && sw.ElapsedMilliseconds >= hardMs)
            || (limits.Nodes > 0L && nodes >= limits.Nodes)
        then
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
    // Lc0 prior-output buffer (one per Worker, reused every MCTS expand).
    let lc0Priors: float32[]     = Array.zeroCreate 256
    // Lc0 CNN input-plane scratch (112*64); the forward pass uses the per-Worker Lc0Scratch below.
    let lc0InBuf: float32[]      = Array.zeroCreate (Lc0Encoder.Planes * 64)
    // Lc0 forward-pass scratch — ONE allocation per Worker, reused every expand (was per-call before). Built
    // only when an Lc0 net is loaded; null otherwise (the Lc0 expand branch is gated on the net's presence).
    let lc0Scratch: Lc0Net.Lc0Scratch =
        match control.Lc0Net with
        | Some n when control.Config.UseLc0 -> Lc0Net.Lc0Scratch(n)
        | _ -> Unchecked.defaultof<Lc0Net.Lc0Scratch>
    let mutable nodes = 0L
    let mutable iters = 0L
    let mutable selDepth = 0
    let mutable rootBest = MoveNone
    let mutable rootScore = 0
    let mutable completedDepth = 0
    let mutable nmpMinPly = 0 // NMP allowed only at ply >= this; set during an NMP-verification region (Feature 7)
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
    member _.Lc0Priors       = lc0Priors
    member _.Lc0InBuf        = lc0InBuf
    member _.Lc0Scratch      = lc0Scratch

    member _.Nodes
        with get () = nodes
        and set v = nodes <- v

    /// Completed MCTS iteration count for this worker. Incremented after each backed-up iteration.
    member _.Iters = iters
    member _.IncIters() = iters <- iters + 1L

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

    /// Rebuild this worker's Position from the immutable root (FEN + replayed moves) and reset per-search state.
    member _.SetupRoot() =
        pos.LoadFen control.RootFen

        match control.Net with
        | Some net -> SfNnue.bindSfNnue net pos
        | None -> ()

        for m in control.RootMoves do
            pos.Make m

        tables.Clear()
        nodes <- 0L
        iters <- 0L
        selDepth <- 0
        rootBest <- MoveNone
        rootScore <- 0
        completedDepth <- 0

let evalPos (w: Worker) (pos: Position) : int =
    match w.Control.Net with
    | Some net -> SfNnue.evalCp net pos
    | None -> 0 // unreachable in play: UCI refuses to search with no net (see Uci.startSearch)

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

    if w.IsMain && (w.Nodes &&& 2047L) = 0L then
        w.Control.CheckTime w.Nodes

    if ply > w.SelDepth then
        w.SelDepth <- ply

    if ply > 0 && w.Control.Stopped then
        0
    elif ply > 0 && isImmediateDraw pos then
        0
    elif ply >= MaxSearchPly then
        (if pos.InCheck then 0 else evalPos w pos)
    else
        let cfg = w.Control.Config
        let useTt = cfg.UseTt
        let usePruning = cfg.UsePruning
        let inCheck = pos.InCheck
        let mutable alpha = alphaIn
        let beta = betaIn
        let mutable ttMove = MoveNone

        if useTt then
            let struct (hit, m, _, _, _, _, _) = w.Control.Tt.Probe pos.Key

            if hit then
                ttMove <- m

        let mutable best = -INF
        let mutable bestMove = MoveNone
        let mutable rawEval = VALUE_NONE
        let mutable cutoff = false

        if not inCheck then
            let sp = evalPos w pos
            rawEval <- sp
            best <- sp

            if sp >= beta then
                cutoff <- true
            elif sp > alpha then
                alpha <- sp

        if cutoff then
            best
        else
            let us = pos.SideToMove
            let ksq = pos.KingSquare us
            let moves = w.MoveBuf.AsSpan(ply * MaxMoves, MaxMoves)
            let scores = w.ScoreBuf.AsSpan(ply * MaxMoves, MaxMoves)
            let mutable mp = mkQSearch pos w.Tables ttMove moves scores
            let mutable movesPlayed = 0
            let mutable m = nextMove &mp false

            while m <> MoveNone && not cutoff do
                let needsCheck =
                    isEnPassant m || fromSq m = ksq || testBit (pos.BlockersForKing us) (fromSq m)

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
                    let v = -(qsearch w pos (-beta) (-alpha) (ply + 1))
                    pos.Unmake m

                    if w.Control.Stopped then
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
                if useTt && not w.Control.Stopped then
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

    if w.IsMain && (w.Nodes &&& 2047L) = 0L then
        w.Control.CheckTime w.Nodes

    if ply > w.SelDepth then
        w.SelDepth <- ply
    // Terminate this PV row up-front so a parent's PV copy stops here when this node bottoms out into
    // qsearch / a draw / a TT cutoff without producing its own continuation (prevents stale PV tails).
    if isPv && ply < MaxSearchPly then
        w.Pv.[ply * MaxSearchPly] <- MoveNone

    if ply > 0 && w.Control.Stopped then
        0
    elif ply > 0 && isImmediateDraw pos then
        0
    elif depthIn <= 0 then
        qsearch w pos alphaIn betaIn ply
    elif ply >= MaxSearchPly then
        (if pos.InCheck then 0 else evalPos w pos)
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

        // 4. pruning block (gated, non-PV, not in check; disabled during a singular exclusion search)
        if not produced && usePruning && not isPv && not inCheck && excludedMove = MoveNone then
            if
                depthIn <= 6
                && staticEval - 120 * depthIn - (if ttPv then 20 else 0) >= beta
                && abs beta < MATE_IN_MAX_PLY
            then
                result <- staticEval
                produced <- true
            // razoring: at shallow depth a static eval far below alpha is verified by qsearch;
            // if even captures can't lift alpha, fail low. Mutually exclusive with RFP/null-move.
            elif
                cfg.UseRazoring
                && depthIn <= 3
                && abs alpha < MATE_IN_MAX_PLY
                && staticEval + (240 + 200 * depthIn) <= alpha
            then
                let v = qsearch w pos alpha (alpha + 1) ply

                if v <= alpha && not w.Control.Stopped then
                    result <- v
                    produced <- true
            elif
                depthIn >= 3
                && ply >= w.NmpMinPly
                && pos.PliesFromNull > 0
                && staticEval >= beta
                && hasNonPawnMaterial pos
            then
                let r = 3 + depthIn / 4 + (if staticEval - beta > 200 then 1 else 0)
                w.Stack.[ssCur].CurrentMove <- MoveNull
                w.Stack.[ssCur].MovedPiece <- NoPiece
                pos.MakeNull()
                let v = -(negamax w pos (-beta) (-beta + 1) (depthIn - r) (ply + 1) false false)
                pos.UnmakeNull()

                if v >= beta && not w.Control.Stopped then
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

                        if vv >= beta && not w.Control.Stopped then
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
                let pcMoves = w.MoveBuf.AsSpan(ply * MaxMoves, MaxMoves)
                let pcScores = w.ScoreBuf.AsSpan(ply * MaxMoves, MaxMoves)
                let mutable pcMp = mkProbCut pos w.Tables ttMove (probCutBeta - staticEval) pcMoves pcScores
                let mutable pcMove = nextMove &pcMp false

                while pcMove <> MoveNone && not produced && not w.Control.Stopped do
                    let needsCheck =
                        isEnPassant pcMove
                        || fromSq pcMove = ksq
                        || testBit (pos.BlockersForKing us) (fromSq pcMove)

                    if (not needsCheck) || isLegal pos pcMove then
                        w.Stack.[ssCur].CurrentMove <- pcMove
                        w.Stack.[ssCur].MovedPiece <- pos.PieceOn(fromSq pcMove)
                        pos.Make pcMove
                        let mutable v = -(qsearch w pos (-probCutBeta) (-probCutBeta + 1) (ply + 1))

                        if v >= probCutBeta then
                            v <- -(negamax w pos (-probCutBeta) (-probCutBeta + 1) (depthIn - 4) (ply + 1) false (not cutNode))

                        pos.Unmake pcMove

                        if not w.Control.Stopped && v >= probCutBeta then
                            // Trust a sufficient existing TT entry over clobbering it; else store + return v.
                            if ttHit && ttDepth >= depthIn - 3 && ttScore <> VALUE_NONE && ttScore >= probCutBeta then
                                result <- ttScore
                            else
                                if useTt then
                                    w.Control.Tt.Store pos.Key (depthIn - 3) BoundLower (valueToTt v ply) staticEval pcMove false

                                result <- v

                            produced <- true

                    pcMove <- if produced || w.Control.Stopped then MoveNone else nextMove &pcMp false

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
            let mutable m = nextMove &mp skipQuiets

            while m <> MoveNone && not cutoff do
                let needsCheck =
                    isEnPassant m || fromSq m = ksq || testBit (pos.BlockersForKing us) (fromSq m)

                if m <> excludedMove && ((not needsCheck) || isLegal pos m) then
                    moveCount <- moveCount + 1

                    let isQuiet =
                        (pos.PieceOn(toSq m) = NoPiece) && not (isEnPassant m) && not (isPromotion m)

                    let givesCheck = pos.GivesCheck m
                    let mutable doMove = true

                    // --- forward pruning: never on the first real move, never when we might be getting mated ---
                    if usePruning && not isPv && not inCheck && best > -MATE_IN_MAX_PLY then
                        let lmrDepth = max 0 (depth - 1 - reduction depth moveCount)

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
                            // SEE — a quiet that walks into a losing exchange
                            elif lmrDepth <= 7 && not (pos.SeeGe m (-25 * lmrDepth * lmrDepth)) then
                                doMove <- false
                        elif (not givesCheck) && depth <= 6 && not (pos.SeeGe m (-90 * depth)) then
                            // SEE — a capture that loses material by static exchange at shallow depth
                            doMove <- false

                    if doMove then
                        let mutable ext = if usePruning && givesCheck && pos.SeeGe m 0 then 1 else 0

                        // Singular / double extension: the TT move, at depth >= 8 with a depth-sufficient
                        // lower-bound TT score, is "singular" if every OTHER move fails low against a
                        // reduced exclusion window. Extend it (twice if it fails low by a clear margin).
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

                            if not w.Control.Stopped && sv < singularBeta then
                                ext <- ext + (if (not isPv) && sv < singularBeta - 16 then 2 else 1)

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
                        let mutable v = 0

                        if moveCount = 1 then
                            v <- -(negamax w pos (-beta) (-alpha) newDepth (ply + 1) isPv (if isPv then false else not cutNode))
                        elif not usePruning then
                            v <- -(negamax w pos (-beta) (-alpha) newDepth (ply + 1) false (not cutNode))
                        else
                            // LMR: table reduction, deeper for non-improving / quiet moves, lighter for captures
                            let mutable r =
                                if depth >= 3 && moveCount > 1 && not givesCheck then
                                    let mutable rr = reduction depth moveCount

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

                        if w.Control.Stopped then
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
                if useTt && not w.Control.Stopped && excludedMove = MoveNone then
                    let bound =
                        if best <= alphaIn then BoundUpper
                        elif best >= beta then BoundLower
                        else BoundExact

                    w.Control.Tt.Store pos.Key depth bound (valueToTt best ply) staticEval bestMove ttPv

                result <- best

        result

// ---------------------------------------------------------------------------
// Reporting (Console only — never printfn)
// ---------------------------------------------------------------------------
let private writeLine (s: string) = System.Console.Out.WriteLine(s)

let private scoreString (score: int) : string =
    if score >= MATE_IN_MAX_PLY then
        "mate " + string ((MATE - score + 1) / 2)
    elif score <= -MATE_IN_MAX_PLY then
        "mate " + string (-((MATE + score + 1) / 2))
    else
        "cp " + string score

let private reportInfo (w: Worker) (depth: int) (score: int) =
    let ms = w.Control.ElapsedMs
    let nodes = w.Control.NodeSum() // aggregate across all workers, not just the main one
    let nps = if ms > 0L then nodes * 1000L / ms else nodes
    let sb = StringBuilder(160)

    sb
        .Append("info depth ")
        .Append(depth)
        .Append(" seldepth ")
        .Append(w.SelDepth)
        .Append(" score ")
        .Append(scoreString score)
        .Append(" nodes ")
        .Append(nodes)
        .Append(" nps ")
        .Append(nps)
        .Append(" time ")
        .Append(ms)
        .Append(" pv")
    |> ignore

    let mutable i = 0
    let mutable cont = true

    while cont && i < MaxSearchPly do
        let mv = w.Pv.[i]

        if mv = MoveNone then
            cont <- false
        else
            sb.Append(' ').Append(toUci mv) |> ignore
            i <- i + 1

    writeLine (sb.ToString())

// ---------------------------------------------------------------------------
// Iterative deepening (aspiration). All workers run it; only the main reports + sets the time stop.
// ---------------------------------------------------------------------------
let iterativeDeepening (w: Worker) (maxDepth: int) : unit =
    let cfg = w.Control.Config
    let mutable prev = evalPos w w.Pos
    let mutable depth = 1

    while depth <= maxDepth && not w.Control.Stopped do
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
                beta <- (alpha + beta) / 2
                alpha <- max (-INF) (score - delta)
                delta <- delta * 2
                attemptDepth <- depth // a fail-low re-search runs at full depth again
            elif score >= beta then
                beta <- min INF (score + delta)
                delta <- delta * 2

                if cfg.UseAspTweaks && attemptDepth > depth - 3 && abs score < MATE_IN_MAX_PLY then
                    attemptDepth <- attemptDepth - 1
            else
                searching <- false

        if not w.Control.Stopped then
            prev <- score
            w.RootScore <- score
            w.RootBest <- w.Pv.[0]
            w.CompletedDepth <- depth

            if w.IsMain then
                reportInfo w depth score

        depth <- depth + 1

        // Between iterations the main worker stops once the soft (optimum) budget is spent; the hard budget
        // stops a running iteration mid-flight via CheckTime. (Best-move-stability / predictive scaling was
        // tried but over-conserved time without SPRT tuning — reverted; MoveOverhead is kept below.)
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
            // one move at 40% of the clock (or 4x soft). The SF optimum/maximum-scaling experiment over the
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
    control.Tt.NewSearch()
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

    let rb = workers.[0].RootBest

    let best =
        if isLegalRoot workers.[0].Pos rb then
            rb
        else
            firstLegalMove workers.[0].Pos

    control.LastBest <- best
    control.LastScore <- workers.[0].RootScore
    writeLine ("bestmove " + toUci best)
    best

// ---------------------------------------------------------------------------
// Test entry: a single fixed-depth, FULL-WINDOW negamax (bypasses aspiration/ID). The correctness oracle.
// ---------------------------------------------------------------------------
let searchToDepthNet (fen: string) (rootMoves: Move[]) (depth: int) (cfg: SearchConfig) (net: SfNetwork option) : struct (int * int64 * Move) =
    let tt = TranspositionTable(max 1 cfg.HashMb)

    let control =
        SearchControl(cfg, { defaultLimits with Depth = depth }, tt, fen, rootMoves, ?net = net)

    let w = Worker(0, true, control)
    w.SetupRoot()
    control.Reset()
    control.StartClock 0L 0L
    let score = negamax w w.Pos (-INF) INF depth 0 true false
    struct (score, w.Nodes, w.Pv.[0])

/// Single-thread iterative deepening until the node budget is exhausted (no UCI stdout). Test/tooling entry.
let searchToNodesNet (fen: string) (rootMoves: Move[]) (nodes: int64) (cfg: SearchConfig) (net: SfNetwork option) : struct (int * int64 * Move) =
    let tt = TranspositionTable(max 1 cfg.HashMb)
    let limits = { defaultLimits with Nodes = nodes }

    let control =
        SearchControl(cfg, limits, tt, fen, rootMoves, ?net = net)

    let w = Worker(0, true, control)
    w.SetupRoot()
    control.Reset()
    control.Tt.NewSearch()
    control.NodeSum <- (fun () -> w.Nodes)
    control.StartClock 0L 0L
    iterativeDeepening w (MaxSearchPly - 1)
    control.Stop()

    let best =
        if isLegalRoot w.Pos w.RootBest then w.RootBest else firstLegalMove w.Pos

    struct (w.RootScore, w.Nodes, best)

let searchToDepth (fen: string) (rootMoves: Move[]) (depth: int) (cfg: SearchConfig) : struct (int * int64 * Move) =
    searchToDepthNet fen rootMoves depth cfg None

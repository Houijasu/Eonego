/// Eonego — staged, lazy move picker.
///
/// [<Struct; IsByRefLike>] value type carrying the two caller-owned PARALLEL stackalloc buffers
/// (Span<Move> moves + Span<int> scores) plus the live stage/cursor state, the Position, and the
/// per-thread History.Tables. Advance is the MODULE function `nextMove (mp: byref<MovePick>) skipQuiets`
/// — NEVER an instance method (a method on a by-ref-like struct would mutate a COPY and the stage would
/// never advance). It is a `while` loop over the stage id (NOT `let rec`: a byref-like parameter can defeat
/// .tail under AOT). Laziness is purely STRUCTURAL: generate(Quiets) is invoked only inside the QuietInit
/// stage, so an early beta cutoff (or always passing skipQuiets) before that stage never materializes quiets.
///
/// LazySMP / lockless: zero module-level mutable state; both buffers are stackalloc'd by the CONSUMING
/// search frame and handed in (a function may never return a Span over its own stackalloc); Tables is one
/// per-thread heap object. Captures occupy [0,EndMoves); losing ("bad") captures are spilled to the FRONT
/// region [0,EndBadCaptures) as good captures are consumed, then replayed AFTER quiets. Quiets are appended
/// above the capture buffer via Moves.Slice(qStart).
module Eonego.MovePick

#nowarn "9" // NativePtr.stackalloc at call sites (AllowUnsafeBlocks set in the .fsproj)

open System
open System.Diagnostics
open System.Runtime.CompilerServices
open Eonego.Bitboard
open Eonego.Move
open Eonego.Position
open Eonego.MoveGeneration
open Eonego.History

// ---------------------------------------------------------------------------
// Stage ids: plain [<Literal>] ints (NOT a DU — Stage is a mutable struct field). The numeric order of
// each chain is load-bearing: an init stage "falls through" by bumping Stage and looping.
// ---------------------------------------------------------------------------
[<Literal>]
let StgMainTT = 0

[<Literal>]
let StgCaptureInit = 1

[<Literal>]
let StgGoodCapture = 2

[<Literal>]
let StgRefutation = 3

[<Literal>]
let StgQuietInit = 4

[<Literal>]
let StgQuiet = 5

[<Literal>]
let StgBadCapture = 6

[<Literal>]
let StgEvasionTT = 7

[<Literal>]
let StgEvasionInit = 8

[<Literal>]
let StgEvasion = 9

[<Literal>]
let StgProbCutTT = 10

[<Literal>]
let StgProbCutInit = 11

[<Literal>]
let StgProbCut = 12

[<Literal>]
let StgQSearchTT = 13

[<Literal>]
let StgQCaptureInit = 14

[<Literal>]
let StgQCapture = 15

[<Literal>]
let StgDone = 16

[<Literal>]
let EvasionBonus = 268435456 // 1 <<< 28 — keeps capture-evasions above any history term, no overflow

// --------------------------------------------------------------------------------------------------
// The layout (explicit `val mutable` form, NOT a struct-record: a record would auto-derive         -
// equality/compare over the Span/ref fields). Span fields are legal ONLY because the container is  -
// IsByRefLike; Position/Tables are ordinary GC references (legal fields in a ref struct).          -
// ---------------------------------------------------------------------------
[<Struct; IsByRefLike>]
type MovePick =
    val mutable Stage: int
    val mutable Cur: int // read cursor into the current segment
    val mutable EndMoves: int // one past the last move in the current segment
    val mutable EndBadCaptures: int // front spill region [0,EndBadCaptures) = losing captures
    val mutable RefIdx: int // refutation cursor (0=killer1, 1=killer2, 2=countermove)
    val mutable Moves: Span<Move> // caller stackalloc, length MaxMoves
    val mutable Scores: Span<int> // caller stackalloc, length MaxMoves (PARALLEL to Moves)
    val mutable Pos: Position
    val mutable Tables: Tables
    val mutable TtMove: Move
    val mutable Killer1: Move
    val mutable Killer2: Move
    val mutable CounterMove: Move
    val mutable Depth: int
    val mutable Threshold: int // ProbCut SEE threshold (0 for the main search)
    // Continuation-history context (the previous moves' piece/to). `-1` prevPc disables the term
    // (root / after null / NoPiece). prev1 = 1-ply-back (ss-1), prev2 = 2-ply-back (ss-2).
    val mutable PrevPc1: int
    val mutable PrevTo1: int
    val mutable PrevPc2: int
    val mutable PrevTo2: int
    // Policy sidecar (null PolNet = off, the default — every factory leaves these inert; Search.fs sets
    // them post-construction on the main picker when EONEGO_POLICY loaded a net). PolFrom/PolTo are this
    // ply's 64-wide logit arrays (Worker-owned slices); PolKey is a 1-wide slice of the Worker's per-ply
    // Zobrist staleness guard — logits are valid for THIS node iff PolKey[0] = Pos.Key (per-ply slots are
    // reused across sibling nodes; a bare bool would serve a stale sibling's logits silently).
    val mutable PolNet: Policy.PolicyNetwork
    val mutable ValNet: NNUE.Network
    val mutable PolFrom: Span<int>
    val mutable PolTo: Span<int>
    val mutable PolKey: Span<uint64>
    // Own-trunk policy net (EONPOL03, null = off). Mutually exclusive with PolNet in practice; when
    // set it fills the SAME per-ply logit arrays, but only at ≤6-piece positions (Policy.ownApplies).
    val mutable OwnNet: Policy.OwnNetwork

    // Explicit constructor: a by-ref-like struct cannot be zero-init'd (`MovePick()` /
    // `Unchecked.defaultof` both fail — Span fields + the byref-as-generic-arg rule). The four cursors
    // always start at 0; everything else is supplied by the factories below.
    new
        (
            pos: Position,
            tables: Tables,
            moves: Span<Move>,
            scores: Span<int>,
            stage: int,
            ttMove: Move,
            k1: Move,
            k2: Move,
            cm: Move,
            depth: int,
            threshold: int,
            prevPc1: int,
            prevTo1: int,
            prevPc2: int,
            prevTo2: int
        ) =
        { Stage = stage
          Cur = 0
          EndMoves = 0
          EndBadCaptures = 0
          RefIdx = 0
          Moves = moves
          Scores = scores
          Pos = pos
          Tables = tables
          TtMove = ttMove
          Killer1 = k1
          Killer2 = k2
          CounterMove = cm
          Depth = depth
          Threshold = threshold
          PrevPc1 = prevPc1
          PrevTo1 = prevTo1
          PrevPc2 = prevPc2
          PrevTo2 = prevTo2
          PolNet = null
          ValNet = Unchecked.defaultof<NNUE.Network>
          PolFrom = Span<int>()
          PolTo = Span<int>()
          PolKey = Span<uint64>()
          OwnNet = null }

// ---------------------------------------------------------------------------
// Helpers (module functions over byref<MovePick> so swaps/scoring persist). A move is a "capture" for
// ordering/refutation purposes iff its destination is occupied OR it is en passant.
// ---------------------------------------------------------------------------
[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let private isCap (pos: Position) (m: Move) : bool =
    pos.PieceOn(toSq m) <> NoPiece || isEnPassant m

/// Swap Moves and Scores at i and j in lockstep (keeps the parallel buffers aligned).
let private swap (mp: byref<MovePick>) (i: int) (j: int) : unit =
    let tm = mp.Moves.[i] in
    mp.Moves.[i] <- mp.Moves.[j]
    mp.Moves.[j] <- tm
    let ts = mp.Scores.[i] in
    mp.Scores.[i] <- mp.Scores.[j]
    mp.Scores.[j] <- ts

/// Select the max-score move in [s,e), swap it into slot s, return s. PRE: s < e.
let private pickBest (mp: byref<MovePick>) (s: int) (e: int) : int =
    let mutable best = s

    for i in s + 1 .. e - 1 do
        if mp.Scores.[i] > mp.Scores.[best] then
            best <- i

    if best <> s then
        swap &mp s best

    s

/// Partial insertion sort: move every element with score >= limit to the front in score-descending
/// order; the rest stay unordered behind. Operates on BOTH parallel buffers (held-element form, NOT swaps).
let private partialInsertionSort (mp: byref<MovePick>) (s: int) (e: int) (limit: int) : unit =
    let mutable sortedEnd = s

    for p in s + 1 .. e - 1 do
        if mp.Scores.[p] >= limit then
            sortedEnd <- sortedEnd + 1
            let tmpM = mp.Moves.[p]
            let tmpS = mp.Scores.[p]
            mp.Moves.[p] <- mp.Moves.[sortedEnd]
            mp.Scores.[p] <- mp.Scores.[sortedEnd]
            let mutable q = sortedEnd

            while q <> s && mp.Scores.[q - 1] < tmpS do
                mp.Moves.[q] <- mp.Moves.[q - 1]
                mp.Scores.[q] <- mp.Scores.[q - 1]
                q <- q - 1

            mp.Moves.[q] <- tmpM
            mp.Scores.[q] <- tmpS

let private scoreCaptures (mp: byref<MovePick>) (s: int) (e: int) : unit =
    let pos = mp.Pos
    let tables = mp.Tables

    for i in s .. e - 1 do
        let m = mp.Moves.[i]
        let pc = pos.PieceOn(fromSq m)

        let capturedPT =
            if isEnPassant m then
                Pawn
            else
                pieceType (pos.PieceOn(toSq m))

        mp.Scores.[i] <- Tunables.CaptScoreMul * pieceValueOf capturedPT + tables.CaptureHistory pc (toSq m) capturedPT

let private scoreQuiets (mp: byref<MovePick>) (s: int) (e: int) : unit =
    let pos = mp.Pos
    let us = pos.SideToMove
    let tables = mp.Tables

    for i in s .. e - 1 do
        let m = mp.Moves.[i]
        let pc = pos.PieceOn(fromSq m)
        let dst = toSq m

        mp.Scores.[i] <-
            tables.MainHistory us (fromTo m)
            + tables.ContHistory1 mp.PrevPc1 mp.PrevTo1 pc dst
            + tables.ContHistory2 mp.PrevPc2 mp.PrevTo2 pc dst

let private scoreEvasions (mp: byref<MovePick>) (s: int) (e: int) : unit =
    let pos = mp.Pos
    let us = pos.SideToMove
    let tables = mp.Tables

    for i in s .. e - 1 do
        let m = mp.Moves.[i]
        let capPc = pos.PieceOn(toSq m)

        if capPc <> NoPiece || isEnPassant m then
            let capturedPT = if isEnPassant m then Pawn else pieceType capPc
            let moverPT = pieceType (pos.PieceOn(fromSq m))
            mp.Scores.[i] <- pieceValueOf capturedPT - pieceValueOf moverPT + EvasionBonus
        else
            mp.Scores.[i] <- tables.MainHistory us (fromTo m)

// ---------------------------------------------------------------------------
// Constructors (module factories). Each takes the caller's TWO stackalloc spans. The TT move is validated
// by pos.IsPseudoLegal (ProbCut also requires it be a capture passing SEE >= threshold). In check, the
// main and qsearch pickers divert to the self-contained evasion chain.
// ---------------------------------------------------------------------------
let mkMain
    (pos: Position)
    (tables: Tables)
    (ttMove: Move)
    (k1: Move)
    (k2: Move)
    (cm: Move)
    (depth: int)
    (prevPc1: int)
    (prevTo1: int)
    (prevPc2: int)
    (prevTo2: int)
    (moves: Span<Move>)
    (scores: Span<int>)
    : MovePick =
    let tt =
        if ttMove <> MoveNone && pos.IsPseudoLegal ttMove then
            ttMove
        else
            MoveNone

    let stage = if pos.InCheck then StgEvasionTT else StgMainTT
    MovePick(pos, tables, moves, scores, stage, tt, k1, k2, cm, depth, 0, prevPc1, prevTo1, prevPc2, prevTo2)

let mkQSearch (pos: Position) (tables: Tables) (ttMove: Move) (moves: Span<Move>) (scores: Span<int>) : MovePick =
    let tt =
        if ttMove <> MoveNone && pos.IsPseudoLegal ttMove then
            ttMove
        else
            MoveNone

    let stage = if pos.InCheck then StgEvasionTT else StgQSearchTT
    MovePick(pos, tables, moves, scores, stage, tt, MoveNone, MoveNone, MoveNone, 0, 0, -1, -1, -1, -1)

let mkProbCut
    (pos: Position)
    (tables: Tables)
    (ttMove: Move)
    (threshold: int)
    (moves: Span<Move>)
    (scores: Span<int>)
    : MovePick =
    let ok =
        ttMove <> MoveNone
        && pos.IsPseudoLegal ttMove
        && isCap pos ttMove
        && pos.SeeGe ttMove threshold

    let tt = if ok then ttMove else MoveNone
    MovePick(pos, tables, moves, scores, StgProbCutTT, tt, MoveNone, MoveNone, MoveNone, 0, threshold, -1, -1, -1, -1)

// ---------------------------------------------------------------------------
// nextMove — the staged advance. MODULE function over byref so mutation persists. Returns MoveNone when
// the stage chain is exhausted. `skipQuiets` (re-read every call; the search may flip it mid-drain)
// suppresses quiets AND quiet refutations, jumping from the refutation/quiet stages straight to bad captures.
// ---------------------------------------------------------------------------
let nextMove (mp: byref<MovePick>) (skipQuiets: bool) : Move =
    let mutable result = MoveNone
    let mutable producing = true

    while producing do
        match mp.Stage with
        // ---- TT stages: emit the (pseudo-legal-validated) TT move first, then fall through ----
        | StgMainTT
        | StgEvasionTT
        | StgProbCutTT
        | StgQSearchTT ->
            mp.Stage <- mp.Stage + 1

            if mp.TtMove <> MoveNone then
                result <- mp.TtMove
                producing <- false

        // ---- capture generation (shared by main / probcut / qsearch) ----
        | StgCaptureInit
        | StgProbCutInit
        | StgQCaptureInit ->
            mp.EndMoves <- generate mp.Pos mp.Moves Captures
            scoreCaptures &mp 0 mp.EndMoves
            mp.Cur <- 0
            mp.EndBadCaptures <- 0
            mp.Stage <- mp.Stage + 1

        // ---- main-search good captures: SEE-split, spill losers to the front ----
        | StgGoodCapture ->
            let mutable found = false

            while not found && mp.Cur < mp.EndMoves do
                let i = pickBest &mp mp.Cur mp.EndMoves
                let m = mp.Moves.[i]
                mp.Cur <- mp.Cur + 1

                if m = mp.TtMove then
                    () // already emitted at TT
                elif mp.Pos.SeeGe m 0 then
                    result <- m
                    found <- true
                    producing <- false // GOOD capture
                else
                    swap &mp (mp.Cur - 1) mp.EndBadCaptures // spill to bad-capture front region
                    mp.EndBadCaptures <- mp.EndBadCaptures + 1

            if not found then
                mp.Stage <- StgRefutation

        // ---- refutations: killer1, killer2, countermove (each a legal, non-capture, non-dup quiet) ----
        | StgRefutation ->
            if skipQuiets then
                mp.Stage <- StgQuietInit // refutations are quiets -> skip
            else
                let mutable found = false

                while not found && mp.RefIdx < 3 do
                    let idx = mp.RefIdx
                    mp.RefIdx <- idx + 1

                    let r =
                        match idx with
                        | 0 -> mp.Killer1
                        | 1 -> mp.Killer2
                        | _ -> mp.CounterMove

                    let isDup = (idx >= 1 && r = mp.Killer1) || (idx >= 2 && r = mp.Killer2)

                    if
                        r <> MoveNone
                        && r <> mp.TtMove
                        && not isDup
                        && not (isCap mp.Pos r)
                        && mp.Pos.IsPseudoLegal r
                    then
                        result <- r
                        found <- true
                        producing <- false

                if not found then
                    mp.Stage <- StgQuietInit

        // ---- quiet generation (THE only generate(Quiets) site -> structural laziness) ----
        | StgQuietInit ->
            if skipQuiets then
                mp.Cur <- 0
                mp.Stage <- StgBadCapture
            else
                let qStart = mp.EndMoves
                let cnt = generate mp.Pos (mp.Moves.Slice qStart) Quiets
                Debug.Assert((qStart + cnt <= MaxMoves), "MovePick: capture+quiet overflow")
                mp.EndMoves <- qStart + cnt
                scoreQuiets &mp qStart mp.EndMoves

                if PosProf.Enabled then
                    PosProf.nQuietInit <- PosProf.nQuietInit + 1L
                    let d = if mp.Depth < 0 then 0 elif mp.Depth > 63 then 63 else mp.Depth
                    PosProf.QuietInitByDepth.[d] <- PosProf.QuietInitByDepth.[d] + 1L

                // Policy fill (lazy — only nodes that actually reach quiet scoring pay the inference,
                // and only above PolMinDepth). The Zobrist guard makes the fill once-per-node and the
                // arrays readable downstream (LMR term in Search.fs) for exactly this position. Two
                // sources fill the SAME 384-wide arrays: the EONPOL02 sidecar (any position) or the
                // EONPOL03 own-trunk net (only at ≤6 pieces — Policy.ownApplies). When the own net is
                // loaded but doesn't apply, no fill happens and the stale-key guard leaves ordering/LMR
                // untouched, exactly as if policy were off for this node.
                let mutable polFilled = false

                if mp.Depth >= Tunables.PolMinDepth then
                    if not (isNull mp.OwnNet) then
                        if Policy.ownApplies mp.Pos then
                            if mp.PolKey.[0] <> mp.Pos.Key then
                                Policy.fillLogitsOwn mp.OwnNet mp.Pos mp.PolFrom mp.PolTo
                                mp.PolKey.[0] <- mp.Pos.Key

                            polFilled <- true
                    elif not (isNull mp.PolNet) then
                        if mp.PolKey.[0] <> mp.Pos.Key then
                            Policy.fillLogits mp.ValNet mp.PolNet mp.Pos mp.PolFrom mp.PolTo
                            mp.PolKey.[0] <- mp.Pos.Key

                        polFilled <- true

                // Ordering blend — INERT at the default PolOrdMul = 0 (Phase-0 re-scope: the LMR term
                // is the v1 consumption; this term buys its own SPRT later). Logit index =
                // moverPieceType*64 + STM-relative square (both EONPOL02 and EONPOL03).
                if polFilled && Tunables.PolOrdMul <> 0 then
                    let stm = mp.Pos.SideToMove

                    for i in qStart .. mp.EndMoves - 1 do
                        let m = mp.Moves.[i]
                        let pt = pieceType (mp.Pos.PieceOn(fromSq m))

                        let ps =
                            mp.PolFrom.[pt * 64 + Policy.relSq stm (fromSq m)]
                            + mp.PolTo.[pt * 64 + Policy.relSq stm (toSq m)]

                        let t = (Tunables.PolOrdMul * ps) >>> Tunables.PolOrdShift
                        let t = max (-Tunables.PolClamp) (min Tunables.PolClamp t)
                        mp.Scores.[i] <- mp.Scores.[i] + t
                partialInsertionSort &mp qStart mp.EndMoves (Tunables.QuietSortLimit * mp.Depth)
                mp.Cur <- qStart
                mp.Stage <- StgQuiet

        // ---- quiets: stream in sorted order, skipping TT + refutations ----
        | StgQuiet ->
            if skipQuiets then
                mp.Cur <- 0
                mp.Stage <- StgBadCapture
            else
                let mutable found = false

                while not found && mp.Cur < mp.EndMoves do
                    let m = mp.Moves.[mp.Cur]
                    mp.Cur <- mp.Cur + 1

                    if m <> mp.TtMove && m <> mp.Killer1 && m <> mp.Killer2 && m <> mp.CounterMove then
                        result <- m
                        found <- true
                        producing <- false

                if not found then
                    mp.Cur <- 0
                    mp.Stage <- StgBadCapture

        // ---- bad captures: replay the spilled losers from the front, last ----
        | StgBadCapture ->
            let mutable found = false

            while not found && mp.Cur < mp.EndBadCaptures do
                let m = mp.Moves.[mp.Cur]
                mp.Cur <- mp.Cur + 1

                if m <> mp.TtMove then
                    result <- m
                    found <- true
                    producing <- false

            if not found then
                (mp.Stage <- StgDone
                 producing <- false)

        // ---- evasion chain (in check): captures-of-checker first via the +EvasionBonus score ----
        | StgEvasionInit ->
            mp.EndMoves <- generate mp.Pos mp.Moves Evasions
            scoreEvasions &mp 0 mp.EndMoves
            mp.Cur <- 0
            mp.Stage <- StgEvasion
        | StgEvasion ->
            let mutable found = false

            while not found && mp.Cur < mp.EndMoves do
                let i = pickBest &mp mp.Cur mp.EndMoves
                let m = mp.Moves.[i]
                mp.Cur <- mp.Cur + 1

                if m <> mp.TtMove then
                    result <- m
                    found <- true
                    producing <- false

            if not found then
                (mp.Stage <- StgDone
                 producing <- false)

        // ---- probcut: captures passing SEE >= threshold only ----
        | StgProbCut ->
            let mutable found = false

            while not found && mp.Cur < mp.EndMoves do
                let i = pickBest &mp mp.Cur mp.EndMoves
                let m = mp.Moves.[i]
                mp.Cur <- mp.Cur + 1

                if m <> mp.TtMove && mp.Pos.SeeGe m mp.Threshold then
                    result <- m
                    found <- true
                    producing <- false

            if not found then
                (mp.Stage <- StgDone
                 producing <- false)

        // ---- qsearch captures (incl. queen push-promotions via the promotion split) ----
        | StgQCapture ->
            let mutable found = false

            while not found && mp.Cur < mp.EndMoves do
                let i = pickBest &mp mp.Cur mp.EndMoves
                let m = mp.Moves.[i]
                mp.Cur <- mp.Cur + 1

                if m <> mp.TtMove then
                    result <- m
                    found <- true
                    producing <- false

            if not found then
                (mp.Stage <- StgDone
                 producing <- false)

        | _ -> producing <- false // StgDone / exhausted -> MoveNone

    result

// ---------------------------------------------------------------------------
// COMPILE-PROBE residue: a trivial factory + byref step kept as a minimal byref-mutation-persistence
// regression guard (MovePickProbeTests). Not used in production.
// ---------------------------------------------------------------------------
let mkProbe (pos: Position) (tables: Tables) (moves: Span<Move>) (scores: Span<int>) : MovePick =
    MovePick(pos, tables, moves, scores, 0, MoveNone, MoveNone, MoveNone, MoveNone, 0, 0, -1, -1, -1, -1)

let probeStep (mp: byref<MovePick>) : Move =
    mp.Stage <- mp.Stage + 1
    mp.Cur <- mp.Cur + 1

    if mp.Cur < mp.Moves.Length then
        mp.Moves.[mp.Cur]
    else
        MoveNone

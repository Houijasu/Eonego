/// Eonego — retrograde search: on-demand exact solving of low-material endings.
///
/// When the game reaches a 3-man ending (two kings + one piece of either color), the engine runs the
/// retrograde algorithm over that material's full state space: a forward init pass classifies the
/// terminals (checkmate / stalemate) with the battle-tested legal generator, then backward induction
/// from the terminals propagates proven win-in-N / loss-in-N / draw values to a fixpoint. The search
/// probes those values at node entry for exact distance-to-mate scores.
///
/// Everything is discovered at search time and lives in RAM for the session: solving is triggered by
/// the actual root position (UCI `position` with few men), one *signature* (which piece, which color)
/// at a time, published lock-free per signature. Nothing is precomputed, shipped, or written to disk.
///
/// Concurrency model: solving happens on one background thread under a lock; the search only ever
/// reads `Volatile.Read`-published immutable arrays (the Reductions-table pattern) — LazySMP-safe.
module Eonego.Retrograde

open System
open System.Diagnostics
open System.Text
open System.Threading
open Eonego.Bitboard
open Eonego.Move
open Eonego.Position
open Eonego.MoveGeneration

// ---------------------------------------------------------------------------
// (1) Value encoding — one sbyte per position of a signature.
//
//     v = RetroIllegal (-128)  index is not a legal chess position
//     v = RetroUnknown (+127)  build-time sentinel; swept to 0 before publication, never observable
//     v = 0                    draw
//     v = +k (k >= 1)          side to move mates in (k-1) plies   [dtm odd in these endings]
//     v = -k (k >= 1)          side to move is mated in (k-1) plies [dtm even; checkmated-now = -1]
//
//     i.e. v = sign * (dtm + 1); the +1 offset keeps "checkmated now" (-1) distinct from draw (0).
//     Max |v| is ~57 (longest pawn-signature win), comfortably inside sbyte.
// ---------------------------------------------------------------------------
[<Literal>]
let RetroIllegal = -128y

[<Literal>]
let RetroUnknown = 127y

/// Distance to mate in plies. PRECONDITION: v is a win/loss value (not RetroIllegal/0).
let inline retroDtm (v: sbyte) : int = abs (int v) - 1

/// Value of the parent position given one successor's value (from the successor's side to move):
/// child mates in (v-1)  -> parent (who just moved into it) is mated in v      -> -(v+1)
/// child mated in (-v-1) -> parent mates in (-v)                               -> +(-v+1)
let inline succToPred (v: sbyte) : sbyte =
    if v = 0y then 0y
    elif v > 0y then -v - 1y
    else -v + 1y

/// Total order "better for the side to move": faster mate > slower mate > draw > later loss > sooner
/// loss. Used by the verification pass to pick the minimax-best successor.
let inline retroOrd (v: sbyte) : int =
    if v > 0y then 1000 - int v
    elif v = 0y then 0
    else -1000 - int v

// ---------------------------------------------------------------------------
// (2) Index math — (stm, whiteKingSq, blackKingSq, pieceSq) -> 0 .. RetroSize-1.
//     The index is color-agnostic: the signature slot (the piece's `Piece` code, owner included)
//     distinguishes a White queen ending from a Black one. No mirroring anywhere.
// ---------------------------------------------------------------------------
[<Literal>]
let RetroSize = 524288 // 2 * 64^3

let inline idxOf (stm: Color) (wk: Square) (bk: Square) (pc: Square) : int =
    (stm <<< 18) ||| (wk <<< 12) ||| (bk <<< 6) ||| pc

let inline idxStm (idx: int) : Color = idx >>> 18
let inline idxWk (idx: int) : Square = (idx >>> 12) &&& 63
let inline idxBk (idx: int) : Square = (idx >>> 6) &&& 63
let inline idxPc (idx: int) : Square = idx &&& 63

// ---------------------------------------------------------------------------
// (3) Arithmetic index legality — no Position involved.
//
//     Legal iff: squares pairwise distinct; kings not adjacent; a pawn is never on rank 1/8; and the
//     side NOT to move is not in check. The last rule is non-trivial only when the piece's owner is
//     to move: the piece must not attack the bare king (king-gives-check is excluded by adjacency,
//     and the bare side has nothing that could check the owner).
//
//     The retraction step of the solver relies on this being computed for EVERY index up front: its
//     only predecessor-validity check is `values.[P] <> RetroIllegal`, which is complete because
//     occupancy collisions, pawn-rank violations, and discovered-check-after-king-unmove all land on
//     indices this function rejected with the predecessor's own full occupancy.
// ---------------------------------------------------------------------------

/// Attack set of the signature piece from `pc` with only the two kings as blockers.
let inline private pieceAttacks (pt: PieceType) (owner: Color) (pc: Square) (occKK: Bitboard) : Bitboard =
    if pt = Queen then queenAttacks pc occKK
    elif pt = Rook then rookAttacks pc occKK
    elif pt = Bishop then bishopAttacks pc occKK
    elif pt = Knight then knightAttacks pc
    else pawnAttacks owner pc // Pawn; King is never a signature piece

/// Is (stm, wk, bk, pc) a legal chess position for signature `pce`?
let arithLegal (pce: Piece) (stm: Color) (wk: Square) (bk: Square) (pc: Square) : bool =
    if wk = bk || wk = pc || bk = pc then false
    elif testBit (kingAttacks wk) bk then false
    else
        let pt = pieceType pce
        let owner = pieceColor pce

        if pt = Pawn && (rankOf pc = 0 || rankOf pc = 7) then
            false
        elif stm = owner then
            // Bare side is not to move: its king must not be attacked by the piece.
            let bareKingSq = if owner = White then bk else wk
            not (testBit (pieceAttacks pt owner pc (bit wk ||| bit bk)) bareKingSq)
        else
            // Owner is not to move: the bare side has only a king, and king-checks are already
            // excluded by the adjacency rule — always consistent.
            true

// ---------------------------------------------------------------------------
// (4) FEN builder — the bridge from an index to the battle-tested Position/movegen machinery.
//     A reused StringBuilder + one reused net-free Position keep the init pass allocation-light.
//     EP is always "-" (no EP capture exists with a bare defending king) and castling always "-"
//     (never representable in the index; real positions with live rights are declined at the probe).
// ---------------------------------------------------------------------------
let internal fenOf (sb: StringBuilder) (pce: Piece) (stm: Color) (wk: Square) (bk: Square) (pc: Square) : string =
    let pieceCh = (if pieceColor pce = White then "PNBRQK" else "pnbrqk").[pieceType pce]
    sb.Clear() |> ignore

    for r = 7 downto 0 do
        let mutable run = 0

        for f = 0 to 7 do
            let sq = mkSquare f r

            let ch =
                if sq = wk then 'K'
                elif sq = bk then 'k'
                elif sq = pc then pieceCh
                else ' '

            if ch = ' ' then
                run <- run + 1
            else
                if run > 0 then
                    sb.Append(char (int '0' + run)) |> ignore
                    run <- 0

                sb.Append ch |> ignore

        if run > 0 then
            sb.Append(char (int '0' + run)) |> ignore

        if r > 0 then
            sb.Append '/' |> ignore

    sb.Append(if stm = White then " w - - 0 1" else " b - - 0 1") |> ignore
    sb.ToString()

// ---------------------------------------------------------------------------
// (5) Init pass (forward): every index gets RetroIllegal, a terminal value, or a legal-move counter.
//
//     Counter protocol: `counter.[idx]` counts ALL legal moves, including out-of-signature ones
//     (captures -> KK, promotions). Only within-signature successors finalized as wins ever
//     decrement it during the backward pass — a draw successor (KxPiece, promotion into a drawn or
//     stalemate position) never does, holding the counter above zero forever, which is exactly
//     "this position has a drawing escape and can never be a loss". The bare side can never win,
//     so there are no init-time decrements at all and BFS level order alone makes loss DTM exact.
//
//     Pawn signatures additionally read the ALREADY-SOLVED promotion signatures (same owner, the
//     promoted piece type) for each promotion move of the just-generated legal move list — never
//     an arithmetic promotion square: the bare king may legally stand there, and that colliding
//     index is RetroIllegal, not a value. A winning promotion (successor lost for the bare side)
//     is queued in `pendingWin.[dtm]` and merged at exactly that BFS level.
// ---------------------------------------------------------------------------
let internal initSignature
    (pce: Piece)
    (promoTables: sbyte[][]) // length 6, PieceType-indexed; consulted only for pawn signatures
    (values: sbyte[]) // RetroSize, pre-filled RetroUnknown
    (counter: byte[]) // RetroSize, zeroed
    (lossQ0: ResizeArray<int>) // out: checkmate indices (LossIn 0)
    (pendingWin: ResizeArray<int>[]) // dtm-indexed promotion win candidates (pawn signatures)
    : unit =
    let owner = pieceColor pce
    let isPawnSig = pieceType pce = Pawn
    let promoRank = if owner = White then 6 else 1
    let pos = Position()
    let sb = StringBuilder(80)
    // Heap buffer, NOT the perft stackalloc idiom: a Span over stackalloc'd memory held across a
    // 524k-iteration loop is unsafe under tiered-JIT on-stack replacement (the test host runs the
    // engine DLL with tiering on; observed as a mid-scan NRE in movegen). Cold build-time code —
    // one small array is free.
    let arr: Move[] = Array.zeroCreate MaxMoves

    for idx = 0 to RetroSize - 1 do
        let buf = Span<Move>(arr)
        let stm = idxStm idx
        let wk = idxWk idx
        let bk = idxBk idx
        let pc = idxPc idx

        if not (arithLegal pce stm wk bk pc) then
            values.[idx] <- RetroIllegal
        else
            pos.LoadFen(fenOf sb pce stm wk bk pc)
            let n = generateLegal pos buf

            if n = 0 then
                if pos.InCheck then
                    values.[idx] <- -1y // checkmate: mated now
                    lossQ0.Add idx
                else
                    values.[idx] <- 0y // stalemate: finalized draw
            else
                counter.[idx] <- byte n

                if isPawnSig && stm = owner && rankOf pc = promoRank then
                    for i = 0 to n - 1 do
                        let m = buf.[i]

                        if isPromotion m then
                            let v = promoTables.[promoType m].[idxOf (flipColor stm) wk bk (toSq m)]
                            Debug.Assert(v <> RetroIllegal) // successor of a legal move is legal

                            if v < 0y then
                                pendingWin.[retroDtm v + 1].Add idx

// ---------------------------------------------------------------------------
// (6) Arithmetic retraction — predecessors of a legal successor index, no Position involved.
//
//     Candidates are filtered by the caller with `values.[P] <> RetroIllegal` ONLY, which is a
//     complete validity test: occupancy collisions and pawn-rank violations land on illegal
//     indices; P's own legality (incl. discovered check after an owner-king un-move) was computed
//     by the init pass with P's full occupancy; and forward-move legality P->S decomposes into
//     clear path (reverse attack set with the kings as blockers), empty destination (S's squares
//     are distinct), and mover-king-safe-after-move — which is exactly S's own legality, and only
//     legal frontier members are ever retracted. The single check retraction adds itself: a pawn
//     double un-push needs the intermediate square king-free (it is not part of P's index).
//
//     No un-captures (nothing to capture) and no un-promotions within a signature (those
//     predecessors live in the pawn signature and are irrelevant when solving a piece signature).
// ---------------------------------------------------------------------------

/// Fills `out` with the predecessor indices of successor `sIdx` (which MUST be legal) and returns
/// the count. Max 35 candidates (8 king retreats + 27 queen retreats); size `out` accordingly.
let internal predecessorsInto (pce: Piece) (sIdx: int) (out: int[]) : int =
    let stmS = idxStm sIdx
    let wk = idxWk sIdx
    let bk = idxBk sIdx
    let pc = idxPc sIdx
    let owner = pieceColor pce
    let mover = flipColor stmS
    let occKK = bit wk ||| bit bk
    let mutable n = 0

    if mover = owner then
        // Owner king retreats (kings-adjacent / attacked-in-P cases die on the RetroIllegal filter).
        let fromK = if owner = White then wk else bk
        let mutable cand = kingAttacks fromK &&& ~~~(occKK ||| bit pc)

        while cand <> 0UL do
            let preSq = popLsb &cand

            out.[n] <-
                (if owner = White then
                     idxOf mover preSq bk pc
                 else
                     idxOf mover wk preSq pc)

            n <- n + 1

        // The piece un-moves.
        let pt = pieceType pce

        if pt = Pawn then
            let push = if owner = White then 8 else -8
            let pre1 = pc - push
            out.[n] <- idxOf mover wk bk pre1 // rank-1/8 or king-square collisions -> RetroIllegal
            n <- n + 1
            let dblRank = if owner = White then 3 else 4

            if rankOf pc = dblRank && not (testBit occKK pre1) then
                out.[n] <- idxOf mover wk bk (pc - 2 * push)
                n <- n + 1
        else
            let mutable cand =
                (if pt = Queen then queenAttacks pc occKK
                 elif pt = Rook then rookAttacks pc occKK
                 elif pt = Bishop then bishopAttacks pc occKK
                 else knightAttacks pc)
                &&& ~~~occKK

            while cand <> 0UL do
                let preSq = popLsb &cand
                out.[n] <- idxOf mover wk bk preSq
                n <- n + 1
    else
        // Bare (defender) king retreats.
        let fromK = if owner = White then bk else wk
        let mutable cand = kingAttacks fromK &&& ~~~(occKK ||| bit pc)

        while cand <> 0UL do
            let preSq = popLsb &cand

            out.[n] <-
                (if owner = White then
                     idxOf mover wk preSq pc
                 else
                     idxOf mover preSq bk pc)

            n <- n + 1

    n

// ---------------------------------------------------------------------------
// (7) The retrograde solve: init, then backward BFS by DTM level, then the draw sweep.
//     Level arithmetic (values encode sign*(dtm+1)): a loss finalized at level d propagates
//     WinIn(d+1) = +(d+2) to its predecessors; when a predecessor's counter of unresolved
//     successors hits zero during win level d+1, it is LossIn(d+2) = -(d+3). Promotion win
//     candidates merge at exactly their level, before that win level is processed.
// ---------------------------------------------------------------------------
let internal solveSignature (pce: Piece) (promoTables: sbyte[][]) : sbyte[] =
    let values = Array.create RetroSize RetroUnknown
    let counter: byte[] = Array.zeroCreate RetroSize
    let lossQ = Array.init 128 (fun _ -> ResizeArray<int>())
    let winQ = Array.init 128 (fun _ -> ResizeArray<int>())
    let pendingWin = Array.init 128 (fun _ -> ResizeArray<int>())
    initSignature pce promoTables values counter lossQ.[0] pendingWin
    let owner = pieceColor pce
    let bare = flipColor owner
    let preds: int[] = Array.zeroCreate 40
    let mutable d = 0

    while d <= 124 do
        // (a) losses at level d: every predecessor is a win at level d+1.
        for s in lossQ.[d] do
            let n = predecessorsInto pce s preds

            for i = 0 to n - 1 do
                let p = preds.[i]

                if values.[p] = RetroUnknown then
                    Debug.Assert(idxStm p = owner) // wins only with the owner to move
                    values.[p] <- sbyte (d + 2)
                    winQ.[d + 1].Add p

        // (b) promotion win candidates merge at exactly their level (same level as step (a) wins,
        //     so ordering between (a) and (b) cannot change a DTM).
        for p in pendingWin.[d + 1] do
            if values.[p] = RetroUnknown then
                Debug.Assert(idxStm p = owner)
                values.[p] <- sbyte (d + 2)
                winQ.[d + 1].Add p

        // (c) wins at level d+1: decrement each predecessor's unresolved-successor counter; at
        //     zero, ALL its moves are proven wins for the opponent -> loss at level d+2.
        for s in winQ.[d + 1] do
            let n = predecessorsInto pce s preds

            for i = 0 to n - 1 do
                let p = preds.[i]

                if values.[p] = RetroUnknown then
                    counter.[p] <- counter.[p] - 1uy

                    if counter.[p] = 0uy then
                        Debug.Assert(idxStm p = bare) // losses only with the bare side to move
                        values.[p] <- sbyte (-(d + 3))
                        lossQ.[d + 2].Add p

        d <- d + 2

    // Fixpoint reached: every remaining unknown legal entry can never be forced into the win/loss
    // lattice — a proven draw. The d <= 124 sweep bound is a hard DTM ceiling: step (c) of the last
    // iteration can still WRITE LossIn(126) values whose predecessors would then be swept as draws
    // (an inconsistent frontier) — irrelevant for 3-man (max observed level is 56, KPK), but assert
    // the frontier stayed empty so any future deeper material class trips loudly here.
    Debug.Assert(lossQ.[126].Count = 0 && winQ.[125].Count = 0)

    for i = 0 to RetroSize - 1 do
        if values.[i] = RetroUnknown then
            values.[i] <- 0y

    values

// ---------------------------------------------------------------------------
// (8) Verification pass — a complete self-consistency proof, independent of all retraction code.
//     For every index: illegal indices must be RetroIllegal; terminals must match the mate/
//     stalemate rules; every other value must equal the minimax (via retroOrd) over succToPred of
//     the REAL legal moves' successor values (captures -> KK draw; promotions -> the promotion
//     signature's table). Returns Some error on the first mismatch, None when the table is proven.
// ---------------------------------------------------------------------------
let internal verifySignature (pce: Piece) (values: sbyte[]) (promoTables: sbyte[][]) : string option =
    let pos = Position()
    let sb = StringBuilder(80)
    // Heap buffer for the same reason as initSignature: no stackalloc span across a 524k loop.
    let arr: Move[] = Array.zeroCreate MaxMoves
    let mutable err: string option = None
    let mutable idx = 0

    while idx < RetroSize && err.IsNone do
        let buf = Span<Move>(arr)
        let stm = idxStm idx
        let wk = idxWk idx
        let bk = idxBk idx
        let pc = idxPc idx

        if not (arithLegal pce stm wk bk pc) then
            if values.[idx] <> RetroIllegal then
                err <- Some("illegal index not marked: " + fenOf sb pce stm wk bk pc)
        else
            pos.LoadFen(fenOf sb pce stm wk bk pc)
            let n = generateLegal pos buf

            let expected =
                if n = 0 then
                    if pos.InCheck then -1y else 0y
                else
                    let mutable best = 0y
                    let mutable bestOrd = System.Int32.MinValue

                    for i = 0 to n - 1 do
                        let m = buf.[i]
                        Debug.Assert(not (isEnPassant m)) // no EP exists in these endings
                        pos.Make m

                        let vSucc =
                            if popCount pos.Occupied = 2 then
                                0y // KxPiece -> bare kings, dead draw
                            else
                                let sq = lsb (pos.Occupied ^^^ pos.Pieces King)
                                let tbl = if isPromotion m then promoTables.[promoType m] else values
                                tbl.[idxOf pos.SideToMove (pos.KingSquare White) (pos.KingSquare Black) sq]

                        pos.Unmake m
                        let cand = succToPred vSucc
                        let o = retroOrd cand

                        if o > bestOrd then
                            bestOrd <- o
                            best <- cand

                    best

            if values.[idx] <> expected then
                err <-
                    Some(
                        "mismatch at "
                        + fenOf sb pce stm wk bk pc
                        + ": got "
                        + string (int values.[idx])
                        + " expected "
                        + string (int expected)
                    )

        idx <- idx + 1

    err

// ---------------------------------------------------------------------------
// (9) Publication + orchestration. One slot per Piece code (10 solvable signatures); a slot is
//     null until its signature is solved, then holds the immutable value table forever. Solving
//     runs under one lock (a pawn signature recursively solves its four promotion signatures
//     first, so leaves publish before the pawn); the search only ever does a Volatile.Read of a
//     slot — the read-only-after-publication pattern that keeps LazySMP workers synchronization-
//     free. Nothing is solved until a low-material root shows up (requestSolveFor).
// ---------------------------------------------------------------------------
// Unsolved sentinel = the shared Array.empty singleton, NOT null (keeps the file clean under the
// nullness analyzer); a slot is replaced exactly once by the immutable solved table.
let private solved: sbyte[][] = Array.init 12 (fun _ -> Array.empty)
let private solveLock = obj ()

let rec private ensureSolvedInLock (pce: Piece) : unit =
    if (Volatile.Read(&solved.[pce])).Length = 0 then
        if pieceType pce = Pawn then
            let owner = pieceColor pce
            ensureSolvedInLock (makePiece owner Knight)
            ensureSolvedInLock (makePiece owner Bishop)
            ensureSolvedInLock (makePiece owner Rook)
            ensureSolvedInLock (makePiece owner Queen)

        let promoTables =
            if pieceType pce = Pawn then
                let owner = pieceColor pce
                let t: sbyte[][] = Array.zeroCreate 6
                t.[Knight] <- Volatile.Read(&solved.[makePiece owner Knight])
                t.[Bishop] <- Volatile.Read(&solved.[makePiece owner Bishop])
                t.[Rook] <- Volatile.Read(&solved.[makePiece owner Rook])
                t.[Queen] <- Volatile.Read(&solved.[makePiece owner Queen])
                t
            else
                Array.empty

        Volatile.Write(&solved.[pce], solveSignature pce promoTables)

/// Synchronously solve a signature (and, for pawns, its promotion signatures first). Idempotent,
/// double-checked-locked — safe to race from tests, the background trigger, and tooling.
let ensureSolved (pce: Piece) : unit =
    if (Volatile.Read(&solved.[pce])).Length = 0 then
        lock solveLock (fun () -> ensureSolvedInLock pce)

let isSolved (pce: Piece) : bool =
    (Volatile.Read(&solved.[pce])).Length <> 0

/// The solved table of a signature (empty until solved). Tooling/tests only — the search probes.
let internal solvedTable (pce: Piece) : sbyte[] = Volatile.Read(&solved.[pce])

/// Signatures worth solving for this root: a 3-man root's own signature; for a 4-man root, the
/// signature each single capture leaves behind (promotion signatures follow inside ensureSolved).
let internal signatureClosure (pos: Position) : int list =
    let nonKings = pos.Occupied ^^^ pos.Pieces King

    match popCount pos.Occupied with
    | 3 -> [ pos.PieceOn(lsb nonKings) ]
    | 4 ->
        let mutable bb = nonKings
        let mutable acc = []

        while bb <> 0UL do
            let pce = pos.PieceOn(popLsb &bb)

            if not (List.contains pce acc) then
                acc <- pce :: acc

        acc
    | _ -> []

/// Root trigger: called when a new game position arrives. Fire-and-forget — a BelowNormal
/// background thread solves whatever the root material makes reachable; probes simply return
/// ValueNone until each signature publishes. Never blocks, never throws (a failed solve leaves
/// its slot null: the search falls through to normal evaluation forever — degraded but correct).
let requestSolveFor (pos: Position) : unit =
    if popCount pos.Occupied <= 4 then
        let sigs = signatureClosure pos

        if sigs |> List.exists (fun pce -> not (isSolved pce)) then
            let th =
                Thread(
                    ThreadStart(fun () ->
                        try
                            for pce in sigs do
                                ensureSolved pce
                        with _ ->
                            ())
                )

            th.IsBackground <- true
            th.Priority <- ThreadPriority.BelowNormal
            th.Start()

/// Probe a real search position. ValueNone unless: exactly 3 men, no live castling rights (a
/// 3-man position with O-O available is real and un-modeled), and the signature has published.
/// The value is from `pos.SideToMove`'s perspective; EpSquare is ignored (no EP capture exists
/// in these endings). Zero allocation.
let probe (pos: Position) : sbyte voption =
    if pos.CastlingRights <> 0 then
        ValueNone
    elif popCount pos.Occupied <> 3 then
        ValueNone
    else
        let sq = lsb (pos.Occupied ^^^ pos.Pieces King)
        let table = Volatile.Read(&solved.[pos.PieceOn sq])

        if table.Length = 0 then
            ValueNone
        else
            let v = table.[idxOf pos.SideToMove (pos.KingSquare White) (pos.KingSquare Black) sq]
            Debug.Assert(v <> RetroIllegal) // a real legal position never maps to an illegal index
            if v = RetroIllegal then ValueNone else ValueSome v

/// Aggregate stats of a solved table: struct (legal, wins, losses, maxWinDtm, maxLossDtm).
/// Draws = legal - wins - losses. Asserts no build sentinel survived to publication.
let internal statsOf (values: sbyte[]) : struct (int * int * int * int * int) =
    let mutable legal = 0
    let mutable wins = 0
    let mutable losses = 0
    let mutable maxWin = 0
    let mutable maxLoss = 0

    for i = 0 to RetroSize - 1 do
        let v = values.[i]
        Debug.Assert(v <> RetroUnknown)

        if v <> RetroIllegal then
            legal <- legal + 1

            if v > 0y then
                wins <- wins + 1
                maxWin <- max maxWin (retroDtm v)
            elif v < 0y then
                losses <- losses + 1
                maxLoss <- max maxLoss (retroDtm v)

    struct (legal, wins, losses, maxWin, maxLoss)

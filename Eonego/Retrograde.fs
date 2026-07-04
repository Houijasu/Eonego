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

#nowarn "9" // NativePtr.stackalloc — AllowUnsafeBlocks is already set in the .fsproj

open System
open System.Diagnostics
open System.Text
open Microsoft.FSharp.NativeInterop
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
    let p = NativePtr.stackalloc<Move> MaxMoves
    let buf = Span<Move>(NativePtr.toVoidPtr p, MaxMoves)

    for idx = 0 to RetroSize - 1 do
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

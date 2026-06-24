/// POLICY network port (for MCTS move priors). The move-index + feature ENCODING is Monty's
/// (3072 input, 7840 SEE-doubled move index); the concrete NET we ship is **Jackal's**
/// (github.com/TomaszJaworski777/Jackal, GPL-3.0) — `p8008192009q.network`, which copies Monty's
/// encoding verbatim but has its own smaller weights (HL=8192 vs Monty's 40960). Monty's net server
/// was unreachable; Jackal's nets are on HuggingFace (Snekkers/networks). See THIRD-PARTY-NOTICES.md.
///
/// This file (step 1 of the port) implements Monty's GEOMETRIC move-index tables exactly:
///   - the per-piece pseudo-attack generators (ray-to-edge for sliders, blockers ignored) used by
///     `DESTINATIONS` in outputs.rs,
///   - the `OFFSETS` prefix-sum (a single running counter across ALL six piece types),
///   - the derived index-space size `NUM_MOVES_INDICES = 2 * (OFFSETS[5][64] + PROMOS + 2 + 8)`.
/// `OFFSETS[5][64]` (the grand geometric total) is COMPUTED here, not assumed — it pins the L2 row
/// count, which the real net file's byte length must equal: 3072*40960 + 40960 + N*20480 + N (all i8).
///
/// Monty squares are LSB=a1, rank-major (square = 8*rank+file) == Eonego LERF, so the bit arithmetic
/// ports verbatim. Move-index logic (promo/castle/dbl/quiet), SEE, map_features and the i8 inference
/// are subsequent steps.
module Eonego.MontyPolicy

open System
open System.Numerics
open System.Buffers.Binary
open Eonego.Bitboard
open Eonego.Move
open Eonego.Position

// ---------------------------------------------------------------------------
// Network constants (policy.rs / outputs.rs). L1=40960 is the release net (datagen uses 16384).
// ---------------------------------------------------------------------------
[<Literal>]
let PolicyInputSize = 3072

// Jackal policy net dims (the concrete net we download/ship; the ENCODING is Monty's, shared).
// l0: 3072 -> HL=8192; pairwise clamped-product activation -> hidden = HL/2 = 4096;
// l1 (transposed): hidden 4096 -> 7840 move indices. QA=QB=128 (Jackal has no Monty FACTOR).
[<Literal>]
let PolicyHl = 8192

[<Literal>]
let PolicyHidden = 4096

[<Literal>]
let PolicyPromos = 88 // 4 * 22

[<Literal>]
let PolicyQA = 128

[<Literal>]
let PolicyQB = 128

// ---------------------------------------------------------------------------
// Geometric pseudo-attack generators (verbatim from outputs.rs / attacks.rs; a1=0 LERF).
// ---------------------------------------------------------------------------
let private fileA = 0x0101010101010101UL
let private fileH = 0x8080808080808080UL

// 15 diagonal masks (attacks.rs DIAGS).
let private diags =
    [| 0x0100000000000000UL
       0x0201000000000000UL
       0x0402010000000000UL
       0x0804020100000000UL
       0x1008040201000000UL
       0x2010080402010000UL
       0x4020100804020100UL
       0x8040201008040201UL
       0x0080402010080402UL
       0x0000804020100804UL
       0x0000008040201008UL
       0x0000000080402010UL
       0x0000000000804020UL
       0x0000000000008040UL
       0x0000000000000080UL |]

/// outputs.rs PAWN: white-perspective forward fan (NW | N | NE), edge-masked.
let private pawnFan (sq: int) : uint64 =
    let bit = 1UL <<< sq
    ((bit &&& ~~~fileA) <<< 7) ||| (bit <<< 8) ||| ((bit &&& ~~~fileH) <<< 9)

let private knightAtt (sq: int) : uint64 =
    let n = 1UL <<< sq
    let h1 = ((n >>> 1) &&& 0x7f7f7f7f7f7f7f7fUL) ||| ((n <<< 1) &&& 0xfefefefefefefefeUL)
    let h2 = ((n >>> 2) &&& 0x3f3f3f3f3f3f3f3fUL) ||| ((n <<< 2) &&& 0xfcfcfcfcfcfcfcfcUL)
    (h1 <<< 16) ||| (h1 >>> 16) ||| (h2 <<< 8) ||| (h2 >>> 8)

let private kingAtt (sq: int) : uint64 =
    let mutable k = 1UL <<< sq
    k <- k ||| (k <<< 8) ||| (k >>> 8)
    k <- k ||| ((k &&& ~~~fileA) >>> 1) ||| ((k &&& ~~~fileH) <<< 1)
    k ^^^ (1UL <<< sq)

/// outputs.rs bishop(): the two full diagonals through sq (blockers ignored), sq excluded.
let private bishopGeo (sq: int) : uint64 =
    let r = sq / 8
    let f = sq % 8
    BinaryPrimitives.ReverseEndianness(diags.[f + r]) ^^^ diags.[7 + f - r]

/// outputs.rs rook(): full rank XOR full file (sq excluded by the XOR).
let private rookGeo (sq: int) : uint64 =
    let r = sq / 8
    let f = sq % 8
    (0xFFUL <<< (r * 8)) ^^^ (fileA <<< f)

let private queenGeo (sq: int) : uint64 = bishopGeo sq ||| rookGeo sq

/// DESTINATIONS[sq].[pc], pc 0..5 = pawn, knight, bishop, rook, queen, king.
let destinations: uint64[,] =
    let d = Array2D.zeroCreate 64 6

    for sq in 0..63 do
        d.[sq, 0] <- pawnFan sq
        d.[sq, 1] <- knightAtt sq
        d.[sq, 2] <- bishopGeo sq
        d.[sq, 3] <- rookGeo sq
        d.[sq, 4] <- queenGeo sq
        d.[sq, 5] <- kingAtt sq

    d

/// OFFSETS[pc].[sq] (sq 0..64): one running counter that DOES NOT reset between piece types, so
/// OFFSETS[5].[64] is the grand total of all geometric destinations across the six piece types.
let offsets: int[,] =
    let o = Array2D.zeroCreate 6 65
    let mutable curr = 0

    for pc in 0..5 do
        for sq in 0..63 do
            o.[pc, sq] <- curr
            curr <- curr + BitOperations.PopCount(destinations.[sq, pc])

        o.[pc, 64] <- curr

    o

/// Grand geometric total (= map_move_to_index's `OFFSETS[5][64]` base for promo/castle/dbl).
let OffsetsBase = offsets.[5, 64]

/// One half of the index space: geometric + promos(88) + castles(2) + double-pushes(8).
let FromTo = OffsetsBase + PolicyPromos + 2 + 8

/// Full L1 row count: the index space is doubled by the good-SEE bit (FROM_TO * good_see + idx).
let NumMovesIndices = 2 * FromTo

/// Expected raw Jackal `.network` length (headerless repr(C), all i8):
/// l0.weights[3072*8192] + l0.biases[8192] + l1.weights[7840*4096] + l1.biases[7840].
let ExpectedNetBytes: int64 =
    int64 PolicyInputSize * int64 PolicyHl
    + int64 PolicyHl
    + int64 NumMovesIndices * int64 PolicyHidden
    + int64 NumMovesIndices

// ---------------------------------------------------------------------------
// Net loader (headerless repr(C): four contiguous int8 arrays, little-endian = raw bytes).
// ---------------------------------------------------------------------------
type PolicyNetwork =
    { L0W: sbyte[] // PolicyInputSize * PolicyHl, row-major [input][neuron]
      L0B: sbyte[] // PolicyHl
      L1W: sbyte[] // NumMovesIndices * PolicyHidden, transposed [moveIndex][hidden]
      L1B: sbyte[] } // NumMovesIndices

type PolicyLoadResult =
    | Loaded of PolicyNetwork
    | Failed of string

let loadBytes (buf: byte[]) : PolicyLoadResult =
    if int64 buf.Length <> ExpectedNetBytes then
        Failed(sprintf "policy net size %d <> expected %d (layout mismatch)" buf.Length ExpectedNetBytes)
    else
        let slice (off: int) (n: int) : sbyte[] =
            let a = Array.zeroCreate<sbyte> n
            System.Buffer.BlockCopy(buf, off, a, 0, n) // i8 == raw byte
            a

        let l0wN = PolicyInputSize * PolicyHl
        let l0bN = PolicyHl
        let l1wN = NumMovesIndices * PolicyHidden
        let l1bN = NumMovesIndices

        Loaded
            { L0W = slice 0 l0wN
              L0B = slice l0wN l0bN
              L1W = slice (l0wN + l0bN) l1wN
              L1B = slice (l0wN + l0bN + l1wN) l1bN }

let load (path: string) : PolicyLoadResult =
    if not (System.IO.File.Exists path) then
        Failed("file not found: " + path)
    else
        try
            loadBytes (System.IO.File.ReadAllBytes path)
        with ex ->
            Failed("could not read file: " + ex.Message)

// ---------------------------------------------------------------------------
// Move -> index (the `idx` in [0, FromTo), WITHOUT the good-SEE doubling).
// Reproduces Monty/Jackal `map_move_to_index`. Eonego flags are coarser than Monty's (Normal /
// Promotion / EnPassant / Castling), so double-push and castle-side are DERIVED from the move.
// The full index is `FromTo * good_see + moveToIndexBase` once Jackal's SEE is ported.
// ---------------------------------------------------------------------------
let moveToIndexBase (pos: Position) (m: Move) : int =
    let stm = pos.SideToMove
    let ksq = BitOperations.TrailingZeroCount(pos.PiecesCT stm King)
    let hm = if (ksq % 8) > 3 then 7 else 0
    let flip = hm ^^^ (if stm = Black then 56 else 0)
    let fromS = fromSq m
    let toS = toSq m
    let src = fromS ^^^ flip
    let dst = toS ^^^ flip

    if isPromotion m then
        // promo_pc - KNIGHT = the 2-bit promo field (0=N,1=B,2=R,3=Q); ffile/tfile from flipped squares.
        let promoIdx = (m >>> 12) &&& 0x3
        OffsetsBase + 22 * promoIdx + (2 * (src % 8) + (dst % 8))
    elif isCastling m then
        // King-side iff the king moves toward higher files (e->g); hm flips file-side, hence the XOR.
        let isKs = if (toS % 8) > (fromS % 8) then 1 else 0
        let isHm = if hm = 0 then 1 else 0
        OffsetsBase + PolicyPromos + (isKs ^^^ isHm)
    else
        let movingPt = pieceType (pos.PieceOn fromS)

        if movingPt = Pawn && abs ((toS / 8) - (fromS / 8)) = 2 then
            OffsetsBase + PolicyPromos + 2 + (src % 8) // double pawn push (one slot per file)
        else
            // Quiet/capture/en-passant: rank `dst` within the piece's geometric destination set.
            let below = destinations.[src, movingPt] &&& ((1UL <<< dst) - 1UL)
            offsets.[movingPt, src] + BitOperations.PopCount(below)

// ---------------------------------------------------------------------------
// Jackal SEE (static exchange evaluation). Port of jackal/chess/src/board/see.rs EXACTLY:
// values, threshold, occupancy setup, x-ray refresh, king legality special-case.
// Used ONLY to compute the good-SEE bit for mapMoveToIndex; NOT used in search (pos.SeeGe).
// ---------------------------------------------------------------------------

/// Jackal SEE piece values [P,N,B,R,Q,K] — different from Eonego's PeSTO values.
let private seeValues = [| 100; 450; 450; 650; 1250; 0 |]

/// Value of a piece (0 for NoPiece / King).
let private seeVal (pc: Piece) : int =
    if pc = NoPiece then 0 else seeValues.[pieceType pc]

/// Jackal `see(pos, mv, threshold) -> bool`.
/// Returns true iff the exchange starting with `mv` beats `threshold`.
let see (pos: Position) (m: Move) (threshold: int) : bool =
    let fromS = fromSq m
    let toS   = toSq m

    // next_victim: for promotions use the promoted piece type; otherwise the mover's type.
    let mutable nextVictim =
        if isPromotion m then promoType m
        else pieceType (pos.PieceOn fromS)

    // move_value: total material delta of the first move.
    let moveValue =
        if isEnPassant m then
            seeValues.[Pawn]                                       // captured pawn
        elif isCastling m then
            0                                                       // KxR encoding: no net gain
        elif isPromotion m then
            // capture value of the to-square piece (0 if non-capture promo)
            // + promoted piece value - pawn value
            seeVal (pos.PieceOn toS) + seeValues.[promoType m] - seeValues.[Pawn]
        else
            seeVal (pos.PieceOn toS)                               // 0 for quiet

    // Step 3: initial balance check.
    let mutable balance = moveValue - threshold
    if balance < 0 then false
    else
        // Step 4: subtract the moved piece (worst case: we lose it).
        balance <- balance - seeValues.[nextVictim]
        if balance >= 0 then true
        else
            // Step 5: occupancy after the first move. Jackal does `exclude(from).include(to)` and, for EP,
            // then `exclude(ep_sq)` where ep_sq == to (the landing square) — so the include(to)/exclude(to)
            // CANCEL: net EP occupancy = exclude(from) only (the captured pawn STAYS, landing sq stays empty).
            // We MUST replicate this quirk for policy-index parity with the oracle, not "correct" SEE.
            let mutable occ =
                if isEnPassant m then
                    pos.Occupied ^^^ (1UL <<< fromS)
                else
                    (pos.Occupied ^^^ (1UL <<< fromS)) ||| (1UL <<< toS)

            // Step 6: slider masks.
            let bishops = pos.Pieces Bishop ||| pos.Pieces Queen
            let rooks   = pos.Pieces Rook   ||| pos.Pieces Queen

            // Step 7: attackers of BOTH colors to the target square.
            let mutable attackers = pos.AttackersTo toS occ &&& occ

            // Step 8: opponent recaptures first.
            let mutable side = 1 - pos.SideToMove   // opposite of STM (White=0,Black=1)
            let mutable looping = true
            let mutable result = false

            while looping do
                // Step 9a: check if side has any attacker.
                let mine = attackers &&& pos.ColorBB side
                if mine = 0UL then
                    looping <- false
                    result <- false   // will be resolved by the outer return
                else
                    // Step 9b: least-valuable attacker.
                    let mutable found = false
                    let mutable ptIdx = 0
                    while not found && ptIdx <= 5 do
                        if (mine &&& pos.PiecesCT side ptIdx) <> 0UL then
                            nextVictim <- ptIdx
                            found <- true
                        ptIdx <- ptIdx + 1

                    // Step 9c: remove one LVA from occupancy.
                    occ <- occ ^^^ (1UL <<< lsb (mine &&& pos.PiecesCT side nextVictim))

                    // Step 9d: x-ray refresh diagonal sliders.
                    if nextVictim = Pawn || nextVictim = Bishop || nextVictim = Queen then
                        attackers <- attackers ||| (bishopAttacks toS occ &&& bishops)
                    // Step 9d: x-ray refresh orthogonal sliders.
                    if nextVictim = Rook || nextVictim = Queen then
                        attackers <- attackers ||| (rookAttacks toS occ &&& rooks)

                    // Step 9e: drop pieces that have left occupancy.
                    attackers <- attackers &&& occ

                    // Step 9f: swap turn.
                    side <- 1 - side

                    // Step 9g: negamax balance update.
                    balance <- -balance - 1 - seeValues.[nextVictim]

                    // Step 9h: if balance >= 0 we (the side just to move) win this exchange.
                    if balance >= 0 then
                        // King legality: if king was the attacker AND opponent still has attackers,
                        // the king capture would be illegal — flip side back so we "lose".
                        if nextVictim = King && (attackers &&& pos.ColorBB side) <> 0UL then
                            side <- 1 - side
                        looping <- false

            // Step 10: the side to move after the loop is the LOSER.
            pos.SideToMove <> side

/// Full policy move index: `FromTo * goodSee + moveToIndexBase`.
/// Maps every legal move to a unique slot in [0, NumMovesIndices).
let mapMoveToIndex (pos: Position) (m: Move) : int =
    let goodSee = if see pos m -108 then 1 else 0
    FromTo * goodSee + moveToIndexBase pos m

// ---------------------------------------------------------------------------
// King-xray attack map (chess_board_utils.rs `generate_attack_map`).
// Removes the OPPONENT king from occupancy so sliders x-ray through it.
// ---------------------------------------------------------------------------

/// <summary>
/// Generate a bitboard of all squares attacked by <paramref name="attacker"/>'s pieces,
/// with the opponent king removed from occupancy (king-xray extension).
/// Matches Jackal's <c>generate_attack_map</c>.
/// </summary>
let generateAttackMap (pos: Position) (attacker: Color) : Bitboard =
    // Remove the defender's king so sliding pieces x-ray through it.
    let defKing = pos.KingSquare(1 - attacker)
    let occ = pos.Occupied ^^^ (1UL <<< defKing)

    let mutable threats = 0UL

    // Rooks + queens (orthogonal sliders).
    let mutable bb = pos.PiecesCT attacker Rook ||| pos.PiecesCT attacker Queen
    while bb <> 0UL do
        let sq = popLsb &bb
        threats <- threats ||| rookAttacks sq occ

    // Bishops + queens (diagonal sliders).
    bb <- pos.PiecesCT attacker Bishop ||| pos.PiecesCT attacker Queen
    while bb <> 0UL do
        let sq = popLsb &bb
        threats <- threats ||| bishopAttacks sq occ

    // King.
    bb <- pos.PiecesCT attacker King
    while bb <> 0UL do
        let sq = popLsb &bb
        threats <- threats ||| kingAttacks sq

    // Knights.
    bb <- pos.PiecesCT attacker Knight
    while bb <> 0UL do
        let sq = popLsb &bb
        threats <- threats ||| knightAttacks sq

    // Pawns — `pawnAttacks c sq` returns squares attacked BY a pawn of color c ON sq.
    bb <- pos.PiecesCT attacker Pawn
    while bb <> 0UL do
        let sq = popLsb &bb
        threats <- threats ||| pawnAttacks attacker sq

    threats

// ---------------------------------------------------------------------------
// 3072-element sparse feature map (threats_3072.rs `map_inputs`).
//
// Layout: 4 planes × 768, where each plane is [stm-pieces(384) | nstm-pieces(384)].
// Planes: 0 = base, 768 = threatened, 1536 = defended, 2304 = threatened+defended.
// Vertical flip if Black-to-move; horizontal mirror if king on files e–h.
// ---------------------------------------------------------------------------

/// <summary>
/// Flip a bitboard vertically (reverse rank order). Equivalent to byte-swapping the u64.
/// Matches Rust's <c>Bitboard::flip_mut()</c>.
/// </summary>
let private flipVertical (bb: Bitboard) : Bitboard =
    System.Buffers.Binary.BinaryPrimitives.ReverseEndianness bb

/// <summary>
/// Map all pieces to their sparse feature indices in [0, 3072).
/// One index per piece (32 at startpos). Matches Jackal's <c>Threats3072::map_inputs</c> exactly.
/// </summary>
let mapFeatures (pos: Position) : int[] =
    let stm = pos.SideToMove
    let flip = (stm = Black)

    // Horizontal mirror: king on files e–h (file > 3) → mirror index within rank.
    // Vertical flip ^56 does NOT change file, so use original stm king square.
    let ksq = pos.KingSquare stm
    let hm = if (ksq % 8) > 3 then 7 else 0

    // Threat/defence maps computed on the original board, then flipped with the rest.
    let mutable threats   = generateAttackMap pos (1 - stm) // squares attacked by opponent
    let mutable defences  = generateAttackMap pos stm        // squares attacked by stm

    if flip then
        threats  <- flipVertical threats
        defences <- flipVertical defences

    // Collect one feature index per piece.
    let result = System.Collections.Generic.List<int>()

    for pt in 0..5 do
        let pieceIdx = 64 * pt

        // --- STM pieces (base offset = 0 within plane) ---
        let mutable stmBb = pos.PiecesCT stm pt
        if flip then stmBb <- flipVertical stmBb

        while stmBb <> 0UL do
            let sq = popLsb &stmBb
            let mutable feat = pieceIdx + (sq ^^^ hm)
            if (threats   >>> sq) &&& 1UL <> 0UL then feat <- feat + 768
            if (defences  >>> sq) &&& 1UL <> 0UL then feat <- feat + 1536
            result.Add feat

        // --- NSTM pieces (base offset = 384 within plane) ---
        let mutable nstmBb = pos.PiecesCT (1 - stm) pt
        if flip then nstmBb <- flipVertical nstmBb

        while nstmBb <> 0UL do
            let sq = popLsb &nstmBb
            let mutable feat = 384 + pieceIdx + (sq ^^^ hm)
            if (threats   >>> sq) &&& 1UL <> 0UL then feat <- feat + 768
            if (defences  >>> sq) &&& 1UL <> 0UL then feat <- feat + 1536
            result.Add feat

    result.ToArray()

// ---------------------------------------------------------------------------
// int8 inference: accumulate → activate → logit → softmax.
// Ports policy_network.rs EXACTLY (QA=QB=128, pairwise-product activation).
// ---------------------------------------------------------------------------

let inline private clampI (x: int) lo hi = max lo (min hi x)

/// Layer 0: bias-initialised accumulator + weight accumulation.
/// acc.[i] = int16(L0B.[i]) + Σ_{f in feats} int16(L0W.[f*HL + i])
let accumulate (net: PolicyNetwork) (feats: int[]) : int16[] =
    let acc = Array.zeroCreate<int16> PolicyHl
    // Initialise with biases.
    for i in 0 .. PolicyHl - 1 do
        acc.[i] <- int16 net.L0B.[i]
    // Accumulate weights for each active feature.
    for f in feats do
        let off = f * PolicyHl
        for i in 0 .. PolicyHl - 1 do
            acc.[i] <- acc.[i] + int16 net.L0W.[off + i]
    acc

/// Pairwise clamped-product activation: pairs acc[n] with acc[n+4096].
/// result.[n] = clamp(acc[n], 0, QA) * clamp(acc[n+4096], 0, QA) / QA   (integer arithmetic)
let activate (acc: int16[]) : int16[] =
    let res = Array.zeroCreate<int16> PolicyHidden
    for n in 0 .. PolicyHidden - 1 do
        let a = clampI (int acc.[n])            0 PolicyQA
        let b = clampI (int acc.[n + PolicyHidden]) 0 PolicyQA
        res.[n] <- int16 (a * b / PolicyQA)
    res

/// Dot-product logit for one move index.
/// logit = (Σ L1W[moveIdx*4096 + n] * hidden[n]) / QA + L1B[moveIdx]) / QB
let logitOf (net: PolicyNetwork) (hidden: int16[]) (moveIdx: int) : float32 =
    let mutable s = 0
    let off = moveIdx * PolicyHidden
    for n in 0 .. PolicyHidden - 1 do
        s <- s + int net.L1W.[off + n] * int hidden.[n]
    (float32 s / float32 PolicyQA + float32 net.L1B.[moveIdx]) / float32 PolicyQB

/// Compute softmax policy priors for `count` legal moves.
/// Returns a float32[] of length `count`, summing to ~1.
let policyPriors (net: PolicyNetwork) (pos: Position) (moves: Move[]) (count: int) : float32[] =
    let hidden = activate (accumulate net (mapFeatures pos))
    let logits = Array.zeroCreate<float32> count
    for i in 0 .. count - 1 do
        logits.[i] <- logitOf net hidden (mapMoveToIndex pos moves.[i])
    if count = 1 then
        [| 1.0f |]
    else
        let mx = Array.max logits
        let e  = Array.map (fun l -> MathF.Exp(l - mx)) logits
        let s  = Array.sum e
        Array.map (fun ei -> ei / s) e

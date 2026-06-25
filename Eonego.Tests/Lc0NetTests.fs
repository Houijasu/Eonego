/// Lc0 hybrid tests. Structural gates first (validate the protobuf loader + BN fold against the real net).
/// Soft-skips when nets/20x256SE-jj-9-75000000.pb is absent (it is large + gitignored).
module Eonego.Tests.Lc0NetTests

open Xunit
open Eonego.Lc0Proto
open Eonego.Move
open Eonego.Bitboard
open Eonego.Position

let private tryNetPath () : string option =
    let mutable dir = System.IO.DirectoryInfo(System.AppContext.BaseDirectory)
    let mutable root = None

    while root.IsNone && not (isNull dir) do
        if System.IO.File.Exists(System.IO.Path.Combine(dir.FullName, "Eonego.slnx")) then
            root <- Some dir.FullName

        dir <- dir.Parent

    match root with
    | Some r ->
        let p = System.IO.Path.Combine(r, "nets", "20x256SE-jj-9-75000000.pb")
        if System.IO.File.Exists p then Some p else None
    | None -> None

let private finite (a: float32[]) =
    Array.forall (fun (x: float32) -> not (System.Single.IsNaN x || System.Single.IsInfinity x)) a

[<Fact>]
let ``Lc0 net loads with the expected 20x256SE architecture`` () =
    match tryNetPath () with
    | None -> () // soft-skip: net not present
    | Some p ->
        match load p with
        | Failed r -> Assert.Fail r
        | Loaded net ->
            // Body
            Assert.Equal(256, net.Channels)
            Assert.Equal(112, net.Input.InC)
            Assert.Equal(256, net.Input.OutC)
            Assert.Equal(3, net.Input.K)
            Assert.Equal(256 * 112 * 9, net.Input.W.Length)
            Assert.Equal(21, net.Tower.Length)

            let r0 = net.Tower.[0]
            Assert.Equal(256, r0.Conv1.OutC)
            Assert.Equal(3, r0.Conv1.K)
            Assert.Equal(256 * 256 * 9, r0.Conv1.W.Length)
            Assert.Equal(256, r0.Conv2.OutC)
            Assert.Equal(32, r0.Se.SeCh)
            Assert.Equal(256, r0.Se.C)
            Assert.Equal(2 * 256, r0.Se.B2.Length)
            Assert.Equal(32 * (2 * 256), r0.Se.W2.Length)

            // Policy head: conv 256->256 1x1, FC (256*64)->1858
            Assert.Equal(256, net.PolicyConv.OutC)
            Assert.Equal(1, net.PolicyConv.K)
            Assert.Equal(1858, net.PolicyB.Length)
            Assert.Equal(1858 * 256 * 64, net.PolicyW.Length)

            // Value head: conv 256->64 1x1, FC (64*64)->128->1 scalar
            Assert.Equal(64, net.ValueConv.OutC)
            Assert.Equal(1, net.ValueConv.K)
            Assert.Equal(128, net.Value1B.Length)
            Assert.Equal(128 * 64 * 64, net.Value1W.Length)
            Assert.Equal(1, net.Value2B.Length)
            Assert.Equal(128, net.Value2W.Length)

            // BN fold must not have produced NaN/Inf (the near-zero-variance eps floor).
            Assert.True(finite net.Input.W && finite net.Input.B, "input conv NaN/Inf after BN fold")
            Assert.True(finite r0.Conv1.W && finite r0.Conv1.B, "res0 conv1 NaN/Inf after BN fold")
            Assert.True(finite r0.Conv2.W && finite r0.Se.W2, "res0 conv2/SE NaN/Inf")
            Assert.True(finite net.PolicyW && finite net.Value1W, "head weights NaN/Inf")

// ---------------------------------------------------------------------------
// Phase 4 gate (net-independent): the 1858 policy map round-trips every legal move with no collisions.
// ---------------------------------------------------------------------------
let private psq (s: string) (o: int) = (int s.[o] - int 'a') + (int s.[o + 1] - int '1') * 8

[<Fact>]
let ``Lc0 1858 policy map round-trips legal moves with no collisions`` () =
    let fens =
        [ "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" // startpos (white, no flip)
          "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1" // startpos (black, exercises flip)
          "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1" // kiwipete (castling)
          "8/PPP2k2/8/8/8/8/2K2ppp/8 w - - 0 1" // white promotions (all 4 types per pawn)
          "8/2k2PPP/8/8/8/8/ppp2K2/8 b - - 0 1" ] // black promotions (flip + promo)

    for fen in fens do
        let pos = Position.OfFen fen
        let stmIsBlack = (pos.SideToMove = Black)
        let moves = Eonego.Tests.TestFixtures.collectLegal pos
        let seen = System.Collections.Generic.HashSet<int>()

        for m in moves do
            let idx = Eonego.Lc0PolicyMap.moveToNNIndex stmIsBlack m
            Assert.True(idx >= 0 && idx < 1858, sprintf "%s -> idx %d out of [0,1858)" (toUci m) idx)
            Assert.True(seen.Add idx, sprintf "policy-index collision %d for move %s in %s" idx (toUci m) fen)

            let flip s = if stmIsBlack then s ^^^ 56 else s
            let uci = Eonego.Lc0PolicyMap.nnIndexToUci.[idx]
            Assert.Equal(flip (fromSq m), psq uci 0)
            // Castling maps to the rook square (king-captures-rook: e1h1/e1a1), not the king's two-square dest.
            let expectedTo =
                if isCastling m then
                    let kf = flip (fromSq m) &&& 7
                    let tf = flip (toSq m) &&& 7
                    (flip (fromSq m) &&& 56) ||| (if tf > kf then 7 else 0)
                else
                    flip (toSq m)

            Assert.Equal(expectedTo, psq uci 2)

            if isPromotion m then
                let pc = (m >>> 12) &&& 0x3 // 0=N,1=B,2=R,3=Q

                if pc = 0 then
                    Assert.Equal(4, uci.Length) // knight promo => plain queen-move slot (no suffix)
                else
                    Assert.Equal(5, uci.Length)
                    let expected = (match pc with | 1 -> 'b' | 2 -> 'r' | _ -> 'q')
                    Assert.Equal(expected, uci.[4])

// ---------------------------------------------------------------------------
// Phase 2 gate (net-independent): 112-plane encoder structure + black-to-move flip.
// ---------------------------------------------------------------------------
[<Fact>]
let ``Lc0 encoder produces the expected 112-plane structure`` () =
    let out = Array.zeroCreate<float32> (112 * 64)
    let planeSum p = Array.sub out (p * 64) 64 |> Array.sum

    // White to move, startpos: no flip.
    let w = Position.OfFen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    Eonego.Lc0Encoder.encodeInto w out
    Assert.Equal(8.0f, planeSum 0) // our pawns
    Assert.Equal(2.0f, planeSum 1) // our knights
    Assert.Equal(1.0f, planeSum 5) // our king
    Assert.Equal(1.0f, out.[5 * 64 + 4]) // king on e1 (sq 4)
    Assert.Equal(8.0f, planeSum 6) // their pawns
    Assert.Equal(1.0f, planeSum 11) // their king
    Assert.Equal(0.0f, planeSum 12) // repetition
    Assert.Equal(8.0f, planeSum (13 + 0)) // h=1 repeat-current = h=0
    Assert.Equal(1.0f, planeSum (13 + 5))
    Assert.Equal(64.0f, planeSum 104) // we_can_000
    Assert.Equal(64.0f, planeSum 105) // we_can_00
    Assert.Equal(64.0f, planeSum 106) // they_000
    Assert.Equal(64.0f, planeSum 107) // they_00
    Assert.Equal(0.0f, planeSum 108) // white to move
    Assert.Equal(0.0f, planeSum 109) // rule50 = 0
    Assert.Equal(0.0f, planeSum 110) // zeros
    Assert.Equal(64.0f, planeSum 111) // all ones

    // Black to move: vertical flip -> our king (black e8) maps to e1 (sq 4); stm plane all-ones.
    let b = Position.OfFen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 5 3"
    Eonego.Lc0Encoder.encodeInto b out
    Assert.Equal(8.0f, planeSum 0) // our (black) pawns, flipped to rank 2
    Assert.Equal(1.0f, planeSum 5) // our king
    Assert.Equal(1.0f, out.[5 * 64 + 4]) // black king e8 -> flipped e1 (sq 4)
    Assert.Equal(64.0f, planeSum 108) // black to move
    Assert.Equal(5.0f * 64.0f, planeSum 109) // rule50 = 5, broadcast

// ---------------------------------------------------------------------------
// Phase 3/5 gate: forward pass scalar==AVX2 parity + sane policy/value (net required).
// The "principled opening move" check is the strongest end-to-end correctness signal without an lc0 oracle:
// a broken conv/encoder/BN/SE/policy-map would yield garbage (uniform or random) policy.
// ---------------------------------------------------------------------------
// Regression: the net encodes castling as king-captures-rook (e1h1/e1a1), so moveToNNIndex must remap the
// king-two-squares castling move to the rook square — otherwise O-O reads the wrong (near-zero) policy slot.
// Position is the Ruy Lopez Berlin after ...Nf6, where 4.O-O is the main line: the net must give O-O a large
// prior. Before the fix O-O's prior was ~0.005 (rank 11/32) and the engine never castled.
[<Fact>]
let ``Lc0 gives castling its main-line prior (king-to-rook encoding)`` () =
    match tryNetPath () with
    | None -> ()
    | Some p ->
        match load p with
        | Failed r -> Assert.Fail r
        | Loaded net ->
            let pos = Position.OfFen "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4"
            let inBuf = Array.zeroCreate<float32> (112 * 64)
            Eonego.Lc0Encoder.encodeInto pos inBuf
            let moves = Eonego.Tests.TestFixtures.collectLegal pos
            let priors = Array.zeroCreate<float32> moves.Length
            let scratch = Eonego.Lc0Net.Lc0Scratch(net)
            Eonego.Lc0Net.lc0PriorsInto true net pos moves moves.Length inBuf scratch priors |> ignore

            let mutable ci = -1
            for i in 0 .. moves.Length - 1 do
                if toUci moves.[i] = "e1g1" then ci <- i

            Assert.True(ci >= 0, "O-O (e1g1) must be a legal move here")
            Assert.True(
                priors.[ci] > 0.3f,
                "O-O prior = " + priors.[ci].ToString("0.000") + " (expected the net's main-line castling prior; was ~0.005 before the king-to-rook fix)"
            )

// Net-free lock on the king-captures-rook remap for ALL FOUR castling moves (white/black x king/queenside).
// The prior regression above only exercises white O-O; this proves the flip + queenside path index correctly.
[<Fact>]
let ``Lc0 policy index maps every castling move to the rook square`` () =
    let idxOf (u: string) = System.Array.IndexOf(Eonego.Lc0PolicyMap.nnIndexToUci, u)
    let h1 = idxOf "e1h1" // kingside rook square, mover perspective
    let a1 = idxOf "e1a1" // queenside rook square, mover perspective
    let e1, g1, c1 = mkSquare 4 0, mkSquare 6 0, mkSquare 2 0
    let e8, g8, c8 = mkSquare 4 7, mkSquare 6 7, mkSquare 2 7
    Assert.True(h1 >= 0 && a1 >= 0, "e1h1/e1a1 must exist in the map")
    // White (stm=white): e1g1->e1h1, e1c1->e1a1.
    Assert.Equal(h1, Eonego.Lc0PolicyMap.moveToNNIndex false (mkCastling e1 g1))
    Assert.Equal(a1, Eonego.Lc0PolicyMap.moveToNNIndex false (mkCastling e1 c1))
    // Black (stm=black): e8g8/e8c8 flip into the mover frame -> the SAME e1h1/e1a1 slots.
    Assert.Equal(h1, Eonego.Lc0PolicyMap.moveToNNIndex true (mkCastling e8 g8))
    Assert.Equal(a1, Eonego.Lc0PolicyMap.moveToNNIndex true (mkCastling e8 c8))

[<Fact>]
let ``Lc0 forward pass: scalar==AVX2 and sane policy+value`` () =
    match tryNetPath () with
    | None -> ()
    | Some p ->
        match load p with
        | Failed r -> Assert.Fail r
        | Loaded net ->
            let pos = Position.OfFen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
            let inBuf = Array.zeroCreate<float32> (112 * 64)
            Eonego.Lc0Encoder.encodeInto pos inBuf
            // Separate scratches: `forward` returns an alias of scratch.Logits, so reusing one scratch would
            // make logitsS/logitsA point at the same (AVX2-overwritten) buffer and void the parity check.
            let scratchS = Eonego.Lc0Net.Lc0Scratch(net)
            let scratchA = Eonego.Lc0Net.Lc0Scratch(net)
            let struct (logitsS, valS) = Eonego.Lc0Net.forward false net scratchS inBuf
            let struct (logitsA, valA) = Eonego.Lc0Net.forward true net scratchA inBuf

            // scalar vs AVX2 parity (epsilon, not bit-exact: FMA + horizontal-sum reorder).
            Assert.True(abs (valS - valA) < 5e-3f, sprintf "value scalar %f vs avx2 %f" valS valA)
            let mutable maxDiff = 0.0f
            for i in 0..1857 do
                maxDiff <- max maxDiff (abs (logitsS.[i] - logitsA.[i]))
            Assert.True(maxDiff < 1e-1f, sprintf "policy logit scalar/avx2 max diff %f" maxDiff)

            // sanity
            Assert.True(valS >= -1.0f && valS <= 1.0f, sprintf "value %f out of [-1,1]" valS)
            Assert.True(
                Array.forall (fun x -> not (System.Single.IsNaN x || System.Single.IsInfinity x)) logitsS,
                "policy logits NaN/Inf"
            )

            // priors sum to ~1 and value q in [0,1]
            let moves = Eonego.Tests.TestFixtures.collectLegal pos
            let priors = Array.zeroCreate<float32> moves.Length
            let q = Eonego.Lc0Net.lc0PriorsInto true net pos moves moves.Length inBuf scratchA priors
            Assert.True(q >= 0.0f && q <= 1.0f, sprintf "q %f" q)
            Assert.True(abs (Array.sum priors - 1.0f) < 1e-3f, sprintf "priors sum %f" (Array.sum priors))

            // top policy move from startpos must be a principled opening.
            let mutable bi = 0
            for i in 1 .. moves.Length - 1 do
                if priors.[i] > priors.[bi] then bi <- i
            let best = toUci moves.[bi]
            let principled = [ "e2e4"; "d2d4"; "g1f3"; "c2c4"; "g2g3"; "b1c3"; "e2e3"; "d2d3"; "f2f4" ]
            Assert.True(List.contains best principled, sprintf "startpos best policy = %s (expected principled opening)" best)

// ---------------------------------------------------------------------------
// Stage-1 gate for batched Lc0 eval: forwardBatch over B packed positions == single forward, per board.
// Each board's cells run the identical weight/accumulation sequence as the single path, so they match tightly.
// ---------------------------------------------------------------------------
[<Fact>]
let ``Lc0 forwardBatch matches single forward for every board`` () =
    match tryNetPath () with
    | None -> ()
    | Some p ->
        match load p with
        | Failed r -> Assert.Fail r
        | Loaded net ->
            let pos = Position.OfFen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
            let planes = 112
            let inBuf = Array.zeroCreate<float32> (planes * 64)
            Eonego.Lc0Encoder.encodeInto pos inBuf

            // single-position reference (AVX2).
            let sref = Eonego.Lc0Net.Lc0Scratch(net)
            let struct (logits1, val1) = Eonego.Lc0Net.forward true net sref inBuf

            // pack B copies of the same position into the batched (channel-major, nPos boards) layout.
            let B = 5
            let batchIn = Array.zeroCreate<float32> (planes * B * 64)
            for ch in 0 .. planes - 1 do
                for b in 0 .. B - 1 do
                    Array.blit inBuf (ch * 64) batchIn (ch * B * 64 + b * 64) 64

            let bscratch = Eonego.Lc0Net.Lc0BatchScratch(net, B)
            let outLogits = Array.zeroCreate<float32> (B * 1858)
            let outValues = Array.zeroCreate<float32> B
            Eonego.Lc0Net.forwardBatch true net bscratch B batchIn outLogits outValues

            for b in 0 .. B - 1 do
                Assert.True(abs (outValues.[b] - val1) < 1e-4f, sprintf "board %d value %f vs single %f" b outValues.[b] val1)
                let mutable maxDiff = 0.0f
                for i in 0 .. 1857 do
                    maxDiff <- max maxDiff (abs (outLogits.[b * 1858 + i] - logits1.[i]))
                Assert.True(maxDiff < 1e-3f, sprintf "board %d policy max diff %f" b maxDiff)

// ---------------------------------------------------------------------------
// int8 viability analysis (gates a future int8-weights effort): per-channel symmetric int8 quant+dequant of
// the dominant weight matrix (PolicyW, 1858 x 256*64 = the biggest weight stream) must barely move the policy
// logits and must NOT change the top move. If this holds, int8 weights are accuracy-safe; if not, finer
// quantization is needed before any kernel work.
// ---------------------------------------------------------------------------
[<Fact>]
let ``Lc0 int8 quantization of PolicyW keeps the policy (top move + logits)`` () =
    match tryNetPath () with
    | None -> ()
    | Some p ->
        match load p with
        | Failed r -> Assert.Fail r
        | Loaded net ->
            let pos = Position.OfFen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
            let inBuf = Array.zeroCreate<float32> (112 * 64)
            Eonego.Lc0Encoder.encodeInto pos inBuf

            // per-row (per-output-channel) symmetric int8 quant+dequant of PolicyW.
            let cols = 256 * 64
            let rows = 1858
            let pw = net.PolicyW
            let dq = Array.zeroCreate<float32> pw.Length

            for r in 0 .. rows - 1 do
                let mutable mx = 0.0f
                for c in 0 .. cols - 1 do
                    mx <- max mx (abs pw.[r * cols + c])

                let scale = if mx > 0.0f then mx / 127.0f else 1.0f

                for c in 0 .. cols - 1 do
                    let q = System.MathF.Round(pw.[r * cols + c] / scale)
                    let qc = max -127.0f (min 127.0f q)
                    dq.[r * cols + c] <- qc * scale

            let netQ = { net with PolicyW = dq }
            let struct (lf, _) = Eonego.Lc0Net.forward true net (Eonego.Lc0Net.Lc0Scratch(net)) inBuf
            let struct (lq, _) = Eonego.Lc0Net.forward true netQ (Eonego.Lc0Net.Lc0Scratch(netQ)) inBuf

            let mutable maxd = 0.0f
            let mutable af = 0
            let mutable aq = 0

            for i in 0 .. 1857 do
                maxd <- max maxd (abs (lf.[i] - lq.[i]))
                if lf.[i] > lf.[af] then af <- i
                if lq.[i] > lq.[aq] then aq <- i

            Assert.Equal(af, aq) // the top policy move must survive int8
            Assert.True(maxd < 1.0f, sprintf "int8 PolicyW max logit diff %f (logits are O(1..10))" maxd)

// ---------------------------------------------------------------------------
// Decisive int8 go/no-go gate (whole-net WEIGHT quant). The PolicyW gate above proves the single biggest
// matrix survives int8; this proves the *entire* forward does: per-output-channel symmetric int8
// quant+dequant of EVERY weight stream (input conv, all 21 residual blocks' conv1/conv2 + SE FCs, the
// policy conv + FC, and the value conv + 2 FCs), then run the full AVX2 forward and require the top policy
// move to be unchanged and the value to barely move. Error compounds across ~45 quantized layers, so this
// is a far stronger statement than the one-matrix gate. Layout is uniformly [outDim][inDim] row-major
// (verified against fcMatvec/conv call sites), so per-output-channel == per-row here.
//   NB: this validates WEIGHT quant only. A real int8 kernel also quantizes ACTIVATIONS to int8 (int32
//   accumulate); activation quant is the separate, harder gate that must pass before any kernel work.
// ---------------------------------------------------------------------------

/// Per-row (per-output-channel) symmetric int8 quant+dequant of a [rows][rowLen] weight matrix.
let private q8 (w: float32[]) (rowLen: int) : float32[] =
    let rows = w.Length / rowLen
    let dq = Array.zeroCreate<float32> w.Length

    for r in 0 .. rows - 1 do
        let b = r * rowLen
        let mutable mx = 0.0f

        for c in 0 .. rowLen - 1 do
            mx <- max mx (abs w.[b + c])

        let scale = if mx > 0.0f then mx / 127.0f else 1.0f

        for c in 0 .. rowLen - 1 do
            let qc = max -127.0f (min 127.0f (System.MathF.Round(w.[b + c] / scale)))
            dq.[b + c] <- qc * scale

    dq

let private q8conv (cb: Eonego.Lc0Proto.Lc0ConvBlock) =
    { cb with W = q8 cb.W (cb.W.Length / cb.OutC) }

let private q8se (se: Eonego.Lc0Proto.Lc0SE) =
    { se with
        W1 = q8 se.W1 (se.W1.Length / se.SeCh) // [seCh][C]
        W2 = q8 se.W2 (se.W2.Length / (2 * se.C)) } // [2C][seCh]

[<Fact>]
let ``Lc0 int8 quantization of the whole net keeps top move and value`` () =
    match tryNetPath () with
    | None -> ()
    | Some p ->
        match load p with
        | Failed r -> Assert.Fail r
        | Loaded net ->
            let pos = Position.OfFen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
            let inBuf = Array.zeroCreate<float32> (112 * 64)
            Eonego.Lc0Encoder.encodeInto pos inBuf

            let netQ =
                { net with
                    Input = q8conv net.Input
                    Tower =
                        net.Tower
                        |> Array.map (fun r ->
                            { r with
                                Conv1 = q8conv r.Conv1
                                Conv2 = q8conv r.Conv2
                                Se = q8se r.Se })
                    PolicyConv = q8conv net.PolicyConv
                    PolicyW = q8 net.PolicyW (net.PolicyW.Length / Eonego.Lc0PolicyMap.NumPolicy)
                    ValueConv = q8conv net.ValueConv
                    Value1W = q8 net.Value1W (net.Value1W.Length / net.Value1B.Length)
                    Value2W = q8 net.Value2W net.Value2W.Length }

            let struct (lf, vf) = Eonego.Lc0Net.forward true net (Eonego.Lc0Net.Lc0Scratch(net)) inBuf
            let struct (lq, vq) = Eonego.Lc0Net.forward true netQ (Eonego.Lc0Net.Lc0Scratch(netQ)) inBuf

            let mutable maxd = 0.0f
            let mutable af = 0
            let mutable aq = 0

            for i in 0 .. 1857 do
                maxd <- max maxd (abs (lf.[i] - lq.[i]))
                if lf.[i] > lf.[af] then af <- i
                if lq.[i] > lq.[aq] then aq <- i

            // The critical property: the move MCTS would prioritise must not change under int8 weights.
            Assert.Equal(af, aq)
            // Value drives leaf Q in a pure-Lc0 design; require it to barely move (eval is in [-1,1]).
            Assert.True(abs (vf - vq) < 0.05f, sprintf "whole-net int8 value %f vs %f (diff %f)" vf vq (abs (vf - vq)))
            Assert.True(maxd < 2.0f, sprintf "whole-net int8 max logit diff %f (logits are O(1..10))" maxd)

// ---------------------------------------------------------------------------
// The harder int8 gate: ACTIVATION quantization. A real int8 kernel feeds each GEMM int8 activations and
// accumulates in int32; activation precision (not weight precision) is where CNNs usually lose accuracy,
// and it compounds through 21 ReLU blocks. Lc0Net.FakeQuantActs snaps every post-ReLU activation to a
// per-tensor symmetric int8 grid (a pessimistic proxy for a real u8 kernel). We check two things:
//   (1) activations-only (fp32 weights) — isolates the new variable;
//   (2) activations + whole-net int8 weights — the actual int8 forward.
// Both must keep the top move and barely move the value. If either fails, int8 conv kernels are a no-go
// until finer (per-channel activation) quant; if both pass, activation quant is green-lit too.
// ---------------------------------------------------------------------------
[<Fact>]
let ``Lc0 int8 activation quantization keeps a near-best move and value`` () =
    match tryNetPath () with
    | None -> ()
    | Some p ->
        match load p with
        | Failed r -> Assert.Fail r
        | Loaded net ->
            // whole-net int8 weights (reuses the per-row helpers above).
            let netQ =
                { net with
                    Input = q8conv net.Input
                    Tower =
                        net.Tower
                        |> Array.map (fun r ->
                            { r with
                                Conv1 = q8conv r.Conv1
                                Conv2 = q8conv r.Conv2
                                Se = q8se r.Se })
                    PolicyConv = q8conv net.PolicyConv
                    PolicyW = q8 net.PolicyW (net.PolicyW.Length / Eonego.Lc0PolicyMap.NumPolicy)
                    ValueConv = q8conv net.ValueConv
                    Value1W = q8 net.Value1W (net.Value1W.Length / net.Value1B.Length)
                    Value2W = q8 net.Value2W net.Value2W.Length }

            let topOf (l: float32[]) =
                let mutable a = 0
                for i in 1 .. 1857 do
                    if l.[i] > l.[a] then a <- i
                a

            // Diverse positions: white/black, castling, sharp tactics, a bare K+P endgame.
            let fens =
                [ "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
                  "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
                  "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
                  "8/2k5/8/8/3K4/8/4P3/8 w - - 0 1" ]

            // Robust criterion: rather than demanding the EXACT top move (which can flip on legitimately-close
            // calls), require the move int8 picks to be near-best under the fp32 net — its fp32 logit within
            // `margin` of the fp32 best. That is exactly what search quality depends on (a near-optimal prior).
            let margin = 0.9f

            try
                for fen in fens do
                    let pos = Position.OfFen fen
                    let inBuf = Array.zeroCreate<float32> (112 * 64)
                    Eonego.Lc0Encoder.encodeInto pos inBuf

                    Eonego.Lc0Net.FakeQuantActs <- false
                    let struct (lRef, vRef) = Eonego.Lc0Net.forward true net (Eonego.Lc0Net.Lc0Scratch(net)) inBuf
                    let bestRef = topOf lRef

                    Eonego.Lc0Net.FakeQuantActs <- true
                    let struct (lA, vA) = Eonego.Lc0Net.forward true net (Eonego.Lc0Net.Lc0Scratch(net)) inBuf // acts-only
                    let struct (lAW, vAW) = Eonego.Lc0Net.forward true netQ (Eonego.Lc0Net.Lc0Scratch(netQ)) inBuf // full int8
                    Eonego.Lc0Net.FakeQuantActs <- false

                    // (1) activations-only regret + value.
                    let regretA = lRef.[bestRef] - lRef.[topOf lA]
                    Assert.True(regretA < margin, sprintf "act-only regret %f at %s" regretA fen)
                    Assert.True(abs (vRef - vA) < 0.06f, sprintf "act-only value %f vs %f at %s" vRef vA fen)

                    // (2) full int8 (acts + weights) regret + value.
                    let regretAW = lRef.[bestRef] - lRef.[topOf lAW]
                    Assert.True(regretAW < margin, sprintf "full-int8 regret %f at %s" regretAW fen)
                    Assert.True(abs (vRef - vAW) < 0.08f, sprintf "full-int8 value %f vs %f at %s" vRef vAW fen)
            finally
                Eonego.Lc0Net.FakeQuantActs <- false // never leak the toggle into other tests

// ---------------------------------------------------------------------------
// End-to-end gate for the REAL int8 forward (Lc0Net.forwardI8): fp32 input conv + int8 tower/heads + fp32 SE,
// using the actual i8Conv / i8Matvec kernels and per-channel scales (not the FakeQuantActs simulation). It
// must track the fp32 forward on diverse positions: the move it prioritises is near-best under fp32 (regret
// bound), and its value barely moves. This is the production-shaped path, so it is the binding accuracy test.
// Sparse policy FC: lc0PriorsIntoI8 computes only the legal moves' logits from polConv, which must be
// BIT-EXACT with softmaxing the legal rows of the full 1858-row i8Matvec (same i8Dot per row).
[<Fact>]
let ``Lc0 sparse policy priors are bit-exact with the full int8 forward`` () =
    match tryNetPath () with
    | None -> ()
    | Some p ->
        match load p with
        | Failed r -> Assert.Fail r
        | Loaded net ->
            let q = Eonego.Lc0Net.quantize net
            let scratch = Eonego.Lc0Net.Lc0Scratch(net)
            let qs = Eonego.Lc0Net.Lc0Int8Scratch(net)

            for fen in
                [ "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4"
                  "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1" ] do
                let pos = Position.OfFen fen
                let moves = Eonego.Tests.TestFixtures.collectLegal pos
                let n = moves.Length
                let inBuf = Array.zeroCreate<float32> (112 * 64)
                Eonego.Lc0Encoder.encodeInto pos inBuf

                // full 1858-row FC path
                let struct (logits, _) = Eonego.Lc0Net.forwardI8 true net q scratch qs inBuf
                let full = Array.zeroCreate<float32> n
                Eonego.Lc0Net.lc0PriorsFromLogits logits 0 pos moves n full

                // sparse path (production)
                let sparse = Array.zeroCreate<float32> n
                Eonego.Lc0Net.lc0PriorsIntoI8 true net q pos moves n inBuf scratch qs sparse |> ignore

                for i in 0 .. n - 1 do
                    Assert.Equal(full.[i], sparse.[i])

// ---------------------------------------------------------------------------
[<Fact>]
let ``Lc0 forwardI8 tracks the fp32 forward (near-best move + value)`` () =
    match tryNetPath () with
    | None -> ()
    | Some p ->
        match load p with
        | Failed r -> Assert.Fail r
        | Loaded net ->
            let q = Eonego.Lc0Net.quantize net
            let scratch = Eonego.Lc0Net.Lc0Scratch(net)
            let qs = Eonego.Lc0Net.Lc0Int8Scratch(net)

            let topOf (l: float32[]) =
                let mutable a = 0
                for i in 1 .. 1857 do
                    if l.[i] > l.[a] then a <- i
                a

            let fens =
                [ "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
                  "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
                  "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
                  "8/2k5/8/8/3K4/8/4P3/8 w - - 0 1"
                  "r2q1rk1/1b1nbppp/p2ppn2/1p6/3NPP2/1BN1B3/PPPQ2PP/2KR3R w - - 6 11" ] // high rule50-free middlegame

            let margin = 0.9f

            for fen in fens do
                let pos = Position.OfFen fen
                let inBuf = Array.zeroCreate<float32> (112 * 64)
                Eonego.Lc0Encoder.encodeInto pos inBuf

                let struct (lRef, vRef) = Eonego.Lc0Net.forward true net (Eonego.Lc0Net.Lc0Scratch(net)) inBuf
                let struct (lI8, vI8) = Eonego.Lc0Net.forwardI8 true net q scratch qs inBuf

                let bestRef = topOf lRef
                let regret = lRef.[bestRef] - lRef.[topOf lI8]
                Assert.True(regret < margin, sprintf "forwardI8 regret %f at %s" regret fen)
                Assert.True(abs (vRef - vI8) < 0.08f, sprintf "forwardI8 value %f vs %f at %s" vRef vI8 fen)

                // logits/value must be finite (no NaN/Inf from a bad scale or saturated dot).
                Assert.True(
                    Array.forall (fun x -> not (System.Single.IsNaN x || System.Single.IsInfinity x)) lI8,
                    sprintf "forwardI8 logits NaN/Inf at %s" fen
                )

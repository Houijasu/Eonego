module Eonego.Benchmarks.Program

#nowarn "9" // NativePtr.stackalloc in MoveGenBench

open System
open System.Numerics
open System.Runtime.Intrinsics.X86
open System.Threading
open Microsoft.FSharp.NativeInterop
open BenchmarkDotNet.Attributes
open BenchmarkDotNet.Running
open Eonego.Bitboard
open Eonego.Move
open Eonego.Position
open Eonego.NNUE
open Eonego.MoveGeneration
open Eonego.History
open Eonego.MovePick
open Eonego.Transposition
open Eonego.Search

/// Walk up to the repo root (holds Eonego.slnx) and load the real FullThreats net if present. Shared by the
/// NNUE benchmarks so they exercise the actual evaluator instead of no-opping on a stale filename.
let private loadFullThreatsNet () : Network option =
    let mutable dir = System.IO.DirectoryInfo(System.AppContext.BaseDirectory)
    let mutable root = None

    while root.IsNone && not (isNull dir) do
        if System.IO.File.Exists(System.IO.Path.Combine(dir.FullName, "Eonego.slnx")) then
            root <- Some dir.FullName

        dir <- dir.Parent

    match root with
    | Some r ->
        let p = System.IO.Path.Combine(r, "nets", "main.nnue")
        if System.IO.File.Exists p then (match load p with Loaded n -> Some n | _ -> None) else None
    | None -> None

/// Bit-serialization loop shoot-out.
///
/// The engine's hottest inner loop walks the set bits of a bitboard (one per
/// piece / target square) during move generation. We compare the two classic
/// ways to advance to the next bit:
///   A. ResetAnd  — clear the lowest set bit with `b &= b - 1`  (BLSR on BMI1)
///   B. ClearXor  — clear the *found* bit with `b ^= (1UL << sq)`
/// plus a direct-intrinsic baseline (TZCNT + BLSR via Bmi1.X64).
///
/// Workload: 4096 bitboards across a spread of densities (8..48 bits), so the
/// result reflects real move-gen occupancy rather than one pathological case.
[<MemoryDiagnoser>]
[<ShortRunJob>]
type BitloopBench() =

    let data: uint64[] = Array.zeroCreate 4096

    [<GlobalSetup>]
    member _.Setup() =
        let rng = Random(20260620)

        let r () =
            uint64 (rng.NextInt64()) ||| ((uint64 (rng.Next(0, 2))) <<< 63)

        for i in 0 .. data.Length - 1 do
            data.[i] <-
                match i % 4 with
                | 0 -> r () &&& r () &&& r () // very sparse (~8 bits)
                | 1 -> r () &&& r () // sparse      (~16 bits)
                | 2 -> r () // medium      (~32 bits)
                | _ -> r () ||| r () // dense       (~48 bits)

    /// A — reset the lowest set bit via `b &= b - 1` (this is what popLsb uses).
    [<Benchmark(Baseline = true)>]
    member _.ResetAnd() =
        let mutable acc = 0

        for i in 0 .. data.Length - 1 do
            let mutable b = data.[i]

            while b <> 0UL do
                let sq = BitOperations.TrailingZeroCount b
                b <- b &&& (b - 1UL)
                acc <- acc + sq

        acc

    /// B — clear the found bit via `b ^= (1UL <<< sq)`.
    [<Benchmark>]
    member _.ClearXor() =
        let mutable acc = 0

        for i in 0 .. data.Length - 1 do
            let mutable b = data.[i]

            while b <> 0UL do
                let sq = BitOperations.TrailingZeroCount b
                b <- b ^^^ (1UL <<< sq)
                acc <- acc + sq

        acc

    /// Direct BMI1 intrinsics: TZCNT + ResetLowestSetBit (BLSR). Falls back to A
    /// on hardware without BMI1 so the run never throws.
    [<Benchmark>]
    member _.Bmi1Direct() =
        let mutable acc = 0

        if Bmi1.X64.IsSupported then
            for i in 0 .. data.Length - 1 do
                let mutable b = data.[i]

                while b <> 0UL do
                    acc <- acc + int (Bmi1.X64.TrailingZeroCount b)
                    b <- Bmi1.X64.ResetLowestSetBit b
        else
            for i in 0 .. data.Length - 1 do
                let mutable b = data.[i]

                while b <> 0UL do
                    let sq = BitOperations.TrailingZeroCount b
                    b <- b &&& (b - 1UL)
                    acc <- acc + sq

        acc

/// Attack-lookup baseline: PEXT vs magic for the sliders, plus a leaper lookup.
/// 4096 (square, random-occupancy) pairs XOR-folded so nothing is optimized away.
[<MemoryDiagnoser>]
[<ShortRunJob>]
type AttackBench() =

    let n = 4096
    let sqs = Array.zeroCreate n: int[]
    let occ = Array.zeroCreate n: uint64[]

    [<GlobalSetup>]
    member _.Setup() =
        let rng = Random(99)

        for i in 0 .. n - 1 do
            sqs.[i] <- rng.Next(64)
            occ.[i] <- uint64 (rng.NextInt64()) ||| ((uint64 (rng.Next(0, 2))) <<< 63)

    [<Benchmark(Baseline = true)>]
    member _.RookMagic() =
        let mutable a = 0UL

        for i in 0 .. n - 1 do
            a <- a ^^^ rookAttacksMagic sqs.[i] occ.[i]

        a

    [<Benchmark>]
    member _.RookPext() =
        let mutable a = 0UL

        for i in 0 .. n - 1 do
            a <- a ^^^ rookAttacksPext sqs.[i] occ.[i]

        a

    [<Benchmark>]
    member _.BishopMagic() =
        let mutable a = 0UL

        for i in 0 .. n - 1 do
            a <- a ^^^ bishopAttacksMagic sqs.[i] occ.[i]

        a

    [<Benchmark>]
    member _.BishopPext() =
        let mutable a = 0UL

        for i in 0 .. n - 1 do
            a <- a ^^^ bishopAttacksPext sqs.[i] occ.[i]

        a

    [<Benchmark>]
    member _.QueenUnified() =
        let mutable a = 0UL

        for i in 0 .. n - 1 do
            a <- a ^^^ queenAttacks sqs.[i] occ.[i]

        a

    [<Benchmark>]
    member _.KnightLookup() =
        let mutable a = 0UL

        for i in 0 .. n - 1 do
            a <- a ^^^ knightAttacks sqs.[i]

        a

/// Move encode/decode/UCI throughput. 4096 mixed moves (normal/promo/ep/castle),
/// folded so nothing is elided. The hot encode/decode/byref paths should report
/// 0 B/op; the cold UCI string paths (ToUCI/ParseUCI) are benched separately to
/// document their allocation profile.
[<MemoryDiagnoser>]
[<ShortRunJob>]
type MoveBench() =

    let n = 4096
    let moves = Array.zeroCreate n: int[]
    let scored = Array.zeroCreate n: ScoredMove[]
    let strs = Array.zeroCreate n: string[]
    let froms = Array.zeroCreate n: int[]
    let dsts = Array.zeroCreate n: int[]

    [<GlobalSetup>]
    member _.Setup() =
        let rng = Random(20260620)
        let promos = [| Knight; Bishop; Rook; Queen |]

        for i in 0 .. n - 1 do
            let from = rng.Next(64)
            let mutable dst = rng.Next(64)

            if dst = from then
                dst <- (dst + 1) &&& 63

            froms.[i] <- from
            dsts.[i] <- dst

            let m =
                match i % 4 with
                | 0 -> mkMove from dst
                | 1 -> mkPromotion from dst (promos.[rng.Next(4)])
                | 2 -> mkEnPassant from dst
                | _ -> mkCastling from dst

            moves.[i] <- m
            scored.[i] <- mkScored m (rng.Next())
            strs.[i] <- toUCI m

    /// Encode then decode: build a move and fold its decoded fields (0 B/op expected).
    [<Benchmark(Baseline = true)>]
    member _.EncodeDecode() =
        let mutable acc = 0

        for i in 0 .. n - 1 do
            let m = mkMove froms.[i] dsts.[i]
            acc <- acc ^^^ fromSq m ^^^ toSq m ^^^ moveFlag m

        acc

    /// Decode-only over pre-built moves.
    [<Benchmark>]
    member _.DecodeOnly() =
        let mutable acc = 0

        for i in 0 .. n - 1 do
            let m = moves.[i]
            acc <- acc ^^^ fromSq m ^^^ toSq m ^^^ moveFlag m

        acc

    /// Butterfly history-key extraction.
    [<Benchmark>]
    member _.FromToKey() =
        let mutable acc = 0

        for i in 0 .. n - 1 do
            acc <- acc ^^^ fromTo moves.[i]

        acc

    /// TT compaction round-trip (pack to uint16, unpack back).
    [<Benchmark>]
    member _.Packed16RoundTrip() =
        let mutable acc = 0

        for i in 0 .. n - 1 do
            acc <- acc ^^^ ofPacked (packed16 moves.[i])

        acc

    /// Copy-free byref read of the [<Struct; IsReadOnly>] ScoredMove.
    [<Benchmark>]
    member _.ScoredByRefFold() =
        let mutable acc = 0

        for i in 0 .. n - 1 do
            let sm = &scored.[i]
            acc <- acc ^^^ sm.Move ^^^ sm.Score

        acc

    /// COLD path: format to a UCI string (allocates).
    [<Benchmark>]
    member _.ToUCI() =
        let mutable acc = 0

        for i in 0 .. n - 1 do
            acc <- acc ^^^ (toUCI moves.[i]).Length

        acc

    /// COLD path: parse from a UCI string (allocates / branches).
    [<Benchmark>]
    member _.ParseUCI() =
        let mutable acc = 0

        for i in 0 .. n - 1 do
            acc <- acc ^^^ parseUCI strs.[i]

        acc

/// Move-generation throughput + allocation profile. The bulk generators and perft must report 0 B/op
/// (caller-owned stackalloc Span<Move>, no module state); perft is the strongest 0-alloc proof since it
/// stackallocs per recursion frame.
[<MemoryDiagnoser>]
[<ShortRunJob>]
type MoveGenBench() =

    let fens =
        [| "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" // startpos
           "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -" // Kiwipete
           "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -" |] // endgame

    let mutable positions: Position[] = [||]

    [<GlobalSetup>]
    member _.Setup() =
        positions <- fens |> Array.map Position.OfFen

    /// Pseudo-legal bulk generation across the fixed position set (0 B/op expected).
    [<Benchmark(Baseline = true)>]
    member _.GenerateNonEvasions() =
        let pbuf = NativePtr.stackalloc<Move> 256
        let buf = Span<Move>(NativePtr.toVoidPtr pbuf, 256)
        let mutable acc = 0

        for pos in positions do
            acc <- acc + generate pos buf NonEvasions

        acc

    /// Fully-legal generation (pseudo-legal + isLegal filter) over the same set (0 B/op expected).
    [<Benchmark>]
    member _.GenerateLegal() =
        let pbuf = NativePtr.stackalloc<Move> 256
        let buf = Span<Move>(NativePtr.toVoidPtr pbuf, 256)
        let mutable acc = 0

        for pos in positions do
            acc <- acc + generateLegal pos buf

        acc

    /// perft throughput headlines — stackalloc per recursion frame, 0 B/op managed.
    [<Benchmark>]
    member _.PerftStartpos4() = perft positions.[0] 4

    [<Benchmark>]
    member _.PerftKiwipete3() = perft positions.[1] 3

/// MovePick drain + early-cutoff + SEE throughput. The picker drain must be 0 B/op: both buffers are
/// caller-owned stackalloc Spans, mkMain/nextMove are allocation-free, and Tables is built once in setup.
/// The lazy-cutoff benchmark takes only the first two moves (a simulated beta cutoff) — structurally it
/// never reaches QuietInit, so generate(Quiets) is never called; it should be both 0 B/op and much cheaper
/// than the full drain.
[<MemoryDiagnoser>]
[<ShortRunJob>]
type MovePickBench() =

    let fens =
        [| "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" // startpos
           "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -" // Kiwipete
           "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -" |] // endgame

    let mutable positions: Position[] = [||]
    let mutable tables = Unchecked.defaultof<Tables>

    [<GlobalSetup>]
    member _.Setup() =
        positions <- fens |> Array.map Position.OfFen
        tables <- Tables()

    /// Construct + fully drain the staged picker over the fixed set (0 B/op expected).
    [<Benchmark(Baseline = true)>]
    member _.PickerDrainFull() =
        let pm = NativePtr.stackalloc<Move> 256
        let moves = Span<Move>(NativePtr.toVoidPtr pm, 256)
        let ps = NativePtr.stackalloc<int> 256
        let scores = Span<int>(NativePtr.toVoidPtr ps, 256)
        let mutable acc = 0

        for pos in positions do
            let mutable mp =
                mkMain pos tables MoveNone MoveNone MoveNone MoveNone 8 -1 -1 -1 -1 moves scores

            let mutable m = nextMove &mp false

            while m <> MoveNone do
                acc <- acc ^^^ m
                m <- nextMove &mp false

        acc

    /// Construct + take only the first two moves (simulated fail-high) — quiets are never generated.
    [<Benchmark>]
    member _.PickerLazyCutoff() =
        let pm = NativePtr.stackalloc<Move> 256
        let moves = Span<Move>(NativePtr.toVoidPtr pm, 256)
        let ps = NativePtr.stackalloc<int> 256
        let scores = Span<int>(NativePtr.toVoidPtr ps, 256)
        let mutable acc = 0

        for pos in positions do
            let mutable mp =
                mkMain pos tables MoveNone MoveNone MoveNone MoveNone 8 -1 -1 -1 -1 moves scores

            acc <- acc ^^^ nextMove &mp false
            acc <- acc ^^^ nextMove &mp false

        acc

    /// SEE throughput: see_ge over every capture of the (capture-rich) positions (0 B/op expected).
    [<Benchmark>]
    member _.SeeGeDrain() =
        let pm = NativePtr.stackalloc<Move> 256
        let buf = Span<Move>(NativePtr.toVoidPtr pm, 256)
        let mutable acc = 0

        for pos in positions do
            let n = generate pos buf Captures

            for i in 0 .. n - 1 do
                if pos.SeeGe buf.[i] 0 then
                    acc <- acc + 1

        acc

/// Static eval throughput for the FullThreats NNUE — the sole evaluator. Soft-skips (does nothing)
/// if `nets/main.nnue` is absent. These four benchmarks DECOMPOSE the per-node cost so the
/// eval-vs-make-time-threat-tracking question can be settled empirically (the previous EvalBench loaded a
/// stale filename and measured nothing; it also only exercised the unbound from-scratch path, never the
/// production incremental delta-apply). All return an XOR-fold so BDN cannot dead-code-eliminate the loop.
///   EvalFromScratch  : unbound positions -> from-scratch oracle (buildAcc). Reference / upper bound.
///   EvalBoundRoot    : net bound, root frame already computed -> FORWARD PASS ONLY (FT product + GEMVs).
///   MakeUnmake       : make/unmake over a move list, NO eval -> pure make-time threat tracking cost.
///   MakeEvalUnmake   : make/eval/unmake -> make + incremental delta-apply + forward.
/// Decomposition: make-time ~= MakeUnmake; incremental-apply ~= (MakeEvalUnmake - MakeUnmake) - EvalBoundRoot;
/// forward ~= EvalBoundRoot. If MakeUnmake dominates MakeEvalUnmake, make-time threat tracking is the bottleneck.
[<MemoryDiagnoser>]
[<ShortRunJob>]
type EvalBench() =

    let fens =
        [| "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" // startpos (phase 24)
           "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -" // Kiwipete (midgame)
           "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -" // rook endgame (low phase)
           "8/8/8/3k4/8/3K4/4P3/8 w - -" |] // pawn endgame (phase 0)

    // Capture-rich position for the bound make/eval/unmake benches (heavy threat churn per move).
    let richFen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -" // Kiwipete

    let mutable positions: Position[] = [||]
    let mutable net: Network option = None
    let mutable bound = Unchecked.defaultof<Position>
    let moves: Move[] = Array.zeroCreate 256
    let mutable nMoves = 0

    [<GlobalSetup>]
    member _.Setup() =
        positions <- fens |> Array.map Position.OfFen
        net <- loadFullThreatsNet ()

        match net with
        | Some n ->
            bound <- Position.OfFen richFen
            Eonego.NNUE.bindNNUE n bound // Active = true; root frame (0) materialized -> incremental path live
            nMoves <- generateLegal bound (Span<Move>(moves))
        | None -> ()

    /// Unbound from-scratch oracle path (reference upper bound). Does nothing if no net is loaded.
    [<Benchmark(Baseline = true)>]
    member _.EvalFromScratch() =
        let mutable acc = 0

        match net with
        | Some n ->
            for pos in positions do
                acc <- acc ^^^ evalCp n pos
        | None -> ()

        acc

    /// Forward pass only: the bound root stays computed across calls, so EnsureComputed is a no-op.
    [<Benchmark>]
    member _.EvalBoundRoot() =
        match net with
        | Some n -> evalCp n bound
        | None -> 0

    /// Forward pass via the AVX2 (double-MultiplyAddAdjacent) GEMV path — same-run baseline for VNNI.
    [<Benchmark>]
    member _.ForwardAvx2() =
        match net with
        | Some n -> evalInternal n bound true false false
        | None -> 0

    /// Forward pass via the AVX-VNNI (vpdpbusd) GEMV path. Compare to ForwardAvx2 IN THE SAME RUN (cross-run
    /// machine state varies ~2x from thermal/turbo, so only the in-run ratio is meaningful).
    [<Benchmark>]
    member _.ForwardVnni() =
        match net with
        | Some n -> evalInternal n bound true true UseSparse
        | None -> 0

    /// Pure make-time threat tracking (UpdatePieceThreats / RayBeyond + dirty recording), no eval.
    [<Benchmark>]
    member _.MakeUnmake() =
        let mutable acc = 0

        if net.IsSome then
            for i in 0 .. nMoves - 1 do
                let m = moves.[i]
                bound.Make m
                acc <- acc ^^^ m
                bound.Unmake m

        acc

    /// Make + incremental delta-apply (1-frame) + forward, the real production per-node cost.
    [<Benchmark>]
    member _.MakeEvalUnmake() =
        let mutable acc = 0

        match net with
        | Some n ->
            for i in 0 .. nMoves - 1 do
                let m = moves.[i]
                bound.Make m
                acc <- acc ^^^ evalCp n bound
                bound.Unmake m
        | None -> ()

        acc

/// Fixed-depth search allocation profile. The Worker (per-thread stack/PV/move/score/quiet buffers) and
/// the TT are built ONCE in Setup, so the per-op figure measures STEADY-STATE search allocation only —
/// expected 0 B/op (preallocated buffers, the byref-like picker on the stack, 0-B/op eval, no boxing).
[<MemoryDiagnoser>]
[<ShortRunJob>]
type SearchBench() =

    let fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -" // Kiwipete (rich tree)
    let mutable worker = Unchecked.defaultof<Worker>
    let mutable net: Network option = None

    [<GlobalSetup>]
    member _.Setup() =
        net <- loadFullThreatsNet () // bind the real net so evalPos exercises NNUE (was None -> eval=0)

        let cfg =
            { defaultConfig with
                Threads = 1
                HashMb = 16
                UseTt = true
                UsePruning = true }

        let tt = TranspositionTable(16)
        let control = SearchControl(cfg, defaultLimits, tt, fen, [||], ?net = net)
        worker <- Worker(0, true, control)
        worker.SetupRoot()
        control.Reset()
        control.StartClock 0L 0L

    /// One fixed-depth full-window search over the preallocated worker. 0 B/op expected.
    [<Benchmark>]
    member _.SearchDepth7() =
        negamax worker worker.Pos (-INF) INF 7 0 true false

/// Phase 1 — NNUE accumulator checkpoint cache A/B (cache-on vs cache-off). Pins the contribution of the
/// lock-free checkpoint to steady-state search nps: cache-off is the pre-Phase-1 baseline; cache-on pays a
/// ~4 KiB BlockCopy+hash per materialization but saves the O(distance) frame-delta walk on hits.
[<MemoryDiagnoser>]
[<ShortRunJob>]
type AccCheckpointSearchBench() =

    let fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -" // Kiwipete (rich tree)
    let mutable workerOn = Unchecked.defaultof<Worker>
    let mutable workerOff = Unchecked.defaultof<Worker>
    let mutable net: Network option = None

    [<GlobalSetup>]
    member _.Setup() =
        net <- loadFullThreatsNet ()

        let cfgOn =
            { defaultConfig with
                Threads = 1
                HashMb = 16
                UseTt = true
                UsePruning = true
                AccCheckpointMb = 4 }

        let cfgOff =
            { defaultConfig with
                Threads = 1
                HashMb = 16
                UseTt = true
                UsePruning = true
                AccCheckpointMb = 0 }

        let ttOn = TranspositionTable(16)
        let controlOn = SearchControl(cfgOn, defaultLimits, ttOn, fen, [||], ?net = net)
        workerOn <- Worker(0, true, controlOn)
        workerOn.SetupRoot()
        controlOn.Reset()
        controlOn.StartClock 0L 0L

        let ttOff = TranspositionTable(16)
        let controlOff = SearchControl(cfgOff, defaultLimits, ttOff, fen, [||], ?net = net)
        workerOff <- Worker(0, true, controlOff)
        workerOff.SetupRoot()
        controlOff.Reset()
        controlOff.StartClock 0L 0L

    [<Benchmark(Baseline = true)>]
    member _.SearchDepth7CacheOff() =
        negamax workerOff workerOff.Pos (-INF) INF 7 0 true false

    [<Benchmark>]
    member _.SearchDepth7CacheOn() =
        negamax workerOn workerOn.Pos (-INF) INF 7 0 true false

/// Multi-threaded entry for benchmarking only: mirrors `Search.go`'s thread topology (N workers sharing one
/// SearchControl - only the lock-free TT/DAG/checkpoint tables are shared mutable state) but stops on a NODE
/// budget instead of depth/time and skips the UCI `bestmove` stdout write, so repeated BenchmarkDotNet
/// invocations stay quiet and the wall-clock budget stays roughly comparable across thread counts.
let private searchNodesMultiThreaded (fen: string) (nodes: int64) (cfg: SearchConfig) (net: Network option) : int64 =
    let tt = TranspositionTable(max 1 cfg.HashMb)
    let limits = { defaultLimits with Nodes = nodes }
    let control = SearchControl(cfg, limits, tt, fen, [||], ?net = net)
    control.Reset()
    control.NewSearch()
    let n = max 1 cfg.Threads
    let workers = Array.init n (fun i -> Worker(i, (i = 0), control))

    for w in workers do
        w.SetupRoot()

    control.NodeSum <-
        (fun () ->
            let mutable s = 0L

            for wk in workers do
                s <- s + wk.Nodes

            s)

    control.StartClock 0L 0L

    let threads =
        [| for i in 1 .. n - 1 ->
               let w = workers.[i]
               let t = Thread(ThreadStart(fun () -> iterativeDeepening w (MaxSearchPly - 1)), 16 * 1024 * 1024)
               t.IsBackground <- true
               t.Start()
               t |]

    iterativeDeepening workers.[0] (MaxSearchPly - 1)
    control.Stop()

    for t in threads do
        t.Join()

    let mutable total = 0L

    for w in workers do
        total <- total + w.Nodes

    total

/// Phase 1 multi-threaded A/B (checkpoint-cache-on vs off), isolating the AccCheckpoint contribution
/// at real concurrency instead of the Threads=1-only `AccCheckpointSearchBench` above.
[<MemoryDiagnoser>]
[<ShortRunJob>]
type AccCheckpointSearchMtBench() =

    let fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -" // Kiwipete (rich tree)
    let nodeBudget = 200_000L
    let mutable net: Network option = None

    [<Params(1, 4, 8)>]
    member val Threads = 1 with get, set

    [<GlobalSetup>]
    member _.Setup() =
        net <- loadFullThreatsNet ()

    [<Benchmark(Baseline = true)>]
    member this.SearchNodesCacheOff() =
        let cfg =
            { defaultConfig with
                Threads = this.Threads
                HashMb = 16
                UseTt = true
                UsePruning = true
                AccCheckpointMb = 0 }

        searchNodesMultiThreaded fen nodeBudget cfg net

    [<Benchmark>]
    member this.SearchNodesCacheOn() =
        let cfg =
            { defaultConfig with
                Threads = this.Threads
                HashMb = 16
                UseTt = true
                UsePruning = true
                AccCheckpointMb = 4 }

        searchNodesMultiThreaded fen nodeBudget cfg net

[<EntryPoint>]
let main argv =
    // `--filter *` runs every benchmark non-interactively when no args are given.
    let args = if Array.isEmpty argv then [| "--filter"; "*" |] else argv

    BenchmarkSwitcher
        .FromTypes(
            [| typeof<BitloopBench>
               typeof<AttackBench>
               typeof<MoveBench>
               typeof<MoveGenBench>
               typeof<MovePickBench>
               typeof<EvalBench>
               typeof<SearchBench>
               typeof<AccCheckpointSearchBench>
               typeof<AccCheckpointSearchMtBench> |]
        )
        .Run(args)
    |> ignore

    0

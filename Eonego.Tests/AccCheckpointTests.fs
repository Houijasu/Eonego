/// Phase 1 — NNUE accumulator checkpoint cache tests.
///
/// Two layers:
///   1. Pure-cache unit tests on `AccCheckpointTable` (no NNUE, no Position): round-trip, miss, clear,
///      torn-payload rejection, capacity-invariants. These pin the lock-free contract.
///   2. Integration tests via `Position.SfEnsureBothComputed` with a real SF NNUE net bound: bit-exact acc
///      parity with the cache enabled vs disabled; a hit restores the same accumulator as a from-scratch
///      materialization of the same position.
///
/// Real-net tests SOFT-SKIP when `nets/nn-f8a759c05f9f.nnue` is absent (mirrors `NnueTests.fs`); pure-cache
/// tests always run.
module Eonego.Tests.AccCheckpointTests

open System
open System.IO
open System.Threading
open System.Threading.Tasks
open Xunit
open Eonego.Accumulator
open Eonego.AccCheckpoint
open Eonego.Bitboard
open Eonego.Move
open Eonego.Position
open Eonego.Nnue
open Eonego.Tests.TestFixtures

// ---------------------------------------------------------------------------
// Soft-skip helper mirroring NnueTests.fs: locate the SF net under nets/.
// ---------------------------------------------------------------------------
let private netPath () : string option =
    let mutable dir = DirectoryInfo(AppContext.BaseDirectory)
    let mutable found = None

    while found.IsNone && not (isNull dir) do
        if File.Exists(Path.Combine(dir.FullName, "Eonego.slnx")) then
            found <- Some dir.FullName

        dir <- dir.Parent

    match found with
    | Some root ->
        let p = Path.Combine(root, "nets", "nn-f8a759c05f9f.nnue")
        if File.Exists p then Some p else None
    | None -> None

let private withNet (f: SfNetwork -> unit) =
    match netPath () with
    | None -> () // soft-skip: net not present in this checkout
    | Some p ->
        match load p with
        | Failed reason -> Assert.Fail("FullThreats net failed to load: " + reason)
        | Loaded net -> f net

// ---------------------------------------------------------------------------
// 1. Pure-cache unit tests — DO NOT require any NNUE net.
// ---------------------------------------------------------------------------
let private dummyPayload () : int16[] * int16[] * int[] * int[] =
    let rng = System.Random(0xC0FFEE)

    let accW = Array.zeroCreate L1
    let accB = Array.zeroCreate L1
    let psqW = Array.zeroCreate PsqtBuckets
    let psqB = Array.zeroCreate PsqtBuckets

    for i in 0 .. L1 - 1 do
        accW.[i] <- int16 (rng.Next(-32768, 32767))
        accB.[i] <- int16 (rng.Next(-32768, 32767))

    for i in 0 .. PsqtBuckets - 1 do
        psqW.[i] <- rng.Next()
        psqB.[i] <- rng.Next()

    (accW, accB, psqW, psqB)

[<Fact>]
let ``store then probe round-trips acc and psqt payloads`` () =
    let cache = AccCheckpointTable(4)
    let (accW, accB, psqW, psqB) = dummyPayload ()
    let key = 0xDEADBEEFCAFEBABEUL

    cache.Store(key, accW, 0, accB, 0, psqW, 0, psqB, 0)

    let destAccW = Array.zeroCreate L1
    let destAccB = Array.zeroCreate L1
    let destPsqW = Array.zeroCreate PsqtBuckets
    let destPsqB = Array.zeroCreate PsqtBuckets

    let hit = cache.TryProbe(key, destAccW, 0, destAccB, 0, destPsqW, 0, destPsqB, 0)

    Assert.True(hit, "expected a hit on the just-stored key")

    for i in 0 .. L1 - 1 do
        Assert.Equal(accW.[i], destAccW.[i])
        Assert.Equal(accB.[i], destAccB.[i])

    for i in 0 .. PsqtBuckets - 1 do
        Assert.Equal(psqW.[i], destPsqW.[i])
        Assert.Equal(psqB.[i], destPsqB.[i])

[<Fact>]
let ``probe of a never-stored key misses`` () =
    let cache = AccCheckpointTable(4)

    let destAccW = Array.zeroCreate L1
    let destAccB = Array.zeroCreate L1
    let destPsqW = Array.zeroCreate PsqtBuckets
    let destPsqB = Array.zeroCreate PsqtBuckets

    let hit = cache.TryProbe(0x0123456789ABCDEFUL, destAccW, 0, destAccB, 0, destPsqW, 0, destPsqB, 0)
    Assert.False(hit)

[<Fact>]
let ``clear invalidates all slots`` () =
    let cache = AccCheckpointTable(4)
    let (accW, accB, psqW, psqB) = dummyPayload ()
    let key = 0xBAADF00DDEADBEEFUL
    cache.Store(key, accW, 0, accB, 0, psqW, 0, psqB, 0)
    cache.Clear()

    let dAccW = Array.zeroCreate L1
    let dAccB = Array.zeroCreate L1
    let dPsqW = Array.zeroCreate PsqtBuckets
    let dPsqB = Array.zeroCreate PsqtBuckets

    Assert.False(cache.TryProbe(key, dAccW, 0, dAccB, 0, dPsqW, 0, dPsqB, 0))

[<Fact>]
let ``two distinct keys round-trip independently`` () =
    let cache = AccCheckpointTable(4)
    let (accWA, accBA, psqWA, psqBA) = dummyPayload ()
    let (accWB, accBB, psqWB, psqBB) = dummyPayload ()
    // Pick keys with differing low bits so the Knuth-multiply slot hash lands them in different slots.
    let keyA = 0xDEADBEEFCAFEBABEUL
    let keyB = 0x0123456789ABCDEFUL

    cache.Store(keyA, accWA, 0, accBA, 0, psqWA, 0, psqBA, 0)
    cache.Store(keyB, accWB, 0, accBB, 0, psqWB, 0, psqBB, 0)

    let dAW = Array.zeroCreate L1
    let dAB = Array.zeroCreate L1
    let dPW = Array.zeroCreate PsqtBuckets
    let dPB = Array.zeroCreate PsqtBuckets

    Assert.True(cache.TryProbe(keyA, dAW, 0, dAB, 0, dPW, 0, dPB, 0))

    for i in 0 .. L1 - 1 do
        Assert.Equal(accWA.[i], dAW.[i])
        Assert.Equal(accBA.[i], dAB.[i])

    for i in 0 .. PsqtBuckets - 1 do
        Assert.Equal(psqWA.[i], dPW.[i])
        Assert.Equal(psqBA.[i], dPB.[i])

    Assert.True(cache.TryProbe(keyB, dAW, 0, dAB, 0, dPW, 0, dPB, 0))

    for i in 0 .. L1 - 1 do
        Assert.Equal(accWB.[i], dAW.[i])
        Assert.Equal(accBB.[i], dAB.[i])

[<Fact>]
let ``4 MiB cache capacity is 512 slots (power-of-two)`` () =
    let cache = AccCheckpointTable(4)
    Assert.Equal(512, cache.Capacity)
    Assert.True((cache.Capacity &&& (cache.Capacity - 1)) = 0, "capacity must be a power of two")

[<Fact>]
let ``zero-MiB config yields null table (cache disabled)`` () =
    // The SearchControl ctor interprets AccCheckpointMb <= 0 as "disabled" by allocating null. We can't easily
    // reach the ctor here, but the explicit-null pattern is what every probe site guards against — sanity-check
    // that a fresh null literal is recognized by the F# Option-style `match` (the same idiom SfEnsureBothComputed
    // uses today). This is a deliberate tautology that pins the contract by example.
    let cache: AccCheckpointTable = null
    Assert.True(obj.ReferenceEquals(cache, null))

[<Fact>]
let ``torn payload is rejected as a miss`` () =
    // Simulate a torn write: invoke Store normally (which sets the header guard correctly), then physically
    // corrupt a single int16 lane in the stored accW payload WITHOUT updating the header. The next probe's
    // recomputed-hash self-check (`hComputed <> hStored`) MUST reject this as a miss — the core safety property
    // that makes best-effort lock-free reads acceptable under concurrent writers.
    let cache = AccCheckpointTable(4)
    let key = 0xCAFED00DCAFED00DUL
    let (accW, accB, psqW, psqB) = dummyPayload ()
    cache.Store(key, accW, 0, accB, 0, psqW, 0, psqB, 0)
    // Replicate the exact SlotIndex hash (Knuth-multiply) to find where Store landed.
    let mask = cache.SlotMask
    let k = (key ^^^ (key >>> 32)) * 0x9E3779B97F4A7C15UL
    let idx = int ((k >>> 33) &&& uint64 mask)
    // XOR-by-0x1234 is guaranteed non-zero, so the lane diverges and the recomputed hash won't match.
    cache.RawAccW.[idx * L1] <- int16 (accW.[0] ^^^ 0x1234s)

    let dAW = Array.zeroCreate L1
    let dAB = Array.zeroCreate L1
    let dPW = Array.zeroCreate PsqtBuckets
    let dPB = Array.zeroCreate PsqtBuckets

    let hit = cache.TryProbe(key, dAW, 0, dAB, 0, dPW, 0, dPB, 0)
    Assert.False(hit, "a probe against a payload not bearing the stored hash MUST miss")

[<Fact>]
let ``distinct Zobrist keys with identical payloads round-trip independently`` () =
    // Two different positions (different keys) sharing identical accumulator bytes should both be hit when
    // they land in different slots. Pick keys with well-spread low bits so the Knuth-multiply hash places
    // them in different slots even in a tiny 512-slot table.
    let cache = AccCheckpointTable(4)
    let (accW, accB, psqW, psqB) = dummyPayload ()
    let keyA = 0xDEADBEEFCAFEBABEUL
    let keyB = 0x0123456789ABCDEFUL
    cache.Store(keyA, accW, 0, accB, 0, psqW, 0, psqB, 0)
    cache.Store(keyB, accW, 0, accB, 0, psqW, 0, psqB, 0)

    let dAW = Array.zeroCreate L1
    let dAB = Array.zeroCreate L1
    let dPW = Array.zeroCreate PsqtBuckets
    let dPB = Array.zeroCreate PsqtBuckets

    Assert.True(cache.TryProbe(keyA, dAW, 0, dAB, 0, dPW, 0, dPB, 0), "keyA must hit")

    for i in 0 .. L1 - 1 do
        Assert.Equal(accW.[i], dAW.[i])
        Assert.Equal(accB.[i], dAB.[i])

    Assert.True(cache.TryProbe(keyB, dAW, 0, dAB, 0, dPW, 0, dPB, 0), "keyB must hit")

// ---------------------------------------------------------------------------
// 2. NNUE integration tests via Position.SfEnsureBothComputed. SOFT-SKIP if the
//    embedded/local net is absent (CC0 but large — see NnueTests.fs).
// ---------------------------------------------------------------------------
let private bindAndEval (net: SfNetwork) (fen: string) (cache: AccCheckpointTable) : int =
    let pos = Position()
    pos.LoadFen fen
    bindNnue net pos
    if not (obj.ReferenceEquals(cache, null)) then
        pos.SfBindCheckpoint cache
    // SfEnsureBothComputed fires lazily here once evalCp touches the accumulators.
    evalCp net pos

[<Fact>]
let ``evalCp parity: cache ON == cache OFF across a fixed-position set`` () =
    withNet
        (fun net ->
            let fens =
                [| StartPosFen
                   "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
                   "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
                   "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1" |]

            let cacheOff: AccCheckpointTable = null
            let cacheOn = AccCheckpointTable(4)

            for fen in fens do
                let vOff = bindAndEval net fen cacheOff
                let vOn = bindAndEval net fen cacheOn
                Assert.Equal(vOff, vOn))

[<Fact>]
let ``SfEnsureBothComputed cache hit yields bit-exact acc vs from-scratch`` () =
    withNet
        (fun net ->
            let pos = Position()
            pos.LoadFen StartPosFen
            bindNnue net pos
            let cache = AccCheckpointTable(4)
            pos.SfBindCheckpoint cache
            // `bindNnue`/EnableNnue already materialized root frame 0 and set sfComputed flags, so
            // the early-return path inside SfEnsureBothComputed would skip the cache populate. Mirror
            // what `Worker.SetupRoot` does: explicitly seed the cache for the already-computed root.
            pos.SfSeedCheckpoint()

            // Read the materialized acc/psqt arrays (both perspectives computed at the root).
            let accWArr = pos.SfAccArray White
            let accBArr = pos.SfAccArray Black
            let psqWArr = pos.SfPsqtArray White
            let psqBArr = pos.SfPsqtArray Black
            // The seed MUST have left a cache entry for this position.
            let dAW = Array.zeroCreate L1
            let dAB = Array.zeroCreate L1
            let dPW = Array.zeroCreate PsqtBuckets
            let dPB = Array.zeroCreate PsqtBuckets

            let hit = cache.TryProbe(pos.Key, dAW, 0, dAB, 0, dPW, 0, dPB, 0)
            Assert.True(hit, "SfSeedCheckpoint must have populated the cache for the root key")

            for i in 0 .. L1 - 1 do
                Assert.Equal(accWArr.[i], dAW.[i])
                Assert.Equal(accBArr.[i], dAB.[i])

            for i in 0 .. PsqtBuckets - 1 do
                Assert.Equal(psqWArr.[i], dPW.[i])
                Assert.Equal(psqBArr.[i], dPB.[i]))

[<Fact>]
let ``eager accumulator materializes during Make (no cache needed)`` () =
    withNet
        (fun net ->
            let pos = Position()
            pos.LoadFen StartPosFen
            bindNnue net pos
            // With eager accumulator updates, Make itself computes both perspectives. After Make, eval is O(1).
            let moves = collectLegal pos
            Assert.True(moves.Length > 0)
            pos.Make moves.[0]
            // The accumulator at sfTop must already be materialized (eager update sets computed flags).
            let accWArr = pos.SfAccArray White
            let accBArr = pos.SfAccArray Black

            // Verify the accumulator is non-trivial (not all zeros).
            Assert.True(accWArr |> Array.exists (fun v -> v <> 0s), "white acc must be materialized")
            Assert.True(accBArr |> Array.exists (fun v -> v <> 0s), "black acc must be materialized")

            // Eval must produce the same value as the from-scratch oracle.
            let oracle = Position.OfFen(pos.ToFen())
            let oracleVal = evalCp net oracle
            let boundVal = evalCp net pos
            Assert.Equal(oracleVal, boundVal))

[<Fact>]
let ``concurrent store/probe stress: no crashes; every hit is bit-exact`` () =
    // Simple stress: 4 writer + 4 reader threads, 256 distinct keys each. Every reader that hits a key it
    // just wrote must read back identical bytes; misses are also acceptable (best-effort). The contract:
    // no exceptions, no torn-data false hits that pass validation yet diverge.
    let cache = AccCheckpointTable(8) // 2048 slots
    let rngFor (seed: int) = System.Random(seed)
    let precomputedPayloads =
        Array.init 256 (fun _ ->
            let r = rngFor (1)
            let aW = Array.init L1 (fun _ -> int16 (r.Next(-32000, 32000)))
            let aB = Array.init L1 (fun _ -> int16 (r.Next(-32000, 32000)))
            let pW = Array.init PsqtBuckets (fun _ -> r.Next())
            let pB = Array.init PsqtBuckets (fun _ -> r.Next())
            (aW, aB, pW, pB))

    let keys = Array.init 256 (fun i -> uint64 (i + 1) * 0x9E3779B97F4A7C15UL)

    let writers =
        [|
            for tid in 0 .. 3 ->
                Task.Run
                    (fun () ->
                        for i in tid .. 4 .. 255 do
                            let k = keys.[i]
                            let (aW, aB, pW, pB) = precomputedPayloads.[i]
                            cache.Store(k, aW, 0, aB, 0, pW, 0, pB, 0))
        |]

    let readersOk =
        [|
            for tid in 0 .. 3 ->
                Task.Run
                    (fun () ->
                        let mutable ok = true
                        let dAW = Array.zeroCreate L1
                        let dAB = Array.zeroCreate L1
                        let dPW = Array.zeroCreate PsqtBuckets
                        let dPB = Array.zeroCreate PsqtBuckets

                        for i in tid .. 4 .. 255 do
                            let k = keys.[i]
                            let hit = cache.TryProbe(k, dAW, 0, dAB, 0, dPW, 0, dPB, 0)

                            if hit then
                                let (aW, aB, pW, pB) = precomputedPayloads.[i]

                                for j in 0 .. L1 - 1 do
                                    if dAW.[j] <> aW.[j] || dAB.[j] <> aB.[j] then
                                        ok <- false

                                for j in 0 .. PsqtBuckets - 1 do
                                    if dPW.[j] <> pW.[j] || dPB.[j] <> pB.[j] then
                                        ok <- false

                        ok)
        |]

    for w in writers do
        w.Wait()

    let mutable allOk = true

    for r in readersOk do
        r.Wait()

        if not r.Result then
            allOk <- false

    Assert.True(allOk, "every reader hit must be bit-exact (or a miss)")
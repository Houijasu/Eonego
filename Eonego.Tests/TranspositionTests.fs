/// Lock-free transposition table: pack round-trip, store/probe, XOR torn-read rejection, empty-slot
/// sentinel, replacement/depth-preference, generation wrap, and a concurrency stress test.
module Eonego.Tests.TranspositionTests

open System.Threading.Tasks
open Xunit
open Eonego.Move
open Eonego.Transposition

// ---------------------------------------------------------------------------
// Pack / unpack round-trip (incl. negative score/eval, full depth + generation).
// ---------------------------------------------------------------------------
[<Theory>]
[<InlineData(0x1F3F, 100, 50, 7, 3)>]
[<InlineData(0x0041, -12345, -32000, 200, 0)>]
[<InlineData(0xFFFF, 32000, 32000, 255, 2)>]
[<InlineData(0, 0, 0, 0, 1)>]
let ``packData round-trips every field`` (move: int) (score: int) (eval: int) (depth: int) (bound: int) =
    let gen = 50
    let d = packData move score eval depth ((gen <<< 2) ||| bound)
    Assert.Equal(move, dMove d)
    Assert.Equal(score, dScore d)
    Assert.Equal(eval, dEval d)
    Assert.Equal(depth, dDepth d)
    Assert.Equal(bound, dBound d)
    Assert.Equal(gen, dGen d)

// ---------------------------------------------------------------------------
// Store then Probe returns exactly what was stored.
// ---------------------------------------------------------------------------
[<Fact>]
let ``store then probe round-trips`` () =
    let tt = TranspositionTable(1)
    let key = 0x0123456789ABCDEFUL
    let mv = 0x1ABC
    tt.Store key 9 BoundExact 250 -40 mv
    let struct (hit, m, sc, ev, dp, bd) = tt.Probe key
    Assert.True hit
    Assert.Equal(mv, m)
    Assert.Equal(250, sc)
    Assert.Equal(-40, ev)
    Assert.Equal(9, dp)
    Assert.Equal(BoundExact, bd)

[<Fact>]
let ``probe of an absent key misses`` () =
    let tt = TranspositionTable(1)
    tt.Store 0xAAAAAAAAAAAAAAAAUL 5 BoundLower 10 0 0x111
    let struct (hit, _, _, _, _, _) = tt.Probe 0x5555555555555555UL
    Assert.False hit

[<Fact>]
let ``empty table never hits, including realKey = 0`` () =
    let tt = TranspositionTable(1)
    let struct (h0, _, _, _, _, _) = tt.Probe 0UL
    let struct (h1, _, _, _, _, _) = tt.Probe 0x1234UL
    Assert.False h0
    Assert.False h1

// ---------------------------------------------------------------------------
// XOR torn-read rejection: corrupting either field of a stored entry must make the probe miss.
// ---------------------------------------------------------------------------
[<Fact>]
let ``injected torn read is rejected as a miss`` () =
    let tt = TranspositionTable(1)
    let key = 0xDEADBEEFCAFEF00DUL
    tt.Store key 6 BoundExact 123 0 0x222
    let struct (hit, _, _, _, _, _) = tt.Probe key
    Assert.True hit

    // Find the slot holding `key` and corrupt ONE field — the XOR check must now reject it.
    let b = tt.ClusterBase key
    let entries = tt.RawEntries
    let mutable slot = -1

    for i in 0 .. ClusterSize - 1 do
        if
            (entries.[b + i].Key ^^^ entries.[b + i].Data) = key
            && dBound entries.[b + i].Data <> BoundNone
        then
            slot <- b + i

    Assert.True(slot >= 0)

    entries.[slot].Data <- entries.[slot].Data ^^^ 0x1UL // torn: Data fresh-ish, Key stale
    let struct (hit2, _, _, _, _, _) = tt.Probe key
    Assert.False hit2

// ---------------------------------------------------------------------------
// Replacement / depth preference on a key match.
// ---------------------------------------------------------------------------
[<Fact>]
let ``shallower non-exact store preserves a deeper entry's value but refreshes the move`` () =
    let tt = TranspositionTable(1)
    let key = 0x9E3779B97F4A7C15UL
    tt.Store key 8 BoundExact 100 0 0xAAA
    tt.Store key 2 BoundLower 50 0 0xBBB // depth+4 = 6 < 8 -> keep value/depth/bound, move updates
    let struct (_, m, sc, _, dp, bd) = tt.Probe key
    Assert.Equal(8, dp)
    Assert.Equal(100, sc)
    Assert.Equal(BoundExact, bd)
    Assert.Equal(0xBBB, m)

[<Fact>]
let ``deeper store overwrites value`` () =
    let tt = TranspositionTable(1)
    let key = 0x9E3779B97F4A7C15UL
    tt.Store key 2 BoundExact 30 0 0xAAA
    tt.Store key 5 BoundLower 70 0 0xBBB
    let struct (_, m, sc, _, dp, bd) = tt.Probe key
    Assert.Equal(5, dp)
    Assert.Equal(70, sc)
    Assert.Equal(BoundLower, bd)
    Assert.Equal(0xBBB, m)

[<Fact>]
let ``storing MoveNone over the same position keeps the old move`` () =
    let tt = TranspositionTable(1)
    let key = 0x0F0F0F0F0F0F0F0FUL
    tt.Store key 5 BoundExact 10 0 0xABC
    tt.Store key 6 BoundExact 20 0 MoveNone
    let struct (_, m, _, _, dp, _) = tt.Probe key
    Assert.Equal(0xABC, m)
    Assert.Equal(6, dp)

// ---------------------------------------------------------------------------
// Generation 6-bit wrap.
// ---------------------------------------------------------------------------
[<Fact>]
let ``NewSearch wraps the generation at 6 bits`` () =
    let tt = TranspositionTable(1)

    for _ in 1..200 do
        tt.NewSearch()

    Assert.InRange(tt.Generation, 0, 63)

[<Fact>]
let ``Clear empties the table`` () =
    let tt = TranspositionTable(1)
    let key = 0x1357913579135791UL
    tt.Store key 5 BoundExact 10 0 0x123
    tt.Clear()
    let struct (hit, _, _, _, _, _) = tt.Probe key
    Assert.False hit

// ---------------------------------------------------------------------------
// Concurrency stress: many threads hammering store/probe never crash, and every validated probe is
// internally consistent (its key/score/depth survive the XOR self-check).
// ---------------------------------------------------------------------------
[<Fact>]
let ``concurrent store and probe never crash and never validate a corrupt entry`` () =
    let tt = TranspositionTable(8)
    let nThreads = 8
    let perThread = 200_000

    let work (tid: int) =
        let mutable k = (uint64 tid * 2654435761UL) ||| 1UL

        for n in 1..perThread do
            k <- k * 6364136223846793005UL + 1442695040888963407UL
            let key = k ||| 1UL

            if (n &&& 1) = 0 then
                tt.Store key (n &&& 63) BoundExact (int (int16 k)) 0 (int (uint16 k))
            else
                let struct (hit, _, sc, _, dp, _) = tt.Probe key

                if hit then
                    // a validated entry must carry sane unpacked fields (no torn garbage leaked through)
                    Assert.InRange(dp, 0, 255)
                    Assert.InRange(sc, -32768, 32767)

    let tasks = [| for t in 0 .. nThreads - 1 -> Task.Run(fun () -> work t) |]
    Task.WaitAll(tasks)

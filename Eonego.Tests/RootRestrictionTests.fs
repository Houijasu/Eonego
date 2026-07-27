/// Root-domain correctness: `go searchmoves` and MultiPV are constrained searches, so their
/// results must neither escape the requested root set nor poison the unrestricted root TT key.
/// Node-limited LazySMP likewise owns one aggregate budget, not one budget per worker.
module Eonego.Tests.RootRestrictionTests

open Xunit
open Eonego.Move
open Eonego.Position
open Eonego.Transposition
open Eonego.Search
open Eonego.Tests.TestFixtures

let private rootWorker (cfg: SearchConfig) (limits: SearchLimits) (tt: TranspositionTable) =
    let control = SearchControl(cfg, limits, tt, StartPosFen, [||])
    control.Reset()
    control.NewSearch()
    let worker = Worker(0, true, control)
    worker.SetupRoot()
    control.NodeSum <- (fun () -> worker.Nodes)
    control.StartClock 0L 0L
    struct (control, worker)

[<Fact>]
let ``searchmoves result never escapes the allowed root set`` () =
    let legal = collectLegal (Position.OfFen StartPosFen)
    let allowed = legal.[legal.Length - 1]
    let limits = { defaultLimits with Depth = 3; SearchMoves = [| allowed |] }
    let control = SearchControl(defaultConfig, limits, TranspositionTable(1), StartPosFen, [||])
    let best = go control
    Assert.Equal(allowed, best)

[<Fact>]
let ``stopped searchmoves fallback stays inside the allowed root set`` () =
    // Reproduce a stop arriving after arm() but before the search thread starts. With no completed
    // iteration, goCore must choose its fallback from SearchMoves rather than all legal root moves.
    let pos = Position.OfFen StartPosFen
    let legal = collectLegal pos
    let allowed = legal.[legal.Length - 1]
    Assert.NotEqual(firstLegalMove pos, allowed) // prove this fixture exercises the fallback restriction

    let limits = { defaultLimits with SearchMoves = [| allowed |] }
    let control = SearchControl(defaultConfig, limits, TranspositionTable(1), StartPosFen, [||])
    arm control
    control.Stop()
    let best = goArmed control
    Assert.Equal(allowed, best)

[<Fact>]
let ``searchmoves root result is not stored under the unrestricted TT key`` () =
    let tt = TranspositionTable(4)
    let legal = collectLegal (Position.OfFen StartPosFen)
    let allowed = legal.[legal.Length - 1]
    let cfg = { defaultConfig with UsePruning = false; UseRetro = false }
    let limits = { defaultLimits with SearchMoves = [| allowed |] }
    let struct (_, worker) = rootWorker cfg limits tt
    let rootKey = worker.Pos.Key

    negamax worker worker.Pos (-INF) INF 3 0 true false |> ignore

    let struct (hit, _, _, _, _, _, _) = tt.Probe rootKey
    Assert.False(hit, "a constrained root bound must not be reusable as an unrestricted root bound")

[<Fact>]
let ``searchmoves root ignores an unrestricted TT cutoff`` () =
    let tt = TranspositionTable(4)
    let legal = collectLegal (Position.OfFen StartPosFen)
    let allowed = legal.[legal.Length - 1]
    let disallowed = legal.[0]
    let cfg = { defaultConfig with UsePruning = false; UseRetro = false }
    let limits = { defaultLimits with SearchMoves = [| allowed |] }
    let struct (_, worker) = rootWorker cfg limits tt
    let impossibleScore = 12_345

    // Exercise the non-PV cutoff branch directly. The cached bound was computed over every legal
    // root move, so it cannot answer a search restricted to `allowed`.
    tt.Store worker.Pos.Key 32 BoundExact impossibleScore 0 disallowed false
    let score = negamax worker worker.Pos (-INF) INF 1 0 false false

    Assert.NotEqual(impossibleScore, score)
    Assert.True(worker.Nodes > 1L, "the restricted root returned directly from the unrestricted TT entry")

[<Fact>]
let ``MultiPV side line cannot replace the primary root TT move`` () =
    let tt = TranspositionTable(4)
    let cfg = { defaultConfig with UsePruning = false; UseRetro = false; MultiPv = 2 }
    let struct (_, worker) = rootWorker cfg defaultLimits tt
    let rootKey = worker.Pos.Key

    negamax worker worker.Pos (-INF) INF 3 0 true false |> ignore
    let primary = worker.Pv.[0]
    Assert.NotEqual(MoveNone, primary)

    // Mirror iterativeDeepening's second MultiPV pass: exclude line 1 at the root, then search the
    // same position again. This pass is useful for reporting but is not the position's exact value.
    worker.AddRootExclusion primary
    worker.RootPvIdx <- 1
    negamax worker worker.Pos (-INF) INF 3 0 true false |> ignore
    let sideLine = worker.Pv.[0]
    Assert.NotEqual(MoveNone, sideLine)
    Assert.NotEqual(primary, sideLine)

    let struct (hit, ttMove, _, _, _, _, _) = tt.Probe rootKey
    Assert.True(not hit || ttMove = primary,
                "a MultiPV side line replaced the unrestricted/primary root TT move")

[<Fact>]
let ``multi-thread node limit is an aggregate budget`` () =
    let threads = 4
    let budget = 32_768L
    let cfg =
        { defaultConfig with
            Threads = threads
            HashMb = 1
            UseTt = false
            UsePruning = false
            UseRetro = false
            UseDFPN = false }

    let limits = { defaultLimits with Nodes = budget }
    let control = SearchControl(cfg, limits, TranspositionTable(1), StartPosFen, [||])
    go control |> ignore

    let actual = control.NodeSum()
    let pollQuantum = 1L <<< Eonego.Tunables.TmPollShift
    let maxExpected = budget + int64 threads * pollQuantum
    Assert.InRange(actual, budget, maxExpected)

/// Time management: pins the computeTimes formulas (legacy + UseTmMtgHarden), the pure dynamic-TM
/// component curves (stability/trend/effort at default tunables), and the read-time soft-scale
/// mechanism on SearchControl — including the scale-set-during-ponder handoff that motivated
/// storing a SCALE instead of a scaled value.
module Eonego.Tests.TimeManTests

open Xunit
open Eonego.Bitboard
open Eonego.Position
open Eonego.Transposition
open Eonego.Search

// ---------------------------------------------------------------------------
// computeTimes — legacy formula pins (mtgHarden = false; defaults: Mtg=30, IncFrac=75, HardPct=40,
// HardMult=4). These guard every later TM stage against silent formula drift.
// ---------------------------------------------------------------------------
[<Fact>]
let ``movetime is a pure hard limit minus overhead`` () =
    Assert.Equal((0L, 990L), computeTimes 10 false { defaultLimits with MoveTime = 1000 } White)

[<Fact>]
let ``movetime never goes below one millisecond`` () =
    Assert.Equal((0L, 1L), computeTimes 500 false { defaultLimits with MoveTime = 100 } White)

[<Fact>]
let ``infinite depth nodes and mate searches have no time stop`` () =
    Assert.Equal((0L, 0L), computeTimes 10 false { defaultLimits with Infinite = true } White)
    Assert.Equal((0L, 0L), computeTimes 10 false { defaultLimits with Depth = 12 } White)
    Assert.Equal((0L, 0L), computeTimes 10 false { defaultLimits with Nodes = 1000L } White)
    Assert.Equal((0L, 0L), computeTimes 10 false { defaultLimits with Mate = 3 } White)

[<Fact>]
let ``no clock means no time stop`` () =
    Assert.Equal((0L, 0L), computeTimes 10 false defaultLimits White)

[<Fact>]
let ``sudden death splits the clock by the fallback movestogo`` () =
    // safety = 60000*2/100 = 1200; avail = 60000 - 10 - 1200 = 58790;
    // soft = 58790/30 + 1000*75/100 = 1959 + 750; hard = min(avail, 24000, 2709*4)
    let lim = { defaultLimits with WTime = 60000; WInc = 1000 }
    Assert.Equal((2709L, 10836L), computeTimes 10 false lim White)

[<Fact>]
let ``explicit movestogo divides the clock directly`` () =
    // avail = 60000 - 10 - 1200 safety = 58790; soft = 58790/20 = 2939; hard = min(avail, 24000, 2939*4)
    let lim = { defaultLimits with WTime = 60000; MovesToGo = 20 }
    Assert.Equal((2939L, 11756L), computeTimes 10 false lim White)

[<Fact>]
let ``black to move reads btime and binc`` () =
    // safety = 30000*2/100 = 600; avail = 30000 - 10 - 600 = 29390;
    // soft = 29390/30 + 500*75/100 = 979 + 375; hard = min(avail, 12000, 1354*4)
    let lim = { defaultLimits with BTime = 30000; BInc = 500; WTime = 1 }
    Assert.Equal((1354L, 5416L), computeTimes 10 false lim Black)

[<Fact>]
let ``legacy movestogo one leaves soft above hard`` () =
    // The known wart the harden flag fixes: soft = avail = 9790 (10000 - 10 - 200 safety) but hard
    // caps at 40% = 4000.
    let lim = { defaultLimits with WTime = 10000; MovesToGo = 1 }
    Assert.Equal((9790L, 4000L), computeTimes 10 false lim White)

// ---------------------------------------------------------------------------
// computeTimes — UseTmMtgHarden (defaults: MtgClamp=50, MtgLowStep=10)
// ---------------------------------------------------------------------------
[<Fact>]
let ``harden lets movestogo one spend ninety percent of the clock`` () =
    // avail = 10000 - 10 - 200 safety = 9790; hardPct = min 90 (40 + 5*10) = 90;
    // hard = 9790*90/100 = 8811; soft clamped to hard.
    let lim = { defaultLimits with WTime = 10000; MovesToGo = 1 }
    Assert.Equal((8811L, 8811L), computeTimes 10 true lim White)

[<Fact>]
let ``harden clamps an absurd movestogo`` () =
    // mtg 200 -> 50: avail = 58790; soft = 58790/50 = 1175; hard = min(avail, 24000, 4700).
    let lim = { defaultLimits with WTime = 60000; MovesToGo = 200 }
    Assert.Equal((1175L, 4700L), computeTimes 10 true lim White)

[<Fact>]
let ``harden leaves a normal movestogo unchanged`` () =
    let lim = { defaultLimits with WTime = 60000; MovesToGo = 20 }
    Assert.Equal(computeTimes 10 false lim White, computeTimes 10 true lim White)

[<Fact>]
let ``harden keeps soft at or below hard across the input grid`` () =
    for time in [ 50; 500; 5000; 60000; 600000 ] do
        for inc in [ 0; 100; 5000 ] do
            for mtg in [ 0; 1; 2; 5; 6; 30; 200 ] do
                let lim = { defaultLimits with WTime = time; WInc = inc; MovesToGo = mtg }
                let (soft, hard) = computeTimes 10 true lim White
                Assert.True(soft >= 1L && hard >= 1L)
                Assert.True(soft <= hard, "soft > hard for time=" + string time + " inc=" + string inc + " mtg=" + string mtg)

// ---------------------------------------------------------------------------
// Dynamic-TM component curves at default tunables
// ---------------------------------------------------------------------------
[<Fact>]
let ``stability shrinks the budget linearly to the floor`` () =
    // Base=110, Slope=5, Min=80: 110 105 100 95 90 85 80, then flat.
    Assert.Equal(110, tmStabilityPct 0)
    Assert.Equal(105, tmStabilityPct 1)
    Assert.Equal(100, tmStabilityPct 2)
    Assert.Equal(80, tmStabilityPct 6)
    Assert.Equal(80, tmStabilityPct 10)

[<Fact>]
let ``trend grows on a falling score and barely saves on a rising one`` () =
    // Slope=100, Min=95, Max=140.
    Assert.Equal(100, tmTrendPct 0)
    Assert.Equal(120, tmTrendPct 20)
    Assert.Equal(140, tmTrendPct 50) // clamped at Max
    Assert.Equal(95, tmTrendPct (-100)) // asymmetric floor
    Assert.Equal(95, tmTrendPct (-5))

[<Fact>]
let ``effort is neutral at even concentration and clamps at the extremes`` () =
    // Off=70, Slope=60, Min=75, Max=125.
    Assert.Equal(100, tmEffortPct 50)
    Assert.Equal(76, tmEffortPct 90)
    Assert.Equal(118, tmEffortPct 20)
    Assert.Equal(75, tmEffortPct 100) // 70 clamped up to Min
    Assert.Equal(125, tmEffortPct 0) // 130 clamped down to Max

// ---------------------------------------------------------------------------
// SearchControl read-time scale mechanism
// ---------------------------------------------------------------------------
let private makeControl (limits: SearchLimits) =
    SearchControl(defaultConfig, limits, TranspositionTable(1), StartPosFen, [||])

[<Fact>]
let ``soft budget is base times scale over one hundred`` () =
    let c = makeControl defaultLimits
    c.StartClock 1000L 4000L
    Assert.Equal(1000L, c.SoftBudgetMs)
    c.SetSoftScale 150L
    Assert.Equal(1500L, c.SoftBudgetMs)

[<Fact>]
let ``scale is clamped to the tunable bounds`` () =
    // Defaults: ScaleMin=60, ScaleMax=180.
    let c = makeControl defaultLimits
    c.StartClock 1000L 4000L
    c.SetSoftScale 10L
    Assert.Equal(600L, c.SoftBudgetMs)
    c.SetSoftScale 500L
    Assert.Equal(1800L, c.SoftBudgetMs)

[<Fact>]
let ``neutral scale is integer-identical to the unscaled budget`` () =
    let c = makeControl defaultLimits
    for soft in [ 1L; 3L; 7L; 999L; 2749L ] do
        c.StartClock soft 4000L
        Assert.Equal(soft, c.SoftBudgetMs)

[<Fact>]
let ``scale accumulated during ponder applies at ponderhit`` () =
    // The race the read-time design kills: the scale is written while the budget is still 0
    // (pondering); arming the real budget at ponderhit must yield base*scale with no reset.
    let c = makeControl { defaultLimits with Ponder = true }
    c.StartClockPonder 1000L 4000L true
    c.SetSoftScale 150L
    Assert.Equal(0L, c.SoftBudgetMs) // still pondering: unbounded regardless of scale
    c.PonderHit()
    Assert.Equal(1000L, c.BaseSoftMs)
    Assert.Equal(1500L, c.SoftBudgetMs)

// ---------------------------------------------------------------------------
// SearchControl read-time HARD scale (UseTmFailLow rider: TmFailLowHard extends the hard cap
// during a fail-low recovery, same race-free store-the-scale design as the soft mechanism).
// ---------------------------------------------------------------------------
[<Fact>]
let ``hard budget is base times hard scale over one hundred`` () =
    let c = makeControl defaultLimits
    c.StartClock 1000L 4000L
    Assert.Equal(4000L, c.HardBudgetMs)
    c.SetHardScale 150L
    Assert.Equal(6000L, c.HardBudgetMs)

[<Fact>]
let ``hard scale is clamped to the TmFailLowHard range`` () =
    // TmFailLowHard range: 100..200 — the hard cap may only grow, and at most double.
    let c = makeControl defaultLimits
    c.StartClock 1000L 4000L
    c.SetHardScale 50L
    Assert.Equal(4000L, c.HardBudgetMs) // clamped up to neutral: never shrink the safety net
    c.SetHardScale 500L
    Assert.Equal(8000L, c.HardBudgetMs) // clamped to 200

[<Fact>]
let ``neutral hard scale is integer-identical to the unscaled budget`` () =
    let c = makeControl defaultLimits
    for hard in [ 1L; 3L; 7L; 999L; 10996L ] do
        c.StartClock 0L hard
        Assert.Equal(hard, c.HardBudgetMs)

/// Campaign tunables — every search margin the SPSA/SPRT tuning campaign varies, as module-level
/// statics initialized ONCE from `EONEGO_T_*` env vars (unset/unparseable -> the default, which is
/// numerically identical to the previously hardcoded literal). Clamps keep every value inside its
/// safety contract (e.g. gravity divisors < int16 max so the history stores can never overflow).
///
/// Compiled FIRST (before Bitboard.fs): depends only on System.Environment, and F# per-file init
/// ordering then guarantees these bindings exist before any consumer module's own init (notably the
/// LMR Reductions table built at Search.fs module init). Reads on the hot path are static readonly
/// loads — the same class of access as the proven Position.UseFinny / Accumulator.UseAvx2 env statics.
///
/// The byte-identity contract: with NO env vars set, node counts must equal the pre-Tunables binary
/// exactly (verified by scripts/nodesweep.ps1). match.py's --a/--b env channels flow these per player,
/// so tuning matches never need a rebuild.
module Eonego.Tunables

let private envInt (name: string) (deflt: int) (lo: int) (hi: int) : int =
    match System.Environment.GetEnvironmentVariable name with
    | null
    | "" -> deflt
    | s ->
        match System.Int32.TryParse s with
        | true, v -> max lo (min hi v)
        | _ -> deflt

// SPSA wave-1 (2026-07-03, 651 iters @ mt60, ~10.4k games): tuned defaults inlined after the
// acceptance SPRT hit H1 at +107.7±28.2 (mt100, 376 games) and the LTC sanity ran +70.4±44.2
// (mt1000, 120 games). Old values in trailing comments; env overrides still take precedence.

// --- Reverse futility (Search.fs step 4) ---
let RfpMargin = envInt "EONEGO_T_RFP_MARGIN" 105 1 1000 // was 120
let RfpTtPvBonus = envInt "EONEGO_T_RFP_TTPV" 21 0 500 // was 20

// --- Razoring ---
let RazorBase = envInt "EONEGO_T_RAZOR_BASE" 224 1 2000 // was 240
let RazorSlope = envInt "EONEGO_T_RAZOR_SLOPE" 202 1 1000 // was 200

// --- Null-move pruning ---
let NmpBase = envInt "EONEGO_T_NMP_BASE" 3 1 8
let NmpDepthDiv = envInt "EONEGO_T_NMP_DDIV" 4 1 16
let NmpEvalMargin = envInt "EONEGO_T_NMP_EVALMARGIN" 212 1 2000 // was 200
// Continuous eval-excess reduction (SF-style): when > 0, the binary "eval-over-beta => +1 ply" term
// becomes min((workingEval - beta) / NmpEvalDiv, NmpEvalMax). DEFAULT 0 = the legacy binary term, so
// the tree is byte-identical until this is set — SPRT the enabled value (roadmap B1). ~200 is the SF
// scale, but Eonego's NNUE eval scale differs, so SPSA NmpEvalDiv (and co-tune NmpBase/NmpDepthDiv).
let NmpEvalDiv = envInt "EONEGO_T_NMP_EVALDIV" 0 0 2000
let NmpEvalMax = envInt "EONEGO_T_NMP_EVALMAX" 6 1 10

// --- ProbCut ---
let ProbCutMargin = envInt "EONEGO_T_PROBCUT_MARGIN" 200 1 1000
let ProbCutImproving = envInt "EONEGO_T_PROBCUT_IMPR" 50 0 500

// --- LMR base table (log(d)*log(m)/div + off; scaled x100 for integer env vars) ---
let LmrDiv100 = envInt "EONEGO_T_LMR_DIV100" 223 100 600 // was 220
let LmrOff100 = envInt "EONEGO_T_LMR_OFF100" 47 0 100 // was 50
// LMR tweak: extra reduction when combined history is below this threshold.
let LmrHistThresh = envInt "EONEGO_T_LMR_HIST" -11499 -30000 0 // was -12000

// --- Late-move (move-count) pruning: moveCount >= (LmpBase + d*d) / (improving ? 1 : 2) ---
let LmpBase = envInt "EONEGO_T_LMP_BASE" 9 1 20 // was 3 — the wave's biggest single move

// --- History pruning: MainHistory < -(Base + Slope*lmrDepth). Keep Base + Slope*6 within the
//     ±MainHistD saturation band or the gate goes dead (soft constraint; SPSA just wastes the region).
let HistPruneBase = envInt "EONEGO_T_HISTPRUNE_BASE" 521 0 3000 // was 500
let HistPruneSlope = envInt "EONEGO_T_HISTPRUNE_SLOPE" 747 0 1200 // was 800
let HistPruneCombinedBase = envInt "EONEGO_T_HISTPRUNE_CBASE" 4000 0 20000
let HistPruneCombinedSlope = envInt "EONEGO_T_HISTPRUNE_CSLOPE" 4000 0 20000

// --- Move-loop futility: staticEval + Base + Slope*lmrDepth <= alpha ---
let FutBase = envInt "EONEGO_T_FUT_BASE" 124 1 1000 // was 120
let FutSlope = envInt "EONEGO_T_FUT_SLOPE" 109 1 1000 // was 110

// --- Capture futility (EONEGO_CAPFUT=1): staticEval + Base + Slope*lmrDepth + capturedValue <= alpha ---
let CaptFutBase = envInt "EONEGO_T_CAPTFUT_BASE" 300 1 1000
let CaptFutSlope = envInt "EONEGO_T_CAPTFUT_SLOPE" 250 1 1000

// --- SEE pruning: quiets -Mult*lmrDepth^2, captures -Mult*depth ---
let SeeQuietMult = envInt "EONEGO_T_SEE_QUIET" 26 1 200 // was 25
let SeeCaptMult = envInt "EONEGO_T_SEE_CAPT" 99 1 500 // was 90

// --- Qsearch delta pruning: rawEval + Base + capturedValue <= alpha ---
let QsDeltaBase = envInt "EONEGO_T_QS_DELTA" 196 1 1000 // was 200

// --- Singular extensions: margin = ttScore - Mul16*depth/16 (x16 scaling gives SPSA sub-integer
//     resolution around the default 2*depth = 32/16) ---
let SingularMul16 = envInt "EONEGO_T_SING_MUL16" 35 4 128 // was 32
let DoubleExtMargin = envInt "EONEGO_T_SING_DBL" 18 1 200 // was 16

// --- Aspiration fail-high re-search depth reduction. 0 = never reduce (measured −24.4 ± 26.2 vs
//     legacy — full-depth re-searches on every noisy fail-high are too expensive); 1 = legacy,
//     always reduce up to depth−3 (asymmetric with fail-low, which re-runs at FULL depth — a slow
//     win arriving as a root fail-high can evaporate at the reduced depth every iteration and stay
//     suppressed forever: b3-b4 fixture pinned at +18 while the subtree is provably +380);
//     2 = ARBITRATED: reduce as legacy, but when the fail-high then EVAPORATES (the re-search comes
//     back below the pre-widen beta — the erasure signature), run ONE full-depth arbitration
//     re-search per iteration. Keeps the efficiency where it pays, removes the suppression. ---
let AspFailHighRed = envInt "EONEGO_T_ASP_FHRED" 1 0 2

// --- Aspiration: initial delta = Init + prev^2/SqDiv ---
let AspInitDelta = envInt "EONEGO_T_ASP_INIT" 8 1 100 // was 10
let AspSqDiv = envInt "EONEGO_T_ASP_SQDIV" 16053 1000 100000 // was 15000
// A3 diversity rider: odd-id helper workers ADD this to their initial aspiration delta (0 = off,
// byte-identical legacy behaviour). Widens helper windows so LazySMP threads diverge by design
// instead of only by TT-arrival races.
let HelperAspOffset = envInt "EONEGO_T_HELPER_ASP" 0 0 100

// --- Correction history: applied = entry/ApplyDiv; update bonus = clamp(err*depth/DepthDiv, ±Clamp) ---
let CorrApplyDiv = envInt "EONEGO_T_CORR_DIV" 16 1 256
let CorrClamp = envInt "EONEGO_T_CORR_CLAMP" 256 16 2047
let CorrDepthDiv = envInt "EONEGO_T_CORR_DDIV" 8 1 64

// --- cont4 (EONEGO_CONT4=1): the ss-4 continuation term is read at 1/Div weight in the LMR
//     history threshold (taught full bonus). Div=2 = the "weighted ss-4" shape. ---
let Cont4Div = envInt "EONEGO_T_CONT4_DIV" 2 1 8

// --- Rule-50 shuffle damping (evaluate() wrapper term, missing from the original port):
//     eval -= eval * rule50 / Div. Pulls stuck positions (fortresses, shuffling) toward the draw
//     score gracefully instead of holding full value until the search proves the rule-50 draw. ---
let Rule50DampDiv = envInt "EONEGO_T_R50_DAMP" 212 50 2048

// --- Time management (game clocks only — fixed-depth/movetime paths never read these).
//     soft = (clock-overhead)/Mtg + inc*IncFrac100/100; hard = min(clock*HardClockPct/100, soft*HardSoftMult).
//     Defaults reproduce the v1 formula exactly (3/4 == 75/100, 0.4 == 40/100 at int precision). ---
let TmMtg = envInt "EONEGO_T_TM_MTG" 30 8 80
let TmIncFrac100 = envInt "EONEGO_T_TM_INCFRAC" 75 0 150
let TmHardClockPct = envInt "EONEGO_T_TM_HARDPCT" 40 10 90
let TmHardSoftMult = envInt "EONEGO_T_TM_HARDMULT" 4 2 12

// --- Dynamic TM scale (the TM campaign): softScalePct rescales the soft budget at READ time
//     (SearchControl.SoftBudgetMs); 100 = neutral = integer-identical to the unscaled budget.
//     Composition: stab · trend/100 · effort/100 · failLow/100, clamped [ScaleMin, ScaleMax].
//     Each component gates on its own EONEGO_TM* config flag (UCI.fs) — off contributes exactly 100. ---
let TmScaleMin = envInt "EONEGO_T_TM_SCALE_MIN" 60 30 100
let TmScaleMax = envInt "EONEGO_T_TM_SCALE_MAX" 180 100 300

// movestogo hardening (EONEGO_TMMTG=1): clamp an absurd user mtg; widen the per-move hard cap when
// few moves remain to the time control (mtg<=5, based on the overhead-adjusted clock); soft <= hard.
let TmMtgClamp = envInt "EONEGO_T_TM_MTGCLAMP" 50 10 80
let TmMtgLowStep = envInt "EONEGO_T_TM_MTGLOWSTEP" 10 0 20

// Best-move stability (EONEGO_TMSTAB=1): stabPct = max Min (Base - Slope*streak), streak capped at
// Cap. Defaults trace 110->105->100->95->90->85->80 — near-neutral by design (the two earlier TM
// scaling attempts over-conserved and regressed; an average move must spend ~its baseline budget).
let TmStabBase = envInt "EONEGO_T_TM_STAB_BASE" 110 100 160
let TmStabSlope = envInt "EONEGO_T_TM_STAB_SLOPE" 5 0 25
let TmStabCap = envInt "EONEGO_T_TM_STAB_CAP" 6 1 10
let TmStabMin = envInt "EONEGO_T_TM_STAB_MIN" 80 50 100

// Score trend (EONEGO_TMTREND=1): trendPct = clamp(100 + fallCp*Slope/100, Min, Max) where fallCp =
// score two completed iterations ago minus now. Asymmetric: a falling score may grow the budget to
// Max, a rising one saves at most (100-Min)% — over-conservation is where the old attempts died.
let TmTrendSlope100 = envInt "EONEGO_T_TM_TREND_SLOPE" 100 0 400
let TmTrendMin = envInt "EONEGO_T_TM_TREND_MIN" 95 50 100
let TmTrendMax = envInt "EONEGO_T_TM_TREND_MAX" 140 100 250

// Root fail-low extension (EONEGO_TMFAILLOW=1): one-shot component armed by a root fail-low at
// depth >= FailLowDepth (earlier fail-lows are aspiration noise). FailLowHard is RESERVED: 100 =
// inert; hard-limit scaling is deliberately unimplemented until SPSA shows the soft bump wants it.
let TmFailLowPct = envInt "EONEGO_T_TM_FAILLOW" 125 100 200
let TmFailLowDepth = envInt "EONEGO_T_TM_FAILLOW_DEPTH" 6 1 20
let TmFailLowHard = envInt "EONEGO_T_TM_FAILLOW_HARD" 100 100 200

// Node effort (EONEGO_TMEFFORT=1): effortPct = clamp(Off + (100-bestFrac)*Slope/100, Min, Max) —
// bestFrac = the best root move's share (%) of this iteration's root nodes. Neutral at 50%
// concentration; a dominating best move shrinks the budget, a contested root grows it.
let TmEffortOff = envInt "EONEGO_T_TM_EFFORT_OFF" 70 40 100
let TmEffortSlope = envInt "EONEGO_T_TM_EFFORT_SLOPE" 60 0 150
let TmEffortMin = envInt "EONEGO_T_TM_EFFORT_MIN" 75 50 100
let TmEffortMax = envInt "EONEGO_T_TM_EFFORT_MAX" 125 100 200

// pollStop clock/node-limit check cadence: every 2^PollShift nodes. 13 (mask 8191) is the legacy
// value and also fixes the STOP GRANULARITY of `go nodes` — the default may only change behind a
// bullet-TC SPRT (candidate: 11 => ~2ms granularity at 1.2Mnps vs ~7ms today).
let TmPollShift = envInt "EONEGO_T_TM_POLLSHIFT" 13 9 14

// --- Root re-verification on stagnation (EONEGO_ROOTVERIFY=1): once the root score has moved less
//     than VerifyBand centipawns over the last 6 completed iterations (>= VerifyDepth), each further
//     iteration designates ONE rotating non-best root move for a full-window, unreduced, PV-quality
//     search. PV nodes skip TT cutoffs, so this is the only way past stale "<= alpha" bounds that
//     null-window scouts re-graft forever (the b3-b4 fixture deadlock). ---
let RootVerifyDepth = envInt "EONEGO_T_ROOTVERIFY_DEPTH" 14 6 64
let RootVerifyBand = envInt "EONEGO_T_ROOTVERIFY_BAND" 12 1 200

// --- Root LMR cap (EONEGO_T_ROOT_LMR_CAP): clamp the LMR reduction of root moves to this many
//     plies. 99 = no cap (byte-identical legacy). The reference engine reduces root moves less than
//     interior moves; a buried root move (negative history + late order) otherwise never gets a deep
//     enough scout to surface a slow win (the b3-b4 fixture pathology). ---
let RootLmrCap = envInt "EONEGO_T_ROOT_LMR_CAP" 99 0 99

// --- History stat bonus: min(Mul*depth - 100, Cap) ---
let StatBonusMul = envInt "EONEGO_T_STATB_MUL" 167 16 1000 // was 160
let StatBonusCap = envInt "EONEGO_T_STATB_CAP" 1735 100 7000 // was 1700
// Stat MALUS (penalty) shape — DEFAULTS IDENTICAL to statBonus (167/1735) so `-statMalus` equals the
// legacy `-statBonus` and the tree stays byte-identical until tuned. SF's malus grows several× faster
// with depth than its bonus; SPRT a steeper StatMalusMul (~350-700) so the butterfly/continuation
// tables stay discriminative (roadmap B2). Own EONEGO_T_STATM_* channels for independent SPSA.
let StatMalusMul = envInt "EONEGO_T_STATM_MUL" 167 16 1500
let StatMalusCap = envInt "EONEGO_T_STATM_CAP" 1735 100 7000

// --- Capture ordering: MVV multiplier in MovePick capture scoring (score = Mul*pieceValue + captHist) ---
let CaptScoreMul = envInt "EONEGO_T_CAPSCORE_MUL" 7 1 20

// --- Quiet partial-sort threshold: moves scoring >= Limit*depth are insertion-sorted to the front ---
let QuietSortLimit = envInt "EONEGO_T_QSORT_LIM" -3000 -10000 0

// --- df-pn mate oracle (EONEGO_DFPN=1 gate lives in UCI.fs): proof-number table size and node
//     budgets. VNodes bounds the verification replay (fail-closed: an over-budget verify just
//     declines to publish). Nodes = 0 means uncapped (the stop flag still bounds the solve). ---
let DFPNMb = envInt "EONEGO_T_DFPN_MB" 32 1 1024
let DFPNNodes = envInt "EONEGO_T_DFPN_NODES" 20_000_000 0 2_000_000_000
let DFPNVNodes = envInt "EONEGO_T_DFPN_VNODES" 5_000_000 1 2_000_000_000

// --- Policy sidecar head (EONEGO_POLICY gate lives in UCI.fs; Phase-0 re-scope 2026-07-06:
//     the LMR term is the v1 consumption, the ordering blend ships INERT at Mul=0).
//     PolMinDepth: no policy inference below this picker depth (qsearch never pays).
//     PolLmrThresh: quiets with from+to logit below this get one extra LMR reduction step.
//     PolOrdMul/Shift/Clamp: scoreQuiets blend term = clamp(Mul*logit >>> Shift, +/-Clamp). ---
let PolMinDepth = envInt "EONEGO_T_POL_MINDEPTH" 3 1 32
let PolLmrThresh = envInt "EONEGO_T_POL_LMR" 0 -1000000 1000000
let PolOrdMul = envInt "EONEGO_T_POL_ORDMUL" 0 0 1024
let PolOrdShift = envInt "EONEGO_T_POL_ORDSHIFT" 8 0 16
let PolClamp = envInt "EONEGO_T_POL_CLAMP" 4000 0 16384

// --- Gravity divisors (also the bonus clamp bounds). MUST stay < 32700: the gravity fixpoint keeps
//     |entry| <= D, so the int16 stores in History.Tables can never overflow. ---
let MainHistD = envInt "EONEGO_T_MAINHIST_D" 7183 1024 32000
let CaptureHistD = envInt "EONEGO_T_CAPTHIST_D" 10692 1024 32000
let ContHistD = envInt "EONEGO_T_CONTHIST_D" 29952 1024 32000
let CorrHistD = envInt "EONEGO_T_CORRHIST_D" 2048 256 32000

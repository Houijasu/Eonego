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

// --- History stat bonus: min(Mul*depth - 100, Cap) ---
let StatBonusMul = envInt "EONEGO_T_STATB_MUL" 167 16 1000 // was 160
let StatBonusCap = envInt "EONEGO_T_STATB_CAP" 1735 100 7000 // was 1700

// --- ABDADA (EONEGO_ABDADA=1): claim nodes at >= ClaimMinDepth; defer a move when a sibling thread
//     owns the child at >= DeferMinDepth (child depth). Shallow nodes churn the table for nothing. ---
let AbdadaClaimMinDepth = envInt "EONEGO_T_ABDADA_CLAIM" 4 1 32
let AbdadaDeferMinDepth = envInt "EONEGO_T_ABDADA_DEFER" 4 1 32

// --- Gravity divisors (also the bonus clamp bounds). MUST stay < 32700: the gravity fixpoint keeps
//     |entry| <= D, so the int16 stores in History.Tables can never overflow. ---
let MainHistD = envInt "EONEGO_T_MAINHIST_D" 7183 1024 32000
let CaptureHistD = envInt "EONEGO_T_CAPTHIST_D" 10692 1024 32000
let ContHistD = envInt "EONEGO_T_CONTHIST_D" 29952 1024 32000
let CorrHistD = envInt "EONEGO_T_CORRHIST_D" 2048 256 32000

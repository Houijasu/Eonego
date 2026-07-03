"""SPSA tuner for Eonego's EONEGO_T_* search margins (campaign step B2).

Each iteration perturbs ALL params simultaneously (Rademacher direction), plays theta+ vs theta-
directly via match.run_match (imported, concurrency-pooled), and steps theta along the direction
scaled by the score. Fishtest conventions: c_k = c / k^0.101, a_k = a / (A + k)^0.602 with
end-targeting c = c_end * N^0.101, a = r_end * c_end^2 * (A + N)^0.602, A = 0.1 * N.

    python spsa.py --iters 2000 --movetime 75 --games-per-iter 8 --concurrency 8
    python spsa.py --resume                     # continue from spsa_state.json
    python spsa.py --fake-objective --iters 800 # synthetic convergence self-test (no engines)

State is checkpointed to spsa_state.json every iteration; Ctrl-C is safe (prints the current best
env string and exits after saving). Final output: paste-ready --a env string + Tunables.fs diffs.
"""

import argparse
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import match  # noqa: E402

def state_file_for(wave):
    """Per-wave checkpoint so a new wave never clobbers an older wave's resumable state."""
    name = "spsa_state.json" if wave == 1 else f"spsa_state_w{wave}.json"
    return os.path.join(HERE, name)


STATE_FILE = state_file_for(1)  # rebound in run() from --wave

# Wave-1 parameters: env name -> (default, min, max, c_end).
# c_end = the +/- perturbation magnitude at the END of the run (~5-10% of the default).
# NOTE: wave-1 winners were inlined as the Tunables defaults (dc3b1f7); this table keeps the OLD
# pre-tune values and exists for reproducibility of the wave-1 run only. New runs: --wave 2.
PARAMS_WAVE1 = {
    "EONEGO_T_RFP_MARGIN":      (120, 40, 400, 10),
    "EONEGO_T_RFP_TTPV":        (20, 0, 100, 4),
    "EONEGO_T_RAZOR_BASE":      (240, 60, 800, 20),
    "EONEGO_T_RAZOR_SLOPE":     (200, 50, 600, 16),
    "EONEGO_T_NMP_EVALMARGIN":  (200, 50, 800, 20),
    "EONEGO_T_LMR_DIV100":      (220, 140, 400, 12),
    "EONEGO_T_LMR_OFF100":      (50, 0, 100, 8),
    "EONEGO_T_LMR_HIST":        (-12000, -25000, -2000, 1000),
    "EONEGO_T_LMP_BASE":        (3, 1, 12, 1),
    "EONEGO_T_HISTPRUNE_BASE":  (500, 0, 2500, 60),
    "EONEGO_T_HISTPRUNE_SLOPE": (800, 100, 1200, 80),
    "EONEGO_T_FUT_BASE":        (120, 30, 500, 12),
    "EONEGO_T_FUT_SLOPE":       (110, 30, 400, 10),
    "EONEGO_T_SEE_QUIET":       (25, 8, 100, 3),
    "EONEGO_T_SEE_CAPT":        (90, 30, 300, 8),
    "EONEGO_T_QS_DELTA":        (200, 50, 600, 20),
    "EONEGO_T_SING_MUL16":      (32, 12, 96, 4),
    "EONEGO_T_SING_DBL":        (16, 4, 80, 3),
    "EONEGO_T_ASP_INIT":        (10, 3, 50, 2),
    "EONEGO_T_ASP_SQDIV":       (15000, 4000, 60000, 1500),
    "EONEGO_T_STATB_MUL":       (160, 60, 500, 16),
    "EONEGO_T_STATB_CAP":       (1700, 500, 5000, 150),
}

# Wave-2 (2026-07-03): the cont4/capture-futility joint retune. Both arms run with the riders ON
# (--shared "EONEGO_CONT4=1,EONEGO_CAPFUT=1"); defaults here = the post-wave-1 inlined values.
# Tight 13-param set (vs 22) concentrates the per-iteration signal on the coupled history bands.
PARAMS_WAVE2 = {
    "EONEGO_T_CONT4_DIV":       (2, 1, 8, 1),
    "EONEGO_T_LMR_HIST":        (-11499, -25000, -2000, 1000),
    "EONEGO_T_LMP_BASE":        (9, 1, 20, 1),
    "EONEGO_T_HISTPRUNE_BASE":  (521, 0, 2500, 60),
    "EONEGO_T_HISTPRUNE_SLOPE": (747, 100, 1200, 80),
    "EONEGO_T_FUT_BASE":        (124, 30, 500, 12),
    "EONEGO_T_FUT_SLOPE":       (109, 30, 400, 10),
    "EONEGO_T_CAPTFUT_BASE":    (300, 50, 800, 25),
    "EONEGO_T_CAPTFUT_SLOPE":   (250, 50, 800, 20),
    "EONEGO_T_SEE_QUIET":       (26, 8, 100, 3),
    "EONEGO_T_SEE_CAPT":        (99, 30, 300, 8),
    "EONEGO_T_STATB_MUL":       (167, 60, 500, 16),
    "EONEGO_T_STATB_CAP":       (1735, 500, 5000, 150),
}

WAVES = {1: PARAMS_WAVE1, 2: PARAMS_WAVE2}
PARAMS = PARAMS_WAVE1  # rebound in run() from --wave; module-level default keeps wave-1 semantics


def env_string(theta):
    return ",".join(f"{k}={int(round(v))}" for k, v in theta.items())


def clamp_theta(theta):
    for k, v in theta.items():
        lo, hi = PARAMS[k][1], PARAMS[k][2]
        theta[k] = min(max(v, lo), hi)
    return theta


def match_args(env_a, env_b, ns):
    """Build a match.run_match argument namespace for one theta+/theta- pairing."""
    argv = [
        "--a", env_a, "--b", env_b,
        "--exe", ns.exe,
        "--movetime", str(ns.movetime),
        "--openings", str(ns.games_per_iter),
        "--opening-plies", "6",
        "--seed", str(ns.match_seed),
        "--concurrency", str(ns.concurrency),
    ]
    if getattr(ns, "shared", ""):
        argv += ["--shared", ns.shared]
    return match.build_parser().parse_args(argv)


def fake_score(theta):
    """Synthetic objective: logistic of distance from a planted optimum + game-like noise.
    Positive score means theta_plus is better (closer to the optimum)."""
    opt = {k: (PARAMS[k][1] + PARAMS[k][2]) / 2.0 * 1.1 for k in PARAMS}
    def loss(t):
        return sum(((t[k] - opt[k]) / max(1.0, PARAMS[k][3] * 10.0)) ** 2 for k in PARAMS)
    return loss  # caller evaluates on theta+ / theta-


def run(ns):
    global PARAMS, STATE_FILE
    PARAMS = WAVES[ns.wave]
    STATE_FILE = state_file_for(ns.wave)

    rng = random.Random(ns.seed)
    start_iter = 1
    theta = {k: float(PARAMS[k][0]) for k in PARAMS}

    if ns.resume and os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            st = json.load(f)
        start_iter = st["iter"] + 1
        theta = {k: float(st["theta"].get(k, PARAMS[k][0])) for k in PARAMS}
        rng.setstate(tuple(st["rng"][0:1] + [tuple(st["rng"][1])] + st["rng"][2:]))
        print(f"resumed at iter {start_iter}", flush=True)

    N = ns.iters
    A = 0.1 * N
    gamma, alpha_exp = 0.101, 0.602
    # Per-param gain schedules from end-targets.
    c0 = {k: PARAMS[k][3] * (N ** gamma) for k in PARAMS}
    a0 = {k: ns.r_end * (PARAMS[k][3] ** 2) * ((A + N) ** alpha_exp) for k in PARAMS}

    loss = fake_score(theta) if ns.fake_objective else None

    try:
        for k_iter in range(start_iter, N + 1):
            ck = {p: c0[p] / (k_iter ** gamma) for p in PARAMS}
            ak = {p: a0[p] / ((A + k_iter) ** alpha_exp) for p in PARAMS}
            delta = {p: rng.choice((-1.0, 1.0)) for p in PARAMS}

            tp = clamp_theta({p: theta[p] + ck[p] * delta[p] for p in PARAMS})
            tm = clamp_theta({p: theta[p] - ck[p] * delta[p] for p in PARAMS})

            if ns.fake_objective:
                # Deterministic-ish signal: which side is closer to the optimum, plus noise.
                diff = loss(tm) - loss(tp)
                score = max(-1.0, min(1.0, 0.02 * diff)) + rng.gauss(0, 0.25)
                score = max(-1.0, min(1.0, score))
                games = 16
            else:
                ns.match_seed = ns.seed + k_iter  # fresh openings every iteration
                r = match.run_match(match_args(env_string(tp), env_string(tm), ns), quiet=True)
                if r["errors"]:
                    raise RuntimeError("; ".join(r["errors"]))
                games = max(1, r["games"])
                score = (r["W"] - r["L"]) / games  # in [-1, 1], + means theta_plus better

            for p in PARAMS:
                theta[p] += (ak[p] / (ck[p] ** 2)) * ck[p] * delta[p] * score
            clamp_theta(theta)

            rs = rng.getstate()
            with open(STATE_FILE, "w") as f:
                json.dump({"iter": k_iter, "theta": theta,
                           "rng": [rs[0], list(rs[1]), rs[2]],
                           "last_score": score, "games": games}, f)

            if k_iter % ns.log_every == 0 or k_iter == N:
                sample = {p.replace("EONEGO_T_", ""): int(round(theta[p]))
                          for p in list(PARAMS)[:6]}
                print(f"it {k_iter:5d}  score {score:+.3f}  {sample}", flush=True)
    except KeyboardInterrupt:
        print("\ninterrupted — state saved", flush=True)

    print("\n=== SPSA result ===")
    print("env string (player A):")
    print("  " + env_string(theta))
    print("Tunables.fs default changes (param: old -> new):")
    for p in PARAMS:
        new = int(round(theta[p]))
        if new != PARAMS[p][0]:
            print(f"  {p}: {PARAMS[p][0]} -> {new}")
    return theta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--movetime", type=int, default=75)
    ap.add_argument("--games-per-iter", type=int, default=8, help="opening PAIRS per iteration (x2 games)")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--exe", default=match.DEFAULT_EXE)
    ap.add_argument("--seed", type=int, default=20260702)
    ap.add_argument("--r-end", type=float, default=0.05, help="learning rate at the final iteration")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--fake-objective", action="store_true", help="synthetic convergence self-test (no engines)")
    ap.add_argument("--wave", type=int, default=1, choices=sorted(WAVES),
                    help="parameter table: 1 = original margins, 2 = cont4/captfut joint retune")
    ap.add_argument("--shared", default="",
                    help="env overrides for BOTH arms, e.g. 'EONEGO_CONT4=1,EONEGO_CAPFUT=1' (wave 2)")
    ns = ap.parse_args()
    # Wave 2 tunes rider params (CONT4_DIV, CAPTFUT_*) that are inert unless the rider flags are on —
    # a forgotten --shared would silently burn the whole run on dead parameters. Default it in.
    if ns.wave == 2 and not ns.shared:
        ns.shared = "EONEGO_CONT4=1,EONEGO_CAPFUT=1"
        print(f"wave 2: defaulting --shared '{ns.shared}'", flush=True)
    ns.match_seed = ns.seed
    run(ns)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

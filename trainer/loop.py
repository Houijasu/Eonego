"""Self-play reinforcement loop orchestrator for the Eonego KGA-specialist NNUE.

Per generation k>=1:
  1. gen   : engine self-play from KGA with net_best (parallel workers) -> data_k
  2. window: concat the last W generations' data
  3. train : QAT on the window, warm-started from net_best -> net_k (+ .pt)
  4. parity: intref(net_k) == engine nnforward  (assert; skip on failure)
  5. match : net_k vs net_best from KGA          (promote net_best if score >= threshold)
  6. log   : Elo vs net_best, tactical-suite score
Generation 0 uses the material eval (no net) as the cold-start teacher.

This calls the engine and the train/parity/match/tactics scripts as subprocesses so each
stage is isolated and reproducible. State (net_best, generation) is tracked in loop_state.json.

    python loop.py --generations 6 --games 800 --workers 4 --depth 8
    python loop.py --generations 6 --resume        # continue from loop_state.json
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
EXE = os.path.join(REPO, "Eonego", "bin", "Release", "net10.0", "Eonego.exe")
KGA = "rnbqkbnr/pppp1ppp/8/8/4Pp2/8/PPPP2PP/RNBQKBNR w KQkq - 0 3"
DATA = os.path.join(HERE, "data")
NETS = os.path.join(HERE, "nets")
STATE = os.path.join(HERE, "loop_state.json")
PY = sys.executable


def run(cmd, **kw):
    print(">>", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, check=True, **kw)


def gen_generation(k, net_best, games, workers, depth, seed_base, random_plies):
    """Run `workers` parallel engine gen processes, merge into data/gen{k}.txt, return its path."""
    merged = os.path.join(DATA, f"gen{k}.txt")
    if os.path.exists(merged):
        recs = sum(1 for ln in open(merged, encoding="utf-8") if not ln.startswith("#"))
        if recs > 100:
            print(f"gen{k}: reusing existing data ({recs} records) at {merged}", flush=True)
            return merged
    per = max(1, games // workers)
    procs, parts = [], []
    for w in range(workers):
        out = os.path.join(DATA, f"gen{k}_{w}.txt")
        parts.append(out)
        cmd = [EXE, "gen", "--start", KGA, "--games", str(per), "--depth", str(depth),
               "--random-plies", str(random_plies), "--max-plies", "200",
               "--seed", str(seed_base + 1009 * w), "--out", out]
        if net_best is not None:
            cmd += ["--net", net_best]
        procs.append(subprocess.Popen(cmd))
    for p in procs:
        p.wait()
    merged = os.path.join(DATA, f"gen{k}.txt")
    with open(merged, "w", encoding="utf-8") as out:
        out.write(f"# gen{k} games={per*workers} depth={depth}\n")
        for part in parts:
            with open(part, encoding="utf-8") as f:
                for line in f:
                    if not line.startswith("#"):
                        out.write(line)
    return merged


def make_window(k, window):
    """Concat data/gen{max(0,k-window+1)..k}.txt -> data/window{k}.txt."""
    lo = max(0, k - window + 1)
    win = os.path.join(DATA, f"window{k}.txt")
    with open(win, "w", encoding="utf-8") as out:
        out.write(f"# window gens {lo}..{k}\n")
        for g in range(lo, k + 1):
            p = os.path.join(DATA, f"gen{g}.txt")
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    for line in f:
                        if not line.startswith("#"):
                            out.write(line)
    return win


def lam_for(k):
    # anneal eval->WDL blend: 1.0 at gen0, easing toward 0.6
    return max(0.6, 1.0 - 0.1 * k)


def train_gen(k, data, init_pt):
    net_out = os.path.join(NETS, f"net{k}.eongnnue")
    pt_out = os.path.join(NETS, f"net{k}.pt")
    cmd = [PY, os.path.join(HERE, "train.py"), "--data", data, "--out", net_out,
           "--save-pt", pt_out, "--K", "160", "--lam", f"{lam_for(k):.2f}",
           "--epochs", "250", "--seed", str(k)]
    if init_pt and os.path.exists(init_pt):
        cmd += ["--init", init_pt]
    run(cmd)
    return net_out, pt_out


def parity_ok(net):
    fens = os.path.join(DATA, "parity_fens.txt")
    r = subprocess.run([PY, os.path.join(HERE, "parity_forward.py"), net, fens])
    return r.returncode == 0


def match_score(net_a, net_b, nodes, openings, seed):
    """Return player-A score fraction vs B from the match driver's FINAL line."""
    r = run([PY, os.path.join(HERE, "match.py"), "--a", net_a, "--b", net_b,
             "--nodes", str(nodes), "--openings", str(openings), "--seed", str(seed)],
            capture_output=True, text=True)
    score = None
    for line in r.stdout.splitlines():
        print(line)
        if "score" in line and "W" in line:
            for tok in line.split():
                pass
        if line.strip().startswith("score"):
            score = float(line.split()[-1])
    # robust: parse 'score <x>' anywhere
    for line in r.stdout.splitlines():
        if " score " in line:
            try:
                score = float(line.split(" score ")[1].split()[0])
            except (IndexError, ValueError):
                pass
    return score if score is not None else 0.0


def tactics_score(net, suite, nodes):
    if not os.path.exists(suite):
        return None
    r = run([PY, os.path.join(HERE, "tactics.py"), "--net", net, "--suite", suite,
             "--nodes", str(nodes)], capture_output=True, text=True)
    print(r.stdout.strip())
    return r.stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=6)
    ap.add_argument("--games", type=int, default=800)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--random-plies", type=int, default=8)
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--match-nodes", type=int, default=20000)
    ap.add_argument("--match-openings", type=int, default=60)
    ap.add_argument("--promote", type=float, default=0.53, help="min A-score vs incumbent to promote")
    ap.add_argument("--suite", default=os.path.join(HERE, "suites", "kga_tactics.tsv"))
    ap.add_argument("--tactics-nodes", type=int, default=50000)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    os.makedirs(NETS, exist_ok=True)
    os.makedirs(DATA, exist_ok=True)

    if args.resume and os.path.exists(STATE):
        st = json.load(open(STATE))
    else:
        st = {"gen": -1, "best": None, "best_pt": None, "history": []}

    for k in range(st["gen"] + 1, args.generations + 1):
        net_best = st["best"]
        print(f"\n===== GENERATION {k} (best={net_best}) =====", flush=True)
        data = gen_generation(k, net_best, args.games, args.workers, args.depth,
                              seed_base=1_000_000 + 100_000 * k, random_plies=args.random_plies)
        window = make_window(k, args.window)
        net_k, pt_k = train_gen(k, window, st["best_pt"])

        if not parity_ok(net_k):
            print(f"PARITY FAILED for {net_k} -- not shipping; stopping.")
            break

        if net_best is None:
            promoted = True
            score = None
        else:
            score = match_score(net_k, net_best, args.match_nodes, args.match_openings, seed=k)
            promoted = score >= args.promote
            print(f"gen{k} match vs best: A-score {score:.3f} -> {'PROMOTE' if promoted else 'keep best'}")

        if promoted:
            st["best"], st["best_pt"] = net_k, pt_k

        tac = tactics_score(st["best"], args.suite, args.tactics_nodes)
        st["gen"] = k
        st["history"].append({"gen": k, "net": net_k, "match_score": score,
                              "promoted": promoted, "tactics": tac})
        json.dump(st, open(STATE, "w"), indent=2)

    print("\nloop done. best =", st["best"])
    print("history:", json.dumps(st["history"], indent=2))


if __name__ == "__main__":
    main()

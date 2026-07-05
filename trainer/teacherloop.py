"""Teacher-distillation reinforcement loop (KGA-anchored, architecture unchanged).

Per generation k:
  1. gen   : engine self-play from KGA with the best net (parallel workers) -> position FENs
  2. label : teacher-label those positions (label.py)                         -> labeled_gen{k}.txt
  3. window: the broad seed pool (teacher_pool0) + the last W generations' teacher labels
  4. train : fine-tune warm-started from the best net's .pt on the window    -> net_gen{k}
  5. match : net_gen{k} vs best from KGA; promote on score >= threshold
This is distribution-matched (positions come from the best net's own play) + strong-teacher
(teacher labels) + warm-start (keeps the good-playing net). Known to cap at the net's
capacity (mid-2000s for the 64-wide region net) — the curve shows where.

    python teacherloop.py --generations 8 --resume
"""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
EXE = os.path.join(REPO, "Eonego", "bin", "Release", "net10.0", "Eonego.exe")
KGA = "rnbqkbnr/pppp1ppp/8/8/4Pp2/8/PPPP2PP/RNBQKBNR w KQkq - 0 3"
DATA = os.path.join(HERE, "data")
NETS = os.path.join(HERE, "nets")
STATE = os.path.join(HERE, "trainloop_state.json")
SEED_POOL = os.path.join(DATA, "teacher_pool0.txt")  # broad material-distribution teacher labels
PY = sys.executable


def run(cmd, **kw):
    print(">>", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, check=True, **kw)


def gen_positions(k, best_net, games, workers, depth, seed_base, random_plies):
    procs, parts = [], []
    per = max(1, games // workers)
    for w in range(workers):
        out = os.path.join(DATA, f"playgen{k}_{w}.txt")
        parts.append(out)
        cmd = [EXE, "gen", "--start", KGA, "--games", str(per), "--depth", str(depth),
               "--random-plies", str(random_plies), "--max-plies", "200",
               "--net", best_net, "--seed", str(seed_base + 1009 * w), "--out", out]
        procs.append(subprocess.Popen(cmd))
    for p in procs:
        p.wait()
    fens = os.path.join(DATA, f"playgen{k}_fens.txt")
    seen = set()
    with open(fens, "w", encoding="utf-8") as out:
        for part in parts:
            if not os.path.exists(part):
                continue
            with open(part, encoding="utf-8") as f:
                for line in f:
                    if line.startswith("#"):
                        continue
                    fen = line.split(";")[0].strip()
                    if fen and fen not in seen:
                        seen.add(fen)
                        out.write(fen + "\n")
    return fens


def label_generation(fens, k, depth, workers):
    out = os.path.join(DATA, f"labeled_gen{k}.txt")
    run([PY, os.path.join(HERE, "label.py"), "--in", fens, "--out", out,
         "--depth", str(depth), "--workers", str(workers)])
    return out


def make_window(k, window):
    win = os.path.join(DATA, f"train_window{k}.txt")
    lo = max(1, k - window + 1)
    srcs = ([SEED_POOL] if os.path.exists(SEED_POOL) else []) + \
           [os.path.join(DATA, f"labeled_gen{g}.txt") for g in range(lo, k + 1)]
    seen = set()
    with open(win, "w", encoding="utf-8") as out:
        out.write(f"# train window: seed + gens {lo}..{k}\n")
        for s in srcs:
            if not os.path.exists(s):
                continue
            with open(s, encoding="utf-8") as f:
                for line in f:
                    if line.startswith("#") or not line.strip():
                        continue
                    if line not in seen:
                        seen.add(line)
                        out.write(line)
    return win


def train_gen(k, data, init_pt, lr, epochs):
    net_out = os.path.join(NETS, f"net_gen{k}.eongnnue")
    pt_out = os.path.join(NETS, f"net_gen{k}.pt")
    if os.path.exists(data + ".npz"):
        os.remove(data + ".npz")
    run([PY, os.path.join(HERE, "train.py"), "--data", data, "--out", net_out,
         "--save-pt", pt_out, "--init", init_pt, "--K", "160", "--lam", "1.0",
         "--lr", str(lr), "--epochs", str(epochs), "--seed", str(k)])
    return net_out, pt_out


def match_score(net_a, net_b, nodes, openings, seed):
    r = run([PY, os.path.join(HERE, "match.py"), "--a", net_a, "--b", net_b,
             "--nodes", str(nodes), "--openings", str(openings), "--seed", str(seed)],
            capture_output=True, text=True)
    score = 0.0
    for line in r.stdout.splitlines():
        print(line)
        if " score " in line:
            try:
                score = float(line.split(" score ")[1].split()[0])
            except (IndexError, ValueError):
                pass
    return score


def tactics(net, suite, nodes):
    if not os.path.exists(suite):
        return None
    r = run([PY, os.path.join(HERE, "tactics.py"), "--net", net, "--suite", suite,
             "--nodes", str(nodes)], capture_output=True, text=True)
    out = r.stdout.strip()
    print(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=8)
    ap.add_argument("--games", type=int, default=1200)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--depth", type=int, default=9)
    ap.add_argument("--random-plies", type=int, default=10)
    ap.add_argument("--label-depth", type=int, default=16)
    ap.add_argument("--label-workers", type=int, default=20)
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--match-nodes", type=int, default=20000)
    ap.add_argument("--match-openings", type=int, default=75)
    ap.add_argument("--promote", type=float, default=0.53)
    ap.add_argument("--suite", default=os.path.join(HERE, "suites", "kga_tactics.tsv"))
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if args.resume and os.path.exists(STATE):
        st = json.load(open(STATE))
    else:
        # seed from net_gen1 (already shown to beat net2)
        st = {"gen": 1,
              "best": os.path.join(NETS, "net_gen1.eongnnue"),
              "best_pt": os.path.join(NETS, "net_gen1.pt"),
              "history": [{"gen": 1, "net": "net_gen1", "match_score": 0.570, "promoted": True}]}
        json.dump(st, open(STATE, "w"), indent=2)

    for k in range(st["gen"] + 1, args.generations + 1):
        best, best_pt = st["best"], st["best_pt"]
        print(f"\n===== GEN {k} (best={os.path.basename(best)}) =====", flush=True)
        fens = gen_positions(k, best, args.games, args.workers, args.depth,
                             seed_base=700000 + 10000 * k, random_plies=args.random_plies)
        label_generation(fens, k, args.label_depth, args.label_workers)
        window = make_window(k, args.window)
        net_k, pt_k = train_gen(k, window, best_pt, args.lr, args.epochs)

        score = match_score(net_k, best, args.match_nodes, args.match_openings, seed=k)
        promoted = score >= args.promote
        print(f"gen{k}: net_gen{k} vs best -> {score:.3f} -> {'PROMOTE' if promoted else 'keep'}", flush=True)
        if promoted:
            st["best"], st["best_pt"] = net_k, pt_k
        tac = tactics(st["best"], args.suite, 50000)
        st["gen"] = k
        st["history"].append({"gen": k, "net": f"net_gen{k}", "match_score": score,
                              "promoted": promoted, "tactics": tac})
        json.dump(st, open(STATE, "w"), indent=2)

    print("\ntrainloop done. best =", st["best"])
    print(json.dumps(st["history"], indent=2))


if __name__ == "__main__":
    main()

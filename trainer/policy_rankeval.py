"""STDLIB-ONLY matched-state rank evaluation — the Phase-2 quality gate.

Compares the policy head against the LIVE history ordering on the SAME in-search states: the
engine's EONEGO_CUTDUMP tap records, at every beta cutoff caused by a non-refutation quiet, the
node's quiet list with the exact MovePick history scores plus the move that cut. For each state we
rank the cutter (a) by the dumped history scores and (b) by the policy head's from+to logits
(engine `dumppolicy` on the state FENs). The head earns Phase 3 only if it ranks cutters better
than history does — offline top-1 against a number history never claimed proves nothing.

    python policy_rankeval.py --exe <Eonego.exe> --net ../nets/main.nnue \
        --policy ../nets/main.policy --cutdump data/cuts.tsv [--tmp data]

Producing cuts.tsv: run 1T fixed-depth searches with EONEGO_CUTDUMP=<path> set (e.g. via
scripts/nodesweep.ps1 -EnvSpec) on a broad FEN set, policy OFF (baseline history scores).
"""

import argparse
import os
import subprocess
import sys

from move_encoder import encode_uci, fen_black_to_move


def rank_of(scores: list[int], target_idx: int) -> int:
    """0-based rank of target by descending score; ties broken pessimistically (strictly-greater)."""
    t = scores[target_idx]
    return sum(1 for s in scores if s > t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--net", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--cutdump", required=True)
    ap.add_argument("--tmp", default=".")
    args = ap.parse_args()

    states = []  # (fen, cutter, [(uci, hist_score)])
    with open(args.cutdump, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4:
                continue
            fen, _depth, cutter, quiets = parts
            pairs = []
            for tok in quiets.split():
                uci, score = tok.rsplit(":", 1)
                pairs.append((uci, int(score)))
            ucis = [u for u, _ in pairs]
            if cutter in ucis and len(pairs) >= 4:
                states.append((fen, cutter, pairs))
    if not states:
        sys.exit("no usable cutdump states")

    fens_path = os.path.join(args.tmp, "rankeval_fens.txt")
    pol_path = os.path.join(args.tmp, "rankeval_pol.txt")
    with open(fens_path, "w", encoding="utf-8") as f:
        for fen, _, _ in states:
            f.write(fen + "\n")
    r = subprocess.run(
        [args.exe, "dumppolicy", "--net", args.net, "--policy", args.policy, "--in", fens_path, "--out", pol_path],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        sys.exit(f"dumppolicy failed: {r.stdout}\n{r.stderr}")

    logits = []  # (from[64], to[64]) per state
    with open(pol_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            logits.append(([int(x) for x in parts[1].split()], [int(x) for x in parts[2].split()]))
    assert len(logits) == len(states)

    n = len(states)
    h_top1 = h_top3 = p_top1 = p_top3 = 0
    h_mrr = p_mrr = 0.0
    for (fen, cutter, pairs), (fl, tl) in zip(states, logits):
        black = fen_black_to_move(fen)
        ucis = [u for u, _ in pairs]
        hist = [s for _, s in pairs]
        pol = []
        for u in ucis:
            fr, to = encode_uci(u, black)
            pol.append(fl[fr] + tl[to])
        ti = ucis.index(cutter)
        hr = rank_of(hist, ti)
        pr = rank_of(pol, ti)
        h_top1 += hr == 0
        h_top3 += hr < 3
        p_top1 += pr == 0
        p_top3 += pr < 3
        h_mrr += 1.0 / (hr + 1)
        p_mrr += 1.0 / (pr + 1)

    print(f"states: {n} (cutter ranked among the node's dumped quiets)")
    print(f"history: top1={h_top1 / n:.3f} top3={h_top3 / n:.3f} mrr={h_mrr / n:.3f}")
    print(f"policy:  top1={p_top1 / n:.3f} top3={p_top3 / n:.3f} mrr={p_mrr / n:.3f}")
    print("VERDICT: " + ("policy BEATS history" if p_mrr > h_mrr else "policy does NOT beat history"))


if __name__ == "__main__":
    main()

"""Score a player (net path or 'material') on the Stockfish-verified KGA tactical suite.

The headline metric: fraction of suite positions where the engine, at a fixed node budget,
plays the verified best move. Reported overall and split by tag (mate / win).

    python tactics.py --net nets/net0.eongnnue --suite suites/kga_tactics.tsv --nodes 50000
    python tactics.py --net material --suite suites/kga_tactics.tsv --nodes 50000
"""

import argparse
import os

import chess
import chess.engine

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
EXE = os.path.join(REPO, "Eonego", "bin", "Release", "net10.0", "Eonego.exe")


def load_suite(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                fen, bm = parts[0], parts[1]
                tag = parts[2] if len(parts) > 2 else "win"
                out.append((fen, bm, tag))
    return out


def score_player(cfg, suite, nodes):
    opts = {"Threads": 1}
    if cfg != "material":
        opts["NnueFile"] = cfg   # before UseNnue: the engine needs the net loaded first
        opts["UseNnue"] = True
    eng = chess.engine.SimpleEngine.popen_uci(EXE)
    eng.configure(opts)
    solved = 0
    by = {}
    try:
        for fen, bm, tag in suite:
            board = chess.Board(fen)
            res = eng.play(board, chess.engine.Limit(nodes=nodes))
            ok = res.move is not None and res.move.uci() == bm
            solved += int(ok)
            t = by.setdefault(tag, [0, 0])
            t[0] += int(ok)
            t[1] += 1
    finally:
        eng.quit()
    return solved, len(suite), by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True, help="net path or 'material'")
    ap.add_argument("--suite", default=os.path.join(HERE, "suites", "kga_tactics.tsv"))
    ap.add_argument("--nodes", type=int, default=50000)
    args = ap.parse_args()

    suite = load_suite(args.suite)
    solved, total, by = score_player(args.net, suite, args.nodes)
    pct = 100.0 * solved / max(1, total)
    tagstr = "  ".join(f"{k}:{v[0]}/{v[1]}" for k, v in sorted(by.items()))
    print(f"TACTICS {args.net}  nodes={args.nodes}  solved {solved}/{total} ({pct:.1f}%)  [{tagstr}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

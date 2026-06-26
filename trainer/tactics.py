"""Score an Eonego configuration on the Stockfish-verified KGA tactical suite.

The headline metric: fraction of suite positions where the engine, at a fixed budget,
plays the verified best move. Reported overall and split by tag (mate / win).

The release engine accepts only Threads + Move Overhead via UCI. `--env` is available
for process-level environment overrides. Budget is `go movetime` by default; `--nodes`
is available for fixed-node runs.

    python tactics.py --movetime 500
"""

import argparse
import os

import chess
import chess.engine

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DEFAULT_EXE = os.path.join(REPO, "Eonego", "bin", "Release", "net10.0", "win-x64", "publish", "Eonego.exe")


def parse_env(spec):
    out = {}
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"bad env override '{part}' (expected NAME=VALUE)")
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    return out


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


def score_player(exe, env_overrides, suite, limit):
    env = dict(os.environ)
    env.update(env_overrides)
    eng = chess.engine.SimpleEngine.popen_uci(exe, env=env)
    eng.configure({"Threads": 1})
    solved = 0
    by = {}
    try:
        for fen, bm, tag in suite:
            board = chess.Board(fen)
            res = eng.play(board, limit)
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
    ap.add_argument("--env", default="", help="process env overrides, e.g. 'COMPlus_TieredCompilation=0'")
    ap.add_argument("--exe", default=DEFAULT_EXE)
    ap.add_argument("--suite", default=os.path.join(HERE, "suites", "kga_tactics.tsv"))
    ap.add_argument("--movetime", type=int, default=500, help="ms per position")
    ap.add_argument("--nodes", type=int, default=0, help="alt budget: go nodes N")
    args = ap.parse_args()

    if not os.path.exists(args.exe):
        raise SystemExit(f"engine not found: {args.exe}\n(build it, or pass --exe)")

    limit = chess.engine.Limit(nodes=args.nodes) if args.nodes > 0 else chess.engine.Limit(time=args.movetime / 1000.0)
    suite = load_suite(args.suite)
    solved, total, by = score_player(args.exe, parse_env(args.env), suite, limit)
    pct = 100.0 * solved / max(1, total)
    tagstr = "  ".join(f"{k}:{v[0]}/{v[1]}" for k, v in sorted(by.items()))
    budget = f"nodes={args.nodes}" if args.nodes > 0 else f"movetime={args.movetime}"
    print(f"TACTICS env=[{args.env}]  {budget}  solved {solved}/{total} ({pct:.1f}%)  [{tagstr}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build a Stockfish-verified KGA tactical suite.

Samples KGA-reachable positions (from gen data or random KGA play), analyses each with
Stockfish at depth D with multipv=2, and keeps the ones where the best move is decisively
better than the second best (a forced mate, or a >=GAP cp swing) -- i.e. a real tactic with
an unambiguous solution. Writes `<fen>\t<bm_uci>\t<tag>` to the suite file.

    python build_suite.py --pool data/gen0_fens.txt --out suites/kga_tactics.tsv --n 120
"""

import argparse
import os
import random

import chess
import chess.engine

HERE = os.path.dirname(os.path.abspath(__file__))
SF = os.environ.get("STOCKFISH", r"C:\Users\Samaritan\bin\stockfish.exe")
KGA_FEN = "rnbqkbnr/pppp1ppp/8/8/4Pp2/8/PPPP2PP/RNBQKBNR w KQkq - 0 3"


def pool_from_file(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t and not t.startswith("#"):
                out.append(t.split(";")[0])
    return out


def random_kga_positions(n, seed):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        b = chess.Board(KGA_FEN)
        for _ in range(rng.randint(6, 24)):
            legal = list(b.legal_moves)
            if not legal or b.is_game_over():
                break
            b.push(rng.choice(legal))
        if not b.is_game_over():
            out.append(b.fen())
    return out


def score_cp(povscore):
    """White-POV score -> a comparable number (mate mapped to large cp)."""
    s = povscore.white()
    if s.is_mate():
        m = s.mate()
        return (100000 - abs(m) * 100) * (1 if m > 0 else -1)
    return s.score()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=None, help="fens/records file; else random KGA play")
    ap.add_argument("--out", default=os.path.join(HERE, "suites", "kga_tactics.tsv"))
    ap.add_argument("--n", type=int, default=120, help="target suite size")
    ap.add_argument("--depth", type=int, default=22)
    ap.add_argument("--gap", type=int, default=200, help="min cp gap best vs 2nd to count as tactical")
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--max-candidates", type=int, default=4000)
    args = ap.parse_args()

    cands = pool_from_file(args.pool) if args.pool else random_kga_positions(args.max_candidates, args.seed)
    rng = random.Random(args.seed)
    rng.shuffle(cands)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    eng = chess.engine.SimpleEngine.popen_uci(SF)
    eng.configure({"Threads": 2, "Hash": 256})

    kept = []
    seen = set()
    try:
        for fen in cands:
            if len(kept) >= args.n:
                break
            try:
                board = chess.Board(fen)
            except ValueError:
                continue
            if board.is_game_over() or board.fen() in seen:
                continue
            if len(list(board.legal_moves)) < 2:
                continue
            seen.add(board.fen())
            info = eng.analyse(board, chess.engine.Limit(depth=args.depth), multipv=2)
            if len(info) < 2:
                continue
            best, second = info[0], info[1]
            s_best = score_cp(best["score"])
            s_second = score_cp(second["score"])
            bm = best["pv"][0]
            is_mate = best["score"].white().is_mate() and best["score"].white().mate() != 0
            tag = "mate" if is_mate else "win"
            # only positions where the side to move has a decisively best move
            stm_sign = 1 if board.turn == chess.WHITE else -1
            margin = (s_best - s_second) * stm_sign
            if is_mate and (best["score"].pov(board.turn).mate() or 0) > 0:
                kept.append((board.fen(), bm.uci(), "mate"))
            elif margin >= args.gap:
                kept.append((board.fen(), bm.uci(), "win"))
    finally:
        eng.quit()

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# fen\tbm_uci\ttag  (Stockfish-verified KGA tactics)\n")
        for fen, bm, tag in kept:
            f.write(f"{fen}\t{bm}\t{tag}\n")
    print(f"wrote {len(kept)} tactical positions to {args.out}")


if __name__ == "__main__":
    main()

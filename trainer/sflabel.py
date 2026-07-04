"""Label FENs with deep teacher-engine eval (WHITE-RELATIVE centipawns), parallelized.

Distillation teacher: external UCI engine. Each worker runs its own engine (1 thread) and
analyses to a fixed depth (or nodes). Mate scores map to a large cp with ply falloff.
Output line format matches the trainer: `<fen>;<cp_white>;<result_white>` (result is a
placeholder 0.5 here — pure-eval distillation uses lambda=1.0; WDL can be added later).

    python sflabel.py --in data/pool_fens.txt --out data/labeled_gen0.txt --depth 16 --workers 20
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import chess
import chess.engine

TEACHER_ENGINE = os.environ.get(
    "TEACHER_ENGINE",
    os.path.join(os.environ.get("USERPROFILE", ""), "bin", "uci-engine.exe"),
)
MATE_CP = 30000  # mate maps to +-(MATE_CP - |mate_in|*100), clamped by the trainer anyway


def _cp_white(score):
    s = score.white()
    if s.is_mate():
        m = s.mate()
        return (MATE_CP - min(abs(m), 200) * 100) * (1 if m > 0 else -1)
    return s.score()


def label_chunk(args):
    fens, depth, nodes, hash_mb, teacher_threads = args
    eng = chess.engine.SimpleEngine.popen_uci(TEACHER_ENGINE)
    eng.configure({"Threads": teacher_threads, "Hash": hash_mb})
    limit = chess.engine.Limit(nodes=nodes) if nodes else chess.engine.Limit(depth=depth)
    out = []
    try:
        for fen in fens:
            try:
                board = chess.Board(fen)
            except ValueError:
                continue
            if board.is_game_over():
                continue
            info = eng.analyse(board, limit)
            out.append((fen, _cp_white(info["score"])))
    finally:
        eng.quit()
    return out


def read_fens(path):
    seen, out = set(), []
    with open(path, encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if not t or t.startswith("#"):
                continue
            fen = t.split(";")[0]  # tolerate fen or fen;cp;res records
            if fen not in seen:
                seen.add(fen)
                out.append(fen)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--nodes", type=int, default=0, help="if >0, use fixed nodes instead of depth")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--teacher-threads", type=int, default=1, help="threads per teacher process (workers * teacher-threads should be <= cores)")
    ap.add_argument("--hash", type=int, default=16)
    args = ap.parse_args()

    fens = read_fens(args.inp)
    w = max(1, args.workers)
    chunks = [fens[i::w] for i in range(w)]  # round-robin -> even load
    payloads = [(c, args.depth, args.nodes, args.hash, args.teacher_threads) for c in chunks if c]

    print(f"labeling {len(fens)} unique FENs with teacher engine "
          f"({'nodes ' + str(args.nodes) if args.nodes else 'depth ' + str(args.depth)}), "
          f"{w} workers x {args.teacher_threads} threads, hash {args.hash}MB", flush=True)

    results = []
    done = 0
    with ProcessPoolExecutor(max_workers=w) as ex:
        for chunk_result in ex.map(label_chunk, payloads):
            results.extend(chunk_result)
            done += len(chunk_result)
            print(f"  labeled {done}/{len(fens)}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(f"# teacher-labeled  "
                f"{'nodes=' + str(args.nodes) if args.nodes else 'depth=' + str(args.depth)}  "
                f"n={len(results)}\n")
        for fen, cp in results:
            f.write(f"{fen};{cp};0.5\n")
    print(f"wrote {len(results)} labeled positions to {args.out}", flush=True)


if __name__ == "__main__":
    main()

"""Label FENs with deep teacher-engine eval (WHITE-RELATIVE centipawns), parallelized,
with CHECKPOINTING: results are appended and flushed to --out as each small batch completes,
and a rerun with the same --out resumes where it left off (already-labeled FENs are skipped).
Pause = kill the process tree at any time; at most the in-flight batches are lost.

Distillation teacher: external UCI engine (TEACHER_ENGINE env; the Eonego publish exe —
2026-07-05 directive: Eonego is its own teacher, no external engines). Each worker runs its
own engine (1 thread by default) and analyses to a fixed depth (or nodes). Mate scores map
to a large cp with ply falloff. Output line format matches the trainer:
`<fen>;<cp_white>;<result_white>` (result is a placeholder 0.5 — join_labels.py grafts the
teacher cp onto the self-play results by FEN key, so OUTPUT ORDER DOES NOT MATTER).

    python label.py --in data/pool_fens.txt --out data/labeled.txt --depth 25 --workers 16
    # ... interrupted; later, the SAME command resumes from the checkpoint.
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

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
    fens, depth, nodes, hash_mb, teacher_threads, max_time = args

    def spawn():
        e = chess.engine.SimpleEngine.popen_uci(TEACHER_ENGINE)
        e.configure({"Threads": teacher_threads, "Hash": hash_mb})
        return e

    # depth + time cap = `go depth D movetime T`: the engine stops at whichever comes first.
    # The cap only trims the pathological tail (positions whose depth-D search runs away) —
    # without it one such position wedges its worker forever (analyse has no builtin timeout).
    if nodes:
        limit = chess.engine.Limit(nodes=nodes)
    elif max_time > 0:
        limit = chess.engine.Limit(depth=depth, time=max_time)
    else:
        limit = chess.engine.Limit(depth=depth)

    eng = spawn()
    out = []
    try:
        for fen in fens:
            try:
                board = chess.Board(fen)
            except ValueError:
                continue
            if board.is_game_over():
                continue
            try:
                info = eng.analyse(board, limit)
                out.append((fen, _cp_white(info["score"])))
            except chess.engine.EngineError:
                # engine died or protocol hiccup: skip this fen, respawn, keep the batch alive
                try:
                    eng.close()
                except Exception:
                    pass
                eng = spawn()
    finally:
        try:
            eng.quit()
        except Exception:
            pass
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


def read_done(path):
    """FENs already present in a previous (possibly partial) output — the checkpoint."""
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if not t or t.startswith("#"):
                continue
            parts = t.split(";")
            if len(parts) >= 2:
                done.add(parts[0])
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--nodes", type=int, default=0, help="if >0, use fixed nodes instead of depth")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--teacher-threads", type=int, default=1,
                    help="threads per teacher process (workers * teacher-threads should be <= cores)")
    ap.add_argument("--hash", type=int, default=16)
    ap.add_argument("--batch", type=int, default=64,
                    help="fens per work unit = checkpoint granularity (flushed on completion)")
    ap.add_argument("--max-time", type=float, default=0.0,
                    help="per-position wall cap in seconds alongside --depth (0 = uncapped)")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore an existing --out and start over (default: resume/append)")
    args = ap.parse_args()

    fens = read_fens(args.inp)
    done = set() if args.fresh else read_done(args.out)
    todo = [f for f in fens if f not in done]

    budget = "nodes " + str(args.nodes) if args.nodes else "depth " + str(args.depth)
    print(f"labeling with teacher engine ({budget}), {args.workers} workers x "
          f"{args.teacher_threads} threads, hash {args.hash}MB, batch {args.batch}", flush=True)
    print(f"  input {len(fens)} unique FENs; checkpoint has {len(done)}; to do {len(todo)}", flush=True)

    if not todo:
        print("nothing to do — checkpoint already covers the input", flush=True)
        return

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    mode = "w" if (args.fresh or not os.path.exists(args.out)) else "a"

    batches = [todo[i:i + args.batch] for i in range(0, len(todo), args.batch)]
    payloads = [(b, args.depth, args.nodes, args.hash, args.teacher_threads, args.max_time)
                for b in batches]

    written = 0
    with open(args.out, mode, encoding="utf-8", buffering=1) as f:
        if mode == "w":
            f.write(f"# teacher-labeled  {budget.replace(' ', '=')}  (checkpointed; order arbitrary)\n")
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futs = [ex.submit(label_chunk, p) for p in payloads]
            try:
                for fut in as_completed(futs):
                    for fen, cp in fut.result():
                        f.write(f"{fen};{cp};0.5\n")
                    f.flush()
                    os.fsync(f.fileno())
                    written += 1
                    if written % 4 == 0 or written == len(futs):
                        done_n = len(done) + min(written * args.batch, len(todo))
                        print(f"  checkpoint {done_n}/{len(fens)} "
                              f"({100.0 * done_n / len(fens):.1f}%)", flush=True)
            except KeyboardInterrupt:
                print("interrupted — checkpoint is flushed; rerun the same command to resume",
                      flush=True)
                for x in futs:
                    x.cancel()
                sys.exit(130)

    print(f"done: {len(done) + len(todo)} labeled positions in {args.out}", flush=True)


if __name__ == "__main__":
    main()

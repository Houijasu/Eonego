"""Self-contained UCI match driver + SPRT for Eonego processes.

Per-player channels:
  --a / --b / --shared          env-var overrides ('NAME=VAL,NAME2=VAL2') — the engine's EONEGO_* knobs
  --options-a / --options-b / --options
                                UCI options sent via `setoption` after `uciok` (same grammar)
  --exe / --exe-b               different binaries (old-vs-new matches)

Concurrency: --concurrency N runs N worker threads, each owning ONE engine pair and playing whole
color-swapped opening PAIRS (both games of a pair on the same worker, so a thermally-throttled or
E-core slot penalizes both players symmetrically). Openings are generated once from --seed and
assigned deterministically by pair index; only completion order varies between runs.

    python match.py --a "" --b "" --movetime 200 --openings 200 --sprt --concurrency 8

`run_match(args)` is importable (used by spsa.py); `main()` is a thin CLI wrapper.
"""

import argparse
import math
import os
import queue
import subprocess
import threading

import chess

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DEFAULT_EXE = os.path.join(REPO, "Eonego", "bin", "Release", "net10.0", "win-x64", "publish", "Eonego.exe")
START_FEN = chess.STARTING_FEN


class UciEngine:
    def __init__(self, name, exe, env_overrides, uci_options=None):
        env = dict(os.environ)
        env.update(env_overrides)
        self.proc = subprocess.Popen([exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     text=True, bufsize=1, env=env)
        self.name = name
        self._send("uci")
        self._wait("uciok")
        opts = dict(uci_options or {})
        # Legacy default: pin Threads=1 unless the caller overrides it explicitly.
        if "Threads" not in opts:
            opts["Threads"] = "1"
        for k, v in opts.items():
            self._send(f"setoption name {k} value {v}")
        self._send("isready")
        self._wait("readyok")

    def _send(self, s):
        self.proc.stdin.write(s + "\n")
        self.proc.stdin.flush()

    def _wait(self, token):
        while True:
            line = self.proc.stdout.readline()
            if line == "":
                raise RuntimeError(f"{self.name}: engine closed waiting for {token}")
            if line.startswith(token):
                return

    def newgame(self):
        self._send("ucinewgame")
        self._send("isready")
        self._wait("readyok")

    def bestmove(self, root_fen, moves, go_cmd):
        mv = " ".join(moves)
        pos = f"position fen {root_fen}" + (f" moves {mv}" if moves else "")
        self._send(pos)
        self._send(go_cmd)
        while True:
            line = self.proc.stdout.readline()
            if line == "":
                raise RuntimeError(f"{self.name}: engine closed during search")
            if line.startswith("bestmove"):
                return line.split()[1]

    def quit(self):
        try:
            self._send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def parse_env(spec):
    """'NAME=VAL,NAME2=VAL2' -> {'NAME': 'VAL', ...}. Empty/None -> {}."""
    out = {}
    if not spec:
        return out
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"bad env override '{part}' (expected NAME=VALUE)")
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    return out


# Same grammar as parse_env; separate name so call sites read as intended.
parse_options = parse_env


def gen_openings(root_fen, n, plies, seed):
    """n distinct opening move-lists from the root: `plies` random legal moves each."""
    import random
    rng = random.Random(seed)
    seen, out = set(), []
    tries = 0
    while len(out) < n and tries < n * 50:
        tries += 1
        b = chess.Board(root_fen)
        ok = True
        for _ in range(plies):
            legal = list(b.legal_moves)
            if not legal:
                ok = False
                break
            b.push(rng.choice(legal))
        if not ok or b.is_game_over():
            continue
        key = b.fen()
        if key in seen:
            continue
        seen.add(key)
        out.append([m.uci() for m in b.move_stack])
    return out


def play_game(root_fen, eng_white, eng_black, opening, go_cmd, max_plies=600):
    board = chess.Board(root_fen)
    for u in opening:
        board.push_uci(u)
    moves = list(opening)
    while not board.is_game_over(claim_draw=True) and len(moves) < max_plies:
        eng = eng_white if board.turn == chess.WHITE else eng_black
        u = eng.bestmove(root_fen, moves, go_cmd)
        try:
            mv = chess.Move.from_uci(u)
        except ValueError:
            return "0-1" if board.turn == chess.WHITE else "1-0"  # illegal = loss
        if mv not in board.legal_moves:
            return "0-1" if board.turn == chess.WHITE else "1-0"
        board.push(mv)
        moves.append(u)
    out = board.outcome(claim_draw=True)
    if out is None:
        return "1/2-1/2"  # hit move cap -> draw
    return out.result()


def sprt_llr(W, D, L, elo0, elo1):
    """GSPRT log-likelihood ratio (Laplace-smoothed so it never degenerates early)."""
    n = W + D + L
    if n == 0:
        return 0.0
    def escore(elo):
        return 1.0 / (1.0 + 10 ** (-elo / 400.0))
    s0, s1 = escore(elo0), escore(elo1)
    den = n + 1.5
    w, d, ll = (W + 0.5) / den, (D + 0.5) / den, (L + 0.5) / den
    mu = w + 0.5 * d
    var = w * (1 - mu) ** 2 + d * (0.5 - mu) ** 2 + ll * (0 - mu) ** 2
    return n * (s1 - s0) * (mu - (s0 + s1) / 2.0) / var


def elo_estimate(W, D, L):
    n = W + D + L
    if n == 0:
        return 0.0, 0.0
    score = (W + 0.5 * D) / n
    score = min(max(score, 1e-6), 1 - 1e-6)
    elo = -400.0 * math.log10(1.0 / score - 1.0)
    var = (W * (1 - score) ** 2 + D * (0.5 - score) ** 2 + L * score ** 2) / (n * n)
    se = math.sqrt(max(var, 1e-12))
    s_hi = min(1 - 1e-6, score + 1.96 * se)
    s_lo = max(1e-6, score - 1.96 * se)
    elo_hi = -400.0 * math.log10(1.0 / s_hi - 1.0)
    elo_lo = -400.0 * math.log10(1.0 / s_lo - 1.0)
    return elo, (elo_hi - elo_lo) / 2.0


def run_match(args, quiet=False):
    """Play the configured match; returns dict(W, D, L, games, elo, err, llr, verdict).

    Thread-pool design: each worker owns one (engA, engB) pair for its lifetime and plays whole
    color-swapped opening pairs pulled from a shared queue. SPRT stop is evaluated on the aggregate
    tally after every game; workers finish their in-flight game, then drain.
    """
    exe_a = args.exe
    exe_b = getattr(args, "exe_b", "") or args.exe
    if not os.path.exists(exe_a):
        raise SystemExit(f"engine not found: {exe_a}\n(build it, or pass --exe)")
    if not os.path.exists(exe_b):
        raise SystemExit(f"engine B not found: {exe_b}")

    go_cmd = f"go nodes {args.nodes}" if args.nodes > 0 else f"go movetime {args.movetime}"

    shared_env = parse_env(args.shared)
    env_a = {**shared_env, **parse_env(args.a)}
    env_b = {**shared_env, **parse_env(args.b)}
    shared_opt = parse_options(getattr(args, "options", ""))
    opt_a = {**shared_opt, **parse_options(getattr(args, "options_a", ""))}
    opt_b = {**shared_opt, **parse_options(getattr(args, "options_b", ""))}

    lower = math.log(args.beta / (1 - args.alpha))
    upper = math.log((1 - args.beta) / args.alpha)

    openings = gen_openings(args.root, args.openings, args.opening_plies, args.seed)
    work = queue.Queue()
    for i in range(len(openings)):
        work.put(i)

    tally = {"W": 0, "D": 0, "L": 0, "games": 0}
    lock = threading.Lock()
    stop = threading.Event()
    errors = []

    def record(res, a_is_white):
        """Post one game result; returns True when the SPRT says stop."""
        with lock:
            if res == "1/2-1/2":
                tally["D"] += 1
            elif (res == "1-0") == a_is_white:
                tally["W"] += 1
            else:
                tally["L"] += 1
            tally["games"] += 1
            W, D, L, g = tally["W"], tally["D"], tally["L"], tally["games"]
            llr = sprt_llr(W, D, L, args.elo0, args.elo1)
            if not quiet and g % 10 == 0:
                elo, err = elo_estimate(W, D, L)
                print(f"g{g:4d}  W{W} D{D} L{L}  elo {elo:+.1f}+-{err:.1f}  LLR {llr:+.2f} "
                      f"[{lower:.2f},{upper:.2f}]", flush=True)
            return args.sprt and g >= args.min_games and (llr >= upper or llr <= lower)

    def worker():
        engA = engB = None
        try:
            engA = UciEngine("A", exe_a, env_a, opt_a)
            engB = UciEngine("B", exe_b, env_b, opt_b)
            while not stop.is_set():
                try:
                    idx = work.get_nowait()
                except queue.Empty:
                    return
                op = openings[idx]
                for a_is_white in (True, False):
                    if stop.is_set():
                        return
                    engA.newgame()
                    engB.newgame()
                    ew, eb = (engA, engB) if a_is_white else (engB, engA)
                    res = play_game(args.root, ew, eb, op, go_cmd, max_plies=args.max_plies)
                    if record(res, a_is_white):
                        stop.set()
        except Exception as ex:  # engine crash etc. — stop the match, surface the error
            with lock:
                errors.append(f"{type(ex).__name__}: {ex}")
            stop.set()
        finally:
            for e in (engA, engB):
                if e is not None:
                    e.quit()

    n_workers = max(1, min(args.concurrency, len(openings)))
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(n_workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    W, D, L, g = tally["W"], tally["D"], tally["L"], tally["games"]
    llr = sprt_llr(W, D, L, args.elo0, args.elo1)
    elo, err = elo_estimate(W, D, L)
    verdict = ("H1 (A stronger)" if llr >= upper
               else "H0 (not stronger)" if llr <= lower
               else "inconclusive")
    return {"W": W, "D": D, "L": L, "games": g, "elo": elo, "err": err,
            "llr": llr, "verdict": verdict, "errors": errors}


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="env overrides for player A, e.g. 'EONEGO_T_RFP_MARGIN=130'")
    ap.add_argument("--b", required=True, help="env overrides for player B")
    ap.add_argument("--shared", default="", help="env overrides applied to BOTH players")
    ap.add_argument("--options-a", default="", help="UCI options for player A, e.g. 'Threads=16,Hash=1024'")
    ap.add_argument("--options-b", default="", help="UCI options for player B")
    ap.add_argument("--options", default="", help="UCI options applied to BOTH players")
    ap.add_argument("--exe", default=DEFAULT_EXE, help="path to Eonego.exe (default: AOT publish build)")
    ap.add_argument("--exe-b", default="", help="path to player B's exe (default: same as --exe) — for old-vs-new binary matches")
    ap.add_argument("--root", default=START_FEN, help="opening root FEN (default: startpos)")
    ap.add_argument("--movetime", type=int, default=200, help="ms per move")
    ap.add_argument("--nodes", type=int, default=0, help="alt budget: go nodes N")
    ap.add_argument("--openings", type=int, default=80)
    ap.add_argument("--opening-plies", type=int, default=6)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--elo0", type=float, default=0.0)
    ap.add_argument("--elo1", type=float, default=10.0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--sprt", action="store_true", help="enable early stop on SPRT bounds")
    ap.add_argument("--min-games", type=int, default=40)
    ap.add_argument("--max-plies", type=int, default=600,
                    help="adjudicate as draw past this many plies; use <=240 when a pre-2d3fd08 binary "
                         "plays (their accumulator stack crashed past ~255 game plies)")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="worker threads, each owning one engine pair and playing whole opening pairs; "
                         "use 8 for 1T-vs-1T matches, 1 when any player runs Threads>=8")
    return ap


def main():
    args = build_parser().parse_args()
    r = run_match(args)
    print(f"\nFINAL  A=[{args.a}|{args.options_a}]  B=[{args.b}|{args.options_b}]  "
          f"budget={'go nodes ' + str(args.nodes) if args.nodes > 0 else 'go movetime ' + str(args.movetime)}"
          f"  concurrency={args.concurrency}")
    print(f"  games {r['games']}  W{r['W']} D{r['D']} L{r['L']}  score {(r['W'] + 0.5 * r['D']) / max(1, r['games']):.3f}")
    print(f"  Elo(A-B) {r['elo']:+.1f} +- {r['err']:.1f}   LLR {r['llr']:+.2f}  -> {r['verdict']}")
    for e in r["errors"]:
        print(f"  ERROR: {e}")
    return 1 if r["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Self-contained UCI match driver + SPRT for Eonego processes.

The release engine exposes only `Threads` and `Move Overhead` via UCI. `--a`/`--b`
remain available as comma-separated per-subprocess environment overrides for comparing
different launches or binaries, but Eonego itself does not consume search-mode env knobs.

    python match.py --a "" --b "" --movetime 200 --openings 200 --sprt
"""

import argparse
import math
import os
import subprocess

import chess

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DEFAULT_EXE = os.path.join(REPO, "Eonego", "bin", "Release", "net10.0", "win-x64", "publish", "Eonego.exe")
START_FEN = chess.STARTING_FEN


class UciEngine:
    def __init__(self, name, exe, env_overrides):
        env = dict(os.environ)
        env.update(env_overrides)
        self.proc = subprocess.Popen([exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     text=True, bufsize=1, env=env)
        self.name = name
        self._send("uci")
        self._wait("uciok")
        self._send("setoption name Threads value 1")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="env overrides for player A, e.g. 'COMPlus_TieredCompilation=0'")
    ap.add_argument("--b", required=True, help="env overrides for player B, e.g. 'COMPlus_TieredCompilation=1'")
    ap.add_argument("--shared", default="", help="env overrides applied to BOTH players")
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
    args = ap.parse_args()

    if not os.path.exists(args.exe):
        raise SystemExit(f"engine not found: {args.exe}\n(build it, or pass --exe)")

    exe_b = args.exe_b or args.exe
    if not os.path.exists(exe_b):
        raise SystemExit(f"engine B not found: {exe_b}")

    go_cmd = f"go nodes {args.nodes}" if args.nodes > 0 else f"go movetime {args.movetime}"

    shared = parse_env(args.shared)
    env_a = {**shared, **parse_env(args.a)}
    env_b = {**shared, **parse_env(args.b)}

    lower = math.log(args.beta / (1 - args.alpha))
    upper = math.log((1 - args.beta) / args.alpha)

    engA = UciEngine("A", args.exe, env_a)
    engB = UciEngine("B", exe_b, env_b)
    openings = gen_openings(args.root, args.openings, args.opening_plies, args.seed)

    W = D = L = 0  # from A's perspective
    games = 0
    try:
        for op in openings:
            for a_is_white in (True, False):
                engA.newgame()
                engB.newgame()
                ew, eb = (engA, engB) if a_is_white else (engB, engA)
                res = play_game(args.root, ew, eb, op, go_cmd, max_plies=args.max_plies)
                if res == "1/2-1/2":
                    D += 1
                elif (res == "1-0") == a_is_white:
                    W += 1
                else:
                    L += 1
                games += 1
                llr = sprt_llr(W, D, L, args.elo0, args.elo1)
                if games % 10 == 0:
                    elo, err = elo_estimate(W, D, L)
                    print(f"g{games:4d}  W{W} D{D} L{L}  elo {elo:+.1f}+-{err:.1f}  LLR {llr:+.2f} "
                          f"[{lower:.2f},{upper:.2f}]", flush=True)
                if args.sprt and games >= args.min_games and (llr >= upper or llr <= lower):
                    break
            else:
                continue
            break
    finally:
        engA.quit()
        engB.quit()

    llr = sprt_llr(W, D, L, args.elo0, args.elo1)
    elo, err = elo_estimate(W, D, L)
    verdict = "H1 (A stronger)" if llr >= upper else "H0 (not stronger)" if llr <= lower else "inconclusive"
    print(f"\nFINAL  A=[{args.a}]  B=[{args.b}]  budget={go_cmd}")
    print(f"  games {games}  W{W} D{D} L{L}  score {(W+0.5*D)/max(1,games):.3f}")
    print(f"  Elo(A-B) {elo:+.1f} +- {err:.1f}   LLR {llr:+.2f}  -> {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

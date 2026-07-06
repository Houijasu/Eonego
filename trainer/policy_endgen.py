"""4-man endgame position generator with Syzygy-labeled policy targets -> gen-v2 records.

TRAINER-SIDE ONLY: the probe is python-chess's native Syzygy reader (.rtbw/.rtbz); the engine
stays zero-dependency and never sees a tablebase. Install BOTH 3-man and 4-man files in --tb
(captures from a 4-man position land in 3-man — child probes need them).

Label hygiene (the CE-noise and capacity guards):
  * target must be a QUIET move (engine isQuiet: no capture, no promotion) and STRICTLY best:
    higher WDL than every alternative move, or — among equal-WDL winning moves — strictly
    smallest post-move DTZ. Ties are DROPPED, not arbitrated (CE against an arbitrary tie
    member is label noise).
  * --per-signature caps each signature and --total caps the file, so the endgame slice enters
    the training mix as a controlled fraction (plan: a few percent of the middlegame corpus).
  * result_white comes from the root WDL — perfectly calibrated WDL-head labels for bucket 0.
  * cp_white is a coarse WDL->cp map (policy training ignores it; only value pipelines read it).

    uv run --with chess python policy_endgen.py --tb <syzygy_dir> \
        --out data/pol1/endgen4.txt --per-signature 8000 --total 60000

    --dry-run generates and validates positions without probing (no files needed yet).

DTZ note: post-move DTZ minimization is an optimal-play PROXY (DTZ optimizes 50-move resets,
not mate distance). Combined with the strict-uniqueness filter this yields clean single-target
labels; positions where DTZ-optimal is ambiguous or unnatural simply don't make the cut.
"""

import argparse
import random
import sys

import chess
import chess.syzygy

PIECE_OF = {"Q": chess.QUEEN, "R": chess.ROOK, "B": chess.BISHOP, "N": chess.KNIGHT, "P": chess.PAWN}

# Signature grammar: strong side pieces + 'v' + weak side pieces (kings implicit).
DEFAULT_SIGNATURES = [
    "KQvKR", "KQvKB", "KQvKN", "KQvKP",
    "KRvKB", "KRvKN", "KRvKP",
    "KBNvK", "KBBvK",
    "KPvKP", "KRvKR", "KQvKQ",
]


def parse_signature(sig: str):
    strong, weak = sig.split("v")
    return [PIECE_OF[c] for c in strong[1:]], [PIECE_OF[c] for c in weak[1:]]


def random_position(rng: random.Random, strong, weak):
    """One random legal position of the signature (either side to move), or None."""
    board = chess.Board.empty()
    squares = rng.sample(chess.SQUARES, 2 + len(strong) + len(weak))
    it = iter(squares)
    board.set_piece_at(next(it), chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(next(it), chess.Piece(chess.KING, chess.BLACK))
    for pt in strong:
        board.set_piece_at(next(it), chess.Piece(pt, chess.WHITE))
    for pt in weak:
        board.set_piece_at(next(it), chess.Piece(pt, chess.BLACK))
    board.turn = rng.choice([chess.WHITE, chess.BLACK])
    board.castling_rights = 0
    if not board.is_valid():  # king adjacency, side-not-to-move in check, pawns on back ranks...
        return None
    if board.is_game_over():
        return None
    return board


def is_quiet(board: chess.Board, mv: chess.Move) -> bool:
    return mv.promotion is None and not board.is_capture(mv)


def label(tb, board: chess.Board, wdl_only: bool = False):
    """(best_quiet_uci, root_wdl_stm) with the strict-uniqueness rule, or None.

    WDL-first: when one move strictly beats every alternative on WDL alone, no DTZ probe runs.
    DTZ is only the tie-breaker among equal-WDL winning moves — and any DTZ failure (missing
    table, or python-chess's DTZ decompression bug on some tables) just drops the position
    (fail-closed; label hygiene beats coverage)."""
    # Broad excepts on every probe: beyond MissingTableError, current python-chess can raise
    # TypeError/IndexError from its decompressor on rare blocks (observed on this 3-4-5 set);
    # fail-closed — a dropped position costs nothing, a bad label poisons training.
    try:
        root_wdl = tb.probe_wdl(board)
    except (chess.syzygy.MissingTableError, KeyError, TypeError, ValueError, IndexError):
        return None

    scored = []  # (wdl_stm, quiet, move)
    for mv in board.legal_moves:
        quiet = is_quiet(board, mv)
        board.push(mv)
        try:
            w = -tb.probe_wdl(board)
        except (chess.syzygy.MissingTableError, KeyError, TypeError, ValueError, IndexError):
            board.pop()
            return None
        board.pop()
        scored.append((w, quiet, mv))

    best_w = max(s[0] for s in scored)
    top = [s for s in scored if s[0] == best_w]

    if len(top) == 1:
        w, quiet, mv = top[0]
        return (mv.uci(), root_wdl) if quiet else None

    if best_w <= 0 or wdl_only:
        return None  # drawn/lost with several equal moves — or DTZ probing disabled: no unique target

    # Equal-WDL winning moves: break the tie by post-move DTZ, strictly.
    dtz = []
    for w, quiet, mv in top:
        board.push(mv)
        try:
            d = abs(tb.probe_dtz(board))
        except (chess.syzygy.MissingTableError, KeyError, TypeError, ValueError, IndexError):
            board.pop()
            return None
        board.pop()
        dtz.append((d, quiet, mv))
    dtz.sort(key=lambda s: s[0])
    if len(dtz) > 1 and dtz[0][0] == dtz[1][0]:
        return None  # DTZ tie: ambiguous
    d, quiet, mv = dtz[0]
    return (mv.uci(), root_wdl) if quiet else None


def wdl_to_result_white(wdl_stm: int, white_to_move: bool) -> float:
    w = wdl_stm if white_to_move else -wdl_stm
    return 1.0 if w > 0 else (0.0 if w < 0 else 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tb", default=None, help="Syzygy directory (.rtbw/.rtbz, 3-man AND 4-man)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--signatures", default=",".join(DEFAULT_SIGNATURES))
    ap.add_argument("--per-signature", type=int, default=8000)
    ap.add_argument("--total", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tries-mult", type=int, default=60, help="sampling attempts per kept record")
    ap.add_argument("--dry-run", action="store_true", help="validate generation only, no probing")
    ap.add_argument(
        "--wdl-only",
        action="store_true",
        help="never probe DTZ: accept only positions whose best move is WDL-strictly-unique "
        "(drops win-speed ties; sidesteps python-chess's fragile DTZ recursion)",
    )
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help="crash-isolated mode: run per-(signature,chunk) SUBPROCESSES of this many records and "
        "concatenate — python-chess's Syzygy prober can die natively (access violation / C-stack "
        "overflow) on rare positions, which no except catches; a crashed chunk is logged and skipped "
        "(downstream policy_merge dedupes across chunks anyway)",
    )
    args = ap.parse_args()

    if args.chunk_size > 0:
        import math
        import os
        import subprocess

        base, ext = os.path.splitext(args.out)
        parts, failed = [], 0
        for sig in args.signatures.split(","):
            sig = sig.strip()
            n_chunks = math.ceil(args.per_signature / args.chunk_size)
            for c in range(n_chunks):
                k = min(args.chunk_size, args.per_signature - c * args.chunk_size)
                part = f"{base}.{sig}.{c}{ext}"
                if not os.path.exists(part):  # checkpoint: rerun resumes
                    cmd = [sys.executable, os.path.abspath(__file__), "--tb", args.tb, "--out", part,
                           "--signatures", sig, "--per-signature", str(k), "--total", str(k),
                           "--seed", str(args.seed + c * 7919), "--max-tries-mult", str(args.max_tries_mult)]
                    if args.wdl_only:
                        cmd.append("--wdl-only")
                    r = subprocess.run(cmd, capture_output=True, text=True)
                    if r.returncode != 0:
                        failed += 1
                        print(f"chunk {sig}#{c} CRASHED (exit {r.returncode}) — skipped")
                        if os.path.exists(part):
                            os.remove(part)
                        continue
                parts.append(part)
        kept = 0
        with open(args.out, "w", encoding="utf-8") as out:
            out.write("# eonego gen v2 endgen4 fen;cp_white;result_white;best_uci\n")
            for part in parts:
                with open(part, encoding="utf-8") as f:
                    for line in f:
                        if line.strip() and not line.startswith("#"):
                            out.write(line)
                            kept += 1
        print(f"CHUNKED TOTAL {kept} records -> {args.out} ({failed} chunks crashed/skipped)")
        return

    if not args.dry_run:
        if not args.tb:
            sys.exit("--tb required (or use --dry-run)")
        # Comma-separated directories (WDL and DTZ files are often distributed in separate folders).
        dirs = [d.strip() for d in args.tb.split(",") if d.strip()]
        tb = chess.syzygy.open_tablebase(dirs[0])
        for d in dirs[1:]:
            tb.add_directory(d)
    rng = random.Random(args.seed)

    kept_total = 0
    with open(args.out, "w", encoding="utf-8") as out:
        out.write("# eonego gen v2 endgen4 fen;cp_white;result_white;best_uci\n")
        for sig in args.signatures.split(","):
            sig = sig.strip()
            strong, weak = parse_signature(sig)
            seen = set()
            kept = tries = 0
            budget = args.per_signature * args.max_tries_mult
            while kept < args.per_signature and tries < budget and kept_total < args.total:
                tries += 1
                board = random_position(rng, strong, weak)
                if board is None:
                    continue
                key = board.board_fen() + (" w" if board.turn else " b")
                if key in seen:
                    continue
                seen.add(key)
                if args.dry_run:
                    kept += 1
                    kept_total += 1
                    continue
                lab = label(tb, board, args.wdl_only)
                if lab is None:
                    continue
                best_uci, root_wdl = lab
                res = wdl_to_result_white(root_wdl, board.turn == chess.WHITE)
                cp_stm = 1000 if root_wdl > 0 else (-1000 if root_wdl < 0 else 0)
                cp_white = cp_stm if board.turn == chess.WHITE else -cp_stm
                out.write(f"{board.fen()};{cp_white};{res:.1f};{best_uci}\n")
                kept += 1
                kept_total += 1
            print(f"{sig}: kept {kept} ({tries} tries)")
    print(f"TOTAL {kept_total} records -> {args.out}")


if __name__ == "__main__":
    main()

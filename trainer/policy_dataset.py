"""Policy-head training data: gen/tbgen records + dumpft FT buffers -> padded quiet-conditional arrays.

Record formats (index-aligned with the dumpft binary by construction — feed the SAME file to
`Eonego.exe dumpft`; # comments skipped by both readers):
  gen v2   : fen;cp_white;result_white;best_uci
  tbgen v2 : fen;cp_white;result_white;best_uci;good_ucis;quiet_ucis
dumpft binary: 1034-byte records [bucket u8][stm u8][psqt i32][eval i32][ft u8*1024]

The loss is QUIET-CONDITIONAL and (for tbgen data) MULTI-LABEL: the softmax runs over the
position's legal quiets, and the target is the SET of WDL-preserving quiets (good_ucis ∩ quiets)
— "100% correct play" = argmax lands anywhere in that set. Rows where the set is empty (only a
tactical move preserves the result) or covers every quiet (zero discrimination, e.g. all moves
lose equally) are dropped. gen-v2 rows degrade to a one-hot set = {best_uci} via python-chess.
"""

import numpy as np

try:
    import chess
except ImportError:  # pragma: no cover
    chess = None

from move_encoder import board_of_fen, encode_uci_piece, fen_black_to_move

L1 = 1024
QMAX = 72  # padded quiet-list width
DUMP_DTYPE = np.dtype(
    [("bucket", "u1"), ("stm", "u1"), ("psqt", "<i4"), ("eval", "<i4"), ("ft", "u1", (L1,))]
)


def read_dump(path):
    recs = np.fromfile(path, dtype=DUMP_DTYPE)
    assert recs.size > 0, f"empty dump {path}"
    return recs


def read_gen(path):
    """Records -> list of (fen, cp_white, result_white, best_uci, good_str, quiet_str).
    good_str/quiet_str are '' for plain gen files. utf-8-sig: the engine writes a BOM."""
    out = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            t = line.strip()
            if not t or t.startswith("#"):
                continue
            parts = t.split(";")
            if len(parts) < 3:
                continue
            best = parts[3].strip() if len(parts) > 3 else ""
            good = parts[4].strip() if len(parts) > 4 else ""
            quiet = parts[5].strip() if len(parts) > 5 else ""
            out.append((parts[0].strip(), int(parts[1]), float(parts[2]), best, good, quiet))
    return out


def quiet_moves(board) -> list[str]:
    """Legal QUIET moves (engine isQuiet) via python-chess — the fallback for gen-v2 records."""
    res = []
    for mv in board.legal_moves:
        if mv.promotion is None and not board.is_capture(mv):
            res.append(mv.uci())
    return res


def build_policy_arrays(records, qmax: int = QMAX):
    """-> (keep_idx, arrays):
    qf/qt (N,qmax) i16  EONPOL02 piece-aware logit indices (pt*64 + rel square, 0-padded)
    qn    (N,)     i16  quiet count
    good  (N,qmax) bool WDL-preserving quiet mask (one-hot of best for gen-v2 rows)
    tgt   (N,)     i32  index of best_uci within the quiet list (reporting)
    wdl   (N,)     i8   STM-relative outcome class: 0=win 1=draw 2=loss"""
    keep, qf, qt, qn, good, tgt, wdl = [], [], [], [], [], [], []
    for i, (fen, _cp, result, best, good_str, quiet_str) in enumerate(records):
        if not best:
            continue
        if quiet_str:
            quiets = quiet_str.split()
            goods = set(good_str.split()) & set(quiets)
        else:
            if chess is None:
                raise RuntimeError("python-chess required for gen-v2 records without quiet lists")
            board = chess.Board(fen)
            quiets = quiet_moves(board)
            goods = {best} if best in quiets else set()
        # Discriminative rows only: at least one good quiet, at least one bad one.
        if len(quiets) < 2 or len(quiets) > qmax or not goods or len(goods) >= len(quiets):
            continue
        black = fen_black_to_move(fen)
        board = board_of_fen(fen)
        pairs = [encode_uci_piece(u, black, board) for u in quiets]
        keep.append(i)
        qf.append([p[0] for p in pairs] + [0] * (qmax - len(pairs)))
        qt.append([p[1] for p in pairs] + [0] * (qmax - len(pairs)))
        qn.append(len(pairs))
        good.append([u in goods for u in quiets] + [False] * (qmax - len(pairs)))
        tgt.append(quiets.index(best) if best in quiets else quiets.index(next(iter(goods))))
        r_stm = result if not black else 1.0 - result
        wdl.append(0 if r_stm > 0.75 else (1 if r_stm > 0.25 else 2))
    arrays = {
        "qf": np.array(qf, dtype=np.int16),
        "qt": np.array(qt, dtype=np.int16),
        "qn": np.array(qn, dtype=np.int16),
        "good": np.array(good, dtype=bool),
        "tgt": np.array(tgt, dtype=np.int32),
        "wdl": np.array(wdl, dtype=np.int8),
    }
    return np.array(keep, dtype=np.int64), arrays


_SIG_ORDER = "KQRBNP"
_SIG_VALS = {"k": 0, "q": 9, "r": 5, "b": 3, "n": 3, "p": 1}


def signature_of_fen(fen: str) -> str:
    """Canonical material signature (e.g. 'KQRvKR'): each side's pieces in K,Q,R,B,N,P order,
    stronger army (piece-value sum; lexicographic tiebreak) first. Color/STM-independent, so a
    signature-disjoint split can never leak mirrored positions across the train/holdout line."""
    w, b = [], []
    for c in fen.split()[0]:
        if c.isalpha():
            (w if c.isupper() else b).append(c.lower())

    def fmt(ps):
        return "".join(sorted((p.upper() for p in ps), key=_SIG_ORDER.index))

    sw, sb = fmt(w), fmt(b)
    vw = sum(_SIG_VALS[p] for p in w)
    vb = sum(_SIG_VALS[p] for p in b)
    return f"{sw}v{sb}" if (vw, sw) >= (vb, sb) else f"{sb}v{sw}"


def signatures_for(records, keep) -> np.ndarray:
    """Material signature per KEPT row (index-aligned with build_policy_arrays' keep)."""
    return np.array([signature_of_fen(records[i][0]) for i in keep])


def compute_a1(net_path, recs, chunk: int = 200_000):
    """Exact integer a1 activations (N, 32) u8 + buckets, through the FROZEN value stacks
    (mirror of NNUE.a1FromFt). Chunked so multi-million-row dumps stay in RAM."""
    import blend_nnue as bn

    net = bn.parse(net_path)
    stacks = [{k: net["regions"][f"s{s}.{k}"] for k, _, _ in bn.STACK_REGIONS} for s in range(bn.STACKS)]
    n = recs.size
    a1_out = np.zeros((n, 32), dtype=np.uint8)
    buckets = recs["bucket"]
    w0s = [st["fc0w"].reshape(32, L1).astype(np.float64) for st in stacks]
    w1s = [st["fc1w"].reshape(32, 64).astype(np.float64) for st in stacks]

    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        ft = recs["ft"][lo:hi].astype(np.float64)
        bks = buckets[lo:hi]
        for b in range(bn.STACKS):
            idx = np.nonzero(bks == b)[0]
            if idx.size == 0:
                continue
            st = stacks[b]
            fc0 = (ft[idx] @ w0s[b].T + st["fc0b"].astype(np.float64)).astype(np.int64)
            x = fc0[:, :31]
            conc = np.zeros((idx.size, 64), dtype=np.int64)
            conc[:, :31] = np.minimum(127, (x * x) >> 21)
            conc[:, 31:62] = np.clip(x >> 7, 0, 127)
            fc1 = (conc.astype(np.float64) @ w1s[b].T + st["fc1b"].astype(np.float64)).astype(np.int64)
            a1_out[lo + idx] = np.clip(fc1 >> 6, 0, 127).astype(np.uint8)
    return a1_out, buckets.astype(np.int64)

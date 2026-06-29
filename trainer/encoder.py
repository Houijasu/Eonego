"""Byte-faithful port of Eonego's NnueRegions + Nnue.assembleInput.

Produces the exact 2577-length feature vector the engine's `featuredump` emits.
Gate: parity_check.py asserts this is byte-identical to the engine over many FENs.

Contract (from NnueRegions.fs / NNUE.fs / Bitboard.fs):
  * Squares: LERF, a1=0, h1=7, a2=8, ... h8=63.  rank=sq>>3, file=sq&7.
  * Piece index pc = color*6 + pieceType ; White=0 Black=1 ; P=0 N=1 B=2 R=3 Q=4 K=5.
  * 204 k*k regions (k=1..8), grouped by k then row-major by top-left (rank,file).
    regionIndex k tr tf = offset[k] + tr*(9-k) + tf ; offset[k]=sum_{j<k}(9-j)^2.
    Accumulator index = region*12 + pc ; each piece +1 in every region containing its square.
  * Feature vector (length 2577): [0,2448) region counts ; [2448] STM (0/1) ;
    [2449,2513) white-king one-hot ; [2513,2577) black-king one-hot.
    (The king is counted BOTH in the region accumulator AND in the one-hot.)
"""

import numpy as np

REGION_COUNT = 204
CHANNELS = 12
ACC_SIZE = 2448           # REGION_COUNT * CHANNELS
INPUT_SIZE = 2577         # ACC_SIZE + 1 STM + 128 king one-hot
PADDED_L1 = 2592
STM_INDEX = ACC_SIZE      # 2448
WKING_BASE = STM_INDEX + 1     # 2449
BKING_BASE = WKING_BASE + 64   # 2513

_PIECE = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
          'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11}


def _build_region_offsets():
    off = [0] * 10
    off[1] = 0
    for k in range(2, 10):
        side = 9 - (k - 1)
        off[k] = off[k - 1] + side * side
    assert off[9] == REGION_COUNT, off[9]
    return off


_REGION_OFFSET = _build_region_offsets()


def _region_index(k, top_rank, top_file):
    return _REGION_OFFSET[k] + top_rank * (9 - k) + top_file


def _build_regions_for_square():
    rfs = [None] * 64
    for sq in range(64):
        r = sq >> 3
        f = sq & 7
        acc = []
        for k in range(1, 9):
            tr_lo = max(0, r - k + 1)
            tr_hi = min(r, 8 - k)
            tf_lo = max(0, f - k + 1)
            tf_hi = min(f, 8 - k)
            for tr in range(tr_lo, tr_hi + 1):
                for tf in range(tf_lo, tf_hi + 1):
                    acc.append(_region_index(k, tr, tf))
        rfs[sq] = np.array(acc, dtype=np.int64)
    return rfs


# regionsForSquare.[sq] -> int64[] of region indices containing sq
_REGIONS_FOR_SQUARE = _build_regions_for_square()
# Precomputed channel-base (region*12) per square, for fast scatter-add.
_CHAN_BASE_FOR_SQUARE = [r * CHANNELS for r in _REGIONS_FOR_SQUARE]


def parse_fen(fen):
    """Return (board[64] piece 0..11 or -1, stm 0/1, wking_sq, bking_sq)."""
    parts = fen.split()
    placement = parts[0]
    stm = 0 if parts[1] == 'w' else 1
    board = [-1] * 64
    rank = 7
    file = 0
    for ch in placement:
        if ch == '/':
            rank -= 1
            file = 0
        elif ch.isdigit():
            file += int(ch)
        else:
            board[rank * 8 + file] = _PIECE[ch]
            file += 1
    return board, stm, board.index(5), board.index(11)


def encode_dense(fen):
    """uint8 array length INPUT_SIZE (engine pads to PADDED_L1 with zeros)."""
    v = np.zeros(INPUT_SIZE, dtype=np.int32)
    board, stm, wking, bking = parse_fen(fen)
    parts = []
    for sq in range(64):
        pc = board[sq]
        if pc >= 0:
            parts.append(_CHAN_BASE_FOR_SQUARE[sq] + pc)  # region*12 + pc indices for this piece
    if parts:
        all_idx = np.concatenate(parts)
        # bincount is robust (no fancy-index broadcast); region*12+pc is always < ACC_SIZE
        v[:ACC_SIZE] = np.bincount(all_idx, minlength=ACC_SIZE)[:ACC_SIZE]
    v[STM_INDEX] = stm
    v[WKING_BASE + wking] = 1
    v[BKING_BASE + bking] = 1
    return v.astype(np.uint8)


def encode_sparse(fen):
    """dict idx->value over nonzero entries in [0,INPUT_SIZE)."""
    v = encode_dense(fen)
    nz = np.nonzero(v)[0]
    return {int(i): int(v[i]) for i in nz}


def encode_batch_sparse(fens):
    """Return (idx_lists, val_lists) parallel to fens; for sparse-L1 training."""
    idxs, vals = [], []
    for fen in fens:
        s = encode_sparse(fen)
        keys = np.fromiter(s.keys(), dtype=np.int64)
        order = np.argsort(keys)
        keys = keys[order]
        vs = np.fromiter((s[int(k)] for k in keys), dtype=np.int64)
        idxs.append(keys)
        vals.append(vs)
    return idxs, vals

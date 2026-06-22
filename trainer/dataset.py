"""Parse gen text records and encode to dense feature matrices (with a file cache)."""

import hashlib
import os

import numpy as np

from encoder import INPUT_SIZE, encode_dense


def parse_records(path):
    """Read `<fen>;<cp_white>;<result_white>` lines -> (fens, cp int32[N], result f32[N])."""
    fens, cps, res = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(";")
            if len(parts) != 3:
                continue
            fens.append(parts[0])
            cps.append(int(parts[1]))
            res.append(float(parts[2]))
    return fens, np.array(cps, dtype=np.int32), np.array(res, dtype=np.float32)


def _sig(fens):
    h = hashlib.sha1()
    h.update(str(len(fens)).encode())
    if fens:
        h.update(fens[0].encode())
        h.update(fens[-1].encode())
    return h.hexdigest()


def encode_dataset(fens, cache=None):
    """Encode fens -> uint8 (N, INPUT_SIZE). Cached as <cache>.npz keyed on a content signature."""
    sig = _sig(fens)
    if cache and os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        if str(d["sig"]) == sig and d["X"].shape == (len(fens), INPUT_SIZE):
            return d["X"]
    X = np.zeros((len(fens), INPUT_SIZE), dtype=np.uint8)
    for i, fen in enumerate(fens):
        X[i] = encode_dense(fen)
    if cache:
        np.savez_compressed(cache, X=X, sig=sig)
    return X

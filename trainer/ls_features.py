"""Parse an lstrace TSV into (X, leaf_eval, root_impact) numpy arrays.

The engine is the single source of feature truth: lstrace dumps the already-quantised feature row, so we
do NOT recompute features here (no FEN re-encoding). Row format (tab-separated):
    rootFen  movepath  f0 .. f{NF-1}  leafEval  rootImpact
`movepath` may be empty (a field of "") for shallow nodes; columns are positional regardless.
"""

import numpy as np
from ls_intref import NF


def load_trace(path):
    X, leaf, impact = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n").rstrip("\r")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2 + NF + 2:  # rootFen, movepath, NF feats, leafEval, rootImpact
                continue
            try:
                x = [int(parts[2 + i]) for i in range(NF)]
                le = int(parts[2 + NF])
                ri = int(parts[2 + NF + 1])
            except ValueError:
                continue
            X.append(x)
            leaf.append(le)
            impact.append(ri)
    return (
        np.array(X, dtype=np.float32),
        np.array(leaf, dtype=np.float32),
        np.array(impact, dtype=np.float32),
    )


if __name__ == "__main__":
    import sys

    X, leaf, impact = load_trace(sys.argv[1])
    print("rows", len(X), "NF", X.shape[1] if len(X) else 0)
    if len(X):
        print("feat mean", np.round(X.mean(0), 1))
        print("|impact| mean/max", float(np.abs(impact).mean()), float(np.abs(impact).max()))

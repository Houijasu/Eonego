"""Phase-1 parity gate: assert the Python encoder == the engine's featuredump, byte for byte.

Usage:
    python parity_check.py [fens_file]
Exits 0 if every FEN matches, 1 otherwise. Pin this as a CI check — any drift here
makes every trained net garbage on the engine.
"""

import os
import subprocess
import sys

from encoder import encode_sparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
EXE = os.path.join(REPO, "Eonego", "bin", "Release", "net10.0", "Eonego.exe")
DEFAULT_FENS = os.path.join(HERE, "data", "parity_fens.txt")


def read_fens(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t and not t.startswith("#"):
                out.append(t)
    return out


def engine_featuredump(fens_path):
    """Map fen -> {idx: val} from the engine's sparse dump."""
    res = {}
    proc = subprocess.run([EXE, "featuredump", "--in", fens_path],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"featuredump exited {proc.returncode}")
    for line in proc.stdout.splitlines():
        line = line.rstrip("\r\n")
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        fen = parts[0]
        sparse = {}
        for tok in parts[2:]:  # parts[1] is "n=<count>"
            if not tok:
                continue
            idx, val = tok.split(":")
            sparse[int(idx)] = int(val)
        res[fen] = sparse
    return res


def main():
    fens_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FENS
    fens = read_fens(fens_path)
    engine = engine_featuredump(fens_path)

    mismatches = 0
    for fen in fens:
        py = encode_sparse(fen)
        eng = engine.get(fen)
        if eng is None:
            print("MISSING in engine output:", fen)
            mismatches += 1
            continue
        if py != eng:
            mismatches += 1
            diff = sorted(set(py) ^ set(eng) | {k for k in (set(py) & set(eng)) if py[k] != eng[k]})
            print("MISMATCH", fen, "(", len(diff), "differing indices )")
            for k in diff[:12]:
                print(f"    idx {k}: py={py.get(k)} eng={eng.get(k)}")

    ok = len(fens) - mismatches
    print(f"PARITY: {ok}/{len(fens)} FENs byte-identical")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())

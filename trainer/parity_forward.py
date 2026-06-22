"""Phase-4 round-trip parity gate: intref(exported net) == engine nnforward, exactly.

Both sides are pure-integer on the SAME exported weights, so they must agree bit-for-bit
(the white-relative raw forward, before sign/clamp). A net that fails this is not shipped.

    python parity_forward.py <net.eongnnue> [fens_file]
"""

import os
import subprocess
import sys

import numpy as np

import intref
from encoder import encode_dense

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


def engine_nnforward(net_path, fens_path):
    proc = subprocess.run([EXE, "nnforward", "--net", net_path, "--in", fens_path],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"nnforward exited {proc.returncode}")
    res = {}
    for line in proc.stdout.splitlines():
        line = line.rstrip("\r\n")
        if not line or line.startswith("#"):
            continue
        fen, cp = line.rsplit("\t", 1)
        res[fen] = int(cp)
    return res


def main():
    net_path = sys.argv[1]
    fens_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_FENS
    net = intref.load(net_path)
    fens = read_fens(fens_path)
    eng = engine_nnforward(net_path, fens_path)

    X = np.stack([encode_dense(f) for f in fens])
    py = intref.int_forward(net, X)

    mism = 0
    for i, f in enumerate(fens):
        if f not in eng:
            print("MISSING in engine:", f)
            mism += 1
            continue
        if int(py[i]) != eng[f]:
            mism += 1
            print(f"MISMATCH {f}  py={int(py[i])} eng={eng[f]}")
    ok = len(fens) - mism
    print(f"FORWARD PARITY: {ok}/{len(fens)} exact (shifts "
          f"{net['Shift1']},{net['Shift2']},{net['Shift3']},{net['Shift4']} qs {net['QuantScale']})")
    return 1 if mism else 0


if __name__ == "__main__":
    raise SystemExit(main())

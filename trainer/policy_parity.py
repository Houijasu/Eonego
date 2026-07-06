"""STDLIB-ONLY policy parity gate: engine `dumppolicy` output == Python integer reference forward,
bit-exact, over a FEN set that MUST include Black-to-move and promotion-heavy positions (STM
mirroring is the classic silent engine/trainer skew). Exit 1 on any mismatch.

    python policy_parity.py --exe <Eonego.exe> --net ../nets/main.nnue \
        --policy ../nets/main.policy --fens data/parity_fens.txt [--tmp data]

Flow: engine dumpft (FT buffers, the head's exact input) + engine dumppolicy (full engine path),
then intref forward from the dumped FT — 128 ints per position compared exactly.
"""

import argparse
import os
import struct
import subprocess
import sys

import policy_intref as ref

REC = struct.Struct("<BBii1024s")  # bucket, stm, psqt, eval, ft — 1034 bytes


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"command failed ({r.returncode}): {' '.join(cmd)}\n{r.stdout}\n{r.stderr}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--net", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--fens", required=True)
    ap.add_argument("--tmp", default=".")
    args = ap.parse_args()

    ft_bin = os.path.join(args.tmp, "parity_ft.bin")
    pol_txt = os.path.join(args.tmp, "parity_pol.txt")
    run([args.exe, "dumpft", "--net", args.net, "--in", args.fens, "--out", ft_bin])
    run([args.exe, "dumppolicy", "--net", args.net, "--policy", args.policy, "--in", args.fens, "--out", pol_txt])

    pol = ref.load(args.policy)
    with open(ft_bin, "rb") as f:
        dump = f.read()
    engine_rows = []
    with open(pol_txt, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                engine_rows.append((parts[0], [int(x) for x in parts[1].split()], [int(x) for x in parts[2].split()]))

    n = len(dump) // REC.size
    assert n == len(engine_rows), f"row mismatch: dumpft {n} vs dumppolicy {len(engine_rows)}"

    mismatches = 0
    for i in range(n):
        _b, _stm, _psqt, _ev, ft_bytes = REC.unpack_from(dump, i * REC.size)
        rf, rt = ref.forward(pol, list(ft_bytes))
        fen, ef, et = engine_rows[i]
        if rf != ef or rt != et:
            mismatches += 1
            if mismatches <= 5:
                print(f"MISMATCH row {i}: {fen}")

    if mismatches:
        print(f"PARITY FAILED ({n} rows, {mismatches} mismatch)")
        sys.exit(1)
    print(f"PARITY OK ({n} rows, 0 mismatch)")


if __name__ == "__main__":
    main()

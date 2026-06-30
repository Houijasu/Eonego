"""Engine<->Python inference parity gate for the EONGLS priority net.

Generates random feature rows, runs them through BOTH the engine `lsforward` command and the Python
int_forward reference, and asserts every output matches bit-for-bit. This is the one parity surface that
matters (features are engine-dumped, so there is no feature-extraction parity to check).

Usage:  python parity_lsforward.py <net.lsnet> <engine-cmd...>
  e.g.  python parity_lsforward.py phase_b.lsnet dotnet ../Eonego/bin/Release/net10.0/Eonego.dll
"""

import os
import random
import subprocess
import sys
import tempfile

from ls_intref import NF, int_forward, load


def main():
    if len(sys.argv) < 3:
        print("usage: python parity_lsforward.py <net.lsnet> <engine-cmd...>")
        sys.exit(2)
    lsnet = sys.argv[1]
    engine_cmd = sys.argv[2:]

    n = 256
    rng = random.Random(20260629)
    rows = [[rng.randint(-127, 127) for _ in range(NF)] for _ in range(n)]

    fd, rowpath = tempfile.mkstemp(suffix=".tsv")
    os.close(fd)
    outpath = rowpath + ".out"
    with open(rowpath, "w", encoding="utf-8") as f:
        for r in rows:
            f.write("\t".join(str(int(v)) for v in r) + "\n")

    subprocess.run(engine_cmd + ["lsforward", "--lsnet", lsnet, "--in", rowpath, "--out", outpath],
                   check=True)
    eng = [int(x) for x in open(outpath, encoding="utf-8").read().split()]

    net = load(lsnet)
    ref = [int_forward(net, [int(v) for v in r]) for r in rows]

    os.unlink(rowpath)
    os.unlink(outpath)

    if len(eng) != len(ref):
        print(f"FAIL: engine produced {len(eng)} rows, reference {len(ref)}")
        sys.exit(1)
    mism = [(i, a, b) for i, (a, b) in enumerate(zip(eng, ref)) if a != b]
    print(f"rows={n} mismatches={len(mism)}")
    if mism:
        for i, a, b in mism[:5]:
            print(f"  row {i}: engine={a} ref={b}")
        sys.exit(1)
    print("PARITY OK")


if __name__ == "__main__":
    main()

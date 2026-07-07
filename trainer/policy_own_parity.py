"""Own-trunk parity: engine `dumppolicyown` (EONPOL03 F# forward) vs the torch checkpoint, on a set
of ≤6-piece FENs. Float nets aren't bit-exact across implementations, so this checks that the
argmax from-move and to-square AGREE and that logits correlate tightly — the point is to catch a
feature-extraction / index mismatch, not floating-point noise.

    uv run --python 3.12 --with torch,numpy python policy_own_parity.py \
        --exe ../scratch-build/pol1/Eonego.exe --ckpt data/pol1/own_deep.pt \
        --policy ../nets/main.ownpolicy --fens data/pol1/eg_fens.txt
"""

import argparse
import subprocess
import sys

import torch

from move_encoder import PIECE_TYPE

SCALE = 256


def features(fen):
    board = [" "] * 64
    ranks = fen.split()[0].split("/")
    for r, row in enumerate(ranks):
        f = 0
        for c in row:
            if c.isdigit():
                f += int(c)
            else:
                board[(7 - r) * 8 + f] = c
                f += 1
    black = fen.split()[1] == "b"
    feats = []
    for sq in range(64):
        c = board[sq]
        if c != " ":
            own = c.isupper() != black
            plane = (0 if own else 6) + PIECE_TYPE[c.lower()]
            rel = sq ^ 56 if black else sq
            feats.append(plane * 64 + rel)
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--fens", required=True)
    ap.add_argument("--tmp", default="data/pol1")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck["model"]
    ftw, ftb = sd["ft.weight"], sd["ft_b"]
    mw, mb = sd["mid.weight"], sd["mid.bias"]
    m2w, m2b = sd["mid2.weight"], sd["mid2.bias"]
    hfw, hfb = sd["hf.weight"], sd["hf.bias"]
    htw, htb = sd["ht.weight"], sd["ht.bias"]

    def torch_forward(fen):
        acc = ftb.clone()
        for f in features(fen):
            acc += ftw[f]
        x = torch.relu(acc)
        x = torch.relu(mw @ x + mb)
        x = torch.relu(m2w @ x + m2b)
        return (hfw @ x + hfb), (htw @ x + htb)

    out = f"{args.tmp}/own_parity.txt"
    r = subprocess.run(
        [args.exe, "dumppolicyown", "--policy", args.policy, "--in", args.fens, "--out", out],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        sys.exit(f"dumppolicyown failed: {r.stdout}\n{r.stderr}")

    eng = {}
    with open(out, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            eng[p[0]] = ([int(x) for x in p[1].split()], [int(x) for x in p[2].split()])

    n = argmax_from_ok = argmax_to_ok = 0
    max_abs = 0
    for fen in eng:
        ef, et = eng[fen]
        tf, tt = torch_forward(fen)
        tf = [int(v * SCALE) for v in tf.tolist()]
        tt = [int(v * SCALE) for v in tt.tolist()]
        n += 1
        if max(range(384), key=lambda i: ef[i]) == max(range(384), key=lambda i: tf[i]):
            argmax_from_ok += 1
        if max(range(384), key=lambda i: et[i]) == max(range(384), key=lambda i: tt[i]):
            argmax_to_ok += 1
        max_abs = max(max_abs, max(abs(a - b) for a, b in zip(ef, tf)),
                      max(abs(a - b) for a, b in zip(et, tt)))

    print(f"positions: {n}")
    print(f"argmax-from agree: {argmax_from_ok}/{n}   argmax-to agree: {argmax_to_ok}/{n}")
    print(f"max |engine-torch| logit diff (scaled x{SCALE}): {max_abs}")
    if argmax_from_ok == n and argmax_to_ok == n and max_abs <= 4:
        print("PARITY OK")
    else:
        print("PARITY MISMATCH — check feature extraction / indexing")
        sys.exit(1)


if __name__ == "__main__":
    main()

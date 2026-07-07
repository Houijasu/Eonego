"""Export an own-trunk policy checkpoint (policy_own.py) to the engine's EONPOL03 float32 format.

    uv run --python 3.12 --with torch,numpy python policy_own_export.py \
        --ckpt data/pol1/own_deep.pt --out ../nets/main.ownpolicy [--scale 256]

EONPOL03 (little-endian): magic "EONPOL03", version 3, then u32 trunk, mid, mid2, nfeat(769),
scale, flags(0); then float32 arrays in forward order:
  ft_w[769*trunk] ft_b[trunk] mid_w[mid*trunk] mid_b[mid] mid2_w[mid2*mid] mid2_b[mid2]
  hf_w[384*mid2] hf_b[384] ht_w[384*mid2] ht_b[384]
The engine reads ≤6 active board features, forwards float32, scales logits by `scale` to int.
Only the 768-feature (no king-pair) checkpoints are supported here (layers==3).
"""

import argparse
import struct

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=int, default=256, help="float logit -> int multiplier in the engine")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ck["args"]
    assert a["layers"] == 3, "EONPOL03 export supports the 3-layer 768-feature trunk only"
    sd = ck["model"]
    trunk, mid, mid2 = a["trunk"], a["hidden2"], a["hidden3"]
    nfeat = sd["ft.weight"].shape[0]  # 769 (768 features + padding row)
    assert sd["ft.weight"].shape[1] == trunk
    assert sd["hf.weight"].shape == (384, mid2)

    def f32(name):
        return sd[name].detach().cpu().numpy().astype("<f4").ravel()

    out = bytearray()
    out += b"EONPOL03"
    out += struct.pack("<iIIIII", 3, trunk, mid, mid2, nfeat, args.scale)
    out += struct.pack("<I", 0)  # flags
    for name in ("ft.weight", "ft_b", "mid.weight", "mid.bias", "mid2.weight", "mid2.bias",
                 "hf.weight", "hf.bias", "ht.weight", "ht.bias"):
        out += f32(name).tobytes()

    with open(args.out, "wb") as f:
        f.write(out)
    print(f"wrote {args.out} ({len(out)} bytes; trunk={trunk} mid={mid} mid2={mid2} scale={args.scale})")


if __name__ == "__main__":
    main()

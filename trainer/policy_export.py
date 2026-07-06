"""Serialize a policy_train.py .npz into the EONPOL02 sidecar the engine loads (Eonego/Policy.fs).

    uv run --with numpy python policy_export.py --npz data/policy_head.npz \
        --net ../nets/main.nnue --out ../nets/main.policy

ftHash comes from the .nnue header (the loader refuses a head whose ftHash differs from the
loaded trunk). Format doc lives in Policy.fs; must end exactly at EOF.
"""

import argparse
import struct

import numpy as np

HEAD_OUT = 384
L1 = 1024
STACKS = 8
FC2IN = 32


def nnue_ft_hash(path: str) -> int:
    """.nnue header: version i32, hash u32, descLen i32, desc, ftHash u32."""
    with open(path, "rb") as f:
        head = f.read(64 * 1024)
    desc_len = struct.unpack_from("<i", head, 8)[0]
    return struct.unpack_from("<I", head, 12 + desc_len)[0]


def export(npz_path: str, ft_hash: int, out_path: str) -> int:
    d = np.load(npz_path)
    hidden = int(d["hidden"])
    assert 32 <= hidden <= 1024 and hidden % 32 == 0, f"unsupported hidden {hidden}"
    has_wdl = "wdl_w" in d
    out = bytearray()
    out += b"EONPOL02"
    out += struct.pack(
        "<iIiiii", 2, ft_hash, hidden, int(d["shift0"]), int(d["wdl_shift"]), 1 if has_wdl else 0
    )

    def arr(name, dtype, shape):
        a = np.ascontiguousarray(d[name], dtype=dtype)
        assert a.shape == shape, f"{name}: {a.shape} != {shape}"
        return a.tobytes()

    out += arr("b0", "<i4", (hidden,))
    out += arr("w0", "i1", (hidden, L1))
    out += arr("bf", "<i4", (HEAD_OUT,))
    out += arr("wf", "i1", (HEAD_OUT, hidden))
    out += arr("bt", "<i4", (HEAD_OUT,))
    out += arr("wt", "i1", (HEAD_OUT, hidden))
    if has_wdl:
        out += arr("wdl_b", "<i4", (STACKS, 3))
        out += arr("wdl_w", "i1", (STACKS, 3, FC2IN))
    with open(out_path, "wb") as f:
        f.write(out)
    return len(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--net", required=True, help=".nnue whose trunk this head was trained on")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ft_hash = nnue_ft_hash(args.net)
    n = export(args.npz, ft_hash, args.out)
    print(f"wrote {args.out} ({n} bytes, ftHash=0x{ft_hash:08X})")


if __name__ == "__main__":
    main()

"""STDLIB-ONLY integer reference forward of the EONPOL01 sidecar — the parity oracle
(ls_intref precedent). Must mirror Eonego/Policy.fs exactly:

  hid[o]  = clamp((b0[o] + sum_j w0[o][j]*ft[j]) >> shift0, 0, 127)   # arithmetic shift = floor
  from[o] = bf[o] + sum_j wf[o][j]*hid[j]
  to[o]   = bt[o] + sum_j wt[o][j]*hid[j]
  wdl[k]  = wdl_b[bucket][k] + sum_j wdl_w[bucket][k][j]*a1[j]        # softmax by the engine

Python's >> on negative ints floors, exactly like F#'s arithmetic >>> on int32.
"""

import struct

HIDDEN = 128
L1 = 1024
STACKS = 8
FC2IN = 32


def parse(buf: bytes) -> dict:
    assert buf[:8] == b"EONPOL01", "bad magic"
    version, ft_hash, hidden, shift0, wdl_shift, flags = struct.unpack_from("<iIiiii", buf, 8)
    assert version == 1 and hidden == HIDDEN
    p = 32

    def i32s(n):
        nonlocal p
        v = list(struct.unpack_from(f"<{n}i", buf, p))
        p += 4 * n
        return v

    def i8s(n):
        nonlocal p
        v = list(struct.unpack_from(f"{n}b", buf, p))
        p += n
        return v

    d = {"ft_hash": ft_hash, "shift0": shift0, "wdl_shift": wdl_shift, "has_wdl": bool(flags & 1)}
    d["b0"] = i32s(HIDDEN)
    d["w0"] = i8s(HIDDEN * L1)
    d["bf"] = i32s(64)
    d["wf"] = i8s(64 * HIDDEN)
    d["bt"] = i32s(64)
    d["wt"] = i8s(64 * HIDDEN)
    if d["has_wdl"]:
        d["wdl_b"] = i32s(STACKS * 3)
        d["wdl_w"] = i8s(STACKS * 3 * FC2IN)
    assert p == len(buf), f"trailing bytes: {len(buf) - p}"
    return d


def load(path: str) -> dict:
    with open(path, "rb") as f:
        return parse(f.read())


def forward(pol: dict, ft) -> tuple[list[int], list[int]]:
    """ft: 1024 ints in [0,127] -> (from_logits[64], to_logits[64])."""
    shift0 = pol["shift0"]
    hid = [0] * HIDDEN
    w0, b0 = pol["w0"], pol["b0"]
    for o in range(HIDDEN):
        base = o * L1
        s = b0[o]
        for j in range(L1):
            v = ft[j]
            if v:
                s += w0[base + j] * v
        hid[o] = min(127, max(0, s >> shift0))

    def head(w, b):
        out = [0] * 64
        for o in range(64):
            base = o * HIDDEN
            s = b[o]
            for j in range(HIDDEN):
                v = hid[j]
                if v:
                    s += w[base + j] * v
            out[o] = s
        return out

    return head(pol["wf"], pol["bf"]), head(pol["wt"], pol["bt"])


def wdl_logits(pol: dict, a1, bucket: int) -> list[int]:
    assert pol["has_wdl"]
    out = [0] * 3
    for k in range(3):
        idx = bucket * 3 + k
        s = pol["wdl_b"][idx]
        base = idx * FC2IN
        for j in range(FC2IN):
            s += pol["wdl_w"][base + j] * a1[j]
        out[k] = s
    return out

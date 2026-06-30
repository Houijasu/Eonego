"""EONGLS v1 (de)serialization + the integer reference forward pass.

This is bit-for-bit identical to the engine (Eonego/LearnedSearch.fs `forwardInto`/`init`) and is the
ONLY engine<->Python parity surface: features are computed and dumped by the engine (lstrace), so Python
never re-derives them. parity_lsforward.py asserts engine `lsforward` == int_forward here on identical rows.

Format (all little-endian):
  magic "EONGLS01" (8 bytes)
  int32 x8: version=1, NF, H1, H2, OutDim(=1), Shift1, Shift2, OutShift
  int8[NF*H1]  W1 (row-major: W1[j*NF + i])   int32[H1] B1
  int8[H1*H2]  W2 (W2[j*H1 + i])              int32[H2] B2
  int8[H2*1]   W3 (W3[i])                      int32[1]  B3
Forward: a1 = clip((x.W1 + B1 + round)>>S1, 0,127); a2 likewise; out = (a2.W3 + B3 + round)>>Sout (no clip).
"""

import struct

NF = 11
MAGIC = b"EONGLS01"

# NOTE: this module is deliberately stdlib-only (no numpy/torch) so the parity gate runs anywhere.
# ls_export builds an LsNet from a trained model (numpy there) and calls serialize() here.


class LsNet:
    def __init__(self, nf, h1, h2, s1, s2, so, W1, b1, W2, b2, W3, b3):
        self.NF, self.H1, self.H2, self.OutDim = nf, h1, h2, 1
        self.Shift1, self.Shift2, self.OutShift = s1, s2, so
        self.W1, self.b1 = W1, b1
        self.W2, self.b2 = W2, b2
        self.W3, self.b3 = W3, b3


def _rsh(acc, s):
    # Arithmetic shift-right with rounding. Python `>>` on negatives floors (arithmetic), matching F# `>>>`.
    return (acc + (1 << (s - 1))) >> s if s > 0 else acc


def _clip127(v):
    return 0 if v < 0 else (127 if v > 127 else v)


def int_forward(net, x):
    """x: sequence of NF ints -> the engine's priority output (signed int)."""
    h1 = [0] * net.H1
    for j in range(net.H1):
        acc = int(net.b1[j])
        base = j * net.NF
        for i in range(net.NF):
            acc += int(x[i]) * int(net.W1[base + i])
        h1[j] = _clip127(_rsh(acc, net.Shift1))

    h2 = [0] * net.H2
    for j in range(net.H2):
        acc = int(net.b2[j])
        base = j * net.H1
        for i in range(net.H1):
            acc += h1[i] * int(net.W2[base + i])
        h2[j] = _clip127(_rsh(acc, net.Shift2))

    acc = int(net.b3[0])
    for i in range(net.H2):
        acc += h2[i] * int(net.W3[i])
    return _rsh(acc, net.OutShift)


def serialize(net):
    out = bytearray()
    out += MAGIC
    out += struct.pack("<8i", 1, net.NF, net.H1, net.H2, net.OutDim,
                       net.Shift1, net.Shift2, net.OutShift)
    out += bytes((int(w) & 0xFF) for w in net.W1)
    out += struct.pack("<%di" % net.H1, *[int(b) for b in net.b1])
    out += bytes((int(w) & 0xFF) for w in net.W2)
    out += struct.pack("<%di" % net.H2, *[int(b) for b in net.b2])
    out += bytes((int(w) & 0xFF) for w in net.W3)
    out += struct.pack("<i", int(net.b3[0]))
    return bytes(out)


def load(path):
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != MAGIC:
        raise ValueError("bad EONGLS magic")
    version, nf, h1, h2, outdim, s1, s2, so = struct.unpack_from("<8i", data, 8)
    if version != 1 or outdim != 1:
        raise ValueError("unexpected EONGLS version/outdim")
    o = 40

    def i8(n):  # signed bytes
        nonlocal o
        a = list(struct.unpack_from("<%db" % n, data, o))
        o += n
        return a

    def i32(n):
        nonlocal o
        a = list(struct.unpack_from("<%di" % n, data, o))
        o += n * 4
        return a

    W1 = i8(nf * h1)
    b1 = i32(h1)
    W2 = i8(h1 * h2)
    b2 = i32(h2)
    W3 = i8(h2 * outdim)
    b3 = i32(outdim)
    return LsNet(nf, h1, h2, s1, s2, so, W1, b1, W2, b2, W3, b3)

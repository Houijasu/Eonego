"""Integer reference forward pass + EONGNNUE v2 (de)serialization.

Mirrors NnueNetwork.forwardWith / NnueTestFixtures.refForward EXACTLY so the
exported file's Python forward == the engine's `nnforward` output, bit for bit.

v2 quant contract (NnueNetwork.fs):
  * int8 weights L1..L4, int16 L5, int32 biases, int32 accumulate.
  * per-layer activation: round = (1<<(shift-1)) if shift>0 else 0
        out = clip( (acc + round) >> shift , 0, 127 )   # arithmetic >>
  * L5: raw int32 / quantScale  (TRUNCATE toward zero, like F#/C# int division)
  * output = WHITE-RELATIVE centipawns (caller applies sign + clamp).
"""

import struct

import numpy as np

MAGIC = b"EONGNNUE"
VERSION = 2
INPUT_SIZE = 2577
L1, L2, L3, L4, OUT = 64, 32, 16, 16, 1


def shift_clip(acc, shift):
    """(acc + round) >> shift, clipped to [0,127]. acc: int64 ndarray. Arithmetic shift."""
    rnd = (1 << (shift - 1)) if shift > 0 else 0
    v = np.right_shift(acc + rnd, shift)  # numpy signed right-shift is arithmetic
    return np.clip(v, 0, 127)


def trunc_div(acc, q):
    """Integer division truncating toward zero (F#/C# semantics)."""
    a = np.asarray(acc, dtype=np.int64)
    out = np.abs(a) // q
    return np.where(a < 0, -out, out)


def int_forward(net, X):
    """X: (N, INPUT_SIZE) of int (0..127). Returns (N,) int64 white-relative cp.

    `net` is a dict with UNPADDED weights:
      L1W (64,2577) L1B (64,) L2W (32,64) L2B (32,) L3W (16,32) L3B (16,)
      L4W (16,16) L4B (16,) L5W (1,16) L5B (1,) QuantScale int, Shift1..4 ints.
    """
    X = np.asarray(X, dtype=np.int64)
    a1 = X @ net["L1W"].astype(np.int64).T + net["L1B"].astype(np.int64)
    l1 = shift_clip(a1, net["Shift1"])
    a2 = l1 @ net["L2W"].astype(np.int64).T + net["L2B"].astype(np.int64)
    l2 = shift_clip(a2, net["Shift2"])
    a3 = l2 @ net["L3W"].astype(np.int64).T + net["L3B"].astype(np.int64)
    l3 = shift_clip(a3, net["Shift3"])
    a4 = l3 @ net["L4W"].astype(np.int64).T + net["L4B"].astype(np.int64)
    l4 = shift_clip(a4, net["Shift4"])
    a5 = l4 @ net["L5W"].astype(np.int64).T + net["L5B"].astype(np.int64)  # (N,1)
    return trunc_div(a5[:, 0], net["QuantScale"])


# --- EONGNNUE v2 byte (de)serialization --------------------------------------

def serialize(net):
    """dict -> EONGNNUE v2 bytes (inverse of NnueNetwork.loadBytes)."""
    buf = bytearray()
    buf += MAGIC
    buf += struct.pack("<7I", VERSION, INPUT_SIZE, L1, L2, L3, L4, OUT)
    buf += struct.pack("<i", int(net["QuantScale"]))
    buf += struct.pack("<4i", int(net["Shift1"]), int(net["Shift2"]),
                       int(net["Shift3"]), int(net["Shift4"]))

    def w8(a):
        buf.extend(np.asarray(a, dtype=np.int8).tobytes())

    def w32(a):
        buf.extend(np.asarray(a, dtype="<i4").tobytes())

    def w16(a):
        buf.extend(np.asarray(a, dtype="<i2").tobytes())

    w8(net["L1W"].reshape(-1)); w32(net["L1B"])
    w8(net["L2W"].reshape(-1)); w32(net["L2B"])
    w8(net["L3W"].reshape(-1)); w32(net["L3B"])
    w8(net["L4W"].reshape(-1)); w32(net["L4B"])
    w16(net["L5W"].reshape(-1)); w32(net["L5B"])
    return bytes(buf)


def loadbytes(data):
    """EONGNNUE v2 bytes -> dict (validates header; raises on mismatch)."""
    off = 0
    if data[:8] != MAGIC:
        raise ValueError("bad magic")
    off = 8
    ver, inp, l1, l2, l3, l4, out = struct.unpack_from("<7I", data, off); off += 28
    (qs,) = struct.unpack_from("<i", data, off); off += 4
    s1, s2, s3, s4 = struct.unpack_from("<4i", data, off); off += 16
    if (ver, inp, l1, l2, l3, l4, out) != (VERSION, INPUT_SIZE, L1, L2, L3, L4, OUT):
        raise ValueError(f"header mismatch: {(ver, inp, l1, l2, l3, l4, out)}")

    def r8(n):
        nonlocal off
        a = np.frombuffer(data, dtype=np.int8, count=n, offset=off).copy()
        off += n
        return a

    def r32(n):
        nonlocal off
        a = np.frombuffer(data, dtype="<i4", count=n, offset=off).astype(np.int32)
        off += n * 4
        return a

    def r16(n):
        nonlocal off
        a = np.frombuffer(data, dtype="<i2", count=n, offset=off).astype(np.int16)
        off += n * 2
        return a

    net = {"QuantScale": qs, "Shift1": s1, "Shift2": s2, "Shift3": s3, "Shift4": s4}
    net["L1W"] = r8(L1 * INPUT_SIZE).reshape(L1, INPUT_SIZE); net["L1B"] = r32(L1)
    net["L2W"] = r8(L2 * L1).reshape(L2, L1); net["L2B"] = r32(L2)
    net["L3W"] = r8(L3 * L2).reshape(L3, L2); net["L3B"] = r32(L3)
    net["L4W"] = r8(L4 * L3).reshape(L4, L3); net["L4B"] = r32(L4)
    net["L5W"] = r16(OUT * L4).reshape(OUT, L4); net["L5B"] = r32(OUT)
    return net


def load(path):
    with open(path, "rb") as f:
        return loadbytes(f.read())

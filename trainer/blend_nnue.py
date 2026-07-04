"""Blend (weight-average) same-architecture FullThreats NNUE nets — a "model soup".

Only meaningful for nets from the same training lineage (fishtest candidates are fine-tunes of the
current master, which is the linearly-mode-connected case where averaging helps); independently
trained nets average to garbage (hidden-neuron permutation). All inputs must share the version word,
architecture hash, FT hash and every per-stack hash — asserted, not assumed.

Layout mirrored from Eonego/NNUE.fs loadBytes (the loader IS the spec):
  u32 version (0x6A448AFA) | u32 archHash | u32 descLen | desc bytes
  u32 ftHash
  i16[L1]                    FT biases            (LEB128-or-raw)
  i8 [ThreatDims*L1]         threat FT weights    (LEB128-or-raw)
  i32[ThreatDims*Psqt]       threat PSQT          (LEB128-or-raw)
  i16[HalfKaDims*L1]         HalfKA FT weights    (LEB128-or-raw)
  i32[HalfKaDims*Psqt]       HalfKA PSQT          (LEB128-or-raw)
  8 x stack: u32 hash | i32[32] fc0b | i8[32*1024] fc0w | i32[32] fc1b | i8[32*64] fc1w
             | i32[1] fc2b | i8[32] fc2w                              (raw)

Output is written fully RAW (no LEB128) — the Eonego loader auto-detects either encoding.

    python blend_nnue.py out.nnue in1.nnue in2.nnue [in3.nnue ...]
"""

import sys
import struct

import numpy as np

L1 = 1024
HALFKA = 22528
THREAT = 60720
PSQT = 8
STACKS = 8
FC0_OUT = 32
FC1_IN = 64
FC1_OUT = 32
FC2_IN = 32
VERSION = 0x6A448AFA
LEB_MAGIC = b"COMPRESSED_LEB128"


class Cursor:
    def __init__(self, buf):
        self.buf = buf
        self.pos = 0

    def u32(self):
        (v,) = struct.unpack_from("<I", self.buf, self.pos)
        self.pos += 4
        return v

    def read_array(self, count, dtype):
        """LEB128-if-magic-present, else raw little-endian; returns np.int64 array."""
        if self.buf[self.pos : self.pos + len(LEB_MAGIC)] == LEB_MAGIC:
            self.pos += len(LEB_MAGIC)
            nbytes = self.u32()
            out = np.empty(count, dtype=np.int64)
            buf = self.buf
            pos = self.pos
            for i in range(count):
                result = 0
                shift = 0
                while True:
                    b = buf[pos]
                    pos += 1
                    result |= (b & 0x7F) << shift
                    shift += 7
                    if not (b & 0x80):
                        break
                if shift < 64 and (b & 0x40):
                    result |= -1 << shift
                out[i] = result
            self.pos = pos
            return out
        width = np.dtype(dtype).itemsize
        out = np.frombuffer(self.buf, dtype=dtype, count=count, offset=self.pos).astype(np.int64)
        self.pos += count * width
        return out


REGIONS = [
    ("ft_biases", L1, "<i2"),
    ("threat_w", THREAT * L1, "<i1"),
    ("threat_psqt", THREAT * PSQT, "<i4"),
    ("halfka_w", HALFKA * L1, "<i2"),
    ("halfka_psqt", HALFKA * PSQT, "<i4"),
]
STACK_REGIONS = [
    ("fc0b", FC0_OUT, "<i4"),
    ("fc0w", FC0_OUT * L1, "<i1"),
    ("fc1b", FC1_OUT, "<i4"),
    ("fc1w", FC1_OUT * FC1_IN, "<i1"),
    ("fc2b", 1, "<i4"),
    ("fc2w", FC2_IN, "<i1"),
]


def parse(path):
    buf = open(path, "rb").read()
    c = Cursor(buf)
    version = c.u32()
    assert version == VERSION, f"{path}: version 0x{version:08X} != 0x{VERSION:08X}"
    arch_hash = c.u32()
    desc_len = c.u32()
    desc = buf[c.pos : c.pos + desc_len]
    c.pos += desc_len
    ft_hash = c.u32()
    regions = {}
    for name, count, dtype in REGIONS:
        regions[name] = c.read_array(count, dtype)

    # Newer serializations pad the output layer's weights from Fc2In=32 to 128 bytes per stack
    # (+96 dead bytes past the real inputs; verified: fc2 biases only land at sane magnitudes
    # under the tail-pad reading). Detect from the remaining byte count; emit unpadded on write.
    stack_bytes = sum(count * np.dtype(dtype).itemsize for _, count, dtype in STACK_REGIONS)
    rem = len(buf) - c.pos
    if rem == STACKS * (4 + stack_bytes):
        fc2_pad = 0
    elif rem == STACKS * (4 + stack_bytes + 96):
        fc2_pad = 96
    else:
        raise AssertionError(f"{path}: unexpected stack region size {rem}")

    stack_hashes = []
    for s in range(STACKS):
        stack_hashes.append(c.u32())
        for name, count, dtype in STACK_REGIONS:
            regions[f"s{s}.{name}"] = c.read_array(count, dtype)
            if name == "fc2w" and fc2_pad:
                pad = buf[c.pos : c.pos + fc2_pad]
                if any(pad):
                    print(f"  WARNING {path} s{s}: nonzero fc2w padding ignored")
                c.pos += fc2_pad
    assert c.pos == len(buf), f"{path}: trailing {len(buf) - c.pos} bytes (layout mismatch)"
    return {
        "arch_hash": arch_hash,
        "desc": desc,
        "ft_hash": ft_hash,
        "stack_hashes": stack_hashes,
        "regions": regions,
    }


def clip_for(dtype):
    info = np.iinfo(np.dtype(dtype))
    return info.min, info.max


def main():
    out_path, in_paths = sys.argv[1], sys.argv[2:]
    assert len(in_paths) >= 2, "need at least 2 input nets"
    nets = []
    for p in in_paths:
        print(f"parsing {p} ...", flush=True)
        nets.append(parse(p))
    base = nets[0]
    for n in nets[1:]:
        assert n["arch_hash"] == base["arch_hash"], "architecture hash mismatch"
        assert n["ft_hash"] == base["ft_hash"], "FT hash mismatch"
        assert n["stack_hashes"] == base["stack_hashes"], "stack hash mismatch"

    all_regions = [(name, count, dtype) for name, count, dtype in REGIONS]
    for s in range(STACKS):
        all_regions += [(f"s{s}.{name}", count, dtype) for name, count, dtype in STACK_REGIONS]

    blended = {}
    for name, count, dtype in all_regions:
        acc = np.zeros(count, dtype=np.float64)
        for n in nets:
            acc += n["regions"][name]
        lo, hi = clip_for(dtype)
        blended[name] = np.clip(np.rint(acc / len(nets)), lo, hi).astype(np.dtype(dtype))
        diffs = sum(int((n["regions"][name] != nets[0]["regions"][name]).sum()) for n in nets[1:])
        print(f"  {name:14s} {count:>10} elems, {diffs:>10} input diffs vs net0", flush=True)

    desc = base["desc"] + f" [Eonego blend of {len(in_paths)} nets]".encode()
    with open(out_path, "wb") as f:
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<I", base["arch_hash"]))
        f.write(struct.pack("<I", len(desc)))
        f.write(desc)
        f.write(struct.pack("<I", base["ft_hash"]))
        for name, count, dtype in REGIONS:
            f.write(blended[name].tobytes())
            if name == "ft_biases":
                pass
        # NOTE: region order above matches REGIONS iteration; stacks follow.
        for s in range(STACKS):
            f.write(struct.pack("<I", base["stack_hashes"][s]))
            for name, count, dtype in STACK_REGIONS:
                f.write(blended[f"s{s}.{name}"].tobytes())
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()

"""Export a graph checkpoint to the EONGR01 native runtime format.

EONGR01 starts with the 36-byte engine header, followed by an explicit tensor table:

    magic      "EONGRW1\0" 8 bytes
    count      u32
    tensor[]:
      nameLen  u32
      name     utf8 bytes
      dtype    u32   (1 = float32)
      rank     u32
      dims     u32[rank]
      nbytes   u64
      data     little-endian float32 bytes, row-major

The native CUDA runtime can parse this without Python, torch, pickle, or reflection.
"""

from __future__ import annotations

import argparse
import struct

import numpy as np
import torch

import graph_dataset as gd

MAGIC = b"EONGR01!"
WEIGHTS_MAGIC = b"EONGRW1\0"
VERSION = 1
HEADER = struct.Struct("<8sIIIIIII")
FLAG_POLICY = 1
FLAG_WDL = 2
TENSOR_HDR = struct.Struct("<II")
TENSOR_BYTES = struct.Struct("<Q")
DTYPE_F32 = 1


def _write_tensor(f, name: str, tensor):
    arr = tensor.detach().cpu().numpy().astype("<f4", copy=False)
    name_b = name.encode("utf-8")
    f.write(struct.pack("<I", len(name_b)))
    f.write(name_b)
    f.write(TENSOR_HDR.pack(DTYPE_F32, arr.ndim))
    for d in arr.shape:
        f.write(struct.pack("<I", int(d)))
    data = np.ascontiguousarray(arr).tobytes()
    f.write(TENSOR_BYTES.pack(len(data)))
    f.write(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--value-scale", type=int, default=600)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    d_model = int(a.get("d_model", a.get("d-model", 128)))
    layers = int(a.get("layers", 3))
    flags = FLAG_POLICY | FLAG_WDL
    sd = ck["model"]
    names = sorted(sd.keys())

    with open(args.out, "wb") as f:
        f.write(HEADER.pack(MAGIC, VERSION, gd.SCHEMA_HASH, d_model, layers, flags, args.value_scale, 0))
        f.write(WEIGHTS_MAGIC)
        f.write(struct.pack("<I", len(names)))
        for name in names:
            _write_tensor(f, name, sd[name])

    print(
        f"wrote {args.out}: d_model={d_model} layers={layers} "
        f"schema={gd.SCHEMA_HASH:#x} tensors={len(names)}"
    )


if __name__ == "__main__":
    main()

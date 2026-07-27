"""Graph model smoke/parity helper for EONGF01 dumps.

This validates that Python can read the engine dump and run the checkpoint on the same fixed schema.
CUDA-runtime numeric parity will be added when eonego_cuda implements inference.
"""

from __future__ import annotations

import argparse

import torch

import graph_dataset as gd
from graph_model import GraphNet
from policy_device import device_summary, select_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--take", type=int, default=16)
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = ap.parse_args()

    try:
        dev = select_device(args.device)
    except RuntimeError as exc:
        ap.error(str(exc))
    print(device_summary(dev))

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    model = GraphNet(int(a.get("d_model", a.get("d-model", 128))), int(a.get("layers", 3))).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    dump = gd.read_dump(args.dump)
    n = min(args.take, dump.node_count.shape[0])
    with torch.no_grad():
        value, wdl, from_l, to_l = model(
            torch.from_numpy(dump.nodes[:n]).to(dev),
            torch.from_numpy(dump.edges[:n]).to(dev),
            torch.from_numpy(dump.globals[:n]).to(dev),
            torch.from_numpy(dump.node_count[:n].astype("int64")).to(dev),
        )
    print(f"rows={n} value={tuple(value.shape)} wdl={tuple(wdl.shape)} from={tuple(from_l.shape)} to={tuple(to_l.shape)}")
    print(f"value_range=[{float(value.min()):.4f},{float(value.max()):.4f}]")


if __name__ == "__main__":
    main()
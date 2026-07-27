"""Train the EONGR01 graph value/policy prototype."""

from __future__ import annotations

import argparse

import numpy as np
import torch

import graph_dataset as gd
from graph_model import GraphNet
from policy_dataset import QMAX, build_policy_arrays, read_gen
from policy_device import device_summary, select_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="gen/tbgen records aligned with --dump")
    ap.add_argument("--dump", required=True, help="dumpgraph output")
    ap.add_argument("--out", required=True, help="torch checkpoint")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--policy-weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = ap.parse_args()

    try:
        dev = select_device(args.device)
    except RuntimeError as exc:
        ap.error(str(exc))
    print(device_summary(dev))
    torch.manual_seed(args.seed)

    dump = gd.read_dump(args.dump)
    records = read_gen(args.data)
    if len(records) != dump.node_count.shape[0]:
        raise SystemExit(f"data/dump mismatch: {len(records)} vs {dump.node_count.shape[0]}")

    cp = np.array([r[1] if not r[0].split()[1] == "b" else -r[1] for r in records], dtype=np.float32)
    target = np.tanh(cp / 600.0).astype(np.float32)
    wdl = np.array([0 if (r[2] if r[0].split()[1] != "b" else 1.0 - r[2]) > 0.75 else (1 if (r[2] if r[0].split()[1] != "b" else 1.0 - r[2]) > 0.25 else 2) for r in records], dtype=np.int64)
    keep_pol, pol = build_policy_arrays(records)
    has_pol = np.zeros(len(records), dtype=bool)
    qf_all = np.zeros((len(records), QMAX), dtype=np.int64)
    qt_all = np.zeros((len(records), QMAX), dtype=np.int64)
    qn_all = np.zeros(len(records), dtype=np.int64)
    good_all = np.zeros((len(records), QMAX), dtype=bool)
    if keep_pol.size:
        has_pol[keep_pol] = True
        qf_all[keep_pol] = pol["qf"].astype(np.int64)
        qt_all[keep_pol] = pol["qt"].astype(np.int64)
        qn_all[keep_pol] = pol["qn"].astype(np.int64)
        good_all[keep_pol] = pol["good"]
    print(f"policy rows: {int(has_pol.sum())}/{len(records)}")

    n = target.shape[0]
    rng = np.random.default_rng(args.seed)
    idx_all = rng.permutation(n)
    n_val = max(64, n // 50) if n >= 128 else max(1, n // 5)
    val_i, tr_i = idx_all[:n_val], idx_all[n_val:]

    model = GraphNet(args.d_model, args.layers).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def batch(idx):
        return (
            torch.from_numpy(dump.nodes[idx]).to(dev),
            torch.from_numpy(dump.edges[idx]).to(dev),
            torch.from_numpy(dump.globals[idx]).to(dev),
            torch.from_numpy(dump.node_count[idx].astype(np.int64)).to(dev),
            torch.from_numpy(target[idx]).to(dev),
            torch.from_numpy(wdl[idx]).to(dev),
            torch.from_numpy(qf_all[idx]).to(dev),
            torch.from_numpy(qt_all[idx]).to(dev),
            torch.from_numpy(qn_all[idx]).to(dev),
            torch.from_numpy(good_all[idx]).to(dev),
            torch.from_numpy(has_pol[idx]).to(dev),
        )

    def policy_loss(from_logits, to_logits, qf, qt, qn, good, mask):
        if args.policy_weight <= 0.0 or not bool(mask.any()):
            return from_logits.sum() * 0.0
        row = torch.nonzero(mask, as_tuple=False).squeeze(1)
        qf = qf[row]
        qt = qt[row]
        qn = qn[row]
        good = good[row]
        fl = from_logits[row].gather(1, qf)
        tl = to_logits[row].gather(1, qt)
        logits = fl + tl
        valid = torch.arange(QMAX, device=dev).unsqueeze(0) < qn.unsqueeze(1)
        logits = logits.masked_fill(~valid, -1.0e9)
        target = good.float()
        target = target / target.sum(dim=1, keepdim=True).clamp_min(1.0)
        return -(target * torch.nn.functional.log_softmax(logits, dim=1)).sum(dim=1).mean()

    def policy_acc(from_logits, to_logits, qf, qt, qn, good, mask):
        if not bool(mask.any()):
            return float("nan")
        row = torch.nonzero(mask, as_tuple=False).squeeze(1)
        qf = qf[row]
        qt = qt[row]
        qn = qn[row]
        good = good[row]
        logits = from_logits[row].gather(1, qf) + to_logits[row].gather(1, qt)
        valid = torch.arange(QMAX, device=dev).unsqueeze(0) < qn.unsqueeze(1)
        logits = logits.masked_fill(~valid, -1.0e9)
        pick = logits.argmax(dim=1)
        return good.gather(1, pick.unsqueeze(1)).float().mean().item()

    for ep in range(args.epochs):
        model.train()
        perm = rng.permutation(tr_i)
        total = 0.0
        seen = 0
        for s in range(0, perm.size, args.batch):
            idx = perm[s : s + args.batch]
            nodes, edges, glob, nc, yv, yw, qf, qt, qn, good, pmask = batch(idx)
            pv, pw, pf, pt = model(nodes, edges, glob, nc)
            loss = (
                torch.nn.functional.smooth_l1_loss(pv, yv)
                + 0.25 * torch.nn.functional.cross_entropy(pw, yw)
                + args.policy_weight * policy_loss(pf, pt, qf, qt, qn, good, pmask)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * idx.size
            seen += idx.size
        model.eval()
        with torch.no_grad():
            nodes, edges, glob, nc, yv, yw, qf, qt, qn, good, pmask = batch(val_i)
            pv, pw, pf, pt = model(nodes, edges, glob, nc)
            mae = torch.mean(torch.abs(pv - yv)).item()
            wdl_acc = (pw.argmax(dim=1) == yw).float().mean().item()
            pol_acc = policy_acc(pf, pt, qf, qt, qn, good, pmask)
        pol_text = "nan" if np.isnan(pol_acc) else f"{pol_acc:.4f}"
        print(
            f"epoch {ep + 1}/{args.epochs} loss={total / max(1, seen):.5f} "
            f"val_value_mae={mae:.5f} val_wdl_acc={wdl_acc:.4f} val_policy_acc={pol_text}"
        )

    torch.save({"model": model.state_dict(), "args": vars(args), "schema_hash": gd.SCHEMA_HASH}, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

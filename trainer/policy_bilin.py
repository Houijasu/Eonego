"""Bilinear-interaction policy head trainer (phase 1: the learning ceiling).

    uv run --python 3.12 --with torch,numpy python policy_bilin.py \
        --data data/pol1/tbgen345.txt --dump data/pol1/tbgen345_ft.bin \
        --out data/pol1/bilin32.npz [--dim 32] [--holdout-sig-frac 0.10]

Two validation metrics per epoch:
  val_preserve     -- random rows from TRAINING signatures (interpolation)
  holdout_preserve -- rows from HELD-OUT signatures the net never saw (generalization)
The gap between them is the memorization measure. --dim 0 = additive ablation baseline.
No WDL head here (policy_train.py owns that path); --take N truncates BOTH inputs for smokes.
"""

import argparse
import zlib

import numpy as np

import policy_dataset as pd
from policy_model import SHIFT0, SHIFT_EMB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dim", type=int, default=32, help="interaction rank; 0 = additive baseline")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--holdout-sig-frac", type=float, default=0.10)
    ap.add_argument("--take", type=int, default=0, help="first N records only (smoke runs)")
    args = ap.parse_args()

    import torch

    from policy_model import BilinearPolicyHead, masked_goodset_ce_bilin

    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    records = pd.read_gen(args.data)
    if args.take:
        records = records[: args.take]
        recs = np.fromfile(args.dump, dtype=pd.DUMP_DTYPE, count=args.take)
    else:
        recs = pd.read_dump(args.dump)
    assert len(records) == recs.size, f"data/dump mismatch: {len(records)} vs {recs.size}"
    keep, arrs = pd.build_policy_arrays(records)
    sigs = pd.signatures_for(records, keep)
    print(f"{len(records)} records -> {keep.size} discriminative rows, {np.unique(sigs).size} signatures")
    assert keep.size >= 64, "not enough usable rows"

    ft_np = recs["ft"][keep]
    qf_np, qt_np = arrs["qf"], arrs["qt"]
    qn_np, good_np = arrs["qn"], arrs["good"]

    # Signature-disjoint holdout: whole signatures leave the training pool (seeded hash), so
    # holdout_preserve measures generalization to material the net NEVER trained on.
    def held(s):
        return (zlib.crc32(f"{s}#{args.seed}".encode()) % 10_000) < int(args.holdout_sig_frac * 10_000)

    held_sigs = {s for s in np.unique(sigs) if held(s)}
    hold_mask = np.array([s in held_sigs for s in sigs])
    hold_i = np.nonzero(hold_mask)[0]
    pool = np.nonzero(~hold_mask)[0]
    print(f"holdout: {len(held_sigs)} signatures, {hold_i.size} rows; pool {pool.size} rows")
    assert pool.size >= 64, "holdout ate the training pool"

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(pool)
    n_val = max(64, int(pool.size * args.val_frac))
    val_i, tr_i = perm[:n_val], perm[n_val:]

    model = BilinearPolicyHead(hidden=args.hidden, dim=args.dim, seed=args.seed).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def batch_tensors(idx):
        ft = torch.from_numpy(ft_np[idx].astype(np.float32)).to(dev)
        qf = torch.from_numpy(qf_np[idx].astype(np.int64)).to(dev)
        qt = torch.from_numpy(qt_np[idx].astype(np.int64)).to(dev)
        qn = torch.from_numpy(qn_np[idx].astype(np.int64)).to(dev)
        good = torch.from_numpy(good_np[idx]).to(dev)
        return ft, qf, qt, qn, good

    def preserve(indices):
        if indices.size == 0:
            return float("nan")
        model.eval()
        hits = tot = 0
        with torch.no_grad():
            for s in range(0, indices.size, args.batch):
                idx = indices[s : s + args.batch]
                ft, qf, qt, qn, good = batch_tensors(idx)
                fl, tl, ef, et = model(ft)
                _, ql = masked_goodset_ce_bilin(fl, tl, ef, et, qf, qt, qn, good)
                arg = ql.argmax(dim=1)
                hits += int(good.gather(1, arg.unsqueeze(1)).sum())
                tot += idx.size
        model.train()
        return hits / tot

    model.train()
    for ep in range(args.epochs):
        ep_perm = rng.permutation(tr_i)
        tot_loss, cnt = 0.0, 0
        for s in range(0, ep_perm.size, args.batch):
            idx = ep_perm[s : s + args.batch]
            ft, qf, qt, qn, good = batch_tensors(idx)
            fl, tl, ef, et = model(ft)
            ce, _ = masked_goodset_ce_bilin(fl, tl, ef, et, qf, qt, qn, good)
            loss = ce.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += float(loss.detach()) * idx.size
            cnt += idx.size
        vp, hp = preserve(val_i), preserve(hold_i)
        print(
            f"epoch {ep + 1}/{args.epochs} loss={tot_loss / cnt:.4f} "
            f"val_preserve={vp:.4f} holdout_preserve={hp:.4f}"
        )

    out = {
        "w0": np.clip(np.round(model.w0.detach().cpu().numpy()), -127, 127).astype(np.int8),
        "b0": np.round(model.b0.detach().cpu().numpy()).astype(np.int32),
        "wf": np.clip(np.round(model.wf.detach().cpu().numpy()), -127, 127).astype(np.int8),
        "bf": np.round(model.bf.detach().cpu().numpy()).astype(np.int32),
        "wt": np.clip(np.round(model.wt.detach().cpu().numpy()), -127, 127).astype(np.int8),
        "bt": np.round(model.bt.detach().cpu().numpy()).astype(np.int32),
        "shift0": np.int32(SHIFT0),
        "shift_emb": np.int32(SHIFT_EMB),
        "hidden": np.int32(args.hidden),
        "dim": np.int32(args.dim),
    }
    if args.dim > 0:
        out["wef"] = np.clip(np.round(model.wef.detach().cpu().numpy()), -127, 127).astype(np.int8)
        out["bef"] = np.round(model.bef.detach().cpu().numpy()).astype(np.int32)
        out["wet"] = np.clip(np.round(model.wet.detach().cpu().numpy()), -127, 127).astype(np.int8)
        out["bet"] = np.round(model.bet.detach().cpu().numpy()).astype(np.int32)
    np.savez(args.out, **out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

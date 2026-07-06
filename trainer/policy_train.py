"""Train the EONPOL policy head (+ optional WDL head) on gen/tbgen records + a dumpft FT dump.

    uv run --with torch,numpy,chess python policy_train.py \
        --data data/pol1/tbgen345.txt --dump data/pol1/tbgen345_ft.bin \
        --out data/pol1/head.npz [--net ../nets/main.nnue]

Loss = multi-label quiet-conditional CE over the WDL-preserving move set (tbgen records; gen-v2
records degrade to one-hot best) + lam_wdl * CE(wdl class). Memory-lean for multi-million-row
corpora: everything stays numpy in RAM (u8/i8), converted to torch per batch. Headline metric =
held-out PRESERVE RATE: fraction of positions where the head's argmax quiet keeps the theoretical
result — the trainable definition of "100% correct play".
"""

import argparse

import numpy as np

import policy_dataset as pd
from policy_model import SHIFT0, WDL_SHIFT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="gen/tbgen records")
    ap.add_argument("--dump", required=True, help="dumpft binary aligned with --data")
    ap.add_argument("--out", required=True, help="output .npz")
    ap.add_argument("--net", default=None, help=".nnue (enables WDL head training via frozen-stack a1)")
    ap.add_argument("--hidden", type=int, default=256, help="pfc0 width (EONPOL02: 32..1024, mult of 32)")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--lam-wdl", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.02)
    args = ap.parse_args()

    import torch

    from policy_model import PolicyHead, WDLHead, masked_goodset_ce

    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    records = pd.read_gen(args.data)
    recs = pd.read_dump(args.dump)
    assert len(records) == recs.size, f"data/dump mismatch: {len(records)} vs {recs.size}"
    keep, arrs = pd.build_policy_arrays(records)
    print(f"{len(records)} records -> {keep.size} discriminative quiet rows")
    assert keep.size >= 64, "not enough usable rows"

    # Numpy-resident training set (u8/i8); torch conversion happens per batch.
    ft_np = recs["ft"][keep]  # (N, 1024) u8
    qf_np, qt_np = arrs["qf"], arrs["qt"]
    qn_np, good_np, tgt_np = arrs["qn"], arrs["good"], arrs["tgt"]

    use_wdl = args.net is not None
    if use_wdl:
        a1_np, buckets_np = pd.compute_a1(args.net, recs)
        a1_np, buckets_np = a1_np[keep], buckets_np[keep]
        wdl_np = arrs["wdl"]

    n = keep.size
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_val = max(64, int(n * args.val_frac))
    val_i, tr_i = perm[:n_val], perm[n_val:]

    model = PolicyHead(hidden=args.hidden, seed=args.seed).to(dev)
    heads = list(model.parameters())
    wdl_head = None
    if use_wdl:
        wdl_head = WDLHead(seed=args.seed).to(dev)
        heads += list(wdl_head.parameters())
    opt = torch.optim.Adam(heads, lr=args.lr)

    def batch_tensors(idx):
        ft = torch.from_numpy(ft_np[idx].astype(np.float32)).to(dev)
        qf = torch.from_numpy(qf_np[idx].astype(np.int64)).to(dev)
        qt = torch.from_numpy(qt_np[idx].astype(np.int64)).to(dev)
        qn = torch.from_numpy(qn_np[idx].astype(np.int64)).to(dev)
        good = torch.from_numpy(good_np[idx]).to(dev)
        return ft, qf, qt, qn, good

    def eval_val():
        model.eval()
        hits = wdl_hits = tot = 0
        with torch.no_grad():
            for s in range(0, val_i.size, args.batch):
                idx = val_i[s : s + args.batch]
                ft, qf, qt, qn, good = batch_tensors(idx)
                fl, tl = model(ft)
                _, ql = masked_goodset_ce(fl, tl, qf, qt, qn, good)
                arg = ql.argmax(dim=1)
                hits += int(good.gather(1, arg.unsqueeze(1)).sum())
                tot += idx.size
                if use_wdl:
                    a1 = torch.from_numpy(a1_np[idx].astype(np.float32)).to(dev)
                    bk = torch.from_numpy(buckets_np[idx]).to(dev)
                    wl = wdl_head(a1, bk)
                    wdl_hits += int((wl.argmax(dim=1) == torch.from_numpy(wdl_np[idx].astype(np.int64)).to(dev)).sum())
        model.train()
        return hits / tot, (wdl_hits / tot if use_wdl else float("nan"))

    model.train()
    for ep in range(args.epochs):
        ep_perm = rng.permutation(tr_i)
        tot_loss, cnt = 0.0, 0
        for s in range(0, ep_perm.size, args.batch):
            idx = ep_perm[s : s + args.batch]
            ft, qf, qt, qn, good = batch_tensors(idx)
            fl, tl = model(ft)
            ce, _ = masked_goodset_ce(fl, tl, qf, qt, qn, good)
            loss = ce.mean()
            if use_wdl:
                a1 = torch.from_numpy(a1_np[idx].astype(np.float32)).to(dev)
                bk = torch.from_numpy(buckets_np[idx]).to(dev)
                wl = wdl_head(a1, bk) / float(1 << WDL_SHIFT)
                wt = torch.from_numpy(wdl_np[idx].astype(np.int64)).to(dev)
                loss = loss + args.lam_wdl * torch.nn.functional.cross_entropy(wl, wt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += float(loss.detach()) * idx.size
            cnt += idx.size
        pr, wacc = eval_val()
        print(
            f"epoch {ep + 1}/{args.epochs} loss={tot_loss / cnt:.4f} "
            f"val_preserve={pr:.4f}" + (f" val_wdl_acc={wacc:.4f}" if use_wdl else "")
        )

    out = {
        "w0": np.clip(np.round(model.w0.detach().cpu().numpy()), -127, 127).astype(np.int8),
        "b0": np.round(model.b0.detach().cpu().numpy()).astype(np.int32),
        "wf": np.clip(np.round(model.wf.detach().cpu().numpy()), -127, 127).astype(np.int8),
        "bf": np.round(model.bf.detach().cpu().numpy()).astype(np.int32),
        "wt": np.clip(np.round(model.wt.detach().cpu().numpy()), -127, 127).astype(np.int8),
        "bt": np.round(model.bt.detach().cpu().numpy()).astype(np.int32),
        "shift0": np.int32(SHIFT0),
        "wdl_shift": np.int32(WDL_SHIFT),
        "hidden": np.int32(args.hidden),
    }
    if use_wdl:
        out["wdl_w"] = np.clip(np.round(wdl_head.w.detach().cpu().numpy()), -127, 127).astype(np.int8)
        out["wdl_b"] = np.round(wdl_head.b.detach().cpu().numpy()).astype(np.int32)
    np.savez(args.out, **out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

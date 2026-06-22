"""Train one generation of the Eonego NNUE and export a v2 EONGNNUE file.

Loss is in win-probability space (the standard NNUE objective):
    wp_eval = sigmoid(cp_white / K)
    target  = lambda * wp_eval + (1 - lambda) * result_white
    loss    = MSE( sigmoid(model_cp / K), target )
All quantities are WHITE-RELATIVE, so no sign juggling. cp labels are clamped to +-CP_CLAMP
before the sigmoid so mate scores don't distort the grid.
"""

import argparse
import os

import numpy as np
import torch

import export
from dataset import encode_dataset, parse_records
from model import EonegoNet

CP_CLAMP = 2000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True, help="EONGNNUE output path")
    ap.add_argument("--init", default=None, help="warm-start .pt state_dict")
    ap.add_argument("--save-pt", default=None, help="also save float state_dict here")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--K", type=float, default=200.0)
    ap.add_argument("--lam", type=float, default=1.0, help="eval vs WDL blend (1=pure eval)")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=25)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    fens, cp, res = parse_records(args.data)
    print(f"records: {len(fens)}")
    X = encode_dataset(fens, cache=args.data + ".npz")

    cp_c = np.clip(cp, -CP_CLAMP, CP_CLAMP).astype(np.float32)
    wp_eval = 1.0 / (1.0 + np.exp(-cp_c / args.K))
    target = (args.lam * wp_eval + (1.0 - args.lam) * res).astype(np.float32)

    n = len(fens)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_val = max(1, int(n * args.val_frac))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device,
          ("(" + torch.cuda.get_device_name(0) + ")" if device.type == "cuda" else ""))

    Xt = torch.from_numpy(X.astype(np.float32)).to(device)
    yt = torch.from_numpy(target).to(device)
    tr_idx_t = torch.from_numpy(tr_idx).to(device)
    val_idx_t = torch.from_numpy(val_idx).to(device)

    model = EonegoNet().to(device)
    if args.init and os.path.exists(args.init):
        model.load_state_dict(torch.load(args.init, map_location=device))
        print("warm-started from", args.init)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    K = args.K

    def wp(cp_pred):
        return torch.sigmoid(cp_pred / K)

    best_val = float("inf")
    best_state = None
    bad = 0
    for ep in range(args.epochs):
        model.train()
        ep_perm = tr_idx_t[torch.randperm(tr_idx_t.numel())]
        tot = 0.0
        for b in range(0, ep_perm.numel(), args.batch):
            idx = ep_perm[b:b + args.batch]
            xb = Xt[idx]
            yb = yt[idx]
            opt.zero_grad()
            pred = wp(model(xb))
            loss = torch.mean((pred - yb) ** 2)
            loss.backward()
            opt.step()
            tot += loss.item() * idx.numel()
        tr_loss = tot / tr_idx_t.numel()

        model.eval()
        with torch.no_grad():
            vpred = wp(model(Xt[val_idx_t]))
            val_loss = float(torch.mean((vpred - yt[val_idx_t]) ** 2))
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"ep {ep:3d}  train {tr_loss:.5f}  val {val_loss:.5f}  best {best_val:.5f}")
        if bad >= args.patience:
            print(f"early stop at ep {ep} (best val {best_val:.5f})")
            break

    model.load_state_dict(best_state)
    model.to("cpu")  # export + degradation run numpy on CPU tensors

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    net = export.save(model, args.out)
    if args.save_pt:
        torch.save(model.state_dict(), args.save_pt)

    # export degradation on the val split (float model vs integer forward)
    Xval = X[val_idx]
    deg = export.degradation(model, net, Xval)
    print(f"shifts: L1={net['Shift1']} L2={net['Shift2']} L3={net['Shift3']} "
          f"L4={net['Shift4']} quantScale={net['QuantScale']}")
    print(f"export degradation cp |float-int|: mean={deg.mean():.3f} max={deg.max():.3f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

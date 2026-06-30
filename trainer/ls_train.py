"""Train the LearnedSearch priority net on an lstrace TSV and export it as EONGLS.

Target = ROOT-IMPACT magnitude: how much expanding a node moved the backed-up root value (self-supervised
from the bootstrap trace — no teacher needed for v0). The net learns to predict expansion utility; at search
time the highest-utility frontier leaf is expanded first. Normalised to ~[0,100] and outlier-clipped.

Usage:  python ls_train.py <trace.tsv> <out.lsnet> [epochs]
"""

import sys
import numpy as np
import torch

from ls_features import load_trace
from ls_model import LsModel
from ls_export import save
from ls_intref import int_forward


def main():
    if len(sys.argv) < 3:
        print("usage: python ls_train.py <trace.tsv> <out.lsnet> [epochs]")
        sys.exit(2)
    trace, out = sys.argv[1], sys.argv[2]
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 300

    X, _leaf, impact = load_trace(trace)
    if len(X) == 0:
        print("no training rows in", trace)
        sys.exit(1)

    # Expansion utility = |root-impact|, clipped to a mate-free band and scaled into the activation range.
    y = np.clip(np.abs(impact), 0.0, 2000.0) / 20.0  # -> [0,100]

    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)

    torch.manual_seed(0)
    model = LsModel()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = torch.nn.MSELoss()

    n = len(Xt)
    bs = min(4096, n)
    idx = np.arange(n)
    for ep in range(epochs):
        np.random.shuffle(idx)
        total = 0.0
        for b in range(0, n, bs):
            sel = idx[b:b + bs]
            xb = Xt[sel]
            yb = yt[sel]
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(sel)
        if ep % 50 == 0 or ep == epochs - 1:
            print(f"epoch {ep:4d}  mse {total / n:.3f}")

    net = save(model, out)

    # Quantisation degradation: |int_forward - float| over a sample.
    errs = []
    for i in range(min(1000, n)):
        fi = float(int_forward(net, [int(v) for v in X[i]]))
        ff = float(model(Xt[i:i + 1]).item())
        errs.append(abs(fi - ff))
    print(f"exported {out}  (H1={net.H1} H2={net.H2} shifts={net.Shift1}/{net.Shift2}/{net.OutShift})  "
          f"mean|int-float|={np.mean(errs):.2f}")


if __name__ == "__main__":
    main()

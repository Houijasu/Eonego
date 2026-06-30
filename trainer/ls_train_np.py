"""Numpy-only trainer for the LearnedSearch priority net (no torch dependency).

A 448-weight MLP (NF -> H1 -> H2 -> 1) with clipped-ReLU [0,127] activations, trained with manual-backprop
Adam on MSE against the |root-impact| target. Inputs are the engine's RAW quantised feature ints (NOT
re-normalised — the engine feeds the same raw ints at inference, so the learned scale must match). Exports
EONGLS via ls_export.from_arrays. Run anywhere numpy is available:

    uv run --with numpy python ls_train_np.py <trace.tsv> <out.lsnet> [epochs]

The torch ls_train.py is the canonical/extensible trainer; this is the dependency-light path.
"""

import sys
import numpy as np

from ls_features import load_trace
from ls_export import save_arrays
from ls_intref import int_forward, NF


def main():
    if len(sys.argv) < 3:
        print("usage: python ls_train_np.py <trace.tsv> <out.lsnet> [epochs]")
        sys.exit(2)
    trace, out = sys.argv[1], sys.argv[2]
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 400
    H1, H2 = 16, 16

    X, _leaf, impact = load_trace(trace)
    n = len(X)
    if n == 0:
        print("no training rows in", trace)
        sys.exit(1)
    X = X.astype(np.float32)
    # Expansion utility = |root-impact|, clipped to a mate-free band and scaled into the activation range.
    y = (np.clip(np.abs(impact), 0.0, 2000.0) / 20.0).astype(np.float32)  # -> [0,100]
    print(f"rows={n}  y mean/max={y.mean():.2f}/{y.max():.2f}")

    rng = np.random.default_rng(0)
    W1 = (rng.standard_normal((H1, NF)) * 0.05).astype(np.float32)
    b1 = np.zeros(H1, np.float32)
    W2 = (rng.standard_normal((H2, H1)) * 0.05).astype(np.float32)
    b2 = np.zeros(H2, np.float32)
    W3 = (rng.standard_normal((1, H2)) * 0.05).astype(np.float32)
    b3 = np.zeros(1, np.float32)
    params = [W1, b1, W2, b2, W3, b3]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    lr, beta1, beta2, eps = 1e-3, 0.9, 0.999, 1e-8
    t = 0
    bs = 8192
    idx = np.arange(n)

    for ep in range(epochs):
        rng.shuffle(idx)
        tot = 0.0
        for s in range(0, n, bs):
            sel = idx[s:s + bs]
            xb, yb = X[sel], y[sel]
            nb = len(sel)
            # forward
            z1 = xb @ W1.T + b1
            a1 = np.clip(z1, 0.0, 127.0)
            z2 = a1 @ W2.T + b2
            a2 = np.clip(z2, 0.0, 127.0)
            pred = (a2 @ W3.T + b3).ravel()
            diff = pred - yb
            tot += float(np.sum(diff * diff))
            # backward
            dout = (2.0 / nb) * diff
            dW3 = dout[None, :] @ a2                      # (1,H2)
            db3 = np.array([dout.sum()], np.float32)
            da2 = (dout[:, None] @ W3) * ((z2 > 0) & (z2 < 127))
            dW2 = da2.T @ a1                              # (H2,H1)
            db2 = da2.sum(0)
            da1 = (da2 @ W2) * ((z1 > 0) & (z1 < 127))
            dW1 = da1.T @ xb                             # (H1,NF)
            db1 = da1.sum(0)
            grads = [dW1.astype(np.float32), db1.astype(np.float32),
                     dW2.astype(np.float32), db2.astype(np.float32),
                     dW3.astype(np.float32), db3]
            t += 1
            for i, (p, g) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * g
                v[i] = beta2 * v[i] + (1 - beta2) * (g * g)
                mhat = m[i] / (1 - beta1 ** t)
                vhat = v[i] / (1 - beta2 ** t)
                p -= lr * mhat / (np.sqrt(vhat) + eps)
        if ep % 50 == 0 or ep == epochs - 1:
            print(f"epoch {ep:4d}  mse {tot / n:.3f}")

    net = save_arrays(W1, b1, W2, b2, W3, b3, out)

    # Quantisation degradation (int vs float) over a sample.
    errs = []
    for i in range(min(1000, n)):
        fi = float(int_forward(net, [int(round(float(vv))) for vv in X[i]]))
        z1 = X[i] @ W1.T + b1
        a1 = np.clip(z1, 0, 127)
        z2 = a1 @ W2.T + b2
        a2 = np.clip(z2, 0, 127)
        ff = float((a2 @ W3.T + b3).ravel()[0])
        errs.append(abs(fi - ff))
    print(f"exported {out}  shifts {net.Shift1}/{net.Shift2}/{net.OutShift}  "
          f"mean|int-float|={np.mean(errs):.2f}")


if __name__ == "__main__":
    main()

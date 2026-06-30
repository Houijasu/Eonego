"""Quantise a trained float LsModel to the EONGLS int8 format and write it.

Per-layer right-shift `s` is the largest power of two that keeps round(W * 2^s) within int8 [-127,127].
Inputs (engine feature row) and hidden activations are small ints in [-127,127] / [0,127], so:
    int_acc = x . round(W*2^s) + round(b*2^s)   ~=   (x.W + b) * 2^s
    (int_acc + round) >> s   ~=   x.W + b        (then clipped, as in ls_intref / the engine).
"""

import numpy as np
from ls_intref import LsNet, serialize


def derive_shift(W, maxw=127):
    m = float(np.max(np.abs(W)))
    if m <= 1e-9:
        return 0
    s = 0
    while s < 30 and (m * (1 << (s + 1))) <= maxw:
        s += 1
    return s


def qlayer(Wf, bf, s):
    Wq = np.clip(np.round(Wf * (1 << s)), -127, 127).astype(np.int32)
    bq = np.round(bf * (1 << s)).astype(np.int32)
    return Wq, bq


def from_arrays(W1f, b1f, W2f, b2f, W3f, b3f):
    """Quantise raw float weight arrays (W1 (H1,NF), W2 (H2,H1), W3 (1,H2)) into an EONGLS LsNet."""
    s1, s2, so = derive_shift(W1f), derive_shift(W2f), derive_shift(W3f)
    W1q, b1q = qlayer(W1f, b1f, s1)
    W2q, b2q = qlayer(W2f, b2f, s2)
    W3q, b3q = qlayer(W3f, b3f, so)

    h1, nf = W1f.shape
    h2 = W2f.shape[0]
    return LsNet(
        nf, h1, h2, s1, s2, so,
        W1q.reshape(-1), b1q,   # row-major W1[j*NF + i]
        W2q.reshape(-1), b2q,   # W2[j*H1 + i]
        W3q.reshape(-1), b3q,   # W3[i]
    )


def from_torch(model):
    return from_arrays(
        model.l1.weight.detach().cpu().numpy(), model.l1.bias.detach().cpu().numpy(),
        model.l2.weight.detach().cpu().numpy(), model.l2.bias.detach().cpu().numpy(),
        model.l3.weight.detach().cpu().numpy(), model.l3.bias.detach().cpu().numpy(),
    )


def save(model, path):
    net = from_torch(model)
    with open(path, "wb") as f:
        f.write(serialize(net))
    return net


def save_arrays(W1f, b1f, W2f, b2f, W3f, b3f, path):
    net = from_arrays(W1f, b1f, W2f, b2f, W3f, b3f)
    with open(path, "wb") as f:
        f.write(serialize(net))
    return net

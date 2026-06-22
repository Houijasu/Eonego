"""Post-training quantization: float EonegoNet -> v2 EONGNNUE integer net + bytes.

Per-layer shift = largest s keeping round(W*2^s) in int8; quantScale = largest q keeping
round(W5*q) in int16. The exported integer weights are the source of truth for the engine;
the hard parity gate (parity_forward.py) checks intref(exported) == engine nnforward exactly.
`degradation` reports how far the integer net drifts from the trained float model.
"""

import numpy as np
import torch

import intref
from model import derive_qscale, derive_shift


def quantize(model):
    """float model -> net dict with UNPADDED int weights + shifts + quantScale."""
    net = {}

    def qlayer(lin, shift):
        scale = float(1 << shift)
        w = torch.clamp(torch.round(lin.weight.detach() * scale), -127, 127)
        b = torch.round(lin.bias.detach() * scale)
        return w.numpy().astype(np.int8), b.numpy().astype(np.int32)

    s = [derive_shift(model.l1.weight), derive_shift(model.l2.weight),
         derive_shift(model.l3.weight), derive_shift(model.l4.weight)]
    net["Shift1"], net["Shift2"], net["Shift3"], net["Shift4"] = s
    net["L1W"], net["L1B"] = qlayer(model.l1, s[0])
    net["L2W"], net["L2B"] = qlayer(model.l2, s[1])
    net["L3W"], net["L3B"] = qlayer(model.l3, s[2])
    net["L4W"], net["L4B"] = qlayer(model.l4, s[3])

    q = derive_qscale(model.l5.weight)
    net["QuantScale"] = q
    w5 = torch.clamp(torch.round(model.l5.weight.detach() * q), -32767, 32767)
    b5 = torch.round(model.l5.bias.detach() * q)
    net["L5W"] = w5.numpy().astype(np.int16)
    net["L5B"] = b5.numpy().astype(np.int32)
    return net


def save(model, path):
    net = quantize(model)
    with open(path, "wb") as f:
        f.write(intref.serialize(net))
    return net


def degradation(model, net, X):
    """abs error array: float model output vs integer forward over X (N,2577)."""
    model.eval()
    with torch.no_grad():
        f = model(torch.from_numpy(X.astype(np.float32))).numpy()
    i = intref.int_forward(net, X).astype(np.float64)
    return np.abs(f - i)

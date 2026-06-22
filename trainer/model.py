"""PyTorch training model for the Eonego region-piececount net (2577->64->32->16->16->1).

The float model mirrors the engine's integer pipeline structurally: each hidden layer is a
linear followed by an integer-rounded clipped-ReLU (round-to-int, clamp [0,127]) via straight-
through estimator, so activations live on the same 0..127 grid the int kernel uses. The output
head is a plain linear producing WHITE-RELATIVE centipawns. Weights stay float during training
and are post-training-quantized at export (see export.py); export degradation is measured, and
the hard parity gate is intref==engine on the exported integer weights.
"""

import math

import torch
import torch.nn as nn


def ste_round(x):
    """Round with a straight-through gradient (identity backward)."""
    return (torch.round(x) - x).detach() + x


def clipped_relu_int(x):
    """Integer-grid clipped ReLU: round to int, clamp to [0,127]. STE gradient."""
    return torch.clamp(ste_round(x), 0.0, 127.0)


def derive_shift(weight, max_bits=31):
    """Largest right-shift s such that round(W*2^s) stays within int8 [-127,127]."""
    m = float(weight.detach().abs().max())
    if m == 0.0:
        return 0
    s = int(math.floor(math.log2(127.0 / m)))
    return max(0, min(max_bits, s))


def derive_qscale(weight):
    """quantScale so round(W5*q) stays within int16 [-32767,32767]."""
    m = float(weight.detach().abs().max())
    if m == 0.0:
        return 1
    q = int(math.floor(32767.0 / m))
    return max(1, q)


class EonegoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(2577, 64)
        self.l2 = nn.Linear(64, 32)
        self.l3 = nn.Linear(32, 16)
        self.l4 = nn.Linear(16, 16)
        self.l5 = nn.Linear(16, 1)
        for lin in (self.l1, self.l2, self.l3, self.l4, self.l5):
            nn.init.uniform_(lin.weight, -0.03, 0.03)
            nn.init.zeros_(lin.bias)

    def forward(self, x):
        """x: (N,2577) float of raw counts (0..127). Returns (N,) white-relative cp."""
        a1 = clipped_relu_int(self.l1(x))
        a2 = clipped_relu_int(self.l2(a1))
        a3 = clipped_relu_int(self.l3(a2))
        a4 = clipped_relu_int(self.l4(a3))
        return self.l5(a4).squeeze(-1)

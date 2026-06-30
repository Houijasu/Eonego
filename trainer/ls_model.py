"""Tiny float MLP for the LearnedSearch priority net: NF -> H1 -> H2 -> 1.

Activations are clamped to [0,127] to mirror the engine's int clipped-ReLU range (export.py quantises the
weights to int8 + per-layer shift; ls_intref does the integer forward). Kept deliberately small and
CPU-cheap — this scores every frontier leaf in the hot search loop.
"""

import torch
import torch.nn as nn
from ls_intref import NF


class LsModel(nn.Module):
    def __init__(self, h1=16, h2=16):
        super().__init__()
        self.l1 = nn.Linear(NF, h1)
        self.l2 = nn.Linear(h1, h2)
        self.l3 = nn.Linear(h2, 1)
        for layer in (self.l1, self.l2, self.l3):
            nn.init.uniform_(layer.weight, -0.05, 0.05)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        a1 = torch.clamp(self.l1(x), 0.0, 127.0)
        a2 = torch.clamp(self.l2(a1), 0.0, 127.0)
        return self.l3(a2).squeeze(-1)

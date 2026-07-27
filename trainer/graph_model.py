"""Compact relation-aware graph model for Eonego EONGF01 dumps."""

from __future__ import annotations

import torch

from graph_dataset import EDGE_FEATURES, GLOBAL_FEATURES, MAX_NODES, NODE_FEATURES

HEAD_OUT = 384


class GraphNet(torch.nn.Module):
    def __init__(self, d_model: int = 128, layers: int = 3):
        super().__init__()
        self.d_model = d_model
        self.layers = layers
        self.node_in = torch.nn.Linear(NODE_FEATURES, d_model)
        self.edge_in = torch.nn.Linear(EDGE_FEATURES, d_model)
        self.global_in = torch.nn.Linear(GLOBAL_FEATURES, d_model)
        self.msg = torch.nn.ModuleList(torch.nn.Linear(d_model * 3, d_model) for _ in range(layers))
        self.upd = torch.nn.ModuleList(torch.nn.Linear(d_model * 2, d_model) for _ in range(layers))
        self.norm = torch.nn.ModuleList(torch.nn.LayerNorm(d_model) for _ in range(layers))
        self.value = torch.nn.Sequential(torch.nn.Linear(d_model * 2, d_model), torch.nn.ReLU(), torch.nn.Linear(d_model, 1))
        self.wdl = torch.nn.Sequential(torch.nn.Linear(d_model * 2, d_model), torch.nn.ReLU(), torch.nn.Linear(d_model, 3))
        self.from_head = torch.nn.Linear(d_model * 2, HEAD_OUT)
        self.to_head = torch.nn.Linear(d_model * 2, HEAD_OUT)

    def forward(self, nodes, edges, globals_, node_count):
        """nodes uint/float (B,32,NF), edges (B,32,32,EF), globals (B,GF)."""
        x = self.node_in(nodes.float() / 255.0)
        e = self.edge_in(edges.float() / 255.0)
        g = self.global_in(globals_.float() / 255.0)
        mask = torch.arange(MAX_NODES, device=x.device).unsqueeze(0) < node_count.long().unsqueeze(1)

        for msg, upd, norm in zip(self.msg, self.upd, self.norm):
            src = x.unsqueeze(2).expand(-1, -1, MAX_NODES, -1)
            dst = x.unsqueeze(1).expand(-1, MAX_NODES, -1, -1)
            m = torch.relu(msg(torch.cat([src, dst, e], dim=-1)))
            edge_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            m = m.masked_fill(~edge_mask.unsqueeze(-1), 0.0).sum(dim=1)
            x = norm(x + torch.relu(upd(torch.cat([x, m], dim=-1))))
            x = x.masked_fill(~mask.unsqueeze(-1), 0.0)

        denom = node_count.clamp_min(1).float().unsqueeze(1)
        pooled = x.sum(dim=1) / denom
        h = torch.cat([pooled, g], dim=1)
        return self.value(h).squeeze(1), self.wdl(h), self.from_head(h), self.to_head(h)
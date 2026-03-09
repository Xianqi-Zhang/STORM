from __future__ import annotations

import torch
from torch import nn


class EmbodimentGraphEncoder(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2) -> None:
        super().__init__()
        self.input_proj = nn.Linear(node_dim, hidden_dim)
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.out_proj = nn.Linear(hidden_dim, out_dim)
        self.act = nn.ReLU()

    def forward(self, nodes: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # nodes: [B, N, D], adj: [B, N, N]
        x = self.act(self.input_proj(nodes))
        for layer in self.layers:
            x = torch.bmm(adj, x)
            x = self.act(layer(x))
        # graph token by mean pooling
        g = x.mean(dim=1)
        return self.out_proj(g)

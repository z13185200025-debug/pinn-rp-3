from __future__ import annotations

import torch
from torch import nn


try:
    from torch_geometric.nn import SAGEConv
except Exception:  # pragma: no cover - optional dependency
    SAGEConv = None


class MeshGraphNet(nn.Module):
    """GraphSAGE-style mesh regressor for CFD point clouds."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128, num_layers: int = 3) -> None:
        super().__init__()
        if SAGEConv is None:
            raise ImportError("torch_geometric is required for MeshGraphNet. Install torch-geometric to use this model.")
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        for _ in range(max(0, num_layers - 1)):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.edge_mlp = nn.Sequential(nn.Linear(3, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor | None = None) -> torch.Tensor:
        h = x
        for conv in self.convs:
            h = torch.relu(conv(h, edge_index))
        if edge_attr is not None and edge_attr.numel() > 0:
            # Edge attributes are currently consumed as a global context term.
            h = h + self.edge_mlp(edge_attr).mean(dim=0, keepdim=True)
        return self.head(h)


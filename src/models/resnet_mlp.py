from __future__ import annotations

import torch
from torch import nn

from .mlp import get_activation


class ResidualBlock(nn.Module):
    """Two-layer fully connected residual block."""

    def __init__(self, hidden_dim: int, activation: str = "silu", dropout: float = 0.0, layer_norm: bool = True) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity()
        self.block = nn.Sequential(
            self.norm1,
            nn.Linear(hidden_dim, hidden_dim),
            get_activation(activation),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            self.norm2,
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.act = get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class ResNetMLP(nn.Module):
    """Residual MLP baseline for deeper tabular regression."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 4,
        activation: str = "silu",
        dropout: float = 0.0,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.input = nn.Sequential(nn.Linear(input_dim, hidden_dim), get_activation(activation))
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, activation=activation, dropout=dropout, layer_norm=layer_norm) for _ in range(num_blocks)]
        )
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input(x)
        h = self.blocks(h)
        return self.output(h)

from __future__ import annotations

import math

import torch
from torch import nn


def get_activation(name: str) -> nn.Module:
    """Return an activation module by name."""
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "silu":
        return nn.SiLU()
    if name == "sin":
        return Sin()
    raise ValueError(f"Unsupported activation: {name}")


class Sin(nn.Module):
    """Sine activation for PINN-style networks."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


class FourierFeatures(nn.Module):
    """Append Fourier features for selected coordinate columns."""

    def __init__(self, input_dim: int, coordinate_indices: list[int] | None = None, num_frequencies: int = 4) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.coordinate_indices = coordinate_indices or []
        self.num_frequencies = num_frequencies
        self.register_buffer("freqs", 2.0 ** torch.arange(num_frequencies, dtype=torch.float32) * math.pi)

    @property
    def output_dim(self) -> int:
        return self.input_dim + len(self.coordinate_indices) * self.num_frequencies * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.coordinate_indices:
            return x
        coords = x[:, self.coordinate_indices]
        feats = []
        for i in range(coords.shape[1]):
            phase = coords[:, i : i + 1] * self.freqs[None, :]
            feats.extend([torch.sin(phase), torch.cos(phase)])
        return torch.cat([x, *feats], dim=1)


class MLPBlock(nn.Module):
    """Configurable dense block."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation: str = "silu",
        dropout: float = 0.0,
        batch_norm: bool = False,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, out_dim)]
        if batch_norm:
            layers.append(nn.BatchNorm1d(out_dim))
        if layer_norm:
            layers.append(nn.LayerNorm(out_dim))
        layers.append(get_activation(activation))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLP(nn.Module):
    """Fully connected regression network for ANN/DNN baselines."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: list[int] | tuple[int, ...] = (128, 128, 128),
        activation: str = "silu",
        dropout: float = 0.0,
        batch_norm: bool = False,
        layer_norm: bool = False,
        residual: bool = False,
        fourier_features: bool = False,
        fourier_num_frequencies: int = 4,
        fourier_coordinate_indices: list[int] | None = None,
    ) -> None:
        super().__init__()
        if fourier_coordinate_indices is None:
            fourier_coordinate_indices = [i for i in [3, 4] if i < input_dim]
        self.fourier = (
            FourierFeatures(input_dim, fourier_coordinate_indices, fourier_num_frequencies)
            if fourier_features
            else nn.Identity()
        )
        prev = self.fourier.output_dim if isinstance(self.fourier, FourierFeatures) else input_dim
        layers: list[nn.Module] = []
        self.residual = residual
        for hidden in hidden_layers:
            layers.append(MLPBlock(prev, hidden, activation, dropout, batch_norm, layer_norm))
            prev = hidden
        self.hidden = nn.ModuleList(layers)
        self.output = nn.Linear(prev, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fourier(x)
        for layer in self.hidden:
            out = layer(h)
            h = h + out if self.residual and out.shape == h.shape else out
        return self.output(h)


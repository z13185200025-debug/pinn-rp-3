from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for wall sequences."""

    def __init__(self, d_model: int, max_len: int = 2048) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerEncoderRegressor(nn.Module):
    """Transformer encoder for axial wall sequences with padding mask support."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.pos(self.proj(x))
        src_key_padding_mask = (mask <= 0) if mask is not None else None
        return self.head(self.encoder(h, src_key_padding_mask=src_key_padding_mask))


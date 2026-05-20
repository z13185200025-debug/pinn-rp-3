from __future__ import annotations

import torch
from torch import nn


class LSTMRegressor(nn.Module):
    """Sequence regressor for wall-wise x histories."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        y, _ = self.rnn(x)
        return self.head(y)


class GRURegressor(nn.Module):
    """GRU sequence regressor for wall temperature/Nu/HTD evolution."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        y, _ = self.rnn(x)
        return self.head(y)


class TemporalBlock(nn.Module):
    """Residual dilated temporal block."""

    def __init__(self, channels: int, dilation: int, dropout: float = 0.0) -> None:
        super().__init__()
        padding = dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=padding, dilation=dilation),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv1d(channels, channels, 3, padding=padding, dilation=dilation),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class TCNRegressor(nn.Module):
    """Residual dilated TCN for padded wall sequences."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128, num_layers: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.input = nn.Conv1d(input_dim, hidden_dim, 1)
        self.blocks = nn.Sequential(*[TemporalBlock(hidden_dim, 2**i, dropout) for i in range(num_layers)])
        self.head = nn.Conv1d(hidden_dim, output_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.input(x.transpose(1, 2))
        y = self.head(self.blocks(h)).transpose(1, 2)
        return y


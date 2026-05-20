from __future__ import annotations

import torch
from torch import nn


class SpectralConv2d(nn.Module):
    """2D spectral convolution used by FNO."""

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int) -> None:
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1 / max(1, in_channels * out_channels)
        self.weights = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(batch, self.weights.shape[1], x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)
        m1 = min(self.modes1, x_ft.size(-2))
        m2 = min(self.modes2, x_ft.size(-1))
        out_ft[:, :, :m1, :m2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :m1, :m2], self.weights[:, :, :m1, :m2])
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


class FNOBlock(nn.Module):
    """Spectral plus pointwise convolution block."""

    def __init__(self, width: int, modes1: int, modes2: int) -> None:
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes1, modes2)
        self.local = nn.Conv2d(width, width, 1)
        self.norm = nn.GroupNorm(1, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(self.norm(self.spectral(x) + self.local(x)))


class FNO2d(nn.Module):
    """Fourier neural operator for regular-grid parameterized fields."""

    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int = 4,
        width: int = 48,
        modes1: int = 12,
        modes2: int = 12,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        self.lift = nn.Conv2d(in_channels, width, 1)
        self.blocks = nn.Sequential(*[FNOBlock(width, modes1, modes2) for _ in range(num_layers)])
        self.head = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(), nn.Conv2d(width, out_channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.lift(x)))

    def physics_loss(self, pred: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Optional PINO hook; not used by default grid training."""
        return pred.sum() * 0


from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Two-convolution block used by U-Net."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(1, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(1, out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet2D(nn.Module):
    """Encoder-decoder U-Net with skip connections for regular x-r grids."""

    def __init__(self, in_channels: int, out_channels: int, base_channels: int = 32, depth: int = 3) -> None:
        super().__init__()
        self.depth = depth
        chs = [base_channels * (2**i) for i in range(depth)]
        self.down = nn.ModuleList()
        prev = in_channels
        for ch in chs:
            self.down.append(ConvBlock(prev, ch))
            prev = ch
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(chs[-1], chs[-1] * 2)
        self.up = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        prev = chs[-1] * 2
        for ch in reversed(chs):
            self.up.append(nn.ConvTranspose2d(prev, ch, 2, stride=2))
            self.up_blocks.append(ConvBlock(ch * 2, ch))
            prev = ch
        self.head = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        h = x
        for block in self.down:
            h = block(h)
            skips.append(h)
            h = self.pool(h)
        h = self.bottleneck(h)
        for up, block, skip in zip(self.up, self.up_blocks, reversed(skips)):
            h = up(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = torch.nn.functional.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = block(torch.cat([h, skip], dim=1))
        return self.head(h)


class CNNRegressor(nn.Module):
    """Lightweight convolutional grid regressor."""

    def __init__(self, in_channels: int, out_channels: int, base_channels: int = 32, depth: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_channels
        for _ in range(depth):
            layers.extend([nn.Conv2d(prev, base_channels, 3, padding=1), nn.GroupNorm(1, base_channels), nn.SiLU()])
            prev = base_channels
        layers.append(nn.Conv2d(prev, out_channels, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


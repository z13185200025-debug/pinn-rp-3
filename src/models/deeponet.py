from __future__ import annotations

import torch
from torch import nn

from .mlp import MLP


class DeepONet(nn.Module):
    """Parameterized operator network with branch and trunk subnets."""

    def __init__(
        self,
        branch_dim: int = 3,
        trunk_dim: int = 2,
        output_dim: int = 4,
        width: int = 128,
        latent_dim: int = 128,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.branch = MLP(branch_dim, output_dim * latent_dim, [width, width], activation="silu")
        self.trunk = MLP(trunk_dim, output_dim * latent_dim, [width, width], activation="silu")
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, branch_input: torch.Tensor, trunk_input: torch.Tensor) -> torch.Tensor:
        b = self.branch(branch_input).view(-1, self.output_dim, self.latent_dim)
        t = self.trunk(trunk_input).view(-1, self.output_dim, self.latent_dim)
        return (b * t).sum(dim=-1) + self.bias


class PhysicsInformedDeepONet(DeepONet):
    """DeepONet skeleton with reserved physics residual hook."""

    def physics_loss(self, pred: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Optional physics residual hook; data training does not depend on it."""
        return pred.sum() * 0


class TabularDeepONet(DeepONet):
    """DeepONet adapter accepting one tabular tensor [D, Tin, qw, x, r]."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x[:, :3], x[:, 3:5])

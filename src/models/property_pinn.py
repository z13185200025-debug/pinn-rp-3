from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .mlp import MLP
from .pinn import PINN


class PropertyNet(nn.Module):
    """Property network mapping thermodynamic/context inputs to positive [rho, Pr]."""

    def __init__(self, input_dim: int = 4, hidden_layers: list[int] | tuple[int, ...] = (64, 64)) -> None:
        super().__init__()
        self.net = MLP(input_dim, 2, hidden_layers=hidden_layers, activation="silu")
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softplus(self.net(x)) + 1e-6


class PropertyPINN(PINN):
    """PINN coupled to a learned RP-3 property relation."""

    def __init__(self, input_dim: int = 5, output_dim: int = 4, property_input_dim: int = 4, property_hidden_layers=(64, 64), **kwargs) -> None:
        super().__init__(input_dim=input_dim, output_dim=output_dim, **kwargs)
        self.property_net = PropertyNet(property_input_dim, property_hidden_layers)

    def property_loss(
        self,
        network_input: torch.Tensor,
        pinn_pred: torch.Tensor,
        target_names: list[str],
        prandtl_data: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Supervise rho/Pr consistency when CFD columns are available."""
        temperature = pinn_pred[:, 0:1]
        if "pressure" in target_names:
            p_idx = target_names.index("pressure")
            pressure_or_case = pinn_pred[:, p_idx : p_idx + 1]
            prop_input = torch.cat([temperature, pressure_or_case, network_input[:, :2]], dim=1)
        else:
            prop_input = torch.cat([temperature, network_input[:, :3]], dim=1)
        props = self.property_net(prop_input)
        rho_idx = target_names.index("density") if "density" in target_names else pinn_pred.shape[1] - 1
        loss = F.mse_loss(torch.relu(pinn_pred[:, rho_idx : rho_idx + 1]), props[:, 0:1])
        if prandtl_data is not None:
            loss = loss + F.mse_loss(prandtl_data, props[:, 1:2])
        return loss


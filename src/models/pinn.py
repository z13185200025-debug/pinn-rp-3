from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .mlp import MLP


def grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """Autograd helper for first derivatives."""
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]


class PINN(nn.Module):
    """PINN mapping case parameters and coordinates to field variables."""

    def __init__(
        self,
        input_dim: int = 6,
        output_dim: int = 4,
        hidden_layers: list[int] | tuple[int, ...] = (128, 128, 128, 128),
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        self.net = MLP(input_dim, output_dim, hidden_layers=hidden_layers, activation=activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PINNLoss(nn.Module):
    """Dimensional supervised and physics loss for a base PINN.

    The network still receives scaled inputs and predicts scaled outputs. Physics
    residuals are computed after inverse scaling, and derivatives use the scaler
    standard deviations through the chain rule.
    """

    def __init__(
        self,
        loss_weights: dict[str, float],
        x_scaler=None,
        y_scaler=None,
        input_names: list[str] | None = None,
        target_names: list[str] | None = None,
        cp_const: float = 2200.0,
        k_const: float = 0.12,
        diameter_scale_to_meter: float = 1.0,
        heat_flux_scale_to_w_m2: float = 1.0,
    ) -> None:
        super().__init__()
        self.weights = loss_weights
        self.cp_const = cp_const
        self.k_const = k_const
        self.input_names = input_names or []
        self.target_names = target_names or []
        self.diameter_scale_to_meter = diameter_scale_to_meter
        self.heat_flux_scale_to_w_m2 = heat_flux_scale_to_w_m2
        self.x_mean = np.asarray(getattr(x_scaler, "mean_", np.zeros(len(self.input_names))), dtype=np.float32)
        self.x_std = np.asarray(getattr(x_scaler, "scale_", np.ones(len(self.input_names))), dtype=np.float32)
        self.y_mean = np.asarray(getattr(y_scaler, "mean_", np.zeros(len(self.target_names))), dtype=np.float32)
        self.y_std = np.asarray(getattr(y_scaler, "scale_", np.ones(len(self.target_names))), dtype=np.float32)
        self.last_components: dict[str, float] = {}

    def _idx(self, names: list[str], name: str) -> int | None:
        return names.index(name) if name in names else None

    def _t(self, arr: np.ndarray, device: torch.device) -> torch.Tensor:
        return torch.as_tensor(arr, dtype=torch.float32, device=device)

    def _unscale_inputs(self, x_scaled: torch.Tensor) -> torch.Tensor:
        return x_scaled * self._t(self.x_std, x_scaled.device) + self._t(self.x_mean, x_scaled.device)

    def _unscale_outputs(self, y_scaled: torch.Tensor) -> torch.Tensor:
        return y_scaled * self._t(self.y_std, y_scaled.device) + self._t(self.y_mean, y_scaled.device)

    def _out(self, y_phys: torch.Tensor, name: str, fallback: int = -1) -> torch.Tensor:
        idx = self._idx(self.target_names, name)
        return y_phys[:, idx : idx + 1] if idx is not None else y_phys[:, fallback : fallback + 1]

    def _derivative_phys(
        self,
        y_scaled_col: torch.Tensor,
        x_scaled: torch.Tensor,
        y_name: str,
        x_name: str,
        extra_denominator: torch.Tensor | None = None,
    ) -> torch.Tensor:
        y_idx = self._idx(self.target_names, y_name)
        x_idx = self._idx(self.input_names, x_name)
        if y_idx is None or x_idx is None:
            return torch.zeros_like(y_scaled_col)
        d_scaled = grad(y_scaled_col, x_scaled)[:, x_idx : x_idx + 1]
        scale = float(self.y_std[y_idx]) / max(float(self.x_std[x_idx]), 1e-12)
        deriv = d_scaled * scale
        if extra_denominator is not None:
            deriv = deriv / extra_denominator.clamp_min(1e-12)
        return deriv

    def forward(self, model: PINN, x_scaled: torch.Tensor, y_true_scaled: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        x_scaled = x_scaled.requires_grad_(True)
        y_pred_scaled = model(x_scaled)
        loss_data = F.mse_loss(y_pred_scaled, y_true_scaled)
        x_phys = self._unscale_inputs(x_scaled)
        y_phys = self._unscale_outputs(y_pred_scaled)

        temp_idx = self._idx(self.target_names, "temperature")
        u_idx = self._idx(self.target_names, "u")
        rho_idx = self._idx(self.target_names, "density")
        if temp_idx is None or u_idx is None or rho_idx is None:
            zero = y_pred_scaled.sum() * 0
            return loss_data, {"data": float(loss_data.detach().cpu()), "continuity": 0.0, "energy": 0.0}

        temperature_scaled = y_pred_scaled[:, temp_idx : temp_idx + 1]
        u_phys = y_phys[:, u_idx : u_idx + 1]
        rho_phys = y_phys[:, rho_idx : rho_idx + 1].clamp_min(1e-6)

        dT_dx = self._derivative_phys(temperature_scaled, x_scaled, "temperature", "x")
        dT_dx_scaled = grad(temperature_scaled, x_scaled)[:, self._idx(self.input_names, "x") : self._idx(self.input_names, "x") + 1]
        x_idx = self._idx(self.input_names, "x")
        t_scale = float(self.y_std[temp_idx])
        x_scale = float(self.x_std[x_idx]) if x_idx is not None else 1.0
        d2T_dx2 = grad(dT_dx_scaled, x_scaled)[:, x_idx : x_idx + 1] * (t_scale / max(x_scale**2, 1e-12)) if x_idx is not None else torch.zeros_like(dT_dx)

        r_idx = self._idx(self.input_names, "r_over_R")
        dT_deta = self._derivative_phys(temperature_scaled, x_scaled, "temperature", "r_over_R")
        if r_idx is not None:
            dT_deta_scaled = grad(temperature_scaled, x_scaled)[:, r_idx : r_idx + 1]
            d2T_deta2 = grad(dT_deta_scaled, x_scaled)[:, r_idx : r_idx + 1] * (t_scale / max(float(self.x_std[r_idx]) ** 2, 1e-12))
        else:
            d2T_deta2 = torch.zeros_like(dT_dx)

        d_idx = self._idx(self.input_names, "diameter")
        radius = x_phys[:, d_idx : d_idx + 1].abs() * float(self.diameter_scale_to_meter) / 2.0 if d_idx is not None else torch.ones_like(dT_dx)
        radius = radius.clamp_min(1e-9)
        r_over_R = x_phys[:, r_idx : r_idx + 1].abs() if r_idx is not None else torch.ones_like(dT_dx)
        r_dim = (r_over_R * radius).clamp_min(radius * 1e-4)
        dT_dr = dT_deta / radius
        d2T_dr2 = d2T_deta2 / (radius**2)

        rho_u_scaled = y_pred_scaled[:, rho_idx : rho_idx + 1] * y_pred_scaled[:, u_idx : u_idx + 1]
        if x_idx is not None:
            d_rho_u_scaled = grad(rho_u_scaled, x_scaled)[:, x_idx : x_idx + 1]
            continuity = d_rho_u_scaled * (float(self.y_std[rho_idx] * self.y_std[u_idx]) / max(x_scale, 1e-12))
        else:
            continuity = torch.zeros_like(dT_dx)
        energy = rho_phys * self.cp_const * u_phys * dT_dx - self.k_const * (d2T_dx2 + d2T_dr2 + dT_dr / r_dim)

        x_coord = x_phys[:, x_idx : x_idx + 1] if x_idx is not None else torch.zeros_like(dT_dx)
        inlet_mask = x_coord <= torch.quantile(x_coord.detach(), 0.05)
        tin_idx = self._idx(self.input_names, "inlet_temperature")
        tin = x_phys[:, tin_idx : tin_idx + 1] if tin_idx is not None else torch.zeros_like(dT_dx)
        temp_phys = y_phys[:, temp_idx : temp_idx + 1]
        loss_inlet = F.mse_loss(temp_phys[inlet_mask], tin[inlet_mask]) if inlet_mask.any() else y_pred_scaled.sum() * 0

        wall_mask = r_over_R >= torch.quantile(r_over_R.detach(), 0.95)
        qw_idx = self._idx(self.input_names, "wall_heat_flux")
        qw = x_phys[:, qw_idx : qw_idx + 1] * float(self.heat_flux_scale_to_w_m2) if qw_idx is not None else torch.zeros_like(dT_dx)
        heat_flux_res = -self.k_const * dT_dr[wall_mask] - qw[wall_mask]
        loss_wall_q = F.mse_loss(heat_flux_res, torch.zeros_like(heat_flux_res)) if wall_mask.any() else y_pred_scaled.sum() * 0
        loss_wall_u = F.mse_loss(u_phys[wall_mask], torch.zeros_like(u_phys[wall_mask])) if wall_mask.any() else y_pred_scaled.sum() * 0
        loss_cont = F.mse_loss(continuity, torch.zeros_like(continuity))
        loss_energy = F.mse_loss(torch.nan_to_num(energy), torch.zeros_like(energy))

        weights = self.weights
        loss = (
            weights.get("data", 1.0) * loss_data
            + weights.get("continuity", 0.0) * loss_cont
            + weights.get("energy", 0.0) * loss_energy
            + weights.get("bc_inlet", 0.0) * loss_inlet
            + weights.get("bc_wall_heat_flux", 0.0) * loss_wall_q
            + weights.get("bc_wall_noslip", 0.0) * loss_wall_u
        )
        parts = {
            "data": float(loss_data.detach().cpu()),
            "continuity": float(loss_cont.detach().cpu()),
            "energy": float(loss_energy.detach().cpu()),
            "bc_inlet": float(loss_inlet.detach().cpu()),
            "bc_wall_heat_flux": float(loss_wall_q.detach().cpu()),
            "bc_wall_noslip": float(loss_wall_u.detach().cpu()),
        }
        self.last_components = parts
        return loss, parts


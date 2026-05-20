from __future__ import annotations

import torch
import torch.nn.functional as F

from .pinn import PINN


class RANSPINN(PINN):
    """PINN with optional auxiliary turbulent quantity outputs."""

    def __init__(self, input_dim: int = 5, base_output_dim: int = 4, auxiliary_names: list[str] | None = None, **kwargs) -> None:
        self.base_output_dim = base_output_dim
        self.auxiliary_names = auxiliary_names or []
        super().__init__(input_dim=input_dim, output_dim=base_output_dim + len(self.auxiliary_names), **kwargs)

    def base_outputs(self, pred: torch.Tensor) -> torch.Tensor:
        return pred[:, : self.base_output_dim]

    def auxiliary_loss(self, pred: torch.Tensor, aux_data: dict[str, torch.Tensor]) -> torch.Tensor:
        """Supervise configured auxiliary outputs only when data exist."""
        loss = pred.sum() * 0
        for i, name in enumerate(self.auxiliary_names):
            if name in aux_data:
                loss = loss + F.mse_loss(pred[:, self.base_output_dim + i : self.base_output_dim + i + 1], aux_data[name])
        return loss


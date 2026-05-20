from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import grouped_regression_metrics, regression_metrics
from .utils import model_output_dirs, save_json
from .visualization import plot_error_distribution, plot_parity


def evaluate_and_save(
    y_true_scaled: np.ndarray,
    y_pred_scaled: np.ndarray,
    y_scaler,
    target_names: list[str],
    model_name: str,
    output_dir: str | Path,
    extra: dict | None = None,
    meta: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Inverse-transform predictions, compute metrics, and save artifacts."""
    output_dir = Path(output_dir)
    dirs = model_output_dirs(model_name, output_dir)
    y_true = y_scaler.inverse_transform(y_true_scaled)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    metrics = regression_metrics(y_true, y_pred, target_names)
    grouped = {}
    if meta is not None and not meta.empty:
        for col in ["case_id", "diameter", "wall_heat_flux", "inlet_temperature"]:
            if col in meta.columns:
                grouped[col] = grouped_regression_metrics(y_true, y_pred, target_names, meta[col].to_numpy(), col)
    if extra:
        metrics.update(extra)
    if grouped:
        metrics["grouped_metrics"] = grouped

    save_json(metrics, dirs["metrics"] / f"{model_name}_metrics.json")
    pred_df = meta.copy().reset_index(drop=True) if meta is not None else pd.DataFrame()
    for i, c in enumerate(target_names):
        pred_df[f"true_{c}"] = y_true[:, i]
    for i, c in enumerate(target_names):
        pred_df[f"pred_{c}"] = y_pred[:, i]
        pred_df[f"error_{c}"] = y_pred[:, i] - y_true[:, i]
    pred_df.to_csv(dirs["predictions"] / f"{model_name}_predictions.csv", index=False)
    plot_parity(y_true, y_pred, target_names, dirs["figures"], model_name)
    plot_error_distribution(y_true, y_pred, target_names, dirs["figures"], model_name)
    return metrics

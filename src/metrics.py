from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_names: list[str]) -> dict[str, float]:
    """Compute per-target and aggregate regression metrics."""
    metrics: dict[str, float] = {}
    eps = 1e-12
    for i, name in enumerate(target_names):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        metrics[f"MAE_{name}"] = float(mean_absolute_error(yt, yp))
        metrics[f"RMSE_{name}"] = float(np.sqrt(mean_squared_error(yt, yp)))
        metrics[f"R2_{name}"] = float(r2_score(yt, yp)) if len(np.unique(yt)) > 1 else float("nan")
        metrics[f"MAPE_{name}"] = float(np.mean(np.abs((yt - yp) / (np.abs(yt) + eps))) * 100.0)
    metrics["relative_L2"] = float(np.linalg.norm(y_true - y_pred) / (np.linalg.norm(y_true) + eps))
    return metrics


def grouped_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str],
    groups,
    prefix: str,
) -> dict[str, dict[str, float]]:
    """Compute metrics per group value."""
    result: dict[str, dict[str, float]] = {}
    groups = np.asarray(groups)
    for value in sorted(group for group in np.unique(groups) if group == group):
        idx = groups == value
        if idx.sum() < 2:
            continue
        result[f"{prefix}_{value}"] = regression_metrics(y_true[idx], y_pred[idx], target_names)
    return result


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    """Compute HTD classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid", context="paper")


def _save(fig: plt.Figure, out_dir: str | Path, name: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", dpi=300)
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)


def plot_case_matrix(summary: pd.DataFrame, out_dir: str | Path) -> None:
    """Plot inlet temperature, wall heat flux, and diameter coverage."""
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        data=summary,
        x="inlet_temperature",
        y="wall_heat_flux",
        hue="diameter",
        style="diameter",
        s=70,
        ax=ax,
    )
    ax.set_title("Case Matrix")
    ax.set_xlabel("Inlet temperature")
    ax.set_ylabel("Wall heat flux")
    _save(fig, out_dir, "case_matrix")


def plot_wall_curve(
    wall: pd.DataFrame,
    y_col: str,
    out_dir: str | Path,
    name: str,
    ylabel: str | None = None,
    max_cases: int = 8,
) -> None:
    """Plot wall axial curves for a small representative sample of cases."""
    if wall.empty or y_col not in wall.columns:
        return
    cases = list(wall["case_id"].drop_duplicates().head(max_cases))
    data = wall[wall["case_id"].isin(cases)]
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.lineplot(data=data, x="x", y=y_col, hue="case_id", units="wall_side", estimator=None, alpha=0.9, ax=ax)
    ax.set_title(f"{ylabel or y_col} along x")
    ax.set_xlabel("x")
    ax.set_ylabel(ylabel or y_col)
    ax.legend(loc="best", fontsize=7)
    _save(fig, out_dir, name)


def plot_cloud(
    full: pd.DataFrame,
    value_col: str,
    out_dir: str | Path,
    name: str,
    case_id: str | None = None,
    max_points: int = 120_000,
) -> None:
    """Plot x-r cloud/contour-like scatter for one case."""
    if value_col not in full.columns or "r_over_R" not in full.columns:
        return
    if case_id is None:
        case_id = str(full["case_id"].iloc[0])
    data = full[full["case_id"] == case_id]
    if len(data) > max_points:
        data = data.sample(max_points, random_state=42)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sc = ax.scatter(data["x"], data["r_over_R"], c=data[value_col], s=4, cmap="viridis", linewidths=0)
    fig.colorbar(sc, ax=ax, label=value_col)
    ax.set_title(f"{value_col} cloud ({case_id})")
    ax.set_xlabel("x")
    ax.set_ylabel("r/R")
    _save(fig, out_dir, name)


def plot_htd_region(full: pd.DataFrame, out_dir: str | Path, case_id: str | None = None) -> None:
    """Plot HTD labels in x-r space for one case."""
    if "HTD_label" not in full.columns:
        return
    if case_id is None:
        case_id = str(full["case_id"].iloc[0])
    data = full[full["case_id"] == case_id]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sc = ax.scatter(data["x"], data["r_over_R"], c=data["HTD_label"], s=4, cmap="coolwarm", vmin=0, vmax=2)
    fig.colorbar(sc, ax=ax, label="HTD label")
    ax.set_title(f"HTD label region ({case_id})")
    ax.set_xlabel("x")
    ax.set_ylabel("r/R")
    _save(fig, out_dir, f"htd_region_{case_id}")


def plot_correlation_heatmap(df: pd.DataFrame, columns: list[str], out_dir: str | Path) -> None:
    """Plot Pearson and Spearman correlation heatmaps."""
    cols = [c for c in columns if c in df.columns]
    if len(cols) < 2:
        return
    data = df[cols].dropna()
    if data.empty:
        return
    for method in ["pearson", "spearman"]:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(data.corr(method=method), cmap="vlag", center=0, ax=ax)
        ax.set_title(f"{method.title()} Correlation")
        _save(fig, out_dir, f"{method}_correlation_heatmap")


def plot_loss_curve(history: dict[str, list[float]], out_dir: str | Path, name: str) -> None:
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key, values in history.items():
        ax.plot(values, label=key)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.set_title(f"{name} Loss Curve")
    ax.legend()
    _save(fig, out_dir, f"{name}_loss_curve")


def plot_parity(y_true, y_pred, target_names: list[str], out_dir: str | Path, name: str) -> None:
    """Plot predicted-vs-true parity plots."""
    n = len(target_names)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), squeeze=False)
    for i, target in enumerate(target_names):
        ax = axes[0, i]
        ax.scatter(y_true[:, i], y_pred[:, i], s=5, alpha=0.35)
        lo = min(y_true[:, i].min(), y_pred[:, i].min())
        hi = max(y_true[:, i].max(), y_pred[:, i].max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_title(target)
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
    _save(fig, out_dir, f"{name}_parity_plot")


def plot_error_distribution(y_true, y_pred, target_names: list[str], out_dir: str | Path, name: str) -> None:
    """Plot error distributions for each target."""
    errors = y_pred - y_true
    fig, axes = plt.subplots(1, len(target_names), figsize=(5 * len(target_names), 4.5), squeeze=False)
    for i, target in enumerate(target_names):
        sns.histplot(errors[:, i], bins=60, kde=True, ax=axes[0, i])
        axes[0, i].set_title(f"{target} error")
    _save(fig, out_dir, f"{name}_error_distribution")


def plot_feature_importance(importances: pd.Series, out_dir: str | Path, name: str) -> None:
    """Plot tree-model feature importances."""
    if importances.empty:
        return
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(importances))))
    importances.sort_values().plot(kind="barh", ax=ax)
    ax.set_title(f"{name} Feature Importance")
    ax.set_xlabel("Importance")
    _save(fig, out_dir, f"{name}_feature_importance")


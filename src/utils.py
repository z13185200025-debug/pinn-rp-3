from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml


def project_root() -> Path:
    """Return the repository root inferred from this file location."""
    return Path(__file__).resolve().parents[1]


def resolve_project_path(path: str | Path | None, base: str | Path | None = None) -> Path | None:
    """Resolve a path relative to the project root unless it is already absolute."""
    if path is None:
        return None
    path = Path(path)
    if path.is_absolute():
        return path
    root = Path(base) if base is not None else project_root()
    return (root / path).resolve()


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary."""
    path = resolve_project_path(path) or Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """Save JSON with stable formatting."""
    path = resolve_project_path(path) or Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def ensure_project_dirs(base: str | Path = ".") -> None:
    """Create the standard output and data directories."""
    base = resolve_project_path(base) or project_root()
    for rel in [
        "data/raw",
        "data/interim",
        "data/processed",
        "outputs/figures",
        "outputs/models",
        "outputs/metrics",
        "outputs/predictions",
        "outputs/reports",
    ]:
        (base / rel).mkdir(parents=True, exist_ok=True)


def model_output_dirs(model_name: str, output_dir: str | Path = "outputs") -> dict[str, Path]:
    """Return and create per-model output directories.

    Artifacts are organized as outputs/<category>/<model_name>/... so figures,
    checkpoints, metrics, and predictions from different models do not mix.
    """
    output_dir = resolve_project_path(output_dir) or Path(output_dir)
    dirs = {
        "figures": output_dir / "figures" / model_name,
        "models": output_dir / "models" / model_name,
        "metrics": output_dir / "metrics" / model_name,
        "predictions": output_dir / "predictions" / model_name,
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def set_seed(seed: int) -> None:
    """Fix common random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device(device: str = "auto") -> torch.device:
    """Resolve a PyTorch device from config."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def save_dataframe(df: pd.DataFrame, path: str | Path, fmt: str = "parquet") -> Path:
    """Save a dataframe as parquet or csv, falling back to csv if parquet fails."""
    path = resolve_project_path(path) or Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        target = path.with_suffix(".parquet")
        try:
            df.to_parquet(target, index=False)
            return target
        except Exception as exc:
            print(f"[WARN] Parquet save failed for {target}: {exc}. Falling back to CSV.")
    target = path.with_suffix(".csv")
    df.to_csv(target, index=False)
    return target


def read_table(path: str | Path) -> pd.DataFrame:
    """Read parquet or csv by extension."""
    path = resolve_project_path(path) or Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def available_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    """Return columns that are present in the dataframe."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        print(f"[WARN] Missing columns skipped: {missing}")
    return [c for c in columns if c in df.columns]


def write_text(path: str | Path, text: str) -> None:
    """Write text under the project root by default."""
    path = resolve_project_path(path) or Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

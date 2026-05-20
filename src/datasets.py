from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .utils import resolve_project_path, save_json


@dataclass
class SplitData:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    train_cases: list[str]
    val_cases: list[str]
    test_cases: list[str]


class TabularDataset(Dataset):
    """PyTorch dataset for tabular regression."""

    def __init__(self, x: np.ndarray, y: np.ndarray, meta: pd.DataFrame | None = None):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.meta = meta.reset_index(drop=True) if meta is not None else None

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class GridDataset(Dataset):
    """Regular x-r grid dataset for CNN/FNO models."""

    def __init__(self, x: np.ndarray, y: np.ndarray, mask: np.ndarray):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.mask = torch.as_tensor(mask, dtype=torch.float32)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"x": self.x[idx], "y": self.y[idx], "mask": self.mask[idx]}


class WallSequenceDataset(Dataset):
    """Padded wall sequence dataset for RNN/TCN/Transformer models."""

    def __init__(self, x: np.ndarray, y: np.ndarray, mask: np.ndarray):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.mask = torch.as_tensor(mask, dtype=torch.float32)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"x": self.x[idx], "y": self.y[idx], "mask": self.mask[idx]}


def _split_cases(cases: np.ndarray, test_size: float, val_size: float, seed: int) -> tuple[list[str], list[str], list[str]]:
    if len(cases) < 3:
        return list(cases), list(cases[:0]), list(cases[:0])
    if test_size <= 0:
        train_cases, val_cases = train_test_split(cases, test_size=min(0.5, max(val_size, 1 / len(cases))), random_state=seed)
        return list(train_cases), list(val_cases), []
    train_val_cases, test_cases = train_test_split(cases, test_size=test_size, random_state=seed)
    if len(train_val_cases) < 2 or val_size <= 0:
        return list(train_val_cases), [], list(test_cases)
    val_fraction = min(0.5, val_size / max(1e-9, 1 - test_size))
    train_cases, val_cases = train_test_split(train_val_cases, test_size=val_fraction, random_state=seed)
    return list(train_cases), list(val_cases), list(test_cases)


def _save_manifest(split: SplitData, split_config: dict[str, Any]) -> None:
    path = split_config.get("manifest_path")
    if not path:
        return
    save_json(
        {
            "split_method": split_config.get("split_method", "case_split"),
            "train_cases": split.train_cases,
            "val_cases": split.val_cases,
            "test_cases": split.test_cases,
        },
        path,
    )


def split_dataframe(df: pd.DataFrame, split_config: dict, holdout_value: float | None = None) -> SplitData:
    """Split data by point, case, diameter, heat flux, or inlet temperature."""
    method = split_config.get("split_method", "case_split")
    test_size = float(split_config.get("test_size", 0.2))
    val_size = float(split_config.get("val_size", 0.1))
    seed = int(split_config.get("random_seed", 42))

    if method == "point_random_split":
        train_val, test = train_test_split(df, test_size=test_size, random_state=seed)
        train, val = train_test_split(train_val, test_size=val_size / max(1e-9, 1 - test_size), random_state=seed)
        split = SplitData(
            train.reset_index(drop=True),
            val.reset_index(drop=True),
            test.reset_index(drop=True),
            sorted(train["case_id"].unique()) if "case_id" in train else [],
            sorted(val["case_id"].unique()) if "case_id" in val else [],
            sorted(test["case_id"].unique()) if "case_id" in test else [],
        )
        _save_manifest(split, split_config)
        return split

    if method == "case_split":
        cases = np.array(sorted(df["case_id"].dropna().unique()))
        train_cases, val_cases, test_cases = _split_cases(cases, test_size, val_size, seed)
        split = SplitData(
            df[df["case_id"].isin(train_cases)].reset_index(drop=True),
            df[df["case_id"].isin(val_cases)].reset_index(drop=True),
            df[df["case_id"].isin(test_cases)].reset_index(drop=True),
            train_cases,
            val_cases,
            test_cases,
        )
        _save_manifest(split, split_config)
        return split

    holdout_map = {
        "leave_one_diameter_out": "diameter",
        "leave_one_heat_flux_out": "wall_heat_flux",
        "leave_one_inlet_temperature_out": "inlet_temperature",
    }
    if method in holdout_map:
        col = holdout_map[method]
        configured = split_config.get("holdout_value")
        value = holdout_value if holdout_value is not None else configured
        value = sorted(df[col].dropna().unique())[-1] if value in (None, "") else float(value)
        test = df[df[col] == value]
        train_val = df[df[col] != value]
        cases = np.array(sorted(train_val["case_id"].dropna().unique()))
        train_cases, val_cases, _ = _split_cases(cases, test_size=0.0, val_size=val_size, seed=seed)
        test_cases = sorted(test["case_id"].dropna().unique())
        split = SplitData(
            train_val[train_val["case_id"].isin(train_cases)].reset_index(drop=True),
            train_val[train_val["case_id"].isin(val_cases)].reset_index(drop=True),
            test.reset_index(drop=True),
            train_cases,
            val_cases,
            test_cases,
        )
        _save_manifest(split, split_config)
        return split

    raise ValueError(f"Unsupported split_method: {method}")


def prepare_arrays(
    split: SplitData,
    input_features: list[str],
    target_features: list[str],
    scaler_dir: str | Path,
    model_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, StandardScaler, list[str], list[str], pd.DataFrame]:
    """Create scaled arrays and persist scalers fit only on train data."""
    scaler_dir = resolve_project_path(scaler_dir) or Path(scaler_dir)
    scaler_dir.mkdir(parents=True, exist_ok=True)
    inputs = [c for c in input_features if c in split.train.columns]
    targets = [c for c in target_features if c in split.train.columns]
    if not inputs or not targets:
        raise ValueError(f"No usable inputs or targets. inputs={inputs}, targets={targets}")

    meta_cols = [c for c in ["case_id", "x", "x_over_L", "r_over_R", "r_signed", "diameter", "inlet_temperature", "wall_heat_flux"] if c in split.test.columns]
    train_cols = list(dict.fromkeys(inputs + targets))
    test_cols = list(dict.fromkeys(meta_cols + inputs + targets))
    train = split.train[train_cols].replace([np.inf, -np.inf], np.nan).dropna()
    val = split.val[train_cols].replace([np.inf, -np.inf], np.nan).dropna()
    test_src = split.test[test_cols].replace([np.inf, -np.inf], np.nan)
    test = test_src.dropna(subset=inputs + targets)
    if train.empty:
        raise ValueError("Training split is empty after dropping NaN/Inf rows.")
    if val.empty:
        val = train.sample(min(len(train), 1024), random_state=42)
    if test.empty:
        test = val.copy()

    x_scaler = StandardScaler().fit(train[inputs])
    y_scaler = StandardScaler().fit(train[targets])
    joblib.dump(x_scaler, scaler_dir / f"{model_name}_x_scaler.joblib")
    joblib.dump(y_scaler, scaler_dir / f"{model_name}_y_scaler.joblib")
    save_json(
        {
            "input_features": inputs,
            "target_features": targets,
            "train_cases": split.train_cases,
            "val_cases": split.val_cases,
            "test_cases": split.test_cases,
        },
        scaler_dir / f"{model_name}_metadata.json",
    )

    def xy(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return x_scaler.transform(frame[inputs]), y_scaler.transform(frame[targets])

    return (*xy(train), *xy(val), *xy(test), x_scaler, y_scaler, inputs, targets, test[meta_cols].reset_index(drop=True))


def make_loaders(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    sample_weights: np.ndarray | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch dataloaders from arrays."""
    sampler = None
    shuffle = True
    if sample_weights is not None:
        sampler = WeightedRandomSampler(torch.as_tensor(sample_weights, dtype=torch.double), len(sample_weights), replacement=True)
        shuffle = False
    train_loader = DataLoader(
        TabularDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(TabularDataset(x_val, y_val), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(TabularDataset(x_test, y_test), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader


def load_grid_dataset(path: str | Path) -> GridDataset:
    """Load a saved grid_dataset.npz."""
    path = resolve_project_path(path) or Path(path)
    data = np.load(path)
    return GridDataset(data["x"], data["y"], data["mask"])


def load_sequence_dataset(path: str | Path) -> WallSequenceDataset:
    """Load a saved wall_sequence_dataset.npz."""
    path = resolve_project_path(path) or Path(path)
    data = np.load(path)
    return WallSequenceDataset(data["x"], data["y"], data["mask"])

from __future__ import annotations

import re

import numpy as np
import pandas as pd


COLUMN_RENAME = {
    "x-coordinate": "x",
    "y-coordinate": "y",
    "z-coordinate": "z",
    "x-velocity": "u",
    "y-velocity": "v",
    "z-velocity": "w",
    "velocity-magnitude": "velocity_magnitude",
    "wall-temperature": "wall_temperature",
    "wall-adjacent-temperature": "wall_adjacent_temperature",
    "nusselt-number": "nusselt_number",
    "dynamic-pressure": "dynamic_pressure",
    "cell-reynolds-number": "cell_reynolds_number",
    "turb-reynolds-number-rey": "turb_reynolds_number",
    "prandtl-number-lam": "prandtl_number_lam",
    "z-vorticity": "omega_z",
    "x-vorticity": "omega_x",
    "y-vorticity": "omega_y",
    "total-enthalpy": "total_enthalpy",
    "internal-energy": "internal_energy",
    "density-all": "density_all",
    "total-temperature": "total_temperature",
    "pressure-coefficient": "pressure_coefficient",
    "absolute-pressure": "absolute_pressure",
}


def snake_case(name: str) -> str:
    """Convert a CFD export column name to snake_case."""
    name = name.strip()
    name = COLUMN_RENAME.get(name, name)
    name = re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_")
    return name.lower()


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to Python-friendly names and drop duplicate columns."""
    df = df.rename(columns={col: snake_case(str(col)) for col in df.columns})
    df = df.loc[:, ~df.columns.duplicated()].copy()
    return df


def coerce_numeric(df: pd.DataFrame, exclude: tuple[str, ...] = ("case_id", "source_file")) -> pd.DataFrame:
    """Convert non-metadata columns to numeric values where possible."""
    for col in df.columns:
        if col not in exclude:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def clean_dataframe(df: pd.DataFrame, ignored_features: list[str] | None = None) -> pd.DataFrame:
    """Standardize names, coerce numeric fields, drop empty rows, and ignore unused columns."""
    df = standardize_columns(df)
    df = coerce_numeric(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=0, how="all")
    ignored = {snake_case(c) for c in (ignored_features or [])}
    drop_cols = [c for c in ignored if c in df.columns]
    return df.drop(columns=drop_cols, errors="ignore")


def warn_missing_columns(df: pd.DataFrame, required: list[str], context: str = "data") -> None:
    """Print a friendly warning for missing required columns."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[WARN] {context} is missing columns: {missing}")


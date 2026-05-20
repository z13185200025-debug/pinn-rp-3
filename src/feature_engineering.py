from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import resolve_project_path, save_json


RADIAL_CANDIDATES = ["radial", "r", "y", "z"]


def _safe_mean_positive(series: pd.Series) -> float:
    vals = series[series > 0]
    if vals.empty:
        return float(series.mean()) if len(series) else np.nan
    return float(vals.mean())


def _gradient_by_group(
    df: pd.DataFrame,
    value_col: str,
    coord_col: str,
    group_cols: list[str],
    out_col: str,
) -> pd.Series:
    """Compute a robust first derivative inside sorted groups."""
    out = pd.Series(np.zeros(len(df), dtype=float), index=df.index)
    if value_col not in df.columns or coord_col not in df.columns or df.empty:
        out.name = out_col
        return out
    for _, idx in df.groupby(group_cols, sort=False).groups.items():
        sub = df.loc[idx, [coord_col, value_col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(sub) < 2 or sub[coord_col].nunique() < 2:
            continue
        sub = sub.groupby(coord_col, as_index=False)[value_col].mean().sort_values(coord_col)
        coords = sub[coord_col].to_numpy(dtype=float)
        values = sub[value_col].to_numpy(dtype=float)
        if np.any(np.diff(coords) <= 0):
            continue
        grad = np.gradient(values, coords)
        out.loc[idx] = np.interp(df.loc[idx, coord_col].to_numpy(dtype=float), coords, grad)
    out.name = out_col
    return out


def _choose_case_radial(case_df: pd.DataFrame, requested: str = "auto") -> tuple[str, float]:
    """Choose the radial coordinate column for one case."""
    if requested != "auto" and requested in case_df.columns:
        vals = case_df[requested].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        return requested, float(vals.abs().max()) if len(vals) else np.nan
    scores: dict[str, float] = {}
    for col in RADIAL_CANDIDATES:
        if col not in case_df.columns:
            continue
        vals = case_df[col].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        if vals.empty:
            continue
        scores[col] = float(vals.max() - vals.min())
    if not scores:
        raise ValueError("No usable radial coordinate found. Expected one of y, z, r, radial.")
    chosen = max(scores, key=scores.get)
    observed_radius = float(case_df[chosen].astype(float).abs().max())
    return chosen, observed_radius


def add_geometry_features(
    df: pd.DataFrame,
    diameter_scale_to_meter: float = 1.0,
    radial_coordinate: str = "auto",
    axial_coordinate: str = "x",
    heat_flux_scale_to_w_m2: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add robust axial/radial coordinates and geometry audit columns."""
    df = df.copy()
    if axial_coordinate not in df.columns:
        raise ValueError(f"Configured axial_coordinate '{axial_coordinate}' not found in columns.")
    if axial_coordinate != "x":
        df["x"] = df[axial_coordinate]

    frames: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    for case_id, case_df in df.groupby("case_id", sort=False):
        case_df = case_df.copy()
        radial_name, observed_radius = _choose_case_radial(case_df, radial_coordinate)
        diameter_physical = float(case_df["diameter"].iloc[0]) * float(diameter_scale_to_meter)
        expected_radius = diameter_physical / 2.0 if diameter_physical else np.nan
        denom = observed_radius if np.isfinite(observed_radius) and observed_radius > 0 else expected_radius
        if not np.isfinite(denom) or denom == 0:
            denom = 1.0
        x_min = float(case_df["x"].min())
        x_max = float(case_df["x"].max())
        x_span = x_max - x_min if x_max > x_min else 1.0
        case_df["radial_coord_name"] = radial_name
        case_df["r_signed"] = case_df[radial_name].astype(float)
        case_df["r_abs"] = case_df["r_signed"].abs()
        case_df["observed_radius"] = observed_radius
        case_df["r_wall"] = denom
        case_df["r_over_R"] = (case_df["r_abs"] / denom).clip(lower=0)
        case_df["x_over_L"] = (case_df["x"] - x_min) / x_span
        case_df["diameter_physical"] = diameter_physical
        case_df["wall_heat_flux_w_m2"] = case_df["wall_heat_flux"] * float(heat_flux_scale_to_w_m2)

        rel_error = (
            abs(observed_radius - expected_radius) / max(abs(expected_radius), 1e-12)
            if np.isfinite(observed_radius) and np.isfinite(expected_radius)
            else np.nan
        )
        if np.isfinite(rel_error) and rel_error > 0.25:
            print(
                f"[WARN] Geometry mismatch for {case_id}: observed_radius={observed_radius:.6g}, "
                f"diameter_physical/2={expected_radius:.6g}, rel_error={rel_error:.3g}. "
                "Using observed radius for r_over_R."
            )
        audits.append(
            {
                "case_id": case_id,
                "radial_coord_name": radial_name,
                "observed_radius": observed_radius,
                "diameter_physical": diameter_physical,
                "expected_radius_from_diameter": expected_radius,
                "relative_radius_error": rel_error,
            }
        )
        frames.append(case_df)
    return pd.concat(frames, ignore_index=True), pd.DataFrame(audits)


def add_physical_features(
    df: pd.DataFrame,
    diameter_scale_to_meter: float = 1.0,
    htd_config: dict[str, Any] | None = None,
    radial_coordinate: str = "auto",
    axial_coordinate: str = "x",
    heat_flux_scale_to_w_m2: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add geometry, thermal, derivative, and placeholder HTD label features."""
    df = df.copy()
    required = ["case_id", "diameter", axial_coordinate]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Cannot engineer features without columns: {missing}")
    df, audit = add_geometry_features(
        df,
        diameter_scale_to_meter=diameter_scale_to_meter,
        radial_coordinate=radial_coordinate,
        axial_coordinate=axial_coordinate,
        heat_flux_scale_to_w_m2=heat_flux_scale_to_w_m2,
    )

    if "temperature" in df.columns:
        df["bulk_temperature"] = df.groupby(["case_id", "x"], sort=False)["temperature"].transform("mean")
    else:
        df["bulk_temperature"] = np.nan
    if "wall_temperature" in df.columns:
        df["Tw_minus_Tb"] = df["wall_temperature"] - df["bulk_temperature"]
    else:
        df["Tw_minus_Tb"] = np.nan

    if "nusselt_number" in df.columns:
        nu_mean = df.groupby("case_id", sort=False)["nusselt_number"].transform(_safe_mean_positive)
        df["Nu_normalized"] = df["nusselt_number"] / nu_mean.replace(0, np.nan)
    else:
        df["Nu_normalized"] = np.nan

    df["r_bin"] = df["r_over_R"].round(3)
    df["x_bin"] = df["x"].round(9)
    df["temperature_gradient_x"] = _gradient_by_group(
        df, "temperature", "x", ["case_id", "r_bin"], "temperature_gradient_x"
    )
    df["temperature_gradient_r"] = _gradient_by_group(
        df, "temperature", "r_abs", ["case_id", "x_bin"], "temperature_gradient_r"
    )
    df["density_gradient_x"] = _gradient_by_group(df, "density", "x", ["case_id", "r_bin"], "density_gradient_x")
    df["density_gradient_r"] = _gradient_by_group(df, "density", "r_abs", ["case_id", "x_bin"], "density_gradient_r")
    df["velocity_gradient_r"] = _gradient_by_group(df, "u", "r_abs", ["case_id", "x_bin"], "velocity_gradient_r")
    df.drop(columns=["r_bin", "x_bin"], inplace=True, errors="ignore")
    df["HTD_label"] = 0
    return df, audit


def create_wall_dataset(df: pd.DataFrame, wall_tol: float = 0.02, htd_config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Build a near-wall dataset with axial derivatives and HTD labels."""
    htd_config = htd_config or {}
    wall_mask = pd.Series(False, index=df.index)
    if "nusselt_number" in df.columns:
        wall_mask |= df["nusselt_number"].fillna(0) > 0
    if "r_over_R" in df.columns:
        wall_mask |= df["r_over_R"].abs() >= (1.0 - wall_tol)
    if wall_mask.sum() == 0 and "wall_temperature" in df.columns:
        wall_mask |= df["wall_temperature"].fillna(0) > 0

    wall = df.loc[wall_mask].copy()
    if wall.empty:
        return wall
    wall["wall_side"] = np.where(wall["r_signed"] >= 0, "positive_r", "negative_r")

    agg: dict[str, str] = {
        "diameter": "first",
        "diameter_physical": "first",
        "inlet_temperature": "first",
        "wall_heat_flux": "first",
        "wall_heat_flux_w_m2": "first",
        "x_over_L": "mean",
        "r_signed": "mean",
        "r_abs": "mean",
        "r_over_R": "mean",
        "r_wall": "mean",
        "bulk_temperature": "mean",
        "wall_temperature": "mean",
        "wall_adjacent_temperature": "mean",
        "nusselt_number": "mean",
        "Tw_minus_Tb": "mean",
        "Nu_normalized": "mean",
        "u": "mean",
        "density": "mean",
        "turb_reynolds_number": "mean",
        "prandtl_number_lam": "mean",
        "omega_z": "mean",
        "pressure": "mean",
    }
    agg = {k: v for k, v in agg.items() if k in wall.columns}
    wall = wall.groupby(["case_id", "wall_side", "x"], as_index=False).agg(agg)
    wall = wall.sort_values(["case_id", "wall_side", "x"]).reset_index(drop=True)

    wall["dTw_dx"] = _gradient_by_group(wall, "wall_temperature", "x", ["case_id", "wall_side"], "dTw_dx")
    wall["dNu_dx"] = _gradient_by_group(wall, "nusselt_number", "x", ["case_id", "wall_side"], "dNu_dx")
    for src, dst in [
        ("u", "u_near_wall"),
        ("density", "density_near_wall"),
        ("turb_reynolds_number", "turb_reynolds_number_near_wall"),
        ("prandtl_number_lam", "prandtl_number_lam_near_wall"),
        ("omega_z", "omega_z_near_wall"),
    ]:
        if src in wall.columns:
            wall[dst] = wall[src]

    wall["HTD_label"] = 0
    for case_id, idx in wall.groupby("case_id", sort=False).groups.items():
        sub = wall.loc[idx]
        if len(sub) < 4 or "dTw_dx" not in sub or "dNu_dx" not in sub:
            continue
        q75 = sub["dTw_dx"].replace([np.inf, -np.inf], np.nan).quantile(htd_config.get("suspicious_dTw_dx_quantile", 0.75))
        q90 = sub["dTw_dx"].replace([np.inf, -np.inf], np.nan).quantile(htd_config.get("severe_dTw_dx_quantile", 0.90))
        q25 = sub["dNu_dx"].replace([np.inf, -np.inf], np.nan).quantile(htd_config.get("suspicious_dNu_dx_quantile", 0.25))
        severe_nu = htd_config.get("severe_Nu_normalized_threshold", 0.70)
        suspicious = (sub["dTw_dx"] > q75) | (sub["dNu_dx"] < q25)
        severe = (sub["dTw_dx"] > q90) & (sub["Nu_normalized"].fillna(1.0) < severe_nu)
        wall.loc[idx, "HTD_label"] = np.select([severe, suspicious], [2, 1], default=0)
    return wall


def attach_wall_labels_to_field(full: pd.DataFrame, wall: pd.DataFrame) -> pd.DataFrame:
    """Propagate wall HTD labels to matching case/x locations in the full field."""
    if wall.empty:
        return full
    label = wall.groupby(["case_id", "x"], as_index=False)["HTD_label"].max()
    full = full.drop(columns=["HTD_label"], errors="ignore").merge(label, on=["case_id", "x"], how="left")
    full["HTD_label"] = full["HTD_label"].fillna(0).astype(int)
    return full


def create_case_summary_dataset(full: pd.DataFrame, wall: pd.DataFrame) -> pd.DataFrame:
    """Create one-row-per-case summary features for risk analysis."""
    rows: list[dict[str, float | str]] = []
    for case_id, case_df in full.groupby("case_id", sort=True):
        w = wall[wall["case_id"] == case_id] if not wall.empty else wall
        row: dict[str, float | str] = {
            "case_id": case_id,
            "diameter": float(case_df["diameter"].iloc[0]),
            "diameter_physical": float(case_df["diameter_physical"].iloc[0]) if "diameter_physical" in case_df else np.nan,
            "inlet_temperature": float(case_df["inlet_temperature"].iloc[0]),
            "wall_heat_flux": float(case_df["wall_heat_flux"].iloc[0]),
            "wall_heat_flux_w_m2": float(case_df["wall_heat_flux_w_m2"].iloc[0]) if "wall_heat_flux_w_m2" in case_df else np.nan,
            "radial_coord_name": str(case_df["radial_coord_name"].iloc[0]) if "radial_coord_name" in case_df else "",
            "observed_radius": float(case_df["observed_radius"].iloc[0]) if "observed_radius" in case_df else np.nan,
            "mean_density": float(case_df["density"].mean()) if "density" in case_df else np.nan,
            "min_density": float(case_df["density"].min()) if "density" in case_df else np.nan,
            "mean_prandtl_number": float(case_df["prandtl_number_lam"].mean()) if "prandtl_number_lam" in case_df else np.nan,
            "mean_turb_reynolds_number": float(case_df["turb_reynolds_number"].mean()) if "turb_reynolds_number" in case_df else np.nan,
        }
        if "pressure" in case_df:
            x_min = case_df["x"].min()
            x_max = case_df["x"].max()
            p_in = case_df.loc[case_df["x"] == x_min, "pressure"].mean()
            p_out = case_df.loc[case_df["x"] == x_max, "pressure"].mean()
            row["pressure_drop"] = float(p_in - p_out)
        else:
            row["pressure_drop"] = np.nan
        if not w.empty:
            row.update(
                {
                    "max_wall_temperature": float(w["wall_temperature"].max()) if "wall_temperature" in w else np.nan,
                    "mean_wall_temperature": float(w["wall_temperature"].mean()) if "wall_temperature" in w else np.nan,
                    "max_Tw_minus_Tb": float(w["Tw_minus_Tb"].max()) if "Tw_minus_Tb" in w else np.nan,
                    "min_nusselt_number": float(w["nusselt_number"].min()) if "nusselt_number" in w else np.nan,
                    "mean_nusselt_number": float(w["nusselt_number"].mean()) if "nusselt_number" in w else np.nan,
                    "max_dTw_dx": float(w["dTw_dx"].max()) if "dTw_dx" in w else np.nan,
                    "min_dNu_dx": float(w["dNu_dx"].min()) if "dNu_dx" in w else np.nan,
                    "htd_ratio": float((w["HTD_label"] >= 1).mean()) if "HTD_label" in w else np.nan,
                    "htd_severe_ratio": float((w["HTD_label"] == 2).mean()) if "HTD_label" in w else np.nan,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def create_grid_arrays(
    full: pd.DataFrame,
    input_cols: list[str],
    target_cols: list[str],
    nx: int,
    nr: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Create regular x-r grids per case for CNN/FNO training."""
    cases = list(full["case_id"].drop_duplicates())
    inputs = [c for c in input_cols if c in full.columns]
    targets = [c for c in target_cols if c in full.columns]
    x_grid = np.linspace(0.0, 1.0, nx)
    r_grid = np.linspace(0.0, 1.0, nr)
    x_arr = np.zeros((len(cases), len(inputs), nx, nr), dtype=np.float32)
    y_arr = np.zeros((len(cases), len(targets), nx, nr), dtype=np.float32)
    mask = np.zeros((len(cases), 1, nx, nr), dtype=np.float32)
    params = np.zeros((len(cases), 3), dtype=np.float32)

    for ci, case_id in enumerate(cases):
        case = full[full["case_id"] == case_id].copy()
        params[ci] = [
            float(case["diameter"].iloc[0]),
            float(case["inlet_temperature"].iloc[0]),
            float(case["wall_heat_flux"].iloc[0]),
        ]
        xi = np.clip(np.searchsorted(x_grid, case["x_over_L"].to_numpy(), side="left"), 0, nx - 1)
        ri = np.clip(np.searchsorted(r_grid, case["r_over_R"].clip(0, 1).to_numpy(), side="left"), 0, nr - 1)
        counts = np.zeros((nx, nr), dtype=np.float32)
        for cidx, col in enumerate(inputs):
            vals = case[col].to_numpy(dtype=np.float32)
            np.add.at(x_arr[ci, cidx], (xi, ri), np.nan_to_num(vals))
        for tidx, col in enumerate(targets):
            vals = case[col].to_numpy(dtype=np.float32)
            np.add.at(y_arr[ci, tidx], (xi, ri), np.nan_to_num(vals))
        np.add.at(counts, (xi, ri), 1.0)
        valid = counts > 0
        mask[ci, 0, valid] = 1.0
        counts_safe = np.where(valid, counts, 1.0)
        x_arr[ci] /= counts_safe[None, :, :]
        y_arr[ci] /= counts_safe[None, :, :]
        for cidx, col in enumerate(inputs):
            if col in {"x_over_L", "r_over_R"}:
                continue
            mean_val = float(np.nanmean(case[col])) if len(case) else 0.0
            x_arr[ci, cidx, ~valid] = mean_val
        for tidx, col in enumerate(targets):
            mean_val = float(np.nanmean(case[col])) if len(case) else 0.0
            y_arr[ci, tidx, ~valid] = mean_val
    meta = {"case_ids": cases, "input_features": inputs, "target_features": targets, "nx": nx, "nr": nr}
    return {"x": x_arr, "y": y_arr, "mask": mask, "case_params": params}, meta


def save_grid_dataset(full: pd.DataFrame, out_dir: str | Path, default_config: dict[str, Any], features_config: dict[str, Any]) -> Path:
    """Save grid_dataset.npz and grid_metadata.json."""
    out_dir = resolve_project_path(out_dir) or Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_cols = ["diameter", "inlet_temperature", "wall_heat_flux", "x_over_L", "r_over_R"]
    input_cols.append("HTD_label")
    target_cols = [c for c in features_config.get("pinn_output_features", []) if c in full.columns]
    arrays, meta = create_grid_arrays(full, input_cols, target_cols, int(default_config.get("grid_nx", 64)), int(default_config.get("grid_nr", 32)))
    path = out_dir / "grid_dataset.npz"
    np.savez_compressed(path, **arrays)
    save_json(meta, out_dir / "grid_metadata.json")
    return path


def create_wall_sequence_arrays(
    wall: pd.DataFrame,
    input_cols: list[str],
    target_cols: list[str],
    sequence_length: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Create padded wall sequences for sequence models."""
    groups = list(wall.groupby(["case_id", "wall_side"], sort=True))
    inputs = [c for c in input_cols if c in wall.columns]
    targets = [c for c in target_cols if c in wall.columns]
    x_arr = np.zeros((len(groups), sequence_length, len(inputs)), dtype=np.float32)
    y_arr = np.zeros((len(groups), sequence_length, len(targets)), dtype=np.float32)
    mask = np.zeros((len(groups), sequence_length), dtype=np.float32)
    ids: list[str] = []
    for gi, ((case_id, side), grp) in enumerate(groups):
        grp = grp.sort_values("x")
        if len(grp) > sequence_length:
            idx = np.linspace(0, len(grp) - 1, sequence_length).round().astype(int)
            grp = grp.iloc[idx]
        n = min(sequence_length, len(grp))
        if n:
            x_arr[gi, :n] = grp[inputs].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=np.float32)
            y_arr[gi, :n] = grp[targets].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=np.float32)
            mask[gi, :n] = 1.0
        ids.append(f"{case_id}|{side}")
    meta = {"sequence_ids": ids, "input_features": inputs, "target_features": targets, "sequence_length": sequence_length}
    return {"x": x_arr, "y": y_arr, "mask": mask}, meta


def save_wall_sequence_dataset(wall: pd.DataFrame, out_dir: str | Path, default_config: dict[str, Any], features_config: dict[str, Any]) -> Path | None:
    """Save wall_sequence_dataset.npz and metadata."""
    if wall.empty:
        return None
    out_dir = resolve_project_path(out_dir) or Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_cols = features_config.get("wall_input_features", [])
    target_cols = [c for c in ["wall_temperature", "nusselt_number", "Tw_minus_Tb", "HTD_label"] if c in wall.columns]
    arrays, meta = create_wall_sequence_arrays(wall, input_cols, target_cols, int(default_config.get("sequence_length", 128)))
    path = out_dir / "wall_sequence_dataset.npz"
    np.savez_compressed(path, **arrays)
    save_json(meta, out_dir / "wall_sequence_metadata.json")
    return path


def build_datasets(
    df: pd.DataFrame,
    default_config: dict[str, Any],
    features_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build full-field, wall, summary, and geometry audit datasets."""
    htd_config = features_config.get("htd_label", {})
    full, audit = add_physical_features(
        df,
        diameter_scale_to_meter=default_config.get("diameter_scale_to_meter", 1.0),
        htd_config=htd_config,
        radial_coordinate=default_config.get("radial_coordinate", "auto"),
        axial_coordinate=default_config.get("axial_coordinate", "x"),
        heat_flux_scale_to_w_m2=default_config.get("heat_flux_scale_to_w_m2", 1.0),
    )
    wall = create_wall_dataset(full, wall_tol=default_config.get("wall_tol", 0.02), htd_config=htd_config)
    full = attach_wall_labels_to_field(full, wall)
    summary = create_case_summary_dataset(full, wall)
    return full, wall, summary, audit


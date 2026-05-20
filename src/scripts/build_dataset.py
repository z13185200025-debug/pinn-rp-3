from __future__ import annotations

import argparse
from pathlib import Path

from src.data_io import load_all_cases
from src.feature_engineering import build_datasets, save_grid_dataset, save_wall_sequence_dataset
from src.preprocessing import clean_dataframe
from src.utils import ensure_project_dirs, load_yaml, resolve_project_path, save_dataframe


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RP-3 processed datasets from DATA/*.dat")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--features_config", default="configs/features.yaml")
    parser.add_argument("--save_raw", action="store_true", help="Also save cleaned concatenated raw table.")
    parser.add_argument("--max_cases", type=int, default=None)
    parser.add_argument("--sample_rows", type=int, default=None, help="Sample rows per case while reading.")
    parser.add_argument("--smoke", action="store_true", help="Use small config defaults for fast smoke tests.")
    args = parser.parse_args()

    default_cfg = load_yaml(args.config)
    features_cfg = load_yaml(args.features_config)
    root = resolve_project_path(".")
    ensure_project_dirs(root)

    smoke_cfg = default_cfg.get("smoke", {})
    max_cases = args.max_cases if args.max_cases is not None else (smoke_cfg.get("max_cases") if args.smoke else None)
    sample_rows = args.sample_rows if args.sample_rows is not None else default_cfg.get("sample_rows_per_case")
    if args.smoke and sample_rows is None:
        sample_rows = smoke_cfg.get("sample_rows")
    data_dir = resolve_project_path(args.data_dir or default_cfg.get("data_dir", "DATA"))
    print(f"[INFO] Loading .dat files from {data_dir}")
    raw = load_all_cases(
        data_dir,
        max_cases=max_cases,
        sample_rows=sample_rows,
        random_seed=int(default_cfg.get("random_seed", 42)),
    )
    clean = clean_dataframe(raw, ignored_features=features_cfg.get("ignored_features", []))
    if args.save_raw:
        raw_path = save_dataframe(clean, "data/raw/all_cases_cleaned", default_cfg.get("save_format", "parquet"))
        print(f"[INFO] Saved cleaned raw data: {raw_path}")

    print("[INFO] Engineering features and building datasets")
    full, wall, summary, audit = build_datasets(clean, default_cfg, features_cfg)

    fmt = default_cfg.get("save_format", "parquet")
    full_path = save_dataframe(full, "data/processed/full_field_dataset", fmt)
    wall_path = save_dataframe(wall, "data/processed/wall_dataset", fmt)
    summary_path = save_dataframe(summary, "data/processed/case_summary_dataset", fmt)
    audit_path = save_dataframe(audit, "data/processed/geometry_audit", fmt)
    grid_path = save_grid_dataset(full, "data/processed", default_cfg, features_cfg)
    seq_path = save_wall_sequence_dataset(wall, "data/processed", default_cfg, features_cfg)
    print(f"[INFO] Saved full-field dataset: {full_path} ({len(full):,} rows)")
    print(f"[INFO] Saved wall dataset: {wall_path} ({len(wall):,} rows)")
    print(f"[INFO] Saved case summary dataset: {summary_path} ({len(summary):,} rows)")
    print(f"[INFO] Saved geometry audit: {audit_path}")
    print(f"[INFO] Saved grid dataset: {grid_path}")
    if seq_path:
        print(f"[INFO] Saved wall sequence dataset: {seq_path}")


if __name__ == "__main__":
    main()

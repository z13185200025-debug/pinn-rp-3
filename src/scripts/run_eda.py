from __future__ import annotations

import argparse
from pathlib import Path

from src.eda import run_basic_eda
from src.utils import load_yaml, read_table, resolve_project_path


def _default_existing(path_stem: str) -> Path:
    for suffix in [".parquet", ".csv"]:
        p = Path(path_stem).with_suffix(suffix)
        p = resolve_project_path(p)
        if p.exists():
            return p
    raise FileNotFoundError(f"Cannot find {path_stem}.parquet or {path_stem}.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EDA and save figures.")
    parser.add_argument("--dataset", default=None, help="Full-field dataset path.")
    parser.add_argument("--wall_dataset", default=None)
    parser.add_argument("--summary_dataset", default=None)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    full_path = resolve_project_path(args.dataset) if args.dataset else _default_existing("data/processed/full_field_dataset")
    wall_path = resolve_project_path(args.wall_dataset) if args.wall_dataset else _default_existing("data/processed/wall_dataset")
    summary_path = resolve_project_path(args.summary_dataset) if args.summary_dataset else _default_existing("data/processed/case_summary_dataset")
    full = read_table(full_path)
    wall = read_table(wall_path)
    summary = read_table(summary_path)
    fig_dir = resolve_project_path(Path(cfg.get("output_dir", "outputs")) / "figures" / "eda")
    run_basic_eda(full, wall, summary, fig_dir, cfg.get("max_eda_cases", 6))
    print(f"[INFO] EDA figures saved under {fig_dir}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.utils import resolve_project_path


def _load_metrics(metrics_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(metrics_dir.rglob("*_metrics.json")):
        if path.name.startswith("legacy_"):
            continue
        with path.open("r", encoding="utf-8") as f:
            row = json.load(f)
        row.setdefault("model_name", path.parent.name if path.parent != metrics_dir else path.name.replace("_metrics.json", ""))
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare saved model metrics.")
    parser.add_argument("--metrics_dir", default="outputs/metrics")
    parser.add_argument("--output_dir", default="outputs/reports")
    args = parser.parse_args()

    rows = _load_metrics(resolve_project_path(args.metrics_dir))
    if not rows:
        raise FileNotFoundError(f"No *_metrics.json files found in {args.metrics_dir}")
    df = pd.DataFrame(rows)
    rename = {
        "MAE_temperature": "MAE_T",
        "RMSE_temperature": "RMSE_T",
        "R2_temperature": "R2_T",
        "MAE_u": "MAE_u",
        "RMSE_u": "RMSE_u",
        "R2_u": "R2_u",
        "MAE_wall_temperature": "MAE_Tw",
        "RMSE_wall_temperature": "RMSE_Tw",
        "R2_wall_temperature": "R2_Tw",
        "MAE_nusselt_number": "MAE_Nu",
        "RMSE_nusselt_number": "RMSE_Nu",
        "R2_nusselt_number": "R2_Nu",
    }
    df = df.rename(columns=rename)
    columns = [
        "model_name",
        "task",
        "split_method",
        "train_cases",
        "test_cases",
        "MAE_T",
        "RMSE_T",
        "R2_T",
        "MAE_u",
        "RMSE_u",
        "R2_u",
        "MAE_Tw",
        "RMSE_Tw",
        "R2_Tw",
        "MAE_Nu",
        "RMSE_Nu",
        "R2_Nu",
        "relative_L2",
        "training_time",
        "parameter_count",
        "notes",
    ]
    for col in columns:
        if col not in df.columns:
            df[col] = None
    out_dir = resolve_project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = df[columns].sort_values(["task", "relative_L2"], na_position="last")
    result.to_csv(out_dir / "model_comparison.csv", index=False)
    result.to_markdown(out_dir / "model_comparison.md", index=False)
    print(f"[INFO] Saved model comparison to {out_dir}")


if __name__ == "__main__":
    main()

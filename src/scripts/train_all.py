from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.utils import project_root


BASELINE_MODELS = {"mlp", "dnn", "resnet_mlp", "random_forest", "svr", "xgboost", "lightgbm"}
GRID_MODELS = {"cnn", "unet", "fno"}
SEQUENCE_MODELS = {"lstm", "gru", "tcn", "transformer"}
FIELD_MODELS = {"deeponet", "property_pinn", "rans_pinn"}
SPECIAL_MODELS = {"pinn", "gnn"}
DEFAULT_MODELS = [
    "mlp",
    "resnet_mlp",
    "pinn",
    "random_forest",
    "cnn",
    "unet",
    "fno",
    "lstm",
    "gru",
    "tcn",
    "transformer",
    "deeponet",
    "property_pinn",
    "rans_pinn",
    "gnn",
]


def _command_for_model(model: str, args: argparse.Namespace) -> list[str]:
    base = [sys.executable, "-m"]
    common_paths = [
        "--config",
        args.config,
        "--default_config",
        args.default_config,
        "--features_config",
        args.features_config,
    ]
    epoch_args = ["--epochs", str(args.epochs)] if args.epochs is not None else []
    if model in BASELINE_MODELS:
        cmd = base + ["src.scripts.train_baselines", "--model", model, "--sample_rows", str(args.sample_rows)] + common_paths + epoch_args
    elif model == "pinn":
        cmd = base + ["src.scripts.train_pinn", "--sample_rows", str(args.pinn_sample_rows)] + common_paths + epoch_args
    elif model in GRID_MODELS:
        cmd = base + ["src.scripts.train_model", "--model", model, "--task", "grid"] + common_paths + epoch_args
        if args.smoke:
            cmd.append("--smoke")
    elif model in SEQUENCE_MODELS:
        cmd = base + ["src.scripts.train_model", "--model", model, "--task", "sequence"] + common_paths + epoch_args
        if args.smoke:
            cmd.append("--smoke")
    elif model in FIELD_MODELS:
        cmd = base + [
            "src.scripts.train_model",
            "--model",
            model,
            "--task",
            "field",
            "--sample_rows",
            str(args.sample_rows),
        ] + common_paths + epoch_args
        if args.smoke:
            cmd.append("--smoke")
    elif model == "gnn":
        cmd = base + ["src.scripts.train_model", "--model", "gnn", "--task", "graph"] + common_paths
        if args.smoke:
            cmd.append("--smoke")
    else:
        raise ValueError(f"Unsupported model in train_all: {model}")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Train multiple RP-3 models sequentially.")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS), help="Comma-separated model list or 'all'.")
    parser.add_argument("--sample_rows", type=int, default=20000)
    parser.add_argument("--pinn_sample_rows", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="Use each model's fast smoke mode where available.")
    parser.add_argument("--fail_fast", action="store_true")
    parser.add_argument("--config", default="configs/model_config.yaml")
    parser.add_argument("--default_config", default="configs/default.yaml")
    parser.add_argument("--features_config", default="configs/features.yaml")
    args = parser.parse_args()

    models = DEFAULT_MODELS if args.models == "all" else [m.strip() for m in args.models.split(",") if m.strip()]
    failures: list[tuple[str, int]] = []
    root = project_root()
    for model in models:
        cmd = _command_for_model(model, args)
        print(f"\n[INFO] Training {model}: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=root)
        if result.returncode != 0:
            failures.append((model, result.returncode))
            print(f"[WARN] {model} failed with exit code {result.returncode}")
            if args.fail_fast:
                raise SystemExit(result.returncode)
    if failures:
        print(f"[WARN] Finished with failures: {failures}")
        raise SystemExit(1)
    print("[INFO] All requested models finished.")


if __name__ == "__main__":
    main()


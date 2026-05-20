from __future__ import annotations

import argparse
from pathlib import Path

from src.datasets import make_loaders, prepare_arrays, split_dataframe
from src.evaluate import evaluate_and_save
from src.models.pinn import PINN, PINNLoss
from src.train import predict_model, save_checkpoint, train_model
from src.utils import get_device, load_yaml, model_output_dirs, read_table, resolve_project_path, set_seed
from src.visualization import plot_loss_curve


def _dataset_path() -> Path:
    for suffix in [".parquet", ".csv"]:
        path = Path("data/processed") / f"full_field_dataset{suffix}"
        path = resolve_project_path(path)
        if path.exists():
            return path
    raise FileNotFoundError("Run build_dataset first; missing full_field_dataset.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the base PINN model.")
    parser.add_argument("--config", default="configs/model_config.yaml")
    parser.add_argument("--features_config", default="configs/features.yaml")
    parser.add_argument("--default_config", default="configs/default.yaml")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--sample_rows", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    model_cfg = load_yaml(args.config)
    if args.epochs is not None:
        model_cfg.setdefault("training", {})["epochs"] = args.epochs
    features_cfg = load_yaml(args.features_config)
    default_cfg = load_yaml(args.default_config)
    set_seed(model_cfg.get("split", {}).get("random_seed", default_cfg.get("random_seed", 42)))
    output_dir = resolve_project_path(default_cfg.get("output_dir", "outputs"))
    dirs = model_output_dirs("pinn", output_dir)
    df = read_table(Path(args.dataset) if args.dataset else _dataset_path())
    if args.sample_rows and len(df) > args.sample_rows:
        df = df.sample(args.sample_rows, random_state=model_cfg.get("split", {}).get("random_seed", 42))

    split = split_dataframe(df, model_cfg.get("split", {}))
    arrays = prepare_arrays(
        split,
        features_cfg["pinn_input_features"],
        features_cfg["pinn_output_features"],
        dirs["models"],
        "pinn",
    )
    x_train, y_train, x_val, y_val, x_test, y_test, x_scaler, y_scaler, inputs, targets, test_meta = arrays
    train_loader, val_loader, test_loader = make_loaders(
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        model_cfg.get("training", {}).get("batch_size", 4096),
        num_workers=default_cfg.get("num_workers", 0),
        pin_memory=default_cfg.get("pin_memory", False),
    )
    pinn_cfg = model_cfg.get("pinn", {})
    model = PINN(len(inputs), len(targets), pinn_cfg.get("hidden_layers", [128, 128, 128, 128]), pinn_cfg.get("activation", "tanh"))
    criterion = PINNLoss(
        pinn_cfg.get("loss_weights", {}),
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        input_names=inputs,
        target_names=targets,
        cp_const=pinn_cfg.get("Cp_const", 2200.0),
        k_const=pinn_cfg.get("k_const", 0.12),
        diameter_scale_to_meter=default_cfg.get("diameter_scale_to_meter", 1.0),
        heat_flux_scale_to_w_m2=default_cfg.get("heat_flux_scale_to_w_m2", 1.0),
    )
    device = get_device(default_cfg.get("device", "auto"))
    model, history, info = train_model(model, train_loader, val_loader, model_cfg, device, criterion=criterion, model_name="pinn", checkpoint_dir=dirs["models"])
    y_true_s, y_pred_s = predict_model(model, test_loader, device)
    extra = {
        **info,
        "model_name": "pinn",
        "task": "field",
        "split_method": model_cfg.get("split", {}).get("split_method", "case_split"),
        "train_cases": int(split.train["case_id"].nunique()),
        "test_cases": int(split.test["case_id"].nunique()),
        "notes": "Dimensional residuals use scaler chain rule. Diameter/heat-flux unit assumptions still need confirmation.",
    }
    evaluate_and_save(y_true_s, y_pred_s, y_scaler, targets, "pinn", output_dir, extra=extra, meta=test_meta)
    plot_loss_curve(history, dirs["figures"], "pinn")
    save_checkpoint(model, dirs["models"] / "pinn.pt", {"inputs": inputs, "targets": targets, **extra})
    print("[INFO] Saved PINN checkpoint, metrics, predictions, and figures.")


if __name__ == "__main__":
    main()

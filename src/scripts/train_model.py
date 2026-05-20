from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from src.datasets import GridDataset, WallSequenceDataset, make_loaders, prepare_arrays, split_dataframe
from src.evaluate import evaluate_and_save
from src.models.cnn import CNNRegressor, UNet2D
from src.models.deeponet import TabularDeepONet
from src.models.fno import FNO2d
from src.models.gnn import MeshGraphNet
from src.models.mlp import MLP
from src.models.property_pinn import PropertyPINN
from src.models.rans_pinn import RANSPINN
from src.models.resnet_mlp import ResNetMLP
from src.models.sequence import GRURegressor, LSTMRegressor, TCNRegressor
from src.models.transformer import TransformerEncoderRegressor
from src.train import predict_model, save_checkpoint, train_model
from src.utils import get_device, load_yaml, model_output_dirs, read_table, resolve_project_path, save_json, set_seed
from src.visualization import plot_loss_curve


def _existing(stem: str) -> Path:
    for suffix in [".parquet", ".csv", ".npz"]:
        path = resolve_project_path(stem + suffix)
        if path and path.exists():
            return path
    raise FileNotFoundError(f"Missing dataset for {stem}. Run build_dataset first.")


def _split_dataset(ds, seed: int):
    n = len(ds)
    if n < 3:
        return ds, ds, ds
    n_test = max(1, int(round(n * 0.2)))
    n_val = max(1, int(round(n * 0.1)))
    n_train = max(1, n - n_test - n_val)
    return random_split(ds, [n_train, n_val, n - n_train - n_val], generator=torch.Generator().manual_seed(seed))


def _standardize_grid(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_mean = x.mean(axis=(0, 2, 3), keepdims=True)
    x_std = x.std(axis=(0, 2, 3), keepdims=True) + 1e-6
    y_mean = y.mean(axis=(0, 2, 3), keepdims=True)
    y_std = y.std(axis=(0, 2, 3), keepdims=True) + 1e-6
    return (x - x_mean) / x_std, (y - y_mean) / y_std


def train_grid_model(args, model_cfg, default_cfg) -> None:
    path = resolve_project_path(args.dataset) if args.dataset else _existing("data/processed/grid_dataset")
    data = np.load(path)
    x, y, mask = data["x"], data["y"], data["mask"]
    if args.smoke:
        x, y, mask = x[: min(4, len(x))], y[: min(4, len(y))], mask[: min(4, len(mask))]
    x, y = _standardize_grid(x, y, mask)
    ds = GridDataset(x, y, mask)
    train_ds, val_ds, test_ds = _split_dataset(ds, model_cfg["split"]["random_seed"])
    batch_size = min(int(model_cfg["training"].get("batch_size", 4)), max(1, len(train_ds)))
    loader_args = {"batch_size": batch_size, "num_workers": default_cfg.get("num_workers", 0), "pin_memory": default_cfg.get("pin_memory", False)}
    loaders = (
        DataLoader(train_ds, shuffle=True, **loader_args),
        DataLoader(val_ds, shuffle=False, **loader_args),
        DataLoader(test_ds, shuffle=False, **loader_args),
    )
    in_ch, out_ch = x.shape[1], y.shape[1]
    if args.model == "cnn":
        model = CNNRegressor(in_ch, out_ch, **{k: v for k, v in model_cfg.get("cnn", {}).items() if k not in {"in_channels", "out_channels"}})
    elif args.model == "unet":
        model = UNet2D(in_ch, out_ch, **{k: v for k, v in model_cfg.get("unet", {}).items() if k not in {"in_channels", "out_channels"}})
    elif args.model == "fno":
        model = FNO2d(in_ch, out_ch, **{k: v for k, v in model_cfg.get("fno", {}).items() if k not in {"in_channels", "out_channels"}})
    else:
        raise ValueError(args.model)
    output_dir = resolve_project_path(default_cfg.get("output_dir", "outputs"))
    dirs = model_output_dirs(args.model, output_dir)
    device = get_device(default_cfg.get("device", "auto"))
    model, history, info = train_model(model, loaders[0], loaders[1], model_cfg, device, model_name=args.model, checkpoint_dir=dirs["models"])
    save_checkpoint(model, dirs["models"] / f"{args.model}.pt", info)
    plot_loss_curve(history, dirs["figures"], args.model)
    save_json({"model_name": args.model, "task": "grid", **info, "relative_L2": history["val_loss"][-1]}, dirs["metrics"] / f"{args.model}_metrics.json")
    print(f"[INFO] Saved {args.model} grid smoke artifacts.")


def train_sequence_model(args, model_cfg, default_cfg) -> None:
    path = resolve_project_path(args.dataset) if args.dataset else _existing("data/processed/wall_sequence_dataset")
    data = np.load(path)
    x, y, mask = data["x"], data["y"], data["mask"]
    if args.smoke:
        x, y, mask = x[: min(8, len(x))], y[: min(8, len(y))], mask[: min(8, len(mask))]
    x_mean, x_std = x.mean(axis=(0, 1), keepdims=True), x.std(axis=(0, 1), keepdims=True) + 1e-6
    y_mean, y_std = y.mean(axis=(0, 1), keepdims=True), y.std(axis=(0, 1), keepdims=True) + 1e-6
    ds = WallSequenceDataset((x - x_mean) / x_std, (y - y_mean) / y_std, mask)
    train_ds, val_ds, test_ds = _split_dataset(ds, model_cfg["split"]["random_seed"])
    batch_size = min(int(model_cfg["training"].get("batch_size", 4)), max(1, len(train_ds)))
    loaders = (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )
    in_dim, out_dim = x.shape[-1], y.shape[-1]
    seq_cfg = model_cfg.get("sequence", {})
    if args.model == "lstm":
        model = LSTMRegressor(in_dim, out_dim, **seq_cfg)
    elif args.model == "gru":
        model = GRURegressor(in_dim, out_dim, **seq_cfg)
    elif args.model == "tcn":
        model = TCNRegressor(in_dim, out_dim, **seq_cfg)
    elif args.model == "transformer":
        model = TransformerEncoderRegressor(in_dim, out_dim, **model_cfg.get("transformer", {}))
    else:
        raise ValueError(args.model)
    output_dir = resolve_project_path(default_cfg.get("output_dir", "outputs"))
    dirs = model_output_dirs(args.model, output_dir)
    device = get_device(default_cfg.get("device", "auto"))
    model, history, info = train_model(model, loaders[0], loaders[1], model_cfg, device, model_name=args.model, checkpoint_dir=dirs["models"])
    save_checkpoint(model, dirs["models"] / f"{args.model}.pt", info)
    plot_loss_curve(history, dirs["figures"], args.model)
    save_json({"model_name": args.model, "task": "sequence", **info, "relative_L2": history["val_loss"][-1]}, dirs["metrics"] / f"{args.model}_metrics.json")
    print(f"[INFO] Saved {args.model} sequence smoke artifacts.")


def train_field_model(args, model_cfg, default_cfg, features_cfg) -> None:
    path = resolve_project_path(args.dataset) if args.dataset else _existing("data/processed/full_field_dataset")
    df = read_table(path)
    if args.max_cases:
        keep = sorted(df["case_id"].unique())[: args.max_cases]
        df = df[df["case_id"].isin(keep)]
    if args.sample_rows and len(df) > args.sample_rows:
        df = df.sample(args.sample_rows, random_state=model_cfg["split"]["random_seed"])
    split = split_dataframe(df, model_cfg["split"])
    inputs = features_cfg["pinn_input_features"]
    targets = features_cfg["pinn_output_features"]
    output_dir = resolve_project_path(default_cfg.get("output_dir", "outputs"))
    dirs = model_output_dirs(args.model, output_dir)
    arrays = prepare_arrays(split, inputs, targets, dirs["models"], args.model)
    x_train, y_train, x_val, y_val, x_test, y_test, x_scaler, y_scaler, input_names, target_names, test_meta = arrays
    loaders = make_loaders(x_train, y_train, x_val, y_val, x_test, y_test, model_cfg["training"].get("batch_size", 1024))
    if args.model == "deeponet":
        deeponet_cfg = {k: v for k, v in model_cfg.get("deeponet", {}).items() if k not in {"branch_dim", "trunk_dim", "output_dim"}}
        model = TabularDeepONet(branch_dim=3, trunk_dim=2, output_dim=len(target_names), **deeponet_cfg)
    elif args.model in {"mlp", "dnn"}:
        model = MLP(len(input_names), len(target_names), **model_cfg.get(args.model, model_cfg.get("mlp", {})))
    elif args.model == "resnet_mlp":
        model = ResNetMLP(len(input_names), len(target_names), **model_cfg.get("resnet_mlp", {}))
    elif args.model == "property_pinn":
        model = PropertyPINN(len(input_names), len(target_names), hidden_layers=model_cfg["property_pinn"]["hidden_layers"], activation=model_cfg["property_pinn"]["activation"])
    elif args.model == "rans_pinn":
        model = RANSPINN(len(input_names), len(target_names), auxiliary_names=[], hidden_layers=model_cfg["rans_pinn"]["hidden_layers"], activation=model_cfg["rans_pinn"]["activation"])
    else:
        raise ValueError(args.model)
    device = get_device(default_cfg.get("device", "auto"))
    model, history, info = train_model(model, loaders[0], loaders[1], model_cfg, device, model_name=args.model, checkpoint_dir=dirs["models"])
    y_true_s, y_pred_s = predict_model(model, loaders[2], device)
    evaluate_and_save(y_true_s, y_pred_s, y_scaler, target_names, args.model, output_dir, extra={"model_name": args.model, "task": "field", **info}, meta=test_meta)
    plot_loss_curve(history, dirs["figures"], args.model)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified training entrypoint.")
    parser.add_argument("--model", required=True, choices=["mlp", "dnn", "resnet_mlp", "cnn", "unet", "fno", "lstm", "gru", "tcn", "transformer", "deeponet", "pinn", "property_pinn", "rans_pinn", "random_forest", "xgboost", "lightgbm", "svr", "gnn"])
    parser.add_argument("--task", default="field", choices=["field", "wall", "grid", "sequence", "graph", "case"])
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--config", default="configs/model_config.yaml")
    parser.add_argument("--default_config", default="configs/default.yaml")
    parser.add_argument("--features_config", default="configs/features.yaml")
    parser.add_argument("--sample_rows", type=int, default=None)
    parser.add_argument("--max_cases", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    model_cfg = load_yaml(args.config)
    default_cfg = load_yaml(args.default_config)
    features_cfg = load_yaml(args.features_config)
    if args.epochs is not None:
        model_cfg["training"]["epochs"] = args.epochs
    if args.smoke:
        model_cfg["training"]["epochs"] = args.epochs or 2
        model_cfg["training"]["batch_size"] = min(model_cfg["training"].get("batch_size", 1024), 4 if args.task in {"grid", "sequence"} else 1024)
    set_seed(model_cfg["split"].get("random_seed", default_cfg.get("random_seed", 42)))

    if args.model == "gnn":
        try:
            MeshGraphNet(4, 2)
        except ImportError as exc:
            print(f"[SKIP] {exc}")
            return
    if args.model in {"random_forest", "xgboost", "lightgbm", "svr", "pinn"}:
        print(f"[INFO] Use existing dedicated script for {args.model}: train_baselines.py or train_pinn.py")
        return
    if args.task == "grid":
        train_grid_model(args, model_cfg, default_cfg)
    elif args.task == "sequence":
        train_sequence_model(args, model_cfg, default_cfg)
    elif args.task == "field":
        train_field_model(args, model_cfg, default_cfg, features_cfg)
    else:
        raise ValueError(f"Task {args.task} is not implemented in unified trainer yet.")


if __name__ == "__main__":
    main()

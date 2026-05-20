from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from src.datasets import make_loaders, prepare_arrays, split_dataframe
from src.evaluate import evaluate_and_save
from src.metrics import regression_metrics
from src.models.mlp import MLP
from src.models.resnet_mlp import ResNetMLP
from src.train import predict_model, save_checkpoint, train_model
from src.utils import available_columns, get_device, load_yaml, model_output_dirs, read_table, resolve_project_path, save_json, set_seed
from src.visualization import plot_feature_importance, plot_loss_curve


def _dataset_path(task: str) -> Path:
    stem = "wall_dataset" if task == "wall" else "full_field_dataset"
    for suffix in [".parquet", ".csv"]:
        path = Path("data/processed") / f"{stem}{suffix}"
        path = resolve_project_path(path)
        if path.exists():
            return path
    raise FileNotFoundError(f"Run build_dataset first; missing data/processed/{stem}.parquet or .csv")


def _features(task: str, features_cfg: dict, model: str) -> tuple[list[str], list[str]]:
    if task == "wall":
        return features_cfg["wall_input_features"], ["wall_temperature", "nusselt_number", "Tw_minus_Tb"]
    if model in {"mlp", "dnn", "resnet_mlp", "random_forest", "svr", "xgboost", "lightgbm"}:
        return features_cfg["pinn_input_features"], features_cfg["pinn_output_features"]
    return features_cfg["main_input_features"], features_cfg["main_output_features"]


def train_sklearn(
    model_name: str,
    df: pd.DataFrame,
    split_cfg: dict,
    input_features: list[str],
    target_features: list[str],
    output_dir: Path,
) -> None:
    dirs = model_output_dirs(model_name, output_dir)
    split = split_dataframe(df, split_cfg)
    inputs = available_columns(split.train, input_features)
    targets = available_columns(split.train, target_features)
    train = split.train[inputs + targets].replace([np.inf, -np.inf], np.nan).dropna()
    test = split.test[inputs + targets].replace([np.inf, -np.inf], np.nan).dropna()
    x_train, y_train = train[inputs], train[targets]
    x_test, y_test = test[inputs], test[targets]

    if model_name == "random_forest":
        estimator = RandomForestRegressor(n_estimators=200, random_state=split_cfg.get("random_seed", 42), n_jobs=-1)
    elif model_name == "svr":
        estimator = MultiOutputRegressor(SVR(C=10.0, epsilon=0.01))
    elif model_name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except Exception as exc:
            raise ImportError("xgboost is not installed. Install xgboost or choose random_forest.") from exc
        estimator = MultiOutputRegressor(XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03))
    elif model_name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except Exception as exc:
            raise ImportError("lightgbm is not installed. Install lightgbm or choose random_forest.") from exc
        estimator = MultiOutputRegressor(LGBMRegressor(n_estimators=500, learning_rate=0.03))
    else:
        raise ValueError(model_name)

    pipe = Pipeline([("x_scaler", StandardScaler()), ("model", estimator)])
    start = time.time()
    pipe.fit(x_train, y_train)
    y_pred = pipe.predict(x_test)
    metrics = regression_metrics(y_test.to_numpy(), y_pred, targets)
    metrics.update(
        {
            "model_name": model_name,
            "task": "regression",
            "split_method": split_cfg.get("split_method", "case_split"),
            "training_time": time.time() - start,
            "parameter_count": 0,
            "train_cases": int(split.train["case_id"].nunique()) if "case_id" in split.train else None,
            "test_cases": int(split.test["case_id"].nunique()) if "case_id" in split.test else None,
        }
    )
    save_json(metrics, dirs["metrics"] / f"{model_name}_metrics.json")
    pred = pd.DataFrame(y_test.to_numpy(), columns=[f"true_{c}" for c in targets])
    for i, c in enumerate(targets):
        pred[f"pred_{c}"] = y_pred[:, i]
    pred.to_csv(dirs["predictions"] / f"{model_name}_predictions.csv", index=False)
    joblib.dump(pipe, dirs["models"] / f"{model_name}.joblib")
    if model_name == "random_forest":
        importances = pd.Series(pipe.named_steps["model"].feature_importances_, index=inputs)
        plot_feature_importance(importances, dirs["figures"], model_name)
    print(f"[INFO] Saved {model_name} metrics and predictions.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline models.")
    parser.add_argument("--model", default="mlp", choices=["mlp", "dnn", "resnet_mlp", "random_forest", "svr", "xgboost", "lightgbm"])
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--task", default="field", choices=["field", "wall"])
    parser.add_argument("--config", default="configs/model_config.yaml")
    parser.add_argument("--features_config", default="configs/features.yaml")
    parser.add_argument("--default_config", default="configs/default.yaml")
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
    df = read_table(Path(args.dataset) if args.dataset else _dataset_path(args.task))
    if args.sample_rows and len(df) > args.sample_rows:
        df = df.sample(args.sample_rows, random_state=model_cfg.get("split", {}).get("random_seed", 42))

    input_features, target_features = _features(args.task, features_cfg, args.model)
    split_cfg = model_cfg.get("split", {})
    if args.model in {"random_forest", "svr", "xgboost", "lightgbm"}:
        train_sklearn(args.model, df, split_cfg, input_features, target_features, output_dir)
        return

    split = split_dataframe(df, split_cfg)
    dirs = model_output_dirs(args.model, output_dir)
    arrays = prepare_arrays(split, input_features, target_features, dirs["models"], args.model)
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
    if args.model in {"mlp", "dnn"}:
        cfg_key = args.model
        model = MLP(len(inputs), len(targets), **model_cfg[cfg_key])
    elif args.model == "resnet_mlp":
        model = ResNetMLP(len(inputs), len(targets), **model_cfg["resnet_mlp"])
    else:
        raise ValueError(args.model)
    device = get_device(default_cfg.get("device", "auto"))
    model, history, info = train_model(model, train_loader, val_loader, model_cfg, device, model_name=args.model, checkpoint_dir=dirs["models"])
    y_true_s, y_pred_s = predict_model(model, test_loader, device)
    extra = {
        **info,
        "model_name": args.model,
        "task": args.task,
        "split_method": split_cfg.get("split_method", "case_split"),
        "train_cases": int(split.train["case_id"].nunique()),
        "test_cases": int(split.test["case_id"].nunique()),
    }
    evaluate_and_save(y_true_s, y_pred_s, y_scaler, targets, args.model, output_dir, extra=extra, meta=test_meta)
    plot_loss_curve(history, dirs["figures"], args.model)
    save_checkpoint(model, dirs["models"] / f"{args.model}.pt", {"inputs": inputs, "targets": targets, **extra})
    print(f"[INFO] Saved {args.model} checkpoint, metrics, predictions, and figures.")


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from .utils import count_parameters, resolve_project_path


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """MSE over valid masked entries."""
    if mask is None:
        return torch.mean((pred - target) ** 2)
    while mask.ndim < pred.ndim:
        mask = mask.unsqueeze(1) if pred.ndim == 4 else mask.unsqueeze(-1)
    mask = mask.to(pred.dtype)
    denom = mask.sum().clamp_min(1.0)
    return (((pred - target) ** 2) * mask).sum() / denom


def _forward_loss(model: nn.Module, batch, criterion, device: torch.device):
    if isinstance(batch, dict):
        xb = batch["x"].to(device)
        yb = batch["y"].to(device)
        mask = batch.get("mask")
        mask = mask.to(device) if mask is not None else None
        pred = model(xb, mask) if xb.ndim == 3 else model(xb)
        if criterion is None:
            loss = masked_mse(pred, yb, mask)
        else:
            try:
                loss = criterion(pred, yb, mask)
            except TypeError:
                loss = masked_mse(pred, yb, mask)
        return loss, pred, yb
    xb, yb = batch
    xb = xb.to(device)
    yb = yb.to(device)
    if criterion.__class__.__name__ == "PINNLoss":
        loss, parts = criterion(model, xb, yb)
        return loss, model(xb), yb, parts
    pred = model(xb)
    loss = criterion(pred, yb) if criterion is not None else torch.mean((pred - yb) ** 2)
    return loss, pred, yb


def _scheduler(config: dict, optimizer, epochs: int, steps_per_epoch: int):
    name = config.get("scheduler", "none")
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    if name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    if name == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=float(config.get("learning_rate", 1e-3)), epochs=epochs, steps_per_epoch=max(1, steps_per_epoch)
        )
    return None


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    config: dict,
    device: torch.device,
    criterion: nn.Module | Callable | None = None,
    model_name: str = "model",
    checkpoint_dir: str | Path | None = None,
) -> tuple[nn.Module, dict[str, list[float]], dict[str, float]]:
    """Train a PyTorch model with AMP, early stopping, and optional masked batches."""
    train_cfg = config.get("training", config)
    epochs = int(train_cfg.get("epochs", 20))
    lr = float(train_cfg.get("learning_rate", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-6))
    patience = int(train_cfg.get("early_stopping_patience", 8))
    min_delta = float(train_cfg.get("early_stopping_min_delta", 0.0))
    grad_accum = max(1, int(train_cfg.get("gradient_accumulation_steps", 1)))
    grad_clip = train_cfg.get("gradient_clip_norm", 10.0)
    use_amp = bool(train_cfg.get("amp", False)) and device.type == "cuda"

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = _scheduler(train_cfg, optimizer, epochs, len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = criterion or nn.MSELoss()
    checkpoint_dir = resolve_project_path(checkpoint_dir) if checkpoint_dir else None
    if checkpoint_dir:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    resume = train_cfg.get("resume")
    if resume:
        payload = torch.load(resolve_project_path(resume), map_location=device)
        model.load_state_dict(payload["state_dict"])

    best_state = None
    best_val = float("inf")
    wait = 0
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "lr": []}
    component_accum: dict[str, list[float]] = {}
    start = time.time()

    for epoch in tqdm(range(epochs), desc=f"Training {model_name}"):
        model.train()
        train_losses = []
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                result = _forward_loss(model, batch, criterion, device)
                parts = result[3] if len(result) == 4 else {}
                loss = result[0] / grad_accum
            scaler.scale(loss).backward()
            if (step + 1) % grad_accum == 0:
                if grad_clip:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None and scheduler.__class__.__name__ == "OneCycleLR":
                    scheduler.step()
            train_losses.append(float((loss * grad_accum).detach().cpu()))
            for key, value in parts.items():
                component_accum.setdefault(f"train_{key}", []).append(float(value))

        model.eval()
        val_losses = []
        for batch in val_loader:
            if criterion.__class__.__name__ == "PINNLoss":
                result = _forward_loss(model, batch, criterion, device)
            else:
                with torch.no_grad():
                    result = _forward_loss(model, batch, criterion, device)
            val_losses.append(float(result[0].detach().cpu()))
        val_loss = float(np.mean(val_losses)) if val_losses else float(np.mean(train_losses))
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        if scheduler is not None and scheduler.__class__.__name__ == "ReduceLROnPlateau":
            scheduler.step(val_loss)
        elif scheduler is not None and scheduler.__class__.__name__ not in {"OneCycleLR"}:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))
        for key, values in component_accum.items():
            if values:
                history.setdefault(key, []).append(float(np.mean(values)))
        component_accum.clear()

        if val_loss < best_val - min_delta:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
            if checkpoint_dir and train_cfg.get("save_best", True):
                save_checkpoint(model, checkpoint_dir / f"{model_name}_best.pt", {"epoch": epoch, "best_val_loss": best_val})
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    if checkpoint_dir:
        save_checkpoint(model, checkpoint_dir / f"{model_name}_last.pt", {"best_val_loss": best_val})
        _save_history(history, checkpoint_dir / f"{model_name}_history.csv")
    info = {"training_time": time.time() - start, "parameter_count": count_parameters(model), "best_val_loss": best_val}
    return model, history, info


def _save_history(history: dict[str, list[float]], path: Path) -> None:
    keys = list(history)
    rows = max((len(v) for v in history.values()), default=0)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", *keys])
        writer.writeheader()
        for i in range(rows):
            row = {"epoch": i}
            for key in keys:
                row[key] = history[key][i] if i < len(history[key]) else ""
            writer.writerow(row)


def predict_model(model: nn.Module, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Return scaled true and predicted arrays for tabular loaders."""
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb.to(device)).cpu().numpy()
            preds.append(pred)
            ys.append(yb.numpy())
    return np.vstack(ys), np.vstack(preds)


def save_checkpoint(model: nn.Module, path: str | Path, metadata: dict | None = None) -> None:
    """Save a PyTorch checkpoint."""
    path = resolve_project_path(path) or Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata or {}}, path)


def load_checkpoint(model: nn.Module, path: str | Path, map_location: str | torch.device = "cpu") -> nn.Module:
    """Load model weights from a checkpoint."""
    payload = torch.load(resolve_project_path(path), map_location=map_location)
    model.load_state_dict(payload["state_dict"])
    return model

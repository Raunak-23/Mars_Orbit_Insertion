"""
src/lstm_training.py
--------------------
Training loop, learning rate scheduling, early stopping, and model checkpointing
for the Multi-Output Multi-Horizon Quantile LSTM telemetry forecasting system.
"""

import copy
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.lstm_dataset import (
    FEATURE_COLS,
    HORIZONS,
    QUANTILES,
    TARGET_COLS,
    create_telemetry_dataloaders,
)
from src.lstm_model import MultiHorizonQuantileLSTM, MultiQuantilePinballLoss


def set_seed(seed: int = 42) -> None:
    """Set random seeds across numpy and torch for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
) -> float:
    """Train for a single epoch and return average training loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        x = batch["x"].to(device)
        y_delta = batch["y_delta"].to(device)

        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y_delta)

        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Evaluate model on a DataLoader and return average loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y_delta = batch["y_delta"].to(device)

            preds = model(x)
            loss = criterion(preds, y_delta)

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(1, n_batches)


def train_telemetry_lstm(
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Optional[Dict[str, Any]] = None,
    checkpoint_dir: Union[str, Path] = "models",
    verbose: bool = True,
) -> Tuple[nn.Module, Dict[str, List[float]], Dict[str, Any]]:
    """
    Train MultiHorizonQuantileLSTM with early stopping, learning rate scheduling,
    and checkpointing.

    Returns:
      - best_model: nn.Module loaded with weights yielding lowest validation loss.
      - history: dict with 'train_loss', 'val_loss', 'lr'.
      - final_config: complete hyperparameters and metadata dictionary.
    """
    default_config = {
        "input_dim": len(FEATURE_COLS),
        "num_targets": len(TARGET_COLS),
        "num_horizons": len(HORIZONS),
        "quantiles": QUANTILES,
        "seq_len": 60,
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "bidirectional": False,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "epochs": 60,
        "patience": 12,
        "grad_clip": 1.0,
        "crossing_penalty_weight": 0.05,
        "seed": 42,
    }

    if config:
        default_config.update(config)
    cfg = default_config

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiHorizonQuantileLSTM(
        input_dim=cfg["input_dim"],
        num_targets=cfg["num_targets"],
        num_horizons=cfg["num_horizons"],
        quantiles=cfg["quantiles"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        bidirectional=cfg["bidirectional"],
    ).to(device)

    criterion = MultiQuantilePinballLoss(
        quantiles=cfg["quantiles"],
        crossing_penalty_weight=cfg["crossing_penalty_weight"],
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=4,
        min_lr=1e-5,
    )

    history = {"train_loss": [], "val_loss": [], "lr": []}

    best_val_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    patience_counter = 0

    chk_dir = Path(checkpoint_dir)
    chk_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = chk_dir / "best_lstm_telemetry_model.pt"
    config_save_path = chk_dir / "lstm_config.json"

    start_time = time.time()
    if verbose:
        mode_str = "BiLSTM" if cfg["bidirectional"] else "Causal LSTM"
        print(
            f"Starting {mode_str} training (epochs={cfg['epochs']}, hidden_dim={cfg['hidden_dim']}, "
            f"layers={cfg['num_layers']}, lr={cfg['lr']}) on {device}..."
        )

    for epoch in range(1, cfg["epochs"] + 1):
        train_loss = train_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_clip=cfg["grad_clip"],
        )

        val_loss = evaluate_loss(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (epoch % 5 == 0 or epoch == 1 or epoch == cfg["epochs"]):
            print(
                f"Epoch [{epoch:03d}/{cfg['epochs']:03d}] | "
                f"Train Pinball: {train_loss:.4f} | "
                f"Val Pinball: {val_loss:.4f} | "
                f"Best Val: {best_val_loss:.4f} (ep {best_epoch}) | "
                f"LR: {current_lr:.6f}"
            )

        if patience_counter >= cfg["patience"]:
            if verbose:
                print(
                    f"Early stopping triggered at epoch {epoch} (patience={cfg['patience']}). Best epoch: {best_epoch}."
                )
            break

    elapsed = time.time() - start_time
    if verbose:
        print(f"Training completed in {elapsed:.2f}s. Best Val Pinball: {best_val_loss:.4f}")

    # Load and save best weights
    model.load_state_dict(best_weights)
    torch.save(best_weights, model_save_path)

    cfg["best_val_loss"] = float(best_val_loss)
    cfg["best_epoch"] = int(best_epoch)
    cfg["training_time_sec"] = float(round(elapsed, 2))
    cfg["target_cols"] = TARGET_COLS
    cfg["feature_cols"] = FEATURE_COLS
    cfg["horizons"] = HORIZONS

    with open(config_save_path, "w") as f:
        json.dump(cfg, f, indent=2)

    return model, history, cfg

"""
src/lstm_evaluation.py
----------------------
Holdout evaluation engine, probabilistic interval calibration assessment,
metrics reporting, predictions CSV generation, and publication-grade visualizations.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader

from src.lstm_dataset import (
    HORIZONS,
    QUANTILES,
    TARGET_COLS,
    reconstruct_physical_predictions,
)


def compute_circular_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute circular mean absolute error across the [0, 360) boundary."""
    diff = np.abs(y_true - y_pred) % 360.0
    circ_diff = np.minimum(diff, 360.0 - diff)
    return float(np.mean(circ_diff))


def compute_pinball_loss_np(y_true: np.ndarray, y_pred_q: np.ndarray, q: float) -> float:
    """Compute empirical pinball loss for a single quantile scalar q."""
    diff = y_true - y_pred_q
    loss = np.maximum(q * diff, (q - 1.0) * diff)
    return float(np.mean(loss))


def evaluate_lstm_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: Optional[torch.device] = None,
    horizons: List[int] = HORIZONS,
    target_cols: List[str] = TARGET_COLS,
    quantiles: List[float] = QUANTILES,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate predictions on a DataLoader, reconstruct physical level forecasts,
    evaluate MAE, RMSE, R2, Pinball Loss, PICP, MPIW for each target and horizon,
    and format both detailed predictions and metrics summary DataFrames.

    Returns:
      - metrics_df: Summary metrics table (target x horizon)
      - predictions_df: Long-form predictions table
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    all_actuals = []
    all_recons = []
    all_timestamps = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y_actual = batch["y_actual"]  # (B, H, T)
            y_terminal = batch["y_terminal"]  # (B, T)
            timestamps = batch["timestamp"]

            # Predict delta quantiles: (B, H, T, Q)
            preds_delta = model(x).cpu()

            # Reconstruct physical level predictions: (B, H, T, Q)
            preds_recon = reconstruct_physical_predictions(
                y_delta_pred=preds_delta,
                y_terminal=y_terminal,
            )

            all_actuals.append(y_actual.numpy())
            all_recons.append(preds_recon.numpy())
            all_timestamps.extend(timestamps)

    # Concatenate across batches:
    # actuals: (N, H, T)
    # recons:  (N, H, T, Q)
    actuals = np.concatenate(all_actuals, axis=0)
    recons = np.concatenate(all_recons, axis=0)
    N, H, T, Q = recons.shape

    q_map = {q_val: idx for idx, q_val in enumerate(quantiles)}
    idx_q50 = q_map[0.50]
    idx_q05 = q_map[0.05]
    idx_q95 = q_map[0.95]
    idx_q10 = q_map[0.10]
    idx_q90 = q_map[0.90]
    idx_q25 = q_map[0.25]
    idx_q75 = q_map[0.75]

    metrics_records = []
    pred_rows = []

    for h_idx, h in enumerate(horizons):
        for t_idx, target_name in enumerate(target_cols):
            y_t = actuals[:, h_idx, t_idx]
            q_preds_t = recons[:, h_idx, t_idx, :]  # (N, Q)

            y_median = q_preds_t[:, idx_q50]

            # Accuracy metrics on median
            if target_name == "Mag_heading":
                mae = compute_circular_mae(y_t, y_median)
                diff = np.abs(y_t - y_median) % 360.0
                circ_diff = np.minimum(diff, 360.0 - diff)
                rmse = float(np.sqrt(np.mean(circ_diff**2)))
            else:
                mae = float(np.mean(np.abs(y_t - y_median)))
                rmse = float(np.sqrt(mean_squared_error(y_t, y_median)))

            # R-squared
            y_var = np.var(y_t)
            if y_var > 1e-6:
                r2 = float(r2_score(y_t, y_median))
            else:
                r2 = 0.0

            # Quantile pinball loss
            pinball_losses = [
                compute_pinball_loss_np(y_t, q_preds_t[:, k], q_val)
                for k, q_val in enumerate(quantiles)
            ]
            mean_pinball = float(np.mean(pinball_losses))

            # Coverage & Interval Widths
            # 90% interval: [q05, q95]
            q05 = q_preds_t[:, idx_q05]
            q95 = q_preds_t[:, idx_q95]
            covered_90 = (y_t >= q05) & (y_t <= q95)
            picp_90 = float(np.mean(covered_90) * 100.0)
            mpiw_90 = float(np.mean(q95 - q05))

            # 80% interval: [q10, q90]
            q10 = q_preds_t[:, idx_q10]
            q90_val = q_preds_t[:, idx_q90]
            covered_80 = (y_t >= q10) & (y_t <= q90_val)
            picp_80 = float(np.mean(covered_80) * 100.0)
            mpiw_80 = float(np.mean(q90_val - q10))

            # 50% interval: [q25, q75]
            q25 = q_preds_t[:, idx_q25]
            q75 = q_preds_t[:, idx_q75]
            covered_50 = (y_t >= q25) & (y_t <= q75)
            picp_50 = float(np.mean(covered_50) * 100.0)
            mpiw_50 = float(np.mean(q75 - q25))

            metrics_records.append({
                "target": target_name,
                "horizon": h,
                "mae": round(mae, 4),
                "rmse": round(rmse, 4),
                "r2": round(r2, 6),
                "mean_pinball": round(mean_pinball, 4),
                "picp_90": round(picp_90, 2),
                "mpiw_90": round(mpiw_90, 4),
                "picp_80": round(picp_80, 2),
                "mpiw_80": round(mpiw_80, 4),
                "picp_50": round(picp_50, 2),
                "mpiw_50": round(mpiw_50, 4),
            })

            # Record per-sample predictions for export
            for i_samp in range(N):
                pred_rows.append({
                    "timestamp": all_timestamps[i_samp],
                    "target": target_name,
                    "horizon": f"t+{h}",
                    "actual": float(round(y_t[i_samp], 4)),
                    "median": float(round(y_median[i_samp], 4)),
                    "q05": float(round(q05[i_samp], 4)),
                    "q10": float(round(q10[i_samp], 4)),
                    "q25": float(round(q25[i_samp], 4)),
                    "q50": float(round(y_median[i_samp], 4)),
                    "q75": float(round(q75[i_samp], 4)),
                    "q90": float(round(q90_val[i_samp], 4)),
                    "q95": float(round(q95[i_samp], 4)),
                })

    metrics_df = pd.DataFrame(metrics_records)
    predictions_df = pd.DataFrame(pred_rows)

    return metrics_df, predictions_df


def plot_loss_convergence(
    history: Dict[str, List[float]],
    best_epoch: int,
    save_path: Path,
) -> None:
    """Plot training and validation pinball loss progression with best epoch marker."""
    sns.set_theme(style="darkgrid")
    fig, ax1 = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], label="Train Pinball Loss", color="#3b82f6", linewidth=2.0)
    ax1.plot(epochs, history["val_loss"], label="Val Pinball Loss", color="#f59e0b", linewidth=2.2)

    ax1.axvline(best_epoch, color="#10b981", linestyle="--", alpha=0.85, label=f"Best Checkpoint (Ep {best_epoch})")
    ax1.scatter([best_epoch], [history["val_loss"][best_epoch - 1]], color="#10b981", s=120, zorder=5)

    ax1.set_title("LSTM Multi-Horizon Training Convergence (Pinball Loss)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Training Epoch", fontsize=11)
    ax1.set_ylabel("Multi-Quantile Pinball Loss", fontsize=11)
    ax1.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_horizon_degradation(
    metrics_df: pd.DataFrame,
    save_path: Path,
) -> None:
    """Plot performance degradation (MAE & RMSE) across t+1 -> t+3 -> t+5."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: MAE Growth across Horizons
    ax1 = axes[0]
    sns.barplot(
        data=metrics_df,
        x="target",
        y="mae",
        hue="horizon",
        palette="crest",
        ax=ax1,
    )
    ax1.set_title("Mean Absolute Error (MAE) Degradation: t+1 -> t+3 -> t+5", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Telemetry Target Variable", fontsize=11)
    ax1.set_ylabel("MAE (Physical Units)", fontsize=11)
    ax1.tick_params(axis="x", rotation=30)
    ax1.legend(title="Horizon (s)", loc="upper left")

    # Panel 2: 90% Interval Coverage Calibration
    ax2 = axes[1]
    sns.barplot(
        data=metrics_df,
        x="target",
        y="picp_90",
        hue="horizon",
        palette="magma",
        ax=ax2,
    )
    ax2.axhline(90.0, color="red", linestyle="--", linewidth=1.5, label="Nominal 90% Target")
    ax2.set_title("90% Prediction Interval Coverage Probability (PICP)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Telemetry Target Variable", fontsize=11)
    ax2.set_ylabel("Empirical Coverage (%)", fontsize=11)
    ax2.tick_params(axis="x", rotation=30)
    ax2.set_ylim(0, 100)
    ax2.legend(title="Horizon (s)", loc="lower left")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_prediction_intervals(
    predictions_df: pd.DataFrame,
    save_path: Path,
    targets_to_plot: Optional[List[str]] = None,
    horizon_str: str = "t+1",
    n_steps: int = 180,
) -> None:
    """
    Plot actual vs predicted median with shaded 50% and 90% prediction intervals
    over a representative continuous segment of the test set.
    """
    if targets_to_plot is None:
        targets_to_plot = ["TOF_Alt", "Mag_heading", "pitch", "Magx"]

    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(len(targets_to_plot), 1, figsize=(16, 4 * len(targets_to_plot)), sharex=False)
    if len(targets_to_plot) == 1:
        axes = [axes]

    units = {
        "TOF_Alt": "meters",
        "Mag_heading": "degrees",
        "pitch": "degrees",
        "Magx": "microTesla",
        "Magy": "microTesla",
        "Magz": "microTesla",
        "angle roll": "degrees",
        "yaw": "degrees",
    }

    for ax, tgt in zip(axes, targets_to_plot):
        sub = predictions_df[(predictions_df["target"] == tgt) & (predictions_df["horizon"] == horizon_str)].iloc[:n_steps]
        x_idx = range(len(sub))

        # Shaded 90% interval [q05, q95]
        ax.fill_between(
            x_idx,
            sub["q05"],
            sub["q95"],
            color="#3b82f6",
            alpha=0.25,
            label="90% Prediction Interval [q05 - q95]",
        )
        # Shaded 50% interval [q25, q75]
        ax.fill_between(
            x_idx,
            sub["q25"],
            sub["q75"],
            color="#3b82f6",
            alpha=0.45,
            label="50% Interval (IQR) [q25 - q75]",
        )
        # Median prediction q50
        ax.plot(x_idx, sub["median"], color="#1d4ed8", linewidth=1.8, label="Predicted Median (q50)")
        # Actual ground truth
        ax.plot(x_idx, sub["actual"], color="#ef4444", linewidth=1.6, linestyle="--", label="Actual Ground Truth")

        unit_str = units.get(tgt, "")
        ax.set_title(f"{tgt} Probabilistic Forecast ({horizon_str}) | Scale: {unit_str}", fontsize=12, fontweight="bold")
        ax.set_ylabel(f"{tgt} ({unit_str})", fontsize=10)
        ax.legend(loc="upper right", frameon=True, fontsize=9)

    axes[-1].set_xlabel(f"Test Sequence Timestep (1-second increments)", fontsize=11)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_residuals_distribution(
    predictions_df: pd.DataFrame,
    save_path: Path,
    targets_to_plot: Optional[List[str]] = None,
    horizon_str: str = "t+1",
) -> None:
    """Plot error residual distribution histograms for representative targets."""
    if targets_to_plot is None:
        targets_to_plot = ["TOF_Alt", "Mag_heading", "pitch", "yaw"]

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, tgt in zip(axes, targets_to_plot):
        sub = predictions_df[(predictions_df["target"] == tgt) & (predictions_df["horizon"] == horizon_str)]
        res = sub["actual"] - sub["median"]
        if tgt == "Mag_heading":
            res = ((res + 180.0) % 360.0) - 180.0

        sns.histplot(res, kde=True, color="#3b82f6", bins=30, ax=ax, stat="density", alpha=0.6)
        ax.axvline(0, color="red", linestyle="--", linewidth=1.2)
        ax.set_title(f"Residual Error: {tgt} ({horizon_str})", fontsize=12, fontweight="bold")
        ax.set_xlabel(f"Residual (Actual - Median)", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

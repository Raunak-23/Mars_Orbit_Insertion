"""
Quantile Random Forest (QRF) Evaluation and Visualization Module.

Computes point metrics (MAE, RMSE, R²), quantile pinball losses,
prediction interval coverage (PICP), and interval width (MPIW).
Generates publication-quality visualizations for spacecraft telemetry prediction.
"""

from dataclasses import dataclass
import os
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_pinball_loss(
    y_true: np.ndarray,
    y_pred_q: np.ndarray,
    alpha: float,
    is_angular: bool = False,
) -> float:
  """Compute empirical pinball (quantile) loss for quantile alpha."""
  diff = y_true - y_pred_q
  if is_angular:
    diff = ((diff + 180.0) % 360.0) - 180.0
  loss = np.maximum(alpha * diff, (alpha - 1.0) * diff)
  return float(np.mean(loss))


def compute_coverage(
    y_true: np.ndarray,
    q_lower: np.ndarray,
    q_upper: np.ndarray,
    is_angular: bool = False,
    y_center: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
  """Compute Prediction Interval Coverage Probability (PICP %) and Mean Prediction Interval Width (MPIW)."""
  if is_angular and y_center is not None:
    ang_err = np.abs(((y_true - y_center + 180.0) % 360.0) - 180.0)
    half_width = np.abs(((q_upper - q_lower + 180.0) % 360.0) - 180.0) / 2.0
    inside = ang_err <= half_width
    picp = float(np.mean(inside) * 100.0)
    mpiw = float(np.mean(half_width * 2.0))
  else:
    inside = (y_true >= q_lower) & (y_true <= q_upper)
    picp = float(np.mean(inside) * 100.0)
    mpiw = float(np.mean(q_upper - q_lower))
  return picp, mpiw


def evaluate_predictions(
    y_true: np.ndarray,
    preds_q: np.ndarray,
    quantiles: List[float],
    is_angular: bool = False,
) -> Dict[str, float]:
  """Compute comprehensive probabilistic and point forecast evaluation metrics."""
  q_indices = {q: idx for idx, q in enumerate(quantiles)}
  median_idx = q_indices.get(0.50, len(quantiles) // 2)
  y_median = preds_q[:, median_idx]

  # Point forecast accuracy
  if is_angular:
    ang_diff = ((y_true - y_median + 180.0) % 360.0) - 180.0
    mae = float(np.mean(np.abs(ang_diff)))
    rmse = float(np.sqrt(np.mean(ang_diff**2)))
    # For angular R2, total variance with respect to circular mean
    ss_res = np.sum(ang_diff**2)
    sin_sum = np.sum(np.sin(np.radians(y_true)))
    cos_sum = np.sum(np.cos(np.radians(y_true)))
    circ_mean = np.degrees(np.arctan2(sin_sum, cos_sum)) % 360.0
    tot_diff = ((y_true - circ_mean + 180.0) % 360.0) - 180.0
    ss_tot = np.sum(tot_diff**2)
    r2 = float(1.0 - (ss_res / (ss_tot + 1e-9)))
  else:
    mae = float(mean_absolute_error(y_true, y_median))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_median)))
    r2 = float(r2_score(y_true, y_median))

  # Quantile pinball losses
  pinball_dict = {}
  pinball_losses = []
  for q, idx in q_indices.items():
    loss = compute_pinball_loss(y_true, preds_q[:, idx], q, is_angular=is_angular)
    pinball_dict[f"pinball_q{int(q*100):02d}"] = loss
    pinball_losses.append(loss)
  mean_pinball = float(np.mean(pinball_losses))

  # Prediction Interval Coverage & Width
  # 90% PI (q0.05 to q0.95)
  picp_90, mpiw_90 = None, None
  if 0.05 in q_indices and 0.95 in q_indices:
    picp_90, mpiw_90 = compute_coverage(
        y_true,
        preds_q[:, q_indices[0.05]],
        preds_q[:, q_indices[0.95]],
        is_angular=is_angular,
        y_center=y_median,
    )

  # 80% PI (q0.10 to q0.90)
  picp_80, mpiw_80 = None, None
  if 0.10 in q_indices and 0.90 in q_indices:
    picp_80, mpiw_80 = compute_coverage(
        y_true,
        preds_q[:, q_indices[0.10]],
        preds_q[:, q_indices[0.90]],
        is_angular=is_angular,
        y_center=y_median,
    )

  # 50% PI (q0.25 to q0.75)
  picp_50, mpiw_50 = None, None
  if 0.25 in q_indices and 0.75 in q_indices:
    picp_50, mpiw_50 = compute_coverage(
        y_true,
        preds_q[:, q_indices[0.25]],
        preds_q[:, q_indices[0.75]],
        is_angular=is_angular,
        y_center=y_median,
    )

  results = {
      "mae": mae,
      "rmse": rmse,
      "r2": r2,
      "mean_pinball": mean_pinball,
      "picp_90": picp_90,
      "mpiw_90": mpiw_90,
      "picp_80": picp_80,
      "mpiw_80": mpiw_80,
      "picp_50": picp_50,
      "mpiw_50": mpiw_50,
      **pinball_dict,
  }
  return results


def plot_trajectory_with_uncertainty(
    test_df: pd.DataFrame,
    preds_dict: Dict[str, np.ndarray],
    target_name: str,
    output_dir: str = "reports/figures",
) -> str:
  """Plot actual vs predicted trajectory with 50%, 80%, and 90% prediction intervals across horizons."""
  os.makedirs(output_dir, exist_ok=True)
  plt.style.use("seaborn-v0_8-whitegrid")

  fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
  time_steps = test_df["elapsed_sec"].values

  horizons = [1, 3, 5]
  quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
  q_idx = {q: i for i, q in enumerate(quantiles)}

  unit = "deg" if target_name == "Mag_heading" else "meters"

  for ax_idx, h in enumerate(horizons):
    ax = axes[ax_idx]
    y_true = test_df[f"target_{target_name}_t+{h}"].values
    preds = preds_dict[h]

    # Mask any NaNs in target if applicable
    valid = ~np.isnan(y_true)
    t = time_steps[valid]
    y_act = y_true[valid]
    p = preds[valid]

    # Plot uncertainty bands
    # 90% interval [q05, q95]
    ax.fill_between(
        t,
        p[:, q_idx[0.05]],
        p[:, q_idx[0.95]],
        color="#3b82f6",
        alpha=0.20,
        label="90% Prediction Interval (q05-q95)",
    )
    # 80% interval [q10, q90]
    ax.fill_between(
        t,
        p[:, q_idx[0.10]],
        p[:, q_idx[0.90]],
        color="#2563eb",
        alpha=0.25,
        label="80% Prediction Interval (q10-q90)",
    )
    # 50% interval [q25, q75]
    ax.fill_between(
        t,
        p[:, q_idx[0.25]],
        p[:, q_idx[0.75]],
        color="#1d4ed8",
        alpha=0.35,
        label="50% Prediction Interval (q25-q75)",
    )

    # Median prediction
    ax.plot(
        t,
        p[:, q_idx[0.50]],
        color="#dc2626",
        linestyle="--",
        linewidth=1.8,
        label="Predicted Median (q50)",
    )

    # Actual target
    ax.plot(
        t,
        y_act,
        color="#0f172a",
        linewidth=1.4,
        alpha=0.85,
        label=f"Actual {target_name}",
    )

    ax.set_title(
        f"Target: {target_name} | Horizon t+{h}s ahead | Out-of-Time Test Set",
        fontsize=12,
        fontweight="bold",
        pad=8,
    )
    ax.set_ylabel(f"{target_name} ({unit})", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    if ax_idx == 0:
      ax.legend(loc="upper right", framealpha=0.92, fontsize=9)

  axes[-1].set_xlabel("Elapsed Mission Time (seconds)", fontsize=11)
  fig.suptitle(
      f"Quantile Random Forest Multi-Horizon Probabilistic Trajectory Forecast: {target_name}",
      fontsize=14,
      fontweight="bold",
      y=0.995,
  )
  plt.tight_layout()
  save_path = os.path.join(output_dir, f"qrf_{target_name}_trajectory_intervals.png")
  plt.savefig(save_path, dpi=300, bbox_inches="tight")
  plt.close()
  return save_path


def plot_residuals_distribution(
    test_df: pd.DataFrame,
    preds_dict: Dict[str, np.ndarray],
    target_name: str,
    output_dir: str = "reports/figures",
) -> str:
  """Plot residual distributions (histogram, KDE, and boxplots) across horizons."""
  os.makedirs(output_dir, exist_ok=True)
  plt.style.use("seaborn-v0_8-whitegrid")

  fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
  horizons = [1, 3, 5]
  palette = ["#0284c7", "#7c3aed", "#e11d48"]
  unit = "deg" if target_name == "Mag_heading" else "m"

  for idx, h in enumerate(horizons):
    ax = axes[idx]
    y_true = test_df[f"target_{target_name}_t+{h}"].values
    valid = ~np.isnan(y_true)
    y_median = preds_dict[h][valid, 3]  # q0.50 index
    residuals = y_true[valid] - y_median

    sns.histplot(
        residuals,
        kde=True,
        ax=ax,
        color=palette[idx],
        stat="density",
        alpha=0.45,
        bins=35,
    )
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)

    ax.axvline(
        0, color="#1e293b", linestyle="--", linewidth=1.5, label="Zero Bias"
    )
    ax.axvline(
        mean_res,
        color="#dc2626",
        linestyle=":",
        linewidth=1.5,
        label=f"Mean: {mean_res:.2f}",
    )

    ax.set_title(
        f"Horizon t+{h}s Residuals\nMean: {mean_res:.3f} | Std: {std_res:.3f} {unit}",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel(f"Residual (Actual - Median Pred) [{unit}]", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.legend(loc="upper right", fontsize=9)

  fig.suptitle(
      f"Quantile Random Forest Residual Distributions across Horizons: {target_name}",
      fontsize=13,
      fontweight="bold",
      y=1.02,
  )
  plt.tight_layout()
  save_path = os.path.join(output_dir, f"qrf_{target_name}_residual_distribution.png")
  plt.savefig(save_path, dpi=300, bbox_inches="tight")
  plt.close()
  return save_path


def plot_horizon_comparison(
    metrics_summary: pd.DataFrame,
    output_dir: str = "reports/figures",
) -> str:
  """Plot metric degradation curves (MAE, RMSE, Pinball Loss, Interval Width) vs Horizon."""
  os.makedirs(output_dir, exist_ok=True)
  plt.style.use("seaborn-v0_8-whitegrid")

  fig, axes = plt.subplots(2, 2, figsize=(14, 10))
  targets = metrics_summary["target"].unique()
  colors = {"Mag_heading": "#0284c7", "TOF_Alt": "#10b981"}

  # 1. MAE vs Horizon
  ax1 = axes[0, 0]
  for tgt in targets:
    sub = metrics_summary[metrics_summary["target"] == tgt]
    ax1.plot(
        sub["horizon"],
        sub["mae"],
        marker="o",
        linewidth=2.2,
        markersize=7,
        label=f"{tgt}",
        color=colors.get(tgt, "#6366f1"),
    )
  ax1.set_title("Median MAE vs Prediction Horizon", fontsize=12, fontweight="bold")
  ax1.set_xlabel("Horizon (seconds ahead)", fontsize=10)
  ax1.set_ylabel("Mean Absolute Error (target units)", fontsize=10)
  ax1.set_xticks([1, 3, 5])
  ax1.legend(fontsize=9)

  # 2. RMSE vs Horizon
  ax2 = axes[0, 1]
  for tgt in targets:
    sub = metrics_summary[metrics_summary["target"] == tgt]
    ax2.plot(
        sub["horizon"],
        sub["rmse"],
        marker="s",
        linewidth=2.2,
        markersize=7,
        label=f"{tgt}",
        color=colors.get(tgt, "#6366f1"),
    )
  ax2.set_title("Median RMSE vs Prediction Horizon", fontsize=12, fontweight="bold")
  ax2.set_xlabel("Horizon (seconds ahead)", fontsize=10)
  ax2.set_ylabel("Root Mean Squared Error", fontsize=10)
  ax2.set_xticks([1, 3, 5])
  ax2.legend(fontsize=9)

  # 3. Mean Pinball Loss vs Horizon
  ax3 = axes[1, 0]
  for tgt in targets:
    sub = metrics_summary[metrics_summary["target"] == tgt]
    ax3.plot(
        sub["horizon"],
        sub["mean_pinball"],
        marker="^",
        linewidth=2.2,
        markersize=7,
        label=f"{tgt}",
        color=colors.get(tgt, "#6366f1"),
    )
  ax3.set_title("Mean Quantile Pinball Loss vs Horizon", fontsize=12, fontweight="bold")
  ax3.set_xlabel("Horizon (seconds ahead)", fontsize=10)
  ax3.set_ylabel("Mean Pinball Loss across 7 Quantiles", fontsize=10)
  ax3.set_xticks([1, 3, 5])
  ax3.legend(fontsize=9)

  # 4. 90% Prediction Interval Coverage (PICP) vs Horizon
  ax4 = axes[1, 1]
  for tgt in targets:
    sub = metrics_summary[metrics_summary["target"] == tgt]
    ax4.plot(
        sub["horizon"],
        sub["picp_90"],
        marker="d",
        linewidth=2.2,
        markersize=7,
        label=f"{tgt} (Nominal: 90%)",
        color=colors.get(tgt, "#6366f1"),
    )
  ax4.axhline(90.0, color="#dc2626", linestyle="--", linewidth=1.5, label="Nominal 90% Target")
  ax4.set_title("90% Prediction Interval Coverage (PICP)", fontsize=12, fontweight="bold")
  ax4.set_xlabel("Horizon (seconds ahead)", fontsize=10)
  ax4.set_ylabel("Empirical Coverage (%)", fontsize=10)
  ax4.set_xticks([1, 3, 5])
  ax4.set_ylim([80, 100])
  ax4.legend(fontsize=9)

  fig.suptitle(
      "Performance and Uncertainty Degradation Across Forecast Horizons (t+1, t+3, t+5)",
      fontsize=14,
      fontweight="bold",
      y=0.995,
  )
  plt.tight_layout()
  save_path = os.path.join(output_dir, "qrf_horizon_metrics_comparison.png")
  plt.savefig(save_path, dpi=300, bbox_inches="tight")
  plt.close()
  return save_path

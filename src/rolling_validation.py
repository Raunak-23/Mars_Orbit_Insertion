"""
Rolling Temporal Validation Module for Spacecraft Telemetry Quantile Random Forest.

Implements expanding-window (walk-forward) time-series cross-validation
to evaluate model robustness and uncertainty calibration across mission phases:
1. Pre-Burn Approach
2. LAM Main Engine Firing (periapsis insertion)
3. Post-Burn Orbital Ascent
"""

import os
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.qrf_evaluation import evaluate_predictions
from src.qrf_pipeline import (
    DEFAULT_QUANTILES,
    FEATURE_COLS,
    HORIZONS,
    TARGET_COLS,
    QuantileRandomForestPipeline,
    TelemetryFeaturePipeline,
    load_best_hyperparameters,
)


class TemporalRollingValidator:
  """Expanding-window rolling temporal cross-validation for spacecraft telemetry."""

  def __init__(
      self,
      n_splits: int = 5,
      min_train_ratio: float = 0.40,
      horizons: List[int] = HORIZONS,
      targets: List[str] = TARGET_COLS,
      quantiles: List[float] = DEFAULT_QUANTILES,
      random_state: int = 42,
  ):
    self.n_splits = n_splits
    self.min_train_ratio = min_train_ratio
    self.horizons = horizons
    self.targets = targets
    self.quantiles = quantiles
    self.random_state = random_state

  def generate_temporal_splits(
      self, df: pd.DataFrame
  ) -> List[Tuple[pd.DataFrame, pd.DataFrame, int, Tuple[float, float], Tuple[float, float]]]:
    """Generate expanding window (walk-forward) temporal splits respecting chronological order.

    Returns list of: (train_df, test_df, fold_idx, (train_start_t, train_end_t), (test_start_t, test_end_t))
    """
    n_total = len(df)
    min_train_size = int(n_total * self.min_train_ratio)
    remaining_size = n_total - min_train_size
    test_size = remaining_size // self.n_splits

    splits = []
    for i in range(self.n_splits):
      train_end = min_train_size + i * test_size
      test_end = train_end + test_size if i < self.n_splits - 1 else n_total

      train_df = df.iloc[:train_end].copy()
      test_df = df.iloc[train_end:test_end].copy()

      train_time_range = (
          train_df["elapsed_sec"].min(),
          train_df["elapsed_sec"].max(),
      )
      test_time_range = (
          test_df["elapsed_sec"].min(),
          test_df["elapsed_sec"].max(),
      )
      splits.append((train_df, test_df, i + 1, train_time_range, test_time_range))

    return splits

  def run_validation(
      self,
      full_df: pd.DataFrame,
      n_estimators: int = 120,
      max_depth: int = 15,
      verbose: bool = True,
  ) -> Dict[str, pd.DataFrame]:
    """Execute rolling validation across all folds, targets, and horizons."""
    splits = self.generate_temporal_splits(full_df)
    if verbose:
      print(f"Executing {self.n_splits}-Fold Expanding Window Rolling Validation:")
      for _, _, fold, (tr_s, tr_e), (te_s, te_e) in splits:
        print(
            f"  Fold {fold}: Train [0s -> {tr_e:.0f}s] ({tr_e:.0f} samples) | "
            f"Test [{te_s:.0f}s -> {te_e:.0f}s] ({te_e - te_s:.0f} samples)"
        )

    all_fold_metrics = []
    oof_prediction_records = []

    best_params_map = load_best_hyperparameters()
    if best_params_map and verbose:
      print("  Using tuned hyperparameters from models/best_hyperparameters.json for rolling validation.")

    for train_df, test_df, fold, tr_span, te_span in splits:
      if verbose:
        print(f"\n--- Training Fold {fold}/{self.n_splits} (Test: t={te_span[0]:.0f}s to {te_span[1]:.0f}s) ---")

      for target in self.targets:
        for h in self.horizons:
          model_tag = f"{target}_t+{h}"
          tuned_params = best_params_map.get(model_tag, {})
          default_depth = 8 if target == "TOF_Alt" else 10
          default_leaf = 15 if target == "TOF_Alt" else 10
          default_feats = 0.7 if target == "TOF_Alt" else "sqrt"

          model = QuantileRandomForestPipeline(
              target_name=target,
              horizon=h,
              n_estimators=tuned_params.get("n_estimators", n_estimators),
              max_depth=tuned_params.get("max_depth", default_depth),
              min_samples_leaf=tuned_params.get("min_samples_leaf", default_leaf),
              min_samples_split=tuned_params.get("min_samples_split", 16),
              max_features=tuned_params.get("max_features", default_feats),
              max_samples=tuned_params.get("max_samples", 0.85),
              random_state=self.random_state,
              quantiles=self.quantiles,
              params=tuned_params,
          )
          model.fit(train_df)

          # Predict on out-of-fold test slice
          valid_test = test_df.dropna(subset=FEATURE_COLS + [f"target_{target}_t+{h}"])
          if len(valid_test) == 0:
            continue

          preds_test = model.predict_quantiles(valid_test)
          y_test_true = valid_test[f"target_{target}_t+{h}"].values

          metrics = evaluate_predictions(
              y_test_true, preds_test, self.quantiles, is_angular=(target == "Mag_heading")
          )
          metrics["fold"] = fold
          metrics["target"] = target
          metrics["horizon"] = h
          metrics["train_span_sec"] = f"{tr_span[0]:.0f}-{tr_span[1]:.0f}"
          metrics["test_span_sec"] = f"{te_span[0]:.0f}-{te_span[1]:.0f}"
          metrics["test_samples"] = len(valid_test)
          all_fold_metrics.append(metrics)

          # Record out-of-fold predictions
          for idx, row_idx in enumerate(valid_test.index):
            rec = {
                "fold": fold,
                "utc_time": valid_test.loc[row_idx, "utc_time"].isoformat(),
                "elapsed_sec": valid_test.loc[row_idx, "elapsed_sec"],
                "target": target,
                "horizon": f"t+{h}",
                "actual": y_test_true[idx],
                "pred_median": preds_test[idx, 3],  # q0.50
                "pred_q05": preds_test[idx, 0],
                "pred_q10": preds_test[idx, 1],
                "pred_q25": preds_test[idx, 2],
                "pred_q50": preds_test[idx, 3],
                "pred_q75": preds_test[idx, 4],
                "pred_q90": preds_test[idx, 5],
                "pred_q95": preds_test[idx, 6],
            }
            oof_prediction_records.append(rec)

    metrics_df = pd.DataFrame(all_fold_metrics)
    oof_df = pd.DataFrame(oof_prediction_records)

    return {"fold_metrics": metrics_df, "oof_predictions": oof_df}

  def plot_rolling_metrics(
      self,
      metrics_df: pd.DataFrame,
      output_dir: str = "reports/figures",
  ) -> str:
    """Plot metric evolution across rolling temporal folds."""
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. MAE by Fold
    ax1 = axes[0, 0]
    sns.lineplot(
        data=metrics_df,
        x="fold",
        y="mae",
        hue="target",
        style="horizon",
        markers=True,
        dashes=False,
        ax=ax1,
        linewidth=2.0,
        markersize=8,
    )
    ax1.set_title("Median MAE Across Rolling Folds", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Rolling Fold Index", fontsize=10)
    ax1.set_ylabel("MAE (target units)", fontsize=10)
    ax1.set_xticks(range(1, self.n_splits + 1))

    # 2. Mean Pinball Loss by Fold
    ax2 = axes[0, 1]
    sns.lineplot(
        data=metrics_df,
        x="fold",
        y="mean_pinball",
        hue="target",
        style="horizon",
        markers=True,
        dashes=False,
        ax=ax2,
        linewidth=2.0,
        markersize=8,
    )
    ax2.set_title("Mean Pinball Loss Across Rolling Folds", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Rolling Fold Index", fontsize=10)
    ax2.set_ylabel("Mean Quantile Loss", fontsize=10)
    ax2.set_xticks(range(1, self.n_splits + 1))

    # 3. 90% Interval Coverage (PICP) by Fold
    ax3 = axes[1, 0]
    sns.lineplot(
        data=metrics_df,
        x="fold",
        y="picp_90",
        hue="target",
        style="horizon",
        markers=True,
        dashes=False,
        ax=ax3,
        linewidth=2.0,
        markersize=8,
    )
    ax3.axhline(90.0, color="#dc2626", linestyle="--", linewidth=1.5, label="Nominal 90% Target")
    ax3.set_title("90% Prediction Interval Coverage (PICP)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Rolling Fold Index", fontsize=10)
    ax3.set_ylabel("Empirical Coverage (%)", fontsize=10)
    ax3.set_xticks(range(1, self.n_splits + 1))
    ax3.set_ylim([-5, 105])

    # 4. 90% Interval Width (MPIW) by Fold
    ax4 = axes[1, 1]
    sns.lineplot(
        data=metrics_df,
        x="fold",
        y="mpiw_90",
        hue="target",
        style="horizon",
        markers=True,
        dashes=False,
        ax=ax4,
        linewidth=2.0,
        markersize=8,
    )
    ax4.set_title("90% Prediction Interval Width (MPIW)", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Rolling Fold Index", fontsize=10)
    ax4.set_ylabel("Interval Width (q95 - q05)", fontsize=10)
    ax4.set_xticks(range(1, self.n_splits + 1))

    fig.suptitle(
        "Expanding-Window Rolling Temporal Validation Performance Dynamics",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    save_path = os.path.join(output_dir, "qrf_rolling_validation_dynamics.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    return save_path

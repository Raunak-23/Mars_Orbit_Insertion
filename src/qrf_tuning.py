"""
Hyperparameter Tuning Module for Spacecraft Telemetry Quantile Random Forest.

Executes RandomizedSearchCV over a regularized hyperparameter space using
TimeSeriesSplit to optimize forecast accuracy (MAE / Pinball Loss) while enforcing
strict generalization constraints to eliminate tree overfitting.
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
  sys.path.insert(0, project_root)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from quantile_forest import RandomForestQuantileRegressor
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from src.qrf_pipeline import (
    FEATURE_COLS,
    HORIZONS,
    TARGET_COLS,
    TelemetryFeaturePipeline,
    resolve_dataset_path,
)

# Expanded hyperparameter search space covering model capacity, tree regularization, and subsampling
PARAM_DISTRIBUTIONS = {
    "n_estimators": [80, 120, 160, 200, 250, 300],
    "max_depth": [6, 8, 10, 12, 14, 18, 22, None],
    "min_samples_split": [4, 8, 12, 16, 24, 32, 48],
    "min_samples_leaf": [2, 4, 6, 10, 15, 20, 30],
    "max_features": ["sqrt", "log2", 0.40, 0.55, 0.70, 0.85, 1.0],
    "max_samples": [0.60, 0.75, 0.85, 0.95, None],
    "bootstrap": [True],
}


def select_best_generalized_params(
    cv_results: pd.DataFrame,
    lambda_std: float = 0.20,
    lambda_gap: float = 0.50,
) -> Tuple[Dict[str, Any], pd.Series]:
  """Select optimal hyperparameters balancing low CV MAE and minimal generalization gap.

  Penalizes models that overfit (large validation-to-train error gap or extreme ratio)
  to ensure robust generalization on unseen out-of-time telemetry.
  """
  scores = []
  for idx, row in cv_results.iterrows():
    val_mae = -row["mean_test_score"]
    val_std = row["std_test_score"]
    tr_mae = -row["mean_train_score"]
    gap = max(0.0, val_mae - tr_mae)
    ratio = val_mae / (tr_mae + 1e-6)

    # Generalization-penalized score:
    # 1. Validation MAE (primary accuracy)
    # 2. Validation fold stability (std)
    # 3. Absolute generalization gap (val_mae - tr_mae)
    # 4. Overfitting ratio penalty if test error is disproportionately higher than train
    ratio_penalty = max(0.0, ratio - 1.2)
    gen_score = val_mae + lambda_std * val_std + lambda_gap * gap + 0.15 * ratio_penalty * val_mae
    scores.append((gen_score, val_mae, tr_mae, gap, idx))

  scores.sort(key=lambda x: x[0])
  best_idx = scores[0][4]
  return cv_results.loc[best_idx, "params"], cv_results.loc[best_idx]


def plot_tuning_tradeoffs(
    tuning_df: pd.DataFrame,
    output_dir: str = "reports/figures",
) -> str:
  """Plot hyperparameter search tradeoff curves: Train MAE vs Validation MAE and Generalization Gap."""
  os.makedirs(output_dir, exist_ok=True)
  plt.style.use("seaborn-v0_8-whitegrid")

  targets = tuning_df["target"].unique()
  fig, axes = plt.subplots(len(targets), 3, figsize=(20, 6 * len(targets)), sharex=False)
  if len(targets) == 1:
    axes = axes[None, :]

  for row_idx, tgt in enumerate(targets):
    sub_tgt = tuning_df[tuning_df["target"] == tgt]
    for col_idx, h in enumerate(HORIZONS):
      ax = axes[row_idx, col_idx]
      sub_h = sub_tgt[sub_tgt["horizon"] == h].copy()
      if sub_h.empty:
        continue

      sub_h["gen_gap"] = sub_h["mean_test_mae"] - sub_h["mean_train_mae"]
      best_cand = sub_h.loc[sub_h["is_best"] == True]

      # Handle depth values for colorbar (replace None with 25 for visual scale)
      depth_vals = sub_h["max_depth"].fillna(25).astype(float)

      # Scatter of candidates: Train MAE vs Validation MAE
      scatter = ax.scatter(
          sub_h["mean_train_mae"],
          sub_h["mean_test_mae"],
          c=depth_vals,
          cmap="viridis",
          s=90,
          alpha=0.80,
          edgecolors="k",
          linewidths=0.8,
          label="Search Candidates",
          zorder=3,
      )
      cbar = fig.colorbar(scatter, ax=ax)
      cbar.set_label("Max Depth (None ~ 25)", fontsize=9)

      # Plot 1:1 line (ideal generalization: Val MAE == Train MAE)
      min_val = min(sub_h["mean_train_mae"].min(), sub_h["mean_test_mae"].min()) * 0.9
      max_val = max(sub_h["mean_train_mae"].max(), sub_h["mean_test_mae"].max()) * 1.1
      diag_line = np.linspace(min_val, max_val, 100)
      ax.plot(
          diag_line,
          diag_line,
          color="#64748b",
          linestyle="--",
          linewidth=1.5,
          alpha=0.7,
          label="Ideal Generalization (y=x)",
          zorder=2,
      )

      # Highlight selected generalized model
      if not best_cand.empty:
        best_tr = best_cand["mean_train_mae"].iloc[0]
        best_val = best_cand["mean_test_mae"].iloc[0]
        best_gap = best_cand["gen_gap"].iloc[0]
        ax.scatter(
            best_tr,
            best_val,
            color="#ef4444",
            s=260,
            marker="*",
            edgecolors="#7f1d1d",
            linewidth=1.8,
            label=f"Selected (Gap: {best_gap:.3f})",
            zorder=5,
        )
        ax.annotate(
            f"Best Model\nTrain: {best_tr:.3f}\nVal: {best_val:.3f}",
            (best_tr, best_val),
            textcoords="offset points",
            xytext=(15, -15),
            ha="left",
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="#fef2f2", ec="#ef4444", lw=1.2),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", color="#b91c1c"),
        )

      unit = "m" if tgt == "TOF_Alt" else "deg"
      ax.set_title(
          f"{tgt} (Horizon t+{h}s) - Train vs Validation MAE [{unit}]\nTrade-Off & Overfitting Control",
          fontsize=11,
          fontweight="bold",
      )
      ax.set_xlabel(f"Mean Train MAE ({unit})", fontsize=9.5)
      ax.set_ylabel(f"Mean Validation MAE ({unit})", fontsize=9.5)
      ax.set_xlim(min_val, max_val)
      ax.set_ylim(min_val, max_val)
      ax.grid(True, linestyle=":", alpha=0.6)
      if row_idx == 0 and col_idx == 0:
        ax.legend(loc="upper left", fontsize=8.5)

  fig.suptitle(
      "RandomizedSearchCV Hyperparameter Tuning: Train vs Validation Error Trade-Off",
      fontsize=14,
      fontweight="bold",
      y=0.995,
  )
  plt.tight_layout()
  save_path = os.path.join(output_dir, "qrf_hyperparameter_tuning_tradeoffs.png")
  plt.savefig(save_path, dpi=300, bbox_inches="tight")
  plt.close()
  return save_path


def run_hyperparameter_tuning(
    dataset_path: Optional[str] = None,
    n_iter: int = 16,
    n_splits: int = 4,
    output_dir: str = "reports",
    models_dir: str = "models",
    random_state: int = 42,
) -> Dict[str, Any]:
  """Run RandomizedSearchCV with TimeSeriesSplit across targets and horizons."""
  os.makedirs(output_dir, exist_ok=True)
  os.makedirs(models_dir, exist_ok=True)

  print("=" * 80)
  print("HYPERPARAMETER TUNING: GENERALIZATION-CONSTRAINED RANDOMIZED SEARCH CV")
  print("=" * 80)

  pipeline = TelemetryFeaturePipeline(feature_cols=FEATURE_COLS, use_residual=True)
  resolved_path = resolve_dataset_path(dataset_path)
  full_df = pipeline.prepare_dataset(file_path=resolved_path)

  # Use chronological training + validation window (85%) for walk-forward CV, keeping test (15%) pure
  train_pool_df, _, test_df = pipeline.chronological_split(
      full_df, train_ratio=0.70, val_ratio=0.15
  )
  tuning_df_pool = pd.concat([train_pool_df, _]).sort_values("utc_time").reset_index(drop=True)
  print(f"Tuning on walk-forward temporal pool: {len(tuning_df_pool)} samples (Holdout test: {len(test_df)} samples)")

  best_params_all = {}
  tuning_history_records = []

  tscv = TimeSeriesSplit(n_splits=n_splits)

  for target in TARGET_COLS:
    for h in HORIZONS:
      target_tag = f"{target}_t+{h}"
      delta_target_col = f"delta_target_{target}_t+{h}"
      print(f"\n--- Tuning Hyperparameters for {target_tag} (Iterations: {n_iter}, CV Folds: {n_splits}) ---")

      valid_train = tuning_df_pool.dropna(subset=FEATURE_COLS + [delta_target_col])
      X = valid_train[FEATURE_COLS].values
      y = valid_train[delta_target_col].values

      # Scale features
      X_scaled = pipeline.scaler.fit_transform(X)

      base_qrf = RandomForestQuantileRegressor(
          random_state=random_state, n_jobs=-1
      )

      search = RandomizedSearchCV(
          estimator=base_qrf,
          param_distributions=PARAM_DISTRIBUTIONS,
          n_iter=n_iter,
          cv=tscv,
          scoring="neg_mean_absolute_error",
          random_state=random_state,
          n_jobs=1,
          verbose=1,
          return_train_score=True,
      )

      search.fit(X_scaled, y)

      # Apply generalization-penalized selection
      cv_res = pd.DataFrame(search.cv_results_)
      best_params, best_row = select_best_generalized_params(cv_res)
      best_params_all[target_tag] = best_params

      print(f"  Selected Model Mean CV MAE: {-best_row['mean_test_score']:.4f}")
      print(f"  Selected Model Train MAE:   {-best_row['mean_train_score']:.4f}")
      print(f"  Generalization Gap:         {-best_row['mean_test_score'] - (-best_row['mean_train_score']):.4f}")
      print(f"  Selected Parameters:        {best_params}")

      # Record results
      for idx, row in cv_res.iterrows():
        is_chosen = (row["params"] == best_params)
        rec = {
            "target": target,
            "horizon": h,
            "iter": idx + 1,
            "mean_test_mae": -row["mean_test_score"],
            "std_test_mae": row["std_test_score"],
            "mean_train_mae": -row["mean_train_score"],
            "gen_gap": -row["mean_test_score"] - (-row["mean_train_score"]),
            "n_estimators": row["params"].get("n_estimators", 150),
            "max_depth": row["params"].get("max_depth", None),
            "min_samples_leaf": row["params"].get("min_samples_leaf", 5),
            "min_samples_split": row["params"].get("min_samples_split", 16),
            "max_features": row["params"].get("max_features", "sqrt"),
            "max_samples": row["params"].get("max_samples", None),
            "is_best": is_chosen,
            "params": json.dumps(row["params"]),
        }
        tuning_history_records.append(rec)

  # Export tuning logs
  tuning_history_df = pd.DataFrame(tuning_history_records)
  tuning_csv_path = os.path.join(output_dir, "qrf_tuning_results.csv")
  tuning_history_df.to_csv(tuning_csv_path, index=False)
  print(f"\nSaved hyperparameter search results to: {tuning_csv_path}")

  # Export best params json
  best_params_json_path = os.path.join(models_dir, "best_hyperparameters.json")
  with open(best_params_json_path, "w", encoding="utf-8") as f:
    json.dump(best_params_all, f, indent=2)
  print(f"Saved best hyperparameters to: {best_params_json_path}")

  # Plot tradeoffs
  plot_path = plot_tuning_tradeoffs(tuning_history_df, output_dir=os.path.join(output_dir, "figures"))
  print(f"Saved hyperparameter tuning tradeoff plots to: {plot_path}")

  return {"best_params": best_params_all, "tuning_history": tuning_history_df}


if __name__ == "__main__":
  run_hyperparameter_tuning(n_iter=16, n_splits=4)


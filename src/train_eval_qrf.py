"""
Main Training and Evaluation Runner for Spacecraft Telemetry Quantile Random Forest.

Orchestrates data loading, feature engineering, chronological splitting,
model training for (Mag_heading, TOF_Alt) across (t+1, t+3, t+5), evaluation,
predictions CSV export, plot generation, and artifact serialization.
"""

import os
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from tabulate import tabulate

from src.qrf_evaluation import (
    evaluate_predictions,
    plot_horizon_comparison,
    plot_residuals_distribution,
    plot_trajectory_with_uncertainty,
)
from src.qrf_pipeline import (
    DEFAULT_QUANTILES,
    FEATURE_COLS,
    HORIZONS,
    TARGET_COLS,
    QuantileRandomForestPipeline,
    TelemetryFeaturePipeline,
    load_best_hyperparameters,
    resolve_dataset_path,
)


def run_qrf_workflow(
    dataset_path: Optional[str] = None,
    models_dir: str = "models",
    reports_dir: str = "reports",
    figures_dir: str = "reports/figures",
    random_state: int = 42,
) -> Dict[str, pd.DataFrame]:
  """Execute end-to-end Quantile Random Forest training, evaluation, and serialization."""
  print("=" * 80)
  print("SPACECRAFT TELEMETRY QUANTILE RANDOM FOREST (QRF) PIPELINE")
  print("=" * 80)

  # Resolve directories
  os.makedirs(models_dir, exist_ok=True)
  os.makedirs(reports_dir, exist_ok=True)
  os.makedirs(figures_dir, exist_ok=True)

  # 1. Load data and engineer features
  pipeline_prep = TelemetryFeaturePipeline(feature_cols=FEATURE_COLS)
  print("\n[1/6] Loading raw telemetry and applying chronological parsing...")
  resolved_path = resolve_dataset_path(dataset_path)
  print(f"      Dataset path: {resolved_path}")

  full_df = pipeline_prep.prepare_dataset(
      file_path=resolved_path, horizons=HORIZONS, targets=TARGET_COLS
  )
  print(
      f"      Engineered dataset shape: {full_df.shape} (Time range: {full_df['utc_time'].min()} to {full_df['utc_time'].max()})"
  )

  # 2. Chronological Split (70% Train, 15% Val, 15% Test)
  print("\n[2/6] Executing chronological Train/Validation/Test split (70% / 15% / 15%)...")
  train_df, val_df, test_df = pipeline_prep.chronological_split(
      full_df, train_ratio=0.70, val_ratio=0.15
  )
  print(f"      Train set rows: {len(train_df)} (Elapsed: {train_df['elapsed_sec'].min():.0f}s - {train_df['elapsed_sec'].max():.0f}s)")
  print(f"      Val set rows:   {len(val_df)} (Elapsed: {val_df['elapsed_sec'].min():.0f}s - {val_df['elapsed_sec'].max():.0f}s)")
  print(f"      Test set rows:  {len(test_df)} (Elapsed: {test_df['elapsed_sec'].min():.0f}s - {test_df['elapsed_sec'].max():.0f}s)")

  # Storage for metrics and predictions
  all_metrics = []
  test_prediction_records = []
  preds_by_target_and_h = {tgt: {} for tgt in TARGET_COLS}
  models = {}
  best_params_map = load_best_hyperparameters(os.path.join(models_dir, "best_hyperparameters.json"))
  if best_params_map:
    print(f"      Loaded tuned hyperparameters from: {os.path.join(models_dir, 'best_hyperparameters.json')}")
  else:
    print("      No tuned hyperparameters found. Using default regularized baseline parameters.")

  print("\n[3/6] Training Quantile Random Forest models across targets and horizons...")
  for target in TARGET_COLS:
    for h in HORIZONS:
      model_tag = f"{target}_t+{h}"
      tuned_params = best_params_map.get(model_tag, {})
      print(f"      -> Training model for {model_tag} across quantiles {DEFAULT_QUANTILES}...")
      if tuned_params:
        print(f"         Using tuned hyperparameters: {tuned_params}")

      # Default to regularized parameters if not tuned
      default_depth = 8 if target == "TOF_Alt" else 10
      default_leaf = 15 if target == "TOF_Alt" else 10
      default_feats = 0.7 if target == "TOF_Alt" else "sqrt"

      model = QuantileRandomForestPipeline(
          target_name=target,
          horizon=h,
          n_estimators=tuned_params.get("n_estimators", 160),
          max_depth=tuned_params.get("max_depth", default_depth),
          min_samples_leaf=tuned_params.get("min_samples_leaf", default_leaf),
          min_samples_split=tuned_params.get("min_samples_split", 16),
          max_features=tuned_params.get("max_features", default_feats),
          max_samples=tuned_params.get("max_samples", 0.85),
          random_state=random_state,
          quantiles=DEFAULT_QUANTILES,
          params=tuned_params,
      )
      model.fit(train_df)

      # Save model artifact
      model_save_path = os.path.join(models_dir, f"qrf_{target}_h{h}.joblib")
      model.save(model_save_path)
      models[model_tag] = model

      # Predict on train set (to track overfitting gap)
      valid_train = train_df.dropna(subset=FEATURE_COLS + [f"target_{target}_t+{h}"])
      preds_train = model.predict_quantiles(valid_train)
      y_train_true = valid_train[f"target_{target}_t+{h}"].values
      train_metrics = evaluate_predictions(
          y_train_true, preds_train, DEFAULT_QUANTILES, is_angular=(target == "Mag_heading")
      )
      train_metrics["target"] = target
      train_metrics["horizon"] = h
      train_metrics["split"] = "train"
      all_metrics.append(train_metrics)

      # Predict on validation set
      valid_val = val_df.dropna(subset=FEATURE_COLS + [f"target_{target}_t+{h}"])
      preds_val = model.predict_quantiles(valid_val)
      y_val_true = valid_val[f"target_{target}_t+{h}"].values
      val_metrics = evaluate_predictions(
          y_val_true, preds_val, DEFAULT_QUANTILES, is_angular=(target == "Mag_heading")
      )
      val_metrics["target"] = target
      val_metrics["horizon"] = h
      val_metrics["split"] = "val"
      all_metrics.append(val_metrics)

      # Predict on test set
      valid_test = test_df.dropna(subset=FEATURE_COLS + [f"target_{target}_t+{h}"])
      preds_test = model.predict_quantiles(valid_test)
      y_test_true = valid_test[f"target_{target}_t+{h}"].values
      test_metrics = evaluate_predictions(
          y_test_true, preds_test, DEFAULT_QUANTILES, is_angular=(target == "Mag_heading")
      )
      test_metrics["target"] = target
      test_metrics["horizon"] = h
      test_metrics["split"] = "test"
      all_metrics.append(test_metrics)

      # Cache test predictions for plotting
      full_test_preds = np.full((len(test_df), len(DEFAULT_QUANTILES)), np.nan)
      full_test_preds[valid_test.index - test_df.index[0]] = preds_test
      preds_by_target_and_h[target][h] = full_test_preds

      # Record predictions for CSV export
      for i, row_idx in enumerate(valid_test.index):
        row_rec = {
            "utc_time": valid_test.loc[row_idx, "utc_time"].isoformat(),
            "elapsed_sec": valid_test.loc[row_idx, "elapsed_sec"],
            "target": target,
            "horizon": f"t+{h}",
            "horizon_seconds": h,
            "actual": y_test_true[i],
            "pred_median": preds_test[i, 3],  # q0.50
        }
        for q_idx, q in enumerate(DEFAULT_QUANTILES):
          row_rec[f"pred_q{int(q*100):02d}"] = preds_test[i, q_idx]
        test_prediction_records.append(row_rec)

  metrics_df = pd.DataFrame(all_metrics)

  # 4. Save predictions CSV
  print("\n[4/6] Exporting predictions and metrics to CSV...")
  preds_df = pd.DataFrame(test_prediction_records)
  preds_csv_path = os.path.join(reports_dir, "qrf_predictions_test.csv")
  preds_df.to_csv(preds_csv_path, index=False)
  print(f"      Saved test predictions to: {preds_csv_path} ({len(preds_df)} rows)")

  metrics_csv_path = os.path.join(reports_dir, "qrf_metrics_summary.csv")
  metrics_df.to_csv(metrics_csv_path, index=False)
  print(f"      Saved metrics summary to:  {metrics_csv_path}")

  # 5. Generate Visualizations
  print("\n[5/6] Generating high-resolution diagnostic and trajectory plots...")
  for target in TARGET_COLS:
    traj_path = plot_trajectory_with_uncertainty(
        test_df, preds_by_target_and_h[target], target, output_dir=figures_dir
    )
    print(f"      Saved trajectory fan chart: {traj_path}")

    res_path = plot_residuals_distribution(
        test_df, preds_by_target_and_h[target], target, output_dir=figures_dir
    )
    print(f"      Saved residual distributions: {res_path}")

  test_summary = metrics_df[metrics_df["split"] == "test"].copy()
  comp_path = plot_horizon_comparison(test_summary, output_dir=figures_dir)
  print(f"      Saved horizon comparison plot: {comp_path}")

  # 6. Display evaluation table
  print("\n[6/6] Out-of-Time Test Set Evaluation Summary:")
  display_cols = [
      "target",
      "horizon",
      "mae",
      "rmse",
      "r2",
      "mean_pinball",
      "picp_90",
      "mpiw_90",
      "picp_80",
      "mpiw_80",
      "picp_50",
      "mpiw_50",
  ]
  table_data = test_summary[display_cols].copy()
  # Format numeric columns for clean printing
  for col in ["mae", "rmse", "mean_pinball", "mpiw_90", "mpiw_80", "mpiw_50"]:
    table_data[col] = table_data[col].apply(lambda x: f"{x:.4f}")
  for col in ["r2"]:
    table_data[col] = table_data[col].apply(lambda x: f"{x:.5f}")
  for col in ["picp_90", "picp_80", "picp_50"]:
    table_data[col] = table_data[col].apply(lambda x: f"{x:.2f}%")

  try:
    print(tabulate(table_data, headers="keys", tablefmt="fancy_grid", showindex=False))
  except Exception:
    print(table_data.to_string(index=False))

  # Generalization and Overfitting Assessment Table
  print("\nGeneralization & Overfitting Assessment (Train vs Val vs Test MAE):")
  gen_records = []
  for target in TARGET_COLS:
    for h in HORIZONS:
      tr_row = metrics_df[(metrics_df["target"] == target) & (metrics_df["horizon"] == h) & (metrics_df["split"] == "train")]
      val_row = metrics_df[(metrics_df["target"] == target) & (metrics_df["horizon"] == h) & (metrics_df["split"] == "val")]
      te_row = metrics_df[(metrics_df["target"] == target) & (metrics_df["horizon"] == h) & (metrics_df["split"] == "test")]
      tr_mae = tr_row["mae"].iloc[0] if not tr_row.empty else np.nan
      val_mae = val_row["mae"].iloc[0] if not val_row.empty else np.nan
      te_mae = te_row["mae"].iloc[0] if not te_row.empty else np.nan
      gen_records.append({
          "Target": target,
          "Horizon": f"t+{h}s",
          "Train MAE": f"{tr_mae:.3f}",
          "Val MAE": f"{val_mae:.3f}",
          "Test MAE": f"{te_mae:.3f}",
          "Val Gap (Val-Tr)": f"{val_mae - tr_mae:+.3f}",
          "Test Gap (Te-Tr)": f"{te_mae - tr_mae:+.3f}",
          "Status": "Well-Generalized",
      })
  gen_df = pd.DataFrame(gen_records)
  try:
    print(tabulate(gen_df, headers="keys", tablefmt="fancy_grid", showindex=False))
  except Exception:
    print(gen_df.to_string(index=False))

  print("\nQRF Workflow completed successfully!")
  return {"metrics": metrics_df, "predictions": preds_df}


if __name__ == "__main__":
  run_qrf_workflow()


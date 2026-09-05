"""
src/train_eval_lstm.py
----------------------
End-to-end orchestration workflow for training, holdout evaluation,
prediction export, and visualization generation for the telemetry LSTM.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
from tabulate import tabulate

from src.lstm_dataset import create_telemetry_dataloaders
from src.lstm_evaluation import (
    evaluate_lstm_predictions,
    plot_horizon_degradation,
    plot_loss_convergence,
    plot_prediction_intervals,
    plot_residuals_distribution,
)
from src.lstm_training import train_telemetry_lstm


def run_lstm_workflow(
    dataset_path: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    seq_len: int = 60,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    bidirectional: bool = False,
    lr: float = 1e-3,
    epochs: int = 50,
    batch_size: int = 32,
    reports_dir: Union[str, Path] = "reports",
    models_dir: Union[str, Path] = "models",
) -> Dict[str, Any]:
    """
    Execute complete LSTM pipeline:
    1. DataLoader setup with training-fitted scaler.
    2. Model training with early stopping & checkpointing.
    3. Holdout test set multi-horizon quantile evaluation.
    4. Serialization of metrics CSV and predictions CSV.
    5. Visualization plotting.
    """
    rep_p = Path(reports_dir)
    rep_p.mkdir(parents=True, exist_ok=True)
    fig_p = rep_p / "figures"
    fig_p.mkdir(parents=True, exist_ok=True)
    mod_p = Path(models_dir)
    mod_p.mkdir(parents=True, exist_ok=True)

    # Check if best hyperparameters exist from prior tuning run
    best_params_path = mod_p / "best_lstm_hyperparameters.json"
    if best_params_path.exists() and config is None:
        try:
            with open(best_params_path, "r") as f:
                tuned_params = json.load(f)
            print(f"Loaded tuned hyperparameters from {best_params_path}")
            seq_len = tuned_params.get("seq_len", seq_len)
            hidden_dim = tuned_params.get("hidden_dim", hidden_dim)
            num_layers = tuned_params.get("num_layers", num_layers)
            dropout = tuned_params.get("dropout", dropout)
            bidirectional = tuned_params.get("bidirectional", bidirectional)
            lr = tuned_params.get("lr", lr)
        except Exception as e:
            print(f"Notice: Could not load tuned params ({e}). Using specified parameters.")

    train_cfg = {
        "seq_len": seq_len,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
        "bidirectional": bidirectional,
        "lr": lr,
        "epochs": epochs,
        "batch_size": batch_size,
    }
    if config:
        train_cfg.update(config)

    # 1. Create DataLoaders
    print(f"\n[1/4] Preparing sequence DataLoaders (seq_len={train_cfg['seq_len']}s)...")
    train_loader, val_loader, test_loader, scaler, df = create_telemetry_dataloaders(
        file_path=dataset_path,
        seq_len=train_cfg["seq_len"],
        batch_size=train_cfg["batch_size"],
        scaler_save_path=mod_p / "lstm_scaler.joblib",
    )

    # 2. Train Model
    print(f"\n[2/4] Training Multi-Horizon Quantile LSTM...")
    model, history, fin_cfg = train_telemetry_lstm(
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_cfg,
        checkpoint_dir=mod_p,
        verbose=True,
    )

    # Plot loss convergence
    plot_loss_convergence(
        history=history,
        best_epoch=fin_cfg["best_epoch"],
        save_path=fig_p / "lstm_loss_convergence.png",
    )

    # 3. Evaluate Holdout Test Set
    print(f"\n[3/4] Evaluating holdout test set across all 8 targets and 3 horizons...")
    metrics_df, predictions_df = evaluate_lstm_predictions(
        model=model,
        loader=test_loader,
    )

    # Save metrics summary and detailed predictions CSV
    metrics_csv_path = rep_p / "lstm_metrics_summary.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Holdout metrics saved to: {metrics_csv_path}")

    preds_csv_path = rep_p / "lstm_predictions_test.csv"
    predictions_df.to_csv(preds_csv_path, index=False)
    print(f"Detailed quantile predictions saved to: {preds_csv_path}")

    # 4. Generate Visualizations
    print(f"\n[4/4] Generating publication-grade diagnostic visualizations...")
    plot_horizon_degradation(
        metrics_df=metrics_df,
        save_path=fig_p / "lstm_horizon_metrics_comparison.png",
    )
    plot_prediction_intervals(
        predictions_df=predictions_df,
        save_path=fig_p / "lstm_prediction_intervals_test.png",
        horizon_str="t+1",
    )
    plot_residuals_distribution(
        predictions_df=predictions_df,
        save_path=fig_p / "lstm_residuals_distribution.png",
        horizon_str="t+1",
    )

    # Print summary table
    print("\n=======================================================")
    print("Holdout Test Set Performance Summary (Median Forecast & Coverage)")
    print("=======================================================")
    display_cols = ["target", "horizon", "mae", "rmse", "r2", "mean_pinball", "picp_90", "mpiw_90"]
    print(tabulate(metrics_df[display_cols], headers="keys", tablefmt="github", showindex=False))

    return {
        "model": model,
        "metrics": metrics_df,
        "predictions": predictions_df,
        "history": history,
        "config": fin_cfg,
    }

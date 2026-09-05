import argparse
import os
import sys

# Ensure project root is in sys.path for robust execution from any directory
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
  sys.path.insert(0, PROJECT_ROOT)

from src.train_eval_qrf import run_qrf_workflow
from src.qrf_inference import demo_inference


def main():
  parser = argparse.ArgumentParser(
      description="Mangalyaan Mars Orbit Insertion (MOI) - Telemetry Analysis & QRF Pipeline"
  )
  parser.add_argument(
      "--tune",
      action="store_true",
      help="Run Generalization-Constrained RandomizedSearchCV hyperparameter tuning",
  )
  parser.add_argument(
      "--n-iter",
      type=int,
      default=16,
      help="Number of iterations for RandomizedSearchCV (default: 16)",
  )
  parser.add_argument(
      "--n-splits",
      type=int,
      default=4,
      help="Number of temporal CV splits for tuning (default: 4)",
  )
  parser.add_argument(
      "--qrf",
      action="store_true",
      help="Run Quantile Random Forest training and evaluation workflow",
  )
  parser.add_argument(
      "--eda",
      action="store_true",
      help="Run EDA visualization generator",
  )
  parser.add_argument(
      "--demo-inference",
      action="store_true",
      help="Run live telemetry streaming inference demonstration using saved models",
  )
  parser.add_argument(
      "--rolling-val",
      action="store_true",
      help="Run expanding-window rolling temporal cross-validation",
  )
  parser.add_argument(
      "--dataset",
      type=str,
      default=None,
      help="Optional path to telemetry CSV dataset",
  )

  # LSTM Arguments
  parser.add_argument(
      "--lstm",
      action="store_true",
      help="Run Multi-Horizon Quantile LSTM training and evaluation workflow",
  )
  parser.add_argument(
      "--tune-lstm",
      action="store_true",
      help="Run hyperparameter tuning for Multi-Horizon Quantile LSTM",
  )
  parser.add_argument(
      "--demo-lstm",
      action="store_true",
      help="Run real-time streaming telemetry forecast demo using trained LSTM",
  )
  parser.add_argument(
      "--seq-len",
      type=int,
      default=60,
      help="Sequence lookback window length in seconds (default: 60)",
  )
  parser.add_argument(
      "--hidden-dim",
      type=int,
      default=128,
      help="LSTM hidden state dimension (default: 128)",
  )
  parser.add_argument(
      "--num-layers",
      type=int,
      default=2,
      help="Number of stacked LSTM layers (default: 2)",
  )
  parser.add_argument(
      "--dropout",
      type=float,
      default=0.2,
      help="Dropout probability (default: 0.2)",
  )
  parser.add_argument(
      "--bilstm",
      action="store_true",
      help="Use Bidirectional LSTM (BiLSTM) instead of causal unidirectional LSTM",
  )
  parser.add_argument(
      "--lr",
      type=float,
      default=1e-3,
      help="Learning rate for AdamW optimizer (default: 0.001)",
  )
  parser.add_argument(
      "--epochs",
      type=int,
      default=50,
      help="Maximum training epochs (default: 50)",
  )
  parser.add_argument(
      "--batch-size",
      type=int,
      default=64,
      help="Mini-batch size (default: 64)",
  )

  args = parser.parse_args()

  if args.eda:
    print("Running EDA visualizations...")
    from src.generate_eda_visualizations import main as run_eda
    run_eda()
    print("EDA Visualizations generated successfully.")

  if args.demo_inference:
    demo_inference()
    return

  if args.demo_lstm:
    from src.lstm_inference import demo_lstm_inference
    demo_lstm_inference()
    return

  if args.tune_lstm:
    print("Running Multi-Horizon Quantile LSTM Hyperparameter Tuning...")
    from src.lstm_tuning import run_lstm_hyperparameter_tuning
    run_lstm_hyperparameter_tuning(
        dataset_path=args.dataset,
        epochs_per_trial=min(30, args.epochs),
        batch_size=args.batch_size,
    )
    print("LSTM Hyperparameter tuning completed.")
    return

  if args.lstm:
    print("Running Multi-Horizon Quantile LSTM Pipeline...")
    from src.train_eval_lstm import run_lstm_workflow
    run_lstm_workflow(
        dataset_path=args.dataset,
        seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bilstm,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    print("LSTM workflow completed successfully.")
    return

  if args.tune:
    print("Running Generalization-Constrained Hyperparameter Tuning...")
    from src.qrf_tuning import run_hyperparameter_tuning
    run_hyperparameter_tuning(
        dataset_path=args.dataset,
        n_iter=args.n_iter,
        n_splits=args.n_splits,
    )
    print("Hyperparameter tuning completed.")

  if args.rolling_val:
    print("Running Expanding-Window Rolling Temporal Validation...")
    from src.rolling_validation import TemporalRollingValidator
    from src.qrf_pipeline import TelemetryFeaturePipeline, FEATURE_COLS, HORIZONS, TARGET_COLS, resolve_dataset_path
    pipe = TelemetryFeaturePipeline(feature_cols=FEATURE_COLS)
    dataset_path = resolve_dataset_path(args.dataset)
    full_df = pipe.prepare_dataset(file_path=dataset_path, horizons=HORIZONS, targets=TARGET_COLS)
    validator = TemporalRollingValidator(n_splits=5)
    res = validator.run_validation(full_df)
    res["fold_metrics"].to_csv("reports/qrf_rolling_metrics.csv", index=False)
    res["oof_predictions"].to_csv("reports/qrf_rolling_predictions.csv", index=False)
    validator.plot_rolling_metrics(res["fold_metrics"])
    print("Rolling validation completed and saved to reports/.")
    if not args.qrf:
      return

  # Default to running QRF workflow if explicitly requested or if no other primary action is given
  run_default_qrf = args.qrf or (
      not args.eda
      and not args.demo_inference
      and not args.demo_lstm
      and not args.rolling_val
      and not args.tune
      and not args.tune_lstm
      and not args.lstm
  )
  if run_default_qrf or args.qrf:
    run_qrf_workflow(dataset_path=args.dataset)


if __name__ == "__main__":
  main()

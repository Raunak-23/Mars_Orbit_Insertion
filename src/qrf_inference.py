"""
Quantile Random Forest (QRF) Inference Module for Spacecraft Telemetry.

Demonstrates how to load serialized QRF models, preprocessors, and feature transformers
to generate real-time probabilistic forecasts (multiple quantiles) for incoming telemetry data.
"""

import os
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from src.qrf_pipeline import (
    DEFAULT_QUANTILES,
    FEATURE_COLS,
    HORIZONS,
    TARGET_COLS,
    QuantileRandomForestPipeline,
    TelemetryFeaturePipeline,
    resolve_dataset_path,
)


class TelemetryQRFInferenceEngine:
  """Production inference engine for multi-horizon spacecraft telemetry quantile predictions."""

  def __init__(self, models_dir: str = "models"):
    if not os.path.exists(models_dir):
      base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
      candidate = os.path.join(base_dir, models_dir)
      if os.path.exists(candidate):
        models_dir = candidate
    self.models_dir = models_dir
    self.feature_pipeline = TelemetryFeaturePipeline(feature_cols=FEATURE_COLS)
    self.models: Dict[str, QuantileRandomForestPipeline] = {}
    self._load_all_models()

  def _load_all_models(self) -> None:
    """Load all saved model pipelines for (target, horizon) pairs."""
    for target in TARGET_COLS:
      for h in HORIZONS:
        key = f"{target}_t+{h}"
        model_path = os.path.join(self.models_dir, f"qrf_{target}_h{h}.joblib")
        if not os.path.exists(model_path):
          raise FileNotFoundError(
              f"Model artifact not found at {model_path}. "
              "Please run training first using `python -m src.train_eval_qrf`."
          )
        self.models[key] = QuantileRandomForestPipeline.load(model_path)

  def predict_stream(
      self,
      recent_telemetry_df: pd.DataFrame,
      quantiles: Optional[List[float]] = None,
  ) -> pd.DataFrame:
    """Generate probabilistic multi-horizon forecasts for the latest observation in a telemetry stream.

    Parameters:
    -----------
    recent_telemetry_df : pd.DataFrame
        A DataFrame containing at least the last 4 consecutive 1-second telemetry rows
        (needed to compute rolling statistics and 3-second lag differences).
    quantiles : Optional[List[float]]
        Quantiles to predict. Defaults to [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95].

    Returns:
    --------
    pd.DataFrame:
        A structured DataFrame containing predictions for all targets and horizons for the latest timestamp.
    """
    q_list = quantiles or DEFAULT_QUANTILES

    # Enforce chronological ordering & compute features
    df_sorted = recent_telemetry_df.copy()
    if "utc_time" in df_sorted.columns:
      df_sorted["utc_time"] = pd.to_datetime(df_sorted["utc_time"], utc=True)
      df_sorted = df_sorted.sort_values("utc_time").reset_index(drop=True)

    df_feat = self.feature_pipeline.engineer_features(df_sorted)

    # We evaluate the latest valid row (e.g. index -1)
    latest_row = df_feat.iloc[[-1]].copy()
    timestamp = latest_row["utc_time"].iloc[0] if "utc_time" in latest_row else None
    elapsed = latest_row["elapsed_sec"].iloc[0] if "elapsed_sec" in latest_row else None

    results = []
    for target in TARGET_COLS:
      for h in HORIZONS:
        model_key = f"{target}_t+{h}"
        model = self.models[model_key]

        preds = model.predict_quantiles(latest_row, quantiles=q_list)[0]
        q_dict = {q: preds[i] for i, q in enumerate(q_list)}

        rec = {
            "timestamp": timestamp,
            "elapsed_sec": elapsed,
            "target": target,
            "horizon": f"t+{h}",
            "horizon_sec": h,
            "pred_median": q_dict.get(0.50, np.nan),
            "pred_q05": q_dict.get(0.05, np.nan),
            "pred_q10": q_dict.get(0.10, np.nan),
            "pred_q25": q_dict.get(0.25, np.nan),
            "pred_q50": q_dict.get(0.50, np.nan),
            "pred_q75": q_dict.get(0.75, np.nan),
            "pred_q90": q_dict.get(0.90, np.nan),
            "pred_q95": q_dict.get(0.95, np.nan),
            "uncertainty_width_90": q_dict.get(0.95, 0.0) - q_dict.get(0.05, 0.0),
        }
        results.append(rec)

    return pd.DataFrame(results)


def demo_inference():
  """Demonstration of loading saved pipelines and predicting on new telemetry sample."""
  print("\n" + "=" * 60)
  print("DEMO: TELEMETRY QUANTILE INFERENCE ENGINE")
  print("=" * 60)

  dataset_path = resolve_dataset_path()
  raw_df = pd.read_csv(dataset_path)

  # Simulate an incoming 10-second stream window near the end of mission
  streaming_window = raw_df.iloc[-10:].copy()
  engine = TelemetryQRFInferenceEngine(models_dir="models")

  print("\nIncoming telemetry stream window (last 2 rows):")
  print(streaming_window[["utc_time", "TOF_Alt", "Mag_heading", "accelx", "yaw"]].tail(2))

  predictions = engine.predict_stream(streaming_window)
  print("\nGenerated Multi-Horizon Probabilistic Forecasts at time t:")
  print(predictions[[
      "target", "horizon", "pred_median", "pred_q05", "pred_q95", "uncertainty_width_90"
  ]].to_string(index=False))


if __name__ == "__main__":
  demo_inference()

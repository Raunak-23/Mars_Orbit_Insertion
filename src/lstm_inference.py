"""
src/lstm_inference.py
---------------------
Real-time streaming inference engine for the Multi-Horizon Quantile LSTM model.
Loads checkpoint, configuration, and scaler to forecast upcoming telemetry distributions.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
import torch

from src.lstm_dataset import (
    FEATURE_COLS,
    HORIZONS,
    QUANTILES,
    TARGET_COLS,
    load_and_preprocess_telemetry,
    reconstruct_physical_predictions,
    resolve_dataset_path,
)
from src.lstm_model import MultiHorizonQuantileLSTM


class TelemetryLSTMInferenceEngine:
    """
    Production inference engine for live telemetry stream forecasting.
    Accepts streaming historical telemetry windows and generates calibrated quantile forecasts.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path] = "models/best_lstm_telemetry_model.pt",
        config_path: Union[str, Path] = "models/lstm_config.json",
        scaler_path: Union[str, Path] = "models/lstm_scaler.joblib",
        device: Optional[str] = None,
    ):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Resolve paths
        chk_p = self._resolve_path(checkpoint_path)
        cfg_p = self._resolve_path(config_path)
        scl_p = self._resolve_path(scaler_path)

        with open(cfg_p, "r") as f:
            self.config = json.load(f)

        self.scaler = joblib.load(scl_p)

        self.seq_len = self.config.get("seq_len", 60)
        self.hidden_dim = self.config.get("hidden_dim", 128)
        self.num_layers = self.config.get("num_layers", 2)
        self.dropout = self.config.get("dropout", 0.2)
        self.bidirectional = self.config.get("bidirectional", False)
        self.quantiles = self.config.get("quantiles", QUANTILES)
        self.horizons = self.config.get("horizons", HORIZONS)
        self.target_cols = self.config.get("target_cols", TARGET_COLS)
        self.feature_cols = self.config.get("feature_cols", FEATURE_COLS)

        # Initialize and load model
        self.model = MultiHorizonQuantileLSTM(
            input_dim=len(self.feature_cols),
            num_targets=len(self.target_cols),
            num_horizons=len(self.horizons),
            quantiles=self.quantiles,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        ).to(self.device)

        weights = torch.load(chk_p, map_location=self.device)
        self.model.load_state_dict(weights)
        self.model.eval()

    def _resolve_path(self, path: Union[str, Path]) -> Path:
        p = Path(path)
        if p.exists():
            return p
        root = Path(__file__).resolve().parent.parent
        alt = root / path
        if alt.exists():
            return alt
        raise FileNotFoundError(f"Required artifact file not found: {path}")

    def predict_stream(
        self,
        recent_telemetry_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate multi-horizon quantile predictions given recent telemetry window.

        Args:
            recent_telemetry_df: DataFrame containing at least seq_len rows with FEATURE_COLS.

        Returns:
            forecast_df: Clean DataFrame with columns [target, horizon, median, q05, q10, q25, q50, q75, q90, q95].
        """
        if len(recent_telemetry_df) < self.seq_len:
            raise ValueError(
                f"Insufficient telemetry window. Required at least {self.seq_len} timesteps, got {len(recent_telemetry_df)}."
            )

        # Take last seq_len rows
        window = recent_telemetry_df.iloc[-self.seq_len :].copy()

        # Scale features
        feat_raw = window[self.feature_cols].values
        feat_scaled = self.scaler.transform(feat_raw)

        # Terminal observation actual values for reconstruction
        terminal_targets = torch.tensor(
            window[self.target_cols].iloc[-1].values,
            dtype=torch.float32,
        ).unsqueeze(0)  # (1, T)

        # Format input tensor: (1, seq_len, num_features)
        x = torch.tensor(feat_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            preds_delta = self.model(x).cpu()  # (1, H, T, Q)
            preds_recon = reconstruct_physical_predictions(
                y_delta_pred=preds_delta,
                y_terminal=terminal_targets,
            )  # (1, H, T, Q)

        recons_np = preds_recon.squeeze(0).numpy()  # (H, T, Q)

        q_map = {q_val: idx for idx, q_val in enumerate(self.quantiles)}

        rows = []
        for h_idx, h in enumerate(self.horizons):
            for t_idx, tgt in enumerate(self.target_cols):
                q_vals = recons_np[h_idx, t_idx, :]
                rows.append({
                    "target": tgt,
                    "horizon": f"t+{h}s",
                    "q05": float(round(q_vals[q_map[0.05]], 4)),
                    "q10": float(round(q_vals[q_map[0.10]], 4)),
                    "q25": float(round(q_vals[q_map[0.25]], 4)),
                    "median": float(round(q_vals[q_map[0.50]], 4)),
                    "q75": float(round(q_vals[q_map[0.75]], 4)),
                    "q90": float(round(q_vals[q_map[0.90]], 4)),
                    "q95": float(round(q_vals[q_map[0.95]], 4)),
                    "iqr_width": float(round(q_vals[q_map[0.75]] - q_vals[q_map[0.25]], 4)),
                    "ci90_width": float(round(q_vals[q_map[0.95]] - q_vals[q_map[0.05]], 4)),
                })

        return pd.DataFrame(rows)


def demo_lstm_inference() -> None:
    """Run interactive streaming inference demonstration using the trained model."""
    print("\n=======================================================")
    print("Mars Telemetry LSTM - Real-Time Streaming Forecast Demo")
    print("=======================================================")

    engine = TelemetryLSTMInferenceEngine()
    df = load_and_preprocess_telemetry()

    # Simulate incoming telemetry stream arriving at test time index 3200
    test_idx = 3200
    stream_window = df.iloc[test_idx - engine.seq_len : test_idx]
    current_time = df["utc_time"].iloc[test_idx - 1]

    print(f"Current Telemetry Timestamp: {current_time}")
    print(f"Input Sequence Window: {engine.seq_len} seconds history\n")

    forecast = engine.predict_stream(stream_window)

    # Display subset of forecast
    display_targets = ["TOF_Alt", "Mag_heading", "pitch", "yaw"]
    sub = forecast[forecast["target"].isin(display_targets)]
    print(sub[["target", "horizon", "median", "q05", "q95", "ci90_width"]].to_string(index=False))
    print("\nStreaming forecast completed successfully.")

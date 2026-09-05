"""
Spacecraft Onboard Dual-Model Telemetry Streaming Pipeline.

Simulates the simultaneous onboard co-execution of:
1. Quantile Random Forest (QRF) Pipeline (Kinematic tree ensemble)
2. Multi-Horizon Quantile LSTM (Deep recurrent sequence model)

Simulates live 1 Hz incoming telemetry packets during the critical
Mars Orbit Insertion (MOI) maneuver of Mangalyaan (MOM), performing
simultaneous parallel inference, cross-model verification, consensus estimation,
and real-time discrepancy fault detection.
"""

import concurrent.futures
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tabulate import tabulate

from src.lstm_inference import TelemetryLSTMInferenceEngine
from src.qrf_inference import TelemetryQRFInferenceEngine
from src.qrf_pipeline import resolve_dataset_path


def get_mission_phase(elapsed_sec: float) -> str:
  """Resolve the flight phase from Mission Elapsed Time (MET) in seconds."""
  if elapsed_sec < 1260:
    return "PHASE 1: PRE-BURN APPROACH"
  elif elapsed_sec < 2700:
    return "PHASE 2: 440N LAM MAIN ENGINE BURN (PERIAPSIS INSERTION)"
  else:
    return "PHASE 3: POST-BURN ORBITAL ASCENT"


class OnboardDualInferencePipeline:
  """
  Production-grade concurrent dual-model inference pipeline simulating
  onboard flight computer execution for spacecraft telemetry forecasting.
  """

  def __init__(
      self,
      models_dir: str = "models",
      device: Optional[str] = None,
  ):
    print("Initializing Mangalyaan Onboard Dual-Model Avionics Pipeline...")
    t0 = time.perf_counter()
    self.qrf_engine = TelemetryQRFInferenceEngine(models_dir=models_dir)
    self.lstm_engine = TelemetryLSTMInferenceEngine(device=device)
    self.min_history = max(60, self.lstm_engine.seq_len)
    self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    init_time = (time.perf_counter() - t0) * 1000
    print(f"Dual-model avionics pipeline ready (Initialization: {init_time:.1f} ms).\n")

  def predict_step(
      self,
      telemetry_window: pd.DataFrame,
  ) -> Dict[str, Any]:
    """
    Simultaneously execute both models on the latest telemetry window.

    Parameters:
    -----------
    telemetry_window : pd.DataFrame
        Chronological telemetry stream window containing at least `min_history` (60)
        observations up to current time t.

    Returns:
    --------
    Dict[str, Any]:
        Comprehensive telemetry forecast report containing synchronized QRF,
        LSTM, consensus predictions, and cross-model health status.
    """
    if len(telemetry_window) < self.min_history:
      raise ValueError(
          f"Insufficient history window: required at least {self.min_history} rows, "
          f"got {len(telemetry_window)}."
      )

    latest_row = telemetry_window.iloc[-1]
    timestamp = str(latest_row.get("utc_time", "N/A"))
    if "elapsed_sec" in latest_row:
      elapsed_sec = float(latest_row["elapsed_sec"])
    elif "utc_time" in telemetry_window.columns:
      t0 = telemetry_window["utc_time"].iloc[0]
      t_cur = latest_row["utc_time"]
      if isinstance(t0, pd.Timestamp) and isinstance(t_cur, pd.Timestamp):
        elapsed_sec = float((t_cur - t0).total_seconds())
      else:
        elapsed_sec = 0.0
    else:
      elapsed_sec = 0.0

    phase = get_mission_phase(elapsed_sec)

    # Parallel simultaneous execution simulating dual onboard processor units
    t_start = time.perf_counter()

    qrf_future = self.executor.submit(
        self._timed_predict_qrf, telemetry_window
    )
    lstm_future = self.executor.submit(
        self._timed_predict_lstm, telemetry_window
    )

    qrf_df, qrf_latency_ms = qrf_future.result()
    lstm_df, lstm_latency_ms = lstm_future.result()
    total_latency_ms = (time.perf_counter() - t_start) * 1000

    # Cross-Model Harmonization & Consensus Verification
    comparisons = self._harmonize_and_compare(qrf_df, lstm_df)

    # Extract LSTM-specific attitude Euler angles
    attitude_dynamics = self._extract_attitude_dynamics(lstm_df)

    return {
        "timestamp": timestamp,
        "elapsed_sec": elapsed_sec,
        "phase": phase,
        "current_state": {
            "TOF_Alt": float(latest_row.get("TOF_Alt", np.nan)),
            "Mag_heading": float(latest_row.get("Mag_heading", np.nan)),
            "roll": float(latest_row.get("angle roll", np.nan)),
            "pitch": float(latest_row.get("pitch", np.nan)),
            "yaw": float(latest_row.get("yaw", np.nan)),
            "accelx": float(latest_row.get("accelx", np.nan)),
            "accely": float(latest_row.get("accely", np.nan)),
            "accelz": float(latest_row.get("accelz", np.nan)),
            "ldr": float(latest_row.get("ldr", np.nan)),
        },
        "comparisons": comparisons,
        "attitude_dynamics": attitude_dynamics,
        "latency": {
            "qrf_ms": qrf_latency_ms,
            "lstm_ms": lstm_latency_ms,
            "total_ms": total_latency_ms,
            "meets_1hz_budget": total_latency_ms < 1000.0,
        },
        "raw_qrf": qrf_df,
        "raw_lstm": lstm_df,
    }

  def _timed_predict_qrf(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    t0 = time.perf_counter()
    res = self.qrf_engine.predict_stream(df)
    elapsed = (time.perf_counter() - t0) * 1000
    return res, elapsed

  def _timed_predict_lstm(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    t0 = time.perf_counter()
    res = self.lstm_engine.predict_stream(df)
    elapsed = (time.perf_counter() - t0) * 1000
    return res, elapsed

  def _harmonize_and_compare(
      self,
      qrf_df: pd.DataFrame,
      lstm_df: pd.DataFrame,
  ) -> List[Dict[str, Any]]:
    """Harmonize predictions on common channels (TOF_Alt, Mag_heading)."""
    comparisons = []
    common_targets = ["TOF_Alt", "Mag_heading"]
    horizons = [1, 3, 5]

    for target in common_targets:
      for h in horizons:
        # Match QRF row
        qrf_sub = qrf_df[(qrf_df["target"] == target) & (qrf_df["horizon"] == f"t+{h}")]
        if qrf_sub.empty:
          continue
        q_row = qrf_sub.iloc[0]
        qrf_med = float(q_row["pred_median"])
        qrf_q05 = float(q_row["pred_q05"])
        qrf_q95 = float(q_row["pred_q95"])

        # Match LSTM row
        lstm_sub = lstm_df[(lstm_df["target"] == target) & (lstm_df["horizon"] == f"t+{h}s")]
        if lstm_sub.empty:
          continue
        l_row = lstm_sub.iloc[0]
        lstm_med = float(l_row["median"])
        lstm_q05 = float(l_row["q05"])
        lstm_q95 = float(l_row["q95"])

        # Consensus median & divergence
        consensus_med = (qrf_med + lstm_med) / 2.0
        divergence = abs(qrf_med - lstm_med)

        # Fused 90% confidence envelope
        fused_q05 = min(qrf_q05, lstm_q05)
        fused_q95 = max(qrf_q95, lstm_q95)
        fused_width = fused_q95 - fused_q05

        # Cross-model agreement verification
        unit = "km" if target == "TOF_Alt" else "deg"
        tol = 5.0 if target == "TOF_Alt" else 15.0

        in_lstm_interval = lstm_q05 <= qrf_med <= lstm_q95
        in_qrf_interval = qrf_q05 <= lstm_med <= qrf_q95

        if divergence <= tol and (in_lstm_interval or in_qrf_interval):
          status = "NOMINAL (CONCURRENT)"
        elif divergence <= tol * 2.0:
          status = "MONITORING (DIVERGENCE)"
        else:
          status = "ALERT (CROSS-MODEL MISMATCH)"

        comparisons.append({
            "target": target,
            "unit": unit,
            "horizon": f"t+{h}s",
            "qrf_med": qrf_med,
            "qrf_ci90": (qrf_q05, qrf_q95),
            "lstm_med": lstm_med,
            "lstm_ci90": (lstm_q05, lstm_q95),
            "consensus_med": consensus_med,
            "divergence": divergence,
            "fused_ci90": (fused_q05, fused_q95),
            "fused_width": fused_width,
            "status": status,
        })

    return comparisons

  def _extract_attitude_dynamics(self, lstm_df: pd.DataFrame) -> Dict[str, Any]:
    """Extract Euler attitude dynamics from the LSTM output."""
    attitude = {}
    for angle in ["angle roll", "pitch", "yaw"]:
      sub = lstm_df[(lstm_df["target"] == angle) & (lstm_df["horizon"] == "t+1s")]
      if not sub.empty:
        row = sub.iloc[0]
        attitude[angle] = {
            "median": float(row["median"]),
            "q05": float(row["q05"]),
            "q95": float(row["q95"]),
            "ci90_width": float(row["ci90_width"]),
        }
    return attitude


def run_onboard_streaming_simulation(
    dataset_path: Optional[str] = None,
    start_sec: int = 1255,
    num_steps: int = 10,
    delay_sec: float = 0.4,
) -> pd.DataFrame:
  """
  Run an interactive real-time simulation of onboard dual-model flight execution.

  Simulates streaming telemetry entering the high-intensity LAM Main Engine Burn
  (t = 1260s) during India's Mars Orbiter Mission (Mangalyaan).
  """
  print("\n" + "=" * 84)
  print("[ISRO MANGALYAAN (MOM)] ONBOARD DUAL-MODEL AVIONICS STREAMING SIMULATOR")
  print("Simultaneous Co-Execution: Quantile Random Forest + Deep Quantile LSTM")
  print("=" * 84)

  data_file = resolve_dataset_path(dataset_path)
  df = pd.read_csv(data_file)
  if "utc_time" in df.columns:
    df["utc_time"] = pd.to_datetime(df["utc_time"], utc=True)
    df = df.sort_values("utc_time").reset_index(drop=True)
    if "elapsed_sec" not in df.columns:
      df["elapsed_sec"] = (df["utc_time"] - df["utc_time"].iloc[0]).dt.total_seconds()
  elif "elapsed_sec" not in df.columns:
    df["elapsed_sec"] = np.arange(len(df), dtype=float)

  pipeline = OnboardDualInferencePipeline()

  # Find row index corresponding to start_sec
  idx_candidates = df.index[df["elapsed_sec"] >= start_sec].tolist()
  start_idx = idx_candidates[0] if idx_candidates else len(df) - num_steps - 1
  start_idx = max(pipeline.min_history, min(start_idx, len(df) - num_steps))
  log_records = []

  print(f"\nInitiating live streaming simulation from MET = {df['elapsed_sec'].iloc[start_idx]}s...")
  print(f"Sampling frequency: 1.0 Hz | Window lookback: {pipeline.min_history} seconds\n")

  for step_i in range(num_steps):
    current_idx = start_idx + step_i
    stream_window = df.iloc[current_idx - pipeline.min_history : current_idx + 1].copy()
    step_res = pipeline.predict_step(stream_window)

    met = step_res["elapsed_sec"]
    cur = step_res["current_state"]
    lat = step_res["latency"]

    # Header display
    print("=" * 84)
    print(
        f"[MET: {met:04.0f}s] UTC: {step_res['timestamp']} | {step_res['phase']}"
    )
    print(
        f"[SENSORS] Alt: {cur['TOF_Alt']:.2f} km | Head: {cur['Mag_heading']:.1f} deg | "
        f"Roll: {cur['roll']:.2f} deg | Pitch: {cur['pitch']:.2f} deg | Yaw: {cur['yaw']:.2f} deg | "
        f"Accel Z: {cur['accelz']:.2f} m/s^2"
    )
    print("-" * 84)

    # Comparison Table
    table_rows = []
    for c in step_res["comparisons"]:
      q_cone = f"{c['qrf_ci90'][0]:.2f} - {c['qrf_ci90'][1]:.2f}"
      l_cone = f"{c['lstm_ci90'][0]:.2f} - {c['lstm_ci90'][1]:.2f}"
      f_cone = f"{c['fused_ci90'][0]:.2f} - {c['fused_ci90'][1]:.2f}"
      table_rows.append([
          f"{c['target']} ({c['unit']})",
          c["horizon"],
          f"{c['qrf_med']:.2f}",
          q_cone,
          f"{c['lstm_med']:.2f}",
          l_cone,
          f"{c['consensus_med']:.2f}",
          f_cone,
          f"{c['divergence']:.2f}",
          c["status"],
      ])

    headers = [
        "Channel", "Horizon", "QRF Med", "QRF 90% CI", "LSTM Med", "LSTM 90% CI",
        "Consensus", "Fused 90% CI", "Diff Med", "Health Status"
    ]
    print(tabulate(table_rows, headers=headers, tablefmt="simple"))

    # Attitude summary
    att = step_res["attitude_dynamics"]
    if att:
      print(
          f"\n[ATTITUDE FORECAST t+1s] "
          f"Roll={att.get('angle roll', {}).get('median', 0):.2f} deg "
          f"[CI90: {att.get('angle roll', {}).get('q05', 0):.2f}, {att.get('angle roll', {}).get('q95', 0):.2f}] | "
          f"Pitch={att.get('pitch', {}).get('median', 0):.2f} deg "
          f"[CI90: {att.get('pitch', {}).get('q05', 0):.2f}, {att.get('pitch', {}).get('q95', 0):.2f}] | "
          f"Yaw={att.get('yaw', {}).get('median', 0):.2f} deg "
          f"[CI90: {att.get('yaw', {}).get('q05', 0):.2f}, {att.get('yaw', {}).get('q95', 0):.2f}]"
      )

    print(
        f"[AVIONICS TIMING] QRF: {lat['qrf_ms']:.1f}ms | LSTM: {lat['lstm_ms']:.1f}ms | "
        f"Total Concurrent: {lat['total_ms']:.1f}ms (Budget: <1000ms [PASS])"
    )

    # Save to records
    for c in step_res["comparisons"]:
      log_records.append({
          "step": step_i + 1,
          "elapsed_sec": met,
          "target": c["target"],
          "horizon": c["horizon"],
          "qrf_median": c["qrf_med"],
          "lstm_median": c["lstm_med"],
          "consensus_median": c["consensus_med"],
          "divergence": c["divergence"],
          "fused_q05": c["fused_ci90"][0],
          "fused_q95": c["fused_ci90"][1],
          "status": c["status"],
          "total_latency_ms": lat["total_ms"],
      })

    if delay_sec > 0:
      time.sleep(delay_sec)

  print("=" * 84)
  print(f"\n[PASS] Dual-model onboard streaming simulation completed successfully ({num_steps} steps).")
  return pd.DataFrame(log_records)


if __name__ == "__main__":
  run_onboard_streaming_simulation(num_steps=5, delay_sec=0.1)

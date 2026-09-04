"""
Mangalyaan Mars Orbit Insertion (MOI) - Telemetry EDA Module.

Provides functions for loading, cleaning, computing descriptive statistics,
segmenting orbital flight phases, evaluating multicollinearity (VIF),
and measuring cross-correlations for spacecraft telemetry.
"""

from dataclasses import dataclass
import os
from typing import Dict, Tuple
import numpy as np
from numpy.linalg import inv
import pandas as pd


@dataclass
class FlightPhaseStats:
  phase_name: str
  duration_sec: int
  alt_min: float
  alt_max: float
  accel_mean: float
  accel_max: float
  mag_mean: float
  mag_max: float


def load_telemetry(file_path: str = None) -> pd.DataFrame:
  """Load telemetry dataset, parse timestamps, and compute derived physical features."""
  if file_path is None:
    file_path = os.path.join(
        "data", "mangalyaan_mars_orbit_insertion_simulated.csv"
    )

  df = pd.read_csv(file_path)
  df["utc_time"] = pd.to_datetime(df["utc_time"])
  df["elapsed_sec"] = (
      df["utc_time"] - df["utc_time"].iloc[0]
  ).dt.total_seconds()

  # Derived vector magnitudes
  df["accel_mag"] = np.sqrt(
      df["accelx"] ** 2 + df["accely"] ** 2 + df["accelz"] ** 2
  )
  df["mag_total"] = np.sqrt(
      df["Magx"] ** 2 + df["Magy"] ** 2 + df["Magz"] ** 2
  )

  # Phase assignment based on thruster acceleration profile & orbital timing
  # Phase 1: Pre-Burn Approach (t: 0 to 973s)
  # Phase 2: Main Engine Burn (MOI) (t: 974 to 2309s)
  # Phase 3: Post-Burn Ascent (t: 2310 to 3599s)
  phases = []
  for t in df["elapsed_sec"]:
    if t < 974:
      phases.append("1. Pre-Burn Approach")
    elif t <= 2309:
      phases.append("2. Main Engine Burn (MOI)")
    else:
      phases.append("3. Post-Burn Ascent")
  df["flight_phase"] = phases

  return df


def compute_correlations(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
  """Compute Pearson (linear) and Spearman (monotonic rank) correlation matrices."""
  numeric_cols = [
      "TOF_Alt",
      "Magx",
      "Magy",
      "Magz",
      "Mag_heading",
      "accelx",
      "accely",
      "accelz",
      "angle roll",
      "pitch",
      "yaw",
      "ldr",
      "accel_mag",
      "mag_total",
  ]
  pearson = df[numeric_cols].corr(method="pearson")
  spearman = df[numeric_cols].corr(method="spearman")
  return pearson, spearman


def compute_vif(df: pd.DataFrame) -> pd.DataFrame:
  """Compute Variance Inflation Factors (VIF) to assess multicollinearity."""
  cols = [
      "TOF_Alt",
      "Magx",
      "Magy",
      "Magz",
      "accelx",
      "accely",
      "accelz",
      "angle roll",
      "pitch",
      "yaw",
      "ldr",
  ]
  corr_mat = df[cols].corr().values
  vif_vals = np.diag(inv(corr_mat))
  return pd.DataFrame({"Feature": cols, "VIF": vif_vals}).sort_values(
      by="VIF", ascending=False
  )


def get_flight_phase_summary(df: pd.DataFrame) -> pd.DataFrame:
  """Aggregate key telemetry metrics across flight phases."""
  grouped = df.groupby("flight_phase")
  summary = grouped.agg(
      duration_sec=("elapsed_sec", "count"),
      alt_min_km=("TOF_Alt", "min"),
      alt_max_km=("TOF_Alt", "max"),
      accel_mean_mps2=("accel_mag", "mean"),
      accel_max_mps2=("accel_mag", "max"),
      mag_mean=("mag_total", "mean"),
      mag_max=("mag_total", "max"),
      pitch_mean_deg=("pitch", "mean"),
      yaw_mean_deg=("yaw", "mean"),
      ldr_mean=("ldr", "mean"),
  )
  return summary


if __name__ == "__main__":
  df = load_telemetry()
  print(f"Loaded {len(df)} telemetry frames.")
  print("\n--- Flight Phase Summary ---")
  print(get_flight_phase_summary(df).to_string())

  print("\n--- Variance Inflation Factors (Multicollinearity) ---")
  print(compute_vif(df).to_string(index=False))

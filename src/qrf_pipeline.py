"""
Quantile Random Forest (QRF) Pipeline for Spacecraft Telemetry.

Provides data loading, temporal feature engineering, multi-horizon target construction,
strict chronological splitting, residual/delta target modeling, and model serialization.
"""

from dataclasses import dataclass
import json
import os
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from quantile_forest import RandomForestQuantileRegressor
from sklearn.preprocessing import StandardScaler

DEFAULT_QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
HORIZONS = [1, 3, 5]
TARGET_COLS = ["Mag_heading", "TOF_Alt"]

FEATURE_COLS = [
    # Instantaneous sensor telemetry at time t
    "TOF_Alt",
    "Magx",
    "Magy",
    "Magz",
    "mag_total",
    "Mag_heading",
    "sin_heading",
    "cos_heading",
    "accelx",
    "accely",
    "accelz",
    "accel_mag",
    "angle roll",
    "pitch",
    "yaw",
    "ldr",
    # Kinetic and dynamic rate-of-change features (historical only, no future leak)
    "delta_TOF_Alt_1s",
    "delta_TOF_Alt_3s",
    "delta2_TOF_Alt_1s",
    "delta2_TOF_Alt_3s",
    "delta_heading_1s",
    "delta_pitch_1s",
    "delta_yaw_1s",
    "delta_roll_1s",
    # Rolling summary statistics over past window
    "accel_mag_roll3_mean",
    "accel_mag_roll3_std",
    "TOF_Alt_roll3_mean",
    "delta_TOF_Alt_roll3_mean",
    "delta_TOF_Alt_roll5_mean",
    # Physical vector projections
    "accel_vertical_proj",
]


def resolve_dataset_path(provided_path: Optional[str] = None) -> str:
  """Resolve dataset path supporting container/linux paths and local repository."""
  candidates = []
  if provided_path:
    candidates.append(provided_path)
  # Standard container / linux mount
  candidates.append("/mnt/data/mangalyaan_mars_orbit_insertion_simulated.csv")
  # Local workspace relative path
  candidates.append(
      os.path.join("data", "mangalyaan_mars_orbit_insertion_simulated.csv")
  )
  # Workspace absolute path
  base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  candidates.append(
      os.path.join(
          base_dir, "data", "mangalyaan_mars_orbit_insertion_simulated.csv"
      )
  )

  for candidate in candidates:
    if os.path.exists(candidate):
      return candidate

  raise FileNotFoundError(
      f"Telemetry dataset not found in candidate locations: {candidates}"
  )


class TelemetryFeaturePipeline:
  """Feature engineering and chronological dataset preparation for spacecraft telemetry."""

  def __init__(
      self,
      feature_cols: Optional[List[str]] = None,
      use_residual: bool = True,
  ):
    self.feature_cols = feature_cols or FEATURE_COLS
    self.use_residual = use_residual
    self.scaler = StandardScaler()
    self.is_fitted = False

  def load_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
    """Load raw telemetry CSV, parse timestamps, and sort chronologically."""
    resolved_path = resolve_dataset_path(file_path)
    df = pd.read_csv(resolved_path)

    # 1. Parse UTC timestamp and enforce chronological sorting
    df["utc_time"] = pd.to_datetime(df["utc_time"], utc=True)
    df = df.sort_values("utc_time").reset_index(drop=True)

    # Calculate elapsed seconds from mission start
    df["elapsed_sec"] = (
        df["utc_time"] - df["utc_time"].iloc[0]
    ).dt.total_seconds()
    return df

  def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
    """Compute physical derived features, cyclical encodings, and historical rates."""
    df_feat = df.copy()

    # Derived vector magnitudes
    df_feat["accel_mag"] = np.sqrt(
        df_feat["accelx"] ** 2 + df_feat["accely"] ** 2 + df_feat["accelz"] ** 2
    )
    df_feat["mag_total"] = np.sqrt(
        df_feat["Magx"] ** 2 + df_feat["Magy"] ** 2 + df_feat["Magz"] ** 2
    )

    # Circular encoding for magnetic heading [0, 360 deg)
    heading_rad = np.radians(df_feat["Mag_heading"])
    df_feat["sin_heading"] = np.sin(heading_rad)
    df_feat["cos_heading"] = np.cos(heading_rad)

    # Historical velocity & acceleration proxies (strictly backward-looking)
    df_feat["delta_TOF_Alt_1s"] = df_feat["TOF_Alt"].diff(1)
    df_feat["delta_TOF_Alt_3s"] = df_feat["TOF_Alt"].diff(3)

    # Second-difference vertical acceleration proxies
    df_feat["delta2_TOF_Alt_1s"] = df_feat["delta_TOF_Alt_1s"].diff(1)
    df_feat["delta2_TOF_Alt_3s"] = df_feat["delta_TOF_Alt_3s"].diff(1)

    # Unrolled angular velocity for heading (avoiding wrap-around discontinuity at 0/360)
    raw_heading_diff = df_feat["Mag_heading"].diff(1)
    df_feat["delta_heading_1s"] = ((raw_heading_diff + 180.0) % 360.0) - 180.0

    # Attitude angular rates
    df_feat["delta_pitch_1s"] = df_feat["pitch"].diff(1)
    df_feat["delta_yaw_1s"] = df_feat["yaw"].diff(1)
    df_feat["delta_roll_1s"] = df_feat["angle roll"].diff(1)

    # Rolling window statistics (past window = 3 samples, strictly historical)
    df_feat["accel_mag_roll3_mean"] = (
        df_feat["accel_mag"].rolling(window=3, min_periods=3).mean()
    )
    df_feat["accel_mag_roll3_std"] = (
        df_feat["accel_mag"].rolling(window=3, min_periods=3).std().fillna(0.0)
    )
    df_feat["TOF_Alt_roll3_mean"] = (
        df_feat["TOF_Alt"].rolling(window=3, min_periods=3).mean()
    )

    # Rolling climb rate indicators
    df_feat["delta_TOF_Alt_roll3_mean"] = (
        df_feat["delta_TOF_Alt_1s"].rolling(window=3, min_periods=1).mean()
    )
    df_feat["delta_TOF_Alt_roll5_mean"] = (
        df_feat["delta_TOF_Alt_1s"].rolling(window=5, min_periods=1).mean()
    )

    # Projected vertical acceleration from spacecraft body frame
    pitch_rad = np.radians(df_feat["pitch"])
    roll_rad = np.radians(df_feat["angle roll"])
    df_feat["accel_vertical_proj"] = (
        df_feat["accelz"] * np.cos(pitch_rad) * np.cos(roll_rad)
        + df_feat["accelx"] * np.sin(pitch_rad)
    )

    return df_feat

  def construct_targets(
      self,
      df: pd.DataFrame,
      horizons: List[int] = HORIZONS,
      targets: List[str] = TARGET_COLS,
  ) -> pd.DataFrame:
    """Construct multi-horizon future prediction targets and residual increments."""
    df_targets = df.copy()
    for target in targets:
      for h in horizons:
        # Standard future level target: Y(t+h)
        target_col = f"target_{target}_t+{h}"
        df_targets[target_col] = df_targets[target].shift(-h)

        # Residual / differential increment target: Delta Y_{t+h}
        delta_col = f"delta_target_{target}_t+{h}"
        if target == "Mag_heading":
          raw_diff = df_targets[target].shift(-h) - df_targets[target]
          df_targets[delta_col] = ((raw_diff + 180.0) % 360.0) - 180.0
        elif target == "TOF_Alt":
          # Second-order kinematic residual: Alt(t+h) - [Alt(t) + h * v_z(t)]
          # delta_TOF_Alt_1s is the vertical climb rate (m/s)
          v_z = df_targets["delta_TOF_Alt_1s"].bfill().fillna(0.0)
          base_extrap = df_targets["TOF_Alt"] + h * v_z
          df_targets[delta_col] = df_targets[target_col] - base_extrap
        else:
          df_targets[delta_col] = df_targets[target].shift(-h) - df_targets[target]

    return df_targets

  def prepare_dataset(
      self,
      file_path: Optional[str] = None,
      horizons: List[int] = HORIZONS,
      targets: List[str] = TARGET_COLS,
  ) -> pd.DataFrame:
    """Execute end-to-end loading, feature engineering, and target construction."""
    raw_df = self.load_data(file_path)
    feat_df = self.engineer_features(raw_df)
    full_df = self.construct_targets(feat_df, horizons=horizons, targets=targets)
    return full_df

  def chronological_split(
      self,
      df: pd.DataFrame,
      train_ratio: float = 0.70,
      val_ratio: float = 0.15,
  ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset chronologically into train, validation, and test sets."""
    n_total = len(df)
    train_end = int(n_total * train_ratio)
    val_end = int(n_total * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df


class QuantileRandomForestPipeline:
  """Trained QRF model container holding preprocessor, regressor, and metadata."""

  def __init__(
      self,
      target_name: str,
      horizon: int,
      use_residual: bool = True,
      n_estimators: int = 150,
      max_depth: Optional[int] = 16,
      min_samples_split: int = 2,
      min_samples_leaf: int = 3,
      max_features: Any = "sqrt",
      max_samples: Optional[float] = None,
      random_state: int = 42,
      quantiles: Optional[List[float]] = None,
      params: Optional[Dict[str, Any]] = None,
  ):
    self.target_name = target_name
    self.horizon = horizon
    self.use_residual = use_residual
    self.target_col = f"target_{target_name}_t+{horizon}"
    self.train_target_col = (
        f"delta_target_{target_name}_t+{horizon}" if use_residual else self.target_col
    )
    self.quantiles = quantiles or DEFAULT_QUANTILES
    self.random_state = random_state

    # Merge custom/tuned hyperparameters if provided
    model_params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features,
        "max_samples": max_samples,
        "random_state": random_state,
        "bootstrap": True,
        "n_jobs": -1,
    }
    if params:
      model_params.update(params)

    self.model = RandomForestQuantileRegressor(**model_params)
    self.scaler = StandardScaler()
    self.feature_cols = FEATURE_COLS

  def fit(self, train_df: pd.DataFrame) -> "QuantileRandomForestPipeline":
    """Clean NaN target rows and train QRF on target or residual."""
    valid_train = train_df.dropna(subset=self.feature_cols + [self.train_target_col])
    X = valid_train[self.feature_cols].values
    y = valid_train[self.train_target_col].values

    X_scaled = self.scaler.fit_transform(X)
    self.model.fit(X_scaled, y)
    return self

  def predict_quantiles(
      self,
      df: pd.DataFrame,
      quantiles: Optional[List[float]] = None,
  ) -> np.ndarray:
    """Generate conditional quantile predictions for input features.

    If use_residual=True, automatically reconstructs the full target level:
    hat{Y}_{t+h}^{(alpha)} = Y_t + hat{delta}_{t+h}^{(alpha)}
    """
    q_list = quantiles or self.quantiles
    X = df[self.feature_cols].values
    X_scaled = self.scaler.transform(X)
    raw_preds = self.model.predict(X_scaled, quantiles=q_list)

    if not self.use_residual:
      return raw_preds

    # Level reconstruction
    current_val = df[self.target_name].values[:, None]
    if self.target_name == "Mag_heading":
      reconstructed = (current_val + raw_preds) % 360.0
    elif self.target_name == "TOF_Alt":
      v_z = df["delta_TOF_Alt_1s"].bfill().fillna(0.0).values[:, None]
      base_alt = current_val + self.horizon * v_z
      reconstructed = base_alt + raw_preds
    else:
      reconstructed = current_val + raw_preds

    return reconstructed

  def save(self, filepath: str) -> None:
    """Serialize model pipeline to disk using joblib."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    payload = {
        "target_name": self.target_name,
        "horizon": self.horizon,
        "use_residual": self.use_residual,
        "target_col": self.target_col,
        "train_target_col": self.train_target_col,
        "quantiles": self.quantiles,
        "feature_cols": self.feature_cols,
        "scaler": self.scaler,
        "model": self.model,
        "random_state": self.random_state,
    }
    joblib.dump(payload, filepath)

  @classmethod
  def load(cls, filepath: str) -> "QuantileRandomForestPipeline":
    """Deserialize model pipeline from disk."""
    payload = joblib.load(filepath)
    instance = cls(
        target_name=payload["target_name"],
        horizon=payload["horizon"],
        use_residual=payload.get("use_residual", True),
        random_state=payload["random_state"],
        quantiles=payload["quantiles"],
    )
    instance.train_target_col = payload.get(
        "train_target_col",
        f"delta_target_{instance.target_name}_t+{instance.horizon}"
        if instance.use_residual
        else instance.target_col,
    )
    instance.feature_cols = payload["feature_cols"]
    instance.scaler = payload["scaler"]
    instance.model = payload["model"]
    return instance


def load_best_hyperparameters(filepath: str = "models/best_hyperparameters.json") -> Dict[str, Any]:
  """Load tuned hyperparameter dictionary from json file if present, else empty dict."""
  if os.path.exists(filepath):
    try:
      with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
    except Exception as e:
      print(f"Warning: Could not load tuned hyperparameters from {filepath}: {e}")
  return {}


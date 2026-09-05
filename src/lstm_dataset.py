"""
src/lstm_dataset.py
-------------------
Telemetry dataset loading, chronological splitting, zero-leakage feature scaling,
and sliding-window sequence dataset generation for multi-horizon LSTM forecasting.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# Canonical 8 Target Variables as specified by user requirements
TARGET_COLS: List[str] = [
    "Mag_heading",
    "Magx",
    "Magy",
    "Magz",
    "angle roll",
    "pitch",
    "yaw",
    "TOF_Alt",
]

# Canonical 12 Feature Variables (8 targets + 4 exogenous telemetry channels)
FEATURE_COLS: List[str] = [
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
]

# Horizons in seconds (1 Hz sampling rate => 1 timestep = 1 second)
HORIZONS: List[int] = [1, 3, 5]

# Evaluated Quantiles for Probabilistic Forecast
QUANTILES: List[float] = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]


def resolve_dataset_path(custom_path: Optional[Union[str, Path]] = None) -> Path:
    """Resolve the dataset file path checking given, workspace, and standard mount locations."""
    if custom_path:
        p = Path(custom_path)
        if p.exists():
            return p

    candidates = [
        Path("data/mangalyaan_mars_orbit_insertion_simulated.csv"),
        Path("/mnt/data/mangalyaan_mars_orbit_insertion_simulated.csv"),
        Path("d:/cur_projects/mars_projection/data/mangalyaan_mars_orbit_insertion_simulated.csv"),
    ]
    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        "Telemetry dataset not found. Checked default workspace and mount locations."
    )


def load_and_preprocess_telemetry(
    file_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Load raw CSV telemetry, parse 'utc_time' to UTC timestamps, sort chronologically,
    and verify monotonic timestamp spacing without missing rows.
    """
    path = resolve_dataset_path(file_path)
    df = pd.read_csv(path)

    if "utc_time" not in df.columns:
        raise ValueError("Dataset missing required 'utc_time' column.")

    df["utc_time"] = pd.to_datetime(df["utc_time"])
    df = df.sort_values("utc_time").reset_index(drop=True)

    # Verify monotonic continuity
    dt_diff = df["utc_time"].diff().dropna()
    is_regular = (dt_diff == pd.Timedelta(seconds=1)).all()
    if not is_regular:
        print("Warning: Timestamps are not perfectly 1-second continuous.")

    return df


class TelemetrySequenceDataset(Dataset):
    """
    PyTorch Dataset generating sliding-window sequences for multi-horizon telemetry forecasting.

    Each item yields:
      - x: (seq_len, num_features) scaled input history
      - y_delta: (num_horizons, num_targets) target change relative to window terminal state
      - y_actual: (num_horizons, num_targets) ground truth actual values (physical scale)
      - y_terminal: (num_targets,) ground truth target value at final input step t
      - timestamp: string of timestamp at sequence terminal step t
    """

    def __init__(
        self,
        features_scaled: np.ndarray,
        targets_raw: np.ndarray,
        timestamps: pd.Series,
        seq_len: int = 60,
        horizons: List[int] = HORIZONS,
        target_indices_in_features: Optional[List[int]] = None,
    ):
        self.features_scaled = torch.tensor(features_scaled, dtype=torch.float32)
        self.targets_raw = torch.tensor(targets_raw, dtype=torch.float32)
        self.timestamps = list(timestamps)
        self.seq_len = seq_len
        self.horizons = horizons
        self.max_h = max(horizons)

        # Indices of the 8 targets inside the 12 features array
        if target_indices_in_features is None:
            self.target_indices = [FEATURE_COLS.index(col) for col in TARGET_COLS]
        else:
            self.target_indices = target_indices_in_features

        # Precompute valid terminal sequence indices t
        # Valid window ends at t where t - seq_len + 1 >= 0 and t + max_h < len(features)
        self.valid_terminal_indices = []
        n_total = len(features_scaled)
        for t in range(self.seq_len - 1, n_total - self.max_h):
            self.valid_terminal_indices.append(t)

        self._precompute_tensors()

    def _precompute_tensors(self) -> None:
        """Precompute all sequences and targets into contiguous tensors for instant DataLoader access."""
        if not self.valid_terminal_indices:
            self.x_all = torch.empty((0, self.seq_len, self.features_scaled.shape[1]), dtype=torch.float32)
            self.y_delta_all = torch.empty((0, len(self.horizons), len(TARGET_COLS)), dtype=torch.float32)
            self.y_actual_all = torch.empty((0, len(self.horizons), len(TARGET_COLS)), dtype=torch.float32)
            self.y_terminal_all = torch.empty((0, len(TARGET_COLS)), dtype=torch.float32)
            return

        x_list = []
        y_delta_list = []
        y_actual_list = []
        y_terminal_list = []

        for t in self.valid_terminal_indices:
            start_idx = t - self.seq_len + 1
            x_list.append(self.features_scaled[start_idx : t + 1])
            y_term = self.targets_raw[t]
            y_terminal_list.append(y_term)

            act_h = [self.targets_raw[t + h] for h in self.horizons]
            act_tensor = torch.stack(act_h, dim=0)
            y_actual_list.append(act_tensor)

            delta_tensor = act_tensor - y_term.unsqueeze(0)
            # Circular correction for Mag_heading (index 0)
            h_diff = delta_tensor[:, 0]
            delta_tensor[:, 0] = ((h_diff + 180.0) % 360.0) - 180.0
            y_delta_list.append(delta_tensor)

        self.x_all = torch.stack(x_list, dim=0)
        self.y_delta_all = torch.stack(y_delta_list, dim=0)
        self.y_actual_all = torch.stack(y_actual_list, dim=0)
        self.y_terminal_all = torch.stack(y_terminal_list, dim=0)

    def filter_terminal_indices(self, min_t: int) -> None:
        """Filter dataset to sequences with terminal index >= min_t and update precomputed buffers."""
        keep_indices = [i for i, t in enumerate(self.valid_terminal_indices) if t >= min_t]
        self.valid_terminal_indices = [self.valid_terminal_indices[i] for i in keep_indices]
        if keep_indices:
            idx_t = torch.tensor(keep_indices, dtype=torch.long)
            self.x_all = self.x_all[idx_t]
            self.y_delta_all = self.y_delta_all[idx_t]
            self.y_actual_all = self.y_actual_all[idx_t]
            self.y_terminal_all = self.y_terminal_all[idx_t]
        else:
            self._precompute_tensors()

    def __len__(self) -> int:
        return len(self.valid_terminal_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t = self.valid_terminal_indices[idx]
        return {
            "x": self.x_all[idx],
            "y_delta": self.y_delta_all[idx],
            "y_actual": self.y_actual_all[idx],
            "y_terminal": self.y_terminal_all[idx],
            "timestamp": str(self.timestamps[t]),
            "t_idx": torch.tensor(t, dtype=torch.long),
        }


def reconstruct_physical_predictions(
    y_delta_pred: torch.Tensor,
    y_terminal: torch.Tensor,
) -> torch.Tensor:
    """
    Reconstruct absolute physical level forecasts from predicted deltas and terminal states.

    Args:
        y_delta_pred: (B, H, T, Q) tensor of predicted quantile deltas.
        y_terminal: (B, T) tensor of target actuals at sequence end step t.

    Returns:
        y_reconstructed: (B, H, T, Q) tensor of predicted physical quantiles.
    """
    # Expand y_terminal to (B, 1, T, 1) for broadcasting
    y_term_expanded = y_terminal.unsqueeze(1).unsqueeze(-1)
    y_reconstructed = y_term_expanded + y_delta_pred

    # Circular modulo reconstruction for Mag_heading (target index 0)
    heading_recon = y_reconstructed[:, :, 0, :]
    y_reconstructed[:, :, 0, :] = torch.remainder(heading_recon, 360.0)

    return y_reconstructed


def create_telemetry_dataloaders(
    file_path: Optional[Union[str, Path]] = None,
    seq_len: int = 60,
    horizons: List[int] = HORIZONS,
    batch_size: int = 32,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    scaler_save_path: Optional[Union[str, Path]] = "models/lstm_scaler.joblib",
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler, pd.DataFrame]:
    """
    Prepare chronological train, validation, and test datasets and DataLoaders.
    Zero data leakage: StandardScaler is fitted strictly on train_df[FEATURE_COLS].
    """
    df = load_and_preprocess_telemetry(file_path)
    n = len(df)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    # Fit StandardScaler strictly on training set features
    scaler = StandardScaler()
    train_feat_scaled = scaler.fit_transform(train_df[FEATURE_COLS].values)

    # Save fitted scaler
    if scaler_save_path:
        p = Path(scaler_save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, p)

    # Transform full dataset using training-fitted scaler
    full_feat_scaled = scaler.transform(df[FEATURE_COLS].values)
    full_targets_raw = df[TARGET_COLS].values

    # Build sequence datasets with proper boundaries:
    # Train: sequences whose target (t + max_h) is within train_end
    # Validation: sequences whose terminal t is in [train_end, val_end)
    # Test: sequences whose terminal t is in [val_end, n)
    max_h = max(horizons)

    # Train dataset slice: [0 : train_end]
    ds_train = TelemetrySequenceDataset(
        features_scaled=full_feat_scaled[:train_end],
        targets_raw=full_targets_raw[:train_end],
        timestamps=df["utc_time"].iloc[:train_end],
        seq_len=seq_len,
        horizons=horizons,
    )

    # For validation and test, sequences look back into preceding telemetry
    # but their terminal index t strictly lies within the val / test partition
    ds_val = TelemetrySequenceDataset(
        features_scaled=full_feat_scaled[:val_end],
        targets_raw=full_targets_raw[:val_end],
        timestamps=df["utc_time"].iloc[:val_end],
        seq_len=seq_len,
        horizons=horizons,
    )
    # Filter validation valid indices to those with terminal t >= train_end
    ds_val.filter_terminal_indices(train_end)

    ds_test = TelemetrySequenceDataset(
        features_scaled=full_feat_scaled,
        targets_raw=full_targets_raw,
        timestamps=df["utc_time"],
        seq_len=seq_len,
        horizons=horizons,
    )
    # Filter test valid indices to those with terminal t >= val_end
    ds_test.filter_terminal_indices(val_end)

    train_loader = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,  # Shuffle train batches across temporal sequences for SGD stability
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,  # Strictly chronological validation
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        ds_test,
        batch_size=batch_size,
        shuffle=False,  # Strictly chronological test
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader, scaler, df

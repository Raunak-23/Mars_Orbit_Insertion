"""
src/lstm_model.py
-----------------
PyTorch model architecture for Multi-Output, Multi-Horizon Quantile LSTM,
supporting Causal (unidirectional) LSTM and Bidirectional LSTM (BiLSTM),
and Multi-Quantile Pinball Loss with monotonicity (non-crossing) regularization.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn

# Default evaluated quantiles
DEFAULT_QUANTILES: List[float] = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]


class MultiHorizonQuantileLSTM(nn.Module):
    """
    Multi-Output, Multi-Horizon Quantile Recurrent Neural Network.

    Architecture:
      - Recurrent Backbone: Configurable LSTM or BiLSTM layers with recurrent dropout.
      - Sequence Pooling: Terminal temporal state h_t represents past context.
      - Multi-Horizon Heads: Horizon-specific dense projection blocks for t+1, t+3, t+5.
      - Quantile Dimension: Directly outputs Q quantiles per target and horizon.

    Shapes:
      - Input X: (B, seq_len, input_dim)
      - Output: (B, num_horizons, num_targets, num_quantiles)
    """

    def __init__(
        self,
        input_dim: int = 12,
        num_targets: int = 8,
        num_horizons: int = 3,
        quantiles: List[float] = DEFAULT_QUANTILES,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_targets = num_targets
        self.num_horizons = num_horizons
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.bidirectional = bidirectional

        # Input projection layer to match hidden dimension and stabilize scale
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout if dropout > 0 else 0.0),
        )

        # Recurrent Backbone
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        backbone_out_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # Horizon-Specific Quantile Prediction Heads
        # Separate projection heads for each horizon to learn distinct near vs far dynamics
        self.horizon_heads = nn.ModuleList()
        for _ in range(num_horizons):
            head = nn.Sequential(
                nn.Linear(backbone_out_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_targets * self.num_quantiles),
            )
            self.horizon_heads.append(head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: (B, seq_len, input_dim) input sequence tensor.
        Returns:
            out: (B, num_horizons, num_targets, num_quantiles)
        """
        # Project inputs
        x_proj = self.input_proj(x)  # (B, seq_len, hidden_dim)

        # LSTM recurrent pass
        lstm_out, _ = self.lstm(x_proj)  # (B, seq_len, backbone_out_dim)

        # Extract terminal representation h_t at final sequence step
        h_t = lstm_out[:, -1, :]  # (B, backbone_out_dim)

        # Project through each horizon-specific head
        horizon_preds = []
        for head in self.horizon_heads:
            pred_flat = head(h_t)  # (B, num_targets * num_quantiles)
            pred_reshaped = pred_flat.view(
                -1, self.num_targets, self.num_quantiles
            )  # (B, T, Q)
            horizon_preds.append(pred_reshaped)

        # Stack across horizons: (B, H, T, Q)
        out = torch.stack(horizon_preds, dim=1)
        return out


class MultiQuantilePinballLoss(nn.Module):
    """
    Multi-Quantile Pinball (Tilted Absolute) Loss Function with optional
    Non-Crossing Monotonicity Regularization.

    For quantile tau in (0, 1) and residual e = y - y_hat:
        L_tau(y, y_hat) = max(tau * e, (tau - 1) * e)

    Monotonicity penalty:
        L_cross = lambda_cross * sum(ReLU(y_hat_k - y_hat_{k+1}))
    """

    def __init__(
        self,
        quantiles: List[float] = DEFAULT_QUANTILES,
        crossing_penalty_weight: float = 0.05,
    ):
        super().__init__()
        self.register_buffer(
            "quantiles", torch.tensor(quantiles, dtype=torch.float32)
        )
        self.crossing_penalty_weight = crossing_penalty_weight

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            y_pred: (B, H, T, Q) predicted quantiles.
            y_true: (B, H, T) ground truth actual targets.
        Returns:
            total_loss: scalar tensor.
        """
        # Expand y_true to (B, H, T, 1) for broadcasting across Q quantiles
        y_true_expanded = y_true.unsqueeze(-1)  # (B, H, T, 1)

        # Residual: e = y - y_hat
        diff = y_true_expanded - y_pred  # (B, H, T, Q)

        # quantiles buffer reshaped to (1, 1, 1, Q)
        q = self.quantiles.view(1, 1, 1, -1)

        # Pinball loss: max(q * diff, (q - 1) * diff)
        pinball = torch.maximum(q * diff, (q - 1.0) * diff)
        mean_pinball = torch.mean(pinball)

        # Monotonicity penalty to discourage quantile crossing:
        # y_pred[..., k] should be <= y_pred[..., k+1]
        if self.crossing_penalty_weight > 0 and y_pred.shape[-1] > 1:
            crossing_violations = torch.relu(y_pred[..., :-1] - y_pred[..., 1:])
            crossing_loss = torch.mean(crossing_violations)
            total_loss = mean_pinball + self.crossing_penalty_weight * crossing_loss
        else:
            total_loss = mean_pinball

        return total_loss

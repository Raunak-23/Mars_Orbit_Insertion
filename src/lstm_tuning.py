"""
src/lstm_tuning.py
------------------
Systematic hyperparameter tuning for Multi-Horizon Quantile LSTM
with train-test generalization gap penalization and diagnostic visualization.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from src.lstm_dataset import create_telemetry_dataloaders
from src.lstm_training import set_seed, train_telemetry_lstm

# Curated high-impact search space across sequence lengths, dimensions, and architectures
PARAM_GRID: List[Dict[str, Any]] = [
    # Causal LSTMs (Primary operational deployment candidates)
    {"seq_len": 30, "hidden_dim": 64, "num_layers": 1, "dropout": 0.1, "lr": 1e-3, "bidirectional": False},
    {"seq_len": 30, "hidden_dim": 128, "num_layers": 2, "dropout": 0.2, "lr": 1e-3, "bidirectional": False},
    {"seq_len": 60, "hidden_dim": 64, "num_layers": 1, "dropout": 0.1, "lr": 1e-3, "bidirectional": False},
    {"seq_len": 60, "hidden_dim": 128, "num_layers": 2, "dropout": 0.2, "lr": 1e-3, "bidirectional": False},
    {"seq_len": 60, "hidden_dim": 128, "num_layers": 2, "dropout": 0.25, "lr": 5e-4, "bidirectional": False},
    {"seq_len": 60, "hidden_dim": 192, "num_layers": 2, "dropout": 0.25, "lr": 1e-3, "bidirectional": False},
    {"seq_len": 120, "hidden_dim": 128, "num_layers": 2, "dropout": 0.2, "lr": 1e-3, "bidirectional": False},
    # Bidirectional LSTM (Offline flight trajectory reconstruction candidate)
    {"seq_len": 60, "hidden_dim": 128, "num_layers": 2, "dropout": 0.2, "lr": 1e-3, "bidirectional": True},
]


def compute_generalization_selection_score(
    train_loss: float,
    val_loss: float,
    gap_weight: float = 0.5,
    ratio_penalty_weight: float = 0.15,
) -> float:
    """
    Compute selection score penalizing both high validation error and severe train-val gap.
    score = Val_Loss + gap_weight * max(0, Val_Loss - Train_Loss) + ratio_penalty
    """
    gen_gap = max(0.0, val_loss - train_loss)
    ratio = val_loss / max(1e-6, train_loss)
    ratio_excess = max(0.0, ratio - 1.25)

    score = val_loss + gap_weight * gen_gap + ratio_penalty_weight * ratio_excess * val_loss
    return score


def run_lstm_hyperparameter_tuning(
    dataset_path: Optional[Union[str, Path]] = None,
    candidate_configs: Optional[List[Dict[str, Any]]] = None,
    epochs_per_trial: int = 25,
    batch_size: int = 64,
    patience: int = 6,
    reports_dir: Union[str, Path] = "reports",
    models_dir: Union[str, Path] = "models",
    verbose: bool = True,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Iterate over candidate LSTM hyperparameter sets, evaluate on validation set,
    compute generalization loss, and select the optimal generalizing model.
    """
    try:
        torch.set_num_threads(4)
    except Exception:
        pass
    if candidate_configs is None:
        configs = PARAM_GRID
    else:
        configs = candidate_configs

    rep_path = Path(reports_dir)
    rep_path.mkdir(parents=True, exist_ok=True)
    fig_path = rep_path / "figures"
    fig_path.mkdir(parents=True, exist_ok=True)

    results = []

    print(f"\n=======================================================")
    print(f"Starting LSTM Hyperparameter Tuning ({len(configs)} configurations)")
    print(f"=======================================================")

    # Cache DataLoaders by seq_len to avoid redundant dataset construction
    loader_cache = {}

    for i, p_cfg in enumerate(configs, start=1):
        s_len = p_cfg["seq_len"]
        if s_len not in loader_cache:
            tr_loader, v_loader, te_loader, _, _ = create_telemetry_dataloaders(
                file_path=dataset_path,
                seq_len=s_len,
                batch_size=batch_size,
                scaler_save_path=None,
            )
            loader_cache[s_len] = (tr_loader, v_loader, te_loader)
        else:
            tr_loader, v_loader, te_loader = loader_cache[s_len]

        run_cfg = copy_cfg = dict(p_cfg)
        run_cfg["epochs"] = epochs_per_trial
        run_cfg["patience"] = patience
        run_cfg["batch_size"] = batch_size

        mode_str = "BiLSTM" if run_cfg["bidirectional"] else "Causal"
        print(
            f"\n[Trial {i:02d}/{len(configs):02d}] seq_len={run_cfg['seq_len']} | "
            f"hidden={run_cfg['hidden_dim']} | layers={run_cfg['num_layers']} | "
            f"mode={mode_str} | lr={run_cfg['lr']} | drop={run_cfg['dropout']}"
        )

        model, history, fin_cfg = train_telemetry_lstm(
            train_loader=tr_loader,
            val_loader=v_loader,
            config=run_cfg,
            checkpoint_dir=models_dir,
            verbose=False,
        )

        tr_loss = history["train_loss"][-1] if history["train_loss"] else float("nan")
        val_loss = fin_cfg["best_val_loss"]
        gen_gap = val_loss - tr_loss
        score = compute_generalization_selection_score(tr_loss, val_loss)

        print(
            f"  --> Train Pinball: {tr_loss:.4f} | Val Pinball: {val_loss:.4f} | "
            f"Gap: {gen_gap:+.4f} | Gen Score: {score:.4f} (ep {fin_cfg['best_epoch']})"
        )

        rec = {
            "trial": i,
            "seq_len": run_cfg["seq_len"],
            "hidden_dim": run_cfg["hidden_dim"],
            "num_layers": run_cfg["num_layers"],
            "dropout": run_cfg["dropout"],
            "lr": run_cfg["lr"],
            "bidirectional": run_cfg["bidirectional"],
            "architecture": mode_str,
            "train_pinball": float(round(tr_loss, 5)),
            "val_pinball": float(round(val_loss, 5)),
            "gen_gap": float(round(gen_gap, 5)),
            "selection_score": float(round(score, 5)),
            "best_epoch": fin_cfg["best_epoch"],
            "is_best": False,
        }
        results.append(rec)
        # Incremental save
        interim_df = pd.DataFrame(results)
        tuning_csv_path = rep_path / "lstm_tuning_results.csv"
        interim_df.to_csv(tuning_csv_path, index=False)

    results_df = pd.DataFrame(results)

    # Identify best configuration based on composite generalization score
    best_idx = results_df["selection_score"].idxmin()
    results_df.loc[best_idx, "is_best"] = True
    best_row = results_df.loc[best_idx]

    # Final save of marked results
    results_df.to_csv(tuning_csv_path, index=False)
    print(f"\nHyperparameter tuning results saved to {tuning_csv_path}")

    # Best hyperparameter dict
    best_params = {
        "seq_len": int(best_row["seq_len"]),
        "hidden_dim": int(best_row["hidden_dim"]),
        "num_layers": int(best_row["num_layers"]),
        "dropout": float(best_row["dropout"]),
        "lr": float(best_row["lr"]),
        "bidirectional": bool(best_row["bidirectional"]),
        "best_epoch": int(best_row["best_epoch"]),
        "val_pinball": float(best_row["val_pinball"]),
        "train_pinball": float(best_row["train_pinball"]),
        "selection_score": float(best_row["selection_score"]),
    }

    best_json_path = Path(models_dir) / "best_lstm_hyperparameters.json"
    with open(best_json_path, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"Optimal hyperparameters saved to {best_json_path}")

    # Generate diagnostic trade-off plots
    plot_tuning_tradeoffs(
        df=results_df,
        save_path=fig_path / "lstm_hyperparameter_tuning_tradeoffs.png",
    )

    return best_params, results_df


def plot_tuning_tradeoffs(df: pd.DataFrame, save_path: Path) -> None:
    """Generate 4-panel publication-grade hyperparameter tuning trade-off diagnostic plot."""
    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: Train Loss vs Val Loss with y=x generalization diagonal
    ax1 = axes[0, 0]
    sns.scatterplot(
        data=df,
        x="train_pinball",
        y="val_pinball",
        hue="architecture",
        size="seq_len",
        sizes=(60, 220),
        palette={"Causal": "#3b82f6", "BiLSTM": "#8b5cf6"},
        ax=ax1,
        alpha=0.9,
    )
    # Ideal generalization diagonal (y = x)
    min_val = min(df["train_pinball"].min(), df["val_pinball"].min()) * 0.95
    max_val = max(df["train_pinball"].max(), df["val_pinball"].max()) * 1.05
    ax1.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.7, label="Zero Gap (y=x)")

    # Highlight best model
    best_pt = df[df["is_best"]].iloc[0]
    ax1.scatter(
        [best_pt["train_pinball"]],
        [best_pt["val_pinball"]],
        color="#10b981",
        s=350,
        marker="*",
        edgecolors="black",
        linewidths=1.5,
        label=f"Selected Best ({best_pt['architecture']}, L={best_pt['seq_len']}, H={best_pt['hidden_dim']})",
        zorder=10,
    )
    ax1.set_title("Train vs. Validation Pinball Loss (Overfitting Diagnostic)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Train Pinball Loss", fontsize=11)
    ax1.set_ylabel("Validation Pinball Loss", fontsize=11)
    ax1.legend(loc="upper left", frameon=True)

    # Panel 2: Validation Loss vs Sequence Length
    ax2 = axes[0, 1]
    sns.boxplot(
        data=df,
        x="seq_len",
        y="val_pinball",
        hue="architecture",
        palette={"Causal": "#3b82f6", "BiLSTM": "#8b5cf6"},
        ax=ax2,
    )
    ax2.set_title("Validation Loss by Sequence Length (Lookback Horizon)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Sequence Length L (seconds)", fontsize=11)
    ax2.set_ylabel("Validation Pinball Loss", fontsize=11)

    # Panel 3: Validation Loss vs Hidden Dimension
    ax3 = axes[1, 0]
    sns.boxplot(
        data=df,
        x="hidden_dim",
        y="val_pinball",
        hue="num_layers",
        palette="viridis",
        ax=ax3,
    )
    ax3.set_title("Validation Loss by Hidden Dimension & Layer Depth", fontsize=13, fontweight="bold")
    ax3.set_xlabel("LSTM Hidden Dimension", fontsize=11)
    ax3.set_ylabel("Validation Pinball Loss", fontsize=11)

    # Panel 4: Causal LSTM vs BiLSTM Generalization Gap Comparison
    ax4 = axes[1, 1]
    sns.barplot(
        data=df,
        x="trial",
        y="selection_score",
        hue="architecture",
        palette={"Causal": "#3b82f6", "BiLSTM": "#8b5cf6"},
        ax=ax4,
    )
    ax4.axhline(best_pt["selection_score"], color="#10b981", linestyle="--", alpha=0.8, label="Best Selection Score")
    ax4.set_title("Composite Overfitting Selection Score Across Trials", fontsize=13, fontweight="bold")
    ax4.set_xlabel("Tuning Trial Index", fontsize=11)
    ax4.set_ylabel("Selection Score (Val Loss + Gap Penalty)", fontsize=11)
    ax4.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Tuning diagnostic visualization saved to {save_path}")

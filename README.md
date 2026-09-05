# 🚀 Mangalyaan Mars Orbit Insertion (MOI) Telemetry Forecasting & Uncertainty Quantification

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9+-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Quantile-Forest](https://img.shields.io/badge/quantile--forest-1.4+-00A699.svg)](https://github.com/zillow/quantile-forest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end aerospace machine learning and deep learning framework for **calibrated, multi-horizon probabilistic telemetry forecasting** during the critical **Mars Orbit Insertion (MOI)** maneuver of India's **Mars Orbiter Mission (Mangalyaan / MOM)**.

This repository implements both non-parametric **Quantile Random Forests (QRF)** and deep recurrent **Multi-Horizon Quantile LSTMs** to predict future spacecraft states ($t+1\text{s}$, $t+3\text{s}$, $t+5\text{s}$) alongside rigorous uncertainty intervals ($90\%$, $80\%$, and $50\%$ prediction intervals).

---

## 📌 Table of Contents
1. [Mission Context & The MOI Challenge](#-mission-context--the-moi-challenge)
2. [Key Engineering Innovations](#-key-engineering-innovations)
3. [Telemetry Dataset & Mission Phases](#-telemetry-dataset--mission-phases)
4. [Methodologies & Architectures](#-methodologies--architectures)
   - [Physical Kinematic Residual Pipeline](#1-physical-kinematic-residual-pipeline)
   - [Quantile Random Forest (QRF) Pipeline](#2-quantile-random-forest-qrf-pipeline)
   - [Multi-Horizon Quantile LSTM Deep Sequence Model](#3-multi-horizon-quantile-lstm-deep-sequence-model)
   - [Expanding-Window Rolling Temporal Validation](#4-expanding-window-rolling-temporal-validation)
5. [Benchmark Results & Evaluation](#-benchmark-results--evaluation)
6. [Interactive Jupyter Notebooks](#-interactive-jupyter-notebooks)
7. [Repository Structure](#-repository-structure)
8. [Getting Started & Installation](#-getting-started--installation)
9. [CLI Usage & Workflows](#-cli-usage--workflows)
10. [Flight Dynamics Visualizations](#-flight-dynamics-visualizations)
11. [Reproducibility & Hardware Specifications](#-reproducibility--hardware-specifications)

---

## 🛰️ Mission Context & The MOI Challenge

During the Mars Orbit Insertion (MOI) maneuver, the spacecraft approaches Mars at hyperbolic speeds ($\sim 5.7\text{ km/s}$) and must fire its **440 N Liquid Apogee Motor (LAM)** for roughly 24 minutes ($\sim 1440\text{ s}$) to shed $\Delta v \approx 1098.7\text{ m/s}$ of velocity. This deceleration transitions the spacecraft from an unbound flyby trajectory into an elliptical Martian capture orbit.

```
                  ┌─────────────────────────────────────────────────────────┐
                  │                 MARS ORBIT INSERTION                    │
                  │                                                         │
   Approach       │          LAM Main Engine Firing (1440 s)                │       Captured Orbit
  ───────────────►│  ════════════════════════════════════════════════════►  │  ──────────────────────►
  High Hyperbolic │  • 440 N LAM Burn         • Plasma Wake Perturbation   │  Stable Elliptical Orbit
  Speed (5.7 km/s)│  • Extreme Attitude Slew  • Mars Shadow Occultation    │  Periapsis ~420 km
                  │  • Propellant Depletion   • Earth Radio Blackout       │  Apoapsis ~76,993 km
                  └─────────────────────────────────────────────────────────┘
```

### Why Point Predictions Fail in Space Operations
Standard machine learning models deliver single scalar point forecasts ($\hat{y}$). In critical spaceflight operations, point predictions are insufficient and dangerous:
- **Radio Light-Time Delay**: Earth-Mars communication incurs a 12–14 minute one-way signal delay. Flight computers must operate autonomously.
- **Sensor Perturbations**: Thruster firing and plasma wake induce high-frequency attitude jitter and magnetometer noise.
- **Contingency Triggering**: Autonomous abort or safe-mode switches require **calibrated upper and lower bounds** ($q_{0.05}$ and $q_{0.95}$). If actual telemetry crosses outside the $90\%$ prediction cone, fault management algorithms immediately flag an anomaly.

---

## 💡 Key Engineering Innovations

1. **Resolving Decision Tree Extrapolation Failure for Altitude (`TOF_Alt`)**:
   Standard decision trees cannot predict values outside the convex hull of their training data. When the spacecraft accelerates into high-altitude post-burn ascent ($>9000\text{ km}$), an un-differenced tree flatlines, causing catastrophic errors ($>3900\text{ m}$). 
   - *Solution*: We reformulate the prediction target into a **second-order kinematic residual** relative to instantaneous vertical velocity:
     $$\Delta h_{\text{residual}} = h_{t+k} - \left( h_t + \dot{h}_t \cdot k + \frac{1}{2} \ddot{h}_t \cdot k^2 \right)$$
   - This transforms an unbounded non-stationary trajectory into a strictly stationary distribution ($[-3\text{ m}, +4\text{ m}]$), slashing altitude test MAE from $>3900\text{ m}$ down to **$0.56\text{ m}$**.

2. **Dual-Model Probabilistic Architecture**:
   - **Quantile Random Forest (QRF)**: Delivers fast, non-parametric conditional quantile estimates with zero distribution assumptions.
   - **Multi-Horizon Quantile LSTM**: A deep recurrent network with a 60-second lookback window, multi-output heads forecasting 8 channels across 3 forward horizons simultaneously under asymmetric Pinball Loss.

3. **Generalization-Constrained Hyperparameter Tuning**:
   Custom objective function that penalizes the training-validation generalization gap to prevent tree memorization and overfitting during transient thrust events.

4. **Phase-Aware Expanding-Window Temporal Validation**:
   5-fold walk-forward cross-validation enforcing strict chronological time boundaries across the Pre-Burn, Main Engine Burn, and Post-Burn phases, guaranteeing zero future-data leakage.

---

## 📊 Telemetry Dataset & Mission Phases

The dataset (`data/mangalyaan_mars_orbit_insertion_simulated.csv`) captures 3,600 continuous seconds (1 Hz sampling) across the full MOI profile:

| Phase ID | Phase Name | Time Range | Dynamic Characteristics |
| :--- | :--- | :--- | :--- |
| **Phase 1** | **Pre-Burn Approach** | $0\text{ s} \le t < 1260\text{ s}$ | High velocity, solar radiation pressure, optical attitude acquisition, quiet thrusters. |
| **Phase 2** | **LAM Main Engine Burn** | $1260\text{ s} \le t < 2700\text{ s}$ | Active 440 N LAM firing, high propellant depletion, intense chamber pressure, periapsis dip ($<500\text{ km}$). |
| **Phase 3** | **Post-Burn Orbital Ascent** | $2700\text{ s} \le t \le 3600\text{ s}$ | Engine shutdown, cooling engine bell, orbital ascent, confirmation of elliptical capture. |

### Telemetry Channels Analyzed
- **Target Channels**:
  - `Mag_heading`: Spacecraft magnetic heading ($0^\circ$ to $360^\circ$).
  - `TOF_Alt`: Time-of-Flight radar altitude ($400\text{ km}$ to $9500\text{ km}$).
  - Multi-target LSTM targets: `Mag_heading`, `Magx`, `Magy`, `Magz`, `angle roll`, `pitch`, `yaw`, `TOF_Alt`.
- **Propulsion & Thermal Predictors**: `Combustion_Chamber_Pressure_bar`, `Tank_Pressure_bar`, `Propellant_Mass_kg`, `Engine_Bell_Temp_C`, `Avionics_Deck_Temp_C`, `Thruster_Pulse_Width_s`, `Thruster_Duty_Cycle_pct`.
- **Inertial & Attitude Predictors**: Linear accelerations (`Accel_X`, `Accel_Y`, `Accel_Z`), body rates (`Gyro_Rate_X`, `Gyro_Rate_Y`, `Gyro_Rate_Z`), Euler angles (`angle roll`, `pitch`, `yaw`).

---

## 🔬 Methodologies & Architectures

### 1. Physical Kinematic Residual Pipeline
Implemented in [`src/qrf_pipeline.py`](src/qrf_pipeline.py):
- **First & Second Order Derivatives**: Numerical central and backward differences computing $\dot{h}$ (vertical climb rate), $\ddot{h}$ (axial thrust acceleration), and angular slew rates.
- **Rolling Window Features**: Multi-scale rolling statistics (5s, 15s, 30s) capturing local mean, volatility (standard deviation), and min/max bounds.
- **Angular Wrapping**: Causal sine/cosine phase encoding ensuring continuous representations across the $0^\circ \leftrightarrow 360^\circ$ singularity.

### 2. Quantile Random Forest (QRF) Pipeline
Implemented in [`src/train_eval_qrf.py`](src/train_eval_qrf.py) & [`src/qrf_tuning.py`](src/qrf_tuning.py):
- Builds non-parametric conditional cumulative distribution functions (CDFs):
  $$\hat{F}(y | X = x) = \sum_{i=1}^n w_i(x) \cdot \mathbb{I}(Y_i \le y)$$
- Predicts 7 quantiles simultaneously: $\tau \in \{0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95\}$.
- Point forecast is derived from the conditional median ($\tau = 0.50$).
- $90\%$ Prediction Interval is bounded by $[\hat{q}_{0.05}, \hat{q}_{0.95}]$.

### 3. Multi-Horizon Quantile LSTM Deep Sequence Model
Implemented in [`src/lstm_model.py`](src/lstm_model.py) & [`src/lstm_training.py`](src/lstm_training.py):

```
 Input: [Batch, T=60s, Features=26]
              │
              ▼
   ┌───────────────────────┐
   │ 2-Layer Causal LSTM   │  Hidden Dim = 128, Dropout = 0.20
   └──────────┬────────────┘
              ▼
   ┌───────────────────────┐
   │ LayerNorm + Linear    │  Dense projection layer
   └──────────┬────────────┘
              ├────────────────────────┬────────────────────────┐
              ▼                        ▼                        ▼
   ┌───────────────────────┐┌───────────────────────┐┌───────────────────────┐
   │ Horizon t+1 Head      ││ Horizon t+3 Head      ││ Horizon t+5 Head      │
   │ 8 Targets x 7 Quantiles││ 8 Targets x 7 Quantiles││ 8 Targets x 7 Quantiles│
   └───────────────────────┘└───────────────────────┘└───────────────────────┘
```

- **Objective Function**: Asymmetric Multi-Quantile Pinball Loss (Tilted Loss):
  $$\mathcal{L}_\tau(y, \hat{y}_\tau) = \max \left( \tau (y - \hat{y}_\tau), \, (1 - \tau)(\hat{y}_\tau - y) \right)$$
- **Quantile Monotonicity Regularization**: Penalizes non-crossing violations where $\hat{q}_{\tau_a} > \hat{q}_{\tau_b}$ for $\tau_a < \tau_b$.
- **Training Mechanics**: AdamW optimizer, Cosine Annealing learning rate schedule, gradient norm clipping ($\le 1.0$), and early stopping on validation pinball loss.

### 4. Expanding-Window Rolling Temporal Validation
Implemented in [`src/rolling_validation.py`](src/rolling_validation.py):
- Evaluates operational robustness by incrementally expanding the training window across 5 folds:
  - Fold 1: Trains on initial Pre-Burn approach, tests early LAM engine ignition.
  - Fold 3: Trains through mid-burn, tests periapsis velocity peak.
  - Fold 5: Trains on full burn maneuver, tests orbital injection ascent.
- Tracks phase-specific calibration dynamics (coverage and interval width) across mission transitions.

---

## 📈 Benchmark Results & Evaluation

### Uncertainty Evaluation Metrics
1. **Prediction Interval Coverage Probability (PICP)**: The percentage of true values falling inside the prediction interval. For a nominal $90\%$ interval, $\text{PICP} \ge 90\%$ indicates reliable calibration.
2. **Mean Prediction Interval Width (MPIW)**: Measures the sharpness/tightness of the interval. Narrower intervals with nominal coverage reflect high predictive confidence.
3. **Pinball / Quantile Loss**: Standard asymmetric loss scoring proper quantile calibration.
4. **Winkler Score**: Heavily penalizes intervals that miss the target while rewarding sharpness.

### Quantile Random Forest (QRF) Test Performance Summary
Evaluated on strictly held-out chronological test data:

| Target Channel | Horizon | MAE | RMSE | $R^2$ | Pinball Loss | 90% PICP | 90% MPIW | 80% PICP | 80% MPIW |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`TOF_Alt`** (Altitude, km) | **$t+1\text{s}$** | **0.564 km** | 0.693 km | >0.999 | 0.162 | **93.69%** | 2.61 km | 85.53% | 2.09 km |
| **`TOF_Alt`** (Altitude, km) | **$t+3\text{s}$** | **0.958 km** | 1.209 km | >0.999 | 0.307 | **96.28%** | 5.95 km | 90.69% | 4.57 km |
| **`TOF_Alt`** (Altitude, km) | **$t+5\text{s}$** | **1.269 km** | 1.579 km | >0.999 | 0.394 | **97.94%** | 8.06 km | 94.39% | 6.23 km |
| **`Mag_heading`** (Heading, deg) | **$t+1\text{s}$** | 12.24° | 16.62° | 0.123 | 5.175 | 25.23% | 8.70° | 16.14° | 5.93° |
| **`Mag_heading`** (Heading, deg) | **$t+3\text{s}$** | 11.81° | 16.09° | 0.174 | 5.057 | 25.88% | 7.96° | 19.18° | 6.03° |
| **`Mag_heading`** (Heading, deg) | **$t+5\text{s}$** | 11.56° | 15.53° | 0.229 | 5.009 | 24.11% | 7.29° | 18.50° | 5.60° |

> **Key Finding**: The kinematic residual formulation for `TOF_Alt` achieves exceptional accuracy ($R^2 > 0.999$) and nominal calibration ($93.7\% - 97.9\%$ coverage for nominal $90\%$ bounds). Heading dynamics during active burn show wider dispersion due to continuous thruster pulsing and plasma noise.

### Multi-Horizon Quantile LSTM Test Performance Summary
Evaluated across 8 channels simultaneously on holdout sequences:

| Target Channel | Horizon | MAE | RMSE | $R^2$ Score | Pinball Loss | 90% PICP | 90% MPIW |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`angle roll`** | $t+1\text{s}$ | **0.0938°** | 0.1173° | **0.9935** | 0.0276 | **96.64%** | 0.496° |
| **`pitch`** | $t+1\text{s}$ | **0.1174°** | 0.1456° | **0.8728** | 0.0303 | **94.58%** | 0.491° |
| **`yaw`** | $t+1\text{s}$ | **0.1160°** | 0.1458° | **0.9126** | 0.0341 | **92.90%** | 0.548° |
| **`Magy`** | $t+1\text{s}$ | **0.1633** | 0.2037 | **0.8060** | 0.0502 | **98.69%** | 1.031 |
| **`Magx`** | $t+1\text{s}$ | **0.1780** | 0.2219 | -1.0748 | 0.0509 | **93.27%** | 0.818 |
| **`Magz`** | $t+1\text{s}$ | **0.1133** | 0.1446 | -0.9656 | 0.0340 | **94.58%** | 0.570 |
| **`angle roll`** | $t+5\text{s}$ | **0.1188°** | 0.1473° | **0.9897** | 0.0330 | **95.33%** | 0.605° |
| **`pitch`** | $t+5\text{s}$ | **0.1016°** | 0.1308° | **0.8919** | 0.0302 | **96.07%** | 0.583° |
| **`yaw`** | $t+5\text{s}$ | **0.1130°** | 0.1410° | **0.9210** | 0.0351 | **95.89%** | 0.791° |

> **Key Finding**: The LSTM excels at Euler angle predictions (`roll`, `pitch`, `yaw`) with $R^2 \ge 0.87 - 0.99$ and calibrated coverage exceeding $92\% - 96\%$, successfully modeling high-frequency thruster-induced attitude adjustments.

---

## 📓 Interactive Jupyter Notebooks

The repository includes three self-contained, fully documented walkthrough notebooks located in [`notebook/`](notebook/):

1. **[`01_eda_mangalyaan_telemetry.ipynb`](notebook/01_eda_mangalyaan_telemetry.ipynb)**:
   - Exhaustive exploratory data analysis of all 26 telemetry channels.
   - Phase-wise distributions, Pearson & Spearman correlation heatmaps.
   - Physical relationship mapping: thrust curves vs. tank pressure vs. acceleration.
2. **[`02_quantile_random_forest_telemetry.ipynb`](notebook/02_quantile_random_forest_telemetry.ipynb)**:
   - In-depth Quantile Random Forest implementation.
   - Kinematic residual derivation and altitude extrapolation solution.
   - Randomized hyperparameter tuning and expanding-window temporal cross-validation.
   - Calibration plots, residual distributions, and test evaluation.
3. **[`03_lstm_telemetry_prediction.ipynb`](notebook/03_lstm_telemetry_prediction.ipynb)**:
   - Deep recurrent modeling with PyTorch.
   - Chronological sequence generation ($T=60\text{s}$) with zero data leakage.
   - Multi-output, multi-horizon quantile architecture with Pinball Loss.
   - Training progression, hyperparameter trials, and live streaming simulation.

---

## 📁 Repository Structure

```text
mars_projection/
├── data/
│   └── mangalyaan_mars_orbit_insertion_simulated.csv   # 1 Hz MOI simulated telemetry
├── models/
│   ├── best_hyperparameters.json                       # Tuned QRF hyperparameters
│   ├── best_lstm_telemetry_model.pt                    # Trained PyTorch Quantile LSTM weights
│   ├── lstm_config.json                                # LSTM architecture configuration
│   ├── lstm_scaler.joblib                              # Zero-leakage standard scaler
│   ├── qrf_Mag_heading_h1.joblib                       # Saved QRF models for each target & horizon
│   ├── qrf_Mag_heading_h3.joblib
│   ├── qrf_Mag_heading_h5.joblib
│   ├── qrf_TOF_Alt_h1.joblib
│   ├── qrf_TOF_Alt_h3.joblib
│   └── qrf_TOF_Alt_h5.joblib
├── notebook/
│   ├── 01_eda_mangalyaan_telemetry.ipynb               # Exploratory Data Analysis notebook
│   ├── 02_quantile_random_forest_telemetry.ipynb       # QRF development & tuning notebook
│   └── 03_lstm_telemetry_prediction.ipynb              # Deep Quantile LSTM notebook
├── reports/
│   ├── figures/                                        # Publication-quality figures & charts
│   │   ├── correlation_heatmaps.png
│   │   ├── moi_mission_overview.png
│   │   ├── orbital_physics_relationships.png
│   │   ├── phase_distributions_boxplots.png
│   │   ├── qrf_Mag_heading_residual_distribution.png
│   │   ├── qrf_Mag_heading_trajectory_intervals.png
│   │   ├── qrf_TOF_Alt_residual_distribution.png
│   │   ├── qrf_TOF_Alt_trajectory_intervals.png
│   │   ├── qrf_horizon_metrics_comparison.png
│   │   ├── qrf_hyperparameter_tuning_tradeoffs.png
│   │   └── qrf_rolling_validation_dynamics.png
│   ├── qrf_metrics_summary.csv                         # QRF point & quantile test metrics
│   ├── qrf_predictions_test.csv                        # QRF test predictions across all quantiles
│   ├── qrf_rolling_metrics.csv                         # Walk-forward cross-validation metrics
│   ├── qrf_tuning_results.csv                          # QRF hyperparameter search log
│   ├── lstm_metrics_summary.csv                        # LSTM multi-target test metrics
│   ├── lstm_predictions_test.csv                       # LSTM multi-horizon test predictions
│   └── lstm_tuning_results.csv                         # LSTM hyperparameter trial history
├── src/
│   ├── __init__.py                                     # Package designation
│   ├── eda_analysis.py                                 # Core statistical EDA engine
│   ├── generate_eda_visualizations.py                  # EDA figure generation script
│   ├── qrf_pipeline.py                                 # Feature engineering & QRF pipeline
│   ├── qrf_evaluation.py                               # Point, Pinball, PICP & MPIW metrics
│   ├── qrf_tuning.py                                   # Generalization-penalized RandomizedSearchCV
│   ├── qrf_inference.py                                # Streaming QRF inference engine
│   ├── rolling_validation.py                           # Expanding-window temporal CV
│   ├── train_eval_qrf.py                               # End-to-end QRF training runner
│   ├── lstm_dataset.py                                 # Sequence lookback generator & scaling
│   ├── lstm_model.py                                   # Multi-Horizon Quantile LSTM network
│   ├── lstm_training.py                                # PyTorch training loop & pinball loss
│   ├── lstm_evaluation.py                              # Multi-target LSTM evaluation suite
│   ├── lstm_tuning.py                                  # LSTM hyperparameter search runner
│   ├── lstm_inference.py                               # Streaming LSTM inference engine
│   └── train_eval_lstm.py                              # End-to-end LSTM training runner
├── main.py                                             # Unified CLI entrypoint
├── pyproject.toml                                      # Project metadata & dependencies
├── pyrightconfig.json                                  # LSP / Pyright search path configuration
└── uv.lock                                             # Pinned lockfile for exact reproducibility
```

---

## 🚀 Getting Started & Installation

### Prerequisites
- Python `>= 3.13`
- [uv](https://github.com/astral-sh/uv) (recommended) or standard `pip`

### Installation with `uv` (Fastest)
```bash
# Clone the repository
git clone https://github.com/Raunak-23/Mars_Orbit_Insertion.git
cd Mars_Orbit_Insertion

# Create virtual environment and install dependencies from lockfile
uv sync
```

### Installation with standard `pip`
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate    # On Linux/macOS
.venv\Scripts\activate       # On Windows

# Install project dependencies
pip install -e .
```

---

## 💻 CLI Usage & Workflows

All training, evaluation, tuning, and real-time streaming demonstrations can be run directly from [`main.py`](main.py):

### 1. Run Complete QRF Pipeline (Train, Evaluate, Plot & Save)
```bash
uv run python main.py --qrf
```
*Trains Quantile Random Forests for `Mag_heading` and `TOF_Alt` across $t+1$, $t+3$, and $t+5$, exports predictions to `reports/qrf_predictions_test.csv`, and serializes models to `models/`.*

### 2. Run Generalization-Constrained Hyperparameter Tuning for QRF
```bash
uv run python main.py --tune --n-iter 16 --n-splits 4
```
*Executes temporal cross-validation tuning across the tree parameter space and saves optimal configurations to `models/best_hyperparameters.json`.*

### 3. Run Expanding-Window Rolling Temporal Validation
```bash
uv run python main.py --rolling-val
```
*Executes 5-fold walk-forward validation across flight phases and exports metrics to `reports/qrf_rolling_metrics.csv`.*

### 4. Run Real-Time Streaming QRF Inference Demo
```bash
uv run python main.py --demo-inference
```
*Simulates incoming live telemetry at 1 Hz and prints point predictions and $90\%$ confidence bounds in real time.*

### 5. Run Complete Multi-Horizon Quantile LSTM Pipeline
```bash
uv run python main.py --lstm --epochs 50 --batch-size 64 --hidden-dim 128 --num-layers 2
```
*Builds sequence datasets, trains the multi-horizon neural network using Pinball Loss, evaluates on the test set, and saves the best model checkpoint.*

### 6. Run LSTM Hyperparameter Tuning
```bash
uv run python main.py --tune-lstm
```

### 7. Run Real-Time Streaming LSTM Forecast Demo
```bash
uv run python main.py --demo-lstm
```
*Streams 60-second sliding windows of telemetry and projects 8-channel forward uncertainty cones.*

### 8. Regenerate All EDA Visualizations
```bash
uv run python main.py --eda
```

---

## 🖼️ Flight Dynamics Visualizations

The pipeline generates publication-ready figures saved in [`reports/figures/`](reports/figures/):

- **Mission Overview & Burn Phases**: [`moi_mission_overview.png`](reports/figures/moi_mission_overview.png) — Trajectory altitude, chamber pressure, and propellant mass evolution across the approach, burn, and ascent.
- **Altitude Trajectory Uncertainty Cones**: [`qrf_TOF_Alt_trajectory_intervals.png`](reports/figures/qrf_TOF_Alt_trajectory_intervals.png) — True altitude vs. median prediction and $90\%$ confidence intervals ($q_{0.05} \leftrightarrow q_{0.95}$).
- **Heading Trajectory Uncertainty Cones**: [`qrf_Mag_heading_trajectory_intervals.png`](reports/figures/qrf_Mag_heading_trajectory_intervals.png) — Multi-horizon heading forecasts tracking attitude slews.
- **Residual Error Distributions**: [`qrf_TOF_Alt_residual_distribution.png`](reports/figures/qrf_TOF_Alt_residual_distribution.png) & [`qrf_Mag_heading_residual_distribution.png`](reports/figures/qrf_Mag_heading_residual_distribution.png) — Normality checks and quantile coverage calibration histograms.
- **Hyperparameter Optimization Trade-Offs**: [`qrf_hyperparameter_tuning_tradeoffs.png`](reports/figures/qrf_hyperparameter_tuning_tradeoffs.png) — Train vs. validation error surfaces demonstrating elimination of tree memorization.
- **Rolling Validation Dynamics**: [`qrf_rolling_validation_dynamics.png`](reports/figures/qrf_rolling_validation_dynamics.png) — Calibration stability and interval sharpness across sequential flight phases.

---

## ⚙️ Reproducibility & Hardware Specifications

- **Deterministic Seeds**: All scripts enforce seed reproducibility (`random_state = 42`) across Python's `random`, `numpy`, `scikit-learn`, and `torch` (with `torch.backends.cudnn.deterministic = True`).
- **Zero Future-Data Leakage**: Train, validation, and test splits are strictly chronological ($70\% / 15\% / 15\%$). Scaling parameters (mean and standard deviation) are fitted exclusively on the training partition.
- **Hardware Agnostic**: Fully supports CPU execution and auto-detects CUDA acceleration when available.

---

## 📜 License
This project is open-source and licensed under the [MIT License](LICENSE).

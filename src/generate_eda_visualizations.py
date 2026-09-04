import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 150

data_path = os.path.join("data", "mangalyaan_mars_orbit_insertion_simulated.csv")
df = pd.read_csv(data_path)
df['utc_time'] = pd.to_datetime(df['utc_time'])
df['elapsed_sec'] = (df['utc_time'] - df['utc_time'].iloc[0]).dt.total_seconds()
df['accel_mag'] = np.sqrt(df['accelx']**2 + df['accely']**2 + df['accelz']**2)
df['mag_total'] = np.sqrt(df['Magx']**2 + df['Magy']**2 + df['Magz']**2)

# Rate of change / velocities
df['alt_rate_kms'] = df['TOF_Alt'].diff() / 1.0 # 1 Hz sampling
df['roll_rate'] = df['angle roll'].diff() / 1.0
df['pitch_rate'] = df['pitch'].diff() / 1.0
df['yaw_rate'] = df['yaw'].diff() / 1.0

# Identify flight phases
# Phase 1: Pre-Burn Approach (t: 0 to 973 s)
# Phase 2: LAM Main Engine Firing (t: 974 to 2309 s)
#   - Periapsis occurs at t = 1054 s (inside Phase 2)
# Phase 3: Post-Burn Orbital Ascent (t: 2310 to 3599 s)
phases = []
for t in df['elapsed_sec']:
    if t < 974:
        phases.append('1. Pre-Burn Approach')
    elif t <= 2309:
        phases.append('2. Main Engine Burn (MOI)')
    else:
        phases.append('3. Post-Burn Ascent')
df['flight_phase'] = phases

phase_stats = df.groupby('flight_phase').agg({
    'TOF_Alt': ['min', 'mean', 'max'],
    'accel_mag': ['min', 'mean', 'max'],
    'mag_total': ['min', 'mean', 'max'],
    'pitch': ['mean', 'std'],
    'yaw': ['mean', 'std'],
    'angle roll': ['mean', 'std'],
    'ldr': ['mean', 'std']
})

print("=== FLIGHT PHASE BREAKDOWN ===")
print(phase_stats.to_string())

# Directory for figures
fig_dir = os.path.join("reports", "figures")
os.makedirs(fig_dir, exist_ok=True)
art_fig_dir = r"C:\Users\apara\.gemini\antigravity-ide\brain\ca81dbab-c529-4423-9eb8-ce9b02da3f05"

# 1. Mission Overview Time Series Plot (Altitude, Acceleration, Magnetic Field, Attitude, LDR)
fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)

# 1a. Altitude & Periapsis
axes[0].plot(df['elapsed_sec'], df['TOF_Alt'], color='#1f77b4', lw=2, label='TOF Altitude (km)')
axes[0].axvline(1054, color='#d62728', linestyle='--', lw=1.5, label='Periapsis Passage (421.7 km @ t=1054s)')
axes[0].axvspan(974, 2309, color='#ff7f0e', alpha=0.15, label='MOI Burn Window (974s - 2309s)')
axes[0].set_ylabel('Altitude (km)', fontweight='bold')
axes[0].set_title('Mangalyaan Mars Orbit Insertion (MOI) - Telemetry Profiles Over 1 Hour', fontsize=14, fontweight='bold')
axes[0].legend(loc='upper right')
axes[0].grid(True, alpha=0.3)

# 1b. Acceleration Components and Total Magnitude
axes[1].plot(df['elapsed_sec'], df['accelx'], color='#2ca02c', lw=1.2, alpha=0.8, label='Accel X (Primary Thrust)')
axes[1].plot(df['elapsed_sec'], df['accely'], color='#bcbd22', lw=1.2, alpha=0.8, label='Accel Y (Cross-track)')
axes[1].plot(df['elapsed_sec'], df['accelz'], color='#17becf', lw=1.2, alpha=0.8, label='Accel Z (Normal)')
axes[1].plot(df['elapsed_sec'], df['accel_mag'], color='#d62728', lw=1.8, label='Total Acceleration Magnitude')
axes[1].axvspan(974, 2309, color='#ff7f0e', alpha=0.15)
axes[1].set_ylabel('Accel (m/s²)', fontweight='bold')
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)

# 1c. Magnetic Field Components & Magnitude
axes[2].plot(df['elapsed_sec'], df['Magx'], color='#9467bd', lw=1.2, alpha=0.8, label='Mag X')
axes[2].plot(df['elapsed_sec'], df['Magy'], color='#8c564b', lw=1.2, alpha=0.8, label='Mag Y')
axes[2].plot(df['elapsed_sec'], df['Magz'], color='#e377c2', lw=1.2, alpha=0.8, label='Mag Z')
axes[2].plot(df['elapsed_sec'], df['mag_total'], color='#7f7f7f', lw=1.8, label='Total Field Magnitude')
axes[2].axvspan(974, 2309, color='#ff7f0e', alpha=0.15)
axes[2].set_ylabel('Magnetic Field (a.u.)', fontweight='bold')
axes[2].legend(loc='upper right')
axes[2].grid(True, alpha=0.3)

# 1d. Spacecraft Attitude (Roll, Pitch, Yaw)
axes[3].plot(df['elapsed_sec'], df['yaw'], color='#ff7f0e', lw=1.5, label='Yaw (°)')
axes[3].plot(df['elapsed_sec'], df['pitch'], color='#1f77b4', lw=1.5, label='Pitch (°)')
axes[3].plot(df['elapsed_sec'], df['angle roll'], color='#2ca02c', lw=1.2, label='Roll (°)')
axes[3].axvspan(974, 2309, color='#ff7f0e', alpha=0.15)
axes[3].set_ylabel('Orientation (°)', fontweight='bold')
axes[3].legend(loc='upper right')
axes[3].grid(True, alpha=0.3)

# 1e. Light Dependent Resistor (LDR) / Illumination
axes[4].plot(df['elapsed_sec'], df['ldr'], color='#e6ab02', lw=1.5, label='LDR (Sunlight / Solar Flux)')
axes[4].axvspan(974, 2309, color='#ff7f0e', alpha=0.15)
axes[4].set_xlabel('Elapsed Time (seconds) [2014-09-24 01:30:00 to 02:29:59 UTC]', fontweight='bold')
axes[4].set_ylabel('LDR Intensity', fontweight='bold')
axes[4].legend(loc='upper right')
axes[4].grid(True, alpha=0.3)

plt.tight_layout()
overview_path = os.path.join(fig_dir, "moi_mission_overview.png")
fig.savefig(overview_path)
fig.savefig(os.path.join(art_fig_dir, "moi_mission_overview.png"))
plt.close(fig)
print(f"Saved: {overview_path}")

# 2. Correlation Heatmaps (Pearson & Spearman)
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

cols_for_corr = ['TOF_Alt', 'Magx', 'Magy', 'Magz', 'Mag_heading', 'accelx', 'accely', 'accelz', 'angle roll', 'pitch', 'yaw', 'ldr', 'accel_mag', 'mag_total']
p_corr = df[cols_for_corr].corr(method='pearson')
s_corr = df[cols_for_corr].corr(method='spearman')

sns.heatmap(p_corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, ax=axes[0], cbar_kws={'label': 'Correlation Coefficient'})
axes[0].set_title('Pearson Linear Correlation Matrix', fontsize=12, fontweight='bold')

sns.heatmap(s_corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, ax=axes[1], cbar_kws={'label': 'Rank Correlation Coefficient'})
axes[1].set_title('Spearman Monotonic Rank Correlation Matrix', fontsize=12, fontweight='bold')

plt.tight_layout()
corr_path = os.path.join(fig_dir, "correlation_heatmaps.png")
fig.savefig(corr_path)
fig.savefig(os.path.join(art_fig_dir, "correlation_heatmaps.png"))
plt.close(fig)
print(f"Saved: {corr_path}")

# 3. Altitude vs Magnetic Field & Altitude vs Acceleration
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 3a. TOF_Alt vs mag_total
axes[0].scatter(df['TOF_Alt'], df['mag_total'], c=df['elapsed_sec'], cmap='viridis', alpha=0.6, s=12)
axes[0].set_xlabel('TOF Altitude (km)', fontweight='bold')
axes[0].set_ylabel('Total Magnetic Field (a.u.)', fontweight='bold')
axes[0].set_title('Inverse Proximity Law: Magnetic Field vs Altitude\n(Color = Mission Elapsed Seconds)', fontsize=11, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# 3b. TOF_Alt vs accel_mag
axes[1].scatter(df['TOF_Alt'], df['accel_mag'], c=df['elapsed_sec'], cmap='plasma', alpha=0.6, s=12)
axes[1].set_xlabel('TOF Altitude (km)', fontweight='bold')
axes[1].set_ylabel('Total Acceleration (m/s²)', fontweight='bold')
axes[1].set_title('Engine Burn Execution vs Altitude\n(Main Braking at Low Altitude)', fontsize=11, fontweight='bold')
axes[1].grid(True, alpha=0.3)

# 3c. 3D Trajectory in Attitude Space (Roll, Pitch, Yaw)
scatter = axes[2].scatter(df['pitch'], df['yaw'], c=df['accel_mag'], cmap='hot', alpha=0.6, s=14)
axes[2].set_xlabel('Pitch Angle (°)', fontweight='bold')
axes[2].set_ylabel('Yaw Angle (°)', fontweight='bold')
axes[2].set_title('Attitude Slewing Under Thrust\n(Color = Acceleration Magnitude)', fontsize=11, fontweight='bold')
cbar = plt.colorbar(scatter, ax=axes[2])
cbar.set_label('Acceleration Magnitude (m/s²)')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
physics_path = os.path.join(fig_dir, "orbital_physics_relationships.png")
fig.savefig(physics_path)
fig.savefig(os.path.join(art_fig_dir, "orbital_physics_relationships.png"))
plt.close(fig)
print(f"Saved: {physics_path}")

# 4. Feature Distributions & Phase Boxplots
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

sns.boxplot(data=df, x='flight_phase', y='accel_mag', palette='Set2', ax=axes[0, 0])
axes[0, 0].set_title('Acceleration by Flight Phase', fontweight='bold')
axes[0, 0].set_ylabel('Accel Mag (m/s²)')
axes[0, 0].tick_params(axis='x', rotation=15)

sns.boxplot(data=df, x='flight_phase', y='mag_total', palette='Set2', ax=axes[0, 1])
axes[0, 1].set_title('Magnetic Field by Flight Phase', fontweight='bold')
axes[0, 1].set_ylabel('Mag Total (a.u.)')
axes[0, 1].tick_params(axis='x', rotation=15)

sns.boxplot(data=df, x='flight_phase', y='pitch', palette='Set2', ax=axes[0, 2])
axes[0, 2].set_title('Pitch Angle Distribution by Phase', fontweight='bold')
axes[0, 2].set_ylabel('Pitch (°)')
axes[0, 2].tick_params(axis='x', rotation=15)

sns.histplot(data=df, x='TOF_Alt', bins=50, kde=True, color='#1f77b4', ax=axes[1, 0])
axes[1, 0].set_title('Altitude Distribution (Heavy Right-Tail)', fontweight='bold')
axes[1, 0].set_xlabel('Altitude (km)')

sns.histplot(data=df, x='ldr', bins=40, kde=True, color='#e6ab02', ax=axes[1, 1])
axes[1, 1].set_title('LDR Sensor Distribution', fontweight='bold')
axes[1, 1].set_xlabel('LDR Value')

sns.scatterplot(data=df, x='Mag_heading', y='yaw', hue='flight_phase', alpha=0.5, s=15, ax=axes[1, 2])
axes[1, 2].set_title('Magnetic Heading vs Yaw Angle', fontweight='bold')
axes[1, 2].set_xlabel('Magnetic Heading (°)')
axes[1, 2].set_ylabel('Yaw (°)')

plt.tight_layout()
dist_path = os.path.join(fig_dir, "phase_distributions_boxplots.png")
fig.savefig(dist_path)
fig.savefig(os.path.join(art_fig_dir, "phase_distributions_boxplots.png"))
plt.close(fig)
print(f"Saved: {dist_path}")

print("All figures successfully created and saved!")

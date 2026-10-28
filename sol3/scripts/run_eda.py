"""
Comprehensive Exploratory Data Analysis & Spectral Decomposition for Azure VM Workload Trace.
Produces workload characterization, autocorrelation analysis, spectral power distributions,
and diagnostic figures saved to sol3/reports/figures/.
"""

import os
import gzip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf

def run():
    os.makedirs("sol3/reports/figures", exist_ok=True)
    print("Loading data from challenge_data...")
    readings_path = "challenge_data/vm_cpu_readings.csv.gz"
    metadata_path = "challenge_data/vm_metadata.csv.gz"

    df_cpu = pd.read_csv(readings_path)
    df_meta = pd.read_csv(metadata_path)

    print(f"CPU Readings shape: {df_cpu.shape}")
    print(f"Metadata shape: {df_meta.shape}")

    # 1. Validation & Integrity
    n_vms_cpu = df_cpu["vm_id"].nunique()
    n_vms_meta = df_meta["vm_id"].nunique()
    timestamps = np.sort(df_cpu["timestamp"].unique())
    n_timestamps = len(timestamps)
    diffs = np.diff(timestamps)
    
    print(f"Unique VMs in CPU data: {n_vms_cpu}")
    print(f"Unique VMs in Metadata: {n_vms_meta}")
    print(f"Number of distinct timestamps: {n_timestamps}")
    print(f"Timestamp cadence: min diff={diffs.min()}s, max diff={diffs.max()}s (expected 300s)")
    print(f"Total time span: {timestamps[-1] - timestamps[0]}s = {(timestamps[-1] - timestamps[0])/3600:.2f} hours")

    # Check nulls
    print(f"Missing values in CPU readings: {df_cpu.isna().sum().to_dict()}")
    print(f"Missing values in Metadata: {df_meta.isna().sum().to_dict()}")

    # 2. Aggregations (Full Target Series)
    # Active VM count per timestamp
    active_vm = df_cpu.groupby("timestamp")["vm_id"].nunique().rename("active_vm_count")
    # Aggregate CPU utilization: mean of VM avg_cpu
    agg_cpu = df_cpu.groupby("timestamp")["avg_cpu"].mean().rename("aggregate_cpu_utilization")
    
    # Also compute p50, p90, p95, max cpu across VMs at each timestamp
    cpu_p50 = df_cpu.groupby("timestamp")["avg_cpu"].median().rename("p50_cpu")
    cpu_p90 = df_cpu.groupby("timestamp")["avg_cpu"].quantile(0.90).rename("p90_cpu")
    cpu_p95 = df_cpu.groupby("timestamp")["avg_cpu"].quantile(0.95).rename("p95_cpu")

    df_targets = pd.concat([active_vm, agg_cpu, cpu_p50, cpu_p90, cpu_p95], axis=1).reset_index()
    df_targets["step"] = np.arange(1, len(df_targets) + 1)

    print("\n--- Summary of Target Series ---")
    print(f"Active VM count: min={active_vm.min()}, max={active_vm.max()}, mean={active_vm.mean():.1f}")
    print(f"Aggregate CPU: min={agg_cpu.min():.2f}%, max={agg_cpu.max():.2f}%, mean={agg_cpu.mean():.2f}%, std={agg_cpu.std():.2f}%")
    print(f"Training window (first 110 steps): CPU mean={agg_cpu.iloc[:110].mean():.2f}%, std={agg_cpu.iloc[:110].std():.2f}%")
    print(f"Holdout window (last 24 steps): CPU mean={agg_cpu.iloc[110:].mean():.2f}%, std={agg_cpu.iloc[110:].std():.2f}%")

    # 3. Category & Tier Breakdown
    df_merged = df_cpu.merge(df_meta[["vm_id", "vm_category", "virtual_core_bucket", "memory_gb_bucket"]], on="vm_id", how="left")
    cat_summary = df_merged.groupby(["timestamp", "vm_category"])["avg_cpu"].mean().unstack()

    # 4. Spectral & Autocorrelation Analysis on aggregate_cpu_utilization (Training only: first 110 steps)
    train_cpu = agg_cpu.iloc[:110].to_numpy()
    acf_vals = acf(train_cpu, nlags=35)
    pacf_vals = pacf(train_cpu, nlags=35)

    # FFT
    detrended = train_cpu - np.mean(train_cpu)
    fft_vals = np.fft.rfft(detrended)
    fft_freqs = np.fft.rfftfreq(len(detrended), d=300) # in Hz (1/s)
    fft_periods_min = (1.0 / (fft_freqs[1:] + 1e-12)) / 60.0 # period in minutes
    fft_power = np.abs(fft_vals[1:]) ** 2

    top_freq_indices = np.argsort(fft_power)[::-1][:5]
    print("\n--- FFT Spectral Analysis (Top 5 Periods in Training Set) ---")
    for idx in top_freq_indices:
        print(f"Period: {fft_periods_min[idx]:.1f} min | Power: {fft_power[idx]:.1f}")

    print("\n--- Autocorrelation (ACF Lags) ---")
    for lag in [1, 2, 3, 6, 12, 24]:
        print(f"Lag {lag} ({(lag*5)} min): ACF = {acf_vals[lag]:.4f}, PACF = {pacf_vals[lag]:.4f}")

    # 5. Diagnostic Figures
    # Figure 1: Full Target Overview with Train/Holdout boundary
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    
    ax1.plot(df_targets["step"], df_targets["aggregate_cpu_utilization"], color="#1f77b4", lw=2, label="Aggregate CPU Mean")
    ax1.plot(df_targets["step"], df_targets["p95_cpu"], color="#ff7f0e", lw=1.5, ls="--", label="P95 CPU (Tail Load)")
    ax1.axvline(110.5, color="red", linestyle=":", lw=2, label="Train / Holdout Boundary (Step 110)")
    ax1.set_ylabel("CPU Utilization (%)", fontsize=11)
    ax1.set_title("Fleet Aggregate CPU Utilization & Tail Load Over Time (5-min cadence)", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, alpha=0.3)

    ax2.plot(df_targets["step"], df_targets["active_vm_count"], color="#2ca02c", lw=2, label="Active VM Count")
    ax2.axvline(110.5, color="red", linestyle=":", lw=2, label="Train / Holdout Boundary")
    ax2.set_xlabel("Observation Step (5-min intervals, ~11.1 hrs total)", fontsize=11)
    ax2.set_ylabel("Active VM Count", fontsize=11)
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig("sol3/reports/figures/eda_targets_overview.png", dpi=200)
    plt.close()

    # Figure 2: Workload Segmentation (Interactive vs Delay-Insensitive)
    fig, ax = plt.subplots(figsize=(11, 5))
    for col in cat_summary.columns:
        ax.plot(np.arange(1, len(cat_summary) + 1), cat_summary[col], lw=2, label=f"Category: {col}")
    ax.axvline(110.5, color="red", linestyle=":", lw=2, label="Train / Holdout Split")
    ax.set_title("Mean CPU Utilization Segmented by Azure Workload Category", fontsize=12, fontweight="bold")
    ax.set_xlabel("Observation Step (5-min intervals)", fontsize=11)
    ax.set_ylabel("Mean CPU Utilization (%)", fontsize=11)
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig("sol3/reports/figures/eda_workload_categories.png", dpi=200)
    plt.close()

    # Figure 3: ACF, PACF, and FFT Power Spectrum
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))
    
    ax1.stem(range(len(acf_vals)), acf_vals)
    ax1.set_title("Autocorrelation Function (ACF)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Lag (5-min steps)")
    ax1.set_ylabel("ACF")
    ax1.axhline(0, color="gray", lw=0.8)
    ax1.grid(True, alpha=0.3)

    ax2.stem(range(len(pacf_vals)), pacf_vals)
    ax2.set_title("Partial Autocorrelation (PACF)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Lag (5-min steps)")
    ax2.set_ylabel("PACF")
    ax2.axhline(0, color="gray", lw=0.8)
    ax2.grid(True, alpha=0.3)

    ax3.plot(fft_periods_min, fft_power, color="#d62728", lw=2)
    ax3.set_title("FFT Power Spectrum vs Period (min)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Period (minutes)")
    ax3.set_ylabel("Spectral Power")
    ax3.set_xlim(10, 300)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig("sol3/reports/figures/eda_spectral_acf.png", dpi=200)
    plt.close()

    print("\nEDA and spectral figures saved successfully to sol3/reports/figures/.")

if __name__ == "__main__":
    run()


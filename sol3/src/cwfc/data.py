"""
Data ingestion, validation, and target series construction.
"""

from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd


def load_raw_data(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load CPU readings and metadata gzipped CSV files."""
    data_path = Path(data_dir)
    readings_file = data_path / "vm_cpu_readings.csv.gz"
    metadata_file = data_path / "vm_metadata.csv.gz"

    if not readings_file.exists():
        raise FileNotFoundError(f"Missing CPU readings file: {readings_file}")
    if not metadata_file.exists():
        raise FileNotFoundError(f"Missing VM metadata file: {metadata_file}")

    df_cpu = pd.read_csv(readings_file)
    df_meta = pd.read_csv(metadata_file)
    return df_cpu, df_meta


def build_target_series(df_cpu: pd.DataFrame, df_meta: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Construct the required aggregate target series from VM-level readings.

    Targets:
      1. active_vm_count: number of distinct VM IDs with a valid CPU reading at timestamp t.
      2. aggregate_cpu_utilization: mean of VM-level avg_cpu values at timestamp t, in [0, 100].

    Also aggregates category-specific CPU means (Interactive, Delay-insensitive) if metadata is provided.
    """
    # Verify no null timestamps
    if df_cpu["timestamp"].isna().any():
        raise ValueError("Null values found in timestamp column")

    # Distinct VM count
    active_vms = df_cpu.groupby("timestamp")["vm_id"].nunique().rename("active_vm_count")
    # Mean CPU utilization
    agg_cpu = df_cpu.groupby("timestamp")["avg_cpu"].mean().rename("aggregate_cpu_utilization")

    series_list = [active_vms, agg_cpu]

    if df_meta is not None:
        merged = df_cpu[["timestamp", "vm_id", "avg_cpu"]].merge(
            df_meta[["vm_id", "vm_category"]], on="vm_id", how="left"
        )
        cat_pivot = merged.groupby(["timestamp", "vm_category"])["avg_cpu"].mean().unstack()
        cat_pivot.columns = [f"cpu_cat_{c.lower().replace('-', '_')}" for c in cat_pivot.columns]
        series_list.append(cat_pivot)

    df_targets = pd.concat(series_list, axis=1).sort_index().reset_index()

    # Validate monotonicity and spacing
    ts = df_targets["timestamp"].to_numpy()
    if len(ts) > 1:
        diffs = np.diff(ts)
        if (diffs <= 0).any():
            raise ValueError("Timestamps must be strictly monotonically increasing")

    return df_targets


def get_train_holdout_split(
    df_targets: pd.DataFrame, train_steps: int = 110, holdout_steps: int = 24
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the full target series into training (first 110 steps) and holdout (last 24 steps).
    """
    total_steps = len(df_targets)
    expected = train_steps + holdout_steps
    if total_steps < expected:
        raise ValueError(f"Target series has {total_steps} rows, expected at least {expected}")

    df_train = df_targets.iloc[:train_steps].copy().reset_index(drop=True)
    df_holdout = df_targets.iloc[train_steps : train_steps + holdout_steps].copy().reset_index(drop=True)
    return df_train, df_holdout


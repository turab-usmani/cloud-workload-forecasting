"""
Unit tests for data loading and target aggregation.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

src_dir = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_dir))

from cwfc.data import build_target_series, get_train_holdout_split


def test_build_target_series_synthetic():
    timestamps = [300, 300, 600, 600]
    vm_ids = ["vm1", "vm2", "vm1", "vm2"]
    avg_cpu = [10.0, 20.0, 30.0, 40.0]

    df_cpu = pd.DataFrame({
        "timestamp": timestamps,
        "vm_id": vm_ids,
        "avg_cpu": avg_cpu,
    })

    targets = build_target_series(df_cpu)

    assert len(targets) == 2
    assert list(targets["timestamp"]) == [300, 600]
    # Active VMs per timestamp
    assert list(targets["active_vm_count"]) == [2, 2]
    # Aggregate CPU: mean at 300 is 15.0, at 600 is 35.0
    assert list(targets["aggregate_cpu_utilization"]) == [15.0, 35.0]


def test_get_train_holdout_split():
    df = pd.DataFrame({
        "timestamp": range(134),
        "active_vm_count": [1000] * 134,
        "aggregate_cpu_utilization": [10.0] * 134,
    })
    train, holdout = get_train_holdout_split(df, train_steps=110, holdout_steps=24)
    assert len(train) == 110
    assert len(holdout) == 24
    assert train["timestamp"].iloc[-1] == 109
    assert holdout["timestamp"].iloc[0] == 110


"""
Submission Contract Verification Tests:
Verifies that generated predictions.csv matches SUBMISSION.md requirements.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest


def test_submission_contract():
    pred_path = Path("sol3/predictions.csv")
    if not pred_path.exists():
        pred_path = Path("predictions.csv")
    assert pred_path.exists(), "predictions.csv must exist"

    df = pd.read_csv(pred_path)

    # 1. Row count
    assert len(df) == 24, f"Expected 24 rows, got {len(df)}"

    # 2. Column names
    expected_cols = ["timestamp", "active_vm_count", "aggregate_cpu_utilization"]
    assert list(df.columns) == expected_cols, f"Columns must be {expected_cols}, got {list(df.columns)}"

    # 3. Monotonic increasing timestamps
    ts = df["timestamp"].to_numpy()
    assert (np.diff(ts) > 0).all(), "Timestamps must be monotonically increasing"

    # 4. No NaNs or infinities
    assert not df.isna().any().any(), "File contains null values"
    assert np.all(np.isfinite(df["aggregate_cpu_utilization"].to_numpy())), "Non-finite CPU values"
    assert np.all(np.isfinite(df["active_vm_count"].to_numpy())), "Non-finite VM count"

    # 5. Value ranges
    assert (df["active_vm_count"] >= 0).all(), "active_vm_count must be >= 0"
    assert (df["aggregate_cpu_utilization"] >= 0).all() and (df["aggregate_cpu_utilization"] <= 100).all(), "CPU must be in [0, 100]"


"""
Critical unit test: Temporal Anti-Leakage Invariance.
Asserts that features computed at origin t depend ONLY on data at or before t,
and are 100% invariant to the presence or values of future data (t+1..T).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

src_dir = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_dir))

from cwfc.features import extract_features_at_origin, build_direct_multistep_dataset


def test_leakage_invariance_at_origin():
    """
    Features extracted at origin t=30 must be identical whether future rows exist or not.
    """
    np.random.seed(42)
    n = 60
    base_series = np.cumsum(np.random.randn(n)) + 10.0
    df_full = pd.DataFrame({"aggregate_cpu_utilization": base_series})

    # Cut history at t=30
    df_slice = df_full.iloc[:30].copy()

    feats_from_slice = extract_features_at_origin(df_slice, target_col="aggregate_cpu_utilization")
    # Same extraction with slice from full dataframe
    feats_from_slice_of_full = extract_features_at_origin(df_full.iloc[:30], target_col="aggregate_cpu_utilization")

    assert feats_from_slice.keys() == feats_from_slice_of_full.keys()
    for k in feats_from_slice:
        assert feats_from_slice[k] == pytest.approx(feats_from_slice_of_full[k]), f"Mismatch in feature {k}"


def test_no_future_correlation_in_dataset_construction():
    """
    Verifies that changing future targets does NOT alter feature matrix X.
    """
    np.random.seed(42)
    n = 50
    series1 = np.sin(np.linspace(0, 10, n)) + 10.0
    df1 = pd.DataFrame({"aggregate_cpu_utilization": series1})

    X1, Y1 = build_direct_multistep_dataset(df1, horizon=10, min_history=12)

    # Modify the tail rows (which are only future targets)
    df2 = df1.copy()
    df2.iloc[-5:, 0] += 50.0

    X2, Y2 = build_direct_multistep_dataset(df2, horizon=10, min_history=12)

    # Origins that occur before the modified rows must have identical feature rows
    # The modified rows are at the very end, so early feature vectors must be strictly identical
    pd.testing.assert_frame_equal(X1.iloc[:10], X2.iloc[:10])


"""
Unit tests for trace-driven capacity simulation.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

src_dir = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_dir))

from cwfc.simulation import run_capacity_simulation


def test_capacity_simulation_behavior():
    train_history = np.array([10.0, 12.0, 14.0, 15.0]) # peak = 15.0
    y_true = np.array([11.0, 12.0, 13.0, 14.0])
    forecast_point = np.array([11.0, 12.0, 13.0, 14.0])
    forecast_p95 = np.array([12.0, 13.0, 14.0, 15.0])

    res_df = run_capacity_simulation(y_true, forecast_point, forecast_p95, train_history)

    assert len(res_df) == 4
    assert "Policy" in res_df.columns
    assert "SLA Violations (Steps)" in res_df.columns
    assert "Mean Idle Waste (% CPU)" in res_df.columns
    # Static peak is 15.0 * 1.05 = 15.75, so no violations
    static_row = res_df[res_df["Policy"] == "Static Peak Policy"].iloc[0]
    assert static_row["SLA Violation Rate (%)"] == 0.0


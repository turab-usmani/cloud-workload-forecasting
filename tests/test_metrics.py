"""
Unit tests for forecasting error and probabilistic metrics.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

# Ensure sol3/src is on path
src_dir = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_dir))

from cwfc.metrics import mae, rmse, smape, normalized_mae, pinball_loss, coverage, sharpness


def test_mae_rmse_exact():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])
    # abs errs: 2, 2, 3 -> mean = 7/3 = 2.3333333333
    assert mae(y_true, y_pred) == pytest.approx(7.0 / 3.0)
    # sq errs: 4, 4, 9 -> mean = 17/3 -> sqrt = 2.380476
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(17.0 / 3.0))


def test_smape_bounds():
    y_true = np.array([10.0, 20.0])
    y_pred = np.array([10.0, 20.0])
    assert smape(y_true, y_pred) == pytest.approx(0.0)


def test_normalized_mae():
    y_true = np.array([10.0, 20.0])
    y_pred = np.array([11.0, 19.0])       # MAE = 1.0
    baseline = np.array([12.0, 18.0])     # MAE = 2.0
    assert normalized_mae(y_true, y_pred, baseline) == pytest.approx(0.5)


def test_pinball_loss():
    y_true = np.array([10.0])
    y_pred = np.array([8.0]) # under-prediction: diff = 2.0
    # For alpha=0.9, loss = 0.9 * 2.0 = 1.8
    assert pinball_loss(y_true, y_pred, alpha=0.9) == pytest.approx(1.8)
    # For alpha=0.1, loss = 0.1 * 2.0 = 0.2
    assert pinball_loss(y_true, y_pred, alpha=0.1) == pytest.approx(0.2)


def test_coverage_and_sharpness():
    y_true = np.array([10.0, 15.0, 20.0, 25.0])
    lb = np.array([9.0, 14.0, 18.0, 26.0]) # 10, 15, 20 inside; 25 outside (25 < 26)
    ub = np.array([11.0, 16.0, 22.0, 30.0])
    assert coverage(y_true, lb, ub) == pytest.approx(75.0)
    assert sharpness(lb, ub) == pytest.approx(3.0)

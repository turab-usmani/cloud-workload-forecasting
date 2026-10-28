"""
Unit tests for forecasting models (Baselines, AR, LightGBM, Ensemble).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

src_dir = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_dir))

from cwfc.baselines import PersistenceForecaster, MovingAverageForecaster, HoltLinearTrendForecaster
from cwfc.statistical import AutoRegressiveForecaster
from cwfc.models import DirectMultiStepLightGBM, DynamicEnsembleForecaster


def test_persistence_forecaster():
    model = PersistenceForecaster()
    model.fit([12.0, 15.0, 18.0])
    preds = model.predict(horizon=24)
    assert len(preds) == 24
    assert np.all(preds == 18.0)


def test_moving_average_forecaster():
    model = MovingAverageForecaster(window=3)
    model.fit([10.0, 20.0, 30.0, 40.0]) # last 3: 20, 30, 40 -> mean = 30.0
    preds = model.predict(horizon=24)
    assert len(preds) == 24
    assert np.all(preds == 30.0)


def test_holt_linear_trend_forecaster():
    model = HoltLinearTrendForecaster(alpha=0.5, beta=0.5, damped=False)
    # Linearly increasing series: 1, 2, 3, 4, 5
    model.fit(np.arange(1, 10, dtype=float))
    preds = model.predict(horizon=5)
    assert len(preds) == 5
    # Predictions should continue the upward trend
    assert preds[-1] > preds[0]


def test_autoregressive_forecaster():
    np.random.seed(42)
    series = np.cumsum(np.random.randn(50)) + 20.0
    model = AutoRegressiveForecaster(lags=3)
    model.fit(series)
    preds = model.predict(horizon=24)
    assert len(preds) == 24
    assert np.all(np.isfinite(preds))


def test_direct_multistep_lightgbm():
    np.random.seed(42)
    n = 60
    series = np.sin(np.linspace(0, 10, n)) + 15.0
    df = pd.DataFrame({"aggregate_cpu_utilization": series})

    model = DirectMultiStepLightGBM(horizon=12, objective="regression_l1")
    model.fit(df, target_col="aggregate_cpu_utilization")
    preds = model.predict(df, target_col="aggregate_cpu_utilization")
    assert len(preds) == 12
    assert np.all(np.isfinite(preds))


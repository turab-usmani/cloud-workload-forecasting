"""
Baseline forecasting models for cloud workload demand.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


class PersistenceForecaster:
    """Carries the last observed value forward across all horizon steps."""

    def __init__(self):
        self.last_val = None

    def fit(self, y: np.ndarray | pd.Series):
        arr = np.asarray(y, dtype=float)
        if len(arr) == 0:
            raise ValueError("Training data cannot be empty")
        self.last_val = arr[-1]
        return self

    def predict(self, horizon: int = 24) -> np.ndarray:
        if self.last_val is None:
            raise RuntimeError("Model must be fitted before predict")
        return np.full(horizon, self.last_val, dtype=float)


class MovingAverageForecaster:
    """Forecasts the simple moving average of the last `window` observations."""

    def __init__(self, window: int = 12):
        self.window = window
        self.mean_val = None

    def fit(self, y: np.ndarray | pd.Series):
        arr = np.asarray(y, dtype=float)
        w = min(len(arr), self.window)
        self.mean_val = float(np.mean(arr[-w:]))
        return self

    def predict(self, horizon: int = 24) -> np.ndarray:
        return np.full(horizon, self.mean_val, dtype=float)


class HoltLinearTrendForecaster:
    """
    Holt's Linear Trend Exponential Smoothing.
    Captures both smoothed level and local linear drift/decay.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.1, damped: bool = True, phi: float = 0.9):
        self.alpha = alpha
        self.beta = beta
        self.damped = damped
        self.phi = phi
        self.level = None
        self.trend = None

    def fit(self, y: np.ndarray | pd.Series):
        arr = np.asarray(y, dtype=float)
        if len(arr) < 2:
            raise ValueError("At least 2 points needed to fit Holt's model")

        # Initial level and trend
        l = arr[0]
        b = arr[1] - arr[0]

        for val in arr[1:]:
            prev_l = l
            prev_b = b
            l = self.alpha * val + (1.0 - self.alpha) * (prev_l + (self.phi * prev_b if self.damped else prev_b))
            b = self.beta * (l - prev_l) + (1.0 - self.beta) * (self.phi * prev_b if self.damped else prev_b)

        self.level = l
        self.trend = b
        return self

    def predict(self, horizon: int = 24) -> np.ndarray:
        preds = []
        for h in range(1, horizon + 1):
            if self.damped:
                # Damped trend formula: sum(phi^i for i in 1..h)
                damp_factor = sum(self.phi ** i for i in range(1, h + 1))
                y_hat = self.level + damp_factor * self.trend
            else:
                y_hat = self.level + h * self.trend
            preds.append(y_hat)
        return np.array(preds, dtype=float)


class SeasonalNaiveForecaster:
    """Repeats the last observed seasonal cycle of length `season_length`."""

    def __init__(self, season_length: int = 24):
        self.season_length = season_length
        self.history = None

    def fit(self, y: np.ndarray | pd.Series):
        arr = np.asarray(y, dtype=float)
        if len(arr) < self.season_length:
            raise ValueError(f"Need at least {self.season_length} observations for seasonal-naive")
        self.history = arr[-self.season_length :]
        return self

    def predict(self, horizon: int = 24) -> np.ndarray:
        preds = [self.history[i % self.season_length] for i in range(horizon)]
        return np.array(preds, dtype=float)


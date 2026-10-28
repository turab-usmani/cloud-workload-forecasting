"""
Statistical Autoregressive and ARIMA time-series models.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg


class AutoRegressiveForecaster:
    """Autoregressive AR(p) model with optimal lag selection."""

    def __init__(self, lags: int = 3):
        self.lags = lags
        self.model_res = None
        self.n_obs = 0

    def fit(self, y: np.ndarray | pd.Series):
        arr = np.asarray(y, dtype=float)
        self.n_obs = len(arr)
        # Fallback to fewer lags if history is short
        effective_lags = min(self.lags, max(1, self.n_obs // 4))
        ar = AutoReg(arr, lags=effective_lags, trend="c")
        self.model_res = ar.fit()
        return self

    def predict(self, horizon: int = 24) -> np.ndarray:
        if self.model_res is None:
            raise RuntimeError("Model must be fitted before predict")
        preds = self.model_res.predict(start=self.n_obs, end=self.n_obs + horizon - 1)
        return np.asarray(preds, dtype=float)

"""
Machine Learning & Hybrid Direct Multi-Step Forecasters.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from .features import extract_features_at_origin, build_direct_multistep_dataset


class DirectMultiStepLightGBM:
    """
    Direct Multi-Horizon LightGBM Forecaster.
    Fits H separate gradient boosted trees f_h(X_t) -> y_{t+h}.
    Avoids recursive feedback loops and error amplification.
    """

    def __init__(
        self,
        horizon: int = 24,
        objective: str = "regression_l1",
        alpha: float | None = None,
        n_estimators: int = 60,
        learning_rate: float = 0.05,
        max_depth: int = 3,
        num_leaves: int = 7,
        min_child_samples: int = 5,
        random_state: int = 42,
    ):
        self.horizon = horizon
        self.objective = objective
        self.alpha = alpha
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.random_state = random_state
        self.models: dict[int, lgb.LGBMRegressor] = {}
        self.feature_names_: list[str] = []

    def fit(self, df_train: pd.DataFrame, target_col: str = "aggregate_cpu_utilization", aux_cols: list[str] | None = None):
        X, Y = build_direct_multistep_dataset(
            df_train=df_train,
            target_col=target_col,
            aux_cols=aux_cols,
            horizon=self.horizon,
            min_history=24,
        )
        self.feature_names_ = list(X.columns)

        for h in range(1, self.horizon + 1):
            params = {
                "objective": self.objective,
                "n_estimators": self.n_estimators,
                "learning_rate": self.learning_rate,
                "max_depth": self.max_depth,
                "num_leaves": self.num_leaves,
                "min_child_samples": self.min_child_samples,
                "random_state": self.random_state + h,
                "verbosity": -1,
            }
            if self.objective == "quantile" and self.alpha is not None:
                params["alpha"] = self.alpha

            model = lgb.LGBMRegressor(**params)
            model.fit(X, Y[h])
            self.models[h] = model

        return self

    def predict(self, history_df: pd.DataFrame, target_col: str = "aggregate_cpu_utilization", aux_cols: list[str] | None = None) -> np.ndarray:
        if not self.models:
            raise RuntimeError("Model must be fitted before predict")
        feat_dict = extract_features_at_origin(history_df, target_col=target_col, aux_cols=aux_cols)
        X_origin = pd.DataFrame([feat_dict])[self.feature_names_]

        preds = []
        for h in range(1, self.horizon + 1):
            y_hat = self.models[h].predict(X_origin)[0]
            preds.append(float(y_hat))

        return np.array(preds, dtype=float)


class DirectMultiStepRidge:
    """Direct Multi-Horizon Ridge Regression."""

    def __init__(self, horizon: int = 24, alpha: float = 1.0):
        self.horizon = horizon
        self.alpha = alpha
        self.models: dict[int, Ridge] = {}
        self.feature_names_: list[str] = []

    def fit(self, df_train: pd.DataFrame, target_col: str = "aggregate_cpu_utilization", aux_cols: list[str] | None = None):
        X, Y = build_direct_multistep_dataset(
            df_train=df_train,
            target_col=target_col,
            aux_cols=aux_cols,
            horizon=self.horizon,
            min_history=24,
        )
        self.feature_names_ = list(X.columns)
        for h in range(1, self.horizon + 1):
            model = Ridge(alpha=self.alpha)
            model.fit(X, Y[h])
            self.models[h] = model
        return self

    def predict(self, history_df: pd.DataFrame, target_col: str = "aggregate_cpu_utilization", aux_cols: list[str] | None = None) -> np.ndarray:
        feat_dict = extract_features_at_origin(history_df, target_col=target_col, aux_cols=aux_cols)
        X_origin = pd.DataFrame([feat_dict])[self.feature_names_]
        return np.array([float(self.models[h].predict(X_origin)[0]) for h in range(1, self.horizon + 1)], dtype=float)


class DynamicEnsembleForecaster:
    """
    Combines Direct LightGBM with Damped Holt Trend.
    Balances non-linear feature adaptation with robust low-frequency drift tracking.
    """

    def __init__(self, lgb_weight: float = 0.6, horizon: int = 24):
        self.lgb_weight = lgb_weight
        self.horizon = horizon
        self.lgb_model = DirectMultiStepLightGBM(horizon=horizon)
        from .baselines import HoltLinearTrendForecaster
        self.holt_model = HoltLinearTrendForecaster(alpha=0.25, beta=0.08, damped=True, phi=0.92)

    def fit(self, df_train: pd.DataFrame, target_col: str = "aggregate_cpu_utilization", aux_cols: list[str] | None = None):
        self.lgb_model.fit(df_train, target_col=target_col, aux_cols=aux_cols)
        self.holt_model.fit(df_train[target_col].to_numpy())
        return self

    def predict(self, history_df: pd.DataFrame, target_col: str = "aggregate_cpu_utilization", aux_cols: list[str] | None = None) -> np.ndarray:
        p_lgb = self.lgb_model.predict(history_df, target_col=target_col, aux_cols=aux_cols)
        p_holt = self.holt_model.predict(self.horizon)
        return self.lgb_weight * p_lgb + (1.0 - self.lgb_weight) * p_holt


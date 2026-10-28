"""
Evaluation metrics for point and probabilistic cloud workload forecasting.
"""

from __future__ import annotations
import numpy as np


def mae(y_true, y_pred) -> float:
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if a.shape != b.shape:
        raise ValueError("y_true and y_pred must have identical shapes")
    return float(np.mean(np.abs(a - b)))


def rmse(y_true, y_pred) -> float:
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if a.shape != b.shape:
        raise ValueError("y_true and y_pred must have identical shapes")
    return float(np.sqrt(np.mean((a - b) ** 2)))


def smape(y_true, y_pred, eps: float = 1e-8) -> float:
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if a.shape != b.shape:
        raise ValueError("y_true and y_pred must have identical shapes")
    denom = np.maximum(np.abs(a) + np.abs(b), eps)
    return float(np.mean(2.0 * np.abs(a - b) / denom) * 100.0)


def normalized_mae(y_true, y_pred, baseline_pred) -> float:
    model_mae = mae(y_true, y_pred)
    base_mae = mae(y_true, baseline_pred)
    if base_mae == 0:
        return 0.0 if model_mae == 0 else float("inf")
    return float(model_mae / base_mae)


def pinball_loss(y_true, y_pred, alpha: float) -> float:
    """Asymmetric quantile (pinball) loss for quantile alpha in (0, 1)."""
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    diff = a - b
    loss = np.maximum(alpha * diff, (alpha - 1.0) * diff)
    return float(np.mean(loss))


def coverage(y_true, lower_bound, upper_bound) -> float:
    """Fraction of true values falling inside [lower_bound, upper_bound]."""
    a = np.asarray(y_true, dtype=float)
    lb = np.asarray(lower_bound, dtype=float)
    ub = np.asarray(upper_bound, dtype=float)
    covered = (a >= lb) & (a <= ub)
    return float(np.mean(covered) * 100.0)


def sharpness(lower_bound, upper_bound) -> float:
    """Mean width of prediction interval."""
    lb = np.asarray(lower_bound, dtype=float)
    ub = np.asarray(upper_bound, dtype=float)
    return float(np.mean(ub - lb))


def evaluate_forecast(y_true, y_pred, baseline_pred=None) -> dict[str, float]:
    """Compute comprehensive diagnostic metrics."""
    res = {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "smape": smape(y_true, y_pred),
    }
    if baseline_pred is not None:
        res["normalized_mae"] = normalized_mae(y_true, y_pred, baseline_pred)
    return res


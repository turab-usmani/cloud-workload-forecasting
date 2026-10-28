"""
Leakage-safe feature extraction for direct multi-step cloud workload forecasting.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def extract_features_at_origin(
    history_df: pd.DataFrame,
    target_col: str = "aggregate_cpu_utilization",
    aux_cols: list[str] | None = None,
) -> dict[str, float]:
    """
    Extract statistical, momentum, and lag features strictly using history up to the current row.
    Guarantees zero future lookahead.
    """
    if len(history_df) < 12:
        raise ValueError("history_df must contain at least 12 observations for feature extraction")

    y = history_df[target_col].to_numpy(dtype=float)
    n = len(y)
    curr = y[-1]

    feats: dict[str, float] = {}

    # 1. Level & Lags
    feats["curr_val"] = curr
    feats["lag_1"] = y[-2] if n >= 2 else curr
    feats["lag_2"] = y[-3] if n >= 3 else curr
    feats["lag_3"] = y[-4] if n >= 4 else curr
    feats["lag_5"] = y[-6] if n >= 6 else curr
    feats["lag_11"] = y[-12] if n >= 12 else curr

    # 2. Velocity & Acceleration
    feats["diff_1"] = curr - feats["lag_1"]
    feats["diff_3"] = curr - feats["lag_3"]
    feats["diff_6"] = curr - (y[-7] if n >= 7 else curr)
    feats["accel"] = (curr - feats["lag_1"]) - (feats["lag_1"] - feats["lag_2"])

    # 3. Rolling Statistics
    for w in [3, 6, 12]:
        window_vals = y[-w:]
        feats[f"roll_mean_{w}"] = float(np.mean(window_vals))
        feats[f"roll_std_{w}"] = float(np.std(window_vals)) if len(window_vals) > 1 else 0.0
        feats[f"roll_min_{w}"] = float(np.min(window_vals))
        feats[f"roll_max_{w}"] = float(np.max(window_vals))
        feats[f"roll_spread_{w}"] = feats[f"roll_max_{w}"] - feats[f"roll_min_{w}"]

    # 4. Exponentially Weighted Moving Averages (EWMA)
    s = pd.Series(y)
    feats["ewma_span3"] = float(s.ewm(span=3).mean().iloc[-1])
    feats["ewma_span6"] = float(s.ewm(span=6).mean().iloc[-1])
    feats["ewma_span12"] = float(s.ewm(span=12).mean().iloc[-1])

    # 5. Ratio & Normalized Position
    feats["ratio_to_mean12"] = curr / (feats["roll_mean_12"] + 1e-6)
    feats["step_idx"] = float(n)

    # 6. Auxiliary Category Signals
    if aux_cols:
        for col in aux_cols:
            if col in history_df.columns:
                arr = history_df[col].to_numpy(dtype=float)
                feats[f"{col}_curr"] = arr[-1]
                feats[f"{col}_diff1"] = arr[-1] - (arr[-2] if len(arr) >= 2 else arr[-1])
                feats[f"{col}_mean6"] = float(np.mean(arr[-6:]))

    return feats


def build_direct_multistep_dataset(
    df_train: pd.DataFrame,
    target_col: str = "aggregate_cpu_utilization",
    aux_cols: list[str] | None = None,
    horizon: int = 24,
    min_history: int = 12,
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """
    Construct feature matrix X and multi-horizon target vectors Y_h for direct forecasting.
    For each valid origin t (from min_history to len(df) - horizon):
      X[t] is derived strictly from rows 0..t.
      For each h in 1..horizon, target y_{t+h} is recorded.
    """
    n_samples = len(df_train)
    max_origin = n_samples - horizon
    if max_origin < min_history:
        # Fallback if horizon is large relative to n_samples
        min_history = max(6, max_origin // 2)

    records = []
    y_targets: dict[int, list[float]] = {h: [] for h in range(1, horizon + 1)}

    for t in range(min_history, max_origin + 1):
        hist_slice = df_train.iloc[:t]
        feat_dict = extract_features_at_origin(hist_slice, target_col=target_col, aux_cols=aux_cols)
        records.append(feat_dict)

        for h in range(1, horizon + 1):
            y_targets[h].append(float(df_train.iloc[t + h - 1][target_col]))

    if not records:
        raise ValueError(f"Insufficient history ({n_samples}) for horizon={horizon} and min_history={min_history}")

    X = pd.DataFrame(records)
    Y = {h: np.array(vals, dtype=float) for h, vals in y_targets.items()}
    return X, Y

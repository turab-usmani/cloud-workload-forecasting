"""
Time-aware, leakage-safe validation split generators.
"""

from __future__ import annotations
from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class TimeSplit:
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp


def rolling_origin_splits(
    frame: pd.DataFrame,
    timestamp_col: str = "timestamp",
    horizon: int = 288,
    min_train_size: int = 7 * 24 * 12,
    step: int | None = None,
) -> list[TimeSplit]:
    """Return expanding-window splits for evenly sampled time series."""
    if horizon <= 0 or min_train_size <= 0:
        raise ValueError("horizon and min_train_size must be positive")
    ts = pd.to_datetime(frame[timestamp_col], utc=True)
    if not ts.is_monotonic_increasing:
        raise ValueError("timestamps must be sorted")
    n = len(frame)
    step = horizon if step is None else step
    out: list[TimeSplit] = []
    train_end_idx = min_train_size
    while train_end_idx + horizon <= n:
        train_end = ts.iloc[train_end_idx - 1]
        val_start = ts.iloc[train_end_idx]
        val_end = ts.iloc[train_end_idx + horizon - 1]
        out.append(TimeSplit(train_end, val_start, val_end))
        train_end_idx += step
    return out


@dataclass(frozen=True)
class TimeSplitIndices:
    split_id: int
    train_end_idx: int  # exclusive upper bound for training indices: [0, train_end_idx)
    val_start_idx: int  # start of validation
    val_end_idx: int    # exclusive upper bound for validation: [val_start_idx, val_end_idx)
    horizon: int


def expanding_window_splits(
    n_samples: int,
    horizon: int = 24,
    min_train_size: int = 38,
    step: int = 12,
) -> list[TimeSplitIndices]:
    """
    Generate rolling-origin expanding-window split indices strictly within the training set.
    Each split uses an expanding historical window to predict the next `horizon` steps,
    faithfully simulating the production 2-hour forecasting task.
    """
    if horizon <= 0 or min_train_size <= 0:
        raise ValueError("horizon and min_train_size must be positive integers")
    if min_train_size + horizon > n_samples:
        raise ValueError(f"min_train_size ({min_train_size}) + horizon ({horizon}) exceeds n_samples ({n_samples})")

    splits: list[TimeSplitIndices] = []
    split_id = 1
    current_train_end = min_train_size

    while current_train_end + horizon <= n_samples:
        val_start = current_train_end
        val_end = current_train_end + horizon
        splits.append(
            TimeSplitIndices(
                split_id=split_id,
                train_end_idx=current_train_end,
                val_start_idx=val_start,
                val_end_idx=val_end,
                horizon=horizon,
            )
        )
        current_train_end += step
        split_id += 1

    return splits

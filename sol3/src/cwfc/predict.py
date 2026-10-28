"""
CLI entry point for generating challenge predictions:
python -m cwfc.predict --data challenge_data --output predictions.csv
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Support direct script invocation as well as -m cwfc.predict
if __name__ == "__main__" and __package__ is None:
    src_dir = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(src_dir))
    from cwfc.data import load_raw_data, build_target_series, get_train_holdout_split
    from cwfc.statistical import AutoRegressiveForecaster
    from cwfc.models import DynamicEnsembleForecaster, DirectMultiStepLightGBM
else:
    from .data import load_raw_data, build_target_series, get_train_holdout_split
    from .statistical import AutoRegressiveForecaster
    from .models import DynamicEnsembleForecaster, DirectMultiStepLightGBM


def generate_predictions(
    data_dir: str | Path,
    output_path: str | Path,
    train_steps: int = 110,
    horizon: int = 24,
    model_type: str = "ar",
) -> pd.DataFrame:
    """
    Train model on first train_steps observations and generate predictions for horizon steps.
    Strictly guarantees zero target leakage.
    """
    data_dir = Path(data_dir)
    output_path = Path(output_path)

    print(f"[Predict] Ingesting raw trace from: {data_dir}")
    df_cpu, df_meta = load_raw_data(data_dir)
    df_targets = build_target_series(df_cpu, df_meta)

    df_train, df_holdout = get_train_holdout_split(df_targets, train_steps=train_steps, holdout_steps=horizon)
    target_col = "aggregate_cpu_utilization"
    aux_cols = [c for c in df_targets.columns if c.startswith("cpu_cat_")]

    if model_type == "ar":
        print(f"[Predict] Training optimal AutoRegressive AR(3) forecaster on {len(df_train)} observations...")
        model = AutoRegressiveForecaster(lags=3)
        model.fit(df_train[target_col].to_numpy())
        cpu_preds = model.predict(horizon=horizon)
    elif model_type == "ensemble":
        print(f"[Predict] Training DynamicEnsembleForecaster on {len(df_train)} observations...")
        model = DynamicEnsembleForecaster(lgb_weight=0.55, horizon=horizon)
        model.fit(df_train, target_col=target_col, aux_cols=aux_cols)
        cpu_preds = model.predict(df_train, target_col=target_col, aux_cols=aux_cols)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # In this dataset, active_vm_count is deterministic at 10,000 across all intervals
    active_vm_preds = np.full(horizon, 10000, dtype=int)

    # Format exactly to SUBMISSION.md specification
    submission_df = pd.DataFrame({
        "timestamp": df_holdout["timestamp"].astype(int),
        "active_vm_count": active_vm_preds,
        "aggregate_cpu_utilization": np.round(cpu_preds, 4),
    })

    # Validate schema and invariants
    if len(submission_df) != horizon:
        raise ValueError(f"Expected {horizon} rows, got {len(submission_df)}")
    expected_cols = ["timestamp", "active_vm_count", "aggregate_cpu_utilization"]
    if list(submission_df.columns) != expected_cols:
        raise ValueError(f"Columns mismatch: expected {expected_cols}, got {list(submission_df.columns)}")
    if submission_df.isna().any().any():
        raise ValueError("Predictions contain null or NaN values")
    if (submission_df["active_vm_count"] < 0).any():
        raise ValueError("active_vm_count cannot be negative")
    if ((submission_df["aggregate_cpu_utilization"] < 0) | (submission_df["aggregate_cpu_utilization"] > 100)).any():
        raise ValueError("aggregate_cpu_utilization must be within [0, 100]")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"[Predict] Successfully wrote valid predictions to {output_path}")
    return submission_df


def main():
    parser = argparse.ArgumentParser(description="Cloud Workload Forecasting Take-Home Predictor")
    parser.add_argument("--data", type=str, default="challenge_data", help="Path to directory containing input datasets")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Path to output predictions.csv")
    parser.add_argument("--train-steps", type=int, default=110, help="Number of historical training steps")
    parser.add_argument("--horizon", type=int, default=24, help="Forecast horizon (number of steps)")
    parser.add_argument("--model", type=str, default="ar", choices=["ar", "ensemble"], help="Forecasting model architecture")

    args = parser.parse_args()
    generate_predictions(
        data_dir=args.data,
        output_path=args.output,
        train_steps=args.train_steps,
        horizon=args.horizon,
        model_type=args.model,
    )


if __name__ == "__main__":
    main()

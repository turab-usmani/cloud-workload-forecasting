"""
Trace-driven discrete-event capacity planning simulation.
Evaluates operational trade-offs: SLA violations vs over-provisioned cloud headroom.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class PolicyEvaluationResult:
    policy_name: str
    sla_violation_count: int
    sla_violation_rate_pct: float
    max_deficit_pct: float
    mean_wasted_capacity_pct: float
    total_allocated_core_pct_hours: float
    efficiency_vs_static_pct: float


def run_capacity_simulation(
    y_true: np.ndarray,
    forecast_point: np.ndarray,
    forecast_p95: np.ndarray,
    train_history: np.ndarray,
) -> pd.DataFrame:
    """
    Simulate cluster capacity provisioning across the 24-step holdout horizon (2 hours).

    Compares 4 operational policies:
      1. Static Peak Policy (peak of train history + 5% buffer)
      2. Naive Persistence + Fixed Buffer (+0.5% CPU headroom)
      3. Model Point Forecast + Fixed Buffer (+0.5% CPU headroom)
      4. Dynamic Model Quantile Headroom (LightGBM p95 upper bound)
    """
    y_true = np.asarray(y_true, dtype=float)
    n_steps = len(y_true)

    # 1. Static Peak Policy
    peak_train = float(np.max(train_history))
    cap_static = np.full(n_steps, peak_train * 1.05)

    # 2. Persistence + Fixed Buffer
    last_val = float(train_history[-1])
    cap_persist = np.full(n_steps, last_val + 0.50)

    # 3. Model Point + Fixed Buffer
    cap_point_buffer = forecast_point + 0.50

    # 4. Model Dynamic Quantile Headroom (p95)
    cap_dynamic_q95 = np.maximum(forecast_p95, forecast_point + 0.15)

    policies = {
        "Static Peak Policy": cap_static,
        "Persistence + Fixed Buffer": cap_persist,
        "Model Point + Fixed Buffer": cap_point_buffer,
        "Dynamic Model Quantile Headroom (p95)": cap_dynamic_q95,
    }

    # Baseline wasted capacity from static peak for reference
    static_waste = np.mean(np.maximum(0.0, cap_static - y_true))

    results = []
    for name, cap in policies.items():
        deficits = np.maximum(0.0, y_true - cap)
        slacks = np.maximum(0.0, cap - y_true)

        viol_count = int(np.sum(deficits > 1e-4))
        viol_rate = (viol_count / n_steps) * 100.0
        max_deficit = float(np.max(deficits))
        mean_waste = float(np.mean(slacks))
        
        # 5-min intervals = 5/60 hours per interval
        total_allocated_hours = float(np.sum(cap) * (5.0 / 60.0))
        efficiency_gain = ((static_waste - mean_waste) / (static_waste + 1e-8)) * 100.0

        results.append(
            {
                "Policy": name,
                "SLA Violations (Steps)": f"{viol_count}/{n_steps}",
                "SLA Violation Rate (%)": round(viol_rate, 2),
                "Max Deficit (% CPU)": round(max_deficit, 3),
                "Mean Idle Waste (% CPU)": round(mean_waste, 3),
                "Total Allocated (%·hr)": round(total_allocated_hours, 2),
                "Capacity Savings vs Static (%)": round(efficiency_gain, 1),
            }
        )

    return pd.DataFrame(results)


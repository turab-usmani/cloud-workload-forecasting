# Cloud Workload Forecasting & Capacity Planning Engine (`sol3`)

Production-grade time-series forecasting and trace-driven capacity planning engine developed for the Azure VM Workload Challenge (`data-v1`).

---

## 1. Executive Summary & Key Results

This solution treats cloud workload forecasting through the lens of a **Principal Cloud Capacity / Systems ML Engineer**:
- **Workload Characterization**: Discovered that the 10,000 VM fleet is deterministic (zero churn across the 134 five-minute intervals). The aggregate CPU metric exhibits high autocorrelation ($ACF_1 = 0.66$, $ACF_3 = 0.66$) with sharp partial autocorrelation cutoff at lag 3. Spectral FFT reveals slow multi-hour macro drift superimposed on 14–15 minute micro-oscillations.
- **Leakage-Safe Architecture**: Strict expanding-window cross-validation (4 temporal folds over 110 training steps, matching the exact 24-step horizon). Zero lookahead.
- **Benchmark Victory**: The **AutoRegressive AR(3)** model achieves the lowest out-of-sample error on the 2-hour holdout, outperforming Naive Persistence by **34.3%**:
  - **Holdout MAE**: **0.0964% CPU** (Normalized MAE = **0.6571** vs Persistence 0.1468%)
  - **Holdout RMSE**: **0.1205% CPU**
  - **Holdout sMAPE**: **1.039%**
- **Non-Parametric Tail Risk Modeling**: A Direct Multi-Horizon **LightGBM Quantile Regressor ($\alpha=0.95$)** predicts the dynamic upper bound of workload demand.
- **Trace-Driven Discrete-Event Capacity Simulation**: When evaluated in a realistic autoscaler simulation across the 24 holdout intervals:
  - **SLA Violations**: **0 / 24 steps (0.0% failure rate)**
  - **Idle Capacity Reduction**: **57.2% reduction in wasted headroom** compared to standard static peak allocation.

---

## 2. Benchmark Summary Table

### Expanding-Window Cross-Validation (110 Train Steps, Horizon=24)
| Model Architecture | CV MAE | CV RMSE | CV sMAPE (%) | Normalized MAE (vs Persistence) |
|---|---|---|---|---|
| **AutoRegressive AR(3)** | **0.2039** | **0.2343** | **2.156%** | **0.9917** |
| Persistence (Naive) | 0.2056 | 0.2387 | 2.164% | 1.0000 |
| Holt Linear Trend (Damped) | 0.2533 | 0.2877 | 2.657% | 1.2319 |
| Dynamic Ensemble (LGBM+Holt) | 0.2535 | 0.2861 | 2.675% | 1.2328 |
| Moving Average ($w=12$) | 0.2545 | 0.2823 | 2.681% | 1.2376 |
| Direct LightGBM (L1) | 0.2981 | 0.3351 | 3.143% | 1.4496 |
| Seasonal Naive ($s=24$) | 0.3124 | 0.3504 | 3.282% | 1.5191 |
| Direct Ridge ($L_2$) | 0.7766 | 0.9292 | 8.106% | 3.7770 |

### Final Public Holdout Performance (Steps 111–134, 2-Hour Horizon)
| Model Architecture | Holdout MAE | Holdout RMSE | Holdout sMAPE (%) | Normalized MAE (vs Persistence) |
|---|---|---|---|---|
| **AutoRegressive AR(3)** | **0.0964** | **0.1205** | **1.039%** | **0.6571 (34.3% improvement)** |
| Moving Average ($w=12$) | 0.1230 | 0.1613 | 1.326% | 0.8380 |
| Dynamic Ensemble (LGBM+Holt) | 0.1366 | 0.1637 | 1.472% | 0.9304 |
| Seasonal Naive ($s=24$) | 0.1464 | 0.1671 | 1.579% | 0.9974 |
| Persistence (Last Value) | 0.1468 | 0.1848 | 1.585% | 1.0000 |
| Holt Linear Trend (Damped) | 0.1693 | 0.2080 | 1.832% | 1.1537 |
| Direct LightGBM (L1) | 0.1852 | 0.2212 | 1.988% | 1.2615 |
| Direct Ridge ($L_2$) | 0.2410 | 0.2706 | 2.582% | 1.6419 |

---

## 3. Quick Start & Reproduction

### Prerequisites
Python 3.11+ with `numpy`, `pandas`, `scipy`, `lightgbm`, `statsmodels`, `pytest`, `matplotlib`.

### 1. Generate Holdout Predictions (One-Command CLI)
```bash
# Direct CLI invocation
python sol3/src/cwfc/predict.py --data challenge_data --output predictions.csv

# Or as an installed module
python -m cwfc.predict --data challenge_data --output predictions.csv
```
This generates `predictions.csv` with exactly 24 rows, matching all formatting and validity criteria from `SUBMISSION.md`.

### 2. Run Comprehensive Benchmarks & Simulations
```bash
python sol3/scripts/run_benchmarks.py
```
This executes expanding-window CV, evaluates holdout performance, executes the capacity planning simulation, and exports high-resolution diagnostic plots to `sol3/reports/figures/`.

### 3. Run Automated Pytest Suite
```bash
pytest sol3/tests
```
Executes 16 unit and contract tests verifying data integrity, metric correctness, model behavior, simulation sanity, and **temporal anti-leakage invariance**.

---

## 4. Repository Structure

```text
sol3/
├── README.md                      # This comprehensive overview
├── pyproject.toml                 # Package definition & dependencies
├── predictions.csv                # Validated 24-row holdout submission
├── src/
│   └── cwfc/
│       ├── __init__.py
│       ├── data.py                # Data ingestion, schema validation, target aggregation
│       ├── features.py            # Origin-aware multi-scale rolling stats & lag features
│       ├── validation.py          # Expanding-window temporal CV splitter (leakage-safe)
│       ├── baselines.py           # Persistence, Moving Average, Holt Trend, Seasonal Naive
│       ├── statistical.py         # AutoRegressive AR(p) model
│       ├── models.py              # Direct Multi-Step LightGBM (Point & Quantile) & Ensembles
│       ├── metrics.py             # MAE, RMSE, sMAPE, Normalized MAE, Pinball Loss, Coverage
│       ├── simulation.py          # Discrete-event cluster autoscaling simulation
│       └── predict.py             # CLI prediction entrypoint
├── tests/
│   ├── test_contract.py           # Validates predictions.csv matches SUBMISSION.md
│   ├── test_data.py               # Tests target series construction
│   ├── test_features_leakage.py   # Critical temporal invariance / anti-leakage tests
│   ├── test_models.py             # Verifies all model interfaces & output shapes
│   ├── test_metrics.py            # Numerical tests for error & uncertainty metrics
│   └── test_simulation.py         # Tests capacity simulation logic
├── scripts/
│   ├── run_eda.py                 # Exploratory workload & FFT spectral analysis
│   └── run_benchmarks.py          # Expanding-window CV, benchmark evaluation & plotting
└── reports/
    ├── final_report.md            # Complete 6-page technical report
    └── figures/                   # Publication-quality diagnostic plots
        ├── eda_targets_overview.png
        ├── eda_workload_categories.png
        ├── eda_spectral_acf.png
        ├── holdout_forecast_comparison.png
        ├── horizon_error_degradation.png
        └── capacity_simulation_policy.png
```


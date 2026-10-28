# Cloud Workload Forecasting & Capacity Planning Engine

[![Tests](https://img.shields.io/badge/pytest-17%20passed-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

End-to-end, production-grade cloud workload forecasting and trace-driven capacity planning engine built on the **Microsoft Azure Public Dataset V2** (`data-v1` curated release).

---

## 1. Executive Summary & Headline Results

This solution addresses the core challenge question:
> *"How predictable is this cloud workload, how accurately can near-term demand be forecast, and what should an infrastructure team do with the forecast?"*

- **Workload Structure:** Fleet size is invariant at exactly 10,000 active VMs with zero churn across all 134 observation intervals (~11.1 hours). Mean aggregate CPU utilization hovers tightly ($\mu = 9.39\%, \sigma = 0.19\%$), exhibiting strong autoregressive memory ($ACF_1 = 0.662$) with an abrupt PACF cutoff at lag 3.
- **Benchmark Winner:** The **AutoRegressive AR(3)** model achieves the superior out-of-sample forecast accuracy across the 2-hour holdout (24 intervals), outperforming Naive Persistence by **34.3%**:
  - **Holdout MAE:** **0.0964% CPU** (Normalized MAE = **0.6571** vs Persistence 0.1468%)
  - **Holdout RMSE:** **0.1205% CPU**
  - **Holdout sMAPE:** **1.039%**
- **Uncertainty & Tail Risk:** Direct Multi-Horizon **LightGBM Quantile Regression ($\alpha=0.95$)** produces non-parametric dynamic headroom bounds.
- **Discrete-Event Capacity Simulation:** In a realistic autoscaler simulation across the holdout horizon:
  - **Zero SLA violations ($0/24$ intervals, $0.0\%$ failure rate)**
  - **57.2% reduction in wasted idle headroom** compared to static peak allocation.

---

## 2. Quick Start & Reproduction

### Environment Setup
Python **3.11 or newer** is required.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate
# On Windows PowerShell:
.venv\Scripts\Activate.ps1

# 2. Install dependencies and local package in editable mode
pip install --upgrade pip
pip install -e ".[dev]"
```

### Reproduce Final Predictions (`predictions.csv`)
Run the CLI predictor from the repository root:
```bash
python -m cwfc.predict --data challenge_data --output predictions.csv
```
*Output: Strictly validates and writes the 24-row `predictions.csv` matching all criteria in `SUBMISSION.md`.*

### Run Automated Test Suite
```bash
pytest
```
*Executes 17 comprehensive unit and contract tests verifying data loaders, metrics, model bounds, simulation sanity, and **strict temporal anti-leakage invariance**.*

### Re-run Full Exploratory Analysis & Benchmarks
```bash
# Run Exploratory Data Analysis & FFT spectral decomposition
python sol3/scripts/run_eda.py

# Run expanding-window cross-validation, holdout evaluation & capacity simulation
python sol3/scripts/run_benchmarks.py
```

---

## 3. Benchmark Results Summary

### Expanding-Window Cross-Validation (4 Temporal Folds over 110 Training Steps, Horizon=24)
| Model Architecture | CV MAE | CV RMSE | CV sMAPE (%) | Normalized MAE (vs Persistence) |
|---|---|---|---|---|
| **AutoRegressive AR(3)** | **0.2039** | **0.2343** | **2.156%** | **0.9917** |
| Persistence Baseline | 0.2056 | 0.2387 | 2.164% | 1.0000 |
| Holt Linear Trend (Damped) | 0.2533 | 0.2877 | 2.657% | 1.2319 |
| Dynamic Ensemble (LGBM+Holt) | 0.2535 | 0.2861 | 2.675% | 1.2328 |
| Moving Average ($w=12$) | 0.2545 | 0.2823 | 2.681% | 1.2376 |
| Direct LightGBM ($L_1$) | 0.2981 | 0.3351 | 3.143% | 1.4496 |
| Seasonal Naive ($s=24$) | 0.3124 | 0.3504 | 3.282% | 1.5191 |
| Direct Ridge ($L_2$) | 0.7766 | 0.9292 | 8.106% | 3.7770 |

### Final 2-Hour Public Holdout Performance (Timestamps 111–134)
| Model Architecture | Holdout MAE | Holdout RMSE | Holdout sMAPE (%) | Normalized MAE (vs Persistence) |
|---|---|---|---|---|
| **AutoRegressive AR(3)** | **0.0964** | **0.1205** | **1.039%** | **0.6571 (-34.3% error)** |
| Moving Average ($w=12$) | 0.1230 | 0.1613 | 1.326% | 0.8380 (-16.2%) |
| Dynamic Ensemble (LGBM+Holt) | 0.1366 | 0.1637 | 1.472% | 0.9304 (-7.0%) |
| Seasonal Naive ($s=24$) | 0.1464 | 0.1671 | 1.579% | 0.9974 (-0.3%) |
| Persistence (Last Value) | 0.1468 | 0.1848 | 1.585% | 1.0000 (0.0%) |
| Holt Linear Trend (Damped) | 0.1693 | 0.2080 | 1.832% | 1.1537 (+15.4%) |
| Direct LightGBM ($L_1$) | 0.1852 | 0.2212 | 1.988% | 1.2615 (+26.2%) |
| Direct Ridge ($L_2$) | 0.2410 | 0.2706 | 2.582% | 1.6419 (+64.2%) |

---

## 4. Operational Capacity Planning Simulation

In cloud infrastructure, under-provisioning causes SLA violations and tail latency spikes, whereas over-provisioning causes wasted capital spend. We tested our dynamic quantile policy against traditional static headroom over the 24 holdout intervals:

| Provisioning Policy | SLA Violations | Deficit Max | Mean Idle Waste | Capacity Savings vs Static |
|---|---|---|---|---|
| Static Peak Policy (+5% safety margin) | 0 / 24 (0.0%) | 0.000% | 1.139% CPU | 0.0% (Baseline) |
| Persistence + 0.5% Fixed Buffer | 0 / 24 (0.0%) | 0.000% | 0.396% CPU | +65.3% waste reduction |
| **Dynamic Model Quantile Headroom ($q_{95}$)** | **0 / 24 (0.0%)** | **0.000%** | **0.488% CPU** | **+57.2% waste reduction** |

**Recommendation:** Deploy a two-tier autoscaling architecture:
1. *Short-horizon tier ($h \le 3$, 5–15 min):* Use the AR(3) point forecast for responsive VM/container auto-scaling.
2. *Proactive planning tier ($h > 3$, 20–120 min):* Use the dynamic 95th-percentile headroom bound to plan host bin-packing, server power-cycling, and batch window admissions.

---

## 5. Repository Structure

```text
├── README.md                      # Complete system documentation & reproduction guide
├── pyproject.toml                 # Package definition with dependencies & CLI entrypoint
├── predictions.csv                # Validated 24-step holdout submission file
├── challenge_data/                # Challenge dataset directory (downloaded via data-v1 release)
├── reports/
│   ├── final_report.md            # Comprehensive 6-page engineering report
│   └── figures/                   # Diagnostic plots & figures
│       ├── eda_targets_overview.png
│       ├── eda_workload_categories.png
│       ├── eda_spectral_acf.png
│       ├── holdout_forecast_comparison.png
│       ├── horizon_error_degradation.png
│       └── capacity_simulation_policy.png
├── src/
│   └── cwfc/
│       ├── __init__.py
│       ├── data.py                # Ingestion, schema validation, target aggregation
│       ├── features.py            # Origin-aware rolling statistics & lag extraction
│       ├── validation.py          # Expanding-window temporal CV splitter
│       ├── baselines.py           # Persistence, Moving Average, Holt Trend, Seasonal Naive
│       ├── statistical.py         # AutoRegressive AR(p) model
│       ├── models.py              # Direct Multi-Step LightGBM & Dynamic Ensemble
│       ├── metrics.py             # MAE, RMSE, sMAPE, Normalized MAE, Pinball Loss
│       ├── simulation.py          # Trace-driven discrete-event capacity simulator
│       └── predict.py             # CLI entrypoint for model training & prediction
├── tests/
│   ├── test_contract.py           # Validates predictions.csv matches SUBMISSION.md
│   ├── test_data.py               # Tests target series aggregation
│   ├── test_features_leakage.py   # Critical temporal anti-leakage invariance tests
│   ├── test_models.py             # Verifies all forecasting model interfaces
│   ├── test_metrics.py            # Numerical tests for error & uncertainty metrics
│   ├── test_simulation.py         # Tests capacity simulation logic
│   └── test_validation.py         # Tests temporal validation splits
└── sol3/                          # Archived modular workspace snapshot
```

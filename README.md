# 🏆 Datathon 2026 — The Gridbreakers

> **Competition:** Datathon 2026 – Round 1
> **Task:** Daily Revenue & COGS Forecasting (Jan 1, 2023 – Jul 1, 2024)
> **Approach:** 3-Model Stacked Ensemble (LightGBM + XGBoost + CatBoost) with Ridge Meta-Learner
> **Final Stack MAE:** 488,323 VND &nbsp;|&nbsp; **R²:** 0.8409

---

## 📁 Directory Structure

```
datathon-2026-round-1/
│
├── 📓 datathon2026_full_pipeline.ipynb   # Main notebook — runs the full pipeline
├── 🐍 datathon2026_full_pipeline.py      # Equivalent Python script
├── 📄 README.md                          # This file
│
├── 📂 submission/
│   └── submission_final.csv              # Final forecast file (548 days)
│
└── 📂 outputs/
    ├── X_train_v7.parquet                # Training feature matrix
    ├── X_test_v7.parquet                 # Test feature matrix
    ├── train_df_v7.parquet               # Full training dataframe
    ├── feature_list_v7.json              # List of 25 selected features
    ├── best_params_lgbm_v7.json          # Best hyperparameters — LightGBM
    ├── best_params_xgb_v7.json           # Best hyperparameters — XGBoost
    ├── best_params_cat_v7.json           # Best hyperparameters — CatBoost
    ├── best_params_margin_v7.json        # Best hyperparameters — Margin model
    ├── shap_importance_v7.csv            # SHAP feature importance scores
    ├── shap_summary_v7.png               # SHAP summary bar chart
    │
    ├── 📂 models/
    │   ├── lgbm_v7.pkl                   # LightGBM model (full refit)
    │   ├── xgb_v7.pkl                    # XGBoost model (full refit)
    │   ├── cat_v7.pkl                    # CatBoost model (full refit)
    │   ├── margin_v7.pkl                 # Gross margin prediction model
    │   └── ridge_v7.pkl                  # Ridge stacking meta-learner
    │
    └── 📂 viz/                           # 18 EDA charts
        ├── viz1_revenue_margin_trend.png
        ├── viz2_sessions_by_source.png
        ├── viz3_order_volume_status.png
        ├── viz4_revenue_by_category.png
        ├── viz5_return_rate_by_category.png
        ├── viz6_rating_by_category_segment.png
        ├── viz7_seasonal_revenue.png
        ├── viz8_promo_vs_no_promo.png
        ├── viz9_inventory_health.png
        ├── viz10_customer_acquisition.png
        ├── viz11_days_to_ship_region.png
        ├── viz12_payment_method_value.png
        ├── viz13_age_gender_heatmap.png
        ├── viz14_annual_margin_trend.png
        ├── viz15_returns_analysis.png
        ├── viz16_revenue_decline_analysis.png
        ├── viz17_source_device_heatmap.png
        └── viz18_cancel_return_rates.png
```

> **Note:** The 13 raw CSV files are not included in this repository due to size constraints. See Step 1 below before running the pipeline.

---

## ⚙️ Requirements

**Python 3.9+**

Install all dependencies with:
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn \
            lightgbm xgboost catboost optuna shap joblib pyarrow
```

---

## 🚀 How to Reproduce Results

### Step 1 — Prepare the data
Place all 13 raw CSV files in the root project folder (same level as the notebook):

```
sales.csv, sample_submission.csv, web_traffic.csv, orders.csv,
order_items.csv, payments.csv, shipments.csv, returns.csv,
reviews.csv, inventory.csv, promotions.csv, products.csv,
customers.csv, geography.csv
```

### Step 2 — Update the folder path
Open the notebook or script and update the `FOLDER` variable to point to your local directory:

```python
FOLDER = "/path/to/your/datathon-2026-round-1"
```

### Step 3 — Run the notebook
```bash
jupyter notebook datathon2026_full_pipeline.ipynb
```
Run all cells top-to-bottom in order (**Kernel → Restart & Run All**).

Or run as a Python script:
```bash
python datathon2026_full_pipeline.py
```

### Step 4 — Verify the output
The submission file will be written to:
```
submission/submission_final.csv
```
The pipeline prints a SHA256 hash for verification:
```
SHA256: 29cab5e398cdc199d03f87845d1fef004cb3f0e667c79530307ed6c4e98263f2
```

> ⚠️ **Optuna tuning (300 trials across 4 models) takes approximately 30–60 minutes** depending on hardware. Random seed = 42 is fixed throughout to ensure full reproducibility.

---

## 🧠 Pipeline Overview

### Part 1 — Data Profiling & Structuring
- Loads all 13 tables, parses dates, fills nulls with sentinel values
- Computes `product_margin` = (price − COGS) / price × 100
- Validates schema: 3,833 training rows and 548 test rows confirmed

### Part 2 — Exploratory Data Analysis (18 Charts)

| # | Chart | Analysis Level |
|---|-------|---------------|
| 1 | Monthly Revenue & Gross Margin Trend | Predictive |
| 2 | Sessions by Traffic Source Over Time | Diagnostic |
| 3 | Order Volume by Year & Status | Diagnostic |
| 4 | Revenue by Product Category & Segment | Descriptive |
| 5 | Return Rate by Product Category | Prescriptive |
| 6 | Average Rating by Category & Segment | Diagnostic |
| 7 | Seasonal Revenue Pattern | Prescriptive |
| 8 | Order Value: Promo vs No Promo | Prescriptive |
| 9 | Inventory Health — Stockout & Overstock | Prescriptive |
| 10 | Customer Acquisition by Channel | Predictive |
| 11 | Days to Ship by Region | Descriptive |
| 12 | Payment Method vs Average Order Value | Prescriptive |
| 13 | Age × Gender Heatmap of Avg Order Value | Descriptive |
| 14 | Annual Gross Margin Trend | Prescriptive |
| 15 | Return Reasons Analysis | Prescriptive |
| 16 | Revenue Decline: Volume vs Price Decomposition | Diagnostic |
| 17 | Order Source × Device Type Heatmap | Prescriptive |
| 18 | Annual Cancellation & Return Rate Trends | Diagnostic |

### Part 3 — Modeling & Prediction

**Model architecture:**
```
LightGBM  (Optuna, 100 trials) ─┐
XGBoost   (Optuna, 100 trials) ─┼─► Ridge Meta-Learner ─► 548-day Forecast
CatBoost  (Optuna, 100 trials) ─┘

+ Margin sub-model (LightGBM, 100 trials)
  → COGS = Revenue × (1 − predicted_margin%)
```

**Validation strategy:** TimeSeriesSplit (5 folds, 30-day gap) + 2022 held-out set

**Forecast method:** Recursive (autoregressive) — lag features for test days are filled from a rolling buffer of prior predictions

**25 features across 5 groups:**

| Group | Features |
|-------|---------|
| Calendar | `year`, `day`, `dayofweek`, `dayofyear`, `month`, `days_from_month_end` |
| Lag & Rolling | `rev_lag_1/6/7/14/28/365`, `rev_roll7/28_mean`, `rev_ewm7/28` |
| DOY Seasonal Priors | `doy_rev_mean/median/std`, `post2018_doy_rev_mean`, `peak_doy_rev_mean`, `month_dow_rev_mean`, `recent_doy_rev_mean`, `log_doy_rev_mean`, `log_recent_doy_mean` |
| Order Density | `doy_orders_mean` |
| Tết Proximity | `days_to_tet` |

---

## 📊 Model Results

### 2022 Hold-out Performance

| Model | MAE (VND) | RMSE (VND) | R² | MAPE |
|-------|-----------|------------|-----|------|
| LightGBM (untuned baseline) | 531,222 | 722,553 | 0.8137 | 18.32% |
| LightGBM (tuned) | 529,217 | 711,700 | 0.8192 | 18.61% |
| XGBoost (tuned) | 512,582 | 698,562 | 0.8258 | 17.67% |
| CatBoost (tuned) | 514,054 | 710,842 | 0.8196 | 17.42% |
| **Ridge Stack (final)** | **488,323** | **667,735** | **0.8409** | **17.85%** |

### Ridge Ensemble Weights

| Model | Weight |
|-------|--------|
| LightGBM | 0.205 |
| XGBoost | 0.332 |
| CatBoost | 0.392 |
| Regularization (α) | 2.6827 |

### Top 10 SHAP Feature Importances (XGBoost)

| Rank | Feature | Mean \|SHAP\| | Business Interpretation |
|------|---------|--------------|------------------------|
| 1 | `rev_lag_1` | 0.1760 | Yesterday's revenue — strongest single predictor |
| 2 | `year` | 0.0857 | Long-term revenue trend direction (2016–2022 decline) |
| 3 | `doy_rev_mean` | 0.0735 | Historical average for this calendar day |
| 4 | `log_doy_rev_mean` | 0.0390 | Log-scale day-of-year seasonal prior |
| 5 | `rev_lag_7` | 0.0371 | Same weekday last week — weekly shopping cycle |
| 6 | `doy_rev_median` | 0.0357 | Robust day-of-year central tendency |
| 7 | `peak_doy_rev_mean` | 0.0280 | Peak-season DOY average (Mar–Jun, Q2 spike) |
| 8 | `rev_lag_14` | 0.0277 | Two weeks ago — medium-term momentum |
| 9 | `days_to_tet` | 0.0220 | Proximity to Tết — dominant seasonal event |
| 10 | `dayofweek` | 0.0194 | Day-of-week seasonality pattern |

---

## 📈 Forecast Summary (Test Period)

| Metric | Value |
|--------|-------|
| Forecast window | Jan 1, 2023 – Jul 1, 2024 (548 days) |
| Mean daily Revenue | 2,086,111 VND |
| Min daily Revenue | 617,912 VND |
| Max daily Revenue | 3,738,177 VND |
| Mean gross margin | 11.41% |
| COGS > Revenue violations | 0 ✓ |
| Submission SHA256 | `29cab5e398cdc199d03f87845d1fef004cb3f0e667c79530307ed6c4e98263f2` |

### Predicted Monthly Revenue (Daily Average)

| Month | Avg Revenue / Day | Days in Window |
|-------|-------------------|---------------|
| Jan 2023 | 1,448,096 VND | 62 |
| Feb 2023 | 1,863,897 VND | 57 |
| Mar 2023 | 2,312,361 VND | 62 |
| Apr 2023 | 2,667,229 VND | 60 |
| May 2023 | 2,632,343 VND | 62 |
| Jun 2023 | 2,530,647 VND | 60 |
| Jul 2023 | 2,029,000 VND | 32 |
| Aug 2023 | 2,051,897 VND | 31 |
| Sep 2023 | 1,953,680 VND | 30 |
| Oct 2023 | 1,808,617 VND | 31 |
| Nov 2023 | 1,464,437 VND | 30 |
| Dec 2023 | 1,341,071 VND | 31 |

The model correctly captures the Q2 seasonal peak (Apr–May at ~2.6M VND/day) and the post-Tết January trough (1.45M VND/day), consistent with the seasonal pattern identified in EDA.

---

## 🔑 Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Log1p transform on Revenue | Raw revenue has strong right skew; log-transform reduces skew to −0.16 and gives proportionally smaller errors on peak days |
| Separate margin sub-model | Ensures COGS is derived consistently from predicted margin, guaranteeing COGS ≤ Revenue on all 548 rows |
| Recursive (autoregressive) forecast | Lag features (rev_lag_1, rev_ewm7, etc.) require prior test-day predictions; batch prediction would introduce leakage |
| Margin buffer clipped to [2%, 25%] | Prevents a single bad margin prediction from cascading into future margin_lag features over the 548-day window |
| DOY priors computed on training data only | Ensures no target leakage from test-period revenue into seasonal prior features |
| Tết dates hard-coded | Vietnamese lunar calendar dates (2012–2024) are fixed and do not require an external data source |
| Random seed = 42 everywhere | Guarantees identical results on every run across all four models and Optuna studies |

---

## 👥 Team

**The Gridbreakers** — Datathon 2026

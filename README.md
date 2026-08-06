# Explainable Machine Learning for BTCIRT Limit Order Book Dynamics

**Nobitex · BTCIRT · Sparse LOB Snapshots · Multi-Study Redesign (v2.0)**

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](.)
[![Tests](https://img.shields.io/badge/tests-46%20passed-brightgreen)](.)
[![License](https://img.shields.io/badge/license-see%20LICENSE-lightgrey)](LICENSE)
[![Research](https://img.shields.io/badge/status-research%20redesign-orange)](.)

> **Is this a good project to share?**  
> **Yes — especially as a research / methodology portfolio piece.**  
> It is strong because it documents *failed designs*, sparse-data honesty, leakage controls, and explainability limits — not because it claims a trading edge. Share it as **explainable microstructure ML under realistic data constraints**, not as a profitable bot.

---

## Table of contents

1. [Why this project matters](#1-why-this-project-matters)
2. [Executive summary of results](#2-executive-summary-of-results)
3. [Research questions (Studies A / B / C / D)](#3-research-questions)
4. [End-to-end project flow](#4-end-to-end-project-flow)
5. [Repository map & code logic](#5-repository-map--code-logic)
6. [Dataset & filtering](#6-dataset--filtering)
7. [Data-quality & observation-gap audit](#7-data-quality--observation-gap-audit)
8. [Mathematical core](#8-mathematical-core)
9. [Label construction](#9-label-construction)
10. [Feature engineering (families & sets)](#10-feature-engineering)
11. [Temporal validation & leakage control](#11-temporal-validation--leakage-control)
12. [Models & hyperparameters](#12-models--hyperparameters)
13. [Full results (this run)](#13-full-results-this-run)
14. [Explainability (SHAP, permutation, ablation)](#14-explainability)
15. [Challenges we hit (and how we handled them)](#15-challenges-we-hit)
16. [What we do *not* claim](#16-what-we-do-not-claim)
17. [Reproduction](#17-reproduction)
18. [Outputs catalog](#18-outputs-catalog)
19. [Limitations & future work](#19-limitations--future-work)
20. [Research integrity](#20-research-integrity)

### At a glance

<p align="center">
  <img src="reports/figures/data_quality/mid_price_timeseries.png" alt="BTCIRT mid-price over the sample" width="90%"/>
</p>

<p align="center"><em>BTCIRT mid-price path across the 44-day sample (sparse LOB snapshots).</em></p>

<p align="center">
  <img src="reports/figures/models/model_comparison_macro_f1.png" alt="Model comparison Macro F1" width="72%"/>
</p>

<p align="center"><em>Study A development-test Macro F1: baselines vs XGBoost vs CatBoost.</em></p>

---

## 1. Why this project matters

Most “LOB → next price” demos quietly assume:

- dense, regular sampling (e.g. every 100 ms),
- exact fixed horizons (10s / 30s / 60s),
- event-level order-flow imbalance (OFI).

**This dataset is not like that.**

Nobitex BTCIRT snapshots arrive with a **median gap of ~69 seconds** (development-test median ~**189 seconds**). Claiming “30-second forecasting” when the next observation is often minutes away is scientifically wrong.

This repository’s contribution is therefore:

| Contribution | Why it matters |
|---|---|
| **Multi-study redesign** | Separates next-observation, next-change, and strict-horizon questions |
| **Actual-delay tracking** | Every label stores the real future delay |
| **Target-timestamp purging** | Prevents label leakage across train/val/test |
| **Trade deduplication** | Fixes repeated `last_trade_*` inflation |
| **Explainability triad** | SHAP + permutation + ablation (not SHAP alone) |
| **Honest failure logs** | Study C underpower & interaction status logged, not hidden |

---

## 2. Executive summary of results

**Run:** redesign v2.0 · seed `42` · pipeline completed  
**Environment:** Python 3.14 · pandas 3.0.5 · sklearn 1.9 · XGBoost 3.4 · CatBoost 1.2.10 · SHAP 0.52 · macOS arm64

### Data

| Metric | Value |
|---|---|
| Raw rows (all symbols) | **835,000** |
| BTCIRT retained | **39,693** (**4.75%**) |
| Date range | **2026-01-04 → 2026-02-18** |
| Unique dates | **44** |
| Median observation gap | **69.35 s** (test ≈ **188.9 s**) |
| Primary features | **80** (`full_no_trade`) |

### Study A — Next observed mid movement (3-class) — **primary**

| Model | Development-test Macro F1 | Notes |
|---|---|---|
| Majority | 0.211 | Always STABLE |
| Previous direction | 0.377 | |
| OBI rule | 0.385 | |
| **XGBoost (primary)** | **0.470** | Frozen specification |
| CatBoost (challenger) | **0.484** | Competitive; not declared superior without multi-fold proof |

Day-level bootstrap (1000 resamples of dates): Macro F1 mean **0.470**, 95% CI **[0.455, 0.481]**.

Hybrid ε (train-only) = **3.664 bps**. Purged sizes: train **29,568** / val **5,630** / development_test **4,492**.

### Study B — Next mid-change direction (binary)

| Item | Value |
|---|---|
| Primary samples (1 per price run) | **25,704** |
| Median time to next change | **~69.6 s** |
| XGBoost Macro F1 (dev-test) | **0.614** |

### Study C — Strict fixed horizons — **pilot / underpowered**

| Horizon | Window | Eligible | Modeling |
|---|---|---|---|
| 10 s | [5, 15] | **0** | Underpowered pilot |
| 30 s | [20, 40] | **0** | Underpowered pilot |
| 60 s | [40, 80] | **29,318** | Eligible overall, but **0** development_test rows after target purge → **no robust claim** |

### Incremental value (development_test Macro F1)

| Quantity | Δ Macro F1 |
|---|---|
| LOB vs price-only (`full_no_trade − price_only`) | **+0.046** |
| Corrected trades (`full_with_trade − full_no_trade`) | **−0.001** |
| Time features (`full_no_trade − full_no_time`) | **+0.004** |

**Interpretation:** LOB state adds measurable lift over returns/volatility alone. Corrected trade intensity does **not** help. Time features add a tiny lift and must be audited for collection artifacts.

---

## 3. Research questions

### Study A — Next observed mid-price movement

> Can the current and recent LOB state predict whether the **next observed** mid-price moves **DOWN / STABLE / UP**?

Uses consecutive snapshots. Stores `actual_delay_seconds`. Does **not** claim an exact clock horizon.

### Study B — Next mid-price *change* direction

> Conditional on the mid eventually changing, can the book predict **UP vs DOWN**?

Skips unchanged mids; primary sample = one observation per flat price-run.

### Study C — Strict 10 / 30 / 60 second forecasting

> On samples whose future observation delay truly falls in a narrow window, is fixed-horizon prediction feasible?

Defaults: 10∈[5,15], 30∈[20,40], 60∈[40,80]. Under sparse sampling this is mostly a **pilot**.

### Study D — Dense data-collection design

Documentation only (`docs/DENSE_DATA_COLLECTION_DESIGN.md`). No fabricated WebSocket data.

---

## 4. End-to-end project flow

```text
raw CSV
  │
  ▼
[1] data_loader.py          filter exchange=nobitex, symbol=BTCIRT (normalized)
  │
  ▼
[2] data_validation.py      crossed books, ordering, missing/inf, duplicates
  │
  ▼
[3] preprocessing.py        keep-last duplicate timestamps; NO 5s grid
  │
  ▼
[4] delay_audit.py          gap histograms & percentiles (overall / date / hour / split)
  │
  ▼
[5] feature_engineering.py  stationary LOB + lags + snapshot OFI proxies + optional trades
  │
  ▼
[6] labels/
      next_observation.py   Study A
      next_price_change.py  Study B
      strict_horizon.py     Study C + horizon overlap
  │
  ▼
[7] splitting/              date split + target-timestamp purge + walk-forward folds
  │
  ▼
[8] models/                 baselines → logistic → XGBoost → CatBoost
  │
  ▼
[9] evaluation/             metrics + day bootstrap + financial sanity (exploratory)
  │
  ▼
[10] explainability/        SHAP + permutation + ablation + interaction fallbacks
  │
  ▼
[11] reporting              tables / figures / frozen spec / reports/final_report.md
```

**One command for the full path:**

```bash
./scripts/run_pipeline.sh
# equivalent:
python -m src.pipeline --config configs/project_config.yaml
```

---

## 5. Repository map & code logic

```text
explainable-btcirt-lob/
├── configs/                 YAML: project + per-study overrides
├── src/
│   ├── pipeline.py          Orchestrator (Studies A/B/C)
│   ├── data_loader.py       Chunked CSV load + BTCIRT filter
│   ├── data_validation.py   Book integrity audits
│   ├── preprocessing.py     Cleaning; refuses dense grid when gaps are large
│   ├── delay_audit.py       Observation-gap statistics
│   ├── feature_engineering.py
│   ├── trade_deduplication.py
│   ├── labels/              Study A / B / C label builders
│   ├── splitting/           Chronological + purge + walk-forward
│   ├── models/              Baselines, logistic, XGB, CatBoost, calibration, two-stage
│   ├── evaluation/          Metrics, bootstrap, financial sanity
│   ├── explainability/      SHAP, grouped SHAP, permutation, ablation, interactions
│   ├── visualization.py     Figures
│   └── reporting.py         Report helpers (run summaries → reports/)
├── docs/                    Design, formulas, validation, limitations
├── tests/                   46 unit tests (units, labels, purge, no-leakage)
├── scripts/                 run_pipeline / study_a|b|c / tests
├── reports/
│   ├── archive/pre_research_redesign/   frozen v1 outputs
│   ├── tables/              CSV results
│   ├── metrics/             JSON metrics
│   ├── figures/             PNG figures
│   ├── models/              frozen_model_specification.json
│   └── final_report.md
└── notebooks/               Thin wrappers that import from src/
```

### Logic highlights by module

| Module | Core logic |
|---|---|
| `data_loader` | Normalize `exchange.lower()`, `symbol.upper()`; keep only `nobitex` + `BTCIRT`; raise if empty |
| `preprocessing` | Sort time; keep **last** duplicate timestamp; skip 5s alignment when median gap ≫ 5s |
| `labels/next_observation` | `target = t+1`; return in bps; hybrid ε from **train only** |
| `labels/next_price_change` | Forward scan to first mid ≠ current; optional one-sample-per-run |
| `labels/strict_horizon` | Match only inside `[lower, upper]`; minimize `|delay − h|` |
| `splitting/purged_split` | Drop rows whose `target_timestamp` crosses the next split boundary |
| `trade_deduplication` | Signature `(price, qty)` → `is_new_trade`; repeated snapshots count as 0 |
| `feature_engineering` | Depth / OBI / WOBI / microprice / snapshot OFI proxy / 120–1200s windows / event lags |
| `models/xgboost_model` | Randomized search on **validation Macro F1**; never on development_test |
| `explainability/*` | TreeSHAP; on interaction failure → dependence fallbacks + JSON status |

Notebooks **must not** re-implement formulas; they import `src/`.

---

## 6. Dataset & filtering

- **File:** `data/raw/market_data_clean_nobitex.csv` (**not committed** — place locally)
- **Exchange / symbol:** `nobitex` / `BTCIRT`
- **LOB:** 8 ask + 8 bid price/qty levels
- **Trades:** `last_trade_price`, `last_trade_qty` (often repeated across snapshots)
- **JSON columns** (`data`, `asks`, `bids`, `last_trade`): unused when flattened columns exist

```python
df["exchange"] = df["exchange"].astype(str).str.strip().str.lower()
df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
df = df[(df["exchange"] == "nobitex") & (df["symbol"] == "BTCIRT")]
```

---

## 7. Data-quality & observation-gap audit

### Quality checks

Missing / infinite values · zero/negative prices & quantities · crossed books (`a1 ≤ b1`) · locked books · invalid ask/bid monotonicity · abnormal spreads · timestamp parse failures · exact & timestamp duplicates.

### Gap reality (this run)

| Segment | Median gap (s) | % gaps > 60s | % gaps > 120s |
|---|---|---|---|
| Overall | **69.35** | **100%** | 18.9% |
| Train | 68.88 | 100% | 1.6% |
| Validation | 71.10 | 100% | 45.3% |
| Development test | **188.88** | 100% | **100%** |

**Consequence:** every gap exceeds 10s and 30s → Study C 10s/30s eligibility = **0**. A 60s window (40–80s) matches many train/val snapshots whose next print is ~70s, but the late-period test gaps (~3 minutes) fall **outside** the 80s upper bound after purging.

Tables: `reports/tables/observation_gap_summary.csv`, `observation_gap_by_date.csv`, `observation_gap_by_hour.csv`.

### Visual evidence

| Observation gaps (log scale) | Snapshots by date |
|:---:|:---:|
| <img src="reports/figures/data_quality/observation_gap_hist_log.png" alt="Observation gap histogram (log)" width="100%"/> | <img src="reports/figures/data_quality/snapshots_by_date.png" alt="Snapshots by date" width="100%"/> |

| Best bid / ask window | Relative spread distribution |
|:---:|:---:|
| <img src="reports/figures/data_quality/best_bid_ask_window.png" alt="Best bid ask window" width="100%"/> | <img src="reports/figures/data_quality/relative_spread_distribution.png" alt="Relative spread distribution" width="100%"/> |

<p align="center">
  <img src="reports/figures/data_quality/depth_profile_example.png" alt="LOB depth profile example" width="70%"/>
</p>

<p align="center"><em>Example depth profile — liquidity is concentrated near the touch.</em></p>

---

## 8. Mathematical core

Formulas use GitHub-flavored LaTeX (`$$ ... $$`). Full catalog: [`docs/FORMULAS.md`](docs/FORMULAS.md).

### Mid-price, spread, relative spread

$$
m_t = \frac{a_{1,t} + b_{1,t}}{2}, \qquad
S_t = a_{1,t} - b_{1,t}, \qquad
\mathrm{RelSpreadBps}_t = 10^{4}\,\frac{S_t}{m_t}
$$

### Log return in basis points

Unit-tested so $10^{4}\log(100.01/100)\approx 1$:

$$
r = 10^{4}\,\log\left(\frac{m_{\mathrm{future}}}{m_{\mathrm{current}}}\right)
$$

### Order-book imbalance (depth $k$)

$$
D^{a}_{k,t}=\sum_{i=1}^{k} q^{a}_{i,t}, \qquad
D^{b}_{k,t}=\sum_{i=1}^{k} q^{b}_{i,t}
$$

$$
\mathrm{OBI}_{k,t}=\frac{D^{b}_{k,t}-D^{a}_{k,t}}{D^{b}_{k,t}+D^{a}_{k,t}+\delta}
$$

where $\delta$ is a tiny constant for numerical stability (e.g. $10^{-12}$).

### Weighted OBI

Weights $w_i=e^{-\lambda(i-1)}$ with default $\lambda=0.5$:

$$
\mathrm{WOBI}_t=
\frac{\sum_{i} w_i q^{b}_{i,t}-\sum_{i} w_i q^{a}_{i,t}}
{\sum_{i} w_i q^{b}_{i,t}+\sum_{i} w_i q^{a}_{i,t}+\delta}
$$

### Snapshot OFI proxy (not event-level OFI)

$$
e^{b}_{t}=
\mathbf{1}_{\{b_t \ge b_{t-1}\}}\,q^{b}_{t}
-
\mathbf{1}_{\{b_t \le b_{t-1}\}}\,q^{b}_{t-1}
$$

(and similarly for the ask side with opposite inequalities)

$$
\mathrm{OFIProxy}_t = e^{b}_{t} - e^{a}_{t}
$$

### Tick size

Empirical mode of positive quote diffs = **10 IRT**. In bps vs huge BTCIRT mids this is $\sim 10^{-6}$ bps, so hybrid $\varepsilon$ relies on return quantiles / spreads, not ticks.

---

## 9. Label construction

### Study A

Next snapshot return → classes with train-only hybrid $\varepsilon$:

$$
\varepsilon=
\max
\bigl(
\varepsilon_{Q},\;
\varepsilon_{\mathrm{tick}},\;
0.5\times \mathrm{MedianSpreadBps}_{\mathrm{train}}
\bigr)
$$

$$
y_t=
\begin{cases}
\mathrm{DOWN} & \text{if } r_t < -\varepsilon \\[4pt]
\mathrm{STABLE} & \text{if } |r_t| \le \varepsilon \\[4pt]
\mathrm{UP} & \text{if } r_t > \varepsilon
\end{cases}
$$

**This run:** $\varepsilon = 3.664$ bps · method = `hybrid` · median delay $\approx 69.35$ s.

**Class mix (development_test):** DOWN **27.3%** · STABLE **46.3%** · UP **26.4%**.

| Study A class mix by split | Next-observation returns vs ε |
|:---:|:---:|
| <img src="reports/figures/labels/study_a_class_distribution.png" alt="Study A class distribution" width="100%"/> | <img src="reports/figures/labels/next_observation_return_epsilon.png" alt="Next observation return with epsilon" width="100%"/> |

| OBI distributions | OBI vs next return |
|:---:|:---:|
| <img src="reports/figures/labels/obi_distributions.png" alt="OBI distributions" width="100%"/> | <img src="reports/figures/labels/obi_vs_next_return.png" alt="OBI vs next return" width="100%"/> |

### Study B

First future mid $\neq$ current → UP / DOWN.  
Primary sample = last snapshot of each unchanged price run (avoids flooding one future event).

### Study C

Keep a future snapshot $s$ only when the delay is inside the strict window:

$$
\mathrm{Lower}_h \le t_s - t_t \le \mathrm{Upper}_h
$$

Defaults: $10\mathrm{s}\rightarrow[5,15]$, $30\mathrm{s}\rightarrow[20,40]$, $60\mathrm{s}\rightarrow[40,80]$.  
Among matches, minimize $|t_s-t_t-h|$. Store `actual_delay_seconds` and `horizon_error_seconds`.

---

## 10. Feature engineering

### Families

Price history · Static liquidity / depth / imbalance · Dynamic liquidity / imbalance · Snapshot OFI proxy · Volatility · Trade activity (corrected) · Time · Data-collection metadata (`observation_gap_seconds`, **excluded** from primary).

### Mandatory feature sets

| Set | Contents |
|---|---|
| `price_only` | Returns + volatility |
| `static_lob` / `dynamic_lob` / `lob_full` | Book structure & dynamics |
| **`full_no_trade`** | Primary model |
| `full_with_trade` | + corrected trades |
| `full_no_time` | Drop cyclical hour/dow |

Dictionary: `reports/tables/feature_dictionary.csv` · `docs/FEATURE_DICTIONARY.md`.

---

## 11. Temporal validation & leakage control

1. Split by **complete calendar dates** (60% / 20% / 20%).
2. Latest block = **`development_test`** — already inspected in prior analysis → **not a pristine holdout**.
3. **Target-timestamp purging:** a training row is kept only if `target_timestamp < validation_start` (same idea for val vs test).
4. Scalers, ε, class weights, and hyperparameter search fit on **training / validation only**.
5. Frozen spec written to `reports/models/frozen_model_specification.json`.
6. **Final independent future holdout (≥7–14 new days) is still pending.**

Tests: `tests/test_purging.py`, `tests/test_no_leakage.py`.

---

## 12. Models & hyperparameters

### Baselines

Majority · stratified random · previous mid-direction · OBI-5 threshold rule (tuned on train).

### Primary: XGBoost

- Study A: `multi:softprob` (3-class)
- Study B: `binary:logistic`
- Search objective: **validation Macro F1**
- Early stopping on validation
- CPU `tree_method=hist`

**Best Study A params (this run):**

```text
max_depth=7, learning_rate=0.02, n_estimators=1500,
min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
gamma=0, reg_alpha=1, reg_lambda=5, max_delta_step=5,
random_state=42
```

**Best Study B params:**

```text
max_depth=6, learning_rate=0.02, n_estimators=500,
min_child_weight=5, subsample=0.8, colsample_bytree=0.5,
gamma=1, reg_alpha=0.1, reg_lambda=20, max_delta_step=5
```

### Challenger: CatBoost

Study A best (this run): `depth=7`, `learning_rate=0.1`, `iterations=750`, `l2_leaf_reg=10`, `border_count=64`, `subsample=1.0` (Bernoulli bootstrap; `bagging_temperature` removed — incompatible with Bernoulli).

Details: [`docs/MODEL_SPECIFICATION.md`](docs/MODEL_SPECIFICATION.md).

---

## 13. Full results (this run)

### 13.1 Study A — model comparison (development_test)

| Model | Macro F1 | Balanced Acc | Log Loss | MCC |
|---|---|---|---|---|
| Majority | 0.211 | 0.333 | — | 0.000 |
| Previous direction | 0.377 | 0.377 | — | 0.083 |
| OBI rule | 0.385 | 0.384 | — | 0.070 |
| XGBoost | **0.470** | **0.484** | **1.029** | **0.218** |
| CatBoost | **0.484** | **0.500** | **1.021** | **0.239** |

XGBoost still used as the **frozen primary** for SHAP / bootstrap continuity; CatBoost is a strong challenger.

<p align="center">
  <img src="reports/figures/models/model_comparison_macro_f1.png" alt="Model comparison Macro F1" width="75%"/>
</p>

| Normalized confusion (XGBoost) | Per-class precision / recall / F1 |
|:---:|:---:|
| <img src="reports/figures/models/xgb_normalized_confusion_test.png" alt="XGBoost confusion matrix" width="100%"/> | <img src="reports/figures/models/xgb_per_class_prf.png" alt="Per-class PRF" width="100%"/> |

| ROC (one-vs-rest) | Precision–recall (one-vs-rest) |
|:---:|:---:|
| <img src="reports/figures/models/xgb_roc_ovr.png" alt="XGBoost ROC OvR" width="100%"/> | <img src="reports/figures/models/xgb_pr_ovr.png" alt="XGBoost PR OvR" width="100%"/> |

| Calibration | Train / validation loss |
|:---:|:---:|
| <img src="reports/figures/models/xgb_calibration.png" alt="XGBoost calibration" width="100%"/> | <img src="reports/figures/models/xgb_train_val_loss.png" alt="Train val loss" width="100%"/> |

| Performance by date | Performance by hour |
|:---:|:---:|
| <img src="reports/figures/models/performance_by_date.png" alt="Performance by date" width="100%"/> | <img src="reports/figures/models/performance_by_hour.png" alt="Performance by hour" width="100%"/> |

### 13.2 Feature-set comparison

| Experiment | #feats | Val Macro F1 | Test Macro F1 |
|---|---|---|---|
| price_only | 12 | 0.367 | 0.417 |
| static_lob | 34 | 0.376 | 0.404 |
| dynamic_lob | 30 | 0.349 | 0.420 |
| lob_full | 64 | 0.374 | 0.456 |
| **full_no_trade** | **80** | **0.384** | **0.464** |
| full_with_trade | 89 | 0.397 | 0.463 |
| full_no_time | 76 | 0.379 | 0.459 |

<p align="center">
  <img src="reports/figures/ablation/feature_set_comparison.png" alt="Feature set comparison" width="80%"/>
</p>

<p align="center"><em>LOB features lift Macro F1 over price-only; corrected trades add essentially nothing.</em></p>

### 13.3 Study B

Binary next-change direction Macro F1 ≈ **0.614** on development_test (val ≈ 0.616). Easier task (no STABLE class; nearly balanced UP/DOWN).

### 13.4 Study C

Honest outcome: **10s/30s impossible** under observed gaps; **60s** has many eligible rows historically concentrated before the sparse test regime, so purged development_test is empty → pilot descriptive stats only.

### 13.5 Financial sanity (exploratory — not a strategy)

Long-only if predict UP (next mid return as proxy):

| Scenario | Fees+slip (bps) | Mean net (bps) |
|---|---|---|
| Zero cost | 0 | +0.92 |
| Low cost | 10 | **−2.35** |
| Moderate cost | 25 | **−7.25** |

> Sparse snapshots **do not** support a realistic HFT execution backtest. Prediction ≠ profitability.

---

## 14. Explainability

We require **agreement** across methods before calling a feature “strongly supported”.
SHAP answers *how the trained model uses features for a prediction*, **not** whether a
feature causes the mid-price to move in the market.

### How to read this section

| Plot / table category | What it answers | What it does *not* say |
|---|---|---|
| **Global importance bar** | Which features move predictions the most *on average* | Direction of effect; causality |
| **Feature families** | Which microstructure *families* dominate attributions | That every member of the family is useful |
| **Permutation importance** | Which features hurt Macro F1 when shuffled on held-out data | Local story for one snapshot |
| **Beeswarm (per class)** | Full distribution of attributions for DOWN / STABLE / UP | A single “typical” row |
| **Dependence** | How SHAP for feature $X$ changes as $X$ varies (often colored by a second feature) | Pure partial effect free of collinearity |
| **Local waterfalls** | Step-by-step attribution for one illustrative prediction | Population-level ranking |
| **Interactions** | Whether features co-contribute in TreeSHAP (or dependence fallbacks) | Causal interaction in the market |

### Top TreeSHAP features (Study A)

**What this table / bar chart says:** mean absolute SHAP over the development-test SHAP sample.
Larger mean \|SHAP\| ⇒ the feature tends to push the model’s class scores more strongly across
snapshots. Rankings are class-averaged for the 3-class model.

| Rank | Feature | Mean \|SHAP\| | Family |
|---|---|---|---|
| 1 | `bid_distance_3_bps` | 0.0595 | Static imbalance |
| 2 | `ask_distance_2_bps` | 0.0454 | Static imbalance |
| 3 | `ask_distance_3_bps` | 0.0345 | Static imbalance |
| 4 | `volatility_120s` | 0.0316 | Volatility |
| 5 | `volatility_300s` | 0.0304 | Volatility |
| 6 | `bid_distance_2_bps` | 0.0304 | Static imbalance |
| 7 | `spread_std_300s` | 0.0203 | Dynamic liquidity |
| 8 | `log_ask_depth_1` | 0.0191 | Static depth |
| 9 | `obi_1` | 0.0179 | Static imbalance |
| 10 | `hour_cos` | 0.0163 | Time |

<p align="center">
  <img src="reports/figures/shap/shap_global_importance.png" alt="Global SHAP importance" width="80%"/>
</p>

<p align="center"><em>Global bar plot — average magnitude of attribution; does not encode sign (UP vs DOWN).</em></p>

### Top feature families (sum of mean \|SHAP\|)

**What this says:** we sum mean \|SHAP\| within each engineered family (static imbalance,
volatility, depth, …). It answers *which family of LOB ideas the model leans on*, not which
single column is “best.”

1. **Static imbalance**  
2. Volatility  
3. Static depth  
4. Snapshot OFI proxy  
5. Static liquidity  

### Permutation importance (top)

**What this says:** features are randomly shuffled within the development-test set (respecting
day blocks where configured); the drop in Macro F1 measures *predictive usefulness under the
evaluation metric*. Agreement with TreeSHAP is stronger evidence than either method alone.

1. `bid_distance_3_bps`  
2. `ask_distance_2_bps`  
3. `relative_spread_bps`  
4. `ask_distance_3_bps`  
5. `hour_cos`  

Distance / imbalance features appear in **both** SHAP and permutation — stronger evidence than `hour_cos` alone.

### Beeswarm plots (class-conditional)

**What beeswarm plots say:** for each class (DOWN / STABLE / UP), every point is one snapshot.
- **Vertical axis:** features ranked by mean \|SHAP\| for that class  
- **Horizontal axis:** SHAP value (left = pushes *against* that class; right = pushes *for* it)  
- **Color:** raw feature value (typically blue = low, red = high)

Use them to see *patterns*: e.g. high imbalance features clustering on the positive side for UP,
or volatility spreading attributions for STABLE. Because Study A is multiclass, we plot
**one beeswarm per class** so attributions are not mixed across DOWN/STABLE/UP.

| DOWN | STABLE | UP |
|:---:|:---:|:---:|
| <img src="reports/figures/shap/shap_beeswarm_down.png" alt="SHAP beeswarm DOWN" width="100%"/> | <img src="reports/figures/shap/shap_beeswarm_stable.png" alt="SHAP beeswarm STABLE" width="100%"/> | <img src="reports/figures/shap/shap_beeswarm_up.png" alt="SHAP beeswarm UP" width="100%"/> |

<p align="center"><em>Class-conditional beeswarms — distribution of attributions, not a single summary number.</em></p>

### Dependence plots

**What dependence plots say:** for a chosen feature $X$, each point is a snapshot plotted as
$(X,\ \mathrm{SHAP}_X)$ for the **UP** class contribution (when multiclass). Color usually
marks an automatically chosen interacting feature. They reveal *how* the model’s attribution
changes as $X$ moves (nonlinear / threshold behavior) and hint at pairwise co-dependence.

They do **not** isolate a causal partial effect: correlated LOB features can share credit.

| OBI (depth 5) | Weighted OBI |
|:---:|:---:|
| <img src="reports/figures/shap/shap_dependence_obi_5.png" alt="SHAP dependence OBI5" width="100%"/> | <img src="reports/figures/shap/shap_dependence_weighted_obi.png" alt="SHAP dependence weighted OBI" width="100%"/> |

| Relative spread | Microprice edge |
|:---:|:---:|
| <img src="reports/figures/shap/shap_dependence_relative_spread_bps.png" alt="SHAP dependence spread" width="100%"/> | <img src="reports/figures/shap/shap_dependence_microprice_edge_bps.png" alt="SHAP dependence microprice" width="100%"/> |

| Normalized snapshot OFI (300s) | Volatility (300s) |
|:---:|:---:|
| <img src="reports/figures/shap/shap_dependence_normalized_snapshot_ofi_300s.png" alt="SHAP dependence OFI" width="100%"/> | <img src="reports/figures/shap/shap_dependence_volatility_300s.png" alt="SHAP dependence volatility" width="100%"/> |

<p align="center"><em>Dependence plots — functional shape of attribution vs feature value (UP-class SHAP).</em></p>

### Local waterfalls (illustrative cases)

**What waterfall plots say:** for **one** snapshot, starting from the model’s expected score
(base value), each bar shows how a feature pushes the score up or down until the final
prediction for the highlighted class. They are case studies, not rankings.

We pick five illustrative development-test cases:

| Case | What it is meant to show |
|---|---|
| **Correct high-conf UP** | Confident correct UP — which LOB features drove the UP score |
| **Correct high-conf DOWN** | Same for DOWN |
| **Correct high-conf STABLE** | Same for STABLE (often spread / low-move cues) |
| **Incorrect high-conf** | Confident but wrong — where the model “trusted” misleading features |
| **Low-conf borderline** | Near-tie probabilities — small opposing attributions |

| Correct high-conf UP | Correct high-conf DOWN |
|:---:|:---:|
| <img src="reports/figures/shap/shap_waterfall_correct_high_conf_UP.png" alt="Waterfall correct UP" width="100%"/> | <img src="reports/figures/shap/shap_waterfall_correct_high_conf_DOWN.png" alt="Waterfall correct DOWN" width="100%"/> |

| Correct high-conf STABLE | Incorrect high-conf |
|:---:|:---:|
| <img src="reports/figures/shap/shap_waterfall_correct_high_conf_STABLE.png" alt="Waterfall correct STABLE" width="100%"/> | <img src="reports/figures/shap/shap_waterfall_incorrect_high_conf.png" alt="Waterfall incorrect" width="100%"/> |

<p align="center">
  <img src="reports/figures/shap/shap_waterfall_low_conf_borderline.png" alt="Waterfall borderline" width="70%"/>
</p>

<p align="center"><em>Local waterfalls — per-snapshot accounting of the model’s score; not causal market effects.</em></p>

### SHAP interactions

**What this category says:** TreeSHAP can estimate pairwise *interaction values* (extra
attribution when two features act together beyond main effects). We reduce the multiclass
tensor to a 2-D mean-absolute interaction matrix when possible and save
`reports/tables/shap_interactions_native.csv`. If that fails, we log the failure and fall back
to colored dependence plots under `reports/figures/shap/` — **never** fabricated interaction
numbers.

**Latest run status:** see `reports/metrics/shap_interaction_status.json`.

Regenerate without full retuning: `python scripts/regenerate_shap_plots.py`

---

## 15. Challenges we hit

This section is intentional — sharing failures is what makes the project credible.

| Challenge | What happened | Resolution |
|---|---|---|
| **Misleading fixed horizons** | 10/30/60s often mapped to the *same* next snapshot | Redesign into Studies A/B/C; store actual delays |
| **Sparse gaps** | Median ~69s; test ~189s; 100% of gaps > 60s | No 5s forward-fill; longer feature windows (120–1200s) |
| **Study C empty horizons** | 0 eligible rows for 10s & 30s | Label as underpowered pilot; do not invent scores |
| **Study C 60s purge** | Eligible overall, but development_test left with 0 after target purge | Descriptive-only; no robust model claim |
| **Repeated last trades** | Same `last_trade_*` across many snapshots | `trade_deduplication.py`; exclude from primary |
| **Tiny tick in bps** | Tick=10 IRT → ~1e−6 bps vs huge mid | Hybrid ε uses quantiles/spreads |
| **CatBoost API** | `bagging_temperature` invalid with Bernoulli bootstrap | Dropped temperature; keep Bernoulli + subsample |
| **SHAP interactions** | Multiclass tensor needs careful 2-D reduction | Reduce over samples/classes or use dependence fallbacks |
| **macOS XGBoost** | Needs `libomp` | `.libs/libomp.dylib` + `DYLD_LIBRARY_PATH` in scripts |
| **Development-test contamination** | Prior analysis already looked at late dates | Renamed; require future holdout |
| **Class imbalance** | STABLE dominates under hybrid ε | Macro F1 primary; balanced sample weights |

Archived v1 metrics live in `reports/archive/pre_research_redesign/` and must **not** be mixed with v2 claims.

---

## 16. What we do *not* claim

- True high-frequency forecasting from these sparse snapshots  
- Exact 30-second prediction when the next print is minutes away  
- Causal effects from SHAP  
- Trading profitability / production alpha  
- True event-level OFI  
- Generalization to all crypto markets  
- That CatBoost “wins” without multi-fold statistical comparison  
- That the development_test is a final holdout  

---

## 17. Reproduction

```bash
git clone git@github.com:OveysSayad/explainable-btcirt-lob.git
cd explainable-btcirt-lob

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Place the raw CSV:
#   data/raw/market_data_clean_nobitex.csv

# macOS XGBoost OpenMP:
#   brew install libomp
#   mkdir -p .libs && ln -sf "$(brew --prefix libomp)/lib/libomp.dylib" .libs/libomp.dylib
export DYLD_LIBRARY_PATH="$(pwd)/.libs:${DYLD_LIBRARY_PATH}"

./scripts/run_pipeline.sh
pytest -q
```

Makefile targets: `make test`, `make all`, `make study-a`, `make study-b`, `make study-c`.

Smoke (fewer trials): `python -m src.pipeline --config configs/smoke_config.yaml`.

---

## 18. Outputs catalog

| Path | Contents |
|---|---|
| `reports/final_report.md` | Paper-style write-up |
| `reports/tables/*.csv` | Metrics, gaps, SHAP, ablations, predictions metadata |
| `reports/ARTIFACT_MANIFEST.md` | Canonical vs archive artifact map |
| `reports/metrics/*.json` | Environment, bootstrap, study metrics, interaction status |
| `reports/figures/` | Data quality, labels, models, SHAP, ablation (embedded in this README) |
| `reports/models/frozen_model_specification.json` | Frozen Study A spec |
| `models/study_a/`, `models/study_b/` | Trained model artifacts |

Figure layout:

```text
reports/figures/
├── data_quality/   # gaps, mid, spread, depth, coverage
├── labels/         # class mix, returns vs ε, OBI
├── models/         # comparison, confusion, ROC/PR, calibration
├── shap/           # beeswarm, dependence, waterfalls
├── ablation/       # feature-set comparison
└── robustness/     # reserved
```
| `reports/archive/pre_research_redesign/` | Pre-redesign freeze |
| `docs/` | Design, formulas, validation, dense-collection plan |

---

## 19. Limitations & future work

See [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

**Highest-priority next steps**

1. Collect **dense** WebSocket LOB + trades (`docs/DENSE_DATA_COLLECTION_DESIGN.md`)  
2. Evaluate the **frozen** model on **≥7–14 new days** never used in analysis  
3. Complete nested walk-forward *scores* (fold Macro F1 distribution), not only fold calendars  
4. Stronger time-feature artifact tests (gap/hour confounds)

---

## 20. Research integrity

- No fabricated metrics, figures, API responses, or SHAP interactions  
- Failed / underpowered components remain documented  
- Prior results archived, not silently overwritten without copy  
- Primary metric Macro F1 chosen for class imbalance — not to hide weak DOWN/UP performance  
- **Final independent holdout evaluation pending**

---

## Citation

```bibtex
@software{explainable_btcirt_lob,
  title  = {Explainable ML for BTCIRT Limit Order Book Dynamics},
  author = {Sayad, Oveys},
  year   = {2026},
  url    = {https://github.com/OveysSayad/explainable-btcirt-lob},
  note   = {Multi-study redesign under sparse Nobitex snapshots}
}
```

## Disclaimer

Research and education only. **Not investment advice.** Cryptocurrency trading involves substantial risk of loss.

---

### Bottom line for readers & recruiters

This repo demonstrates **research engineering**: questioning a bad horizon design, quantifying sparsity, preventing leakage, comparing models fairly, and explaining predictions without overselling. That is rarer — and more valuable — than another notebook that reports a high F1 on shuffled LOB rows.

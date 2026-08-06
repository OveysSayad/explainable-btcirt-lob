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
| **Honest failure logs** | SHAP interactions & Study C underpower documented, not hidden |

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

---

## 8. Mathematical core

Mid, spread, relative spread (bps):

\[
m_t=\frac{a_{1,t}+b_{1,t}}{2},\quad
S_t=a_{1,t}-b_{1,t},\quad
\text{RelSpreadBps}_t=10^{4}\frac{S_t}{m_t}
\]

Log return (bps) — unit-tested so \(10^{4}\log(100.01/100)\approx 1\):

\[
r=10^{4}\log\left(\frac{m_{\text{future}}}{m_{\text{current}}}\right)
\]

Order-book imbalance at depth \(k\):

\[
\text{OBI}_{k}=\frac{D^{b}_{k}-D^{a}_{k}}{D^{b}_{k}+D^{a}_{k}+\delta}
\]

Weighted OBI with \(w_i=e^{-\lambda(i-1)}\) (default \(\lambda=0.5\)).

**Snapshot OFI proxy** (Cont-style between snapshots — **not** event OFI):

\[
e^{b}_{t}=\mathbf{1}_{b_t\ge b_{t-1}}q^{b}_{t}-\mathbf{1}_{b_t\le b_{t-1}}q^{b}_{t-1}
\]

Full catalog: [`docs/FORMULAS.md`](docs/FORMULAS.md).

### Tick size

Empirical mode of positive quote diffs = **10 IRT**. In bps vs huge BTCIRT mids this is ~**10⁻⁶ bps**, so hybrid ε relies on return quantiles / spreads, not ticks.

---

## 9. Label construction

### Study A

Next snapshot return → classes with train-only hybrid ε:

\[
\varepsilon=\max(\varepsilon_Q,\varepsilon_{\text{tick}},0.5\cdot\text{MedianSpreadBps}_{\text{train}})
\]

\[
y=\begin{cases}
\text{DOWN}&r<-\varepsilon\\
\text{STABLE}&|r|\le\varepsilon\\
\text{UP}&r>\varepsilon
\end{cases}
\]

This run: \(\varepsilon=3.664\) bps · method `hybrid` · median delay ≈ 69.35 s.

Class mix (development_test): DOWN 27.3% · STABLE 46.3% · UP 26.4%.

### Study B

First future mid ≠ current → UP/DOWN. Primary = last snapshot of each unchanged run (avoids flooding one future event).

### Study C

Match future \(s\) only if \(\text{Lower}_h \le t_s-t_t \le \text{Upper}_h\). Store `actual_delay_seconds` and `horizon_error_seconds`.

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

### Top TreeSHAP features (Study A)

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

### Top feature families (sum of mean \|SHAP\|)

1. **Static imbalance**  
2. Volatility  
3. Static depth  
4. Snapshot OFI proxy  
5. Static liquidity  

### Permutation importance (top)

1. `bid_distance_3_bps`  
2. `ask_distance_2_bps`  
3. `relative_spread_bps`  
4. `ask_distance_3_bps`  
5. `hour_cos`  

Distance / imbalance features appear in **both** SHAP and permutation — stronger evidence than `hour_cos` alone.

### SHAP interactions

Native multiclass interaction tensor could not be saved as a 2-D table (`shape=(80,80,3)`). Status: **`success=false`** in `reports/metrics/shap_interaction_status.json`. Dependence-plot **fallbacks** were generated instead. **No fabricated interaction values.**

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
| **SHAP interactions** | Multiclass 3-D tensor vs 2-D DataFrame | Log failure + dependence fallbacks |
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
| `reports/metrics/*.json` | Environment, bootstrap, study metrics, interaction status |
| `reports/figures/` | Data quality, models, SHAP, fallbacks |
| `reports/models/frozen_model_specification.json` | Frozen Study A spec |
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

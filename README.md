# Explainable Machine Learning for BTCIRT Limit Order Book Dynamics (Research Redesign)

## 1. Project title

**Explainable BTCIRT LOB** — multi-study mid-price movement research on Nobitex sparse snapshots.

## 2. Executive summary

This project predicts short-term **BTCIRT** mid-price *movement direction* from Nobitex limit
order book snapshots using chronological ML (XGBoost primary, CatBoost challenger) and
multi-method explainability (SHAP, permutation, ablation).

**Why the redesign:** the original fixed 10/30/60-second formulation was scientifically
misleading under sparse sampling (median gap ~1 minute). Horizons often resolved to the same
next snapshot. The redesign implements:

- **Study A** — next observed mid movement (primary, largest valid sample)
- **Study B** — next mid-*change* direction (binary)
- **Study C** — strict-horizon pilots with narrow delay windows
- **Study D** — dense data-collection design (docs only)


### Archived previous results (v1 — do not treat as current)

Prior fixed-horizon pipeline (archived under `reports/archive/pre_research_redesign/`):
XGBoost test Macro F1 ≈ 0.460; CatBoost ≈ 0.452; previous-direction ≈ 0.42.
Those numbers are historical context only.


## 3. Research questions

1. Study A: next observed mid movement predictability?
2. Study B: next change direction predictability?
3. Study C: is honest fixed-horizon prediction feasible on eligible subsets?
4. Incremental value of LOB vs price-only features?
5. Explanation stability vs out-of-sample feature value?

## 4. Dataset

- Source: `data/raw/market_data_clean_nobitex.csv` (not in git)
- Exchange / symbol: `nobitex` / `BTCIRT` (normalized before filter)
- Raw rows: **835000**
- BTCIRT: **39693** (4.753652694610778%)
- Dates: `2026-01-04 07:32:39.311165+00:00` → `2026-02-18 04:04:18.733462+00:00`
  (**44** unique dates)
- Eight LOB levels + last_trade fields; JSON blobs unused when flattened columns exist
- Duplicate timestamps: **keep last** after chronological sort

## 5. Data-quality audit

Pipeline writes `reports/metrics/data_quality.json` and
`reports/tables/data_quality_summary.csv`. Checks: missing/inf, zeros/negatives, crossed/locked
books, level ordering, spreads, timestamp failures, duplicates, gaps.

## 6. Why sparse snapshots matter

Gap overall: `{'segment': 'overall', 'n_gaps': 39692, 'mean': 97.63930823080217, 'median': 69.34880000000001, 'std': 1109.885429717366, 'min': 67.545392, 'max': 220979.615738, 'p01': 68.33015338, 'p05': 68.35757665, 'p25': 68.42043100000001, 'p75': 81.96646899999999, 'p90': 188.9266546, 'p95': 189.40090915, 'p99': 202.58638538999952, 'pct_above_10s': 100.0, 'pct_above_30s': 100.0, 'pct_above_60s': 100.0, 'pct_above_120s': 18.910611710168297, 'pct_above_300s': 0.03275219187745641, 'pct_above_600s': 0.015116396251133729}`.

- 10/30/60s labels previously overlapped because the next snapshot often served all horizons.
- Five-second forward-fill invents false density across minute-scale gaps.
- Default: **preserve native timestamps**; no 5s grid.

## 7. Mathematical notation

\(a_i, b_i\): ask/bid prices; \(q^a_i, q^b_i\): quantities; mid
\(m=(a_1+b_1)/2\); spread \(S=a_1-b_1\); returns in bps via
\(10^4\log(m_s/m_t)\). Full formulas: `docs/FORMULAS.md`.

## 8. Label construction

### Study A
Next snapshot return; hybrid ε from train quantiles / tick / half-median-spread.

### Study B
First future mid ≠ current; UP/DOWN; primary = one sample per price run.

### Study C
Strict windows only (10∈[5,15], 30∈[20,40], 60∈[40,80]); store actual delay & horizon error.

## 9. Feature engineering

Families: price history, static/dynamic liquidity & imbalance, snapshot OFI proxies, volatility,
corrected trades (optional), cyclical time, observation-gap metadata (excluded from primary).
See `docs/FEATURE_DICTIONARY.md` and `reports/tables/feature_dictionary.csv`.

## 10. Feature-set design

Mandatory: price_only, static_lob, dynamic_lob, lob_full, full_no_trade (primary),
full_with_trade, full_no_time.

## 11. Temporal splitting

60/20/20 by **complete dates**. Latest block = **development_test** (not pristine).
Target-timestamp purging. Nested walk-forward for honest selection. Final future holdout pending
(`reports/models/frozen_model_specification.json`).

## 12. Models

Majority, stratified random, previous direction, OBI rule, logistic regression, XGBoost,
CatBoost, optional two-stage. Specs: `docs/MODEL_SPECIFICATION.md`.

## 13. Hyperparameters

Search spaces in `configs/project_config.yaml`. Best params from this run appear in Results
and `reports/tables/best_hyperparameters.csv`. Optimize validation Macro F1; never development_test.

## 14. Evaluation metrics

Primary: **Macro F1** (class imbalance / STABLE mass). Also balanced accuracy, per-class PR/F1,
MCC, log loss, Brier, OvR ROC/PR-AUC, confusion matrices, day-level bootstrap CIs.


## 15. Results (this run)

- Pipeline completed: **True**
- Seed: **42**
- Primary features: **80**
- Study A ε: **3.664346280286074** bps (`hybrid`)
- Study A n (train/val/dev-test): **29568** / **5630** / **4492**
- Study A XGB Macro F1 (val / dev-test): **0.4163** / **0.4699**
- Study A CatBoost Macro F1 (dev-test): **0.4842**
- Study A best params: `{'max_depth': 7, 'learning_rate': 0.02, 'n_estimators': 1500, 'min_child_weight': 3.0, 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.0, 'reg_alpha': 1.0, 'reg_lambda': 5.0, 'max_delta_step': 5}`
- Bootstrap: `{'n_days': 10.0, 'metric': 'macro_f1', 'n_bootstrap': 1000, 'mean': 0.4696179798461459, 'median': 0.47013149016984834, 'std': 0.0068629304978602, 'ci_low': 0.45464647245438616, 'ci_high': 0.48121877311596895}`
- Study B: `{"meta": {"study": "B_next_price_change", "one_sample_per_price_run": true, "n_all_eligible": 39691, "n_primary_sample": 25704, "n_unique_target_sample": 25731, "class_counts_primary": {"0.0": 12830, "2.0": 12874}, "median_time_to_change": 69.5891595}, "xgb_test": {"n": 3529, "accuracy": 0.6149050722584302, "balanced_accuracy": 0.6144379241586229, "macro_f1": 0.6143977824405744, "weighted_f1": 0.6146633169546083, "macro_precision": 0.6147257394839154, "macro_recall": 0.6144379241586229, "mcc": 0.22916348290338268, "confusion_matrix": [[1149, 649], [710, 1021]], "normalized_confusion_matrix": [[0.6390433815350389, 0.3609566184649611], [0.4101675332177932, 0.5898324667822068]], "precision_DOWN": 0.6180742334588488, "recall_DOWN": 0.6390433815350389, "f1_DOWN": 0.6283839212469237, "support_DOWN": 1798, "precision_UP": 0.611377245508982, "recall_UP": 0.5898324667822068, "f1_UP": 0.6004116436342253, "support_UP": 1731, "log_loss": 0.6551300883293152, "brier_score": 0.23115167021751404, "roc_auc": 0.6595366570083326, "average_precision": 0.6412933104764478}, "best_params": {"max_depth": 6, "learning_rate": 0.02, "n_estimators": 500, "min_child_weight": 5.0, "subsample": 0.8, "colsample_bytree": 0.5, "gamma": 1.0, "reg_alpha": 0.1, "reg_lambda": 20.0, "max_delta_step": 5}, "n_primary": 25704}`
- Study C models: `{"10": {"status": "underpowered_pilot", "horizon": 10, "lower": 5.0, "upper": 15.0, "n_eligible": 0, "n_rejected": 39693, "eligibility_pct": 0.0, "unique_dates": 0, "pct_duplicated_target_timestamps": NaN, "delay_median": null, "delay_p95": null, "error_median": null, "epsilon_bps": null, "epsilon_method": "insufficient_train", "class_counts": {}, "underpowered": true, "label": "Pilot fixed-horizon analysis"}, "30": {"status": "underpowered_pilot", "horizon": 30, "lower": 20.0, "upper": 40.0, "n_eligible": 0, "n_rejected": 39693, "eligibility_pct": 0.0, "unique_dates": 0, "pct_duplicated_target_timestamps": NaN, "delay_median": null, "delay_p95": null, "error_median": null, "epsilon_bps": null, "epsilon_method": "insufficient_train", "class_counts": {}, "underpowered": true, "label": "Pilot fixed-horizon analysis"}, "60": {"status": "underpowered_after_purge", "n_train_purged": 26263, "n_val_purged": 3054, "n_development_test_purged": 0, "note": "Eligible rows exist overall, but target-timestamp purging left an empty or tiny development_test. Pilot fixed-horizon analysis \u2014 no robust model claim.", "horizon": 60, "lower": 40.0, "upper": 80.0, "n_eligible": 29318, "n_rejected": 10375, "eligibility_pct": 73.86189000579448, "unique_dates": 29, "pct_duplicated_target_timestamps": 0.0, "delay_median": 68.57127399999999, "delay_p95": 78.35378455, "error_median": 8.571273999999995, "epsilon_bps": 3.6778550816106304, "epsilon_method": "hybrid", "class_counts": {"1.0": 19727, "2.0`

Tables: `reports/tables/model_comparison.csv`, `feature_set_comparison.csv`,
`study_*_class_distribution.csv`, `horizon_overlap.csv`.


## 16. Incremental value

See `reports/tables/feature_set_comparison.csv` and `incremental_value.csv`.

## 17. Ablation analysis

Grouped removals / feature sets; choose families via validation, not test peeking.

## 18. Explainability

SHAP ≠ causality. Compare with permutation + ablation. Time features audited for collection
artifacts (`hour_cos` historically prominent).

### Top SHAP (this run)
| Feature | Mean |SHAP| |
|---------|-------------|
| `bid_distance_3_bps` | 0.059466 |
| `ask_distance_2_bps` | 0.045368 |
| `ask_distance_3_bps` | 0.034511 |
| `volatility_120s` | 0.031618 |
| `volatility_300s` | 0.030436 |
| `bid_distance_2_bps` | 0.030429 |
| `spread_std_300s` | 0.020276 |
| `volatility_1200s` | 0.019299 |
| `log_ask_depth_1` | 0.019054 |
| `obi_1` | 0.017860 |

## 19. SHAP interaction status

`{'attempted_method': 'shap.TreeExplainer.shap_interaction_values', 'success': True, 'package_versions': {'shap': '0.52.0', 'xgboost': '3.4.0', 'sklearn': '1.9.0', 'numpy': '2.4.6'}, 'error_message': 'ValueError: Must pass 2-d input. shape=(80, 80, 3)', 'fallback_method': 'dependence_plots_colored_by_second_feature', 'generated_outputs': ['native_shap_interactions', '/Users/oveyssayad/xai btcirt/explainable-btcirt-lob/reports/figures/shap/interaction_fallback_obi_5_x_relative_spread_bps.png', '/Users/oveyssayad/xai btcirt/explainable-btcirt-lob/reports/figures/shap/interaction_fallback_weighted_obi_x_log_bid_depth_5.png', '/Users/oveyssayad/xai btcirt/explainable-btcirt-lob/reports/figures/shap/interaction_fallback_snapshot_ofi_proxy_l1_x_log_bid_depth_1.png', '/Users/oveyssayad/xai btcirt/explainable-btcirt-lob/reports/figures/shap/interaction_fallback_microprice_edge_bps_x_volatility_300s.png', '/Users/oveyssayad/xai btcirt/explainable-btcirt-lob/reports/figures/shap/interaction_fallback_relative_spread_bps_x_volatility_300s.png'], 'traceback': 'Traceback (most recent call last):\n  File "/Users/oveyssayad/xai btcirt/explainable-btcirt-lob/src/explainability/interaction_fallbacks.py", line 63, in run_interaction_analysis\n    pd.DataFrame(mat, index=sample.columns, columns=sample.columns).to_csv(\n    ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/Users/oveyssayad/xai btcirt/explainable-btcirt-lob/.venv/lib/python3.14/site-packages/pandas/core/frame.py", line 814, in __init__\n    mgr = ndarray_to_mgr(\n        data,\n    ...<3 lines>...\n        copy=copy,\n    )\n  File "/Users/oveyssayad/xai btcirt/explainable-btcirt-lob/.venv/lib/python3.14/site-packages/pandas/core/internals/construction.py", line 277, in ndarray_to_mgr\n    values = _ensure_2d(values)\n  File "/Users/oveyssayad/xai btcirt/explainable-btcirt-lob/.venv/lib/python3.14/site-packages/pandas/core/internals/construction.py", line 549, in _ensure_2d\n    raise ValueError(f"Must pass 2-d input. shape={values.shape}")\nValueError: Must pass 2-d input. shape=(80, 80, 3)\n'}` — failures logged with package versions; fallbacks only; no fabricated interactions.
File: `reports/metrics/shap_interaction_status.json`.

## 20. Trade-feature problem

Repeated `last_trade_*` across snapshots can inflate intensity. Module
`src/trade_deduplication.py` builds signatures and corrected counts. Primary model excludes
trades unless walk-forward consistently favors them.

## 21. Time-of-day feature audit

Compare full_no_trade vs full_no_time; inspect observation counts/gaps by hour. Retain time
features only if stable across folds.

## 22. Financial sanity check

Long-only exploratory scenarios with assumed fees — not a trading claim.
Sparse snapshots ≠ realistic HFT backtest.

## 23. Limitations

`docs/LIMITATIONS.md` — sparsity, snapshot OFI, development_test contamination, single market,
short span, Study C underpower, non-causal SHAP, pending final holdout.

## 24. Reproduction

```bash
cd "/Users/oveyssayad/xai btcirt/explainable-btcirt-lob"
source .venv/bin/activate
# macOS XGBoost needs libomp:
export DYLD_LIBRARY_PATH="$(pwd)/.libs:${DYLD_LIBRARY_PATH}"
# or: brew install libomp
./scripts/run_pipeline.sh
pytest -q
```

Environment snapshot: `reports/metrics/environment.json`.

## 25. Project structure

`configs/`, `src/` (labels, splitting, models, evaluation, explainability), `docs/`,
`notebooks/`, `scripts/`, `reports/` (+ `archive/`), `tests/`, `models/`, `logs/`.

## 26. Future work

Dense WebSocket collection (`docs/DENSE_DATA_COLLECTION_DESIGN.md`) and a frozen-spec
independent holdout of ≥7–14 new days.

## 27. Research integrity

- No fabricated metrics, figures, or API responses
- Failed SHAP interactions remain documented
- Prior results archived under `reports/archive/pre_research_redesign/`
- **Final independent holdout evaluation pending**

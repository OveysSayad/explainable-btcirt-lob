# Explainable BTCIRT LOB — Research Redesign Final Report

## 1. Abstract

This report documents a redesigned research pipeline for Nobitex **BTCIRT** limit-order-book
snapshots. The redesign abandons a single misleading fixed-horizon formulation in favor of
three empirical studies: (A) next-observed mid movement, (B) next mid-change direction, and
(C) strict fixed-horizon pilots with narrow delay windows. Models are chronological,
target-timestamp-purged, and explained with SHAP, permutation importance, and feature-set
ablation. **No results are fabricated**; missing or underpowered analyses are labeled as such.

## 2. Introduction

Sparse LOB snapshots (median gap on the order of a minute) cannot support claims of exact
10/30/60-second high-frequency forecasting unless future observations actually fall near those
horizons. Prior pipeline versions matched distant snapshots into short horizons, creating
overlapping labels and overstated robustness. This redesign separates scientifically distinct
questions and stores actual delays for every label.

## 3. Research questions

1. **Study A:** Can the current LOB state predict the next *observed* mid-price movement?
2. **Study B:** Conditional on a mid change eventually occurring, can the book predict its direction?
3. **Study C:** On the subset with strict timing, is fixed-horizon prediction feasible (pilot)?
4. Do LOB features improve over price-history-only models?
5. Are explanations stable, and do SHAP rankings agree with ablation / permutation evidence?

## 4. Related methodological context

Classical microstructure (spread, depth, OBI, microprice) and Cont-style order-flow imbalance
motivate features. Snapshot OFI proxies are explicitly *not* event OFI. Tree ensembles + TreeSHAP
provide CPU-friendly nonlinear modeling with local/global attributions.

## 5. Data

- File: `data/raw/market_data_clean_nobitex.csv`
- Filter: `exchange=nobitex`, `symbol=BTCIRT` (normalized)
- Raw rows: **835000**
- BTCIRT rows: **39693** (4.753652694610778%)
- Range: `2026-01-04 07:32:39.311165+00:00` → `2026-02-18 04:04:18.733462+00:00`
- Unique dates: **44**
- Eight LOB levels; JSON columns unused when flattened fields are complete
- Duplicate timestamps: keep last (documented in preprocessing meta)

## 6. Data-quality audit

See `reports/metrics/data_quality.json` and `reports/tables/data_quality_summary.csv`.
Checks include missing/infinite values, zero/negative prices & quantities, crossed/locked books,
invalid ask/bid ordering, spreads, timestamp parse failures, and observation gaps.

## 7. Sparse-sampling problem

Overall gap summary (seconds): `{'segment': 'overall', 'n_gaps': 39692, 'mean': 97.63930823080217, 'median': 69.34880000000001, 'std': 1109.885429717366, 'min': 67.545392, 'max': 220979.615738, 'p01': 68.33015338, 'p05': 68.35757665, 'p25': 68.42043100000001, 'p75': 81.96646899999999, 'p90': 188.9266546, 'p95': 189.40090915, 'p99': 202.58638538999952, 'pct_above_10s': 100.0, 'pct_above_30s': 100.0, 'pct_above_60s': 100.0, 'pct_above_120s': 18.910611710168297, 'pct_above_300s': 0.03275219187745641, 'pct_above_600s': 0.015116396251133729}`.

Because median gaps are large relative to 5–60 seconds, the pipeline **does not** forward-fill
onto a 5-second grid. Short clock-time rolling windows are often empty; defaults use
120–1200s windows plus observation lags.

## 8. Study redesign

| Study | Label | Classes | Timing claim |
|-------|-------|---------|--------------|
| A | Next snapshot | DOWN/STABLE/UP | Next observation (actual delay stored) |
| B | First mid change | DOWN/UP | Eventual change (time-to-change stored) |
| C | Strict windows | DOWN/STABLE/UP | Only if delay ∈ configured band |
| D | Docs | — | Future dense collection design |

## 9. Feature engineering

Primary feature set: **full_no_trade** (price history + static/dynamic LOB + snapshot OFI proxies
+ volatility + optional time). Corrected trade features are evaluated separately. Formulas:
`docs/FORMULAS.md`. Dictionary: `reports/tables/feature_dictionary.csv`.

## 10. Label construction

- Study A hybrid ε (train-only): **3.664346280286074** bps
  (method `hybrid`)
- Study A labeled n≈**29568+5630+4492**
  (train/val/dev-test after purge)
- Study B primary samples: **25704**
- Study C: see `reports/metrics/study_c_label_meta.json` (often underpowered)

## 11. Models

Baselines (majority, stratified, previous direction, OBI rule), multinomial logistic regression,
XGBoost (primary), CatBoost (challenger). Optional two-stage hierarchy is implemented but not
forced as primary. Hyperparameters selected on validation Macro F1 only.

## 12. Validation

Date split 60/20/20; latest block = **development_test** (not pristine). Target-timestamp
purging. Day-level bootstrap. Frozen spec:
`reports/models/frozen_model_specification.json`.
**Final independent holdout evaluation pending.**

## 13. Results

Pipeline completed: **True**. Seed: **42**.

### Study A (development_test)

- XGBoost Macro F1: **0.4699**
- XGBoost Balanced Acc: **0.4837**
- XGBoost Log Loss: **1.0291**
- CatBoost Macro F1: **0.4842**
- Val Macro F1 (XGB): **0.4163**
- Best params: `{'max_depth': 7, 'learning_rate': 0.02, 'n_estimators': 1500, 'min_child_weight': 3.0, 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.0, 'reg_alpha': 1.0, 'reg_lambda': 5.0, 'max_delta_step': 5}`
- Bootstrap Macro F1 CI: `{'n_days': 10.0, 'metric': 'macro_f1', 'n_bootstrap': 1000, 'mean': 0.4696179798461459, 'median': 0.47013149016984834, 'std': 0.0068629304978602, 'ci_low': 0.45464647245438616, 'ci_high': 0.48121877311596895}`

### Study B

`{"xgb_test": {"n": 3529, "accuracy": 0.6149050722584302, "balanced_accuracy": 0.6144379241586229, "macro_f1": 0.6143977824405744, "weighted_f1": 0.6146633169546083, "macro_precision": 0.6147257394839154, "macro_recall": 0.6144379241586229, "mcc": 0.22916348290338268, "confusion_matrix": [[1149, 649], [710, 1021]], "normalized_confusion_matrix": [[0.6390433815350389, 0.3609566184649611], [0.4101675332177932, 0.5898324667822068]], "precision_DOWN": 0.6180742334588488, "recall_DOWN": 0.6390433815350389, "f1_DOWN": 0.6283839212469237, "support_DOWN": 1798, "precision_UP": 0.611377245508982, "recall_UP": 0.5898324667822068, "f1_UP": 0.6004116436342253, "support_UP": 1731, "log_loss": 0.6551300883293152, "brier_score": 0.23115167021751404, "roc_auc": 0.6595366570083326, "average_precision": 0.6412933104764478}, "best_params": {"max_depth": 6, "learning_rate": 0.02, "n_estimators": 500, "min_child_weight": 5.0, "subsample": 0.8, "colsample_bytree": 0.5, "gamma": 1.0, "reg_alpha": 0.1, "reg_lambda": 20.0, "max_delta_step": 5}, "n_primary": 25704, "meta_keys": ["study", "one_sample_per_price_run", "n_all_eligible", "n_primary_sample", "n_unique_target_sample", "class_counts_primary", "median_time_to_change"]}`

### Study C (pilot)

`{"10": {"status": "underpowered_pilot", "horizon": 10, "lower": 5.0, "upper": 15.0, "n_eligible": 0, "n_rejected": 39693, "eligibility_pct": 0.0, "unique_dates": 0, "pct_duplicated_target_timestamps": NaN, "delay_median": null, "delay_p95": null, "error_median": null, "epsilon_bps": null, "epsilon_method": "insufficient_train", "class_counts": {}, "underpowered": true, "label": "Pilot fixed-horizon analysis"}, "30": {"status": "underpowered_pilot", "horizon": 30, "lower": 20.0, "upper": 40.0, "n_eligible": 0, "n_rejected": 39693, "eligibility_pct": 0.0, "unique_dates": 0, "pct_duplicated_target_timestamps": NaN, "delay_median": null, "delay_p95": null, "error_median": null, "epsilon_bps": null, "epsilon_method": "insufficient_train", "class_counts": {}, "underpowered": true, "label": "Pilot fixed-horizon analysis"}, "60": {"status": "underpowered_after_purge", "n_train_purged": 26263, "n_val_purged": 3054, "n_development_test_purged": 0, "note": "Eligible rows exist overall, but target-timestamp purging left an empty or tiny development_test. Pilot fixed-horizon analysis \u2014 no robust model claim.", "horizon": 60, "lower": 40.0, "upper": 80.0, "n_eligible": 29318, "n_rejected": 10375, "eligibility_pct": 73.86189000579448, "unique_dates": 29, "pct_duplicated_target_timestamps": 0.0, "delay_median": 68.57127399999999, "delay_p95": 78.35378455, "error_median": 8.571273999999995, "epsilon_bps": 3.6778550816106304, "epsilon_method": "hybrid", "class_counts": {"1.0": 19727, "2.0": 4812, "0.0": 4779}, "underpowered": false, "label": "Strict fixed-horizon"}}`

Interpretation: compare models to baselines in `reports/tables/model_comparison.csv`. Do not
over-claim Study C if status is `underpowered_pilot`.

## 14. Incremental value of LOB data

| experiment      |   n_features |   train_seconds |   val_macro_f1 |   test_macro_f1 |   test_balanced_accuracy |   test_log_loss |
|:----------------|-------------:|----------------:|---------------:|----------------:|-------------------------:|----------------:|
| price_only      |           12 |        0.183216 |       0.367099 |        0.417327 |                 0.420496 |         1.08069 |
| static_lob      |           34 |        0.299289 |       0.376092 |        0.403953 |                 0.448314 |         1.07252 |
| dynamic_lob     |           30 |        0.219859 |       0.348684 |        0.419544 |                 0.426392 |         1.08285 |
| lob_full        |           64 |        0.401065 |       0.373665 |        0.455979 |                 0.473743 |         1.05503 |
| full_no_trade   |           80 |        0.501581 |       0.384475 |        0.463696 |                 0.479663 |         1.04636 |
| full_with_trade |           89 |        0.525221 |       0.396691 |        0.462761 |                 0.473771 |         1.04157 |
| full_no_time    |           76 |        0.466235 |       0.378889 |        0.459345 |                 0.475424 |         1.05096 |

Incremental LOB value ≈ MacroF1(full_no_trade) − MacroF1(price_only). Uncertainty: day bootstrap
where available. Prefer validation / fold evidence over a single development_test number.

## 15. Explainability

Top SHAP features (Study A sample):

- `bid_distance_3_bps`: mean |SHAP| = 0.059466
- `ask_distance_2_bps`: mean |SHAP| = 0.045368
- `ask_distance_3_bps`: mean |SHAP| = 0.034511
- `volatility_120s`: mean |SHAP| = 0.031618
- `volatility_300s`: mean |SHAP| = 0.030436
- `bid_distance_2_bps`: mean |SHAP| = 0.030429
- `spread_std_300s`: mean |SHAP| = 0.020276
- `volatility_1200s`: mean |SHAP| = 0.019299
- `log_ask_depth_1`: mean |SHAP| = 0.019054
- `obi_1`: mean |SHAP| = 0.017860

Grouped importance and permutation tables: `reports/tables/shap_*.csv`,
`permutation_importance.csv`. High SHAP without ablation/permutation support is treated as
unstable or potentially artifactual (e.g., time-of-day / collection patterns).

## 16. Ablation

See feature-set / ablation tables above. Families that hurt development_test when included
(historically: uncorrected trades) must not be forced into the primary model.

## 17. Robustness

Regime and hour analyses are generated when figures/tables exist under
`reports/figures/robustness/` and `reports/tables/regime_performance.csv`.
Gap regimes matter because collection density varies.

## 18. Strict-horizon pilot analysis

Horizon overlap:

|   horizon_a |   horizon_b |   n_common_current_observations |   pct_same_target_timestamp |   pct_same_label |   corr_future_return |
|------------:|------------:|--------------------------------:|----------------------------:|-----------------:|---------------------:|
|          10 |          30 |                               0 |                         nan |              nan |                  nan |
|          10 |          60 |                               0 |                         nan |              nan |                  nan |
|          30 |          60 |                               0 |                         nan |              nan |                  nan |

Identical performance across 10/30/60s is **not** robustness when target timestamps overlap.

## 19. Financial sanity check

Exploratory long-only diagnostic only (`reports/metrics/study_a_financial_sanity.json`).
**Sparse snapshots do not support a realistic HFT execution backtest.** Prediction ≠ profitability.

## 20. Limitations

See `docs/LIMITATIONS.md`. Snapshot sparsity, OFI proxy limits, development_test contamination,
single market, short date span, small strict-horizon samples, non-causal SHAP.

## 21. Conclusion

The redesign makes scientific claims match what the data can support: next-observation and
next-change predictability, with optional underpowered strict-horizon pilots. LOB incremental
value and explanation stability should be judged from the tables/figures produced by this run.

## 22. Future research

Dense WebSocket collection (`docs/DENSE_DATA_COLLECTION_DESIGN.md`), independent future holdout,
event-level OFI, and cost-aware decision evaluation.

---
SHAP interaction status: `{'attempted_method': 'shap.TreeExplainer.shap_interaction_values', 'success': True, 'package_versions': {'shap': '0.52.0', 'xgboost': '3.4.0', 'sklearn': '1.9.0', 'numpy': '2.4.6'}, 'error_message': None, 'fallback_method': None, 'generated_outputs': ['native_shap_interactions', 'reports/tables/shap_interactions_native.csv']}`
Generated from pipeline summary; archived prior results live in
`reports/archive/pre_research_redesign/`.

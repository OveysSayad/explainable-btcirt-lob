# Artifact Manifest — Canonical vs Archive

**Purpose:** Prevent confusion between redesign (v2) outputs and leftover pre-redesign (v1) files.

**Last cleanup:** 2026-08-06 (audit fixes)

---

## Canonical current artifacts (use these)

### Models
| Path | Description |
|------|-------------|
| `models/study_a/xgboost_study_a.joblib` | Primary Study A XGBoost (80 features) |
| `models/study_a/catboost_study_a.joblib` | Study A CatBoost challenger |
| `models/study_a/logistic_regression.joblib` | Study A logistic baseline |
| `models/study_b/xgboost_study_b.joblib` | Study B binary XGBoost |
| `reports/models/frozen_model_specification.json` | Frozen Study A specification |

### Metrics
| Path | Description |
|------|-------------|
| `reports/metrics/study_a_*.json` | Study A labels, baselines, XGB/CatBoost, bootstrap, financial sanity |
| `reports/metrics/study_b_*.json` | Study B labels + XGB |
| `reports/metrics/study_c_label_meta.json` | Strict-horizon eligibility (often underpowered) |
| `reports/metrics/pipeline_summary.json` | Full run summary |
| `reports/metrics/shap_interaction_status.json` | Interaction attempt status |
| `reports/metrics/environment.json` | Package versions |

### Tables
Prefer `study_*`, `observation_gap_*`, `feature_set_comparison.csv`, `model_comparison.csv`,
`permutation_importance.csv`, `shap_*.csv`, `regime_performance.csv`, `hyperparameter_trials.csv`,
`feature_correlation_clusters.csv`, `per_fold_metrics.csv`.

### Figures
Prefer subdirectory layout:

```text
reports/figures/data_quality/
reports/figures/labels/
reports/figures/models/
reports/figures/shap/
reports/figures/ablation/
reports/figures/robustness/
```

---

## Archives (historical — do not treat as current results)

| Path | Contents |
|------|----------|
| `reports/archive/pre_research_redesign/` | Freeze before multi-study redesign |
| `reports/archive/v1_leftover_artifacts/` | Stale root models/metrics/figures left after redesign |

---

## Intentionally limited / documented gaps

| Item | Status |
|------|--------|
| Study C 10s / 30s models | Underpowered (0 eligible rows) |
| Study C 60s development_test models | Underpowered after target purge |
| Nested walk-forward Macro F1 per fold | Calendar only in `per_fold_metrics.csv` (scores not nested-retrained) |
| Full SHAP beeswarm/waterfall suite | Generated under `reports/figures/shap/` (beeswarm per class, dependence, waterfalls) |

---

## Reproduction

```bash
./scripts/run_pipeline.sh
pytest -q
```

Canonical results must match `models/study_a/` and `reports/metrics/study_*` after a clean run.

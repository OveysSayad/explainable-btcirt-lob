"""Report and README generation for the redesigned multi-study pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, default=_json_default)
    logger.info("Saved JSON %s", path)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj)
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:  # noqa: BLE001
            pass
    if isinstance(obj, set):
        return list(obj)
    return str(obj)


def _fmt_metric(metrics: dict[str, Any] | None, key: str) -> str:
    if not metrics or key not in metrics:
        return "N/A"
    val = metrics[key]
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def _study_a(summary: dict[str, Any]) -> dict[str, Any]:
    return summary.get("study_a") or {}


def _load_table(paths: Any, name: str) -> pd.DataFrame | None:
    p = paths.tables / name
    if p.exists():
        return pd.read_csv(p)
    return None


def write_final_report(paths: Any, summary: dict[str, Any]) -> Path:
    """Write paper-like final report using actual pipeline outputs only."""
    sa = _study_a(summary)
    sb = summary.get("study_b") or {}
    sc = summary.get("study_c") or {}
    catalog = summary.get("catalog", {})
    gap = summary.get("gap_overall", {})
    shap_top = (sa.get("shap") or summary.get("shap") or {}).get("top_features", [])[:10]
    boot = sa.get("bootstrap") or {}
    inter = sa.get("interaction") or {}

    shap_lines = "\n".join(
        f"- `{r.get('feature')}`: mean |SHAP| = {float(r.get('mean_abs_shap', float('nan'))):.6f}"
        for r in shap_top
    ) or "- Unavailable / pending"

    feat_cmp = _load_table(paths, "feature_set_comparison.csv")
    feat_md = (
        feat_cmp.to_markdown(index=False)
        if feat_cmp is not None and len(feat_cmp)
        else "_See reports/tables/feature_set_comparison.csv_"
    )
    overlap = _load_table(paths, "horizon_overlap.csv")
    overlap_md = (
        overlap.to_markdown(index=False)
        if overlap is not None and len(overlap)
        else "_See reports/tables/horizon_overlap.csv_"
    )

    text = f"""# Explainable BTCIRT LOB — Research Redesign Final Report

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
- Raw rows: **{catalog.get('total_rows', 'N/A')}**
- BTCIRT rows: **{catalog.get('btcirt_rows', 'N/A')}** ({catalog.get('pct_retained', 'N/A')}%)
- Range: `{catalog.get('btcirt_min_timestamp', 'N/A')}` → `{catalog.get('btcirt_max_timestamp', 'N/A')}`
- Unique dates: **{catalog.get('btcirt_unique_dates', 'N/A')}**
- Eight LOB levels; JSON columns unused when flattened fields are complete
- Duplicate timestamps: keep last (documented in preprocessing meta)

## 6. Data-quality audit

See `reports/metrics/data_quality.json` and `reports/tables/data_quality_summary.csv`.
Checks include missing/infinite values, zero/negative prices & quantities, crossed/locked books,
invalid ask/bid ordering, spreads, timestamp parse failures, and observation gaps.

## 7. Sparse-sampling problem

Overall gap summary (seconds): `{gap}`.

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

- Study A hybrid ε (train-only): **{sa.get('meta', {}).get('epsilon_bps', 'N/A')}** bps
  (method `{sa.get('meta', {}).get('epsilon_method', 'N/A')}`)
- Study A labeled n≈**{sa.get('n_train', 'N/A')}+{sa.get('n_val', 'N/A')}+{sa.get('n_test', 'N/A')}**
  (train/val/dev-test after purge)
- Study B primary samples: **{(sb.get('meta') or {}).get('n_primary_sample', sb.get('n_primary', 'N/A'))}**
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

Pipeline completed: **{summary.get('pipeline_completed')}**. Seed: **{summary.get('random_seed')}**.

### Study A (development_test)

- XGBoost Macro F1: **{_fmt_metric(sa.get('xgb_test'), 'macro_f1')}**
- XGBoost Balanced Acc: **{_fmt_metric(sa.get('xgb_test'), 'balanced_accuracy')}**
- XGBoost Log Loss: **{_fmt_metric(sa.get('xgb_test'), 'log_loss')}**
- CatBoost Macro F1: **{_fmt_metric(sa.get('cat_test'), 'macro_f1')}**
- Val Macro F1 (XGB): **{_fmt_metric(sa.get('xgb_val'), 'macro_f1')}**
- Best params: `{sa.get('best_params')}`
- Bootstrap Macro F1 CI: `{boot}`

### Study B

`{json.dumps({k: v for k, v in sb.items() if k != 'meta'} | {'meta_keys': list((sb.get('meta') or {}).keys())}, default=str)[:2000]}`

### Study C (pilot)

`{json.dumps(sc.get('models', sc), default=str)[:2500]}`

Interpretation: compare models to baselines in `reports/tables/model_comparison.csv`. Do not
over-claim Study C if status is `underpowered_pilot`.

## 14. Incremental value of LOB data

{feat_md}

Incremental LOB value ≈ MacroF1(full_no_trade) − MacroF1(price_only). Uncertainty: day bootstrap
where available. Prefer validation / fold evidence over a single development_test number.

## 15. Explainability

Top SHAP features (Study A sample):

{shap_lines}

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

{overlap_md}

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
SHAP interaction status: `{inter}`
Generated from pipeline summary; archived prior results live in
`reports/archive/pre_research_redesign/`.
"""
    out = paths.reports / "final_report.md"
    out.write_text(text, encoding="utf-8")
    logger.info("Wrote final report %s", out)
    return out


def write_readme(paths: Any, summary: dict[str, Any]) -> Path:
    """Write comprehensive README with redesign methodology and actual results."""
    sa = _study_a(summary)
    sb = summary.get("study_b") or {}
    sc = summary.get("study_c") or {}
    catalog = summary.get("catalog", {})
    gap = summary.get("gap_overall", {})
    executed = bool(summary.get("pipeline_completed"))
    shap_top = (sa.get("shap") or summary.get("shap") or {}).get("top_features", [])[:10]
    boot = sa.get("bootstrap") or {}
    inter = sa.get("interaction") or {}

    archived_note = """
### Archived previous results (v1 — do not treat as current)

Prior fixed-horizon pipeline (archived under `reports/archive/pre_research_redesign/`):
XGBoost test Macro F1 ≈ 0.460; CatBoost ≈ 0.452; previous-direction ≈ 0.42.
Those numbers are historical context only.
"""

    if executed:
        results = f"""
## 15. Results (this run)

- Pipeline completed: **True**
- Seed: **{summary.get('random_seed')}**
- Primary features: **{summary.get('n_features_primary', summary.get('n_features'))}**
- Study A ε: **{(sa.get('meta') or {}).get('epsilon_bps')}** bps (`{(sa.get('meta') or {}).get('epsilon_method')}`)
- Study A n (train/val/dev-test): **{sa.get('n_train')}** / **{sa.get('n_val')}** / **{sa.get('n_test')}**
- Study A XGB Macro F1 (val / dev-test): **{_fmt_metric(sa.get('xgb_val'), 'macro_f1')}** / **{_fmt_metric(sa.get('xgb_test'), 'macro_f1')}**
- Study A CatBoost Macro F1 (dev-test): **{_fmt_metric(sa.get('cat_test'), 'macro_f1')}**
- Study A best params: `{sa.get('best_params')}`
- Bootstrap: `{boot}`
- Study B: `{json.dumps(sb, default=str)[:1500]}`
- Study C models: `{json.dumps(sc.get('models', {}), default=str)[:1500]}`

Tables: `reports/tables/model_comparison.csv`, `feature_set_comparison.csv`,
`study_*_class_distribution.csv`, `horizon_overlap.csv`.
"""
    else:
        results = """
## 15. Results

> **PENDING** — run `./scripts/run_pipeline.sh` to populate metrics. No fabricated numbers.
"""

    shap_md = "\n".join(
        f"| `{r.get('feature')}` | {float(r.get('mean_abs_shap', float('nan'))):.6f} |"
        for r in shap_top
    ) or "| _pending_ | |"

    readme = f"""# Explainable Machine Learning for BTCIRT Limit Order Book Dynamics (Research Redesign)

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

{archived_note}

## 3. Research questions

1. Study A: next observed mid movement predictability?
2. Study B: next change direction predictability?
3. Study C: is honest fixed-horizon prediction feasible on eligible subsets?
4. Incremental value of LOB vs price-only features?
5. Explanation stability vs out-of-sample feature value?

## 4. Dataset

- Source: `data/raw/market_data_clean_nobitex.csv` (not in git)
- Exchange / symbol: `nobitex` / `BTCIRT` (normalized before filter)
- Raw rows: **{catalog.get('total_rows', 'run pipeline')}**
- BTCIRT: **{catalog.get('btcirt_rows', 'run pipeline')}** ({catalog.get('pct_retained', 'n/a')}%)
- Dates: `{catalog.get('btcirt_min_timestamp', 'n/a')}` → `{catalog.get('btcirt_max_timestamp', 'n/a')}`
  (**{catalog.get('btcirt_unique_dates', 'n/a')}** unique dates)
- Eight LOB levels + last_trade fields; JSON blobs unused when flattened columns exist
- Duplicate timestamps: **keep last** after chronological sort

## 5. Data-quality audit

Pipeline writes `reports/metrics/data_quality.json` and
`reports/tables/data_quality_summary.csv`. Checks: missing/inf, zeros/negatives, crossed/locked
books, level ordering, spreads, timestamp failures, duplicates, gaps.

## 6. Why sparse snapshots matter

Gap overall: `{gap}`.

- 10/30/60s labels previously overlapped because the next snapshot often served all horizons.
- Five-second forward-fill invents false density across minute-scale gaps.
- Default: **preserve native timestamps**; no 5s grid.

## 7. Mathematical notation

\\(a_i, b_i\\): ask/bid prices; \\(q^a_i, q^b_i\\): quantities; mid
\\(m=(a_1+b_1)/2\\); spread \\(S=a_1-b_1\\); returns in bps via
\\(10^4\\log(m_s/m_t)\\). Full formulas: `docs/FORMULAS.md`.

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

{results}

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
{shap_md}

## 19. SHAP interaction status

`{inter}` — failures logged with package versions; fallbacks only; no fabricated interactions.
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
export DYLD_LIBRARY_PATH="$(pwd)/.libs:${{DYLD_LIBRARY_PATH}}"
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
"""
    out = paths.root / "README.md"
    out.write_text(readme, encoding="utf-8")
    logger.info("Wrote README %s", out)
    return out

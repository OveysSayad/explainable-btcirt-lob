#!/usr/bin/env python3
"""Regenerate full Study A SHAP plot suite from the frozen model (no retuning)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import get_seed, load_config, resolve_paths, set_global_seed
from src.data_loader import load_btcirt
from src.delay_audit import audit_observation_gaps
from src.explainability.shap_analysis import run_shap_analysis
from src.feature_engineering import engineer_features, get_feature_set
from src.labels.next_observation import build_study_a_labels
from src.models.logistic_regression import prepare_xy
from src.preprocessing import preprocess
from src.splitting.purged_split import purge_by_target_timestamp
from src.splitting.temporal_split import chronological_date_split, masks_from_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("regenerate_shap")


def main() -> int:
    config = load_config(ROOT / "configs" / "project_config.yaml")
    paths = resolve_paths(config)
    seed = get_seed(config)
    set_global_seed(seed)

    raw = load_btcirt(paths.raw, config)
    clean, _ = preprocess(raw, config)
    clean = clean.sort_values("timestamp").reset_index(drop=True)
    prelim = chronological_date_split(
        clean,
        float(config["splitting"]["train_fraction"]),
        float(config["splitting"]["validation_fraction"]),
        float(config["splitting"]["development_test_fraction"]),
    )
    prelim_masks = masks_from_split(len(clean), prelim)
    gap_audit = audit_observation_gaps(clean, paths.tables, prelim_masks)
    clean = gap_audit["frame_with_gaps"]

    feat = engineer_features(clean, config).sort_values("timestamp").reset_index(drop=True)
    split = chronological_date_split(
        feat,
        float(config["splitting"]["train_fraction"]),
        float(config["splitting"]["validation_fraction"]),
        float(config["splitting"]["development_test_fraction"]),
    )
    masks = masks_from_split(len(feat), split)

    study_a_df, meta_a = build_study_a_labels(
        feat, config, masks["train"], masks["val"]
    )
    logger.info("Study A epsilon_bps=%s method=%s", meta_a["epsilon_bps"], meta_a["epsilon_method"])

    tr, va, te = purge_by_target_timestamp(
        study_a_df["timestamp"],
        study_a_df["target_timestamp"],
        masks["train"],
        masks["val"],
        masks["development_test"],
    )
    masks_a = {"train": tr, "val": va, "development_test": te}
    labeled = study_a_df["label"].notna().to_numpy()
    for k in masks_a:
        masks_a[k] = masks_a[k] & labeled

    # Train-median impute (same as pipeline)
    feature_cols = list(
        joblib.load(paths.models / "study_a" / "xgboost_study_a.joblib")["feature_cols"]
    )
    med = (
        study_a_df.loc[masks_a["train"], feature_cols]
        .replace([np.inf, -np.inf], np.nan)
        .median(numeric_only=True)
    )
    for c in feature_cols:
        study_a_df[c] = study_a_df[c].replace([np.inf, -np.inf], np.nan)
        m = med.get(c, 0.0)
        if m is None or (isinstance(m, float) and np.isnan(m)):
            m = 0.0
        study_a_df[c] = study_a_df[c].fillna(float(m))

    bundle = joblib.load(paths.models / "study_a" / "xgboost_study_a.joblib")
    model = bundle["model"]
    feature_cols = list(bundle["feature_cols"])

    Xte, yte, idxte = prepare_xy(
        study_a_df, feature_cols, masks_a["development_test"]
    )
    proba = model.predict_proba(Xte)
    pred = {
        "index": idxte.to_numpy(),
        "y_true": yte,
        "y_pred": np.argmax(proba, axis=1),
        "y_proba": proba,
    }

    summary = run_shap_analysis(
        model,
        study_a_df,
        feature_cols,
        masks_a["development_test"],
        config,
        paths,
        prefix="study_a",
        predictions=pred,
    )
    figs = summary.get("generated_figures", [])
    logger.info("Generated %s SHAP figures:", len(figs))
    for f in figs:
        logger.info("  - %s", f)
    if summary.get("error"):
        logger.error("%s", summary["error"])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

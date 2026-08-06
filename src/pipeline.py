"""Redesigned multi-study research pipeline for sparse BTCIRT LOB data."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import get_seed, load_config, resolve_paths, set_global_seed
from src.data_loader import load_btcirt, save_parquet, summarize_raw_catalog
from src.data_validation import (
    audit_order_book_quality,
    audit_timestamp_gaps,
    save_quality_report,
    validate_schema,
)
from src.delay_audit import audit_observation_gaps
from src.evaluation.bootstrap import day_level_bootstrap_ci
from src.evaluation.financial_sanity_check import long_only_sanity
from src.explainability.ablation import run_feature_set_and_ablation
from src.explainability.interaction_fallbacks import run_interaction_analysis
from src.explainability.permutation_importance import permutation_importance_table
from src.explainability.shap_analysis import run_shap_analysis
from src.feature_engineering import (
    engineer_features,
    feature_dictionary_frame,
    get_feature_set,
)
from src.labels.next_observation import build_study_a_labels
from src.labels.next_price_change import build_study_b_labels
from src.labels.strict_horizon import build_study_c_labels, cross_horizon_overlap
from src.models.catboost_model import train_catboost
from src.models.logistic_regression import prepare_xy, train_logistic
from src.models.naive_baselines import run_baselines
from src.models.xgboost_model import train_xgboost
from src.preprocessing import preprocess
from src.reporting import save_json, write_final_report, write_readme
from src.reporting_figures import generate_redesign_figures
from src.splitting.purged_split import assert_targets_respect_boundaries, purge_by_target_timestamp
from src.splitting.temporal_split import chronological_date_split, masks_from_split
from src.splitting.walk_forward import nested_walk_forward_folds

logger = logging.getLogger(__name__)


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8"),
        ],
        force=True,
    )


def _impute(df: pd.DataFrame, cols: list[str], train_mask: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    med = out.loc[train_mask, cols].replace([np.inf, -np.inf], np.nan).median(numeric_only=True)
    for c in cols:
        out[c] = out[c].replace([np.inf, -np.inf], np.nan)
        m = med.get(c, 0.0)
        if m is None or (isinstance(m, float) and np.isnan(m)):
            m = 0.0
        out[c] = out[c].fillna(float(m))
    return out


def _prev_direction(df: pd.DataFrame, eps: float) -> np.ndarray:
    prev = df["mid_price"].shift(1)
    r = 10_000.0 * np.log(df["mid_price"] / prev)
    pred = np.full(len(df), 1, dtype=int)
    valid = r.notna()
    pred[valid & (r < -eps)] = 0
    pred[valid & (r.abs() <= eps)] = 1
    pred[valid & (r > eps)] = 2
    pred[0] = 1
    return pred


def run_pipeline(config_path: str | Path | None = None) -> dict[str, Any]:
    """Execute redesigned Studies A/B/C pipeline."""
    config = load_config(config_path)
    paths = resolve_paths(config)
    setup_logging(paths.logs)
    seed = get_seed(config)
    set_global_seed(seed)

    summary: dict[str, Any] = {
        "pipeline_completed": False,
        "random_seed": seed,
        "errors": [],
        "stages_completed": [],
        "redesign_version": "2.0.0",
    }
    study_a_frame: pd.DataFrame | None = None
    masks_a_keep: dict[str, np.ndarray] | None = None
    xgb_keep: dict[str, Any] | None = None
    meta_a_keep: dict[str, Any] | None = None
    primary_cols_keep: list[str] = []
    feat_keep: pd.DataFrame | None = None

    try:
        # Environment
        import platform
        import sklearn
        import xgboost
        import catboost
        import shap

        env = {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "catboost": catboost.__version__,
            "shap": shap.__version__,
        }
        save_json(env, paths.metrics / "environment.json")

        # Stage: data audit
        logger.info("=== Data audit ===")
        catalog = summarize_raw_catalog(paths.raw, int(config["data"]["chunk_size"]))
        print("\n=== RAW CATALOG ===")
        for k, v in catalog.items():
            print(f"{k}: {v}")
        print("===================\n")
        raw = load_btcirt(paths.raw, config)
        schema = validate_schema(raw)
        book_q = audit_order_book_quality(raw)
        gaps_legacy = audit_timestamp_gaps(raw.sort_values("timestamp"))
        save_quality_report(catalog, schema, book_q, gaps_legacy, paths.metrics, paths.tables)
        summary["catalog"] = catalog
        summary["stages_completed"].append("data_audit")

        # Preprocess
        clean, prep_meta = preprocess(raw, config, gap_stats=gaps_legacy)
        save_json(prep_meta, paths.metrics / "preprocessing_meta.json")
        save_parquet(clean, paths.interim)

        # Preliminary date split for gap-by-split (before labels)
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
        summary["gap_overall"] = gap_audit["overall"]
        summary["stages_completed"].append("gap_audit")

        # Features
        logger.info("=== Feature engineering ===")
        feat = engineer_features(clean, config)
        feat_dict = feature_dictionary_frame()
        feat_dict.to_csv(paths.tables / "feature_dictionary.csv", index=False)
        primary_cols = get_feature_set(feat, "full_no_trade", include_time=True, include_trade=False)
        summary["n_features_primary"] = len(primary_cols)
        summary["stages_completed"].append("features")

        # Date split on features
        feat = feat.sort_values("timestamp").reset_index(drop=True)
        split = chronological_date_split(
            feat,
            float(config["splitting"]["train_fraction"]),
            float(config["splitting"]["validation_fraction"]),
            float(config["splitting"]["development_test_fraction"]),
        )
        masks = masks_from_split(len(feat), split)
        save_json(split.metadata, paths.metrics / "split_metadata.json")
        pd.DataFrame(
            [
                {"split": "train", "n": len(split.train_idx), "dates": ",".join(split.train_dates)},
                {"split": "validation", "n": len(split.val_idx), "dates": ",".join(split.val_dates)},
                {
                    "split": "development_test",
                    "n": len(split.test_idx),
                    "dates": ",".join(split.test_dates),
                },
            ]
        ).to_csv(paths.tables / "fold_definitions.csv", index=False)

        # -------------------- Study A --------------------
        study_a_results: dict[str, Any] = {}
        if config.get("study_a", {}).get("enabled", True):
            logger.info("=== Study A: next observation ===")
            study_a_df, meta_a = build_study_a_labels(
                feat, config, masks["train"], masks["val"]
            )
            # Target purge
            tr, va, te = purge_by_target_timestamp(
                study_a_df["timestamp"],
                study_a_df["target_timestamp"],
                masks["train"],
                masks["val"],
                masks["development_test"],
            )
            assert_targets_respect_boundaries(
                study_a_df["timestamp"], study_a_df["target_timestamp"], tr, va, te
            )
            masks_a = {"train": tr, "val": va, "development_test": te}
            # require labels
            labeled = study_a_df["label"].notna().to_numpy()
            for k in masks_a:
                masks_a[k] = masks_a[k] & labeled

            study_a_df = _impute(study_a_df, primary_cols, masks_a["train"])
            save_json(
                {
                    **{k: v for k, v in meta_a.items() if k != "candidates"},
                    "candidates": meta_a["candidates"].to_dict(orient="records")
                    if hasattr(meta_a.get("candidates"), "to_dict")
                    else meta_a.get("candidates"),
                },
                paths.metrics / "study_a_label_meta.json",
            )
            # class dist
            dist_rows = []
            for split_name, mask in masks_a.items():
                vc = study_a_df.loc[mask, "label"].value_counts().reindex([0, 1, 2], fill_value=0)
                for code, name in [(0, "DOWN"), (1, "STABLE"), (2, "UP")]:
                    dist_rows.append(
                        {
                            "split": split_name,
                            "class_code": code,
                            "class_name": name,
                            "count": int(vc.loc[code]),
                            "percentage": float(100 * vc.loc[code] / vc.sum()) if vc.sum() else 0,
                        }
                    )
            pd.DataFrame(dist_rows).to_csv(
                paths.tables / "study_a_class_distribution.csv", index=False
            )
            if meta_a.get("tick"):
                pd.DataFrame([meta_a["tick"]]).to_csv(
                    paths.tables / "tick_size_estimation.csv", index=False
                )

            # Walk-forward folds metadata
            folds = nested_walk_forward_folds(
                study_a_df, masks_a["train"], masks_a["val"], int(config["splitting"]["walk_forward_folds"])
            )
            save_json(folds, paths.metrics / "study_a_walk_forward_folds.json")

            # Baselines
            y_train = study_a_df.loc[masks_a["train"], "label"].astype(int).to_numpy()
            y_eval = {
                k: study_a_df.loc[m, "label"].astype(int).to_numpy() for k, m in masks_a.items()
            }
            prev = _prev_direction(study_a_df, float(meta_a["epsilon_bps"]))
            base = run_baselines(
                y_train,
                y_eval,
                obi_train=study_a_df.loc[masks_a["train"], "obi_5"].to_numpy(),
                obi_eval={k: study_a_df.loc[m, "obi_5"].to_numpy() for k, m in masks_a.items()},
                prev_dir_eval={k: prev[m] for k, m in masks_a.items()},
                seed=seed,
            )
            save_json(base, paths.metrics / "study_a_baseline_metrics.json")

            # Models
            logistic = train_logistic(
                study_a_df, primary_cols, masks_a, config, paths.models / "study_a"
            )
            n_trials = int(config["optimization"]["xgboost_trials_study_a"])
            xgb = train_xgboost(
                study_a_df,
                primary_cols,
                masks_a,
                config,
                paths.models / "study_a",
                n_trials=n_trials,
                model_name="xgboost_study_a",
            )
            cat = train_catboost(
                study_a_df,
                primary_cols,
                masks_a,
                config,
                paths.models / "study_a",
                n_trials=int(config["optimization"]["catboost_trials"]),
                model_name="catboost_study_a",
            )
            save_json(
                {
                    "best_params": xgb["best_params"],
                    "metrics_val": xgb["metrics_val"],
                    "metrics_development_test": xgb["metrics_development_test"],
                    "software": xgb.get("software"),
                    "search_history": xgb.get("search_history", []),
                    "best_val_macro_f1": xgb.get("best_val_macro_f1"),
                },
                paths.metrics / "study_a_xgboost_metrics.json",
            )
            save_json(
                {
                    "best_params": cat["best_params"],
                    "metrics_val": cat["metrics_val"],
                    "metrics_development_test": cat["metrics_development_test"],
                },
                paths.metrics / "study_a_catboost_metrics.json",
            )

            # Feature-set / ablation
            abl = run_feature_set_and_ablation(study_a_df, masks_a, config, paths.tables)
            # Bootstrap
            pred = xgb["predictions_development_test"]
            boot = day_level_bootstrap_ci(
                study_a_df.loc[pred["index"], "timestamp"],
                pred["y_true"],
                pred["y_pred"],
                pred["y_proba"],
                n_bootstrap=int(config["bootstrap"]["iterations"]),
                seed=seed,
            )
            save_json(boot, paths.metrics / "study_a_bootstrap_macro_f1.json")

            # SHAP + permutation + interactions
            import joblib

            bundle = joblib.load(paths.models / "study_a" / "xgboost_study_a.joblib")
            shap_sum = run_shap_analysis(
                bundle["model"],
                study_a_df,
                primary_cols,
                masks_a["development_test"],
                config,
                paths,
                prefix="study_a",
                predictions=pred,
            )
            Xte, yte, idxte = prepare_xy(
                study_a_df, primary_cols, masks_a["development_test"]
            )
            dates = (
                study_a_df.loc[idxte, "timestamp"].dt.date.astype(str).to_numpy()
            )
            perm = permutation_importance_table(
                bundle["model"], Xte, yte, primary_cols, dates, seed=seed
            )
            perm.to_csv(paths.tables / "permutation_importance.csv", index=False)
            # stability placeholder from single run ranks
            stab = shap_sum.get("top_features", [])
            pd.DataFrame(stab).assign(rank=lambda d: np.arange(1, len(d) + 1)).to_csv(
                paths.tables / "shap_stability.csv", index=False
            )
            Xdf = pd.DataFrame(Xte, columns=primary_cols)
            # ensure observation_gap available for fallback pair
            if "observation_gap_seconds" in study_a_df.columns:
                Xdf["observation_gap_seconds"] = study_a_df.loc[
                    idxte, "observation_gap_seconds"
                ].to_numpy()
            inter = run_interaction_analysis(bundle["model"], Xdf[primary_cols], config, paths)

            # Predictions parquet
            pred_df = pd.DataFrame(
                {
                    "current_timestamp": study_a_df.loc[pred["index"], "timestamp"].to_numpy(),
                    "target_timestamp": study_a_df.loc[pred["index"], "target_timestamp"].to_numpy(),
                    "actual_delay_seconds": study_a_df.loc[
                        pred["index"], "actual_delay_seconds"
                    ].to_numpy(),
                    "actual_class": pred["y_true"],
                    "predicted_class": pred["y_pred"],
                    "probability_down": pred["y_proba"][:, 0],
                    "probability_stable": pred["y_proba"][:, 1],
                    "probability_up": pred["y_proba"][:, 2],
                    "current_mid_price": study_a_df.loc[pred["index"], "mid_price"].to_numpy(),
                    "realized_return_bps": study_a_df.loc[
                        pred["index"], "next_observation_return_bps"
                    ].to_numpy(),
                    "observation_gap_seconds": study_a_df.loc[
                        pred["index"], "observation_gap_seconds"
                    ].to_numpy(),
                    "model_name": "xgboost",
                    "feature_set": "full_no_trade",
                }
            )
            pred_df.to_parquet(
                paths.tables / "study_a_development_test_predictions.parquet", index=False
            )

            # Financial sanity
            fin = {}
            for name, sc in config.get("financial_sanity_check", {}).get("scenarios", {}).items():
                fin[name] = long_only_sanity(
                    study_a_df.loc[pred["index"], "mid_price"].to_numpy(),
                    study_a_df.loc[pred["index"], "next_observation_return_bps"].to_numpy(),
                    pred["y_pred"],
                    fee_bps=float(sc.get("fee_bps", 0)),
                    slippage_bps=float(sc.get("slippage_bps", 0)),
                )
            save_json(fin, paths.metrics / "study_a_financial_sanity.json")

            # Frozen spec
            frozen = {
                "study": "A_next_observation",
                "feature_set": "full_no_trade",
                "label_method": "next_observation",
                "epsilon_method": meta_a["epsilon_method"],
                "epsilon_bps": meta_a["epsilon_bps"],
                "model_class": "XGBoostClassifier",
                "hyperparameters": xgb["best_params"],
                "random_seed": seed,
                "software": env,
                "training_date_range": [
                    split.train_dates[0],
                    split.train_dates[-1],
                ],
                "note": "Final independent holdout evaluation pending.",
            }
            save_json(frozen, paths.models_report / "frozen_model_specification.json")

            study_a_results = {
                "meta": {
                    k: (dict(v) if hasattr(v, "items") and not isinstance(v, dict) else v)
                    for k, v in meta_a.items()
                    if k != "candidates"
                },
                "xgb_val": xgb["metrics_val"],
                "xgb_test": xgb["metrics_development_test"],
                "cat_test": cat["metrics_development_test"],
                "best_params": xgb["best_params"],
                "bootstrap": boot,
                "shap": {
                    k: v
                    for k, v in shap_sum.items()
                    if k != "explanation_object"
                },
                "ablation": abl.to_dict(orient="records"),
                "interaction": inter,
                "n_train": int(masks_a["train"].sum()),
                "n_val": int(masks_a["val"].sum()),
                "n_test": int(masks_a["development_test"].sum()),
            }
            # JSON-safe class counts
            if "class_counts" in study_a_results["meta"]:
                study_a_results["meta"]["class_counts"] = {
                    str(int(float(k))): int(v)
                    for k, v in study_a_results["meta"]["class_counts"].items()
                }
            summary["study_a"] = study_a_results
            summary["epsilons"] = {"study_a_bps": meta_a["epsilon_bps"]}
            summary["best_hyperparameters"] = xgb["best_params"]
            summary["xgboost_metrics"] = {
                "metrics_val": xgb["metrics_val"],
                "metrics_test": xgb["metrics_development_test"],
            }
            summary["non_overlapping_test_metrics"] = xgb["metrics_development_test"]
            summary["shap"] = shap_sum
            # Keep handles for post-hoc figure generation
            study_a_frame = study_a_df
            masks_a_keep = masks_a
            xgb_keep = xgb
            meta_a_keep = meta_a
            primary_cols_keep = list(primary_cols)
            feat_keep = feat
            summary["stages_completed"].append("study_a")

        # -------------------- Study B --------------------
        if config.get("study_b", {}).get("enabled", True):
            logger.info("=== Study B: next price change ===")
            study_b_df, meta_b = build_study_b_labels(
                feat,
                one_sample_per_price_run=bool(
                    config.get("study_b", {}).get("one_sample_per_price_run", True)
                ),
            )
            primary = study_b_df["study_b_primary_sample"].to_numpy()
            tr, va, te = purge_by_target_timestamp(
                study_b_df["timestamp"],
                study_b_df["target_timestamp"],
                masks["train"] & primary,
                masks["val"] & primary,
                masks["development_test"] & primary,
            )
            masks_b = {"train": tr, "val": va, "development_test": te}
            cols_b = get_feature_set(study_b_df, "full_no_trade")
            study_b_df = _impute(study_b_df, cols_b, masks_b["train"])
            save_json(meta_b, paths.metrics / "study_b_label_meta.json")
            vc = study_b_df.loc[primary, "label"].value_counts()
            pd.DataFrame(
                [
                    {"class_code": int(k), "count": int(v), "percentage": float(100 * v / vc.sum())}
                    for k, v in vc.items()
                ]
            ).to_csv(paths.tables / "study_b_class_distribution.csv", index=False)

            if masks_b["train"].sum() > 100 and masks_b["development_test"].sum() > 50:
                xgb_b = train_xgboost(
                    study_b_df,
                    cols_b,
                    masks_b,
                    config,
                    paths.models / "study_b",
                    n_trials=int(config["optimization"]["xgboost_trials_study_b"]),
                    binary=True,
                    model_name="xgboost_study_b",
                )
                save_json(
                    {
                        "best_params": xgb_b["best_params"],
                        "metrics_val": xgb_b["metrics_val"],
                        "metrics_development_test": xgb_b["metrics_development_test"],
                    },
                    paths.metrics / "study_b_xgboost_metrics.json",
                )
                pred = xgb_b["predictions_development_test"]
                pd.DataFrame(
                    {
                        "current_timestamp": study_b_df.loc[pred["index"], "timestamp"].to_numpy(),
                        "target_timestamp": study_b_df.loc[
                            pred["index"], "target_timestamp"
                        ].to_numpy(),
                        "actual_delay_seconds": study_b_df.loc[
                            pred["index"], "actual_delay_seconds"
                        ].to_numpy(),
                        "actual_class": pred["y_true"],
                        "predicted_class": pred["y_pred"],
                        "probability_down": pred["y_proba"][:, 0],
                        "probability_up": pred["y_proba"][:, 1],
                        "model_name": "xgboost_binary",
                    }
                ).to_parquet(
                    paths.tables / "study_b_development_test_predictions.parquet", index=False
                )
                summary["study_b"] = {
                    "meta": meta_b,
                    "xgb_test": xgb_b["metrics_development_test"],
                    "best_params": xgb_b["best_params"],
                    "n_primary": meta_b["n_primary_sample"],
                }
            else:
                summary["study_b"] = {
                    "meta": meta_b,
                    "error": "insufficient purged sample for modeling",
                }
            summary["stages_completed"].append("study_b")

        # -------------------- Study C --------------------
        if config.get("study_c", {}).get("enabled", True):
            logger.info("=== Study C: strict horizons (pilot) ===")
            labeled_c, meta_c = build_study_c_labels(
                feat, config, masks["train"], masks["val"]
            )
            overlap = cross_horizon_overlap(labeled_c)
            overlap.to_csv(paths.tables / "horizon_overlap.csv", index=False)
            save_json(meta_c, paths.metrics / "study_c_label_meta.json")
            # class dist table
            crows = []
            for h, frame in labeled_c.items():
                elig = frame["target_index"] >= 0
                if "label" not in frame.columns:
                    continue
                vc = frame.loc[elig, "label"].value_counts()
                for code, cnt in vc.items():
                    crows.append(
                        {
                            "horizon": h,
                            "class_code": int(code),
                            "count": int(cnt),
                            "underpowered": meta_c["horizons"][str(h)]["underpowered"],
                        }
                    )
            pd.DataFrame(crows).to_csv(
                paths.tables / "study_c_class_distribution.csv", index=False
            )

            study_c_model = {}
            for h, frame in labeled_c.items():
                hmeta = meta_c["horizons"][str(h)]
                if hmeta["underpowered"]:
                    logger.warning(
                        "Study C %ss underpowered (n=%s); descriptive only",
                        h,
                        hmeta["n_eligible"],
                    )
                    study_c_model[str(h)] = {
                        "status": "underpowered_pilot",
                        **hmeta,
                    }
                    continue
                elig = (frame["target_index"] >= 0) & frame["label"].notna()
                tr, va, te = purge_by_target_timestamp(
                    frame["timestamp"],
                    frame["target_timestamp"],
                    masks["train"] & elig.to_numpy(),
                    masks["val"] & elig.to_numpy(),
                    masks["development_test"] & elig.to_numpy(),
                )
                masks_h = {"train": tr, "val": va, "development_test": te}
                n_tr, n_va, n_te = int(tr.sum()), int(va.sum()), int(te.sum())
                if n_tr < 200 or n_va < 50 or n_te < 50:
                    logger.warning(
                        "Study C %ss insufficient purged splits "
                        "(train=%s val=%s development_test=%s); descriptive only",
                        h,
                        n_tr,
                        n_va,
                        n_te,
                    )
                    study_c_model[str(h)] = {
                        "status": "underpowered_after_purge",
                        "n_train_purged": n_tr,
                        "n_val_purged": n_va,
                        "n_development_test_purged": n_te,
                        "note": (
                            "Eligible rows exist overall, but target-timestamp purging "
                            "left an empty or tiny development_test. "
                            "Pilot fixed-horizon analysis — no robust model claim."
                        ),
                        **hmeta,
                    }
                    # Still save eligible prediction-less descriptive parquet columns
                    elig_idx = np.where(elig.to_numpy())[0]
                    pd.DataFrame(
                        {
                            "current_timestamp": frame.loc[elig_idx, "timestamp"].to_numpy(),
                            "target_timestamp": frame.loc[elig_idx, "target_timestamp"].to_numpy(),
                            "actual_delay_seconds": frame.loc[
                                elig_idx, "actual_delay_seconds"
                            ].to_numpy(),
                            "horizon_error_seconds": frame.loc[
                                elig_idx, "horizon_error_seconds"
                            ].to_numpy(),
                            "actual_class": frame.loc[elig_idx, "label"].to_numpy(),
                            "requested_horizon_seconds": h,
                            "model_name": "none_underpowered_after_purge",
                        }
                    ).to_parquet(
                        paths.tables / f"study_c_{h}s_predictions.parquet", index=False
                    )
                    continue
                cols = get_feature_set(frame, "full_no_trade")
                frame_i = _impute(frame, cols, masks_h["train"])
                xgb_h = train_xgboost(
                    frame_i,
                    cols,
                    masks_h,
                    config,
                    paths.models / "study_c",
                    n_trials=int(config["optimization"]["xgboost_trials_study_c"]),
                    model_name=f"xgboost_study_c_{h}s",
                )
                study_c_model[str(h)] = {
                    "status": "modeled",
                    "metrics_test": xgb_h["metrics_development_test"],
                    "best_params": xgb_h["best_params"],
                    "n_train_purged": n_tr,
                    "n_val_purged": n_va,
                    "n_development_test_purged": n_te,
                    **hmeta,
                }
                pred = xgb_h["predictions_development_test"]
                pd.DataFrame(
                    {
                        "current_timestamp": frame_i.loc[pred["index"], "timestamp"].to_numpy(),
                        "target_timestamp": frame_i.loc[
                            pred["index"], "target_timestamp"
                        ].to_numpy(),
                        "actual_delay_seconds": frame_i.loc[
                            pred["index"], "actual_delay_seconds"
                        ].to_numpy(),
                        "horizon_error_seconds": frame_i.loc[
                            pred["index"], "horizon_error_seconds"
                        ].to_numpy(),
                        "actual_class": pred["y_true"],
                        "predicted_class": pred["y_pred"],
                        "probability_down": pred["y_proba"][:, 0],
                        "probability_stable": pred["y_proba"][:, 1],
                        "probability_up": pred["y_proba"][:, 2],
                        "requested_horizon_seconds": h,
                    }
                ).to_parquet(
                    paths.tables / f"study_c_{h}s_predictions.parquet", index=False
                )
            summary["study_c"] = {"meta": meta_c, "models": study_c_model, "overlap": overlap.to_dict(orient="records")}
            summary["stages_completed"].append("study_c")

        # Delay summary table
        delay_rows = []
        if "study_a" in summary:
            delay_rows.append(
                {
                    "study": "A",
                    "median_delay": summary["study_a"]["meta"].get("delay_median"),
                    "p95_delay": summary["study_a"]["meta"].get("delay_p95"),
                }
            )
        if "study_b" in summary and "meta" in summary["study_b"]:
            delay_rows.append(
                {
                    "study": "B",
                    "median_delay": summary["study_b"]["meta"].get("median_time_to_change"),
                }
            )
        pd.DataFrame(delay_rows).to_csv(paths.tables / "label_delay_summary.csv", index=False)

        # Model comparison table for Study A
        if "study_a" in summary:
            rows = []
            for name, key in [
                ("majority", "majority_development_test"),
                ("previous_direction", "previous_direction_development_test"),
                ("obi_rule", "obi_rule_development_test"),
            ]:
                m = json.loads((paths.metrics / "study_a_baseline_metrics.json").read_text()).get(
                    key, {}
                )
                rows.append({"model": name, **{k: v for k, v in m.items() if not isinstance(v, list)}})
            rows.append(
                {
                    "model": "xgboost",
                    **{
                        k: v
                        for k, v in summary["study_a"]["xgb_test"].items()
                        if not isinstance(v, list)
                    },
                }
            )
            rows.append(
                {
                    "model": "catboost",
                    **{
                        k: v
                        for k, v in summary["study_a"]["cat_test"].items()
                        if not isinstance(v, list)
                    },
                }
            )
            pd.DataFrame(rows).to_csv(paths.tables / "model_comparison.csv", index=False)
            # per-class
            m = summary["study_a"]["xgb_test"]
            pd.DataFrame(
                [
                    {
                        "model": "xgboost",
                        "class": c,
                        "precision": m.get(f"precision_{c}"),
                        "recall": m.get(f"recall_{c}"),
                        "f1": m.get(f"f1_{c}"),
                        "support": m.get(f"support_{c}"),
                    }
                    for c in ["DOWN", "STABLE", "UP"]
                ]
            ).to_csv(paths.tables / "per_class_metrics.csv", index=False)
            pd.DataFrame([{"study": "A", **summary["best_hyperparameters"]}]).to_csv(
                paths.tables / "best_hyperparameters.csv", index=False
            )
            # Hyperparameter search history
            hist = xgb_keep.get("search_history", []) if xgb_keep else []
            if not hist:
                hist = json.loads(
                    (paths.metrics / "study_a_xgboost_metrics.json").read_text()
                ).get("search_history", [])
            trial_rows = []
            for h in hist:
                row = {
                    "trial": h.get("trial"),
                    "val_macro_f1": h.get("val_macro_f1"),
                    "seconds": h.get("seconds"),
                }
                row.update({f"param_{k}": v for k, v in (h.get("params") or {}).items()})
                trial_rows.append(row)
            pd.DataFrame(trial_rows).to_csv(
                paths.tables / "hyperparameter_trials.csv", index=False
            )

            # Correlation clusters (|corr| > 0.95)
            if feat_keep is not None and primary_cols_keep:
                corr = (
                    feat_keep.loc[masks_a_keep["train"], primary_cols_keep]
                    .replace([np.inf, -np.inf], np.nan)
                    .corr()
                    .abs()
                )
                pairs = []
                cols = list(corr.columns)
                for i, a in enumerate(cols):
                    for b in cols[i + 1 :]:
                        v = corr.loc[a, b]
                        if pd.notna(v) and float(v) > 0.95:
                            pairs.append(
                                {
                                    "feature_a": a,
                                    "feature_b": b,
                                    "abs_pearson": float(v),
                                    "cluster_rule": "abs_corr>0.95",
                                }
                            )
                pd.DataFrame(pairs).to_csv(
                    paths.tables / "feature_correlation_clusters.csv", index=False
                )

            # Walk-forward fold calendar (scores not nested-retrained in this release)
            folds_raw = json.loads(
                (paths.metrics / "study_a_walk_forward_folds.json").read_text()
            )
            fold_rows = []
            for fdef in folds_raw if isinstance(folds_raw, list) else []:
                if not isinstance(fdef, dict):
                    continue
                fold_rows.append(
                    {
                        "fold": fdef.get("fold"),
                        "n_train": fdef.get("n_train"),
                        "n_val": fdef.get("n_val"),
                        "train_dates": ",".join(fdef.get("train_dates") or []),
                        "val_dates": ",".join(fdef.get("val_dates") or []),
                        "macro_f1": None,
                        "note": (
                            "Fold calendar only; nested re-training scores not computed "
                            "in this pipeline release."
                        ),
                    }
                )
            pd.DataFrame(fold_rows).to_csv(paths.tables / "per_fold_metrics.csv", index=False)

            # Simple regime performance on development_test (thresholds from train)
            if study_a_frame is not None and masks_a_keep is not None and xgb_keep is not None:
                from sklearn.metrics import f1_score, balanced_accuracy_score, log_loss

                pred = xgb_keep["predictions_development_test"]
                te = study_a_frame.loc[pred["index"]].copy()
                te["y_true"] = pred["y_true"]
                te["y_pred"] = pred["y_pred"]
                te["y_proba"] = list(pred["y_proba"])
                train = study_a_frame.loc[masks_a_keep["train"]]
                vol_col = "volatility_300s" if "volatility_300s" in te.columns else None
                spread_col = "relative_spread_bps"
                gap_col = "observation_gap_seconds"
                regime_rows = []

                def _regime_metrics(name: str, mask: np.ndarray) -> None:
                    if mask.sum() < 30:
                        return
                    yt = te.loc[mask, "y_true"].to_numpy()
                    yp = te.loc[mask, "y_pred"].to_numpy()
                    proba = np.vstack(te.loc[mask, "y_proba"].to_numpy())
                    regime_rows.append(
                        {
                            "regime": name,
                            "n": int(mask.sum()),
                            "macro_f1": float(
                                f1_score(yt, yp, average="macro", zero_division=0)
                            ),
                            "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
                            "log_loss": float(log_loss(yt, proba, labels=[0, 1, 2])),
                        }
                    )

                if vol_col:
                    q1, q2 = train[vol_col].quantile([0.33, 0.66])
                    _regime_metrics("volatility_low", te[vol_col] <= q1)
                    _regime_metrics(
                        "volatility_medium",
                        (te[vol_col] > q1) & (te[vol_col] <= q2),
                    )
                    _regime_metrics("volatility_high", te[vol_col] > q2)
                if spread_col in te.columns:
                    q1, q2 = train[spread_col].quantile([0.33, 0.66])
                    _regime_metrics("spread_narrow", te[spread_col] <= q1)
                    _regime_metrics(
                        "spread_medium",
                        (te[spread_col] > q1) & (te[spread_col] <= q2),
                    )
                    _regime_metrics("spread_wide", te[spread_col] > q2)
                if gap_col in te.columns:
                    _regime_metrics("gap_lt_60", te[gap_col] < 60)
                    _regime_metrics(
                        "gap_60_180",
                        (te[gap_col] >= 60) & (te[gap_col] < 180),
                    )
                    _regime_metrics("gap_ge_180", te[gap_col] >= 180)
                pd.DataFrame(regime_rows).to_csv(
                    paths.tables / "regime_performance.csv", index=False
                )

            # Canonical redesign figures
            if (
                feat_keep is not None
                and study_a_frame is not None
                and masks_a_keep is not None
                and xgb_keep is not None
                and meta_a_keep is not None
            ):
                generate_redesign_figures(
                    feat=feat_keep,
                    study_a_df=study_a_frame,
                    masks_a=masks_a_keep,
                    primary_cols=primary_cols_keep,
                    meta_a=meta_a_keep,
                    xgb=xgb_keep,
                    paths=paths,
                    dpi=int(config.get("reporting", {}).get("figure_dpi", 200)),
                )
                summary["stages_completed"].append("figures")

        # Empty Study C prediction placeholders when underpowered
        for h in (10, 30):
            p = paths.tables / f"study_c_{h}s_predictions.parquet"
            if not p.exists():
                pd.DataFrame(
                    {
                        "note": [
                            f"No eligible samples for strict {h}s window; underpowered pilot."
                        ]
                    }
                ).to_parquet(p, index=False)
        # Documentation
        summary["pipeline_completed"] = True
        summary["n_features"] = summary.get("n_features_primary")
        write_final_report(paths, summary)
        write_readme(paths, summary)
        save_json(
            {
                k: v
                for k, v in summary.items()
                if k not in {"study_a", "study_b", "study_c"}
                or True
            },
            paths.metrics / "pipeline_summary.json",
        )
        # Also write redesign-aware CONFIG_CHANGES
        (paths.reports / "CONFIG_CHANGES.md").write_text(
            (paths.reports / "archive" / "pre_research_redesign" / "CONFIG_CHANGES.md").read_text()
            + "\n\n## Research redesign v2\n\n"
            "- Replaced fixed-horizon-as-primary with Studies A/B/C.\n"
            "- Study A = next observation; Study B = next price change; Study C = strict windows.\n"
            "- Target-timestamp purging replaces fixed 60s purge.\n"
            "- Trade deduplication module added; trades excluded from primary feature set.\n"
            "- Rolling windows remain 120–1200s given sparse gaps.\n"
            "- Development test is not a pristine holdout.\n",
            encoding="utf-8",
        )
        summary["stages_completed"].append("documentation")
        logger.info("Redesigned pipeline completed")
        return summary

    except Exception as exc:  # noqa: BLE001
        logger.exception("Pipeline failed: %s", exc)
        summary["errors"].append(str(exc))
        summary["traceback"] = traceback.format_exc()
        try:
            write_readme(paths, summary)
            write_final_report(paths, summary)
            save_json(summary, paths.metrics / "pipeline_summary.json")
        except Exception:  # noqa: BLE001
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Redesigned BTCIRT LOB pipeline")
    parser.add_argument("--config", default="configs/project_config.yaml")
    args = parser.parse_args(argv)
    run_pipeline(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

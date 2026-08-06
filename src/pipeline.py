"""End-to-end research pipeline for explainable BTCIRT LOB prediction."""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.ablation import run_ablation
from src.baselines import run_baselines
from src.config import get_seed, load_config, resolve_paths, set_global_seed
from src.data_loader import load_btcirt, save_parquet, summarize_raw_catalog
from src.data_validation import (
    audit_order_book_quality,
    audit_timestamp_gaps,
    save_quality_report,
    validate_schema,
)
from src.evaluation import day_level_bootstrap_ci, evaluate_classification, metrics_to_frame
from src.explainability import run_shap_analysis
from src.feature_engineering import (
    engineer_features,
    feature_dictionary,
    get_model_feature_columns,
)
from src.label_engineering import construct_all_labels, label_distribution
from src.preprocessing import preprocess
from src.reporting import save_json, write_final_report, write_readme
from src.robustness import run_horizon_robustness, trading_sanity_check
from src.temporal_split import (
    assert_no_overlap,
    chronological_date_split,
    masks_from_split,
    non_overlapping_indices,
    walk_forward_folds,
)
from src.train_catboost import train_catboost
from src.train_logistic import train_logistic
from src.train_xgboost import train_xgboost
from src.visualization import (
    plot_ablation,
    plot_bid_ask_window,
    plot_calibration,
    plot_class_distribution,
    plot_confusion,
    plot_correlation_heatmap,
    plot_depth_profile,
    plot_future_return_with_epsilon,
    plot_horizon_comparison,
    plot_microprice_edge,
    plot_mid_price,
    plot_missing_values,
    plot_model_comparison,
    plot_obi_distributions,
    plot_obi_vs_return,
    plot_per_class_prf,
    plot_performance_by_date,
    plot_performance_by_hour,
    plot_pr_ovr,
    plot_roc_ovr,
    plot_snapshots_by_date,
    plot_snapshots_by_hour,
    plot_spread_dist,
    plot_spread_series,
    plot_timestamp_gap_hist,
    plot_xgb_loss,
)

logger = logging.getLogger(__name__)


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "pipeline.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
        force=True,
    )


def _impute_train_stats(
    df: pd.DataFrame, feature_cols: list[str], train_mask: np.ndarray
) -> pd.DataFrame:
    """Fill NaN/inf using training-split medians only (no leakage)."""
    out = df.copy()
    train_medians = (
        out.loc[train_mask, feature_cols]
        .replace([np.inf, -np.inf], np.nan)
        .median(numeric_only=True)
    )
    for col in feature_cols:
        out[col] = out[col].replace([np.inf, -np.inf], np.nan)
        med = train_medians.get(col, 0.0)
        if med is None or (isinstance(med, float) and np.isnan(med)):
            med = 0.0
        out[col] = out[col].fillna(float(med))
    return out


def run_pipeline(config_path: str | Path | None = None) -> dict[str, Any]:
    """Execute the full research pipeline and return a summary dict."""
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
    }

    try:
        # ------------------------------------------------------------------
        # Stage 1: Data audit
        # ------------------------------------------------------------------
        logger.info("=== Stage 1: Data audit ===")
        catalog = summarize_raw_catalog(
            paths.raw, chunk_size=int(config["data"]["chunk_size"])
        )
        # Print required audit facts
        print("\n=== RAW DATA CATALOG ===")
        print(f"Unique exchanges: {catalog['unique_exchanges']}")
        print(f"Unique symbols: {catalog['unique_symbols']}")
        print(f"Total rows: {catalog['total_rows']}")
        print(f"BTCIRT rows: {catalog['btcirt_rows']}")
        print(f"BTCIRT min timestamp: {catalog['btcirt_min_timestamp']}")
        print(f"BTCIRT max timestamp: {catalog['btcirt_max_timestamp']}")
        print(f"BTCIRT unique dates: {catalog['btcirt_unique_dates']}")
        print(f"Percentage retained: {catalog['pct_retained']:.4f}%")
        print("========================\n")
        save_json(catalog, paths.metrics / "raw_catalog.json")

        raw_df = load_btcirt(paths.raw, config)
        schema = validate_schema(raw_df)
        book_q = audit_order_book_quality(raw_df)
        gaps = audit_timestamp_gaps(raw_df.sort_values("timestamp"))
        save_quality_report(catalog, schema, book_q, gaps, paths.metrics, paths.tables)
        summary["catalog"] = catalog
        summary["gap_stats"] = gaps
        summary["stages_completed"].append("data_audit")

        # ------------------------------------------------------------------
        # Stage 2: Preprocessing
        # ------------------------------------------------------------------
        logger.info("=== Stage 2: Preprocessing ===")
        clean_df, prep_meta = preprocess(raw_df, config, gap_stats=gaps)
        save_json(prep_meta, paths.metrics / "preprocessing_meta.json")
        if config["output"]["save_processed_data"]:
            save_parquet(clean_df, paths.interim)
        summary["n_clean_rows"] = len(clean_df)
        summary["stages_completed"].append("preprocessing")

        # ------------------------------------------------------------------
        # Stage 3: Feature engineering
        # ------------------------------------------------------------------
        logger.info("=== Stage 3: Feature engineering ===")
        feat_df = engineer_features(clean_df, config)
        feature_cols = get_model_feature_columns(
            feat_df, include_trade=bool(config["features"].get("include_trade_features", True))
        )
        feat_dict = feature_dictionary()
        feat_dict.to_csv(paths.tables / "feature_dictionary.csv", index=False)
        summary["n_features"] = len(feature_cols)
        summary["feature_cols"] = feature_cols

        # Feature quality
        miss = feat_df[feature_cols].isna().mean().sort_values(ascending=False)
        inf_counts = np.isinf(
            feat_df[feature_cols].select_dtypes(include=[np.number]).to_numpy()
        ).sum(axis=0)
        quality = pd.DataFrame(
            {
                "feature": feature_cols,
                "missing_pct": [100.0 * miss.get(c, 0.0) for c in feature_cols],
                "mean": feat_df[feature_cols].mean(numeric_only=True).reindex(feature_cols).to_numpy(),
                "std": feat_df[feature_cols].std(numeric_only=True).reindex(feature_cols).to_numpy(),
                "min": feat_df[feature_cols].min(numeric_only=True).reindex(feature_cols).to_numpy(),
                "p25": feat_df[feature_cols].quantile(0.25).reindex(feature_cols).to_numpy(),
                "p50": feat_df[feature_cols].quantile(0.50).reindex(feature_cols).to_numpy(),
                "p75": feat_df[feature_cols].quantile(0.75).reindex(feature_cols).to_numpy(),
                "max": feat_df[feature_cols].max(numeric_only=True).reindex(feature_cols).to_numpy(),
            }
        )
        quality.to_csv(paths.tables / "feature_quality_summary.csv", index=False)
        corr = feat_df[feature_cols].corr().abs()
        high_corr_pairs = []
        for i, a in enumerate(feature_cols):
            for b in feature_cols[i + 1 :]:
                val = corr.loc[a, b]
                if val > 0.95:
                    high_corr_pairs.append({"feature_a": a, "feature_b": b, "abs_corr": float(val)})
        pd.DataFrame(high_corr_pairs).to_csv(
            paths.tables / "high_correlation_pairs.csv", index=False
        )
        # Decision: do not auto-drop correlated features; document pairs instead.
        save_json(
            {
                "decision": "Retain correlated features; document pairs with |corr|>0.95",
                "n_high_corr_pairs": len(high_corr_pairs),
            },
            paths.metrics / "feature_correlation_decision.json",
        )
        summary["stages_completed"].append("features")

        # ------------------------------------------------------------------
        # Stage 4: Labels and splits
        # ------------------------------------------------------------------
        logger.info("=== Stage 4: Labels and splits ===")
        # Ensure contiguous chronological index before splitting / labeling
        feat_df = feat_df.sort_values("timestamp").reset_index(drop=True)
        # Preliminary split by dates on feature frame (before labels use train epsilon)
        split = chronological_date_split(
            feat_df,
            train_fraction=float(config["split"]["train_fraction"]),
            validation_fraction=float(config["split"]["validation_fraction"]),
            test_fraction=float(config["split"]["test_fraction"]),
            purge_seconds=float(config["split"]["purge_seconds"]),
        )
        assert_no_overlap(split)
        masks = masks_from_split(len(feat_df), split)
        save_json(split.metadata, paths.metrics / "split_metadata.json")
        pd.DataFrame(
            [
                {"split": "train", "dates": ",".join(split.train_dates), "n": len(split.train_idx)},
                {"split": "validation", "dates": ",".join(split.val_dates), "n": len(split.val_idx)},
                {"split": "test", "dates": ",".join(split.test_dates), "n": len(split.test_idx)},
            ]
        ).to_csv(paths.tables / "split_summary.csv", index=False)

        train_mask_series = pd.Series(masks["train"], index=feat_df.index)
        labeled_df, epsilons = construct_all_labels(feat_df, config, train_mask_series)
        summary["epsilons"] = epsilons
        save_json(epsilons, paths.metrics / "epsilons.json")

        # Class distributions
        dist_rows = []
        for split_name, mask in masks.items():
            dist = label_distribution(labeled_df.loc[mask, "label"])
            dist["split"] = split_name
            dist["horizon"] = config["labels"]["primary_horizon_seconds"]
            dist_rows.append(dist)
        for h in sorted(
            {int(config["labels"]["primary_horizon_seconds"])}
            | {int(x) for x in config["labels"]["robustness_horizons_seconds"]}
        ):
            dist = label_distribution(labeled_df[f"label_{h}s"])
            dist["split"] = "all"
            dist["horizon"] = h
            dist_rows.append(dist)
        class_dist = pd.concat(dist_rows, ignore_index=True)
        class_dist.to_csv(paths.tables / "class_distribution.csv", index=False)

        # Walk-forward metadata (test untouched)
        if config["split"].get("use_walk_forward_validation", True):
            folds = walk_forward_folds(
                labeled_df,
                split.train_idx,
                split.val_idx,
                n_folds=int(config["split"].get("n_walk_forward_folds", 3)),
                purge_seconds=float(config["split"]["purge_seconds"]),
            )
            save_json(
                [
                    {
                        "fold": f["fold"],
                        "n_train": f["n_train"],
                        "n_val": f["n_val"],
                        "train_dates": f["train_dates"],
                        "val_dates": f["val_dates"],
                    }
                    for f in folds
                ],
                paths.metrics / "walk_forward_folds.json",
            )

        # Impute features using train medians only
        labeled_df = _impute_train_stats(labeled_df, feature_cols, masks["train"])
        # Drop rows without primary label for modeling
        valid_label = labeled_df["label"].notna()
        for k in masks:
            masks[k] = masks[k] & valid_label.to_numpy()

        if config["output"]["save_processed_data"]:
            save_parquet(labeled_df, paths.processed)
            save_parquet(
                labeled_df[
                    ["timestamp", "mid_price", "label", "future_return_bps", "epsilon"]
                    + [c for c in labeled_df.columns if c.startswith("label_")]
                ],
                paths.labels,
            )
        summary["stages_completed"].append("labels_splits")

        # EDA plots
        dpi = int(config["output"]["figure_dpi"])
        ts_sorted = labeled_df.sort_values("timestamp")
        gap_series = ts_sorted["timestamp"].diff().dt.total_seconds().dropna()
        plot_timestamp_gap_hist(gap_series, paths.figures / "timestamp_gap_hist.png", dpi)
        plot_snapshots_by_date(ts_sorted, paths.figures / "snapshots_by_date.png", dpi)
        plot_snapshots_by_hour(ts_sorted, paths.figures / "snapshots_by_hour.png", dpi)
        plot_mid_price(ts_sorted, paths.figures / "mid_price_timeseries.png", dpi)
        plot_spread_series(ts_sorted, paths.figures / "relative_spread_timeseries.png", dpi)
        plot_spread_dist(ts_sorted, paths.figures / "relative_spread_distribution.png", dpi)
        plot_bid_ask_window(ts_sorted, paths.figures / "best_bid_ask_window.png", dpi)
        plot_depth_profile(ts_sorted, paths.figures / "depth_profile_example.png", dpi)
        plot_class_distribution(
            class_dist[class_dist["split"].isin(["train", "val", "test"])],
            paths.figures / "class_distribution_splits.png",
            dpi,
        )
        plot_future_return_with_epsilon(
            labeled_df["future_return_bps"],
            float(epsilons.get("30s", labeled_df["epsilon"].iloc[0])),
            paths.figures / "future_return_epsilon.png",
            dpi,
        )
        plot_obi_distributions(labeled_df, paths.figures / "obi_distributions.png", dpi)
        plot_obi_vs_return(labeled_df, paths.figures / "obi_vs_future_return.png", dpi)
        plot_microprice_edge(labeled_df, paths.figures / "microprice_edge_distribution.png", dpi)
        plot_correlation_heatmap(
            labeled_df,
            feature_cols[:20],
            paths.figures / "correlation_heatmap.png",
            dpi,
        )
        plot_missing_values(
            feat_df, feature_cols, paths.figures / "missing_value_percentages.png", dpi
        )

        # ------------------------------------------------------------------
        # Stage 5: Baselines
        # ------------------------------------------------------------------
        logger.info("=== Stage 5: Baselines ===")
        baseline_results = run_baselines(
            labeled_df,
            feature_cols,
            masks,
            epsilon=float(epsilons.get("30s", 1.0)),
        )
        save_json(baseline_results, paths.metrics / "baseline_metrics.json")
        summary["baseline_metrics"] = baseline_results
        summary["stages_completed"].append("baselines")

        # ------------------------------------------------------------------
        # Stage 6–7: Logistic + XGBoost + CatBoost
        # ------------------------------------------------------------------
        logger.info("=== Stage 6: Logistic regression ===")
        logistic_results = {}
        if config["models"]["logistic_regression"].get("enabled", True):
            logistic_results = train_logistic(
                labeled_df, feature_cols, masks, config, paths.models
            )
            save_json(
                {k: v for k, v in logistic_results.items() if k.startswith("metrics_")},
                paths.metrics / "logistic_metrics.json",
            )
        summary["logistic_metrics"] = logistic_results
        summary["stages_completed"].append("logistic")

        logger.info("=== Stage 6b: XGBoost ===")
        xgb_results = {}
        if config["models"]["xgboost"].get("enabled", True):
            xgb_results = train_xgboost(
                labeled_df, feature_cols, masks, config, paths.models
            )
            save_json(
                {
                    "best_params": xgb_results.get("best_params"),
                    "best_val_macro_f1": xgb_results.get("best_val_macro_f1"),
                    "metrics_train": xgb_results.get("metrics_train"),
                    "metrics_val": xgb_results.get("metrics_val"),
                    "metrics_test": xgb_results.get("metrics_test"),
                    "software": xgb_results.get("software"),
                    "random_seed": xgb_results.get("random_seed"),
                },
                paths.metrics / "xgboost_metrics.json",
            )
        summary["xgboost_metrics"] = xgb_results
        summary["best_hyperparameters"] = xgb_results.get("best_params")
        summary["stages_completed"].append("xgboost")

        logger.info("=== Stage 7: CatBoost challenger ===")
        cat_results = {}
        if config["models"]["catboost"].get("enabled", True):
            try:
                cat_results = train_catboost(
                    labeled_df, feature_cols, masks, config, paths.models
                )
                save_json(
                    {
                        "best_params": cat_results.get("best_params"),
                        "best_val_macro_f1": cat_results.get("best_val_macro_f1"),
                        "metrics_train": cat_results.get("metrics_train"),
                        "metrics_val": cat_results.get("metrics_val"),
                        "metrics_test": cat_results.get("metrics_test"),
                    },
                    paths.metrics / "catboost_metrics.json",
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("CatBoost failed: %s", exc)
                summary["errors"].append(f"CatBoost failed: {exc}")
                save_json({"error": str(exc)}, paths.metrics / "catboost_metrics.json")
        summary["catboost_metrics"] = cat_results
        summary["stages_completed"].append("catboost")

        # ------------------------------------------------------------------
        # Stage 8: Evaluation plots + non-overlapping test
        # ------------------------------------------------------------------
        logger.info("=== Stage 8: Evaluation ===")
        comparison_rows = []
        for name, key in [
            ("majority", "majority_test"),
            ("previous_direction", "previous_direction_test"),
            ("obi_rule", "obi_rule_test"),
        ]:
            m = baseline_results.get(key, {})
            comparison_rows.append({"model": name, **{k: v for k, v in m.items() if not isinstance(v, list)}})
        for name, res in [
            ("logistic", logistic_results.get("metrics_test", {})),
            ("xgboost", xgb_results.get("metrics_test", {})),
            ("catboost", cat_results.get("metrics_test", {})),
        ]:
            if res:
                comparison_rows.append(
                    {"model": name, **{k: v for k, v in res.items() if not isinstance(v, list)}}
                )
        comparison_df = pd.DataFrame(comparison_rows)
        comparison_df.to_csv(paths.tables / "model_comparison.csv", index=False)

        # Per-class metrics table
        per_class_rows = []
        for model_name, metrics in [
            ("xgboost", xgb_results.get("metrics_test", {})),
            ("logistic", logistic_results.get("metrics_test", {})),
            ("catboost", cat_results.get("metrics_test", {})),
        ]:
            if not metrics:
                continue
            for cls in ["DOWN", "STABLE", "UP"]:
                per_class_rows.append(
                    {
                        "model": model_name,
                        "class": cls,
                        "precision": metrics.get(f"precision_{cls}"),
                        "recall": metrics.get(f"recall_{cls}"),
                        "f1": metrics.get(f"f1_{cls}"),
                        "support": metrics.get(f"support_{cls}"),
                    }
                )
        pd.DataFrame(per_class_rows).to_csv(paths.tables / "per_class_metrics.csv", index=False)

        # Non-overlapping test
        non_idx = non_overlapping_indices(
            labeled_df["timestamp"],
            split.test_idx,
            step_seconds=float(config["split"]["non_overlapping_step_seconds"]),
        )
        non_mask = np.zeros(len(labeled_df), dtype=bool)
        non_mask[non_idx] = True
        non_mask &= valid_label.to_numpy()
        non_metrics = {}
        if "predictions_test" in xgb_results:
            # Rebuild predictions on non-overlapping subset via stored model path
            from src.train_logistic import prepare_xy

            X_non, y_non, idx_non = prepare_xy(labeled_df, feature_cols, non_mask)
            import joblib

            bundle = joblib.load(paths.models / "xgboost_model.joblib")
            model = bundle["model"]
            pred_non = model.predict(X_non)
            proba_non = model.predict_proba(X_non)
            non_metrics = evaluate_classification(y_non, pred_non, proba_non)
            if len(idx_non):
                boot = day_level_bootstrap_ci(
                    labeled_df.loc[idx_non, "timestamp"],
                    y_non,
                    pred_non,
                    proba_non,
                    seed=seed,
                )
                non_metrics["bootstrap_macro_f1"] = boot
        save_json(non_metrics, paths.metrics / "non_overlapping_test_metrics.json")
        save_json(
            xgb_results.get("metrics_test", {}),
            paths.metrics / "test_metrics.json",
        )
        summary["non_overlapping_test_metrics"] = non_metrics

        # Save test predictions parquet
        if "predictions_test" in xgb_results:
            pred = xgb_results["predictions_test"]
            idx = pred["index"]
            pred_df = pd.DataFrame(
                {
                    "timestamp": labeled_df.loc[idx, "timestamp"].to_numpy(),
                    "actual_class": pred["y_true"],
                    "predicted_class": pred["y_pred"],
                    "prob_DOWN": pred["y_proba"][:, 0],
                    "prob_STABLE": pred["y_proba"][:, 1],
                    "prob_UP": pred["y_proba"][:, 2],
                    "mid_price": labeled_df.loc[idx, "mid_price"].to_numpy(),
                    "future_return_bps": labeled_df.loc[idx, "future_return_bps"].to_numpy(),
                }
            )
            for col in ["obi_5", "weighted_obi", "relative_spread_bps", "microprice_edge_bps"]:
                if col in labeled_df.columns:
                    pred_df[col] = labeled_df.loc[idx, col].to_numpy()
            pred_df.to_parquet(paths.tables / "test_predictions.parquet", index=False)

        # Model plots
        if comparison_df is not None and "macro_f1" in comparison_df.columns:
            plot_model_comparison(
                comparison_df.dropna(subset=["macro_f1"]),
                paths.figures / "model_comparison_macro_f1.png",
                dpi,
            )
        if xgb_results.get("metrics_test"):
            m = xgb_results["metrics_test"]
            plot_confusion(
                np.asarray(m["normalized_confusion_matrix"]),
                paths.figures / "xgb_normalized_confusion_test.png",
                "XGBoost Normalized Confusion Matrix (Test)",
                dpi,
            )
            plot_per_class_prf(m, paths.figures / "xgb_per_class_prf.png", dpi)
            pred = xgb_results["predictions_test"]
            plot_roc_ovr(pred["y_true"], pred["y_proba"], paths.figures / "xgb_roc_ovr.png", dpi)
            plot_pr_ovr(pred["y_true"], pred["y_proba"], paths.figures / "xgb_pr_ovr.png", dpi)
            plot_calibration(
                pred["y_true"], pred["y_proba"], paths.figures / "xgb_calibration.png", dpi
            )
            plot_performance_by_date(
                labeled_df.loc[pred["index"], "timestamp"],
                pred["y_true"],
                pred["y_pred"],
                paths.figures / "performance_by_test_date.png",
                dpi,
            )
            plot_performance_by_hour(
                labeled_df.loc[pred["index"], "timestamp"],
                pred["y_true"],
                pred["y_pred"],
                paths.figures / "performance_by_hour.png",
                dpi,
            )
        if xgb_results.get("evals_result"):
            plot_xgb_loss(
                xgb_results["evals_result"],
                paths.figures / "xgb_train_val_loss.png",
                dpi,
            )
        summary["stages_completed"].append("evaluation")

        # ------------------------------------------------------------------
        # Stage 9: SHAP
        # ------------------------------------------------------------------
        logger.info("=== Stage 9: Explainability (SHAP) ===")
        shap_summary = {}
        try:
            import joblib

            bundle = joblib.load(paths.models / "xgboost_model.joblib")
            shap_summary = run_shap_analysis(
                model=bundle["model"],
                df=labeled_df,
                feature_cols=feature_cols,
                test_idx=np.where(masks["test"])[0],
                predictions=xgb_results.get("predictions_test", {}),
                config=config,
                paths=paths,
            )
            save_json(shap_summary, paths.metrics / "shap_summary.json")
        except Exception as exc:  # noqa: BLE001
            logger.exception("SHAP analysis failed: %s", exc)
            summary["errors"].append(f"SHAP failed: {exc}")
            save_json({"error": str(exc), "traceback": traceback.format_exc()}, paths.metrics / "shap_summary.json")
        summary["shap"] = shap_summary
        summary["stages_completed"].append("shap")

        # ------------------------------------------------------------------
        # Stage 10: Ablation + horizon robustness
        # ------------------------------------------------------------------
        logger.info("=== Stage 10: Ablation and robustness ===")
        top_feats = [r["feature"] for r in shap_summary.get("top_features", [])]
        ablation_df = pd.DataFrame()
        if config["ablation"].get("enabled", True):
            ablation_df = run_ablation(
                labeled_df,
                masks,
                config,
                best_params=xgb_results.get("best_params"),
                top_features=top_feats or None,
                tables_dir=paths.tables,
            )
            plot_ablation(ablation_df, paths.figures / "ablation_comparison.png", dpi)
        summary["ablation_table"] = ablation_df

        horizon_df = pd.DataFrame()
        if config["robustness"].get("enabled", True):
            horizon_df = run_horizon_robustness(
                labeled_df,
                masks,
                config,
                best_params=xgb_results.get("best_params"),
                tables_dir=paths.tables,
            )
            if "test_macro_f1" in horizon_df.columns:
                plot_horizon_comparison(
                    horizon_df.dropna(subset=["test_macro_f1"]),
                    paths.figures / "horizon_comparison.png",
                    dpi,
                )
        summary["horizon_table"] = horizon_df

        # Trading sanity
        if "predictions_test" in xgb_results:
            trade = trading_sanity_check(
                labeled_df,
                split.test_idx,
                xgb_results["predictions_test"]["y_pred"],
                xgb_results["predictions_test"]["index"],
                config,
            )
            save_json(trade, paths.metrics / "trading_sanity.json")
            summary["trading_sanity"] = trade
        summary["stages_completed"].append("ablation_robustness")

        # ------------------------------------------------------------------
        # Stage 11: Documentation
        # ------------------------------------------------------------------
        logger.info("=== Stage 11: Documentation ===")
        summary["pipeline_completed"] = True
        write_final_report(paths, summary)
        write_readme(paths, summary)
        save_json(
            {
                k: v
                for k, v in summary.items()
                if k not in {"ablation_table", "horizon_table", "feature_cols"}
            },
            paths.metrics / "pipeline_summary.json",
        )
        summary["stages_completed"].append("documentation")
        logger.info("Pipeline completed successfully")
        return summary

    except Exception as exc:  # noqa: BLE001
        logger.exception("Pipeline failed: %s", exc)
        summary["errors"].append(str(exc))
        summary["traceback"] = traceback.format_exc()
        # Still write pending README if possible
        try:
            write_readme(paths, summary)
            write_final_report(paths, summary)
            save_json(summary, paths.metrics / "pipeline_summary.json")
        except Exception:  # noqa: BLE001
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Explainable BTCIRT LOB pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/project_config.yaml",
        help="Path to YAML config",
    )
    args = parser.parse_args(argv)
    run_pipeline(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

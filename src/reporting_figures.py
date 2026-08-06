"""Generate redesign figures into reports/figures/{data_quality,labels,models,...}."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import visualization as viz

logger = logging.getLogger(__name__)


def generate_redesign_figures(
    *,
    feat: pd.DataFrame,
    study_a_df: pd.DataFrame,
    masks_a: dict[str, np.ndarray],
    primary_cols: list[str],
    meta_a: dict[str, Any],
    xgb: dict[str, Any],
    paths: Any,
    dpi: int = 200,
) -> list[str]:
    """
    Write canonical Study A / data-quality figures for the redesign.

    Returns list of relative paths written.
    """
    written: list[str] = []
    fig = paths.figures
    dq = fig / "data_quality"
    labels = fig / "labels"
    models = fig / "models"
    ablation = fig / "ablation"
    for d in (dq, labels, models, ablation, fig / "shap", fig / "robustness"):
        d.mkdir(parents=True, exist_ok=True)

    def _w(path: Path) -> None:
        written.append(str(path.relative_to(paths.reports) if paths.reports in path.parents else path))

    gaps = feat["timestamp"].sort_values().diff().dt.total_seconds().dropna()
    viz.plot_timestamp_gap_hist(gaps, dq / "observation_gap_hist.png", dpi)
    _w(dq / "observation_gap_hist.png")
    # log-scale gap hist
    import matplotlib.pyplot as plt

    fig_g, ax = plt.subplots(figsize=(9, 5))
    clipped = gaps.clip(upper=gaps.quantile(0.99))
    ax.hist(clipped, bins=50, color="#1f4e79", edgecolor="white")
    ax.set_yscale("log")
    ax.set_xlabel("Timestamp gap (seconds)")
    ax.set_ylabel("Count (log)")
    ax.set_title("Observation Gaps (log-count)")
    viz._save(fig_g, dq / "observation_gap_hist_log.png", dpi)
    _w(dq / "observation_gap_hist_log.png")

    viz.plot_snapshots_by_date(feat, dq / "snapshots_by_date.png", dpi)
    _w(dq / "snapshots_by_date.png")
    viz.plot_snapshots_by_hour(feat, dq / "snapshots_by_hour.png", dpi)
    _w(dq / "snapshots_by_hour.png")
    viz.plot_mid_price(feat, dq / "mid_price_timeseries.png", dpi)
    _w(dq / "mid_price_timeseries.png")
    viz.plot_spread_series(feat, dq / "relative_spread_timeseries.png", dpi)
    _w(dq / "relative_spread_timeseries.png")
    viz.plot_spread_dist(feat, dq / "relative_spread_distribution.png", dpi)
    _w(dq / "relative_spread_distribution.png")
    viz.plot_bid_ask_window(feat, dq / "best_bid_ask_window.png", dpi)
    _w(dq / "best_bid_ask_window.png")
    viz.plot_depth_profile(feat, dq / "depth_profile_example.png", dpi)
    _w(dq / "depth_profile_example.png")
    viz.plot_missing_values(feat, primary_cols, dq / "missing_value_percentages.png", dpi)
    _w(dq / "missing_value_percentages.png")

    # Labels
    dist = pd.read_csv(paths.tables / "study_a_class_distribution.csv")
    viz.plot_class_distribution(dist, labels / "study_a_class_distribution.png", dpi)
    _w(labels / "study_a_class_distribution.png")
    eps = float(meta_a.get("epsilon_bps", 0.0))
    viz.plot_future_return_with_epsilon(
        study_a_df["next_observation_return_bps"],
        eps,
        labels / "next_observation_return_epsilon.png",
        dpi,
    )
    _w(labels / "next_observation_return_epsilon.png")

    # Features
    viz.plot_obi_distributions(feat, labels / "obi_distributions.png", dpi)
    _w(labels / "obi_distributions.png")
    # OBI vs next return
    tmp = study_a_df[["obi_5", "next_observation_return_bps"]].dropna()
    tmp = tmp.rename(columns={"next_observation_return_bps": "future_return_bps"})
    viz.plot_obi_vs_return(tmp, labels / "obi_vs_next_return.png", dpi)
    _w(labels / "obi_vs_next_return.png")
    viz.plot_microprice_edge(feat, labels / "microprice_edge_distribution.png", dpi)
    _w(labels / "microprice_edge_distribution.png")
    viz.plot_correlation_heatmap(
        feat, primary_cols[:25], labels / "feature_correlation_heatmap.png", dpi
    )
    _w(labels / "feature_correlation_heatmap.png")

    # Models
    mc = pd.read_csv(paths.tables / "model_comparison.csv")
    viz.plot_model_comparison(mc, models / "model_comparison_macro_f1.png", dpi)
    _w(models / "model_comparison_macro_f1.png")

    pred = xgb["predictions_development_test"]
    y_true, y_pred, y_proba = pred["y_true"], pred["y_pred"], pred["y_proba"]
    cm = np.asarray(xgb["metrics_development_test"].get("normalized_confusion_matrix"))
    if cm is not None and getattr(cm, "size", 0):
        viz.plot_confusion(
            np.asarray(cm, dtype=float),
            models / "xgb_normalized_confusion_test.png",
            "XGBoost Normalized Confusion (development_test)",
            dpi,
        )
        _w(models / "xgb_normalized_confusion_test.png")
    viz.plot_per_class_prf(xgb["metrics_development_test"], models / "xgb_per_class_prf.png", dpi)
    _w(models / "xgb_per_class_prf.png")
    viz.plot_roc_ovr(y_true, y_proba, models / "xgb_roc_ovr.png", dpi)
    _w(models / "xgb_roc_ovr.png")
    viz.plot_pr_ovr(y_true, y_proba, models / "xgb_pr_ovr.png", dpi)
    _w(models / "xgb_pr_ovr.png")
    viz.plot_calibration(y_true, y_proba, models / "xgb_calibration.png", dpi)
    _w(models / "xgb_calibration.png")
    if xgb.get("evals_result"):
        viz.plot_xgb_loss(xgb["evals_result"], models / "xgb_train_val_loss.png", dpi)
        _w(models / "xgb_train_val_loss.png")

    ts = study_a_df.loc[pred["index"], "timestamp"]
    viz.plot_performance_by_date(
        ts, y_true, y_pred, models / "performance_by_date.png", dpi
    )
    _w(models / "performance_by_date.png")
    viz.plot_performance_by_hour(
        ts, y_true, y_pred, models / "performance_by_hour.png", dpi
    )
    _w(models / "performance_by_hour.png")

    # Ablation / feature sets
    fs = pd.read_csv(paths.tables / "feature_set_comparison.csv")
    fig_a, ax = __import__("matplotlib.pyplot", fromlist=["pyplot"]).subplots(figsize=(10, 5))
    ax.bar(fs["experiment"], fs["test_macro_f1"], color="#1f4e79")
    ax.set_ylabel("Macro F1 (development_test)")
    ax.set_title("Feature-set / Ablation Comparison")
    ax.tick_params(axis="x", rotation=35)
    viz._save(fig_a, ablation / "feature_set_comparison.png", dpi)
    _w(ablation / "feature_set_comparison.png")

    logger.info("Wrote %s redesign figures", len(written))
    return written

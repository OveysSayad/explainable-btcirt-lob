"""Plotting utilities for EDA and model evaluation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from sklearn.preprocessing import label_binarize

logger = logging.getLogger(__name__)


def _save(fig: plt.Figure, path: Path, dpi: int = 200) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure %s", path)


def plot_timestamp_gap_hist(gaps: pd.Series, path: Path, dpi: int = 200) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    clipped = gaps.clip(upper=gaps.quantile(0.99))
    ax.hist(clipped, bins=50, color="#1f4e79", edgecolor="white")
    ax.set_xlabel("Timestamp gap (seconds)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of BTCIRT Snapshot Timestamp Gaps")
    _save(fig, path, dpi)


def plot_snapshots_by_date(df: pd.DataFrame, path: Path, dpi: int = 200) -> None:
    counts = df.groupby(df["timestamp"].dt.date).size()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(counts.index.astype(str), counts.values, color="#1f4e79")
    ax.set_xlabel("Date")
    ax.set_ylabel("Snapshots")
    ax.set_title("Number of BTCIRT Snapshots by Date")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    _save(fig, path, dpi)


def plot_snapshots_by_hour(df: pd.DataFrame, path: Path, dpi: int = 200) -> None:
    counts = df.groupby(df["timestamp"].dt.hour).size()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(counts.index, counts.values, color="#2e75b6")
    ax.set_xlabel("Hour of day (UTC)")
    ax.set_ylabel("Snapshots")
    ax.set_title("Number of BTCIRT Snapshots by Hour")
    _save(fig, path, dpi)


def plot_mid_price(df: pd.DataFrame, path: Path, dpi: int = 200) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["timestamp"], df["mid_price"], color="#1f4e79", linewidth=0.8)
    ax.set_xlabel("Timestamp (UTC)")
    ax.set_ylabel("Mid-price (IRT)")
    ax.set_title("BTCIRT Mid-Price Time Series")
    _save(fig, path, dpi)


def plot_spread_series(df: pd.DataFrame, path: Path, dpi: int = 200) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["timestamp"], df["relative_spread_bps"], color="#c45911", linewidth=0.7)
    ax.set_xlabel("Timestamp (UTC)")
    ax.set_ylabel("Relative spread (bps)")
    ax.set_title("BTCIRT Relative Bid-Ask Spread")
    _save(fig, path, dpi)


def plot_spread_dist(df: pd.DataFrame, path: Path, dpi: int = 200) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    s = df["relative_spread_bps"].replace([np.inf, -np.inf], np.nan).dropna()
    ax.hist(s.clip(upper=s.quantile(0.99)), bins=50, color="#c45911", edgecolor="white")
    ax.set_xlabel("Relative spread (bps)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Relative Spread")
    _save(fig, path, dpi)


def plot_bid_ask_window(df: pd.DataFrame, path: Path, dpi: int = 200, n: int = 400) -> None:
    sub = df.iloc[:n]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(sub["timestamp"], sub["best_ask"], label="Best ask", color="#c00000")
    ax.plot(sub["timestamp"], sub["best_bid"], label="Best bid", color="#00b050")
    ax.set_xlabel("Timestamp (UTC)")
    ax.set_ylabel("Price (IRT)")
    ax.set_title("Best Bid and Ask over a Selected Interval")
    ax.legend()
    _save(fig, path, dpi)


def plot_depth_profile(df: pd.DataFrame, path: Path, dpi: int = 200) -> None:
    row = df.iloc[len(df) // 2]
    ask_qty = [row[f"asks_qty_{i}"] for i in range(1, 9)]
    bid_qty = [row[f"bids_qty_{i}"] for i in range(1, 9)]
    levels = np.arange(1, 9)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(levels - 0.15, bid_qty, width=0.3, label="Bid qty", color="#00b050")
    ax.bar(levels + 0.15, ask_qty, width=0.3, label="Ask qty", color="#c00000")
    ax.set_xlabel("Order-book level")
    ax.set_ylabel("Quantity")
    ax.set_title("Example BTCIRT Depth Profile (Levels 1–8)")
    ax.legend()
    _save(fig, path, dpi)


def plot_class_distribution(dist_df: pd.DataFrame, path: Path, dpi: int = 200) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    piv = dist_df.pivot(index="split", columns="class_name", values="percentage")
    piv.plot(kind="bar", ax=ax, color=["#c00000", "#7f7f7f", "#00b050"])
    ax.set_ylabel("Percentage")
    ax.set_title("Label Distribution by Split")
    ax.legend(title="Class")
    _save(fig, path, dpi)


def plot_future_return_with_epsilon(
    returns: pd.Series, epsilon: float, path: Path, dpi: int = 200
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    r = returns.dropna()
    ax.hist(r.clip(r.quantile(0.01), r.quantile(0.99)), bins=60, color="#1f4e79", edgecolor="white")
    ax.axvline(-epsilon, color="#c00000", linestyle="--", label=f"-ε={-epsilon:.2f}")
    ax.axvline(epsilon, color="#00b050", linestyle="--", label=f"+ε={epsilon:.2f}")
    ax.set_xlabel("Future return (bps)")
    ax.set_ylabel("Count")
    ax.set_title("Future Mid-Price Return with STABLE Thresholds")
    ax.legend()
    _save(fig, path, dpi)


def plot_obi_distributions(df: pd.DataFrame, path: Path, dpi: int = 200) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, col in zip(axes, ["obi_1", "obi_5", "obi_8"]):
        ax.hist(df[col].dropna(), bins=40, color="#2e75b6", edgecolor="white")
        ax.set_title(col)
        ax.set_xlabel("OBI")
    fig.suptitle("Order Book Imbalance Distributions")
    fig.tight_layout()
    _save(fig, path, dpi)


def plot_obi_vs_return(df: pd.DataFrame, path: Path, dpi: int = 200) -> None:
    sub = df[["obi_5", "future_return_bps"]].dropna()
    if len(sub) > 20000:
        sub = sub.sample(20000, random_state=42)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(sub["obi_5"], sub["future_return_bps"], s=5, alpha=0.25, color="#1f4e79")
    ax.set_xlabel("obi_5")
    ax.set_ylabel("Future return (bps)")
    ax.set_title("OBI vs Future Mid-Price Return")
    _save(fig, path, dpi)


def plot_microprice_edge(df: pd.DataFrame, path: Path, dpi: int = 200) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    s = df["microprice_edge_bps"].dropna()
    ax.hist(s.clip(s.quantile(0.01), s.quantile(0.99)), bins=50, color="#7030a0", edgecolor="white")
    ax.set_xlabel("Microprice edge (bps)")
    ax.set_ylabel("Count")
    ax.set_title("Microprice Edge Distribution")
    _save(fig, path, dpi)


def plot_correlation_heatmap(
    df: pd.DataFrame, cols: list[str], path: Path, dpi: int = 200
) -> None:
    cols = [c for c in cols if c in df.columns][:25]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90, fontsize=7)
    ax.set_yticklabels(cols, fontsize=7)
    ax.set_title("Feature Correlation Heatmap")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    _save(fig, path, dpi)


def plot_missing_values(df: pd.DataFrame, cols: list[str], path: Path, dpi: int = 200) -> None:
    miss = 100.0 * df[cols].isna().mean().sort_values(ascending=False).head(30)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(miss.index[::-1], miss.values[::-1], color="#833c0c")
    ax.set_xlabel("Missing (%)")
    ax.set_title("Missing-Value Percentages by Feature")
    _save(fig, path, dpi)


def plot_model_comparison(metrics_df: pd.DataFrame, path: Path, dpi: int = 200) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(metrics_df["model"], metrics_df["macro_f1"], color="#1f4e79")
    ax.set_ylabel("Macro F1")
    ax.set_title("Model Comparison — Macro F1 (Test)")
    ax.tick_params(axis="x", rotation=30)
    _save(fig, path, dpi)


def plot_confusion(cm: np.ndarray, path: Path, title: str, dpi: int = 200) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks([0, 1, 2], ["DOWN", "STABLE", "UP"])
    ax.set_yticks([0, 1, 2], ["DOWN", "STABLE", "UP"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, path, dpi)


def plot_per_class_prf(metrics: dict[str, Any], path: Path, dpi: int = 200) -> None:
    classes = ["DOWN", "STABLE", "UP"]
    precision = [metrics.get(f"precision_{c}", 0) for c in classes]
    recall = [metrics.get(f"recall_{c}", 0) for c in classes]
    f1 = [metrics.get(f"f1_{c}", 0) for c in classes]
    x = np.arange(len(classes))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, precision, width, label="Precision")
    ax.bar(x, recall, width, label="Recall")
    ax.bar(x + width, f1, width, label="F1")
    ax.set_xticks(x, classes)
    ax.set_ylim(0, 1)
    ax.set_title("Per-Class Precision, Recall, and F1")
    ax.legend()
    _save(fig, path, dpi)


def plot_roc_ovr(
    y_true: np.ndarray, y_proba: np.ndarray, path: Path, dpi: int = 200
) -> None:
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, name in enumerate(["DOWN", "STABLE", "UP"]):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("One-vs-Rest ROC Curves")
    ax.legend()
    _save(fig, path, dpi)


def plot_pr_ovr(
    y_true: np.ndarray, y_proba: np.ndarray, path: Path, dpi: int = 200
) -> None:
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, name in enumerate(["DOWN", "STABLE", "UP"]):
        p, r, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
        ax.plot(r, p, label=name)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("One-vs-Rest Precision-Recall Curves")
    ax.legend()
    _save(fig, path, dpi)


def plot_calibration(
    y_true: np.ndarray, y_proba: np.ndarray, path: Path, dpi: int = 200
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, name in enumerate(["DOWN", "STABLE", "UP"]):
        try:
            pt, pp = calibration_curve((y_true == i).astype(int), y_proba[:, i], n_bins=10)
            ax.plot(pp, pt, marker="o", label=name)
        except Exception:  # noqa: BLE001
            continue
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curves (OvR)")
    ax.legend()
    _save(fig, path, dpi)


def plot_xgb_loss(evals: dict[str, Any], path: Path, dpi: int = 200) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    # xgboost evals_result keys vary by version
    for split_name, metrics in evals.items():
        for metric_name, values in metrics.items():
            ax.plot(values, label=f"{split_name}-{metric_name}")
    ax.set_xlabel("Boosting round")
    ax.set_ylabel("Loss / metric")
    ax.set_title("XGBoost Training and Validation Curves")
    ax.legend()
    _save(fig, path, dpi)


def plot_performance_by_date(
    timestamps: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path: Path,
    dpi: int = 200,
) -> None:
    from sklearn.metrics import f1_score

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(timestamps).dt.date.astype(str),
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    scores = df.groupby("date").apply(
        lambda g: f1_score(g["y_true"], g["y_pred"], average="macro", zero_division=0)
        if len(g) > 5
        else np.nan
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(scores.index, scores.values, marker="o", color="#1f4e79")
    ax.set_xlabel("Test date")
    ax.set_ylabel("Macro F1")
    ax.set_title("Performance by Test Date")
    ax.tick_params(axis="x", rotation=45)
    _save(fig, path, dpi)


def plot_performance_by_hour(
    timestamps: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path: Path,
    dpi: int = 200,
) -> None:
    from sklearn.metrics import f1_score

    df = pd.DataFrame(
        {
            "hour": pd.to_datetime(timestamps).dt.hour,
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    scores = df.groupby("hour").apply(
        lambda g: f1_score(g["y_true"], g["y_pred"], average="macro", zero_division=0)
        if len(g) > 5
        else np.nan
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(scores.index, scores.values, color="#2e75b6")
    ax.set_xlabel("Hour of day (UTC)")
    ax.set_ylabel("Macro F1")
    ax.set_title("Performance by Hour of Day")
    _save(fig, path, dpi)


def plot_horizon_comparison(horizon_df: pd.DataFrame, path: Path, dpi: int = 200) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        horizon_df["horizon_seconds"].astype(str) + "s",
        horizon_df["test_macro_f1"],
        color="#1f4e79",
    )
    ax.set_ylabel("Test Macro F1")
    ax.set_title("Performance Across Forecast Horizons")
    _save(fig, path, dpi)


def plot_ablation(ablation_df: pd.DataFrame, path: Path, dpi: int = 200) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(ablation_df["experiment"], ablation_df["test_macro_f1"], color="#1f4e79")
    ax.set_xlabel("Test Macro F1")
    ax.set_title("Feature-Family Ablation Comparison")
    _save(fig, path, dpi)

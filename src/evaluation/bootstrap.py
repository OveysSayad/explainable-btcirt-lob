"""Day-level bootstrap confidence intervals."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.classification_metrics import evaluate_classification


def day_level_bootstrap_ci(
    timestamps: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
    metric: str = "macro_f1",
    labels: list[int] | None = None,
) -> dict[str, Any]:
    """Bootstrap by resampling calendar dates with replacement."""
    dates = pd.to_datetime(timestamps).dt.date.astype(str).to_numpy()
    unique = np.unique(dates)
    if len(unique) < 3:
        return {"n_days": float(len(unique)), "note": "insufficient days"}
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_bootstrap):
        sample = rng.choice(unique, size=len(unique), replace=True)
        mask = np.isin(dates, sample)
        if mask.sum() == 0:
            continue
        proba = y_proba[mask] if y_proba is not None else None
        m = evaluate_classification(y_true[mask], y_pred[mask], proba, labels=labels)
        if metric in m and np.isfinite(m[metric]):
            scores.append(m[metric])
    if not scores:
        return {"n_days": float(len(unique)), "note": "bootstrap failed"}
    arr = np.asarray(scores)
    return {
        "n_days": float(len(unique)),
        "metric": metric,
        "n_bootstrap": len(scores),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "ci_low": float(np.quantile(arr, 0.025)),
        "ci_high": float(np.quantile(arr, 0.975)),
    }


def compare_models_bootstrap(
    timestamps: pd.Series,
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
    labels: list[int] | None = None,
) -> dict[str, Any]:
    """Day-level bootstrap of Macro-F1 difference (A - B)."""
    dates = pd.to_datetime(timestamps).dt.date.astype(str).to_numpy()
    unique = np.unique(dates)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_bootstrap):
        sample = rng.choice(unique, size=len(unique), replace=True)
        mask = np.isin(dates, sample)
        if mask.sum() == 0:
            continue
        fa = evaluate_classification(y_true[mask], pred_a[mask], labels=labels)["macro_f1"]
        fb = evaluate_classification(y_true[mask], pred_b[mask], labels=labels)["macro_f1"]
        diffs.append(fa - fb)
    arr = np.asarray(diffs)
    return {
        "mean_delta_f1": float(arr.mean()),
        "median_delta_f1": float(np.median(arr)),
        "ci_low": float(np.quantile(arr, 0.025)),
        "ci_high": float(np.quantile(arr, 0.975)),
        "prob_delta_gt_0": float(np.mean(arr > 0)),
    }

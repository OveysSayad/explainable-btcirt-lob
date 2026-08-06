"""Evaluation metrics and bootstrap confidence intervals."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

CLASS_NAMES = ["DOWN", "STABLE", "UP"]


def _safe_log_loss(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    try:
        return float(log_loss(y_true, y_proba, labels=[0, 1, 2]))
    except Exception as exc:  # noqa: BLE001
        logger.warning("log_loss failed: %s", exc)
        return float("nan")


def _safe_roc_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    try:
        return float(
            roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro", labels=[0, 1, 2])
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("ROC-AUC failed: %s", exc)
        return float("nan")


def _multiclass_brier(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Mean one-vs-rest Brier score across classes."""
    scores = []
    for c in range(3):
        y_bin = (y_true == c).astype(int)
        scores.append(brier_score_loss(y_bin, y_proba[:, c]))
    return float(np.mean(scores))


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute a full suite of multiclass classification metrics."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    valid = ~np.isnan(y_true.astype(float))
    y_true = y_true[valid].astype(int)
    y_pred = y_pred[valid].astype(int)
    if y_proba is not None:
        y_proba = np.asarray(y_proba)[valid]

    if len(y_true) == 0:
        return {"n": 0, "error": "empty evaluation set"}

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )
    metrics: dict[str, Any] = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan"),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
        "normalized_confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=[0, 1, 2], normalize="true"
        ).tolist(),
    }
    for i, name in enumerate(CLASS_NAMES):
        metrics[f"precision_{name}"] = float(precision[i])
        metrics[f"recall_{name}"] = float(recall[i])
        metrics[f"f1_{name}"] = float(f1[i])
        metrics[f"support_{name}"] = int(support[i])

    if y_proba is not None:
        metrics["log_loss"] = _safe_log_loss(y_true, y_proba)
        metrics["brier_score"] = _multiclass_brier(y_true, y_proba)
        metrics["roc_auc_ovr_macro"] = _safe_roc_auc(y_true, y_proba)
    return metrics


def day_level_bootstrap_ci(
    timestamps: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    n_bootstrap: int = 200,
    seed: int = 42,
    metric: str = "macro_f1",
) -> dict[str, float]:
    """Day-level bootstrap confidence interval for a chosen metric."""
    dates = pd.to_datetime(timestamps).dt.date.astype(str).to_numpy()
    unique_dates = np.unique(dates)
    if len(unique_dates) < 3:
        return {"n_days": float(len(unique_dates)), "note": "insufficient days"}

    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_bootstrap):
        sample_dates = rng.choice(unique_dates, size=len(unique_dates), replace=True)
        mask = np.isin(dates, sample_dates)
        if mask.sum() == 0:
            continue
        proba = y_proba[mask] if y_proba is not None else None
        m = evaluate_classification(y_true[mask], y_pred[mask], proba)
        if metric in m and np.isfinite(m[metric]):
            scores.append(m[metric])
    if not scores:
        return {"n_days": float(len(unique_dates)), "note": "bootstrap failed"}
    arr = np.asarray(scores)
    return {
        "n_days": float(len(unique_dates)),
        "metric": metric,
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "ci_low": float(np.quantile(arr, 0.025)),
        "ci_high": float(np.quantile(arr, 0.975)),
    }


def metrics_to_frame(metrics_by_model: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Flatten nested metrics dict into a comparison table."""
    rows = []
    for name, metrics in metrics_by_model.items():
        row = {"model": name}
        for k, v in metrics.items():
            if isinstance(v, (list, dict)):
                continue
            row[k] = v
        rows.append(row)
    return pd.DataFrame(rows)

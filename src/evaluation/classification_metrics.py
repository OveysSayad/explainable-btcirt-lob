"""Classification metrics for multiclass and binary studies."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
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


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    labels: list[int] | None = None,
) -> dict[str, Any]:
    """Compute classification metrics; Macro F1 is the primary metric."""
    labels = labels or [0, 1, 2]
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if len(y_true) == 0:
        return {"n": 0, "error": "empty"}

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    metrics: dict[str, Any] = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0, labels=labels)
        ),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)
        ),
        "mcc": float(matthews_corrcoef(y_true, y_pred))
        if len(np.unique(y_true)) > 1
        else float("nan"),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "normalized_confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=labels, normalize="true"
        ).tolist(),
    }
    name_map = {0: "DOWN", 1: "STABLE", 2: "UP"}
    if labels == [0, 1]:
        name_map = {0: "DOWN", 1: "UP"}
    for i, lab in enumerate(labels):
        name = name_map.get(lab, str(lab))
        metrics[f"precision_{name}"] = float(precision[i])
        metrics[f"recall_{name}"] = float(recall[i])
        metrics[f"f1_{name}"] = float(f1[i])
        metrics[f"support_{name}"] = int(support[i])

    if y_proba is not None:
        y_proba = np.asarray(y_proba)
        try:
            metrics["log_loss"] = float(log_loss(y_true, y_proba, labels=labels))
        except Exception as exc:  # noqa: BLE001
            metrics["log_loss"] = float("nan")
            logger.warning("log_loss failed: %s", exc)
        try:
            if y_proba.shape[1] == 2:
                metrics["brier_score"] = float(brier_score_loss(y_true, y_proba[:, 1]))
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
                metrics["average_precision"] = float(
                    average_precision_score(y_true, y_proba[:, 1])
                )
            else:
                briers = [
                    brier_score_loss((y_true == lab).astype(int), y_proba[:, i])
                    for i, lab in enumerate(labels)
                ]
                metrics["brier_score"] = float(np.mean(briers))
                metrics["roc_auc_ovr_macro"] = float(
                    roc_auc_score(
                        y_true, y_proba, multi_class="ovr", average="macro", labels=labels
                    )
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("probabilistic metrics failed: %s", exc)
    return metrics

"""Naive and rule-based baselines."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.metrics import f1_score

from src.evaluation.classification_metrics import evaluate_classification

logger = logging.getLogger(__name__)


def run_baselines(
    y_train: np.ndarray,
    y_eval: dict[str, np.ndarray],
    obi_train: np.ndarray | None = None,
    obi_eval: dict[str, np.ndarray] | None = None,
    prev_dir_eval: dict[str, np.ndarray] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Fit majority / stratified / OBI-rule / previous-direction baselines."""
    classes, counts = np.unique(y_train.astype(int), return_counts=True)
    majority = int(classes[np.argmax(counts)])
    probs = counts / counts.sum()
    rng = np.random.default_rng(seed)
    results: dict[str, Any] = {"majority_class": majority}

    for split, y in y_eval.items():
        n = len(y)
        maj_pred = np.full(n, majority)
        results[f"majority_{split}"] = evaluate_classification(
            y, maj_pred, _onehot(maj_pred, classes)
        )
        strat = rng.choice(classes, size=n, p=probs)
        results[f"stratified_{split}"] = evaluate_classification(
            y, strat, _onehot(strat, classes)
        )
        if prev_dir_eval and split in prev_dir_eval:
            pred = prev_dir_eval[split]
            results[f"previous_direction_{split}"] = evaluate_classification(
                y, pred, _onehot(pred, classes)
            )

    if obi_train is not None and obi_eval is not None:
        best_t, best_s = 0.1, -1.0
        for t in np.linspace(0.01, 0.5, 25):
            pred = _obi_pred(obi_train, t)
            s = f1_score(y_train.astype(int), pred, average="macro", zero_division=0)
            if s > best_s:
                best_s, best_t = s, float(t)
        results["obi_threshold"] = best_t
        for split, obi in obi_eval.items():
            pred = _obi_pred(obi, best_t)
            results[f"obi_rule_{split}"] = evaluate_classification(
                y_eval[split], pred, _onehot(pred, classes)
            )
    return results


def _obi_pred(obi: np.ndarray, thr: float) -> np.ndarray:
    pred = np.full(len(obi), 1, dtype=int)
    pred[obi > thr] = 2
    pred[obi < -thr] = 0
    return pred


def _onehot(pred: np.ndarray, classes: np.ndarray) -> np.ndarray:
    # Always 3 columns for consistency in multiclass studies
    proba = np.zeros((len(pred), 3))
    for i, p in enumerate(pred.astype(int)):
        if 0 <= p <= 2:
            proba[i, p] = 1.0
    return proba

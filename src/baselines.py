"""Simple baseline predictors for short-horizon direction."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)


class MajorityBaseline:
    """Always predict the most frequent training class."""

    def __init__(self) -> None:
        self.majority_class_: int = 1

    def fit(self, y: np.ndarray) -> MajorityBaseline:
        values, counts = np.unique(y[~np.isnan(y)].astype(int), return_counts=True)
        self.majority_class_ = int(values[np.argmax(counts)])
        return self

    def predict(self, n: int) -> np.ndarray:
        return np.full(n, self.majority_class_, dtype=int)

    def predict_proba(self, n: int) -> np.ndarray:
        proba = np.zeros((n, 3), dtype=float)
        proba[:, self.majority_class_] = 1.0
        return proba


class PreviousDirectionBaseline:
    """Predict using the most recent historical mid-price direction."""

    def __init__(self, epsilon: float) -> None:
        self.epsilon = epsilon

    def predict(
        self, mid_price: pd.Series, timestamps: pd.Series
    ) -> tuple[np.ndarray, np.ndarray]:
        prev = mid_price.shift(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            hist_ret = 10_000.0 * np.log(mid_price / prev)
        pred = np.full(len(mid_price), 1, dtype=float)  # default STABLE
        valid = hist_ret.notna()
        r = hist_ret.to_numpy()
        pred[valid & (r < -self.epsilon)] = 0
        pred[valid & (np.abs(r) <= self.epsilon)] = 1
        pred[valid & (r > self.epsilon)] = 2
        # First row has no history -> STABLE
        pred[0] = 1
        proba = np.zeros((len(pred), 3))
        for i, p in enumerate(pred.astype(int)):
            proba[i, p] = 1.0
        return pred.astype(int), proba


class OBIRuleBaseline:
    """Thresholded OBI rule learned from training data."""

    def __init__(self) -> None:
        self.threshold_: float = 0.1

    def fit(
        self,
        obi: np.ndarray,
        y: np.ndarray,
        candidate_thresholds: np.ndarray | None = None,
    ) -> OBIRuleBaseline:
        if candidate_thresholds is None:
            candidate_thresholds = np.linspace(0.01, 0.5, 25)
        best_t = candidate_thresholds[0]
        best_score = -1.0
        y = y.astype(int)
        for t in candidate_thresholds:
            pred = self._predict_with_threshold(obi, float(t))
            score = f1_score(y, pred, average="macro", zero_division=0)
            if score > best_score:
                best_score = score
                best_t = float(t)
        self.threshold_ = best_t
        logger.info("OBI rule threshold=%.4f (macro-F1=%.4f on fit set)", best_t, best_score)
        return self

    @staticmethod
    def _predict_with_threshold(obi: np.ndarray, threshold: float) -> np.ndarray:
        pred = np.full(len(obi), 1, dtype=int)
        pred[obi > threshold] = 2
        pred[obi < -threshold] = 0
        return pred

    def predict(self, obi: np.ndarray) -> np.ndarray:
        return self._predict_with_threshold(obi, self.threshold_)

    def predict_proba(self, obi: np.ndarray) -> np.ndarray:
        pred = self.predict(obi)
        proba = np.zeros((len(pred), 3))
        for i, p in enumerate(pred):
            proba[i, p] = 1.0
        return proba


def run_baselines(
    df: pd.DataFrame,
    feature_cols: list[str],
    split_masks: dict[str, np.ndarray],
    epsilon: float,
) -> dict[str, Any]:
    """Fit baselines on train and evaluate on val/test."""
    from src.evaluation import evaluate_classification

    y = df["label"].to_numpy()
    results: dict[str, Any] = {}

    # Majority
    maj = MajorityBaseline().fit(y[split_masks["train"]])
    for split_name, mask in split_masks.items():
        pred = maj.predict(int(mask.sum()))
        proba = maj.predict_proba(int(mask.sum()))
        metrics = evaluate_classification(y[mask], pred, proba)
        results[f"majority_{split_name}"] = metrics
    results["majority_class"] = maj.majority_class_

    # Previous direction (uses only past mid)
    prev = PreviousDirectionBaseline(epsilon=epsilon)
    pred_all, proba_all = prev.predict(df["mid_price"], df["timestamp"])
    for split_name, mask in split_masks.items():
        metrics = evaluate_classification(y[mask], pred_all[mask], proba_all[mask])
        results[f"previous_direction_{split_name}"] = metrics

    # OBI rule: tune threshold on validation
    obi = df["obi_5"].to_numpy()
    rule = OBIRuleBaseline()
    # Fit threshold using validation if available else train
    fit_mask = split_masks["val"] if split_masks["val"].any() else split_masks["train"]
    rule.fit(obi[fit_mask], y[fit_mask])
    for split_name, mask in split_masks.items():
        pred = rule.predict(obi[mask])
        proba = rule.predict_proba(obi[mask])
        metrics = evaluate_classification(y[mask], pred, proba)
        results[f"obi_rule_{split_name}"] = metrics
    results["obi_rule_threshold"] = rule.threshold_

    return results

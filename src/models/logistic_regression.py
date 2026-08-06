"""Logistic regression trainer."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.classification_metrics import evaluate_classification

logger = logging.getLogger(__name__)


def prepare_xy(
    df: pd.DataFrame,
    feature_cols: list[str],
    mask: np.ndarray,
    label_col: str = "label",
) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    """Extract finite features/labels; impute with split medians only."""
    sub = df.loc[mask, feature_cols + [label_col]].dropna(subset=[label_col])
    X = sub[feature_cols].replace([np.inf, -np.inf], np.nan)
    med = X.median(numeric_only=True)
    X = X.fillna(med).fillna(0.0)
    y = sub[label_col].astype(int).to_numpy()
    return X.to_numpy(dtype=float), y, sub.index


def train_logistic(
    df: pd.DataFrame,
    feature_cols: list[str],
    masks: dict[str, np.ndarray],
    config: dict[str, Any],
    models_dir: Path,
    label_col: str = "label",
    binary: bool = False,
) -> dict[str, Any]:
    """Train scaled logistic regression (multinomial or binary)."""
    X_train, y_train, _ = prepare_xy(df, feature_cols, masks["train"], label_col)
    mapping = None
    if binary:
        mapping = {0: 0, 2: 1}
        y_train = np.array([mapping[int(v)] for v in y_train])

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    solver="lbfgs",
                    penalty="l2",
                    C=1.0,
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=int(config["project"]["random_seed"]),
                ),
            ),
        ]
    )
    pipe.fit(X_train, y_train)
    results: dict[str, Any] = {"feature_cols": feature_cols, "binary": binary}
    for split, mask in masks.items():
        X, y, idx = prepare_xy(df, feature_cols, mask, label_col)
        if binary:
            y = np.array([mapping[int(v)] for v in y])
        pred = pipe.predict(X)
        proba = pipe.predict_proba(X)
        metrics = evaluate_classification(
            y, pred, proba, labels=[0, 1] if binary else [0, 1, 2]
        )
        results[f"metrics_{split}"] = metrics
        results[f"predictions_{split}"] = {
            "index": idx.to_numpy(),
            "y_true": y,
            "y_pred": pred,
            "y_proba": proba,
        }
    clf = pipe.named_steps["clf"]
    coef = pd.DataFrame(clf.coef_.T, columns=[f"coef_{c}" for c in clf.classes_])
    coef.insert(0, "feature", feature_cols)
    results["coefficients"] = coef
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, models_dir / "logistic_regression.joblib")
    logger.info(
        "Logistic val macro-F1=%.4f",
        results.get("metrics_val", {}).get("macro_f1", float("nan")),
    )
    return results

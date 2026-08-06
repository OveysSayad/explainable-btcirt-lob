"""Multinomial logistic regression baseline."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation import evaluate_classification

logger = logging.getLogger(__name__)


def prepare_xy(
    df: pd.DataFrame,
    feature_cols: list[str],
    mask: np.ndarray,
    label_col: str = "label",
) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    """Extract finite feature matrix and labels for a split mask."""
    sub = df.loc[mask, feature_cols + [label_col]].copy()
    sub = sub.dropna(subset=[label_col])
    X = sub[feature_cols].replace([np.inf, -np.inf], np.nan)
    # Median impute within split to avoid leakage across splits at call sites
    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0.0)
    y = sub[label_col].astype(int).to_numpy()
    return X.to_numpy(dtype=float), y, sub.index


def train_logistic(
    df: pd.DataFrame,
    feature_cols: list[str],
    split_masks: dict[str, np.ndarray],
    config: dict[str, Any],
    models_dir: Path,
) -> dict[str, Any]:
    """Train scaled multinomial logistic regression on training data only."""
    cfg = config["models"]["logistic_regression"]
    X_train, y_train, _ = prepare_xy(df, feature_cols, split_masks["train"])
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    solver="lbfgs",
                    penalty=cfg.get("penalty", "l2"),
                    C=float(cfg.get("C", 1.0)),
                    max_iter=int(cfg.get("max_iter", 2000)),
                    class_weight=cfg.get("class_weight", "balanced"),
                    random_state=int(config["project"]["random_seed"]),
                ),
            ),
        ]
    )
    pipe.fit(X_train, y_train)

    results: dict[str, Any] = {"feature_cols": feature_cols}
    for split_name, mask in split_masks.items():
        X, y, idx = prepare_xy(df, feature_cols, mask)
        if len(y) == 0:
            continue
        pred = pipe.predict(X)
        proba = pipe.predict_proba(X)
        results[f"metrics_{split_name}"] = evaluate_classification(y, pred, proba)
        results[f"predictions_{split_name}"] = {
            "index": idx.to_numpy(),
            "y_true": y,
            "y_pred": pred,
            "y_proba": proba,
        }

    clf: LogisticRegression = pipe.named_steps["clf"]
    coef = clf.coef_  # shape (n_classes, n_features)
    coef_df = pd.DataFrame(coef.T, columns=[f"coef_{c}" for c in clf.classes_])
    coef_df.insert(0, "feature", feature_cols)
    for c in clf.classes_:
        coef_df[f"sign_{c}"] = np.sign(coef_df[f"coef_{c}"])
    results["coefficients"] = coef_df

    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, models_dir / "logistic_regression.joblib")
    coef_df.to_csv(
        Path(config.get("_project_root", "."))
        / "reports"
        / "tables"
        / "logistic_coefficients.csv",
        index=False,
    )
    logger.info(
        "Logistic regression val macro-F1=%.4f",
        results.get("metrics_val", {}).get("macro_f1", float("nan")),
    )
    return results

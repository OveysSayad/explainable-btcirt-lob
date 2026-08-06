"""CatBoost challenger trainer."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from catboost import CatBoostClassifier
from sklearn.utils.class_weight import compute_class_weight

from src.evaluation.classification_metrics import evaluate_classification
from src.models.logistic_regression import prepare_xy

logger = logging.getLogger(__name__)


def _sample_params(rng: np.random.Generator) -> dict[str, Any]:
    # Bernoulli bootstrap allows subsample; bagging_temperature requires Bayesian.
    return {
        "depth": int(rng.choice([4, 5, 6, 7, 8])),
        "learning_rate": float(rng.choice([0.01, 0.03, 0.05, 0.075, 0.10])),
        "iterations": int(rng.choice([300, 500, 750, 1000])),
        "l2_leaf_reg": float(rng.choice([1, 3, 5, 10, 20, 50])),
        "random_strength": float(rng.choice([0.0, 0.5, 1.0, 2.0])),
        "border_count": int(rng.choice([32, 64, 128])),
        "subsample": float(rng.choice([0.7, 0.8, 0.9, 1.0])),
    }


def train_catboost(
    df: Any,
    feature_cols: list[str],
    masks: dict[str, np.ndarray],
    config: dict[str, Any],
    models_dir: Path,
    n_trials: int = 25,
    label_col: str = "label",
    binary: bool = False,
    model_name: str = "catboost_model",
) -> dict[str, Any]:
    """Train CatBoost with Bernoulli bootstrap to allow subsample."""
    seed = int(config["project"]["random_seed"])
    rng = np.random.default_rng(seed + 7)
    early = int(config.get("optimization", {}).get("early_stopping_rounds", 50))
    X_train, y_train, _ = prepare_xy(df, feature_cols, masks["train"], label_col)
    X_val, y_val, _ = prepare_xy(df, feature_cols, masks["val"], label_col)
    mapping = None
    if binary:
        mapping = {0: 0, 2: 1}
        y_train = np.array([mapping[int(v)] for v in y_train])
        y_val = np.array([mapping[int(v)] for v in y_val])
    if len(y_train) < 500:
        n_trials = min(n_trials, 8)

    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights = {int(c): float(w) for c, w in zip(classes, weights)}

    best_score = -np.inf
    best_model = None
    best_params: dict[str, Any] = {}
    history = []
    for trial in range(n_trials):
        params = _sample_params(rng)
        model = CatBoostClassifier(
            loss_function="Logloss" if binary else "MultiClass",
            eval_metric="TotalF1:average=Macro" if not binary else "Logloss",
            random_seed=seed,
            class_weights=class_weights,
            bootstrap_type="Bernoulli",
            od_type="Iter",
            od_wait=early,
            verbose=False,
            allow_writing_files=False,
            **params,
        )
        t0 = time.time()
        model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True, verbose=False)
        elapsed = time.time() - t0
        pred = model.predict(X_val).astype(int).ravel()
        proba = model.predict_proba(X_val)
        metrics = evaluate_classification(
            y_val, pred, proba, labels=[0, 1] if binary else [0, 1, 2]
        )
        score = metrics["macro_f1"]
        history.append({"trial": trial, "params": params, "val_macro_f1": score, "seconds": elapsed})
        logger.info("CatBoost trial %s/%s macro-F1=%.4f", trial + 1, n_trials, score)
        if score > best_score:
            best_score, best_model, best_params = score, model, params

    if best_model is None:
        raise RuntimeError("CatBoost failed")

    results: dict[str, Any] = {
        "feature_cols": feature_cols,
        "best_params": best_params,
        "best_val_macro_f1": float(best_score),
        "search_history": history,
        "binary": binary,
    }
    for split, mask in masks.items():
        X, y, idx = prepare_xy(df, feature_cols, mask, label_col)
        if binary:
            y = np.array([mapping[int(v)] for v in y])
        pred = best_model.predict(X).astype(int).ravel()
        proba = best_model.predict_proba(X)
        results[f"metrics_{split}"] = evaluate_classification(
            y, pred, proba, labels=[0, 1] if binary else [0, 1, 2]
        )
        results[f"predictions_{split}"] = {
            "index": idx.to_numpy(),
            "y_true": y,
            "y_pred": pred,
            "y_proba": proba,
        }
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": best_model, "feature_cols": feature_cols, "best_params": best_params},
        models_dir / f"{model_name}.joblib",
    )
    best_model.save_model(str(models_dir / f"{model_name}.cbm"))
    return results

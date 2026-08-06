"""XGBoost trainer with CPU-friendly randomized search."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight

from src.evaluation.classification_metrics import evaluate_classification
from src.models.logistic_regression import prepare_xy

logger = logging.getLogger(__name__)


def _sample_params(rng: np.random.Generator) -> dict[str, Any]:
    return {
        "max_depth": int(rng.choice([2, 3, 4, 5, 6, 7])),
        "learning_rate": float(rng.choice([0.01, 0.02, 0.03, 0.05, 0.075, 0.10])),
        "n_estimators": int(rng.choice([200, 300, 500, 750, 1000, 1500])),
        "min_child_weight": float(rng.choice([1, 3, 5, 10, 20, 50])),
        "subsample": float(rng.choice([0.6, 0.7, 0.8, 0.9, 1.0])),
        "colsample_bytree": float(rng.choice([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])),
        "gamma": float(rng.choice([0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 5.0])),
        "reg_alpha": float(rng.choice([0.0, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0])),
        "reg_lambda": float(rng.choice([0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0])),
        "max_delta_step": int(rng.choice([0, 1, 5])),
    }


def train_xgboost(
    df: Any,
    feature_cols: list[str],
    masks: dict[str, np.ndarray],
    config: dict[str, Any],
    models_dir: Path,
    n_trials: int = 30,
    label_col: str = "label",
    binary: bool = False,
    model_name: str = "xgboost_model",
) -> dict[str, Any]:
    """Optimize XGBoost on validation Macro F1; never on development test."""
    seed = int(config["project"]["random_seed"])
    rng = np.random.default_rng(seed)
    early = int(config.get("optimization", {}).get("early_stopping_rounds", 50))
    X_train, y_train, _ = prepare_xy(df, feature_cols, masks["train"], label_col)
    X_val, y_val, _ = prepare_xy(df, feature_cols, masks["val"], label_col)
    mapping = None
    if binary:
        mapping = {0: 0, 2: 1}
        y_train = np.array([mapping[int(v)] for v in y_train])
        y_val = np.array([mapping[int(v)] for v in y_val])
    if len(y_train) < 500:
        n_trials = min(n_trials, 10)

    sw = compute_sample_weight("balanced", y_train)
    base: dict[str, Any] = {
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": seed,
        "verbosity": 0,
    }
    if binary:
        base.update({"objective": "binary:logistic", "eval_metric": "logloss"})
    else:
        base.update(
            {"objective": "multi:softprob", "num_class": 3, "eval_metric": "mlogloss"}
        )

    best_score = -np.inf
    best_model = None
    best_params: dict[str, Any] = {}
    history = []
    for trial in range(n_trials):
        params = _sample_params(rng)
        model = xgb.XGBClassifier(**base, **params, early_stopping_rounds=early)
        t0 = time.time()
        model.fit(
            X_train,
            y_train,
            sample_weight=sw,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False,
        )
        elapsed = time.time() - t0
        pred = model.predict(X_val)
        proba = model.predict_proba(X_val)
        metrics = evaluate_classification(
            y_val, pred, proba, labels=[0, 1] if binary else [0, 1, 2]
        )
        score = metrics["macro_f1"]
        history.append(
            {"trial": trial, "params": params, "val_macro_f1": score, "seconds": elapsed}
        )
        logger.info("XGB trial %s/%s macro-F1=%.4f", trial + 1, n_trials, score)
        if score > best_score:
            best_score, best_model, best_params = score, model, params

    if best_model is None:
        raise RuntimeError("XGBoost failed")

    results: dict[str, Any] = {
        "feature_cols": feature_cols,
        "best_params": best_params,
        "best_val_macro_f1": float(best_score),
        "search_history": history,
        "evals_result": best_model.evals_result(),
        "software": {"xgboost": xgb.__version__},
        "random_seed": seed,
        "binary": binary,
    }
    for split, mask in masks.items():
        X, y, idx = prepare_xy(df, feature_cols, mask, label_col)
        if len(y) == 0:
            results[f"metrics_{split}"] = {"error": "empty_split", "n": 0}
            results[f"predictions_{split}"] = {
                "index": np.array([], dtype=int),
                "y_true": np.array([]),
                "y_pred": np.array([]),
                "y_proba": np.zeros((0, 2 if binary else 3)),
            }
            continue
        if binary:
            y = np.array([mapping[int(v)] for v in y])
        pred = best_model.predict(X)
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
        {
            "model": best_model,
            "feature_cols": feature_cols,
            "best_params": best_params,
            "binary": binary,
        },
        models_dir / f"{model_name}.joblib",
    )
    best_model.save_model(str(models_dir / f"{model_name}.json"))
    return results

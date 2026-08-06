"""CatBoost challenger model (CPU-friendly)."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from catboost import CatBoostClassifier
from sklearn.utils.class_weight import compute_class_weight

from src.evaluation import evaluate_classification
from src.train_logistic import prepare_xy

logger = logging.getLogger(__name__)


def _sample_params(rng: np.random.Generator) -> dict[str, Any]:
    return {
        "depth": int(rng.integers(4, 7)),
        "learning_rate": float(rng.uniform(0.03, 0.10)),
        "iterations": int(rng.choice([300, 500, 800, 1000])),
        "l2_leaf_reg": float(rng.uniform(1.0, 10.0)),
        "subsample": float(rng.uniform(0.7, 1.0)),
        "colsample_bylevel": float(rng.uniform(0.6, 1.0)),
        "min_data_in_leaf": int(rng.choice([10, 20, 50, 100])),
    }


def train_catboost(
    df: Any,
    feature_cols: list[str],
    split_masks: dict[str, np.ndarray],
    config: dict[str, Any],
    models_dir: Path,
) -> dict[str, Any]:
    """Train CatBoost with fewer trials than XGBoost."""
    cfg = config["models"]["catboost"]
    seed = int(config["project"]["random_seed"])
    rng = np.random.default_rng(seed + 7)
    n_trials = int(cfg.get("optimization_trials", 15))
    early_stopping = int(cfg.get("early_stopping_rounds", 50))

    X_train, y_train, _ = prepare_xy(df, feature_cols, split_masks["train"])
    X_val, y_val, _ = prepare_xy(df, feature_cols, split_masks["val"])

    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights = {int(c): float(w) for c, w in zip(classes, weights)}

    best_score = -np.inf
    best_model = None
    best_params: dict[str, Any] = {}
    history: list[dict[str, Any]] = []

    for trial in range(n_trials):
        params = _sample_params(rng)
        model = CatBoostClassifier(
            loss_function="MultiClass",
            eval_metric="TotalF1:average=Macro",
            random_seed=seed,
            thread_count=int(cfg.get("thread_count", -1)),
            class_weights=class_weights,
            bootstrap_type="Bernoulli",
            od_type="Iter",
            od_wait=early_stopping,
            verbose=False,
            allow_writing_files=False,
            **params,
        )
        t0 = time.time()
        model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            use_best_model=True,
            verbose=False,
        )
        elapsed = time.time() - t0
        pred = model.predict(X_val).astype(int).ravel()
        proba = model.predict_proba(X_val)
        metrics = evaluate_classification(y_val, pred, proba)
        score = metrics["macro_f1"]
        history.append({"trial": trial, "params": params, "val_macro_f1": score, "seconds": elapsed})
        logger.info("CatBoost trial %s/%s macro-F1=%.4f", trial + 1, n_trials, score)
        if score > best_score:
            best_score = score
            best_model = model
            best_params = params

    if best_model is None:
        raise RuntimeError("CatBoost optimization failed")

    results: dict[str, Any] = {
        "feature_cols": feature_cols,
        "best_params": best_params,
        "best_val_macro_f1": float(best_score),
        "search_history": history,
        "random_seed": seed,
    }
    for split_name, mask in split_masks.items():
        X, y, idx = prepare_xy(df, feature_cols, mask)
        if len(y) == 0:
            continue
        pred = best_model.predict(X).astype(int).ravel()
        proba = best_model.predict_proba(X)
        results[f"metrics_{split_name}"] = evaluate_classification(y, pred, proba)
        results[f"predictions_{split_name}"] = {
            "index": idx.to_numpy(),
            "y_true": y,
            "y_pred": pred,
            "y_proba": proba,
        }

    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": best_model, "feature_cols": feature_cols, "best_params": best_params},
        models_dir / "catboost_model.joblib",
    )
    best_model.save_model(str(models_dir / "catboost_model.cbm"))
    logger.info("Best CatBoost val macro-F1=%.4f", best_score)
    return results

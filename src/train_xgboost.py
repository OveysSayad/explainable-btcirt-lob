"""XGBoost training with CPU-friendly hyperparameter search."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight

from src.evaluation import evaluate_classification
from src.train_logistic import prepare_xy

logger = logging.getLogger(__name__)


def _sample_params(rng: np.random.Generator) -> dict[str, Any]:
    """Sample a CPU-friendly hyperparameter set."""
    return {
        "max_depth": int(rng.integers(3, 7)),
        "learning_rate": float(rng.uniform(0.03, 0.10)),
        "n_estimators": int(rng.choice([300, 500, 800, 1000, 1200, 1500])),
        "min_child_weight": float(rng.choice([5, 10, 20, 30, 40, 50])),
        "subsample": float(rng.uniform(0.70, 1.00)),
        "colsample_bytree": float(rng.uniform(0.60, 1.00)),
        "reg_alpha": float(rng.uniform(0.0, 5.0)),
        "reg_lambda": float(rng.uniform(1.0, 20.0)),
        "gamma": float(rng.uniform(0.0, 2.0)),
    }


def train_xgboost(
    df: pd.DataFrame,
    feature_cols: list[str],
    split_masks: dict[str, np.ndarray],
    config: dict[str, Any],
    models_dir: Path,
) -> dict[str, Any]:
    """
    Train XGBoost with randomized search on validation macro-F1 / mlogloss.

    Hyperparameters are selected using validation only; the final test set
    is not used for model selection.
    """
    cfg = config["models"]["xgboost"]
    seed = int(config["project"]["random_seed"])
    rng = np.random.default_rng(seed)
    n_trials = int(cfg.get("optimization_trials", 30))
    early_stopping = int(cfg.get("early_stopping_rounds", 50))

    X_train, y_train, _ = prepare_xy(df, feature_cols, split_masks["train"])
    X_val, y_val, _ = prepare_xy(df, feature_cols, split_masks["val"])
    sw = compute_sample_weight("balanced", y_train)

    base = {
        "objective": cfg.get("objective", "multi:softprob"),
        "num_class": int(cfg.get("num_class", 3)),
        "eval_metric": cfg.get("eval_metric", "mlogloss"),
        "tree_method": cfg.get("tree_method", "hist"),
        "n_jobs": int(cfg.get("n_jobs", -1)),
        "random_state": seed,
        "verbosity": 0,
    }

    best_score = -np.inf
    best_model = None
    best_params: dict[str, Any] = {}
    history: list[dict[str, Any]] = []

    for trial in range(n_trials):
        params = _sample_params(rng)
        model = xgb.XGBClassifier(**base, **params, early_stopping_rounds=early_stopping)
        t0 = time.time()
        model.fit(
            X_train,
            y_train,
            sample_weight=sw,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False,
        )
        elapsed = time.time() - t0
        val_pred = model.predict(X_val)
        val_proba = model.predict_proba(X_val)
        val_metrics = evaluate_classification(y_val, val_pred, val_proba)
        score = val_metrics["macro_f1"]
        history.append(
            {
                "trial": trial,
                "params": params,
                "val_macro_f1": score,
                "val_log_loss": val_metrics.get("log_loss"),
                "best_iteration": int(getattr(model, "best_iteration", -1) or -1),
                "seconds": elapsed,
            }
        )
        logger.info(
            "XGB trial %s/%s macro-F1=%.4f params=%s",
            trial + 1,
            n_trials,
            score,
            params,
        )
        if score > best_score:
            best_score = score
            best_model = model
            best_params = params

    if best_model is None:
        raise RuntimeError("XGBoost optimization failed to produce a model")

    # Refit is already done on best trial with early stopping; keep that model.
    results: dict[str, Any] = {
        "feature_cols": feature_cols,
        "best_params": best_params,
        "best_val_macro_f1": float(best_score),
        "search_history": history,
        "software": {
            "xgboost": xgb.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "random_seed": seed,
    }

    for split_name, mask in split_masks.items():
        X, y, idx = prepare_xy(df, feature_cols, mask)
        if len(y) == 0:
            continue
        pred = best_model.predict(X)
        proba = best_model.predict_proba(X)
        results[f"metrics_{split_name}"] = evaluate_classification(y, pred, proba)
        results[f"predictions_{split_name}"] = {
            "index": idx.to_numpy(),
            "y_true": y,
            "y_pred": pred,
            "y_proba": proba,
        }

    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "xgboost_model.joblib"
    joblib.dump(
        {
            "model": best_model,
            "feature_cols": feature_cols,
            "best_params": best_params,
            "seed": seed,
        },
        model_path,
    )
    # Also native XGBoost format
    best_model.save_model(str(models_dir / "xgboost_model.json"))

    # Training history curves from best model
    evals = best_model.evals_result()
    results["evals_result"] = evals

    hyp_df = pd.DataFrame([{**best_params, "val_macro_f1": best_score}])
    tables = Path(config["_project_root"]) / "reports" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    hyp_df.to_csv(tables / "best_hyperparameters.csv", index=False)

    logger.info(
        "Best XGBoost val macro-F1=%.4f params=%s",
        best_score,
        best_params,
    )
    return results

"""Feature-set incremental value and grouped ablation."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight

from src.evaluation.classification_metrics import evaluate_classification
from src.feature_engineering import get_feature_set
from src.models.logistic_regression import prepare_xy

logger = logging.getLogger(__name__)


def _fit_eval(
    df: pd.DataFrame,
    cols: list[str],
    masks: dict[str, np.ndarray],
    seed: int,
    binary: bool = False,
) -> dict[str, Any]:
    if not cols:
        return {"error": "no features", "n_features": 0}
    Xtr, ytr, _ = prepare_xy(df, cols, masks["train"])
    Xva, yva, _ = prepare_xy(df, cols, masks["val"])
    Xte, yte, _ = prepare_xy(df, cols, masks["development_test"])
    mapping = None
    labels = [0, 1, 2]
    if binary:
        mapping = {0: 0, 2: 1}
        ytr = np.array([mapping[int(v)] for v in ytr])
        yva = np.array([mapping[int(v)] for v in yva])
        yte = np.array([mapping[int(v)] for v in yte])
        labels = [0, 1]
    sw = compute_sample_weight("balanced", ytr)
    params: dict[str, Any] = {
        "max_depth": 4,
        "learning_rate": 0.05,
        "n_estimators": 400,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "reg_lambda": 5.0,
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": seed,
        "verbosity": 0,
        "early_stopping_rounds": 40,
    }
    if binary:
        params.update({"objective": "binary:logistic", "eval_metric": "logloss"})
    else:
        params.update({"objective": "multi:softprob", "num_class": 3, "eval_metric": "mlogloss"})
    model = xgb.XGBClassifier(**params)
    t0 = time.time()
    model.fit(Xtr, ytr, sample_weight=sw, eval_set=[(Xva, yva)], verbose=False)
    elapsed = time.time() - t0
    pred = model.predict(Xte)
    proba = model.predict_proba(Xte)
    m = evaluate_classification(yte, pred, proba, labels=labels)
    mv = evaluate_classification(yva, model.predict(Xva), model.predict_proba(Xva), labels=labels)
    return {
        "n_features": len(cols),
        "train_seconds": elapsed,
        "val_macro_f1": mv["macro_f1"],
        "test_macro_f1": m["macro_f1"],
        "test_balanced_accuracy": m["balanced_accuracy"],
        "test_log_loss": m.get("log_loss"),
        "y_pred_test": pred,
        "y_true_test": yte,
    }


def run_feature_set_and_ablation(
    df: pd.DataFrame,
    masks: dict[str, np.ndarray],
    config: dict[str, Any],
    tables_dir: Path,
    binary: bool = False,
) -> pd.DataFrame:
    """Mandatory feature-set comparison and grouped ablations."""
    seed = int(config["project"]["random_seed"])
    experiments = [
        ("price_only", "price_only"),
        ("static_lob", "static_lob"),
        ("dynamic_lob", "dynamic_lob"),
        ("lob_full", "lob_full"),
        ("full_no_trade", "full_no_trade"),
        ("full_with_trade", "full_with_trade"),
        ("full_no_time", "full_no_time"),
    ]
    rows = []
    preds: dict[str, np.ndarray] = {}
    y_ref = None
    for name, key in experiments:
        cols = get_feature_set(
            df,
            key,
            include_time=(key != "full_no_time"),
            include_trade=(key == "full_with_trade"),
        )
        logger.info("Feature-set %s (%s cols)", name, len(cols))
        res = _fit_eval(df, cols, masks, seed, binary=binary)
        rows.append({"experiment": name, **{k: v for k, v in res.items() if not isinstance(v, np.ndarray)}})
        if "y_pred_test" in res:
            preds[name] = res["y_pred_test"]
            y_ref = res["y_true_test"]

    # Incremental values
    def _get(name: str, field: str = "test_macro_f1") -> float:
        for r in rows:
            if r["experiment"] == name:
                return float(r.get(field, np.nan))
        return float("nan")

    incr = {
        "IncrementalLOBValue": _get("full_no_trade") - _get("price_only"),
        "IncrementalTradeValue": _get("full_with_trade") - _get("full_no_trade"),
        "IncrementalTimeValue": _get("full_no_trade") - _get("full_no_time"),
    }
    out = pd.DataFrame(rows)
    tables_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(tables_dir / "feature_set_comparison.csv", index=False)
    out.to_csv(tables_dir / "ablation_results.csv", index=False)
    pd.DataFrame([incr]).to_csv(tables_dir / "incremental_value.csv", index=False)
    return out

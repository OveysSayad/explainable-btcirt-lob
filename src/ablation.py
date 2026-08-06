"""Feature-family ablation experiments."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight

from src.evaluation import evaluate_classification
from src.feature_engineering import FEATURE_FAMILIES, get_model_feature_columns
from src.train_logistic import prepare_xy

logger = logging.getLogger(__name__)


def _family_members(df: pd.DataFrame, families: list[str]) -> list[str]:
    cols = []
    for fam in families:
        for f in FEATURE_FAMILIES.get(fam, []):
            if f in df.columns:
                cols.append(f)
    return list(dict.fromkeys(cols))


def _train_eval_xgb(
    df: pd.DataFrame,
    feature_cols: list[str],
    split_masks: dict[str, np.ndarray],
    config: dict[str, Any],
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not feature_cols:
        return {"error": "no features", "n_features": 0}

    seed = int(config["project"]["random_seed"])
    X_train, y_train, _ = prepare_xy(df, feature_cols, split_masks["train"])
    X_val, y_val, _ = prepare_xy(df, feature_cols, split_masks["val"])
    X_test, y_test, _ = prepare_xy(df, feature_cols, split_masks["test"])
    sw = compute_sample_weight("balanced", y_train)

    default_params = {
        "max_depth": 4,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "min_child_weight": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
    }
    if params:
        default_params.update(params)

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=seed,
        verbosity=0,
        early_stopping_rounds=40,
        **default_params,
    )
    t0 = time.time()
    model.fit(
        X_train,
        y_train,
        sample_weight=sw,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    elapsed = time.time() - t0
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    metrics = evaluate_classification(y_test, pred, proba)
    val_pred = model.predict(X_val)
    val_proba = model.predict_proba(X_val)
    val_metrics = evaluate_classification(y_val, val_pred, val_proba)
    return {
        "n_features": len(feature_cols),
        "train_seconds": elapsed,
        "val_macro_f1": val_metrics["macro_f1"],
        "test_macro_f1": metrics["macro_f1"],
        "test_balanced_accuracy": metrics["balanced_accuracy"],
        "test_log_loss": metrics.get("log_loss"),
        "features": feature_cols,
    }


def run_ablation(
    df: pd.DataFrame,
    split_masks: dict[str, np.ndarray],
    config: dict[str, Any],
    best_params: dict[str, Any] | None = None,
    top_features: list[str] | None = None,
    tables_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Train XGBoost under feature-family ablation settings.

    Uses fixed/best hyperparameters (no re-tuning on test) for fair comparison.
    """
    include_trade = bool(config["features"].get("include_trade_features", True))
    all_feats = get_model_feature_columns(df, include_trade=include_trade)

    experiments: list[tuple[str, list[str]]] = []
    experiments.append(("all_core_features", all_feats))

    drop_map = {
        "without_imbalance": ["Order-book imbalance"],
        "without_liquidity": ["Liquidity"],
        "without_depth": ["Depth"],
        "without_temporal": ["Price dynamics", "Volatility", "Time"],
        "without_ofi_proxy": ["Order flow"],
        "without_trade": ["Trade activity"],
    }
    for name, families in drop_map.items():
        drop = set(_family_members(df, families))
        experiments.append((name, [f for f in all_feats if f not in drop]))

    snapshot_families = [
        "Liquidity",
        "Depth",
        "Order-book imbalance",
        "Price dynamics",
    ]
    experiments.append(("only_snapshot_features", _family_members(df, snapshot_families)))
    experiments.append(
        (
            "only_temporal_features",
            _family_members(df, ["Price dynamics", "Volatility", "Time", "Order flow"]),
        )
    )

    if top_features:
        top_k = [f for f in top_features if f in all_feats][
            : int(config["ablation"]["top_k_features"])
        ]
    else:
        # Fallback: OBI-centric strong features
        top_k = [
            f
            for f in [
                "obi_5",
                "weighted_obi",
                "microprice_edge_bps",
                "relative_spread_bps",
                "normalized_ofi_300s",
                "volatility_300s",
                "return_60s",
                "log_bid_depth_5",
                "log_ask_depth_5",
                "obi5_mean_300s",
            ]
            if f in all_feats
        ]
    experiments.append(("only_strongest_ten_features", top_k))

    rows = []
    for name, cols in experiments:
        logger.info("Ablation experiment: %s (%s features)", name, len(cols))
        result = _train_eval_xgb(df, cols, split_masks, config, params=best_params)
        rows.append(
            {
                "experiment": name,
                "n_features": result.get("n_features", 0),
                "train_seconds": result.get("train_seconds", np.nan),
                "val_macro_f1": result.get("val_macro_f1", np.nan),
                "test_macro_f1": result.get("test_macro_f1", np.nan),
                "test_balanced_accuracy": result.get("test_balanced_accuracy", np.nan),
                "test_log_loss": result.get("test_log_loss", np.nan),
                "error": result.get("error"),
            }
        )

    out = pd.DataFrame(rows)
    if tables_dir is not None:
        tables_dir.mkdir(parents=True, exist_ok=True)
        out.to_csv(tables_dir / "ablation_results.csv", index=False)
    return out

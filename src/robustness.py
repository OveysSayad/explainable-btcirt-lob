"""Horizon robustness and optional trading sanity check."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight

from src.evaluation import evaluate_classification
from src.feature_engineering import get_model_feature_columns
from src.label_engineering import construct_labels_for_horizon, label_distribution
from src.train_logistic import prepare_xy

logger = logging.getLogger(__name__)


def run_horizon_robustness(
    df: pd.DataFrame,
    split_masks: dict[str, np.ndarray],
    config: dict[str, Any],
    best_params: dict[str, Any] | None = None,
    tables_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Retrain/evaluate primary model for multiple forecast horizons.

    Epsilon is fit independently from each horizon's training returns.
    """
    horizons = [int(h) for h in config["robustness"]["horizons_seconds"]]
    include_trade = bool(config["features"].get("include_trade_features", True))
    feature_cols = get_model_feature_columns(df, include_trade=include_trade)
    seed = int(config["project"]["random_seed"])
    rows = []

    params = {
        "max_depth": 4,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "min_child_weight": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
    }
    if best_params:
        params.update({k: v for k, v in best_params.items() if k in params or True})

    for h in horizons:
        labeled, eps = construct_labels_for_horizon(
            df,
            horizon_seconds=h,
            config=config,
            epsilon=None,
            train_mask=pd.Series(split_masks["train"], index=df.index),
        )
        label_col = f"label_{h}s"
        dist = label_distribution(labeled[label_col])
        # Temporary rename for prepare_xy
        tmp = labeled.copy()
        tmp["label"] = tmp[label_col]

        X_train, y_train, _ = prepare_xy(tmp, feature_cols, split_masks["train"])
        X_val, y_val, _ = prepare_xy(tmp, feature_cols, split_masks["val"])
        X_test, y_test, _ = prepare_xy(tmp, feature_cols, split_masks["test"])
        if len(y_train) < 50 or len(y_test) < 20:
            rows.append(
                {
                    "horizon_seconds": h,
                    "epsilon": eps,
                    "error": "insufficient labeled rows",
                    "n_train": len(y_train),
                    "n_test": len(y_test),
                }
            )
            continue

        sw = compute_sample_weight("balanced", y_train)
        model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
            verbosity=0,
            early_stopping_rounds=40,
            **{k: v for k, v in params.items() if k != "n_estimators"},
            n_estimators=int(params.get("n_estimators", 500)),
        )
        model.fit(
            X_train,
            y_train,
            sample_weight=sw,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)
        metrics = evaluate_classification(y_test, pred, proba)
        rows.append(
            {
                "horizon_seconds": h,
                "epsilon": eps,
                "n_train": len(y_train),
                "n_val": len(y_val),
                "n_test": len(y_test),
                "stable_pct_train": float(
                    dist.loc[dist["class_name"] == "STABLE", "percentage"].iloc[0]
                )
                if len(dist)
                else np.nan,
                "test_macro_f1": metrics["macro_f1"],
                "test_balanced_accuracy": metrics["balanced_accuracy"],
                "test_log_loss": metrics.get("log_loss"),
                "test_accuracy": metrics["accuracy"],
                "f1_DOWN": metrics.get("f1_DOWN"),
                "f1_STABLE": metrics.get("f1_STABLE"),
                "f1_UP": metrics.get("f1_UP"),
            }
        )
        logger.info("Horizon %ss test macro-F1=%.4f epsilon=%.4f", h, metrics["macro_f1"], eps)

    out = pd.DataFrame(rows)
    if tables_dir is not None:
        tables_dir.mkdir(parents=True, exist_ok=True)
        out.to_csv(tables_dir / "horizon_comparison.csv", index=False)
    return out


def trading_sanity_check(
    df: pd.DataFrame,
    test_idx: np.ndarray,
    y_pred: np.ndarray,
    pred_index: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Exploratory long-only diagnostic (NOT a claim of trading profitability).

    UP -> long next horizon return; otherwise flat.
    Costs: fee + spread + slippage in bps.
    """
    cfg = config.get("trading_sanity", {})
    if not cfg.get("enabled", True):
        return {"enabled": False}

    fee = float(cfg.get("fee_bps", 10.0))
    spread_cost = float(cfg.get("spread_cost_bps", 5.0))
    slip = float(cfg.get("slippage_bps", 5.0))
    cost = fee + spread_cost + slip

    # Align predictions to dataframe rows
    rets = df.loc[pred_index, "future_return_bps"].to_numpy()
    signal = (y_pred == 2).astype(float)
    gross = signal * rets
    net = np.where(signal > 0, gross - cost, 0.0)
    # Only evaluate on finite returns
    valid = np.isfinite(net) & np.isfinite(rets)
    if valid.sum() == 0:
        return {"enabled": True, "error": "no valid returns"}

    return {
        "enabled": True,
        "label": "exploratory_long_only_diagnostic",
        "disclaimer": (
            "Prediction performance does not imply trading profitability. "
            "This long-only diagnostic includes configurable costs and is exploratory."
        ),
        "cost_bps_per_trade": cost,
        "n_signals": int(signal[valid].sum()),
        "mean_gross_bps": float(np.nanmean(gross[valid])),
        "mean_net_bps": float(np.nanmean(net[valid])),
        "sum_net_bps": float(np.nansum(net[valid])),
        "hit_rate_when_long": float(
            np.nanmean((rets[valid][signal[valid] == 1] > 0).astype(float))
        )
        if (signal[valid] == 1).any()
        else float("nan"),
    }

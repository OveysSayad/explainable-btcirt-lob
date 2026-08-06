"""Anchored nested walk-forward folds inside train+validation."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.splitting.purged_split import purge_by_target_timestamp

logger = logging.getLogger(__name__)


def nested_walk_forward_folds(
    df: pd.DataFrame,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    n_folds: int = 5,
    target_col: str = "target_timestamp",
) -> list[dict[str, Any]]:
    """
    Anchored expanding walk-forward folds on development dates only.

    Development test remains untouched. Target-timestamp purging is applied
    inside each fold.
    """
    combo = train_mask | val_mask
    subset = df.loc[combo].copy()
    subset["_orig"] = np.where(combo)[0]
    dates = sorted(
        subset["timestamp"].dt.tz_convert("UTC").dt.date.astype(str).unique().tolist()
    )
    if len(dates) < n_folds + 1:
        n_folds = max(1, len(dates) - 1)
        logger.warning("Reduced walk-forward folds to %s", n_folds)

    blocks = np.array_split(dates, n_folds + 1)
    folds: list[dict[str, Any]] = []
    for k in range(n_folds):
        tr_dates = [d for b in blocks[: k + 1] for d in b]
        va_dates = list(blocks[k + 1])
        ds = subset["timestamp"].dt.tz_convert("UTC").dt.date.astype(str)
        tr = subset.loc[ds.isin(tr_dates), "_orig"].to_numpy()
        va = subset.loc[ds.isin(va_dates), "_orig"].to_numpy()
        if len(tr) == 0 or len(va) == 0:
            continue
        n = len(df)
        tr_m = np.zeros(n, dtype=bool)
        va_m = np.zeros(n, dtype=bool)
        te_m = np.zeros(n, dtype=bool)
        tr_m[tr] = True
        va_m[va] = True
        if target_col in df.columns:
            tr_m, va_m, _ = purge_by_target_timestamp(
                df["timestamp"], df[target_col], tr_m, va_m, te_m
            )
        folds.append(
            {
                "fold": k + 1,
                "train_idx": np.where(tr_m)[0],
                "val_idx": np.where(va_m)[0],
                "train_dates": tr_dates,
                "val_dates": va_dates,
                "n_train": int(tr_m.sum()),
                "n_val": int(va_m.sum()),
            }
        )
    logger.info("Created %s nested walk-forward folds", len(folds))
    return folds

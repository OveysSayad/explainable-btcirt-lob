"""Date-based chronological train/validation/development-test split."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SplitResult:
    """Chronological split indices and metadata."""

    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    train_dates: list[str]
    val_dates: list[str]
    test_dates: list[str]
    metadata: dict[str, Any]


def _dates(ts: pd.Series) -> pd.Series:
    return ts.dt.tz_convert("UTC").dt.date.astype(str)


def chronological_date_split(
    df: pd.DataFrame,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
    test_fraction: float = 0.20,
) -> SplitResult:
    """
    Split by complete calendar dates.

    The latest block is labeled ``development_test`` (not a pristine holdout)
    because prior analysis already inspected this period.
    """
    if not np.isclose(train_fraction + validation_fraction + test_fraction, 1.0):
        raise ValueError("Split fractions must sum to 1")
    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("DataFrame must be sorted by timestamp")

    dates = sorted(_dates(df["timestamp"]).unique().tolist())
    n = len(dates)
    if n < 3:
        raise ValueError(f"Need >=3 dates; found {n}")
    n_train = max(1, int(np.floor(n * train_fraction)))
    n_val = max(1, int(np.floor(n * validation_fraction)))
    n_test = n - n_train - n_val
    if n_test < 1:
        n_train = max(1, n_train - 1)
        n_test = n - n_train - n_val

    train_dates = dates[:n_train]
    val_dates = dates[n_train : n_train + n_val]
    test_dates = dates[n_train + n_val :]
    ds = _dates(df["timestamp"])
    train_idx = np.where(ds.isin(train_dates))[0]
    val_idx = np.where(ds.isin(val_dates))[0]
    test_idx = np.where(ds.isin(test_dates))[0]
    meta = {
        "n_dates": n,
        "train_dates": train_dates,
        "val_dates": val_dates,
        "development_test_dates": test_dates,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_development_test": int(len(test_idx)),
        "note": (
            "Latest block is development_test, not a pristine final holdout. "
            "A future independent holdout is required for confirmation."
        ),
    }
    logger.info("Date split: %s", meta)
    return SplitResult(
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        train_dates=train_dates,
        val_dates=val_dates,
        test_dates=test_dates,
        metadata=meta,
    )


def masks_from_split(n_rows: int, split: SplitResult) -> dict[str, np.ndarray]:
    """Boolean masks for train/val/development_test."""
    train = np.zeros(n_rows, dtype=bool)
    val = np.zeros(n_rows, dtype=bool)
    test = np.zeros(n_rows, dtype=bool)
    train[split.train_idx] = True
    val[split.val_idx] = True
    test[split.test_idx] = True
    return {"train": train, "val": val, "development_test": test}

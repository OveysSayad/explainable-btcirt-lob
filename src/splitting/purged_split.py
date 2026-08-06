"""Target-timestamp purging across split boundaries."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def purge_by_target_timestamp(
    timestamps: pd.Series,
    target_timestamps: pd.Series,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Drop rows whose target timestamp crosses a split boundary.

    Rules
    -----
    - Training rows require target_timestamp < first validation timestamp
    - Validation rows require target_timestamp < first development-test timestamp
    - Also drop validation rows with current timestamp <= last train target boundary
    """
    ts = pd.to_datetime(timestamps, utc=True)
    tgt = pd.to_datetime(target_timestamps, utc=True)

    train = train_mask.copy()
    val = val_mask.copy()
    test = test_mask.copy()

    if train.any() and val.any():
        val_start = ts[val].min()
        train &= tgt.notna().to_numpy() & (tgt < val_start).to_numpy()
        # validation must start after train period
        train_end = ts[train].max() if train.any() else ts.min()
        val &= (ts > train_end).to_numpy()
        val &= tgt.notna().to_numpy()

    if val.any() and test.any():
        test_start = ts[test].min()
        val &= (tgt < test_start).to_numpy()
        val_end = ts[val].max() if val.any() else ts.min()
        test &= (ts > val_end).to_numpy()
        test &= tgt.notna().to_numpy()

    logger.info(
        "Target purge retained train=%s val=%s development_test=%s",
        int(train.sum()),
        int(val.sum()),
        int(test.sum()),
    )
    return train, val, test


def assert_targets_respect_boundaries(
    timestamps: pd.Series,
    target_timestamps: pd.Series,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
) -> None:
    """Raise if any retained target crosses a later split start."""
    ts = pd.to_datetime(timestamps, utc=True)
    tgt = pd.to_datetime(target_timestamps, utc=True)
    if train_mask.any() and val_mask.any():
        val_start = ts[val_mask].min()
        bad = train_mask & tgt.notna().to_numpy() & (tgt >= val_start).to_numpy()
        if bad.any():
            raise AssertionError(f"{bad.sum()} train targets cross validation start")
    if val_mask.any() and test_mask.any():
        test_start = ts[test_mask].min()
        bad = val_mask & tgt.notna().to_numpy() & (tgt >= test_start).to_numpy()
        if bad.any():
            raise AssertionError(
                f"{bad.sum()} validation targets cross development_test start"
            )

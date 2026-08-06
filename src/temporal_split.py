"""Compatibility shim for chronological splits (legacy purge_seconds API)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.splitting.purged_split import purge_by_target_timestamp
from src.splitting.temporal_split import SplitResult, masks_from_split
from src.splitting.temporal_split import chronological_date_split as _date_split


def chronological_date_split(
    df: pd.DataFrame,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
    purge_seconds: float | None = None,
) -> SplitResult:
    """
    Date split with optional legacy seconds-based purge near boundaries.

    Prefer target-timestamp purging in the redesign pipeline.
    """
    split = _date_split(df, train_fraction, validation_fraction, test_fraction)
    if purge_seconds is None or float(purge_seconds) <= 0:
        return split
    return SplitResult(
        *apply_purge(
            df["timestamp"],
            split.train_idx,
            split.val_idx,
            split.test_idx,
            purge_seconds=float(purge_seconds),
        ),
        train_dates=split.train_dates,
        val_dates=split.val_dates,
        test_dates=split.test_dates,
        metadata={**split.metadata, "legacy_purge_seconds": float(purge_seconds)},
    )


def apply_purge(timestamps, train_idx, val_idx, test_idx, purge_seconds=60.0):
    """Legacy seconds-based purge retained for older tests."""
    ts = pd.to_datetime(timestamps)
    purge = pd.Timedelta(seconds=purge_seconds)
    train_idx = np.asarray(train_idx)
    val_idx = np.asarray(val_idx)
    test_idx = np.asarray(test_idx)
    if len(train_idx) and len(val_idx):
        val_start = ts.iloc[val_idx].min()
        train_idx = train_idx[ts.iloc[train_idx] <= (val_start - purge)]
    if len(val_idx) and len(test_idx):
        test_start = ts.iloc[test_idx].min()
        val_idx = val_idx[ts.iloc[val_idx] <= (test_start - purge)]
    return train_idx, val_idx, test_idx


def assert_no_overlap(split: SplitResult) -> None:
    sets = [
        set(split.train_idx.tolist()),
        set(split.val_idx.tolist()),
        set(split.test_idx.tolist()),
    ]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise AssertionError("overlap")


def non_overlapping_indices(timestamps, indices, step_seconds=30.0):
    if len(indices) == 0:
        return indices
    ts = pd.to_datetime(timestamps.iloc[indices]).sort_values()
    ordered = ts.index.to_numpy()
    selected = [ordered[0]]
    last = ts.iloc[0]
    step = pd.Timedelta(seconds=step_seconds)
    for idx, t in zip(ordered[1:], ts.iloc[1:]):
        if t >= last + step:
            selected.append(idx)
            last = t
    return np.array(selected, dtype=int)


__all__ = [
    "SplitResult",
    "chronological_date_split",
    "masks_from_split",
    "purge_by_target_timestamp",
    "apply_purge",
    "assert_no_overlap",
    "non_overlapping_indices",
]

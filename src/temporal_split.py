"""Chronological and walk-forward temporal splitting utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SplitResult:
    """Container for a chronological train/validation/test split."""

    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    train_dates: list[str]
    val_dates: list[str]
    test_dates: list[str]
    purge_seconds: float
    metadata: dict[str, Any]


def _dates_from_timestamps(ts: pd.Series) -> pd.Series:
    return ts.dt.tz_convert("UTC").dt.date.astype(str)


def chronological_date_split(
    df: pd.DataFrame,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
    test_fraction: float = 0.20,
    purge_seconds: float = 60.0,
) -> SplitResult:
    """
    Split by unique calendar dates (chronological), then purge boundary rows.

    Important: ``df`` must already be sorted by timestamp with a contiguous
    RangeIndex (0..n-1). Callers should do:

        df = df.sort_values("timestamp").reset_index(drop=True)

    Purge removes observations near split boundaries whose labels could overlap.
    """
    if not np.isclose(train_fraction + validation_fraction + test_fraction, 1.0):
        raise ValueError("Split fractions must sum to 1.0")

    out = df
    if not out["timestamp"].is_monotonic_increasing:
        raise ValueError(
            "DataFrame must be sorted by timestamp before chronological_date_split"
        )
    dates = sorted(_dates_from_timestamps(out["timestamp"]).unique().tolist())
    n_dates = len(dates)
    if n_dates < 3:
        raise ValueError(f"Need at least 3 unique dates for split; found {n_dates}")

    n_train = max(1, int(np.floor(n_dates * train_fraction)))
    n_val = max(1, int(np.floor(n_dates * validation_fraction)))
    # Remaining goes to test
    n_test = n_dates - n_train - n_val
    if n_test < 1:
        # Borrow from train
        n_train = max(1, n_train - 1)
        n_test = n_dates - n_train - n_val
    if n_test < 1 or n_val < 1:
        raise ValueError(f"Cannot form split with {n_dates} dates")

    train_dates = dates[:n_train]
    val_dates = dates[n_train : n_train + n_val]
    test_dates = dates[n_train + n_val :]

    date_series = _dates_from_timestamps(out["timestamp"])
    train_mask = date_series.isin(train_dates)
    val_mask = date_series.isin(val_dates)
    test_mask = date_series.isin(test_dates)

    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0]

    # Apply purge around boundaries
    train_idx, val_idx, test_idx = apply_purge(
        out["timestamp"].to_numpy(),
        train_idx,
        val_idx,
        test_idx,
        purge_seconds=purge_seconds,
    )

    meta = {
        "n_dates": n_dates,
        "n_train_dates": len(train_dates),
        "n_val_dates": len(val_dates),
        "n_test_dates": len(test_dates),
        "n_train_rows": int(len(train_idx)),
        "n_val_rows": int(len(val_idx)),
        "n_test_rows": int(len(test_idx)),
        "train_date_start": train_dates[0],
        "train_date_end": train_dates[-1],
        "val_date_start": val_dates[0],
        "val_date_end": val_dates[-1],
        "test_date_start": test_dates[0],
        "test_date_end": test_dates[-1],
    }
    logger.info("Chronological split: %s", meta)
    return SplitResult(
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        train_dates=train_dates,
        val_dates=val_dates,
        test_dates=test_dates,
        purge_seconds=purge_seconds,
        metadata=meta,
    )


def apply_purge(
    timestamps: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    purge_seconds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove observations near split boundaries within purge_seconds.

    - Drop train rows within purge_seconds before first val timestamp
    - Drop val rows within purge_seconds before first test timestamp
    - Drop val rows within purge_seconds after last train timestamp
    - Drop test rows within purge_seconds after last val timestamp
    """
    ts = pd.to_datetime(timestamps)
    purge = pd.Timedelta(seconds=purge_seconds)

    if len(train_idx) and len(val_idx):
        train_end = ts[train_idx].max()
        val_start = ts[val_idx].min()
        train_idx = train_idx[ts[train_idx] <= (val_start - purge)]
        val_idx = val_idx[ts[val_idx] >= (train_end + purge)]

    if len(val_idx) and len(test_idx):
        val_end = ts[val_idx].max()
        test_start = ts[test_idx].min()
        val_idx = val_idx[ts[val_idx] <= (test_start - purge)]
        test_idx = test_idx[ts[test_idx] >= (val_end + purge)]

    return train_idx, val_idx, test_idx


def assert_no_overlap(split: SplitResult) -> None:
    """Raise if train/val/test indices overlap."""
    sets = [
        set(split.train_idx.tolist()),
        set(split.val_idx.tolist()),
        set(split.test_idx.tolist()),
    ]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise AssertionError("Train/validation/test indices overlap")


def walk_forward_folds(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    n_folds: int = 3,
    purge_seconds: float = 60.0,
) -> list[dict[str, Any]]:
    """
    Anchored walk-forward folds inside the train+validation period.

    Fold k trains on the earliest expanding block and validates on the next block.
    """
    # Combine train+val chronologically for folding; final test remains untouched.
    combo_idx = np.sort(np.concatenate([train_idx, val_idx]))
    subset = df.iloc[combo_idx].copy()
    subset["_orig_idx"] = combo_idx
    dates = sorted(_dates_from_timestamps(subset["timestamp"]).unique().tolist())
    if len(dates) < n_folds + 1:
        n_folds = max(1, len(dates) - 1)
        logger.warning("Reducing walk-forward folds to %s given %s dates", n_folds, len(dates))

    # Split dates into n_folds+1 blocks; first block always in train (anchored)
    blocks = np.array_split(dates, n_folds + 1)
    folds: list[dict[str, Any]] = []
    for k in range(n_folds):
        train_dates = [d for block in blocks[: k + 1] for d in block]
        val_dates = list(blocks[k + 1])
        date_series = _dates_from_timestamps(subset["timestamp"])
        tr = subset.loc[date_series.isin(train_dates), "_orig_idx"].to_numpy()
        va = subset.loc[date_series.isin(val_dates), "_orig_idx"].to_numpy()
        if len(tr) == 0 or len(va) == 0:
            continue
        # Purge between fold train/val
        ts = df["timestamp"].to_numpy()
        tr, va, _ = apply_purge(ts, tr, va, np.array([], dtype=int), purge_seconds)
        folds.append(
            {
                "fold": k + 1,
                "train_idx": tr,
                "val_idx": va,
                "train_dates": train_dates,
                "val_dates": val_dates,
                "n_train": int(len(tr)),
                "n_val": int(len(va)),
            }
        )
    logger.info("Created %s walk-forward folds", len(folds))
    return folds


def non_overlapping_indices(
    timestamps: pd.Series,
    indices: np.ndarray,
    step_seconds: float = 30.0,
) -> np.ndarray:
    """
    Subsample indices so consecutive selected timestamps are >= step_seconds apart.

    Used to evaluate performance without heavily overlapping labels.
    """
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


def masks_from_split(n_rows: int, split: SplitResult) -> dict[str, np.ndarray]:
    """Boolean masks for train/val/test."""
    train = np.zeros(n_rows, dtype=bool)
    val = np.zeros(n_rows, dtype=bool)
    test = np.zeros(n_rows, dtype=bool)
    train[split.train_idx] = True
    val[split.val_idx] = True
    test[split.test_idx] = True
    return {"train": train, "val": val, "test": test}

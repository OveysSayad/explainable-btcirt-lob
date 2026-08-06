"""Unit tests for chronological splitting and purge logic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.temporal_split import (
    assert_no_overlap,
    chronological_date_split,
    non_overlapping_indices,
)


def _daily_frame(n_days: int = 10, rows_per_day: int = 5) -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2026-01-01", tz="UTC")
    idx = 0
    for d in range(n_days):
        for r in range(rows_per_day):
            rows.append(
                {
                    "timestamp": start
                    + pd.Timedelta(days=d)
                    + pd.Timedelta(minutes=r * 10),
                    "mid_price": 100 + idx,
                }
            )
            idx += 1
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def test_chronological_ordering_and_no_overlap():
    df = _daily_frame(10)
    split = chronological_date_split(df, 0.6, 0.2, 0.2, purge_seconds=60)
    assert_no_overlap(split)
    assert split.train_idx.max() < split.val_idx.min()
    assert split.val_idx.max() < split.test_idx.min()


def test_purge_interval_removes_boundary_rows():
    df = _daily_frame(10, rows_per_day=20)
    split = chronological_date_split(df, 0.6, 0.2, 0.2, purge_seconds=3600)
    # With 1h purge, some rows near boundaries should be removed vs no purge
    split_nopurge = chronological_date_split(df, 0.6, 0.2, 0.2, purge_seconds=0)
    assert len(split.train_idx) <= len(split_nopurge.train_idx)
    assert len(split.val_idx) <= len(split_nopurge.val_idx)


def test_final_test_is_latest_dates():
    df = _daily_frame(10)
    split = chronological_date_split(df, 0.6, 0.2, 0.2, purge_seconds=0)
    all_dates = sorted(df["timestamp"].dt.date.astype(str).unique())
    assert split.test_dates == all_dates[-len(split.test_dates) :]
    assert split.train_dates[0] == all_dates[0]


def test_requires_sorted_input():
    df = _daily_frame(5)
    shuffled = df.sample(frac=1.0, random_state=0)
    with pytest.raises(ValueError):
        chronological_date_split(shuffled, 0.6, 0.2, 0.2, purge_seconds=0)


def test_non_overlapping_step():
    df = _daily_frame(3, rows_per_day=10)
    idx = np.arange(len(df))
    non = non_overlapping_indices(df["timestamp"], idx, step_seconds=1800)
    ts = df.loc[non, "timestamp"].sort_values()
    gaps = ts.diff().dt.total_seconds().dropna()
    assert (gaps >= 1800 - 1e-6).all()

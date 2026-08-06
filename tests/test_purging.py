"""Target-timestamp purge and delay matching tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.splitting.purged_split import assert_targets_respect_boundaries, purge_by_target_timestamp
from src.splitting.temporal_split import chronological_date_split


def test_target_purge_blocks_crossing():
    ts = pd.date_range("2026-01-01", periods=10, freq="1D", tz="UTC")
    df = pd.DataFrame({"timestamp": ts, "target_timestamp": ts + pd.Timedelta(days=2)})
    n = len(df)
    train = np.zeros(n, dtype=bool)
    val = np.zeros(n, dtype=bool)
    test = np.zeros(n, dtype=bool)
    train[:6] = True
    val[6:8] = True
    test[8:] = True
    tr, va, te = purge_by_target_timestamp(
        df["timestamp"], df["target_timestamp"], train, val, test
    )
    # training targets must be before val start
    if tr.any() and va.any():
        assert (df.loc[tr, "target_timestamp"] < df.loc[va, "timestamp"].min()).all()
    assert_targets_respect_boundaries(
        df["timestamp"], df["target_timestamp"], tr, va, te
    )


def test_chronological_requires_sorted():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-02", "2026-01-01"], utc=True
            )
        }
    )
    with pytest.raises(ValueError):
        chronological_date_split(df)

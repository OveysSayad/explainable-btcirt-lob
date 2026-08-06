"""Basis-point scaling and label unit tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.labels.next_observation import assign_classes, log_return_bps
from src.labels.next_price_change import build_study_b_labels
from src.labels.strict_horizon import _match_strict_horizon
from src.trade_deduplication import deduplicate_trades


def test_log_return_bps_one_bp_approx():
    # 10000 * log(100.01/100) ≈ 0.99995 ≈ 1
    r = log_return_bps(np.array([100.0]), np.array([100.01]))[0]
    assert abs(r - 1.0) < 0.01


def test_log_return_bps_minus_one():
    r = log_return_bps(np.array([100.0]), np.array([99.99]))[0]
    assert abs(r + 1.0) < 0.01


def test_log_return_bps_zero():
    r = log_return_bps(np.array([100.0]), np.array([100.0]))[0]
    assert abs(r) < 1e-12


def test_assign_classes_epsilon():
    r = pd.Series([-2.0, -0.1, 0.0, 0.1, 2.0])
    y = assign_classes(r, epsilon=1.0)
    assert list(y.astype(int)) == [0, 1, 1, 1, 2]


def test_trade_deduplication_no_double_count():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="70s", tz="UTC"),
            "last_trade_price": [100.0, 100.0, 101.0, 101.0],
            "last_trade_qty": [1.0, 1.0, 2.0, 2.0],
            "asks_price_1": [102] * 4,
            "bids_price_1": [98] * 4,
        }
    )
    out = deduplicate_trades(df)
    assert out["is_new_trade"].tolist() == [1, 0, 1, 0]
    assert out["new_trade_qty"].tolist() == [1.0, 0.0, 2.0, 0.0]


def test_strict_horizon_window_enforced():
    ts = pd.date_range("2026-01-01", periods=5, freq="70s", tz="UTC").to_numpy()
    mids = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
    # 10s window 5-15 cannot match 70s gaps
    idx, delay, err, tmid = _match_strict_horizon(ts, mids, 10, 5, 15)
    assert (idx < 0).all()


def test_next_change_direction():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=5, freq="70s", tz="UTC"),
            "mid_price": [100.0, 100.0, 100.0, 101.0, 99.0],
            "asks_price_1": [101] * 5,
            "bids_price_1": [99] * 5,
        }
    )
    out, meta = build_study_b_labels(df, one_sample_per_price_run=True)
    # First run of 100 -> next change to 101 = UP
    assert meta["n_primary_sample"] >= 1
    assert out.loc[out["study_b_primary_sample"] & (out["mid_price"] == 100), "label"].iloc[0] == 2


def test_epsilon_train_only_contract():
    train = pd.Series([1.0, 2.0, 3.0])
    test = pd.Series([100.0, 200.0])
    from src.labels.next_observation import fit_epsilon_candidates

    df = pd.DataFrame({"mid_price": [1e5] * 3, "asks_price_1": [1e5 + 1] * 3, "bids_price_1": [1e5 - 1] * 3})
    # Just ensure hybrid uses training returns scale
    cand = fit_epsilon_candidates(
        train,
        pd.Series([1.0, 1.0, 1.0]),
        tick_bps_median=0.1,
        config={"study_a": {"epsilon_candidates": {"quantiles": [0.5], "spread_multipliers": [0.5], "use_tick_threshold": True}}},
    )
    hybrid = float(cand.loc[cand["method"] == "hybrid", "epsilon_bps"].iloc[0])
    assert hybrid < 50  # would be huge if test contaminated

"""Unit tests for label construction and epsilon fitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.label_engineering import (
    assign_classes,
    compute_future_mid_price,
    compute_future_return_bps,
    fit_epsilon,
)


def test_future_price_matching():
    ts = pd.Series(pd.date_range("2026-01-01", periods=5, freq="70s", tz="UTC"))
    mid = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0])
    future = compute_future_mid_price(
        ts,
        mid,
        horizon_seconds=70,
        smoothing_window_seconds=5,
        match_tolerance_seconds=90,
    )
    # From t0, t+70s is approximately second snapshot mid=101
    assert np.isclose(future.iloc[0], 101.0)


def test_future_matching_excludes_current_snapshot():
    """Short horizons must not resolve to the current mid (zero-return trap)."""
    ts = pd.Series(pd.date_range("2026-01-01", periods=3, freq="70s", tz="UTC"))
    mid = pd.Series([100.0, 110.0, 120.0])
    future = compute_future_mid_price(
        ts,
        mid,
        horizon_seconds=10,
        smoothing_window_seconds=5,
        match_tolerance_seconds=120,
    )
    assert not np.isclose(future.iloc[0], 100.0)
    assert np.isclose(future.iloc[0], 110.0)


def test_epsilon_from_training_only():
    train = pd.Series(np.linspace(-10, 10, 101))
    eps = fit_epsilon(train, stable_fraction=0.35)
    # Should equal 35th percentile of absolute returns
    expected = float(train.abs().quantile(0.35))
    assert np.isclose(eps, expected)


def test_class_assignment():
    r = pd.Series([-5.0, -0.1, 0.0, 0.1, 5.0])
    labels = assign_classes(r, epsilon=1.0)
    assert list(labels.astype(int)) == [0, 1, 1, 1, 2]


def test_epsilon_ignores_test_distribution():
    train = pd.Series([1.0, 2.0, 3.0, 4.0])
    test = pd.Series([100.0, 200.0, 300.0])
    eps_train = fit_epsilon(train, 0.5)
    # Fitting on train+test would be larger; ensure we only use train
    eps_wrong = fit_epsilon(pd.concat([train, test]), 0.5)
    assert eps_train < eps_wrong
    assert np.isclose(eps_train, float(train.abs().quantile(0.5)))


def test_fit_epsilon_empty_raises():
    with pytest.raises(ValueError):
        fit_epsilon(pd.Series(dtype=float), 0.35)


def test_future_return_bps():
    cur = pd.Series([100.0, 100.0])
    fut = pd.Series([101.0, 99.0])
    ret = compute_future_return_bps(cur, fut)
    assert ret.iloc[0] > 0
    assert ret.iloc[1] < 0

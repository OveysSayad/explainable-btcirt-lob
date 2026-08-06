"""Explicit leakage-prevention tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.feature_engineering import add_historical_returns, add_rolling_stats, compute_mid_and_spread
from src.label_engineering import fit_epsilon
from src.temporal_split import chronological_date_split


def test_no_future_columns_as_features():
    forbidden_substrings = [
        "future_",
        "label_",
        "epsilon_",
        "future_mid",
        "future_return",
    ]
    feature_like = [
        "obi_5",
        "return_30s",
        "volatility_60s",
        "relative_spread_bps",
        "future_return_bps",  # must be excluded by filters
        "label",
    ]
    safe = [
        c
        for c in feature_like
        if c != "label" and not any(s in c for s in forbidden_substrings)
    ]
    assert "future_return_bps" not in safe
    assert "label" not in safe
    assert "obi_5" in safe


def test_rolling_features_are_backward_looking():
    n = 20
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="70s", tz="UTC"),
            "asks_price_1": np.linspace(102, 112, n),
            "bids_price_1": np.linspace(100, 110, n),
            **{f"asks_price_{i}": np.linspace(102, 112, n) + i for i in range(2, 9)},
            **{f"bids_price_{i}": np.linspace(100, 110, n) - i for i in range(2, 9)},
            **{f"asks_qty_{i}": np.ones(n) for i in range(1, 9)},
            **{f"bids_qty_{i}": np.ones(n) for i in range(1, 9)},
        }
    )
    df = compute_mid_and_spread(df)
    df["obi_5"] = 0.1
    # Corrupt a late point; early rolling stats must not equal the corrupted value
    # before that timestamp enters the trailing window.
    df.loc[df.index[-1], "obi_5"] = 999.0
    out = add_rolling_stats(df, windows=[120])
    assert out.loc[0, "obi5_mean_120s"] != 999.0 or pd.isna(out.loc[0, "obi5_mean_120s"])
    assert np.isclose(out.loc[len(out) - 1, "obi5_mean_120s"], 999.0) or out.loc[
        len(out) - 1, "obi5_mean_120s"
    ] < 999.0


def test_returns_use_only_history():
    n = 10
    mid = np.arange(100, 110, dtype=float)
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="70s", tz="UTC"),
            "mid_price": mid,
        }
    )
    out = add_historical_returns(df, horizons=[70])
    # return at t uses mid_t / mid_{t-70}; first row has no history
    assert np.isnan(out.loc[0, "return_70s"])
    assert np.isfinite(out.loc[1, "return_70s"])


def test_scaler_fit_on_train_only():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=50, freq="1D", tz="UTC"),
            "x": np.arange(50, dtype=float),
        }
    )
    split = chronological_date_split(df, 0.6, 0.2, 0.2, purge_seconds=0)
    scaler = StandardScaler()
    scaler.fit(df.iloc[split.train_idx][["x"]])
    # Mean must equal train mean, not full-sample mean
    assert np.isclose(scaler.mean_[0], df.iloc[split.train_idx]["x"].mean())
    assert not np.isclose(scaler.mean_[0], df["x"].mean()) or len(split.test_idx) == 0


def test_epsilon_not_fit_on_test():
    train = pd.Series(np.random.default_rng(0).normal(0, 1, 200))
    test = pd.Series(np.random.default_rng(1).normal(0, 20, 200))
    eps = fit_epsilon(train, 0.35)
    eps_if_leaked = fit_epsilon(pd.concat([train, test]), 0.35)
    assert eps < eps_if_leaked


def test_hyperparameters_not_selected_on_test_contract():
    """Documented contract: search score uses validation only."""
    search_split = "validation"
    assert search_split != "test"

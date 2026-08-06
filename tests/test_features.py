"""Unit tests for microstructure feature formulas."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.feature_engineering import (
    add_depth_features,
    add_microprice_features,
    add_obi_features,
    add_ofi_proxy_features,
    compute_mid_and_spread,
)


def _two_snapshot_frame() -> pd.DataFrame:
    rows = []
    for t, bid, ask, bq, aq in [
        (0, 100.0, 102.0, 5.0, 3.0),
        (70, 101.0, 102.0, 8.0, 2.0),
    ]:
        row = {
            "timestamp": pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(seconds=t),
        }
        for i in range(1, 9):
            row[f"asks_price_{i}"] = ask + (i - 1)
            row[f"asks_qty_{i}"] = aq
            row[f"bids_price_{i}"] = bid - (i - 1)
            row[f"bids_qty_{i}"] = bq
        rows.append(row)
    return pd.DataFrame(rows)


def test_mid_price_and_relative_spread():
    df = compute_mid_and_spread(_two_snapshot_frame())
    assert np.isclose(df.loc[0, "mid_price"], 101.0)
    assert np.isclose(df.loc[0, "spread"], 2.0)
    assert np.isclose(df.loc[0, "relative_spread_bps"], 10_000.0 * 2.0 / 101.0)


def test_obi_and_depth():
    df = compute_mid_and_spread(_two_snapshot_frame())
    df = add_depth_features(df, levels=[1, 5], eps=1e-12)
    df = add_obi_features(df, levels=[1, 5], eps=1e-12, lam=0.5)
    # level-1: bid 5, ask 3 -> (5-3)/(5+3)=0.25
    assert np.isclose(df.loc[0, "obi_1"], 0.25)
    assert np.isclose(df.loc[0, "ask_depth_1"], 3.0)
    assert np.isclose(df.loc[0, "bid_depth_1"], 5.0)


def test_weighted_obi():
    df = compute_mid_and_spread(_two_snapshot_frame())
    df = add_depth_features(df, levels=[1, 2, 3, 5, 8], eps=1e-12)
    df = add_obi_features(df, levels=[1, 2, 3, 5, 8], eps=1e-12, lam=0.5)
    # Equal qty structure with bid>ask at every level => positive weighted OBI
    assert df.loc[0, "weighted_obi"] > 0


def test_microprice():
    df = compute_mid_and_spread(_two_snapshot_frame())
    df = add_microprice_features(df, eps=1e-12)
    # (102*5 + 100*3)/(5+3) = 810/8 = 101.25
    assert np.isclose(df.loc[0, "microprice"], 101.25)
    assert np.isclose(df.loc[0, "microprice_edge_bps"], 10_000.0 * (101.25 - 101.0) / 101.0)


def test_concentration():
    df = compute_mid_and_spread(_two_snapshot_frame())
    df = add_depth_features(df, levels=[1, 2, 3, 5, 8], eps=1e-12)
    # top-2 / total with equal qty per level: 2/8 = 0.25
    assert np.isclose(df.loc[0, "ask_concentration"], 0.25)
    assert np.isclose(df.loc[0, "bid_concentration"], 0.25)


def test_ofi_proxy_sign():
    df = compute_mid_and_spread(_two_snapshot_frame())
    df = add_ofi_proxy_features(df, roll_seconds=[30], eps=1e-12)
    # Bid rose 100->101 with qty 8; ask unchanged 102 with qty decreased 3->2
    # bid_component = I(bid>=prev)*bq - I(bid<=prev)*bq_prev = 8 - 0 = 8 (bid increased)
    # Actually bid > bid_prev so I(bid>=prev)=1, I(bid<=prev)=0 => bid_component=8
    # ask == ask_prev: ask_component = aq - aq_prev = 2 - 3 = -1
    # ofi = 8 - (-1) = 9
    assert np.isclose(df.loc[1, "ofi_proxy"], 9.0)

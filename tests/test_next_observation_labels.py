"""Additional redesign tests: labels, delays, leakage contracts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.labels.next_observation import build_study_a_labels
from src.labels.next_price_change import build_study_b_labels
from src.labels.strict_horizon import build_study_c_labels, cross_horizon_overlap
from src.trade_deduplication import add_corrected_trade_features


def _lob_frame(n: int = 20, gap_s: float = 10.0) -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2026-01-01", tz="UTC")
    mid = 100.0
    for i in range(n):
        if i > 0 and i % 4 == 0:
            mid += 1.0 if (i // 4) % 2 == 0 else -1.0
        bid = mid - 0.5
        ask = mid + 0.5
        row = {
            "timestamp": start + pd.Timedelta(seconds=i * gap_s),
            "mid_price": mid,
            "last_trade_price": mid,
            "last_trade_qty": 1.0,
            "asks_price_1": ask,
            "bids_price_1": bid,
        }
        for lvl in range(1, 9):
            row[f"asks_price_{lvl}"] = ask + (lvl - 1)
            row[f"bids_price_{lvl}"] = bid - (lvl - 1)
            row[f"asks_qty_{lvl}"] = 1.0
            row[f"bids_qty_{lvl}"] = 1.0 + (0.5 if lvl == 1 else 0)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.loc[1, "last_trade_price"] = df.loc[0, "last_trade_price"]
    df.loc[1, "last_trade_qty"] = df.loc[0, "last_trade_qty"]
    return df


def test_next_observation_label_fields():
    df = _lob_frame(15, gap_s=70)
    train = np.zeros(len(df), dtype=bool)
    train[:10] = True
    val = np.zeros(len(df), dtype=bool)
    val[10:13] = True
    cfg = {
        "study_a": {
            "primary_epsilon_method": "hybrid",
            "epsilon_candidates": {
                "quantiles": [0.35],
                "spread_multipliers": [0.5],
                "use_tick_threshold": True,
            },
        }
    }
    out, meta = build_study_a_labels(df, cfg, train, val)
    assert "actual_delay_seconds" in out.columns
    assert "next_observation_return_bps" in out.columns
    assert "target_timestamp" in out.columns
    assert meta["epsilon_bps"] > 0
    assert pd.isna(out.loc[len(out) - 1, "label"])


def test_one_sample_per_price_run():
    df = _lob_frame(20, gap_s=70)
    out, meta = build_study_b_labels(df, one_sample_per_price_run=True)
    assert meta["n_primary_sample"] <= meta["n_all_eligible"]
    assert out["study_b_primary_sample"].sum() == meta["n_primary_sample"]


def test_strict_horizon_and_overlap():
    df = _lob_frame(40, gap_s=10)
    train = np.zeros(len(df), dtype=bool)
    train[:25] = True
    val = np.zeros(len(df), dtype=bool)
    val[25:32] = True
    config = {
        "study_a": {
            "primary_epsilon_method": "hybrid",
            "epsilon_candidates": {
                "quantiles": [0.35],
                "spread_multipliers": [0.5],
                "use_tick_threshold": True,
            },
        },
        "study_c": {
            "horizons": {
                10: {"lower_seconds": 5, "upper_seconds": 15},
                30: {"lower_seconds": 20, "upper_seconds": 40},
                60: {"lower_seconds": 40, "upper_seconds": 80},
            },
            "minimum_total_samples": 5,
            "minimum_samples_per_class": 1,
            "minimum_unique_dates": 1,
        },
    }
    labeled, meta = build_study_c_labels(df, config, train, val)
    assert 10 in labeled and 30 in labeled
    elig10 = (labeled[10]["target_index"] >= 0).sum()
    assert elig10 > 0
    d = labeled[10].loc[labeled[10]["target_index"] >= 0, "actual_delay_seconds"]
    assert (d >= 5).all() and (d <= 15).all()
    overlap = cross_horizon_overlap(labeled)
    assert not overlap.empty


def test_corrected_trades_no_double_count():
    df = _lob_frame(6, gap_s=70)
    out = add_corrected_trade_features(df, windows=[300])
    assert out.loc[1, "is_new_trade"] == 0
    assert "new_trade_count_300s" in out.columns

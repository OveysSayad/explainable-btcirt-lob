"""Compatibility shim for label helpers used by older tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.labels.next_observation import assign_classes, fit_epsilon_candidates, log_return_bps

__all__ = [
    "assign_classes",
    "fit_epsilon",
    "fit_epsilon_candidates",
    "log_return_bps",
    "compute_future_return_bps",
    "compute_future_mid_price",
]


def fit_epsilon(train_returns, stable_fraction=0.35):
    abs_ret = pd.Series(train_returns).dropna().abs()
    if abs_ret.empty:
        raise ValueError("Cannot fit epsilon: empty training returns")
    return float(abs_ret.quantile(stable_fraction))


def compute_future_return_bps(current_mid, future_mid):
    return pd.Series(
        log_return_bps(current_mid, future_mid),
        index=getattr(current_mid, "index", None),
    )


def compute_future_mid_price(
    timestamps,
    mid_prices,
    horizon_seconds,
    smoothing_window_seconds,
    match_tolerance_seconds,
):
    """Strictly-future nearest mid within tolerance (legacy fixed-horizon helper)."""
    del smoothing_window_seconds  # unused; retained for API compatibility
    ts = pd.to_datetime(timestamps, utc=True)
    ts_ns = np.asarray([int(t.value) for t in ts], dtype=np.int64)
    mid = np.asarray(mid_prices, dtype=float)
    order = np.argsort(ts_ns)
    sorted_ts = ts_ns[order]
    sorted_mid = mid[order]
    h = int(horizon_seconds * 1e9)
    tol = int(match_tolerance_seconds * 1e9)
    future = np.full(len(ts_ns), np.nan)
    for i, t0 in enumerate(ts_ns):
        center = t0 + h
        first = np.searchsorted(sorted_ts, t0, side="right")
        lo = np.searchsorted(sorted_ts, center - tol, side="left")
        hi = np.searchsorted(sorted_ts, center + tol, side="right")
        lo = max(lo, first)
        if hi > lo:
            idx = int(np.argmin(np.abs(sorted_ts[lo:hi] - center)))
            future[i] = float(sorted_mid[lo:hi][idx])
    return pd.Series(future, index=getattr(timestamps, "index", None), name="future_mid_price")
"""Label construction for short-horizon mid-price direction."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CLASS_DOWN = 0
CLASS_STABLE = 1
CLASS_UP = 2
CLASS_NAMES = {0: "DOWN", 1: "STABLE", 2: "UP"}


def compute_future_mid_price(
    timestamps: pd.Series,
    mid_prices: pd.Series,
    horizon_seconds: float,
    smoothing_window_seconds: float,
    match_tolerance_seconds: float,
) -> pd.Series:
    """
    Compute smoothed future mid-price near t+h.

    Preferred: median mid within [t+h-w, t+h+w] using **strictly future**
    snapshots only (timestamp > t). This prevents zero-return leakage when
    short horizons fall inside sparse gaps and the nearest point would
    otherwise be the current snapshot.

    Fallback: nearest strictly future snapshot to t+h within
    match_tolerance_seconds.
    """
    ts = pd.to_datetime(timestamps, utc=True)
    # Integer nanoseconds since epoch (Timestamp.value is always ns)
    ts_ns = np.asarray([int(t.value) for t in ts], dtype=np.int64)
    mid = mid_prices.to_numpy(dtype=float)
    order = np.argsort(ts_ns)
    sorted_ts = ts_ns[order]
    sorted_mid = mid[order]

    h_ns = int(horizon_seconds * 1_000_000_000)
    w_ns = int(smoothing_window_seconds * 1_000_000_000)
    tol_ns = int(match_tolerance_seconds * 1_000_000_000)
    centers = ts_ns + h_ns

    future = np.full(len(timestamps), np.nan, dtype=float)
    for i, (t0, center) in enumerate(zip(ts_ns, centers)):
        # Strictly future snapshots only
        first_future = np.searchsorted(sorted_ts, t0, side="right")
        if first_future >= len(sorted_ts):
            continue

        lo = np.searchsorted(sorted_ts, center - w_ns, side="left")
        hi = np.searchsorted(sorted_ts, center + w_ns, side="right")
        lo = max(lo, first_future)
        if hi > lo:
            future[i] = float(np.median(sorted_mid[lo:hi]))
            continue

        # Fallback: nearest future snapshot to the horizon center
        lo = np.searchsorted(sorted_ts, center - tol_ns, side="left")
        hi = np.searchsorted(sorted_ts, center + tol_ns, side="right")
        lo = max(lo, first_future)
        if hi > lo:
            window_ts = sorted_ts[lo:hi]
            window_mid = sorted_mid[lo:hi]
            idx = int(np.argmin(np.abs(window_ts - center)))
            future[i] = float(window_mid[idx])

    return pd.Series(future, index=timestamps.index, name="future_mid_price")


def compute_future_return_bps(
    current_mid: pd.Series, future_mid: pd.Series
) -> pd.Series:
    """Future log return in basis points."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = 10_000.0 * np.log(future_mid / current_mid)
    return pd.Series(ret, index=current_mid.index, name="future_return_bps")


def fit_epsilon(
    train_returns: pd.Series,
    stable_fraction: float = 0.35,
) -> float:
    """
    Fit STABLE-class epsilon from training returns only.

    epsilon = quantile(|train returns|, stable_fraction)
    """
    abs_ret = train_returns.dropna().abs()
    if abs_ret.empty:
        raise ValueError("Cannot fit epsilon: empty training returns")
    eps = float(abs_ret.quantile(stable_fraction))
    logger.info(
        "Fitted epsilon=%.6f from %s training returns (stable_fraction=%.2f)",
        eps,
        len(abs_ret),
        stable_fraction,
    )
    return eps


def assign_classes(future_return_bps: pd.Series, epsilon: float) -> pd.Series:
    """Map future returns to DOWN/STABLE/UP using epsilon thresholds."""
    labels = np.full(len(future_return_bps), np.nan)
    valid = future_return_bps.notna()
    r = future_return_bps.to_numpy()
    labels[valid & (r < -epsilon)] = CLASS_DOWN
    labels[valid & (np.abs(r) <= epsilon)] = CLASS_STABLE
    labels[valid & (r > epsilon)] = CLASS_UP
    return pd.Series(labels, index=future_return_bps.index, name="label").astype(
        "Float64"
    )


def label_distribution(labels: pd.Series) -> pd.DataFrame:
    """Return counts and percentages by class."""
    s = labels.dropna().astype(int)
    counts = s.value_counts().reindex([0, 1, 2], fill_value=0)
    pct = 100.0 * counts / counts.sum() if counts.sum() else counts.astype(float)
    return pd.DataFrame(
        {
            "class_code": [0, 1, 2],
            "class_name": ["DOWN", "STABLE", "UP"],
            "count": counts.to_numpy(),
            "percentage": pct.to_numpy(),
        }
    )


def construct_labels_for_horizon(
    df: pd.DataFrame,
    horizon_seconds: int,
    config: dict[str, Any],
    epsilon: float | None = None,
    train_mask: pd.Series | None = None,
) -> tuple[pd.DataFrame, float]:
    """
    Construct labels for a single horizon.

    If epsilon is None, fit from train_mask (required) using training returns only.
    """
    labels_cfg = config["labels"]
    out = df.copy()
    future_mid = compute_future_mid_price(
        out["timestamp"],
        out["mid_price"],
        horizon_seconds=horizon_seconds,
        smoothing_window_seconds=float(labels_cfg["future_smoothing_window_seconds"]),
        match_tolerance_seconds=float(labels_cfg["label_match_tolerance_seconds"]),
    )
    out[f"future_mid_price_{horizon_seconds}s"] = future_mid
    ret = compute_future_return_bps(out["mid_price"], future_mid)
    out[f"future_return_bps_{horizon_seconds}s"] = ret

    if epsilon is None:
        if train_mask is None:
            raise ValueError("train_mask is required when epsilon is not provided")
        epsilon = fit_epsilon(
            ret.loc[train_mask],
            stable_fraction=float(labels_cfg["stable_class_target_fraction"]),
        )

    out[f"label_{horizon_seconds}s"] = assign_classes(ret, epsilon)
    out[f"epsilon_{horizon_seconds}s"] = epsilon
    return out, float(epsilon)


def construct_all_labels(
    df: pd.DataFrame,
    config: dict[str, Any],
    train_mask: pd.Series,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Construct labels for primary and robustness horizons."""
    horizons = [int(config["labels"]["primary_horizon_seconds"])] + [
        int(h) for h in config["labels"]["robustness_horizons_seconds"]
    ]
    horizons = sorted(set(horizons))
    out = df.copy()
    epsilons: dict[str, float] = {}
    for h in horizons:
        out, eps = construct_labels_for_horizon(
            out, h, config, epsilon=None, train_mask=train_mask
        )
        epsilons[f"{h}s"] = eps
        # Convenience alias for primary
        if h == int(config["labels"]["primary_horizon_seconds"]):
            out["label"] = out[f"label_{h}s"]
            out["future_return_bps"] = out[f"future_return_bps_{h}s"]
            out["future_mid_price"] = out[f"future_mid_price_{h}s"]
            out["epsilon"] = eps
    return out, epsilons

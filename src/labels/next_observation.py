"""Study A: next observed mid-price movement labels."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CLASS_DOWN = 0
CLASS_STABLE = 1
CLASS_UP = 2


def log_return_bps(current: pd.Series | np.ndarray, future: pd.Series | np.ndarray) -> np.ndarray:
    """
    Mid-price log return in basis points.

    Formula
    -------
    10000 * log(future / current)

    Units: basis points of log-return (≈ percentage points × 100).
    """
    cur = np.asarray(current, dtype=float)
    fut = np.asarray(future, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10_000.0 * np.log(fut / cur)


def estimate_tick_size(df: pd.DataFrame) -> dict[str, Any]:
    """
    Estimate empirical tick size from positive quote price changes.

    Method: mode of positive differences across best ask/bid and nearby levels.
    """
    diffs = []
    for col in [
        "asks_price_1",
        "bids_price_1",
        "asks_price_2",
        "bids_price_2",
    ]:
        if col not in df.columns:
            continue
        d = df[col].diff().abs()
        diffs.append(d[d > 0])
    if not diffs:
        raise ValueError("Cannot estimate tick size: no price columns")
    all_d = pd.concat(diffs, ignore_index=True)
    # Round to reduce float noise then take mode
    rounded = all_d.round(8)
    mode = float(rounded.mode().iloc[0])
    mid = df["mid_price"] if "mid_price" in df.columns else (
        df["asks_price_1"] + df["bids_price_1"]
    ) / 2.0
    tick_bps = 10_000.0 * mode / mid
    result = {
        "estimated_tick_size_irt": mode,
        "estimated_tick_size_bps_median": float(tick_bps.median()),
        "estimated_tick_size_bps_mean": float(tick_bps.mean()),
        "tick_estimation_method": "mode_positive_quote_diffs",
        "n_positive_diffs": int(len(all_d)),
    }
    logger.info("Tick estimate: %s", result)
    return result


def fit_epsilon_candidates(
    train_returns: pd.Series,
    train_spread_bps: pd.Series,
    tick_bps_median: float,
    config: dict[str, Any],
) -> pd.DataFrame:
    """
    Fit candidate epsilon thresholds on training data only.

    Candidates: quantile, tick, spread-multiple, and hybrid.
    """
    cfg = config.get("study_a", {}).get("epsilon_candidates", {})
    abs_r = train_returns.dropna().abs()
    med_spread = float(train_spread_bps.dropna().median())
    rows = []
    for q in cfg.get("quantiles", [0.35]):
        eps = float(abs_r.quantile(float(q)))
        rows.append({"method": f"quantile_{q}", "epsilon_bps": eps})
    if cfg.get("use_tick_threshold", True):
        rows.append({"method": "tick", "epsilon_bps": float(tick_bps_median)})
    for c in cfg.get("spread_multipliers", [0.5]):
        rows.append(
            {"method": f"spread_x_{c}", "epsilon_bps": float(c) * med_spread}
        )
    # Hybrid
    q_primary = float(cfg.get("quantiles", [0.35])[0])
    eps_q = float(abs_r.quantile(q_primary))
    hybrid = max(eps_q, float(tick_bps_median), 0.5 * med_spread)
    rows.append({"method": "hybrid", "epsilon_bps": float(hybrid)})
    return pd.DataFrame(rows)


def select_epsilon(
    candidates: pd.DataFrame,
    train_returns: pd.Series,
    val_returns: pd.Series | None,
    primary_method: str = "hybrid",
) -> tuple[float, str, pd.DataFrame]:
    """
    Select epsilon by named primary method (default hybrid).

    Validation returns are used only to report resulting class balance,
    not to cherry-pick epsilon from test performance.
    """
    row = candidates.loc[candidates["method"] == primary_method]
    if row.empty:
        row = candidates.loc[candidates["method"] == "hybrid"]
    eps = float(row.iloc[0]["epsilon_bps"])
    method = str(row.iloc[0]["method"])

    def _bal(returns: pd.Series) -> dict[str, float]:
        y = assign_classes(returns, eps)
        vc = y.value_counts(normalize=True)
        return {f"pct_{int(k)}": float(100 * v) for k, v in vc.items()}

    summary = candidates.copy()
    summary["selected"] = summary["method"] == method
    logger.info("Selected epsilon method=%s value=%.6f bps", method, eps)
    if val_returns is not None:
        logger.info("Val class balance under selected epsilon: %s", _bal(val_returns))
    return eps, method, summary


def assign_classes(returns: pd.Series | np.ndarray, epsilon: float) -> pd.Series:
    """Map returns to DOWN/STABLE/UP using epsilon in bps."""
    r = pd.Series(returns)
    labels = pd.Series(np.nan, index=r.index, dtype="Float64")
    valid = r.notna()
    labels.loc[valid & (r < -epsilon)] = CLASS_DOWN
    labels.loc[valid & (r.abs() <= epsilon)] = CLASS_STABLE
    labels.loc[valid & (r > epsilon)] = CLASS_UP
    return labels


def build_study_a_labels(
    df: pd.DataFrame,
    config: dict[str, Any],
    train_mask: np.ndarray,
    val_mask: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Study A: label using the immediately next observed snapshot.

    Research question
    -----------------
    Can the current LOB state predict the direction of the next observed
    mid-price movement (without claiming an exact clock horizon)?
    """
    out = df.sort_values("timestamp").reset_index(drop=True).copy()
    if "mid_price" not in out.columns:
        out["mid_price"] = (out["asks_price_1"] + out["bids_price_1"]) / 2.0
    if "relative_spread_bps" not in out.columns:
        spread = out["asks_price_1"] - out["bids_price_1"]
        out["relative_spread_bps"] = 10_000.0 * spread / out["mid_price"]

    out["current_timestamp"] = out["timestamp"]
    out["target_timestamp"] = out["timestamp"].shift(-1)
    out["actual_delay_seconds"] = (
        out["target_timestamp"] - out["current_timestamp"]
    ).dt.total_seconds()
    out["current_mid_price"] = out["mid_price"]
    out["next_mid_price"] = out["mid_price"].shift(-1)
    out["next_observation_return_bps"] = log_return_bps(
        out["current_mid_price"], out["next_mid_price"]
    )

    tick_info = estimate_tick_size(out.loc[train_mask])
    candidates = fit_epsilon_candidates(
        out.loc[train_mask, "next_observation_return_bps"],
        out.loc[train_mask, "relative_spread_bps"],
        tick_bps_median=float(tick_info["estimated_tick_size_bps_median"]),
        config=config,
    )
    val_ret = (
        out.loc[val_mask, "next_observation_return_bps"] if val_mask is not None else None
    )
    method = config.get("study_a", {}).get("primary_epsilon_method", "hybrid")
    eps, method, cand_table = select_epsilon(
        candidates,
        out.loc[train_mask, "next_observation_return_bps"],
        val_ret,
        primary_method=method,
    )
    out["label"] = assign_classes(out["next_observation_return_bps"], eps)
    out["epsilon_bps"] = eps
    out["epsilon_method"] = method
    # Drop last row (no next observation)
    valid = out["target_timestamp"].notna() & out["label"].notna()
    meta = {
        "study": "A_next_observation",
        "epsilon_bps": eps,
        "epsilon_method": method,
        "tick": tick_info,
        "candidates": cand_table,
        "n_labeled": int(valid.sum()),
        "class_counts": out.loc[valid, "label"]
        .value_counts()
        .sort_index()
        .to_dict(),
        "delay_median": float(out.loc[valid, "actual_delay_seconds"].median()),
        "delay_p95": float(out.loc[valid, "actual_delay_seconds"].quantile(0.95)),
    }
    return out, meta

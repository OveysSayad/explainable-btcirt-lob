"""Revised microstructure feature engineering for sparse LOB snapshots."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.config import numeric_eps
from src.trade_deduplication import add_corrected_trade_features

logger = logging.getLogger(__name__)

FEATURE_FAMILIES: dict[str, list[str]] = {
    "Price history": [
        "return_120s",
        "return_300s",
        "return_600s",
        "return_1200s",
        "mid_return_lag_1",
        "mid_return_lag_2",
        "mid_return_lag_3",
        "mid_return_lag_5",
    ],
    "Static liquidity": [
        "relative_spread_bps",
        "ask_range_8_bps",
        "bid_range_8_bps",
        "ask_concentration",
        "bid_concentration",
        "ask_slope",
        "bid_slope",
    ],
    "Static depth": [
        "log_ask_depth_1",
        "log_ask_depth_2",
        "log_ask_depth_3",
        "log_ask_depth_5",
        "log_ask_depth_8",
        "log_bid_depth_1",
        "log_bid_depth_2",
        "log_bid_depth_3",
        "log_bid_depth_5",
        "log_bid_depth_8",
    ],
    "Static imbalance": [
        "obi_1",
        "obi_2",
        "obi_3",
        "obi_5",
        "obi_8",
        "weighted_obi",
        "microprice_edge_bps",
        "ask_distance_1_bps",
        "ask_distance_2_bps",
        "ask_distance_3_bps",
        "ask_distance_5_bps",
        "ask_distance_8_bps",
        "bid_distance_1_bps",
        "bid_distance_2_bps",
        "bid_distance_3_bps",
        "bid_distance_5_bps",
        "bid_distance_8_bps",
    ],
    "Dynamic liquidity": [
        "spread_change_lag_1",
        "spread_change_lag_2",
        "spread_change_lag_3",
        "spread_change_lag_5",
        "spread_mean_300s",
        "spread_std_300s",
        "spread_mean_600s",
        "spread_std_600s",
    ],
    "Dynamic imbalance": [
        "obi5_change_lag_1",
        "obi5_change_lag_2",
        "obi5_change_lag_3",
        "obi5_change_lag_5",
        "microprice_edge_change_lag_1",
        "microprice_edge_change_lag_2",
        "microprice_edge_change_lag_3",
        "microprice_edge_change_lag_5",
        "obi5_mean_300s",
        "obi5_std_300s",
        "obi5_mean_600s",
        "obi5_std_600s",
        "weighted_obi_mean_300s",
        "microprice_edge_mean_300s",
    ],
    "Snapshot order-flow proxy": [
        "snapshot_ofi_proxy_l1",
        "weighted_snapshot_ofi_proxy",
        "snapshot_ofi_proxy_300s",
        "snapshot_ofi_proxy_600s",
        "normalized_snapshot_ofi_300s",
        "normalized_snapshot_ofi_600s",
        "bid_depth_change_lag_1",
        "ask_depth_change_lag_1",
    ],
    "Volatility": [
        "volatility_120s",
        "volatility_300s",
        "volatility_600s",
        "volatility_1200s",
    ],
    "Trade activity": [
        "new_trade_count_300s",
        "new_trade_count_600s",
        "new_trade_volume_300s",
        "new_trade_volume_600s",
        "trade_intensity_300s",
        "trade_intensity_600s",
        "signed_trade_imbalance_300s",
        "signed_trade_imbalance_600s",
        "time_since_new_trade",
    ],
    "Time": ["hour_sin", "hour_cos", "dow_sin", "dow_cos"],
    "Data-collection metadata": ["observation_gap_seconds"],
}


def _asof_lag(
    timestamps: pd.Series, values: pd.Series, lag_seconds: float, tolerance: float
) -> tuple[np.ndarray, np.ndarray]:
    """Backward as-of lag with actual lag delay tracking."""
    base = pd.DataFrame({"timestamp": timestamps, "value": values}).sort_values("timestamp")
    query = pd.DataFrame(
        {
            "timestamp": timestamps,
            "query_ts": timestamps - pd.Timedelta(seconds=lag_seconds),
            "orig": np.arange(len(timestamps)),
        }
    ).sort_values("query_ts")
    merged = pd.merge_asof(
        query,
        base.rename(columns={"timestamp": "hist_ts"}),
        left_on="query_ts",
        right_on="hist_ts",
        direction="backward",
        tolerance=pd.Timedelta(seconds=tolerance),
    ).sort_values("orig")
    vals = merged["value"].to_numpy()
    # Compute actual lag in seconds; NaT hist_ts yields NaN delay
    hist = pd.to_datetime(merged["hist_ts"], utc=True)
    cur = pd.to_datetime(timestamps, utc=True).reset_index(drop=True)
    actual = (cur - hist.reset_index(drop=True)).dt.total_seconds().to_numpy()
    return vals, actual


def engineer_features(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """
    Build stationary microstructure features for sparse snapshots.

    Time windows default to 120–1200s because median gaps are ~69s.
    Observation-lag features complement irregular clock spacing.
    """
    eps = numeric_eps(config)
    feats = config.get("features", {})
    out = df.sort_values("timestamp").reset_index(drop=True).copy()

    out["best_ask"] = out["asks_price_1"]
    out["best_bid"] = out["bids_price_1"]
    out["mid_price"] = (out["best_ask"] + out["best_bid"]) / 2.0
    out["spread"] = out["best_ask"] - out["best_bid"]
    out["relative_spread_bps"] = 10_000.0 * out["spread"] / out["mid_price"]
    out["observation_gap_seconds"] = out["timestamp"].diff().dt.total_seconds()

    levels = list(feats.get("distance_levels", [1, 2, 3, 5, 8]))
    for i in levels:
        out[f"ask_distance_{i}_bps"] = (
            10_000.0 * (out[f"asks_price_{i}"] - out["mid_price"]) / out["mid_price"]
        )
        out[f"bid_distance_{i}_bps"] = (
            10_000.0 * (out["mid_price"] - out[f"bids_price_{i}"]) / out["mid_price"]
        )

    ask_qty = np.column_stack([out[f"asks_qty_{i}"].to_numpy() for i in range(1, 9)])
    bid_qty = np.column_stack([out[f"bids_qty_{i}"].to_numpy() for i in range(1, 9)])
    depth_levels = list(feats.get("cumulative_depth_levels", [1, 2, 3, 5, 8]))
    for k in depth_levels:
        ad = ask_qty[:, :k].sum(axis=1)
        bd = bid_qty[:, :k].sum(axis=1)
        out[f"ask_depth_{k}"] = ad
        out[f"bid_depth_{k}"] = bd
        out[f"log_ask_depth_{k}"] = np.log1p(ad)
        out[f"log_bid_depth_{k}"] = np.log1p(bd)
        out[f"obi_{k}"] = (bd - ad) / (bd + ad + eps)

    lam = float(feats.get("weighted_obi_lambda", 0.5))
    w = np.array([np.exp(-lam * (i - 1)) for i in range(1, 9)])
    w_bid = (bid_qty * w).sum(axis=1)
    w_ask = (ask_qty * w).sum(axis=1)
    out["weighted_obi"] = (w_bid - w_ask) / (w_bid + w_ask + eps)
    out["microprice"] = (
        out["best_ask"] * out["bids_qty_1"] + out["best_bid"] * out["asks_qty_1"]
    ) / (out["bids_qty_1"] + out["asks_qty_1"] + eps)
    out["microprice_edge_bps"] = (
        10_000.0 * (out["microprice"] - out["mid_price"]) / out["mid_price"]
    )
    out["ask_range_8_bps"] = (
        10_000.0 * (out["asks_price_8"] - out["asks_price_1"]) / out["mid_price"]
    )
    out["bid_range_8_bps"] = (
        10_000.0 * (out["bids_price_1"] - out["bids_price_8"]) / out["mid_price"]
    )
    out["ask_concentration"] = (ask_qty[:, 0] + ask_qty[:, 1]) / (
        ask_qty.sum(axis=1) + eps
    )
    out["bid_concentration"] = (bid_qty[:, 0] + bid_qty[:, 1]) / (
        bid_qty.sum(axis=1) + eps
    )
    out["ask_slope"] = out["ask_range_8_bps"] / (out["log_ask_depth_8"] + eps)
    out["bid_slope"] = out["bid_range_8_bps"] / (out["log_bid_depth_8"] + eps)

    # Snapshot OFI proxy (NOT event OFI)
    bid = out["best_bid"].to_numpy()
    ask = out["best_ask"].to_numpy()
    bq = out["bids_qty_1"].to_numpy()
    aq = out["asks_qty_1"].to_numpy()
    bid_prev, ask_prev = np.roll(bid, 1), np.roll(ask, 1)
    bq_prev, aq_prev = np.roll(bq, 1), np.roll(aq, 1)
    bid_prev[0] = ask_prev[0] = bq_prev[0] = aq_prev[0] = np.nan
    e_b = np.where(bid >= bid_prev, bq, 0.0) - np.where(bid <= bid_prev, bq_prev, 0.0)
    e_a = np.where(ask <= ask_prev, aq, 0.0) - np.where(ask >= ask_prev, aq_prev, 0.0)
    out["snapshot_ofi_proxy_l1"] = e_b - e_a
    out["weighted_snapshot_ofi_proxy"] = out["snapshot_ofi_proxy_l1"]  # L1 weighted equiv
    out["bid_depth_change_lag_1"] = out["bid_depth_1"] - out["bid_depth_1"].shift(1)
    out["ask_depth_change_lag_1"] = out["ask_depth_1"] - out["ask_depth_1"].shift(1)

    # Observation lags
    for lag in feats.get("observation_lags", [1, 2, 3, 5]):
        prev_mid = out["mid_price"].shift(lag)
        out[f"mid_return_lag_{lag}"] = 10_000.0 * np.log(out["mid_price"] / prev_mid)
        out[f"obi5_change_lag_{lag}"] = out["obi_5"] - out["obi_5"].shift(lag)
        out[f"spread_change_lag_{lag}"] = (
            out["relative_spread_bps"] - out["relative_spread_bps"].shift(lag)
        )
        out[f"microprice_edge_change_lag_{lag}"] = (
            out["microprice_edge_bps"] - out["microprice_edge_bps"].shift(lag)
        )

    # Time-based returns with lag tracking
    tol = float(feats.get("lag_match_tolerance_seconds", 90))
    for h in feats.get("time_windows_seconds", [120, 300, 600, 1200]):
        lagged, actual = _asof_lag(out["timestamp"], out["mid_price"], h, tol)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"return_{h}s"] = 10_000.0 * np.log(out["mid_price"].to_numpy() / lagged)
        out[f"return_{h}s_actual_lag"] = actual
        out[f"return_{h}s_lag_error"] = np.abs(actual - h)

    # Rolling stats / volatility / OFI aggregates
    indexed = out.set_index("timestamp").sort_index()
    prev_mid = indexed["mid_price"].shift(1)
    inst = 10_000.0 * np.log(indexed["mid_price"] / prev_mid)
    sq = inst.pow(2)
    for h in feats.get("time_windows_seconds", [120, 300, 600, 1200]):
        indexed[f"volatility_{h}s"] = np.sqrt(sq.rolling(f"{h}s", min_periods=1).sum())
    for h in [300, 600]:
        indexed[f"spread_mean_{h}s"] = indexed["relative_spread_bps"].rolling(
            f"{h}s", min_periods=1
        ).mean()
        indexed[f"spread_std_{h}s"] = indexed["relative_spread_bps"].rolling(
            f"{h}s", min_periods=1
        ).std()
        indexed[f"obi5_mean_{h}s"] = indexed["obi_5"].rolling(f"{h}s", min_periods=1).mean()
        indexed[f"obi5_std_{h}s"] = indexed["obi_5"].rolling(f"{h}s", min_periods=1).std()
        indexed[f"weighted_obi_mean_{h}s"] = indexed["weighted_obi"].rolling(
            f"{h}s", min_periods=1
        ).mean()
        indexed[f"microprice_edge_mean_{h}s"] = indexed["microprice_edge_bps"].rolling(
            f"{h}s", min_periods=1
        ).mean()
        indexed[f"snapshot_ofi_proxy_{h}s"] = indexed["snapshot_ofi_proxy_l1"].rolling(
            f"{h}s", min_periods=1
        ).sum()
        depth = indexed["bids_qty_1"] + indexed["asks_qty_1"]
        indexed[f"normalized_snapshot_ofi_{h}s"] = indexed[
            f"snapshot_ofi_proxy_{h}s"
        ] / (depth.rolling(f"{h}s", min_periods=1).mean() + eps)
    out = indexed.reset_index()

    # Corrected trades
    out = add_corrected_trade_features(out, windows=[300, 600])

    if feats.get("include_time_features", True):
        hour = out["timestamp"].dt.hour + out["timestamp"].dt.minute / 60.0
        out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
        dow = out["timestamp"].dt.dayofweek.astype(float)
        out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
        out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    logger.info("Engineered features; families=%s", list(FEATURE_FAMILIES))
    return out


def get_feature_set(
    df: pd.DataFrame,
    name: str,
    include_time: bool = True,
    include_trade: bool = False,
    include_gap: bool = False,
) -> list[str]:
    """Return feature columns for a named feature-set experiment."""
    families: list[str]
    if name == "price_only":
        families = ["Price history", "Volatility"]
    elif name == "static_lob":
        families = ["Static liquidity", "Static depth", "Static imbalance"]
    elif name == "dynamic_lob":
        families = ["Dynamic liquidity", "Dynamic imbalance", "Snapshot order-flow proxy"]
    elif name == "lob_full":
        families = [
            "Static liquidity",
            "Static depth",
            "Static imbalance",
            "Dynamic liquidity",
            "Dynamic imbalance",
            "Snapshot order-flow proxy",
        ]
    elif name == "full_no_trade":
        families = [
            "Price history",
            "Static liquidity",
            "Static depth",
            "Static imbalance",
            "Dynamic liquidity",
            "Dynamic imbalance",
            "Snapshot order-flow proxy",
            "Volatility",
        ]
        if include_time:
            families.append("Time")
    elif name == "full_with_trade":
        families = [
            "Price history",
            "Static liquidity",
            "Static depth",
            "Static imbalance",
            "Dynamic liquidity",
            "Dynamic imbalance",
            "Snapshot order-flow proxy",
            "Volatility",
            "Trade activity",
        ]
        if include_time:
            families.append("Time")
    elif name == "full_no_time":
        families = [
            "Price history",
            "Static liquidity",
            "Static depth",
            "Static imbalance",
            "Dynamic liquidity",
            "Dynamic imbalance",
            "Snapshot order-flow proxy",
            "Volatility",
        ]
    else:
        raise ValueError(f"Unknown feature set: {name}")

    cols: list[str] = []
    for fam in families:
        for f in FEATURE_FAMILIES.get(fam, []):
            if f in df.columns:
                cols.append(f)
    if include_gap and "observation_gap_seconds" in df.columns:
        cols.append("observation_gap_seconds")
    if not include_trade:
        cols = [c for c in cols if c not in FEATURE_FAMILIES["Trade activity"]]
    # unique preserve order
    return list(dict.fromkeys(cols))


def feature_dictionary_frame() -> pd.DataFrame:
    """Build feature dictionary rows for reporting."""
    rows = []
    for family, feats in FEATURE_FAMILIES.items():
        for f in feats:
            rows.append(
                {
                    "feature_name": f,
                    "feature_family": family,
                    "formula": "see docs/FORMULAS.md and feature_engineering.py",
                    "source_columns": "LOB levels / mid / trades",
                    "window_type": "time_or_event_lag",
                    "window_length": "see name suffix",
                    "stationarity_transformation": "bps / log1p / ratio",
                    "economic_interpretation": family,
                    "known_limitations": (
                        "Snapshot OFI is a proxy; short clock windows may be empty"
                        if "ofi" in f or "snapshot" in f
                        else "Sparse sampling may coarsen intended window"
                    ),
                    "included_in_primary_model": family
                    not in {"Trade activity", "Data-collection metadata"},
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Compatibility helpers for unit tests (compose subsets of engineer_features)
# ---------------------------------------------------------------------------


def compute_mid_and_spread(df: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
    """Add mid-price, spread, and relative spread in bps."""
    out = df.copy()
    out["best_ask"] = out["asks_price_1"]
    out["best_bid"] = out["bids_price_1"]
    out["mid_price"] = (out["best_ask"] + out["best_bid"]) / 2.0
    out["spread"] = out["best_ask"] - out["best_bid"]
    out["relative_spread_bps"] = 10_000.0 * out["spread"] / out["mid_price"]
    return out


def add_depth_features(
    df: pd.DataFrame, levels: list[int] | None = None, eps: float = 1e-12
) -> pd.DataFrame:
    """Cumulative depth, log-depth, and top-of-book concentration."""
    out = df.copy()
    levels = levels or [1, 2, 3, 5, 8]
    ask_qty = np.column_stack([out[f"asks_qty_{i}"].to_numpy() for i in range(1, 9)])
    bid_qty = np.column_stack([out[f"bids_qty_{i}"].to_numpy() for i in range(1, 9)])
    for k in levels:
        ad = ask_qty[:, :k].sum(axis=1)
        bd = bid_qty[:, :k].sum(axis=1)
        out[f"ask_depth_{k}"] = ad
        out[f"bid_depth_{k}"] = bd
        out[f"log_ask_depth_{k}"] = np.log1p(ad)
        out[f"log_bid_depth_{k}"] = np.log1p(bd)
    out["ask_concentration"] = (ask_qty[:, 0] + ask_qty[:, 1]) / (ask_qty.sum(axis=1) + eps)
    out["bid_concentration"] = (bid_qty[:, 0] + bid_qty[:, 1]) / (bid_qty.sum(axis=1) + eps)
    return out


def add_obi_features(
    df: pd.DataFrame,
    levels: list[int] | None = None,
    eps: float = 1e-12,
    lam: float = 0.5,
) -> pd.DataFrame:
    """Order-book imbalance and exponentially weighted OBI."""
    out = df.copy()
    levels = levels or [1, 2, 3, 5, 8]
    ask_qty = np.column_stack([out[f"asks_qty_{i}"].to_numpy() for i in range(1, 9)])
    bid_qty = np.column_stack([out[f"bids_qty_{i}"].to_numpy() for i in range(1, 9)])
    for k in levels:
        ad = ask_qty[:, :k].sum(axis=1)
        bd = bid_qty[:, :k].sum(axis=1)
        out[f"obi_{k}"] = (bd - ad) / (bd + ad + eps)
    w = np.array([np.exp(-lam * (i - 1)) for i in range(1, 9)])
    w_bid = (bid_qty * w).sum(axis=1)
    w_ask = (ask_qty * w).sum(axis=1)
    out["weighted_obi"] = (w_bid - w_ask) / (w_bid + w_ask + eps)
    return out


def add_microprice_features(df: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
    """Quantity-weighted microprice and edge vs mid in bps."""
    out = df.copy()
    out["microprice"] = (
        out["asks_price_1"] * out["bids_qty_1"] + out["bids_price_1"] * out["asks_qty_1"]
    ) / (out["bids_qty_1"] + out["asks_qty_1"] + eps)
    mid = out["mid_price"] if "mid_price" in out.columns else (
        out["asks_price_1"] + out["bids_price_1"]
    ) / 2.0
    out["microprice_edge_bps"] = 10_000.0 * (out["microprice"] - mid) / mid
    return out


def add_ofi_proxy_features(
    df: pd.DataFrame, roll_seconds: list[int] | None = None, eps: float = 1e-12
) -> pd.DataFrame:
    """
    Snapshot order-flow imbalance proxy (Cont-style between snapshots).

    Also aliases ``ofi_proxy`` for legacy tests. This is NOT event-level OFI.
    """
    out = df.sort_values("timestamp").reset_index(drop=True).copy()
    bid = out["bids_price_1"].to_numpy()
    ask = out["asks_price_1"].to_numpy()
    bq = out["bids_qty_1"].to_numpy()
    aq = out["asks_qty_1"].to_numpy()
    bid_prev, ask_prev = np.roll(bid, 1), np.roll(ask, 1)
    bq_prev, aq_prev = np.roll(bq, 1), np.roll(aq, 1)
    bid_prev[0] = ask_prev[0] = bq_prev[0] = aq_prev[0] = np.nan
    e_b = np.where(bid >= bid_prev, bq, 0.0) - np.where(bid <= bid_prev, bq_prev, 0.0)
    e_a = np.where(ask <= ask_prev, aq, 0.0) - np.where(ask >= ask_prev, aq_prev, 0.0)
    out["snapshot_ofi_proxy_l1"] = e_b - e_a
    out["ofi_proxy"] = out["snapshot_ofi_proxy_l1"]  # legacy alias
    if roll_seconds:
        indexed = out.set_index("timestamp").sort_index()
        for h in roll_seconds:
            indexed[f"snapshot_ofi_proxy_{h}s"] = indexed["snapshot_ofi_proxy_l1"].rolling(
                f"{h}s", min_periods=1
            ).sum()
        out = indexed.reset_index()
    return out


def add_historical_returns(
    df: pd.DataFrame, horizons: list[int] | None = None, tolerance: float = 90.0
) -> pd.DataFrame:
    """Clock-time historical mid returns with as-of lag matching."""
    out = df.copy()
    horizons = horizons or [120, 300, 600, 1200]
    for h in horizons:
        lagged, actual = _asof_lag(out["timestamp"], out["mid_price"], float(h), tolerance)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"return_{h}s"] = 10_000.0 * np.log(out["mid_price"].to_numpy() / lagged)
        out[f"return_{h}s_actual_lag"] = actual
    return out


def add_rolling_stats(df: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
    """Backward-looking rolling means for OBI / spread (time-indexed)."""
    out = df.copy()
    windows = windows or [300, 600]
    indexed = out.set_index("timestamp").sort_index()
    if "obi_5" not in indexed.columns:
        indexed["obi_5"] = 0.0
    if "relative_spread_bps" not in indexed.columns and "spread" in indexed.columns:
        mid = indexed.get("mid_price")
        if mid is not None:
            indexed["relative_spread_bps"] = 10_000.0 * indexed["spread"] / mid
    for h in windows:
        if "obi_5" in indexed.columns:
            indexed[f"obi5_mean_{h}s"] = indexed["obi_5"].rolling(f"{h}s", min_periods=1).mean()
        if "relative_spread_bps" in indexed.columns:
            indexed[f"spread_mean_{h}s"] = indexed["relative_spread_bps"].rolling(
                f"{h}s", min_periods=1
            ).mean()
    return indexed.reset_index()

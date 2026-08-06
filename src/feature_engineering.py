"""Market microstructure feature engineering for BTCIRT LOB snapshots."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_FAMILIES: dict[str, list[str]] = {
    "Liquidity": [
        "relative_spread_bps",
        "ask_range_8_bps",
        "bid_range_8_bps",
        "ask_concentration",
        "bid_concentration",
        "delta_spread_5s",
        "delta_spread_15s",
        "delta_spread_30s",
        "spread_mean_120s",
        "spread_std_120s",
        "spread_mean_300s",
        "spread_std_300s",
    ],
    "Depth": [
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
    "Order-book imbalance": [
        "obi_1",
        "obi_2",
        "obi_3",
        "obi_5",
        "obi_8",
        "weighted_obi",
        "microprice_edge_bps",
        "delta_obi5_5s",
        "delta_obi5_15s",
        "delta_obi5_30s",
        "delta_microprice_edge_5s",
        "delta_microprice_edge_15s",
        "delta_microprice_edge_30s",
        "obi5_mean_120s",
        "obi5_std_120s",
        "obi5_mean_300s",
        "obi5_std_300s",
    ],
    "Order flow": [
        "ofi_proxy",
        "ofi_proxy_120s",
        "ofi_proxy_300s",
        "ofi_proxy_600s",
        "normalized_ofi_120s",
        "normalized_ofi_300s",
        "normalized_ofi_600s",
    ],
    "Price dynamics": [
        "ask_distance_1_bps",
        "ask_distance_3_bps",
        "ask_distance_5_bps",
        "ask_distance_8_bps",
        "bid_distance_1_bps",
        "bid_distance_3_bps",
        "bid_distance_5_bps",
        "bid_distance_8_bps",
        "return_5s",
        "return_15s",
        "return_30s",
        "return_60s",
        "return_120s",
    ],
    "Volatility": [
        "volatility_120s",
        "volatility_300s",
        "volatility_600s",
    ],
    "Trade activity": [
        "last_trade_edge_bps",
        "log_last_trade_qty",
        "trade_sign",
        "trade_imbalance_120s",
        "trade_imbalance_300s",
        "trade_intensity_120s",
        "trade_intensity_300s",
        "time_since_last_trade",
    ],
    "Time": [
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
    ],
}


def _eps(config: dict[str, Any]) -> float:
    return float(config["features"]["numeric_epsilon"])


def compute_mid_and_spread(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mid-price and relative spread in basis points."""
    out = df.copy()
    out["best_ask"] = out["asks_price_1"]
    out["best_bid"] = out["bids_price_1"]
    out["mid_price"] = (out["best_ask"] + out["best_bid"]) / 2.0
    out["spread"] = out["best_ask"] - out["best_bid"]
    out["relative_spread_bps"] = 10_000.0 * out["spread"] / out["mid_price"]
    return out


def add_price_distance_features(df: pd.DataFrame, levels: list[int]) -> pd.DataFrame:
    """Ask/bid distance from mid in basis points."""
    out = df.copy()
    mid = out["mid_price"]
    for i in levels:
        out[f"ask_distance_{i}_bps"] = (
            10_000.0 * (out[f"asks_price_{i}"] - mid) / mid
        )
        out[f"bid_distance_{i}_bps"] = (
            10_000.0 * (mid - out[f"bids_price_{i}"]) / mid
        )
    return out


def add_depth_features(
    df: pd.DataFrame, levels: list[int], eps: float
) -> pd.DataFrame:
    """Cumulative depth and log-depth features."""
    out = df.copy()
    ask_qty = np.column_stack([out[f"asks_qty_{i}"].to_numpy() for i in range(1, 9)])
    bid_qty = np.column_stack([out[f"bids_qty_{i}"].to_numpy() for i in range(1, 9)])
    for k in levels:
        ask_depth = ask_qty[:, :k].sum(axis=1)
        bid_depth = bid_qty[:, :k].sum(axis=1)
        out[f"ask_depth_{k}"] = ask_depth
        out[f"bid_depth_{k}"] = bid_depth
        out[f"log_ask_depth_{k}"] = np.log1p(ask_depth)
        out[f"log_bid_depth_{k}"] = np.log1p(bid_depth)
    out["ask_depth_total"] = ask_qty.sum(axis=1)
    out["bid_depth_total"] = bid_qty.sum(axis=1)
    out["ask_concentration"] = (ask_qty[:, 0] + ask_qty[:, 1]) / (
        out["ask_depth_total"] + eps
    )
    out["bid_concentration"] = (bid_qty[:, 0] + bid_qty[:, 1]) / (
        out["bid_depth_total"] + eps
    )
    return out


def add_obi_features(
    df: pd.DataFrame, levels: list[int], eps: float, lam: float
) -> pd.DataFrame:
    """Order-book imbalance and weighted OBI."""
    out = df.copy()
    for k in levels:
        bid_d = out[f"bid_depth_{k}"]
        ask_d = out[f"ask_depth_{k}"]
        out[f"obi_{k}"] = (bid_d - ask_d) / (bid_d + ask_d + eps)

    weights = np.array([np.exp(-lam * (i - 1)) for i in range(1, 9)], dtype=float)
    ask_qty = np.column_stack([out[f"asks_qty_{i}"].to_numpy() for i in range(1, 9)])
    bid_qty = np.column_stack([out[f"bids_qty_{i}"].to_numpy() for i in range(1, 9)])
    w_bid = (bid_qty * weights).sum(axis=1)
    w_ask = (ask_qty * weights).sum(axis=1)
    out["weighted_obi"] = (w_bid - w_ask) / (w_bid + w_ask + eps)
    return out


def add_microprice_features(df: pd.DataFrame, eps: float) -> pd.DataFrame:
    """Microprice and microprice edge in bps."""
    out = df.copy()
    out["microprice"] = (
        out["best_ask"] * out["bids_qty_1"] + out["best_bid"] * out["asks_qty_1"]
    ) / (out["bids_qty_1"] + out["asks_qty_1"] + eps)
    out["microprice_edge_bps"] = (
        10_000.0 * (out["microprice"] - out["mid_price"]) / out["mid_price"]
    )
    return out


def add_range_features(df: pd.DataFrame) -> pd.DataFrame:
    """Order-book range across 8 levels in bps."""
    out = df.copy()
    mid = out["mid_price"]
    out["ask_range_8_bps"] = 10_000.0 * (out["asks_price_8"] - out["asks_price_1"]) / mid
    out["bid_range_8_bps"] = 10_000.0 * (out["bids_price_1"] - out["bids_price_8"]) / mid
    return out


def _asof_lag_values(
    timestamps: pd.Series, values: pd.Series, lag_seconds: float
) -> np.ndarray:
    """
    Backward-looking as-of lag: value at the latest timestamp <= t - lag.

    Uses only historical information (no future leakage).
    """
    base = pd.DataFrame({"timestamp": timestamps, "value": values}).sort_values(
        "timestamp"
    )
    query = pd.DataFrame(
        {
            "timestamp": timestamps,
            "query_ts": timestamps - pd.Timedelta(seconds=lag_seconds),
            "orig_idx": np.arange(len(timestamps)),
        }
    ).sort_values("query_ts")
    merged = pd.merge_asof(
        query,
        base.rename(columns={"timestamp": "hist_ts"}),
        left_on="query_ts",
        right_on="hist_ts",
        direction="backward",
    )
    merged = merged.sort_values("orig_idx")
    return merged["value"].to_numpy()


def add_historical_returns(
    df: pd.DataFrame, horizons: list[int]
) -> pd.DataFrame:
    """Backward-looking mid-price log returns in basis points."""
    out = df.copy()
    mid = out["mid_price"]
    for h in horizons:
        lagged = _asof_lag_values(out["timestamp"], mid, h)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"return_{h}s"] = 10_000.0 * np.log(mid.to_numpy() / lagged)
    return out


def add_realized_volatility(
    df: pd.DataFrame, horizons: list[int]
) -> pd.DataFrame:
    """
    Realized volatility from rolling squared short-horizon log returns.

    Implementation:
    1. Compute consecutive snapshot log-return in bps: r_t = 10000 * log(mid_t / mid_{t-1}).
    2. For each horizon H, volatility_H = sqrt( sum_{s in (t-H, t]} r_s^2 ).
    Uses time-based rolling windows (not row counts) with min_periods=1.

    Note: with ~69s median gaps, configure H >= 120s so windows contain history.
    """
    out = df.set_index("timestamp").sort_index()
    prev_mid = out["mid_price"].shift(1)
    instant_ret = 10_000.0 * np.log(out["mid_price"] / prev_mid)
    sq = instant_ret.pow(2)
    for h in horizons:
        out[f"volatility_{h}s"] = np.sqrt(
            sq.rolling(f"{h}s", min_periods=1).sum()
        )
    out = out.reset_index()
    return out


def add_dynamic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Deltas of OBI, spread, and microprice edge."""
    out = df.copy()
    for h in (5, 15, 30):
        out[f"delta_obi5_{h}s"] = out["obi_5"] - _asof_lag_values(
            out["timestamp"], out["obi_5"], h
        )
        out[f"delta_spread_{h}s"] = out["relative_spread_bps"] - _asof_lag_values(
            out["timestamp"], out["relative_spread_bps"], h
        )
        out[f"delta_microprice_edge_{h}s"] = out[
            "microprice_edge_bps"
        ] - _asof_lag_values(out["timestamp"], out["microprice_edge_bps"], h)
    return out


def add_rolling_stats(df: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
    """
    Backward-looking rolling mean/std for OBI and spread.

    Default windows are lengthened for sparse LOB snapshots (~1 minute gaps).
    Rolling uses the trailing window ending at t (no future points).
    """
    if windows is None:
        windows = [120, 300]
    out = df.set_index("timestamp").sort_index()
    for col, prefix in [("obi_5", "obi5"), ("relative_spread_bps", "spread")]:
        for window in windows:
            rolled = out[col].rolling(f"{window}s", min_periods=1)
            out[f"{prefix}_mean_{window}s"] = rolled.mean()
            out[f"{prefix}_std_{window}s"] = rolled.std()
    return out.reset_index()


def add_ofi_proxy_features(df: pd.DataFrame, roll_seconds: list[int], eps: float) -> pd.DataFrame:
    """
    Snapshot-based order-flow imbalance proxy (NOT true event OFI).

    Cont et al.-style best-level contribution using consecutive snapshots.
    """
    out = df.copy()
    bid = out["best_bid"].to_numpy()
    ask = out["best_ask"].to_numpy()
    bq = out["bids_qty_1"].to_numpy()
    aq = out["asks_qty_1"].to_numpy()

    bid_prev = np.roll(bid, 1)
    ask_prev = np.roll(ask, 1)
    bq_prev = np.roll(bq, 1)
    aq_prev = np.roll(aq, 1)
    bid_prev[0] = np.nan
    ask_prev[0] = np.nan
    bq_prev[0] = np.nan
    aq_prev[0] = np.nan

    bid_component = np.where(bid >= bid_prev, bq, 0.0) - np.where(
        bid <= bid_prev, bq_prev, 0.0
    )
    ask_component = np.where(ask <= ask_prev, aq, 0.0) - np.where(
        ask >= ask_prev, aq_prev, 0.0
    )
    ofi = bid_component - ask_component
    out["ofi_proxy"] = ofi
    out["snapshot_ofi_proxy"] = ofi

    indexed = out.set_index("timestamp").sort_index()
    depth_l1 = indexed["bids_qty_1"] + indexed["asks_qty_1"]
    for h in roll_seconds:
        indexed[f"ofi_proxy_{h}s"] = indexed["ofi_proxy"].rolling(
            f"{h}s", min_periods=1
        ).sum()
        avg_depth = depth_l1.rolling(f"{h}s", min_periods=1).mean()
        indexed[f"normalized_ofi_{h}s"] = indexed[f"ofi_proxy_{h}s"] / (avg_depth + eps)
    out = indexed.reset_index()
    return out


def add_trade_features(df: pd.DataFrame, eps: float) -> pd.DataFrame:
    """Optional trade-related features using new last-trade detection."""
    out = df.copy()
    if "last_trade_price" not in out.columns or "last_trade_qty" not in out.columns:
        logger.warning("Trade columns missing; skipping trade features")
        return out

    price = out["last_trade_price"]
    qty = out["last_trade_qty"]
    # New trade when price or qty changes relative to previous snapshot.
    new_trade = (price != price.shift(1)) | (qty != qty.shift(1))
    new_trade = new_trade.fillna(True)
    out["is_new_trade"] = new_trade.astype(int)

    out["last_trade_edge_bps"] = np.where(
        new_trade,
        10_000.0 * (price - out["mid_price"]) / out["mid_price"],
        np.nan,
    )
    out["log_last_trade_qty"] = np.where(new_trade, np.log1p(qty), np.nan)

    # Quote rule + tick rule for trade sign
    sign = np.zeros(len(out), dtype=float)
    tp = price.to_numpy()
    best_ask = out["best_ask"].to_numpy()
    best_bid = out["best_bid"].to_numpy()
    buy = tp >= best_ask
    sell = tp <= best_bid
    sign[buy] = 1.0
    sign[sell] = -1.0
    # Tick rule for trades inside the spread
    inside = (~buy) & (~sell) & new_trade.to_numpy()
    prev_tp = np.roll(tp, 1)
    prev_tp[0] = np.nan
    tick_up = inside & (tp > prev_tp)
    tick_dn = inside & (tp < prev_tp)
    sign[tick_up] = 1.0
    sign[tick_dn] = -1.0
    sign[~new_trade.to_numpy()] = np.nan
    out["trade_sign"] = sign

    signed_qty = np.where(new_trade, sign * qty, 0.0)
    trade_count = new_trade.astype(float).to_numpy()

    indexed = out.set_index("timestamp").sort_index()
    signed = pd.Series(signed_qty, index=indexed.index)
    counts = pd.Series(trade_count, index=indexed.index)
    for h in (120, 300):
        indexed[f"trade_imbalance_{h}s"] = signed.rolling(f"{h}s", min_periods=1).sum()
        indexed[f"trade_intensity_{h}s"] = counts.rolling(f"{h}s", min_periods=1).sum()

    # Time since last new trade (seconds)
    trade_ts = indexed.index.to_series().where(indexed["is_new_trade"].astype(bool))
    last_trade_ts = trade_ts.ffill()
    indexed["time_since_last_trade"] = (
        indexed.index.to_series() - last_trade_ts
    ).dt.total_seconds()

    indexed["last_trade_edge_bps"] = indexed["last_trade_edge_bps"].ffill()
    indexed["log_last_trade_qty"] = indexed["log_last_trade_qty"].ffill()
    indexed["trade_sign"] = indexed["trade_sign"].ffill().fillna(0.0)
    return indexed.reset_index()


def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclical hour and day-of-week features."""
    out = df.copy()
    ts = out["timestamp"]
    hour = ts.dt.hour + ts.dt.minute / 60.0
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    dow = ts.dt.dayofweek.astype(float)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    return out


def feature_dictionary() -> pd.DataFrame:
    """Build a data dictionary for all engineered features."""
    rows: list[dict[str, str]] = []
    descriptions = {
        "relative_spread_bps": (
            "Liquidity",
            "10000*(best_ask-best_bid)/mid",
            "Relative bid-ask spread in basis points; higher values imply thinner liquidity.",
        ),
        "obi_5": (
            "Order-book imbalance",
            "(bid_depth_5-ask_depth_5)/(bid_depth_5+ask_depth_5+eps)",
            "Signed pressure from top-5 depth; positive suggests buy-side dominance.",
        ),
        "weighted_obi": (
            "Order-book imbalance",
            "exp-decay weighted qty imbalance",
            "Imbalance emphasizing near-touch levels.",
        ),
        "microprice_edge_bps": (
            "Order-book imbalance",
            "10000*(microprice-mid)/mid",
            "Microprice displacement from mid; anticipates short-horizon mid moves.",
        ),
        "ofi_proxy": (
            "Order flow",
            "snapshot Cont-style best-level OFI",
            "Approximation of order-flow imbalance from consecutive snapshots, not event OFI.",
        ),
        "volatility_60s": (
            "Volatility",
            "sqrt(sum of squared snapshot log-returns over 60s)",
            "Short-horizon realized volatility regime indicator.",
        ),
    }
    for family, feats in FEATURE_FAMILIES.items():
        for feat in feats:
            if feat in descriptions:
                fam, formula, interp = descriptions[feat]
            else:
                fam, formula, interp = family, "see feature_engineering.py", family
            rows.append(
                {
                    "feature": feat,
                    "family": fam,
                    "formula": formula,
                    "interpretation": interp,
                }
            )
    return pd.DataFrame(rows)


def get_model_feature_columns(
    df: pd.DataFrame, include_trade: bool = True
) -> list[str]:
    """Return engineered feature columns present in the frame."""
    columns: list[str] = []
    for family, feats in FEATURE_FAMILIES.items():
        if family == "Trade activity" and not include_trade:
            continue
        for f in feats:
            if f in df.columns:
                columns.append(f)
    # Preserve order, unique
    seen: set[str] = set()
    ordered: list[str] = []
    for c in columns:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


def engineer_features(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Run the full feature engineering pipeline."""
    eps = _eps(config)
    feats_cfg = config["features"]
    out = df.copy()
    out = compute_mid_and_spread(out)
    out = add_price_distance_features(out, feats_cfg["distance_levels"])
    out = add_depth_features(out, feats_cfg["depth_levels"], eps)
    out = add_obi_features(
        out, feats_cfg["depth_levels"], eps, float(feats_cfg["weighted_obi_lambda"])
    )
    out = add_microprice_features(out, eps)
    out = add_range_features(out)
    out = add_historical_returns(out, feats_cfg["return_horizons_seconds"])
    out = add_realized_volatility(out, feats_cfg["volatility_horizons_seconds"])
    out = add_dynamic_features(out)
    out = add_rolling_stats(out, windows=list(feats_cfg.get("rolling_stat_windows_seconds", [120, 300])))
    out = add_ofi_proxy_features(out, feats_cfg["ofi_roll_seconds"], eps)
    if feats_cfg.get("include_trade_features", True):
        out = add_trade_features(out, eps)
    if feats_cfg.get("include_cyclical_time", True):
        out = add_cyclical_time_features(out)

    feature_cols = get_model_feature_columns(
        out, include_trade=bool(feats_cfg.get("include_trade_features", True))
    )
    logger.info(
        "Engineered %s model features from %s rows",
        len(feature_cols),
        len(out),
    )
    return out

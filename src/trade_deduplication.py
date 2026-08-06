"""Trade deduplication for repeated last_trade snapshots."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_trade_signature(df: pd.DataFrame) -> pd.Series:
    """
    Build a fallback trade signature from price/quantity.

    Preferred identifiers (trade id / trade timestamp) are used when present.
    Otherwise signature = (last_trade_price, last_trade_qty).

    Limitation
    ----------
    Two consecutive identical genuine trades would be under-counted.
    """
    if "trade_id" in df.columns:
        return df["trade_id"].astype(str)
    if "last_trade_timestamp" in df.columns:
        return (
            df["last_trade_timestamp"].astype(str)
            + "|"
            + df["last_trade_price"].astype(str)
            + "|"
            + df["last_trade_qty"].astype(str)
        )
    return (
        df["last_trade_price"].astype(str) + "|" + df["last_trade_qty"].astype(str)
    )


def deduplicate_trades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mark new trades and zero out repeated last_trade fields.

    Returns a copy with:
    trade_signature, is_new_trade, new_trade_qty, trade_count_increment,
    seconds_since_new_trade.
    """
    out = df.copy()
    if "last_trade_price" not in out.columns or "last_trade_qty" not in out.columns:
        logger.warning("Trade columns missing; skipping deduplication")
        out["is_new_trade"] = 0
        out["new_trade_qty"] = 0.0
        out["trade_count_increment"] = 0
        out["seconds_since_new_trade"] = np.nan
        return out

    out = out.sort_values("timestamp").reset_index(drop=True)
    out["trade_signature"] = build_trade_signature(out)
    prev = out["trade_signature"].shift(1)
    is_new = out["trade_signature"].ne(prev) | prev.isna()
    out["is_new_trade"] = is_new.astype(int)
    out["new_trade_qty"] = np.where(is_new, out["last_trade_qty"].astype(float), 0.0)
    out["trade_count_increment"] = out["is_new_trade"]

    trade_ts = out["timestamp"].where(is_new)
    last_new = trade_ts.ffill()
    out["seconds_since_new_trade"] = (
        out["timestamp"] - last_new
    ).dt.total_seconds()

    n_rep = int((~is_new).sum())
    logger.info(
        "Trade deduplication: %s/%s snapshots are repeated last_trade signatures",
        n_rep,
        len(out),
    )
    return out


def infer_trade_sign(df: pd.DataFrame) -> pd.Series:
    """
    Infer trade aggressor sign using quotes, then tick rule.

    +1 buyer-initiated, -1 seller-initiated, 0 unknown.
    Only defined on new trades.
    """
    sign = np.zeros(len(df), dtype=float)
    if "is_new_trade" not in df.columns:
        return pd.Series(sign, index=df.index, name="trade_sign")

    new = df["is_new_trade"].astype(bool).to_numpy()
    tp = df["last_trade_price"].to_numpy(dtype=float)
    ask = df["asks_price_1"].to_numpy(dtype=float)
    bid = df["bids_price_1"].to_numpy(dtype=float)
    buy = new & (tp >= ask)
    sell = new & (tp <= bid)
    sign[buy] = 1.0
    sign[sell] = -1.0
    inside = new & (~buy) & (~sell)
    prev_tp = np.roll(tp, 1)
    prev_tp[0] = np.nan
    sign[inside & (tp > prev_tp)] = 1.0
    sign[inside & (tp < prev_tp)] = -1.0
    sign[~new] = np.nan
    return pd.Series(sign, index=df.index, name="trade_sign")


def add_corrected_trade_features(
    df: pd.DataFrame, windows: list[int] | None = None
) -> pd.DataFrame:
    """Aggregate corrected trade intensity/imbalance over time windows."""
    if windows is None:
        windows = [300, 600]
    out = deduplicate_trades(df)
    out["trade_sign"] = infer_trade_sign(out)
    signed = np.where(
        out["is_new_trade"].astype(bool),
        out["trade_sign"].fillna(0.0) * out["new_trade_qty"],
        0.0,
    )
    indexed = out.set_index("timestamp").sort_index()
    signed_s = pd.Series(signed, index=indexed.index)
    counts = indexed["trade_count_increment"].astype(float)
    vol = indexed["new_trade_qty"].astype(float)
    for h in windows:
        indexed[f"new_trade_count_{h}s"] = counts.rolling(f"{h}s", min_periods=1).sum()
        indexed[f"new_trade_volume_{h}s"] = vol.rolling(f"{h}s", min_periods=1).sum()
        indexed[f"trade_intensity_{h}s"] = indexed[f"new_trade_count_{h}s"]
        indexed[f"signed_trade_imbalance_{h}s"] = signed_s.rolling(
            f"{h}s", min_periods=1
        ).sum()
    indexed["time_since_new_trade"] = indexed["seconds_since_new_trade"]
    return indexed.reset_index()

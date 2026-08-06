"""Data loading utilities for Nobitex BTCIRT LOB snapshots."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

LOB_PRICE_QTY_COLUMNS = [
    f"{side}_{field}_{level}"
    for side in ("asks", "bids")
    for level in range(1, 9)
    for field in ("price", "qty")
]

TRADE_COLUMNS = ["last_trade_price", "last_trade_qty"]

CORE_COLUMNS = ["id", "timestamp", "exchange", "symbol"]


def required_columns(config: dict[str, Any]) -> list[str]:
    """Return the flattened columns required for the pipeline."""
    cols = CORE_COLUMNS + LOB_PRICE_QTY_COLUMNS + TRADE_COLUMNS
    return cols


def _normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize exchange/symbol string columns."""
    out = df.copy()
    if "exchange" in out.columns:
        out["exchange"] = out["exchange"].astype(str).str.strip().str.lower()
    if "symbol" in out.columns:
        out["symbol"] = out["symbol"].astype(str).str.strip().str.upper()
    return out


def summarize_raw_catalog(
    raw_path: Path,
    chunk_size: int = 500_000,
) -> dict[str, Any]:
    """Scan the raw CSV for unique exchanges/symbols and row counts."""
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {raw_path}. "
            "Place market_data_clean_nobitex.csv under data/raw/."
        )

    exchanges: set[str] = set()
    symbols: set[str] = set()
    total_rows = 0
    btcirt_rows = 0
    btcirt_timestamps: list[pd.Timestamp] = []

    usecols = ["timestamp", "exchange", "symbol"]
    for chunk in pd.read_csv(
        raw_path,
        usecols=usecols,
        chunksize=chunk_size,
        dtype={"exchange": str, "symbol": str},
        low_memory=False,
    ):
        chunk = _normalize_frame(chunk)
        exchanges.update(chunk["exchange"].dropna().unique().tolist())
        symbols.update(chunk["symbol"].dropna().unique().tolist())
        total_rows += len(chunk)
        mask = (chunk["exchange"] == "nobitex") & (chunk["symbol"] == "BTCIRT")
        btcirt_rows += int(mask.sum())
        if mask.any():
            ts = pd.to_datetime(chunk.loc[mask, "timestamp"], utc=True, errors="coerce")
            btcirt_timestamps.extend(ts.dropna().tolist())

    if btcirt_rows == 0:
        raise ValueError(
            "BTCIRT is absent after filtering. "
            f"Found exchanges={sorted(exchanges)}, symbols={sorted(symbols)}. "
            "Cannot continue without BTCIRT rows."
        )

    ts_series = pd.Series(btcirt_timestamps)
    catalog = {
        "unique_exchanges": sorted(exchanges),
        "unique_symbols": sorted(symbols),
        "total_rows": int(total_rows),
        "btcirt_rows": int(btcirt_rows),
        "pct_retained": float(100.0 * btcirt_rows / total_rows) if total_rows else 0.0,
        "btcirt_min_timestamp": str(ts_series.min()),
        "btcirt_max_timestamp": str(ts_series.max()),
        "btcirt_unique_dates": int(ts_series.dt.date.nunique()),
    }
    logger.info("Raw catalog summary: %s", json.dumps(catalog, indent=2))
    return catalog


def load_btcirt(
    raw_path: Path,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Load and filter Nobitex BTCIRT rows from the raw CSV."""
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {raw_path}")

    exchange = str(config["data"]["exchange"]).strip().lower()
    symbol = str(config["data"]["symbol"]).strip().upper()
    chunk_size = int(config["data"]["chunk_size"])
    drop_json = bool(config["data"].get("drop_json_columns", True))
    json_cols = set(config["data"].get("json_columns", []))

    available = pd.read_csv(raw_path, nrows=0).columns.tolist()
    usecols = [c for c in required_columns(config) if c in available]
    missing_core = [c for c in CORE_COLUMNS + LOB_PRICE_QTY_COLUMNS if c not in available]
    if missing_core:
        raise ValueError(f"Missing required columns in raw CSV: {missing_core}")

    # Decision: JSON-like columns (data/asks/bids/last_trade) are unused when
    # flattened asks_*/bids_*/last_trade_* columns are complete.
    if not drop_json:
        usecols = [c for c in available if c not in json_cols or c in usecols]

    dtypes: dict[str, Any] = {
        "id": "Int64",
        "exchange": "string",
        "symbol": "string",
    }
    for col in LOB_PRICE_QTY_COLUMNS + TRADE_COLUMNS:
        if col in usecols:
            dtypes[col] = "float64"

    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        raw_path,
        usecols=usecols,
        chunksize=chunk_size,
        dtype={k: v for k, v in dtypes.items() if k in usecols and k != "id"},
        low_memory=False,
    ):
        chunk = _normalize_frame(chunk)
        mask = (chunk["exchange"] == exchange) & (chunk["symbol"] == symbol)
        if mask.any():
            frames.append(chunk.loc[mask].copy())

    if not frames:
        raise ValueError(
            f"No rows found for exchange={exchange!r}, symbol={symbol!r}. "
            "Do not silently continue without BTCIRT data."
        )

    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    logger.info(
        "Loaded %s BTCIRT rows from %s (exchange=%s, symbol=%s). "
        "JSON columns dropped=%s because flattened LOB columns are complete.",
        len(df),
        raw_path,
        exchange,
        symbol,
        drop_json,
    )
    return df


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame to Parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Saved parquet: %s (%s rows)", path, len(df))


def load_parquet(path: Path) -> pd.DataFrame:
    """Load a Parquet file."""
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df

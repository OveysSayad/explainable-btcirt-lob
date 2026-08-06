"""Order-book and schema validation for BTCIRT LOB snapshots."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data_loader import LOB_PRICE_QTY_COLUMNS, TRADE_COLUMNS

logger = logging.getLogger(__name__)


def validate_schema(df: pd.DataFrame) -> dict[str, Any]:
    """Validate presence of expected columns."""
    expected = (
        ["id", "timestamp", "exchange", "symbol"]
        + LOB_PRICE_QTY_COLUMNS
        + TRADE_COLUMNS
    )
    missing = [c for c in expected if c not in df.columns]
    present = [c for c in expected if c in df.columns]
    report = {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "missing_expected_columns": missing,
        "present_expected_columns": present,
        "schema_ok": len(missing) == 0,
    }
    if missing:
        logger.warning("Missing expected columns: %s", missing)
    return report


def _ask_prices(df: pd.DataFrame) -> pd.DataFrame:
    return df[[f"asks_price_{i}" for i in range(1, 9)]]


def _bid_prices(df: pd.DataFrame) -> pd.DataFrame:
    return df[[f"bids_price_{i}" for i in range(1, 9)]]


def _ask_qtys(df: pd.DataFrame) -> pd.DataFrame:
    return df[[f"asks_qty_{i}" for i in range(1, 9)]]


def _bid_qtys(df: pd.DataFrame) -> pd.DataFrame:
    return df[[f"bids_qty_{i}" for i in range(1, 9)]]


def audit_order_book_quality(df: pd.DataFrame) -> dict[str, Any]:
    """Count order-book quality violations without silently dropping them."""
    n = len(df)
    asks = _ask_prices(df)
    bids = _bid_prices(df)
    ask_q = _ask_qtys(df)
    bid_q = _bid_qtys(df)

    ask_sorted = asks.diff(axis=1).iloc[:, 1:]
    bid_sorted = -bids.diff(axis=1).iloc[:, 1:]

    asks_not_nondecreasing = (ask_sorted < -1e-12).any(axis=1)
    bids_not_nonincreasing = (bid_sorted < -1e-12).any(axis=1)

    zero_or_neg_ask_price = (asks <= 0).any(axis=1)
    zero_or_neg_bid_price = (bids <= 0).any(axis=1)
    negative_ask_qty = (ask_q < 0).any(axis=1)
    negative_bid_qty = (bid_q < 0).any(axis=1)

    best_ask = df["asks_price_1"]
    best_bid = df["bids_price_1"]
    crossed = best_ask < best_bid
    locked = best_ask == best_bid
    valid_spread = best_ask > best_bid

    infinite_mask = np.isinf(df[LOB_PRICE_QTY_COLUMNS + TRADE_COLUMNS].to_numpy()).any(
        axis=1
    )
    missing_lob = df[LOB_PRICE_QTY_COLUMNS].isna().any(axis=1)

    report = {
        "n_rows": n,
        "invalid_timestamp": int(df["timestamp"].isna().sum()),
        "duplicate_timestamps": int(df["timestamp"].duplicated().sum()),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "missing_lob_values": int(missing_lob.sum()),
        "infinite_values": int(infinite_mask.sum()),
        "zero_or_negative_ask_prices": int(zero_or_neg_ask_price.sum()),
        "zero_or_negative_bid_prices": int(zero_or_neg_bid_price.sum()),
        "negative_ask_quantities": int(negative_ask_qty.sum()),
        "negative_bid_quantities": int(negative_bid_qty.sum()),
        "asks_not_nondecreasing": int(asks_not_nondecreasing.sum()),
        "bids_not_nonincreasing": int(bids_not_nonincreasing.sum()),
        "crossed_books": int(crossed.sum()),
        "locked_books": int(locked.sum()),
        "valid_ask_gt_bid": int(valid_spread.sum()),
        "pct_crossed": float(100.0 * crossed.mean()) if n else 0.0,
        "pct_locked": float(100.0 * locked.mean()) if n else 0.0,
        "pct_asks_unordered": float(100.0 * asks_not_nondecreasing.mean()) if n else 0.0,
        "pct_bids_unordered": float(100.0 * bids_not_nonincreasing.mean()) if n else 0.0,
    }
    logger.info("Order-book quality audit: %s", json.dumps(report, indent=2))
    return report


def audit_timestamp_gaps(df: pd.DataFrame) -> dict[str, Any]:
    """Compute timestamp gap statistics (seconds)."""
    ts = df["timestamp"].sort_values()
    gaps = ts.diff().dt.total_seconds().dropna()
    if gaps.empty:
        return {"n_gaps": 0}

    def pct_gt(threshold: float) -> float:
        return float(100.0 * (gaps > threshold).mean())

    report = {
        "n_gaps": int(len(gaps)),
        "median_gap_seconds": float(gaps.median()),
        "mean_gap_seconds": float(gaps.mean()),
        "std_gap_seconds": float(gaps.std()),
        "min_gap_seconds": float(gaps.min()),
        "max_gap_seconds": float(gaps.max()),
        "p05_gap_seconds": float(gaps.quantile(0.05)),
        "p25_gap_seconds": float(gaps.quantile(0.25)),
        "p75_gap_seconds": float(gaps.quantile(0.75)),
        "p95_gap_seconds": float(gaps.quantile(0.95)),
        "p99_gap_seconds": float(gaps.quantile(0.99)),
        "pct_gaps_gt_10s": pct_gt(10),
        "pct_gaps_gt_30s": pct_gt(30),
        "pct_gaps_gt_60s": pct_gt(60),
        "pct_gaps_gt_300s": pct_gt(300),
        "unique_dates": int(ts.dt.date.nunique()),
        "min_timestamp": str(ts.min()),
        "max_timestamp": str(ts.max()),
    }
    # Decision note for grid alignment
    median_gap = report["median_gap_seconds"]
    report["five_second_grid_justified"] = bool(median_gap <= 10.0)
    report["grid_alignment_decision"] = (
        "Align to 5s grid"
        if report["five_second_grid_justified"]
        else (
            f"Do NOT align to 5s grid: median gap is {median_gap:.1f}s. "
            "Using native snapshot timestamps; label matching uses widened tolerance."
        )
    )
    logger.info("Timestamp gap audit: %s", json.dumps(report, indent=2))
    return report


def build_quality_masks(df: pd.DataFrame) -> pd.DataFrame:
    """Return boolean quality flags per row for cleaning decisions."""
    asks = _ask_prices(df)
    bids = _bid_prices(df)
    ask_q = _ask_qtys(df)
    bid_q = _bid_qtys(df)

    ask_sorted_ok = (asks.diff(axis=1).iloc[:, 1:] >= -1e-12).all(axis=1)
    bid_sorted_ok = ((-bids.diff(axis=1).iloc[:, 1:]) >= -1e-12).all(axis=1)
    prices_ok = (asks > 0).all(axis=1) & (bids > 0).all(axis=1)
    qtys_ok = (ask_q >= 0).all(axis=1) & (bid_q >= 0).all(axis=1)
    not_crossed = df["asks_price_1"] > df["bids_price_1"]
    finite = np.isfinite(df[LOB_PRICE_QTY_COLUMNS].to_numpy()).all(axis=1)
    complete = ~df[LOB_PRICE_QTY_COLUMNS].isna().any(axis=1)
    valid_ts = df["timestamp"].notna()

    flags = pd.DataFrame(
        {
            "ask_sorted_ok": ask_sorted_ok.to_numpy(),
            "bid_sorted_ok": bid_sorted_ok.to_numpy(),
            "prices_ok": prices_ok.to_numpy(),
            "qtys_ok": qtys_ok.to_numpy(),
            "not_crossed": not_crossed.to_numpy(),
            "finite": finite,
            "complete": complete.to_numpy(),
            "valid_ts": valid_ts.to_numpy(),
        },
        index=df.index,
    )
    flags["is_valid_book"] = flags.all(axis=1)
    return flags


def save_quality_report(
    catalog: dict[str, Any],
    schema: dict[str, Any],
    book_quality: dict[str, Any],
    gap_stats: dict[str, Any],
    metrics_dir: Path,
    tables_dir: Path,
) -> dict[str, Any]:
    """Persist data-quality JSON and CSV summary."""
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    full = {
        "catalog": catalog,
        "schema": schema,
        "order_book_quality": book_quality,
        "timestamp_gaps": gap_stats,
        "notes": {
            "json_columns": (
                "Raw JSON-like columns data/asks/bids/last_trade are not used; "
                "flattened asks_*/bids_*/last_trade_* columns are complete."
            ),
            "duplicate_timestamp_rule": (
                "Keep the last observation for each duplicate timestamp "
                "(most recent within the duplicate group)."
            ),
        },
    }
    out_json = metrics_dir / "data_quality.json"
    with out_json.open("w", encoding="utf-8") as handle:
        json.dump(full, handle, indent=2, default=str)

    rows = []
    for section, payload in [
        ("catalog", catalog),
        ("schema", {k: v for k, v in schema.items() if not isinstance(v, list)}),
        ("order_book_quality", book_quality),
        ("timestamp_gaps", gap_stats),
    ]:
        for key, value in payload.items():
            rows.append({"section": section, "metric": key, "value": value})
    summary = pd.DataFrame(rows)
    summary.to_csv(tables_dir / "data_quality_summary.csv", index=False)
    logger.info("Saved data quality report to %s", out_json)
    return full

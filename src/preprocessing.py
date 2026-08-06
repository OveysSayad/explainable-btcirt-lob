"""Cleaning, deduplication, and optional time-grid alignment."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.data_validation import build_quality_masks

logger = logging.getLogger(__name__)


def drop_exact_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop exact duplicated rows."""
    before = len(df)
    out = df.drop_duplicates()
    removed = before - len(out)
    logger.info("Dropped %s exact duplicate rows", removed)
    return out, removed


def resolve_duplicate_timestamps(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Resolve duplicate timestamps deterministically.

    Rule: keep the last row for each timestamp after chronological sort.
    This prefers the most recently recorded snapshot within a tie group.
    """
    before = len(df)
    out = df.sort_values(["timestamp", "id"], kind="mergesort").drop_duplicates(
        subset=["timestamp"], keep="last"
    )
    removed = before - len(out)
    logger.info(
        "Duplicate timestamp rule=keep_last; removed %s rows; remaining %s",
        removed,
        len(out),
    )
    return out, removed


def clean_invalid_books(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove rows with invalid LOB structure."""
    flags = build_quality_masks(df)
    kept = df.loc[flags["is_valid_book"]].copy()
    stats = {
        "rows_before": int(len(df)),
        "rows_after": int(len(kept)),
        "rows_removed": int((~flags["is_valid_book"]).sum()),
        "removed_invalid_timestamp": int((~flags["valid_ts"]).sum()),
        "removed_incomplete": int((~flags["complete"]).sum()),
        "removed_infinite": int((~flags["finite"]).sum()),
        "removed_bad_prices": int((~flags["prices_ok"]).sum()),
        "removed_bad_qtys": int((~flags["qtys_ok"]).sum()),
        "removed_ask_order": int((~flags["ask_sorted_ok"]).sum()),
        "removed_bid_order": int((~flags["bid_sorted_ok"]).sum()),
        "removed_crossed": int((~flags["not_crossed"]).sum()),
    }
    logger.info("Cleaned invalid books: %s", stats)
    return kept, stats


def sort_chronologically(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by timestamp ascending."""
    return df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)


def align_to_time_grid(
    df: pd.DataFrame,
    interval_seconds: int,
    maximum_snapshot_age_seconds: float,
) -> pd.DataFrame:
    """
    Align snapshots to a regular time grid using as-of join.

    Uses the most recent valid snapshot at or before each grid timestamp,
    only when the snapshot is no older than maximum_snapshot_age_seconds.
    Does not forward-fill across long gaps.
    """
    if df.empty:
        return df

    df = sort_chronologically(df)
    start = df["timestamp"].iloc[0].ceil(f"{interval_seconds}s")
    end = df["timestamp"].iloc[-1].floor(f"{interval_seconds}s")
    if start > end:
        logger.warning("Time grid empty given data range; returning unsorted native data")
        return df

    grid = pd.DataFrame(
        {"grid_timestamp": pd.date_range(start=start, end=end, freq=f"{interval_seconds}s")}
    )
    left = grid.sort_values("grid_timestamp")
    right = df.sort_values("timestamp")
    merged = pd.merge_asof(
        left,
        right,
        left_on="grid_timestamp",
        right_on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta(seconds=maximum_snapshot_age_seconds),
    )
    merged = merged.dropna(subset=["timestamp"]).copy()
    merged["snapshot_age_seconds"] = (
        merged["grid_timestamp"] - merged["timestamp"]
    ).dt.total_seconds()
    merged["timestamp"] = merged["grid_timestamp"]
    merged = merged.drop(columns=["grid_timestamp"])
    logger.info(
        "Aligned to %ss grid: %s grid rows with valid snapshots (max age %ss)",
        interval_seconds,
        len(merged),
        maximum_snapshot_age_seconds,
    )
    return merged.reset_index(drop=True)


def preprocess(
    df: pd.DataFrame,
    config: dict[str, Any],
    gap_stats: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Full preprocessing pipeline for BTCIRT LOB data."""
    meta: dict[str, Any] = {"steps": []}
    out, n_exact = drop_exact_duplicates(df)
    meta["steps"].append({"drop_exact_duplicates": n_exact})

    out = sort_chronologically(out)
    out, n_dup_ts = resolve_duplicate_timestamps(out)
    meta["steps"].append(
        {
            "resolve_duplicate_timestamps": n_dup_ts,
            "rule": "keep_last",
        }
    )

    out, clean_stats = clean_invalid_books(out)
    meta["steps"].append({"clean_invalid_books": clean_stats})
    out = sort_chronologically(out)

    align = bool(config["data"].get("align_to_grid", False))
    justified = True
    if gap_stats is not None:
        justified = bool(gap_stats.get("five_second_grid_justified", False))

    if align and justified:
        out = align_to_time_grid(
            out,
            interval_seconds=int(config["data"]["base_interval_seconds"]),
            maximum_snapshot_age_seconds=float(
                config["data"]["maximum_snapshot_age_seconds"]
            ),
        )
        meta["grid_alignment"] = "applied"
    else:
        reason = (
            gap_stats.get("grid_alignment_decision")
            if gap_stats
            else "align_to_grid=false in config"
        )
        meta["grid_alignment"] = f"skipped: {reason}"
        logger.info("Skipping grid alignment: %s", meta["grid_alignment"])

    # Basic mid/spread for interim convenience
    out["best_ask"] = out["asks_price_1"]
    out["best_bid"] = out["bids_price_1"]
    out["mid_price"] = (out["best_ask"] + out["best_bid"]) / 2.0
    out["spread"] = out["best_ask"] - out["best_bid"]
    out["relative_spread_bps"] = 10_000.0 * out["spread"] / out["mid_price"]

    meta["rows_final"] = int(len(out))
    meta["min_timestamp"] = str(out["timestamp"].min()) if len(out) else None
    meta["max_timestamp"] = str(out["timestamp"].max()) if len(out) else None
    return out, meta

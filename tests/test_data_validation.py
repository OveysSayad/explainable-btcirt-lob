"""Unit tests for order-book validation and duplicate handling."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_validation import audit_order_book_quality, build_quality_masks
from src.preprocessing import resolve_duplicate_timestamps


def _book_row(**overrides):
    row = {
        "id": 1,
        "timestamp": pd.Timestamp("2026-01-01", tz="UTC"),
        "exchange": "nobitex",
        "symbol": "BTCIRT",
    }
    for i in range(1, 9):
        row[f"asks_price_{i}"] = 100.0 + i
        row[f"asks_qty_{i}"] = 1.0
        row[f"bids_price_{i}"] = 100.0 - i
        row[f"bids_qty_{i}"] = 1.0
    row["last_trade_price"] = 100.0
    row["last_trade_qty"] = 0.1
    row.update(overrides)
    return row


def test_ask_prices_sorted_correctly():
    df = pd.DataFrame([_book_row()])
    flags = build_quality_masks(df)
    assert bool(flags.loc[0, "ask_sorted_ok"])


def test_ask_prices_unsorted_detected():
    df = pd.DataFrame([_book_row(asks_price_2=105.0, asks_price_3=104.0)])
    report = audit_order_book_quality(df)
    assert report["asks_not_nondecreasing"] == 1


def test_bid_prices_sorted_correctly():
    df = pd.DataFrame([_book_row()])
    flags = build_quality_masks(df)
    assert bool(flags.loc[0, "bid_sorted_ok"])


def test_bid_prices_unsorted_detected():
    df = pd.DataFrame([_book_row(bids_price_2=97.0, bids_price_3=98.0)])
    report = audit_order_book_quality(df)
    assert report["bids_not_nonincreasing"] == 1


def test_positive_prices():
    df = pd.DataFrame([_book_row(asks_price_1=0.0)])
    flags = build_quality_masks(df)
    assert not bool(flags.loc[0, "prices_ok"])


def test_non_negative_quantities():
    df = pd.DataFrame([_book_row(bids_qty_1=-1.0)])
    flags = build_quality_masks(df)
    assert not bool(flags.loc[0, "qtys_ok"])


def test_crossed_book_detection():
    df = pd.DataFrame([_book_row(asks_price_1=99.0, bids_price_1=100.0)])
    report = audit_order_book_quality(df)
    assert report["crossed_books"] == 1


def test_duplicate_timestamp_keep_last():
    rows = [
        _book_row(id=1, asks_qty_1=1.0),
        _book_row(id=2, asks_qty_1=9.0),
    ]
    df = pd.DataFrame(rows)
    out, removed = resolve_duplicate_timestamps(df)
    assert removed == 1
    assert len(out) == 1
    assert float(out.iloc[0]["asks_qty_1"]) == 9.0

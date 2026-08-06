"""Financial sanity check and model comparison helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def long_only_sanity(
    mid: np.ndarray,
    future_return_bps: np.ndarray,
    y_pred: np.ndarray,
    up_code: int = 2,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> dict[str, Any]:
    """
    Exploratory long-only diagnostic (NOT a trading claim).

    UP -> capture next return minus costs; else flat.
    Sparse snapshots do not support realistic HFT execution.
    """
    signal = (y_pred == up_code).astype(float)
    cost = fee_bps + slippage_bps
    gross = signal * future_return_bps
    net = np.where(signal > 0, gross - cost, 0.0)
    valid = np.isfinite(net)
    if valid.sum() == 0:
        return {"enabled": True, "error": "no valid returns"}
    return {
        "enabled": True,
        "disclaimer": (
            "Exploratory only. Sparse snapshots do not support a realistic "
            "high-frequency execution backtest. Prediction ≠ profitability."
        ),
        "n_signals": int(signal[valid].sum()),
        "mean_gross_bps": float(np.nanmean(gross[valid])),
        "mean_net_bps": float(np.nanmean(net[valid])),
        "sum_net_bps": float(np.nansum(net[valid])),
        "hit_rate_when_long": float(
            np.nanmean((future_return_bps[valid][signal[valid] == 1] > 0).astype(float))
        )
        if (signal[valid] == 1).any()
        else float("nan"),
        "fee_bps": fee_bps,
        "slippage_bps": slippage_bps,
    }

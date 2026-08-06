"""Study B: next mid-price change direction (binary)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.labels.next_observation import CLASS_DOWN, CLASS_UP, log_return_bps

logger = logging.getLogger(__name__)


def build_study_b_labels(
    df: pd.DataFrame,
    one_sample_per_price_run: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Study B: first future observation where mid-price differs from current.

    Classes
    -------
    DOWN if next_changed_mid < current_mid
    UP if next_changed_mid > current_mid

    Sampling
    --------
    If ``one_sample_per_price_run`` is True, keep only the last observation
    of each unchanged-price run (representative per flat spell). This is the
    primary specification to avoid uncontrolled duplication of the same
    future change event.
    """
    out = df.sort_values("timestamp").reset_index(drop=True).copy()
    if "mid_price" not in out.columns:
        out["mid_price"] = (out["asks_price_1"] + out["bids_price_1"]) / 2.0

    mids = out["mid_price"].to_numpy(dtype=float)
    ts = out["timestamp"].to_numpy()
    n = len(out)
    next_idx = np.full(n, -1, dtype=int)
    n_unchanged = np.zeros(n, dtype=int)

    # Forward scan for first changed mid
    j = 1
    for i in range(n):
        if j <= i:
            j = i + 1
        while j < n and np.isclose(mids[j], mids[i]):
            j += 1
        if j < n:
            next_idx[i] = j
            n_unchanged[i] = j - i - 1
        else:
            next_idx[i] = -1
            n_unchanged[i] = n - i - 1

    valid = next_idx >= 0
    out["next_change_index"] = next_idx
    out["number_of_unchanged_snapshots_before_change"] = n_unchanged
    # Initialize with UTC-aware NaT so later assignment accepts tz-aware timestamps
    out["next_change_timestamp"] = pd.Series(
        pd.NaT, index=out.index, dtype="datetime64[ns, UTC]"
    )
    out["next_changed_mid_price"] = np.nan
    if valid.any():
        out.loc[valid, "next_change_timestamp"] = out.loc[
            next_idx[valid], "timestamp"
        ].to_numpy()
        out.loc[valid, "next_changed_mid_price"] = mids[next_idx[valid]]
    out["time_to_next_price_change_seconds"] = (
        out["next_change_timestamp"] - out["timestamp"]
    ).dt.total_seconds()
    out["current_mid_price"] = out["mid_price"]
    out["next_change_return_bps"] = log_return_bps(
        out["current_mid_price"], out["next_changed_mid_price"]
    )
    direction = np.full(n, np.nan)
    direction[valid & (mids[next_idx] < mids)] = CLASS_DOWN
    direction[valid & (mids[next_idx] > mids)] = CLASS_UP
    out["label"] = pd.Series(direction, dtype="Float64")
    out["next_change_direction"] = out["label"].map({0: "DOWN", 2: "UP"})
    out["target_timestamp"] = out["next_change_timestamp"]
    out["actual_delay_seconds"] = out["time_to_next_price_change_seconds"]

    # Price-run representative sample
    price_changed = ~np.isclose(mids, np.roll(mids, 1))
    price_changed[0] = True
    run_id = np.cumsum(price_changed)
    out["price_run_id"] = run_id
    # last index of each run among valid rows
    if one_sample_per_price_run:
        keep = np.zeros(n, dtype=bool)
        for rid, idx in out.loc[valid].groupby("price_run_id").groups.items():
            # keep last observation in the run
            keep[idx[-1]] = True
        out["study_b_primary_sample"] = keep
    else:
        out["study_b_primary_sample"] = valid

    # Unique target-event conservative sample: one current row per unique target ts
    conservative = np.zeros(n, dtype=bool)
    if valid.any():
        tmp = out.loc[valid, ["target_timestamp"]].copy()
        tmp["orig"] = np.where(valid)[0]
        # keep earliest current observation per target
        chosen = tmp.groupby("target_timestamp", sort=False)["orig"].first()
        conservative[chosen.to_numpy()] = True
    out["study_b_unique_target_sample"] = conservative

    meta = {
        "study": "B_next_price_change",
        "one_sample_per_price_run": one_sample_per_price_run,
        "n_all_eligible": int(valid.sum()),
        "n_primary_sample": int(out["study_b_primary_sample"].sum()),
        "n_unique_target_sample": int(conservative.sum()),
        "class_counts_primary": out.loc[out["study_b_primary_sample"], "label"]
        .value_counts()
        .sort_index()
        .to_dict(),
        "median_time_to_change": float(
            out.loc[out["study_b_primary_sample"], "time_to_next_price_change_seconds"].median()
        ),
    }
    logger.info("Study B labels: %s", meta)
    return out, meta

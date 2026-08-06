"""Observation-gap audit for sparse LOB snapshots."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

GAP_THRESHOLDS = [10, 30, 60, 120, 300, 600]


def compute_gaps(timestamps: pd.Series) -> pd.Series:
    """
    Compute consecutive observation gaps in seconds.

    Parameters
    ----------
    timestamps :
        Chronologically ordered timestamps.

    Returns
    -------
    pd.Series
        Gap in seconds for each observation relative to the previous one
        (first row is NaN).
    """
    ts = pd.to_datetime(timestamps, utc=True)
    return ts.diff().dt.total_seconds()


def summarize_gaps(gaps: pd.Series, label: str = "overall") -> dict[str, Any]:
    """Summarize gap distribution including high-percentile tails."""
    g = gaps.dropna()
    if g.empty:
        return {"segment": label, "n_gaps": 0}
    out: dict[str, Any] = {
        "segment": label,
        "n_gaps": int(len(g)),
        "mean": float(g.mean()),
        "median": float(g.median()),
        "std": float(g.std()),
        "min": float(g.min()),
        "max": float(g.max()),
        "p01": float(g.quantile(0.01)),
        "p05": float(g.quantile(0.05)),
        "p25": float(g.quantile(0.25)),
        "p75": float(g.quantile(0.75)),
        "p90": float(g.quantile(0.90)),
        "p95": float(g.quantile(0.95)),
        "p99": float(g.quantile(0.99)),
    }
    for thr in GAP_THRESHOLDS:
        out[f"pct_above_{thr}s"] = float(100.0 * (g > thr).mean())
    return out


def audit_observation_gaps(
    df: pd.DataFrame,
    tables_dir: Path | None = None,
    split_masks: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """
    Full observation-gap audit overall, by date, by hour, and by split.

    Assumptions
    -----------
    Rows are unique timestamps after deduplication. Gaps reflect the
    collection process rather than exchange event spacing.
    """
    out = df.sort_values("timestamp").copy()
    out["observation_gap_seconds"] = compute_gaps(out["timestamp"])
    gaps = out["observation_gap_seconds"]

    overall = summarize_gaps(gaps, "overall")
    by_date_rows = []
    dates = out["timestamp"].dt.tz_convert("UTC").dt.date.astype(str)
    for d, idx in out.groupby(dates).groups.items():
        by_date_rows.append(summarize_gaps(out.loc[idx, "observation_gap_seconds"], str(d)))
    by_date = pd.DataFrame(by_date_rows)

    by_hour_rows = []
    hours = out["timestamp"].dt.hour
    for h, idx in out.groupby(hours).groups.items():
        row = summarize_gaps(out.loc[idx, "observation_gap_seconds"], f"hour_{int(h):02d}")
        row["hour"] = int(h)
        by_hour_rows.append(row)
    by_hour = pd.DataFrame(by_hour_rows).sort_values("hour") if by_hour_rows else pd.DataFrame()

    by_split = []
    if split_masks:
        for name, mask in split_masks.items():
            by_split.append(summarize_gaps(out.loc[mask, "observation_gap_seconds"], name))

    if tables_dir is not None:
        tables_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([overall] + by_split).to_csv(
            tables_dir / "observation_gap_summary.csv", index=False
        )
        by_date.to_csv(tables_dir / "observation_gap_by_date.csv", index=False)
        by_hour.to_csv(tables_dir / "observation_gap_by_hour.csv", index=False)

    logger.info(
        "Gap audit: median=%.2fs mean=%.2fs p95=%.2fs max=%.2fs",
        overall.get("median", np.nan),
        overall.get("mean", np.nan),
        overall.get("p95", np.nan),
        overall.get("max", np.nan),
    )
    return {
        "overall": overall,
        "by_date": by_date,
        "by_hour": by_hour,
        "by_split": by_split,
        "frame_with_gaps": out,
    }

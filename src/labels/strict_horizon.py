"""Study C: strict fixed-horizon labels with narrow delay windows."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.labels.next_observation import assign_classes, fit_epsilon_candidates, log_return_bps, select_epsilon, estimate_tick_size

logger = logging.getLogger(__name__)


def _match_strict_horizon(
    timestamps: np.ndarray,
    mids: np.ndarray,
    horizon: float,
    lower: float,
    upper: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    For each t, find future s minimizing |delay - h| subject to lower <= delay <= upper.

    Returns target index (-1 if none), actual delay, horizon error, target mid.
    """
    n = len(timestamps)
    # timestamps as int64 ns
    ts = pd.to_datetime(timestamps, utc=True)
    ts_ns = np.asarray([int(t.value) for t in ts], dtype=np.int64)
    h_ns = int(horizon * 1e9)
    lo_ns = int(lower * 1e9)
    hi_ns = int(upper * 1e9)

    target_idx = np.full(n, -1, dtype=int)
    actual_delay = np.full(n, np.nan)
    horizon_error = np.full(n, np.nan)
    target_mid = np.full(n, np.nan)

    for i in range(n):
        left = ts_ns[i] + lo_ns
        right = ts_ns[i] + hi_ns
        # searchsorted on sorted timestamps (assumed sorted)
        a = np.searchsorted(ts_ns, left, side="left")
        b = np.searchsorted(ts_ns, right, side="right")
        a = max(a, i + 1)  # strictly future
        if b <= a:
            continue
        window = ts_ns[a:b]
        center = ts_ns[i] + h_ns
        local = int(np.argmin(np.abs(window - center)))
        j = a + local
        delay = (ts_ns[j] - ts_ns[i]) / 1e9
        target_idx[i] = j
        actual_delay[i] = delay
        horizon_error[i] = abs(delay - horizon)
        target_mid[i] = mids[j]
    return target_idx, actual_delay, horizon_error, target_mid


def build_study_c_labels(
    df: pd.DataFrame,
    config: dict[str, Any],
    train_mask: np.ndarray,
    val_mask: np.ndarray | None = None,
) -> tuple[dict[int, pd.DataFrame], dict[str, Any]]:
    """
    Build strict-horizon labels for configured windows only.

    This is a pilot analysis when eligibility is low under sparse sampling.
    """
    base = df.sort_values("timestamp").reset_index(drop=True).copy()
    if "mid_price" not in base.columns:
        base["mid_price"] = (base["asks_price_1"] + base["bids_price_1"]) / 2.0
    if "relative_spread_bps" not in base.columns:
        spread = base["asks_price_1"] - base["bids_price_1"]
        base["relative_spread_bps"] = 10_000.0 * spread / base["mid_price"]

    horizons_cfg = config.get("study_c", {}).get("horizons", {})
    min_total = int(config.get("study_c", {}).get("minimum_total_samples", 1000))
    min_per_class = int(config.get("study_c", {}).get("minimum_samples_per_class", 100))
    min_dates = int(config.get("study_c", {}).get("minimum_unique_dates", 10))

    tick_info = estimate_tick_size(base.loc[train_mask])
    results: dict[int, pd.DataFrame] = {}
    meta: dict[str, Any] = {"study": "C_strict_horizons", "horizons": {}, "tick": tick_info}

    for h_str, bounds in horizons_cfg.items():
        h = int(h_str)
        lower = float(bounds["lower_seconds"])
        upper = float(bounds["upper_seconds"])
        out = base.copy()
        tidx, delay, err, tmid = _match_strict_horizon(
            out["timestamp"].to_numpy(),
            out["mid_price"].to_numpy(dtype=float),
            horizon=h,
            lower=lower,
            upper=upper,
        )
        out["requested_horizon_seconds"] = h
        out["target_index"] = tidx
        out["actual_delay_seconds"] = delay
        out["horizon_error_seconds"] = err
        out["target_mid_price"] = tmid
        out["target_timestamp"] = pd.Series(
            pd.NaT, index=out.index, dtype="datetime64[ns, UTC]"
        )
        valid = tidx >= 0
        if valid.any():
            out.loc[valid, "target_timestamp"] = out.loc[
                tidx[valid], "timestamp"
            ].to_numpy()
        out["current_mid_price"] = out["mid_price"]
        out["future_return_bps"] = log_return_bps(out["current_mid_price"], out["target_mid_price"])

        # Epsilon from training eligible returns only
        train_eligible = train_mask & valid
        if train_eligible.sum() < 10:
            eps = float("nan")
            method = "insufficient_train"
            cand = pd.DataFrame()
            out["label"] = np.nan
        else:
            candidates = fit_epsilon_candidates(
                out.loc[train_eligible, "future_return_bps"],
                out.loc[train_eligible, "relative_spread_bps"],
                tick_bps_median=float(tick_info["estimated_tick_size_bps_median"]),
                config=config,
            )
            val_eligible = (
                (val_mask & valid) if val_mask is not None else None
            )
            val_ret = (
                out.loc[val_eligible, "future_return_bps"] if val_eligible is not None else None
            )
            eps, method, cand = select_epsilon(
                candidates,
                out.loc[train_eligible, "future_return_bps"],
                val_ret,
                primary_method=config.get("study_a", {}).get("primary_epsilon_method", "hybrid"),
            )
            out["label"] = assign_classes(out["future_return_bps"], eps)
            out["epsilon_bps"] = eps
            out["epsilon_method"] = method

        eligible = out.loc[valid].copy()
        n_eligible = int(valid.sum())
        n_rejected = int((~valid).sum())
        dates = (
            eligible["timestamp"].dt.tz_convert("UTC").dt.date.nunique()
            if n_eligible
            else 0
        )
        class_counts = (
            eligible["label"].value_counts().to_dict() if "label" in eligible else {}
        )
        underpowered = (
            n_eligible < min_total
            or dates < min_dates
            or any(int(v) < min_per_class for v in class_counts.values())
            if class_counts
            else True
        )
        dup_targets = (
            float(100.0 * eligible["target_timestamp"].duplicated().mean())
            if n_eligible
            else np.nan
        )
        hmeta = {
            "horizon": h,
            "lower": lower,
            "upper": upper,
            "n_eligible": n_eligible,
            "n_rejected": n_rejected,
            "eligibility_pct": float(100.0 * n_eligible / len(out)),
            "unique_dates": int(dates),
            "pct_duplicated_target_timestamps": dup_targets,
            "delay_median": float(eligible["actual_delay_seconds"].median()) if n_eligible else None,
            "delay_p95": float(eligible["actual_delay_seconds"].quantile(0.95)) if n_eligible else None,
            "error_median": float(eligible["horizon_error_seconds"].median()) if n_eligible else None,
            "epsilon_bps": eps if np.isfinite(eps) else None,
            "epsilon_method": method,
            "class_counts": class_counts,
            "underpowered": underpowered,
            "label": "Pilot fixed-horizon analysis" if underpowered else "Strict fixed-horizon",
        }
        meta["horizons"][str(h)] = hmeta
        results[h] = out
        logger.info("Study C %ss: %s", h, hmeta)

    return results, meta


def cross_horizon_overlap(
    labeled_by_horizon: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """Measure target-timestamp and label overlap across strict horizons."""
    horizons = sorted(labeled_by_horizon.keys())
    rows = []
    for i, h1 in enumerate(horizons):
        for h2 in horizons[i + 1 :]:
            a = labeled_by_horizon[h1]
            b = labeled_by_horizon[h2]
            va = a["target_index"] >= 0
            vb = b["target_index"] >= 0
            common = va & vb
            n_common = int(common.sum())
            if n_common == 0:
                rows.append(
                    {
                        "horizon_a": h1,
                        "horizon_b": h2,
                        "n_common_current_observations": 0,
                        "pct_same_target_timestamp": np.nan,
                        "pct_same_label": np.nan,
                        "corr_future_return": np.nan,
                    }
                )
                continue
            same_target = (
                a.loc[common, "target_timestamp"].to_numpy()
                == b.loc[common, "target_timestamp"].to_numpy()
            )
            same_label = np.full(n_common, np.nan)
            if "label" in a.columns and "label" in b.columns:
                same_label = (
                    a.loc[common, "label"].to_numpy() == b.loc[common, "label"].to_numpy()
                )
                pct_same_label = float(100.0 * np.nanmean(same_label.astype(float)))
            else:
                pct_same_label = np.nan
            r1 = a.loc[common, "future_return_bps"]
            r2 = b.loc[common, "future_return_bps"]
            corr = float(pd.Series(r1).corr(pd.Series(r2)))
            rows.append(
                {
                    "horizon_a": h1,
                    "horizon_b": h2,
                    "n_common_current_observations": n_common,
                    "pct_same_target_timestamp": float(100.0 * np.mean(same_target)),
                    "pct_same_label": pct_same_label,
                    "corr_future_return": corr,
                }
            )
    return pd.DataFrame(rows)

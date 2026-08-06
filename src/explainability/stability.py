"""SHAP rank stability across folds/regimes."""

from __future__ import annotations

import pandas as pd


def shap_stability_table(rank_frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Aggregate feature ranks across folds."""
    if not rank_frames:
        return pd.DataFrame()
    all_ranks = pd.concat(rank_frames, ignore_index=True)
    return (
        all_ranks.groupby("feature")["rank"]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
        .sort_values("mean")
    )

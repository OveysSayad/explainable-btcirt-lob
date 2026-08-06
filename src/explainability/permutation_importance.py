"""Permutation importance within dates."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)


def permutation_importance_table(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_cols: list[str],
    dates: np.ndarray,
    n_repeats: int = 5,
    seed: int = 42,
    binary: bool = False,
) -> pd.DataFrame:
    """
    Permute each feature within calendar dates and measure Macro-F1 drop.

    Within-date permutation avoids fully destroying temporal structure.
    """
    rng = np.random.default_rng(seed)
    labels = [0, 1] if binary else [0, 1, 2]
    base_pred = model.predict(X)
    base = f1_score(y, base_pred, average="macro", zero_division=0, labels=labels)
    rows = []
    unique_dates = np.unique(dates)
    for j, name in enumerate(feature_cols):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            for d in unique_dates:
                m = dates == d
                idx = np.where(m)[0]
                if len(idx) < 2:
                    continue
                Xp[idx, j] = Xp[rng.permutation(idx), j]
            pred = model.predict(Xp)
            score = f1_score(y, pred, average="macro", zero_division=0, labels=labels)
            drops.append(base - score)
        rows.append(
            {
                "feature": name,
                "mean_f1_decrease": float(np.mean(drops)),
                "std_f1_decrease": float(np.std(drops)),
                "base_macro_f1": float(base),
            }
        )
    out = pd.DataFrame(rows).sort_values("mean_f1_decrease", ascending=False)
    out["rank"] = np.arange(1, len(out) + 1)
    return out

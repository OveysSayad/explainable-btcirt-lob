"""Model comparison table helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd


def metrics_to_frame(metrics_by_model: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Flatten nested metrics into a comparison table."""
    rows = []
    for name, metrics in metrics_by_model.items():
        row = {"model": name}
        for k, v in metrics.items():
            if isinstance(v, (list, dict)):
                continue
            row[k] = v
        rows.append(row)
    return pd.DataFrame(rows)

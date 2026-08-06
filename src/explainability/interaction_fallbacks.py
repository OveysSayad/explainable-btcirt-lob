"""SHAP interaction attempt with documented fallbacks."""

from __future__ import annotations

import json
import logging
import traceback
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def run_interaction_analysis(
    model: Any,
    X: pd.DataFrame,
    config: dict[str, Any],
    paths: Any,
    max_n: int = 2000,
) -> dict[str, Any]:
    """
    Attempt native SHAP interactions; on failure use dependence fallbacks.

    Never fabricates interaction values.
    """
    import shap
    import xgboost
    import sklearn

    status: dict[str, Any] = {
        "attempted_method": "shap.TreeExplainer.shap_interaction_values",
        "success": False,
        "package_versions": {
            "shap": getattr(shap, "__version__", "unknown"),
            "xgboost": xgboost.__version__,
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
        },
        "error_message": None,
        "fallback_method": None,
        "generated_outputs": [],
    }
    sample = X.iloc[: min(max_n, len(X))]
    fig_dir = paths.figures / "shap"
    fig_dir.mkdir(parents=True, exist_ok=True)
    try:
        explainer = shap.TreeExplainer(model)
        inter = explainer.shap_interaction_values(sample)
        # Multiclass returns (n, F, F, C) or list — reduce to 2D mean |interaction|
        if isinstance(inter, list):
            mat = np.mean([np.abs(a).mean(axis=0) for a in inter], axis=0)
        else:
            arr = np.asarray(inter)
            if arr.ndim == 4:
                # (n, F, F, C) → mean over samples and classes
                mat = np.abs(arr).mean(axis=(0, 3))
            elif arr.ndim == 3:
                mat = np.abs(arr).mean(axis=0)
            else:
                raise ValueError(f"Unexpected interaction shape: {arr.shape}")
        if mat.ndim != 2:
            raise ValueError(f"Reduced interaction matrix not 2-d: {mat.shape}")
        pd.DataFrame(mat, index=sample.columns, columns=sample.columns).to_csv(
            paths.tables / "shap_interactions_native.csv"
        )
        status["success"] = True
        status["generated_outputs"].extend(
            ["native_shap_interactions", "reports/tables/shap_interactions_native.csv"]
        )
    except Exception as exc:  # noqa: BLE001
        status["success"] = False
        status["error_message"] = f"{type(exc).__name__}: {exc}"
        status["traceback"] = traceback.format_exc()
        status["fallback_method"] = "dependence_plots_colored_by_second_feature"
        logger.warning("Native SHAP interactions failed: %s", exc)
        pairs = [
            ("obi_5", "relative_spread_bps"),
            ("weighted_obi", "log_bid_depth_5"),
            ("snapshot_ofi_proxy_l1", "log_bid_depth_1"),
            ("microprice_edge_bps", "volatility_300s"),
            ("relative_spread_bps", "volatility_300s"),
            ("hour_cos", "observation_gap_seconds"),
        ]
        try:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(sample)
            for a, b in pairs:
                if a not in sample.columns or b not in sample.columns:
                    continue
                plt.figure(figsize=(7, 5))
                vals = sv[2] if isinstance(sv, list) else (sv[:, :, -1] if getattr(sv, "ndim", 2) == 3 else sv)
                if isinstance(vals, np.ndarray) and vals.ndim == 2:
                    shap.dependence_plot(a, vals, sample, interaction_index=b, show=False)
                plt.title(f"Dependence fallback: {a} colored by {b}")
                out = fig_dir / f"interaction_fallback_{a}_x_{b}.png"
                plt.tight_layout()
                plt.savefig(out, dpi=200, bbox_inches="tight")
                plt.close()
                status["generated_outputs"].append(str(out))
        except Exception as exc2:  # noqa: BLE001
            status["fallback_error"] = str(exc2)
            logger.warning("Fallback dependence plots failed: %s", exc2)

    paths.metrics.mkdir(parents=True, exist_ok=True)
    with (paths.metrics / "shap_interaction_status.json").open("w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)
    return status

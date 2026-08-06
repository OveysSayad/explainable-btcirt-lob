"""SHAP analysis with grouped aggregation and stability hooks."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.feature_engineering import FEATURE_FAMILIES
from src.models.logistic_regression import prepare_xy

logger = logging.getLogger(__name__)


def run_shap_analysis(
    model: Any,
    df: pd.DataFrame,
    feature_cols: list[str],
    test_mask: np.ndarray,
    config: dict[str, Any],
    paths: Any,
    label_col: str = "label",
    binary: bool = False,
    prefix: str = "study_a",
) -> dict[str, Any]:
    """TreeSHAP global/local analysis on a temporally stratified sample."""
    cfg = config.get("shap", {})
    max_n = int(cfg.get("maximum_global_sample", 10000))
    seed = int(config["project"]["random_seed"])

    X, y, idx = prepare_xy(df, feature_cols, test_mask, label_col)
    if len(X) == 0:
        return {"error": "no test rows for SHAP"}
    rng = np.random.default_rng(seed)
    n = min(max_n, len(X))
    # time-block stratification via index order
    order = np.arange(len(X))
    blocks = np.array_split(order, min(10, len(order)))
    chosen = []
    per = max(1, n // len(blocks))
    for b in blocks:
        take = min(per, len(b))
        chosen.extend(rng.choice(b, size=take, replace=False).tolist())
    chosen = list(dict.fromkeys(chosen))[:n]
    Xs = pd.DataFrame(X[chosen], columns=feature_cols)
    sample_idx = idx.to_numpy()[chosen]

    explainer = shap.TreeExplainer(model)
    explanation = explainer(Xs)
    values = explanation.values
    if values.ndim == 3:
        mean_abs = np.abs(values).mean(axis=0)
        overall = mean_abs.mean(axis=1)
        rows = []
        for i, f in enumerate(feature_cols):
            row = {"feature": f, "mean_abs_shap": float(overall[i])}
            if mean_abs.shape[1] >= 3:
                row.update(
                    {
                        "mean_abs_shap_DOWN": float(mean_abs[i, 0]),
                        "mean_abs_shap_STABLE": float(mean_abs[i, 1]),
                        "mean_abs_shap_UP": float(mean_abs[i, 2]),
                    }
                )
            elif mean_abs.shape[1] == 2:
                row.update(
                    {
                        "mean_abs_shap_DOWN": float(mean_abs[i, 0]),
                        "mean_abs_shap_UP": float(mean_abs[i, 1]),
                    }
                )
            rows.append(row)
    else:
        overall = np.abs(values).mean(axis=0)
        rows = [
            {"feature": f, "mean_abs_shap": float(overall[i])}
            for i, f in enumerate(feature_cols)
        ]
    imp = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)

    feat_to_fam = {f: fam for fam, fs in FEATURE_FAMILIES.items() for f in fs}
    imp["family"] = imp["feature"].map(lambda x: feat_to_fam.get(x, "Other"))
    grouped = (
        imp.groupby("family", as_index=False)["mean_abs_shap"]
        .sum()
        .sort_values("mean_abs_shap", ascending=False)
    )
    total = grouped["mean_abs_shap"].sum()
    grouped["normalized"] = grouped["mean_abs_shap"] / total if total else 0.0

    paths.tables.mkdir(parents=True, exist_ok=True)
    fig_dir = paths.figures / "shap"
    fig_dir.mkdir(parents=True, exist_ok=True)
    imp.to_csv(paths.tables / f"{prefix}_shap_global_importance.csv", index=False)
    grouped.to_csv(paths.tables / f"{prefix}_shap_grouped_importance.csv", index=False)
    # also canonical names for Study A primary
    if prefix == "study_a":
        imp.to_csv(paths.tables / "shap_global_importance.csv", index=False)
        grouped.to_csv(paths.tables / "shap_grouped_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    top = imp.head(20)
    ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1], color="#1f4e79")
    ax.set_xlabel("Mean |SHAP|")
    ax.set_title(f"Global SHAP Importance ({prefix})")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{prefix}_shap_global.png", dpi=config.get("reporting", {}).get("figure_dpi", 200))
    plt.close(fig)

    return {
        "n_shap_samples": len(Xs),
        "top_features": imp.head(15).to_dict(orient="records"),
        "grouped": grouped.to_dict(orient="records"),
        "sample_index": sample_idx.tolist(),
        "disclaimer": (
            "SHAP explains how the model uses features. "
            "SHAP does not prove causal relationships."
        ),
    }

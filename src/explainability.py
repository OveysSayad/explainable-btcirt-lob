"""SHAP TreeExplainer utilities for XGBoost."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap

from src.feature_engineering import FEATURE_FAMILIES

logger = logging.getLogger(__name__)


def select_shap_sample(
    df: pd.DataFrame,
    test_idx: np.ndarray,
    feature_cols: list[str],
    sample_size: int,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Select a representative test subsample for SHAP.

    Covers the full test time range, includes all classes, and avoids
    taking only adjacent observations via stratified time-block sampling.
    """
    sub = df.iloc[test_idx].copy()
    # Impute within the test subsample using its own medians only for SHAP matrix
    # construction after pipeline-level train-median imputation; still drop rows
    # missing labels.
    sub = sub.dropna(subset=["label"])
    if len(sub) == 0:
        raise ValueError("No valid test rows for SHAP sample")
    X_all = sub[feature_cols].replace([np.inf, -np.inf], np.nan)
    X_all = X_all.fillna(X_all.median(numeric_only=True)).fillna(0.0)
    sub = sub.copy()
    sub[feature_cols] = X_all

    n = min(sample_size, len(sub))
    rng = np.random.default_rng(seed)

    # Time-block stratification: split test into 10 chronological blocks
    sub = sub.sort_values("timestamp")
    blocks = np.array_split(np.arange(len(sub)), min(10, len(sub)))
    per_block = max(1, n // len(blocks))
    chosen: list[int] = []
    for block in blocks:
        block_df = sub.iloc[block]
        # Ensure class coverage when possible
        take = min(per_block, len(block_df))
        # Stratify by label within block
        parts = []
        for cls in [0, 1, 2]:
            cls_idx = block_df.index[block_df["label"] == cls].to_numpy()
            if len(cls_idx) == 0:
                continue
            k = max(1, take // 3)
            k = min(k, len(cls_idx))
            parts.extend(rng.choice(cls_idx, size=k, replace=False).tolist())
        chosen.extend(parts)

    chosen = list(dict.fromkeys(chosen))  # unique preserve order
    if len(chosen) < n:
        remaining = sub.index.difference(chosen).to_numpy()
        need = min(n - len(chosen), len(remaining))
        if need > 0:
            chosen.extend(rng.choice(remaining, size=need, replace=False).tolist())
    chosen = chosen[:n]
    sample = sub.loc[chosen]
    X = sample[feature_cols].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    return X, sample.index.to_numpy()


def compute_shap_values(
    model: Any,
    X: pd.DataFrame,
) -> shap.Explanation:
    """Compute TreeSHAP values for an XGBoost model."""
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X)
    return explanation


def global_importance(
    explanation: shap.Explanation,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Mean |SHAP| overall and per class."""
    values = explanation.values
    # multiclass: (n, n_features, n_classes) or (n, n_features)
    rows = []
    if values.ndim == 3:
        mean_abs = np.abs(values).mean(axis=0)  # (features, classes)
        overall = mean_abs.mean(axis=1)
        for i, feat in enumerate(feature_cols):
            rows.append(
                {
                    "feature": feat,
                    "mean_abs_shap": float(overall[i]),
                    "mean_abs_shap_DOWN": float(mean_abs[i, 0]),
                    "mean_abs_shap_STABLE": float(mean_abs[i, 1]),
                    "mean_abs_shap_UP": float(mean_abs[i, 2]),
                }
            )
    else:
        overall = np.abs(values).mean(axis=0)
        for i, feat in enumerate(feature_cols):
            rows.append({"feature": feat, "mean_abs_shap": float(overall[i])})
    out = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)
    return out


def family_importance(importance_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate SHAP importance by feature family."""
    feat_to_family = {}
    for family, feats in FEATURE_FAMILIES.items():
        for f in feats:
            feat_to_family[f] = family
    df = importance_df.copy()
    df["family"] = df["feature"].map(lambda x: feat_to_family.get(x, "Other"))
    agg = (
        df.groupby("family", as_index=False)["mean_abs_shap"]
        .sum()
        .sort_values("mean_abs_shap", ascending=False)
    )
    return agg


def regime_stability(
    df: pd.DataFrame,
    sample_idx: np.ndarray,
    explanation: shap.Explanation,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Compare top SHAP features across regimes."""
    sample = df.loc[sample_idx]
    values = explanation.values
    if values.ndim == 3:
        abs_shap = np.abs(values).mean(axis=2)
    else:
        abs_shap = np.abs(values)

    def top_features(mask: np.ndarray, label: str) -> pd.DataFrame:
        if mask.sum() < 10:
            return pd.DataFrame({"regime": [label], "feature": [None], "mean_abs_shap": [np.nan]})
        mean_abs = abs_shap[mask].mean(axis=0)
        order = np.argsort(-mean_abs)[:10]
        return pd.DataFrame(
            {
                "regime": label,
                "feature": [feature_cols[i] for i in order],
                "mean_abs_shap": mean_abs[order],
                "rank": np.arange(1, len(order) + 1),
            }
        )

    vol = sample["volatility_300s"] if "volatility_300s" in sample.columns else sample.filter(like="volatility_").iloc[:, 0]
    spread = sample["relative_spread_bps"]
    ts = sample["timestamp"]
    mid_ts = ts.min() + (ts.max() - ts.min()) / 2

    frames = [
        top_features((vol <= vol.median()).to_numpy(), "low_volatility"),
        top_features((vol > vol.median()).to_numpy(), "high_volatility"),
        top_features((spread <= spread.median()).to_numpy(), "narrow_spread"),
        top_features((spread > spread.median()).to_numpy(), "wide_spread"),
        top_features((ts <= mid_ts).to_numpy(), "test_first_half"),
        top_features((ts > mid_ts).to_numpy(), "test_second_half"),
    ]
    # By hour buckets
    hour = ts.dt.hour
    for h_label, h_mask in [
        ("hours_0_8", (hour < 8).to_numpy()),
        ("hours_8_16", ((hour >= 8) & (hour < 16)).to_numpy()),
        ("hours_16_24", (hour >= 16).to_numpy()),
    ]:
        frames.append(top_features(h_mask, h_label))

    return pd.concat(frames, ignore_index=True)


def select_local_examples(
    df: pd.DataFrame,
    pred_idx: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, int]:
    """Pick illustrative local prediction indices (positional in pred arrays)."""
    conf = y_proba.max(axis=1)
    correct = y_true == y_pred

    def best(mask: np.ndarray) -> int | None:
        if mask.sum() == 0:
            return None
        local = np.where(mask)[0]
        return int(local[np.argmax(conf[local])])

    examples = {
        "correct_high_conf_UP": best(correct & (y_pred == 2) & (conf > 0.5)),
        "correct_high_conf_DOWN": best(correct & (y_pred == 0) & (conf > 0.5)),
        "correct_high_conf_STABLE": best(correct & (y_pred == 1) & (conf > 0.5)),
        "incorrect_high_conf": best((~correct) & (conf > 0.5)),
        "low_conf_borderline": int(np.argmin(conf)),
    }
    return {k: v for k, v in examples.items() if v is not None}


def run_shap_analysis(
    model: Any,
    df: pd.DataFrame,
    feature_cols: list[str],
    test_idx: np.ndarray,
    predictions: dict[str, Any],
    config: dict[str, Any],
    paths: Any,
) -> dict[str, Any]:
    """End-to-end SHAP analysis for the final XGBoost model."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = config["explainability"]
    sample_size = int(cfg["shap_sample_size"])
    seed = int(config["project"]["random_seed"])

    X, sample_idx = select_shap_sample(df, test_idx, feature_cols, sample_size, seed)
    logger.info("Computing SHAP on %s samples x %s features", len(X), X.shape[1])
    explanation = compute_shap_values(model, X)

    imp = global_importance(explanation, feature_cols)
    fam = family_importance(imp)
    stability = regime_stability(df, sample_idx, explanation, feature_cols)

    paths.tables.mkdir(parents=True, exist_ok=True)
    paths.shap.mkdir(parents=True, exist_ok=True)
    paths.figures.mkdir(parents=True, exist_ok=True)
    imp.to_csv(paths.tables / "shap_global_importance.csv", index=False)
    fam.to_csv(paths.tables / "shap_feature_family_importance.csv", index=False)
    stability.to_csv(paths.tables / "shap_stability_results.csv", index=False)

    # Bar plot
    top = imp.head(20)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1], color="#1f4e79")
    ax.set_xlabel("Mean |SHAP| value")
    ax.set_title("Global SHAP Feature Importance (TreeSHAP)")
    fig.tight_layout()
    fig.savefig(paths.figures / "shap_global_importance.png", dpi=config["output"]["figure_dpi"])
    plt.close(fig)

    # Beeswarm for UP class if multiclass
    try:
        values = explanation.values
        if values.ndim == 3:
            for class_i, name in enumerate(["DOWN", "STABLE", "UP"]):
                plt.figure(figsize=(10, 8))
                shap.summary_plot(
                    values[:, :, class_i],
                    X,
                    show=False,
                    max_display=20,
                )
                plt.title(f"SHAP Beeswarm — class {name}")
                plt.tight_layout()
                plt.savefig(
                    paths.figures / f"shap_beeswarm_{name.lower()}.png",
                    dpi=config["output"]["figure_dpi"],
                    bbox_inches="tight",
                )
                plt.close()
        else:
            plt.figure(figsize=(10, 8))
            shap.summary_plot(values, X, show=False, max_display=20)
            plt.tight_layout()
            plt.savefig(
                paths.figures / "shap_beeswarm.png",
                dpi=config["output"]["figure_dpi"],
                bbox_inches="tight",
            )
            plt.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("SHAP beeswarm failed: %s", exc)

    # Dependence plots
    dep_features = [
        "obi_5",
        "weighted_obi",
        "relative_spread_bps",
        "microprice_edge_bps",
        "normalized_ofi_300s",
        "volatility_300s",
    ]
    for feat in dep_features:
        if feat not in feature_cols:
            continue
        try:
            plt.figure(figsize=(8, 5))
            # Use UP class contribution when available
            if explanation.values.ndim == 3:
                shap.dependence_plot(
                    feat,
                    explanation.values[:, :, 2],
                    X,
                    show=False,
                    interaction_index="auto",
                )
                plt.title(f"SHAP Dependence: {feat} (UP class)")
            else:
                shap.dependence_plot(feat, explanation.values, X, show=False)
                plt.title(f"SHAP Dependence: {feat}")
            plt.tight_layout()
            plt.savefig(
                paths.figures / f"shap_dependence_{feat}.png",
                dpi=config["output"]["figure_dpi"],
                bbox_inches="tight",
            )
            plt.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Dependence plot for %s failed: %s", feat, exc)

    # Interactions on smaller sample
    interaction_notes = []
    if cfg.get("calculate_interactions", True):
        inter_n = min(int(cfg["maximum_interaction_sample_size"]), len(X))
        X_inter = X.iloc[:inter_n]
        try:
            explainer = shap.TreeExplainer(model)
            inter = explainer.shap_interaction_values(X_inter)
            # inter may be list per class
            interaction_notes.append(
                "Computed SHAP interaction values on "
                f"{inter_n} samples. Interactions explain co-dependence in "
                "model attributions, not causal effects."
            )
            # Save mean |interaction| for selected pairs if matrix available
            if isinstance(inter, list):
                inter_mat = np.abs(inter[2]).mean(axis=0)  # UP class
            else:
                inter_mat = np.abs(inter).mean(axis=0)
            pairs = [
                ("obi_5", "relative_spread_bps"),
                ("obi_5", "log_bid_depth_5"),
                ("normalized_ofi_300s", "log_bid_depth_5"),
                ("microprice_edge_bps", "volatility_300s"),
                ("relative_spread_bps", "volatility_300s"),
            ]
            rows = []
            for a, b in pairs:
                if a in feature_cols and b in feature_cols:
                    i, j = feature_cols.index(a), feature_cols.index(b)
                    rows.append(
                        {
                            "feature_a": a,
                            "feature_b": b,
                            "mean_abs_interaction": float(inter_mat[i, j]),
                        }
                    )
            if rows:
                pd.DataFrame(rows).to_csv(
                    paths.tables / "shap_interactions.csv", index=False
                )
        except Exception as exc:  # noqa: BLE001
            interaction_notes.append(f"Interaction calculation failed: {exc}")
            logger.warning("SHAP interactions failed: %s", exc)

    # Local examples
    local = {}
    try:
        y_true = predictions["y_true"]
        y_pred = predictions["y_pred"]
        y_proba = predictions["y_proba"]
        examples = select_local_examples(
            df, predictions["index"], y_true, y_pred, y_proba
        )
        # Map prediction-array positions to SHAP sample when possible
        for name, pos in examples.items():
            orig_index = int(predictions["index"][pos])
            local[name] = {
                "row_index": orig_index,
                "y_true": int(y_true[pos]),
                "y_pred": int(y_pred[pos]),
                "proba": y_proba[pos].tolist(),
                "timestamp": str(df.loc[orig_index, "timestamp"]),
                "interpretation": (
                    f"Local case '{name}': predicted class {int(y_pred[pos])} "
                    f"with probs {np.round(y_proba[pos], 3).tolist()}. "
                    "SHAP attributions explain model use of features, not causality."
                ),
            }
            # Waterfall if in SHAP sample
            if orig_index in sample_idx:
                loc = int(np.where(sample_idx == orig_index)[0][0])
                try:
                    plt.figure(figsize=(10, 6))
                    if explanation.values.ndim == 3:
                        shap.plots.waterfall(explanation[loc, :, int(y_pred[pos])], show=False)
                    else:
                        shap.plots.waterfall(explanation[loc], show=False)
                    plt.tight_layout()
                    plt.savefig(
                        paths.figures / f"shap_waterfall_{name}.png",
                        dpi=config["output"]["figure_dpi"],
                        bbox_inches="tight",
                    )
                    plt.close()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Waterfall %s failed: %s", name, exc)
        pd.DataFrame(local).T.to_csv(paths.tables / "shap_local_examples.csv")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Local SHAP examples failed: %s", exc)

    return {
        "n_shap_samples": len(X),
        "top_features": imp.head(15).to_dict(orient="records"),
        "family_importance": fam.to_dict(orient="records"),
        "local_examples": local,
        "interaction_notes": interaction_notes,
        "disclaimer": (
            "SHAP explains how the model uses features. "
            "SHAP does not prove causal relationships."
        ),
    }

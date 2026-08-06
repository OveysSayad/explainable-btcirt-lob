"""SHAP analysis with global, beeswarm, dependence, waterfall, and grouped plots."""

from __future__ import annotations

import logging
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

DEPENDENCE_FEATURES = [
    "obi_5",
    "weighted_obi",
    "relative_spread_bps",
    "microprice_edge_bps",
    "normalized_snapshot_ofi_300s",
    "volatility_300s",
]


def _figure_dpi(config: dict[str, Any]) -> int:
    return int(
        config.get("reporting", {}).get(
            "figure_dpi",
            config.get("output", {}).get("figure_dpi", 200),
        )
    )


def select_local_examples(
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


def regime_stability(
    df: pd.DataFrame,
    sample_idx: np.ndarray,
    explanation: shap.Explanation,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Compare top SHAP features across regimes within the SHAP sample."""
    sample = df.loc[sample_idx]
    values = explanation.values
    if values.ndim == 3:
        abs_shap = np.abs(values).mean(axis=2)
    else:
        abs_shap = np.abs(values)

    def top_features(mask: np.ndarray, label: str) -> pd.DataFrame:
        if mask.sum() < 10:
            return pd.DataFrame(
                {"regime": [label], "feature": [None], "mean_abs_shap": [np.nan]}
            )
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

    vol_col = (
        "volatility_300s"
        if "volatility_300s" in sample.columns
        else next((c for c in sample.columns if c.startswith("volatility_")), None)
    )
    vol = sample[vol_col] if vol_col else pd.Series(0.0, index=sample.index)
    spread = (
        sample["relative_spread_bps"]
        if "relative_spread_bps" in sample.columns
        else pd.Series(0.0, index=sample.index)
    )
    ts = sample["timestamp"]
    mid_ts = ts.min() + (ts.max() - ts.min()) / 2
    hour = ts.dt.hour

    frames = [
        top_features((vol <= vol.median()).to_numpy(), "low_volatility"),
        top_features((vol > vol.median()).to_numpy(), "high_volatility"),
        top_features((spread <= spread.median()).to_numpy(), "narrow_spread"),
        top_features((spread > spread.median()).to_numpy(), "wide_spread"),
        top_features((ts <= mid_ts).to_numpy(), "test_first_half"),
        top_features((ts > mid_ts).to_numpy(), "test_second_half"),
        top_features((hour < 8).to_numpy(), "hours_0_8"),
        top_features(((hour >= 8) & (hour < 16)).to_numpy(), "hours_8_16"),
        top_features((hour >= 16).to_numpy(), "hours_16_24"),
    ]
    return pd.concat(frames, ignore_index=True)


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
    predictions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """TreeSHAP global/local analysis with full plot suite on development_test."""
    cfg = config.get("shap", {})
    max_n = int(cfg.get("maximum_global_sample", 10000))
    seed = int(config["project"]["random_seed"])
    dpi = _figure_dpi(config)

    X, y, idx = prepare_xy(df, feature_cols, test_mask, label_col)
    if len(X) == 0:
        return {"error": "no test rows for SHAP"}

    rng = np.random.default_rng(seed)
    n = min(max_n, len(X))
    order = np.arange(len(X))
    blocks = np.array_split(order, min(10, len(order)))
    chosen: list[int] = []
    per = max(1, n // len(blocks))
    for b in blocks:
        take = min(per, len(b))
        chosen.extend(rng.choice(b, size=take, replace=False).tolist())
    chosen = list(dict.fromkeys(chosen))[:n]

    # Force-include local example rows so waterfalls can always render.
    example_positions: dict[str, int] = {}
    if predictions is not None:
        example_positions = select_local_examples(
            np.asarray(predictions["y_true"]),
            np.asarray(predictions["y_pred"]),
            np.asarray(predictions["y_proba"]),
        )
        pred_index = np.asarray(predictions["index"])
        idx_to_pos = {int(i): p for p, i in enumerate(idx.to_numpy())}
        for pos in example_positions.values():
            orig = int(pred_index[pos])
            if orig in idx_to_pos:
                local_pos = idx_to_pos[orig]
                if local_pos not in chosen:
                    chosen.append(local_pos)

    chosen = chosen[: max(n, len(chosen))]
    Xs = pd.DataFrame(X[chosen], columns=feature_cols)
    sample_idx = idx.to_numpy()[chosen]

    logger.info(
        "Computing SHAP on %s samples x %s features (%s)",
        len(Xs),
        Xs.shape[1],
        prefix,
    )
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
    if prefix == "study_a":
        imp.to_csv(paths.tables / "shap_global_importance.csv", index=False)
        grouped.to_csv(paths.tables / "shap_grouped_importance.csv", index=False)
        grouped.to_csv(paths.tables / "shap_feature_family_importance.csv", index=False)

    generated: list[str] = []

    # Global bar
    fig, ax = plt.subplots(figsize=(10, 8))
    top = imp.head(20)
    ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1], color="#1f4e79")
    ax.set_xlabel("Mean |SHAP|")
    ax.set_title(f"Global SHAP Importance ({prefix})")
    fig.tight_layout()
    out = fig_dir / f"{prefix}_shap_global.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    generated.append(out.name)
    # Also write canonical alias used in docs
    if prefix == "study_a":
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1], color="#1f4e79")
        ax.set_xlabel("Mean |SHAP| value")
        ax.set_title("Global SHAP Feature Importance (TreeSHAP)")
        fig.tight_layout()
        alias = fig_dir / "shap_global_importance.png"
        fig.savefig(alias, dpi=dpi)
        plt.close(fig)
        generated.append(alias.name)

    # Beeswarm per class (multiclass) or single
    try:
        if values.ndim == 3:
            class_names = (
                ["DOWN", "STABLE", "UP"]
                if values.shape[2] >= 3 and not binary
                else ["DOWN", "UP"]
            )
            for class_i, name in enumerate(class_names[: values.shape[2]]):
                plt.figure(figsize=(10, 8))
                shap.summary_plot(
                    values[:, :, class_i],
                    Xs,
                    show=False,
                    max_display=20,
                )
                plt.title(f"SHAP Beeswarm — class {name}")
                plt.tight_layout()
                path = fig_dir / f"shap_beeswarm_{name.lower()}.png"
                plt.savefig(path, dpi=dpi, bbox_inches="tight")
                plt.close()
                generated.append(path.name)
        else:
            plt.figure(figsize=(10, 8))
            shap.summary_plot(values, Xs, show=False, max_display=20)
            plt.title("SHAP Beeswarm")
            plt.tight_layout()
            path = fig_dir / "shap_beeswarm.png"
            plt.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close()
            generated.append(path.name)
    except Exception as exc:  # noqa: BLE001
        logger.warning("SHAP beeswarm failed: %s", exc)

    # Dependence plots (UP class when multiclass)
    for feat in DEPENDENCE_FEATURES:
        if feat not in feature_cols:
            continue
        try:
            plt.figure(figsize=(8, 5))
            if values.ndim == 3:
                up_i = min(2, values.shape[2] - 1)
                shap.dependence_plot(
                    feat,
                    values[:, :, up_i],
                    Xs,
                    show=False,
                    interaction_index="auto",
                )
                plt.title(f"SHAP Dependence: {feat} (UP class)")
            else:
                shap.dependence_plot(feat, values, Xs, show=False)
                plt.title(f"SHAP Dependence: {feat}")
            plt.tight_layout()
            path = fig_dir / f"shap_dependence_{feat}.png"
            plt.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close()
            generated.append(path.name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Dependence plot for %s failed: %s", feat, exc)

    # Local waterfalls + examples table
    local: dict[str, Any] = {}
    if predictions is not None:
        try:
            y_true = np.asarray(predictions["y_true"])
            y_pred = np.asarray(predictions["y_pred"])
            y_proba = np.asarray(predictions["y_proba"])
            pred_index = np.asarray(predictions["index"])
            examples = example_positions or select_local_examples(
                y_true, y_pred, y_proba
            )
            sample_lookup = {int(i): p for p, i in enumerate(sample_idx)}

            for name, pos in examples.items():
                orig_index = int(pred_index[pos])
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
                if orig_index not in sample_lookup:
                    logger.warning(
                        "Waterfall %s: row %s not in SHAP sample", name, orig_index
                    )
                    continue
                loc = sample_lookup[orig_index]
                try:
                    plt.figure(figsize=(10, 6))
                    if values.ndim == 3:
                        cls = int(y_pred[pos])
                        cls = min(cls, values.shape[2] - 1)
                        shap.plots.waterfall(
                            explanation[loc, :, cls], show=False
                        )
                    else:
                        shap.plots.waterfall(explanation[loc], show=False)
                    plt.tight_layout()
                    path = fig_dir / f"shap_waterfall_{name}.png"
                    plt.savefig(path, dpi=dpi, bbox_inches="tight")
                    plt.close()
                    generated.append(path.name)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Waterfall %s failed: %s", name, exc)
                    plt.close("all")

            pd.DataFrame(local).T.to_csv(paths.tables / "shap_local_examples.csv")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Local SHAP examples failed: %s", exc)

    stability = pd.DataFrame()
    if cfg.get("stability_analysis", True):
        try:
            stability = regime_stability(df, sample_idx, explanation, feature_cols)
            stability.to_csv(paths.tables / "shap_stability_results.csv", index=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("SHAP regime stability failed: %s", exc)

    return {
        "n_shap_samples": len(Xs),
        "top_features": imp.head(15).to_dict(orient="records"),
        "grouped": grouped.to_dict(orient="records"),
        "sample_index": sample_idx.tolist(),
        "local_examples": local,
        "generated_figures": generated,
        "disclaimer": (
            "SHAP explains how the model uses features. "
            "SHAP does not prove causal relationships."
        ),
    }

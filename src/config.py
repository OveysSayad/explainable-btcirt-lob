"""Configuration loading with optional YAML inheritance."""

from __future__ import annotations

import logging
import random
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved filesystem paths for the project."""

    root: Path
    raw: Path
    interim: Path
    processed: Path
    models: Path
    figures: Path
    tables: Path
    metrics: Path
    shap: Path
    logs: Path
    reports: Path
    models_report: Path


def find_project_root(start: Path | None = None) -> Path:
    """Locate project root by walking upward for configs/project_config.yaml."""
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "configs" / "project_config.yaml").exists():
            return candidate
        if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
            return candidate
    return current


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base."""
    out = deepcopy(base)
    for key, value in override.items():
        if key == "inherits":
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """
    Load YAML configuration.

    Supports an optional ``inherits`` key pointing at another YAML file
    relative to the configs directory (or project root).
    """
    root = find_project_root()
    path = Path(config_path) if config_path else root / "configs" / "project_config.yaml"
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}

    if "inherits" in raw:
        parent_name = raw["inherits"]
        parent_path = root / "configs" / parent_name
        if not parent_path.exists():
            parent_path = root / parent_name
        parent = load_config(parent_path)
        config = _deep_merge(parent, raw)
    else:
        config = raw

    config["_config_path"] = str(path)
    config["_project_root"] = str(root)
    return config


def resolve_paths(config: dict[str, Any]) -> ProjectPaths:
    """Resolve and create important project directories."""
    root = Path(config["_project_root"])
    data = config["data"]
    output = config.get("output", {})
    paths_cfg = config.get("paths", {})

    def _resolve(rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else root / p

    paths = ProjectPaths(
        root=root,
        raw=_resolve(data["raw_path"]),
        interim=_resolve(data.get("interim_path", "data/interim/btcirt_clean.parquet")),
        processed=_resolve(data.get("processed_path", "data/processed/btcirt_features.parquet")),
        models=_resolve(output.get("models_dir", "models")),
        figures=_resolve(paths_cfg.get("figures", "reports/figures")),
        tables=_resolve(paths_cfg.get("tables", "reports/tables")),
        metrics=_resolve(paths_cfg.get("metrics", "reports/metrics")),
        shap=_resolve(paths_cfg.get("shap", "reports/shap")),
        logs=_resolve(output.get("logs_dir", "logs")),
        reports=_resolve(output.get("reports_dir", "reports")),
        models_report=_resolve(paths_cfg.get("models_report", "reports/models")),
    )
    for directory in [
        paths.interim.parent,
        paths.processed.parent,
        paths.models,
        paths.figures,
        paths.tables,
        paths.metrics,
        paths.shap,
        paths.logs,
        paths.models_report,
        paths.figures / "data_quality",
        paths.figures / "labels",
        paths.figures / "models",
        paths.figures / "shap",
        paths.figures / "ablation",
        paths.figures / "robustness",
    ]:
        directory.mkdir(parents=True, exist_ok=True)
    return paths


def set_global_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import os

        os.environ["PYTHONHASHSEED"] = str(seed)
    except Exception:  # noqa: BLE001
        pass
    logger.info("Global random seed set to %s", seed)


def get_seed(config: dict[str, Any]) -> int:
    """Return configured random seed."""
    return int(config["project"]["random_seed"])


def numeric_eps(config: dict[str, Any]) -> float:
    """Return denominator epsilon for numerical stability."""
    return float(
        config.get("numeric", {}).get(
            "denominator_epsilon",
            config.get("features", {}).get("numeric_epsilon", 1e-12),
        )
    )

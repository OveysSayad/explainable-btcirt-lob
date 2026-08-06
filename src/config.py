"""Configuration loading and path helpers."""

from __future__ import annotations

import logging
import random
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
    labels: Path
    models: Path
    figures: Path
    tables: Path
    metrics: Path
    shap: Path
    logs: Path
    reports: Path


def find_project_root(start: Path | None = None) -> Path:
    """Locate the project root by walking upward for configs/project_config.yaml."""
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "configs" / "project_config.yaml").exists():
            return candidate
        if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
            return candidate
    return current


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML configuration and resolve relative paths against the project root."""
    root = find_project_root()
    path = Path(config_path) if config_path else root / "configs" / "project_config.yaml"
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        config: dict[str, Any] = yaml.safe_load(handle)

    config["_config_path"] = str(path)
    config["_project_root"] = str(root)
    return config


def resolve_paths(config: dict[str, Any]) -> ProjectPaths:
    """Resolve all important project paths from configuration."""
    root = Path(config["_project_root"])
    data = config["data"]
    output = config["output"]

    def _resolve(rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else root / p

    paths = ProjectPaths(
        root=root,
        raw=_resolve(data["raw_path"]),
        interim=_resolve(data["interim_path"]),
        processed=_resolve(data["processed_path"]),
        labels=_resolve(data["labels_path"]),
        models=_resolve(output["models_dir"]),
        figures=_resolve(config.get("paths", {}).get("figures", "reports/figures")),
        tables=_resolve(config.get("paths", {}).get("tables", "reports/tables")),
        metrics=_resolve(config.get("paths", {}).get("metrics", "reports/metrics")),
        shap=_resolve(config.get("paths", {}).get("shap", "reports/shap")),
        logs=_resolve(output["logs_dir"]),
        reports=_resolve(output["reports_dir"]),
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
    ]:
        directory.mkdir(parents=True, exist_ok=True)
    return paths


def set_global_seed(seed: int) -> None:
    """Set random seeds for reproducibility across libraries."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import os

        os.environ["PYTHONHASHSEED"] = str(seed)
    except Exception:  # noqa: BLE001
        pass
    logger.info("Global random seed set to %s", seed)


def get_seed(config: dict[str, Any]) -> int:
    """Return the configured random seed."""
    return int(config["project"]["random_seed"])

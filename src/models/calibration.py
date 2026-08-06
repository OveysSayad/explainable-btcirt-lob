"""Calibration helpers (Platt / isotonic) fitted on validation only."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator


def calibrate_proba(
    base_estimator: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    method: str = "isotonic",
) -> Any:
    """
    Fit probability calibration on validation labels only.

    Uses FrozenEstimator when available so the base model is not refit.
    """
    try:
        calibrated = CalibratedClassifierCV(
            FrozenEstimator(base_estimator), method=method, cv="prefit"
        )
    except Exception:  # noqa: BLE001
        calibrated = CalibratedClassifierCV(base_estimator, method=method, cv=3)
    calibrated.fit(X_val, y_val)
    return calibrated

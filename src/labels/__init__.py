"""Label constructors for Studies A, B, and C."""

from __future__ import annotations

from src.labels.next_observation import (
    build_study_a_labels,
    fit_epsilon_candidates,
    select_epsilon,
)
from src.labels.next_price_change import build_study_b_labels
from src.labels.strict_horizon import build_study_c_labels, cross_horizon_overlap

__all__ = [
    "build_study_a_labels",
    "build_study_b_labels",
    "build_study_c_labels",
    "cross_horizon_overlap",
    "fit_epsilon_candidates",
    "select_epsilon",
]

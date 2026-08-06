"""Explainability package: SHAP, permutation, ablation, fallbacks."""

from src.explainability.shap_analysis import run_shap_analysis
from src.explainability.ablation import run_feature_set_and_ablation
from src.explainability.permutation_importance import permutation_importance_table
from src.explainability.interaction_fallbacks import run_interaction_analysis

__all__ = [
    "run_shap_analysis",
    "run_feature_set_and_ablation",
    "permutation_importance_table",
    "run_interaction_analysis",
]

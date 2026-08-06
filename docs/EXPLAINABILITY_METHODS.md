# Explainability Methods

## Complementary evidence

A feature/family is “strongly supported” only when **SHAP**, **permutation
importance**, and **grouped ablation** agree directionally.

## SHAP

- TreeSHAP for XGBoost when compatible
- Global mean |SHAP|, class-specific bars, beeswarm, dependence, local waterfalls
- Grouped SHAP by feature family
- Stability of ranks across folds (when multiple folds available)

**SHAP is not causal** and high SHAP does not imply positive out-of-sample value
(trade features previously showed this tension).

## Permutation importance

Permute within dates when possible; report mean Macro-F1 drop.

## Ablation / feature sets

Mandatory Price / Static LOB / Dynamic LOB / Full / Trade / No-time comparisons.
Selection uses validation / nested folds — not development-test cherry-picking.

## Interaction analysis

Attempt native SHAP interactions; on failure log exact error and package
versions to `reports/metrics/shap_interaction_status.json` and use fallbacks
(dependence colored by second feature, 2D PDP, conditional tables).
**Never fabricate interaction values.**

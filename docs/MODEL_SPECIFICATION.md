# Model Specification

## Baselines

- Majority class (train mode)
- Stratified random (train frequencies, fixed seed)
- Previous mid-direction (train ε)
- OBI-5 rule (threshold tuned on train Macro F1)

## Logistic regression

`StandardScaler` (train-only) + multinomial logistic with balanced class weights.
Search: L2/elasticnet, C grid, optional `l1_ratio`.

## XGBoost (primary)

- Study A: `multi:softprob`, 3 classes
- Study B: `binary:logistic` (DOWN/UP mapped to 0/1)
- Objective for search: validation Macro F1; secondary log loss
- Early stopping on validation
- Default trials: A=50, B=30, C=20 (reduced automatically if tiny samples)
- CPU `tree_method=hist`, `random_state=42`

Search space includes depth, learning rate, estimators, subsample, colsample,
gamma, reg_alpha/lambda, min_child_weight, max_delta_step.

## CatBoost (challenger)

Multiclass (or binary for Study B). Compared fold-wise / on development_test;
superiority claimed only if stable.

## Optional two-stage (Study A)

Stage 1 MOVE vs STABLE; Stage 2 UP vs DOWN conditional on MOVE.
Not primary unless walk-forward supports it.

## Calibration

Platt / isotonic evaluated on validation only; development_test never used to fit
calibration.

## Frozen specification

`reports/models/frozen_model_specification.json` records study, features, ε,
model class, hyperparameters, seed, software versions, training date range.

**Final independent holdout evaluation pending.**

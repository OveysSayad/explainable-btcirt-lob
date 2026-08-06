# Validation Design

## Date-based chronological split

Earliest 60% of dates → train; next 20% → validation; latest 20% →
**development_test**.

The development_test has already influenced prior analysis and is **not** a
pristine final holdout.

## Target-timestamp purging

For each row, store exact `target_timestamp`.

- Train: `target_timestamp < validation_start`
- Validation: `target_timestamp < development_test_start`

Drop rows whose target crosses a boundary. Same rule inside walk-forward folds.
Tests in `tests/test_purging.py` enforce this contract.

## Nested walk-forward

Anchored expanding outer folds; within each training region, tune ε / features /
hyperparameters using only that fold's train/val. Development_test is a final
*development* estimate after freezing the specification.

## Day-level bootstrap

Bootstrap **dates** with replacement (≥1000 iterations), not independent rows.
Report 95% CIs for Macro F1 and model differences.

## Final future holdout protocol

Collect ≥7–14 new days after freezing the model specification. Evaluate once.
Until then: **Final independent holdout evaluation pending.**

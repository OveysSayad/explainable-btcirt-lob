# Configuration changes after data inspection

These adjustments are documented so the pipeline remains honest about sampling constraints.

1. **`align_to_grid: false`**
   - Median inter-snapshot gap ≈ 69.3 seconds (test window median ≈ 189 seconds).
   - A 5-second grid is not justified.

2. **`label_match_tolerance_seconds: 300`**
   - Short horizons with a 120s tolerance produced **zero** test labels in the sparse final dates.
   - Matching uses **strictly future** snapshots only (never the current row).

3. **Rolling / OFI / volatility windows lengthened to 120–600 seconds**
   - Original 10–60 second time-based rolling windows were empty under ~69s gaps and yielded all-NaN features.
   - Feature names reflect the windows actually used (`volatility_120s`, `normalized_ofi_300s`, etc.).

4. **Nominal 10s / 30s / 60s label horizons often resolve to the same next snapshot**
   - Horizon comparison metrics can be nearly identical; this is a data-frequency limitation, not a coding error.

5. **macOS OpenMP**
   - XGBoost requires `libomp`. Use `scripts/run_pipeline.sh` or `brew install libomp`.


## Research redesign v2

- Replaced fixed-horizon-as-primary with Studies A/B/C.
- Study A = next observation; Study B = next price change; Study C = strict windows.
- Target-timestamp purging replaces fixed 60s purge.
- Trade deduplication module added; trades excluded from primary feature set.
- Rolling windows remain 120–1200s given sparse gaps.
- Development test is not a pristine holdout.

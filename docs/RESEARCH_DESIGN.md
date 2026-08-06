# Research Design — Explainable BTCIRT LOB

## Positioning

This project studies **sparse** Nobitex BTCIRT limit-order-book (LOB) snapshots.
It does **not** claim true high-frequency forecasting from dense event streams.

## Four studies

### Study A — Next observed mid-price movement (primary)

**Question:** Can the current and recent LOB state predict the direction of the
*next observed* mid-price movement (`DOWN` / `STABLE` / `UP`)?

Labels use the immediately following snapshot. Actual delay is stored; no false
claim of an exact 10/30/60-second horizon.

### Study B — Next mid-price *change* direction

**Question:** Conditional on the mid eventually changing, can the LOB predict
whether the next observed change is `UP` or `DOWN`?

Primary sample: one representative observation per unchanged-price run.

### Study C — Strict fixed-horizon pilot

Retain samples only when the actual future observation delay falls in:

| Horizon | Window |
|---------|--------|
| 10s | [5, 15] |
| 30s | [20, 40] |
| 60s | [40, 80] |

If samples are small, results are labeled **Pilot fixed-horizon analysis**.

### Study D — Dense data-collection design

Documentation and safe scaffolding only (`docs/DENSE_DATA_COLLECTION_DESIGN.md`).
No fabricated WebSocket data.

## Valid vs invalid claims

**Valid:** next-observation predictability; next-change direction; limited
strict-horizon pilots; incremental LOB value; explanation stability; SHAP ≠ OOS value.

**Invalid:** true HFT from sparse snapshots; exact 30s forecasts when the next
snapshot is minutes away; causal SHAP; trading P&L from F1; event-level OFI;
cross-market generalization; production trading readiness.

## Validation

- Chronological date splits (60/20/20)
- Latest block = **development_test** (not a pristine holdout)
- Target-timestamp purging across boundaries
- Nested walk-forward metadata for specification honesty
- Final independent future holdout **pending**

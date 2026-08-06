# Limitations

1. **Sparse snapshots** — median gap ≈ 69s; not HFT event data.
2. **No event-level OFI** — snapshot proxies only.
3. **Development_test contamination** — prior inspection; not final holdout.
4. **Short calendar span** — ~44 dates in one market regime window.
5. **Single venue/symbol** — Nobitex BTCIRT only.
6. **No causal claims** from SHAP or ablation.
7. **Execution realism** — financial sanity is exploratory only.
8. **Study C sample size** — strict windows often underpowered.
9. **Trade dedup imperfect** — identical consecutive trades may be under-counted.
10. **Tick in bps tiny** — absolute IRT tick ÷ huge mid yields negligible bps;
    hybrid ε relies mainly on return quantiles and spreads.
11. **Final future holdout pending.**

# Synthetic Overfit — Run Notes

Living log. See [FINDINGS.md](FINDINGS.md) for synthesis.

---

## Final results (2026-05-28)

| Phase | Status | rel_mae | pearson | dcr | depth-offset | steps |
|-------|--------|---------|---------|-----|--------------|-------|
| P1 | PASS | 0.026 | 0.989 | 1.0 | no | ~1200 |
| P2 | PASS | 0.046 | 0.985 | 4.54 | no | ~6000 |
| P3 default | FAIL | 0.055 | 0.943 | **1.30** | no | 4300 best |
| P3 offset | PASS | 0.059 | 0.987 | **4.00** | yes | ~100 |
| P4 offset | PASS | 0.061 | 0.984 | 4.00 | yes | ~100 |
| P5 offset | PASS | 0.095 | 0.985 | 4.00 | yes | ~300 |

### Key learnings
1. Count scale must use depth multipliers 5–40, not production log2 depth 22–25.
2. NB NLL floor ~4 at count~500 — use rel_mae for pass criteria.
3. dcr probe on unmasked P2 rises to ~4.5 when model fits (uses y_meta when trained with varying depth).
4. **Imputation dcr≈1.3 without offset → Q5 confirmed on v2.**
5. E29 depth-offset fixes P3–P5 in <300 steps each.

---

## Ablation matrix (2026-05-28, `run_experiments.py`)

18 synthetic runs E01–E22. Full table in `REPORT.md`.

## Real chr19 diagnostics (2026-05-28, `run_real_experiments.py`)

See `REAL_EXPERIMENTS.md` for batch plan.

| ID | Status | loss | dcr | imp_pearson |
|----|--------|------|-----|-------------|
| R01 | FAIL (Q5) | 1.11→0.005 | **1.001** | 0.51 |
| R02 | FAIL | 17.8→0.015 | **1.000** | ~0 (offset raw 2^d) |
| R03–R04 | PASS | reconstruct, no mask | — | — |
| R05–R06 | FAIL | 8-batch cycle diverges | ~1.0 | — |

**Key:** raw offset works on synthetic (depth 2–5) but NOT on real log2 depth ~24 at count scale 0–6.

Batch 2 (done):

| ID | Result | dcr | imp_p |
|----|--------|-----|-------|
| R07 | FAIL | 0.993 | 0.07 |
| R08 | FAIL | 1.000 | 0.01 (raw offset) |
| R15 | **PASS** | **4.18** | 0.99 |
| R16 | **PASS** | **4.00** | — (no FiLM) |
| R17 | **PASS** | **4.14** | 0.99 (assay-only mask) |

**Fix for real data:** `depth_center=24` → μ = 2^(d−24)·exp(η).

## Batch 3 (done)

| ID | dcr | Notes |
|----|-----|-------|
| R18 | 4.01 | 8-batch cycle: dcr OK, loss diverges (imp_p=0.14) |
| R19 | 0.996 | count_peak default — Q5 persists |
| R19b | 4.02 | count_peak + centered offset PASS |
| R20 default | 1.00 | 3-epoch smoke, 50 batches/epoch |
| R20 offset | 4.00 | dcr≈4 from epoch 0 |

## Metadata + imputation (M01–M10)

See `META_CONDITIONING.md`. Headlines: default imp_p=0.12; offset imp_p≈0.99; x_meta→Z requires enc FiLM; y depth via offset; runtype unused.

**M08–M10 (follow-ups):** masked-bin dcr≈4.24 fixes probe dilution bug; observed x_meta ablation does not hurt imputation; mixed-mask training keeps y_dcr≈4 but imp_p≈0.

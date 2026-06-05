# Synthetic Overfit — Problems & Fixes

## Resolved (harness)

- Tensor layout, count scale, NLL criteria, DSF binomial — see prior entries
- GradientMonitor crash when decoder FiLM disabled (E06) — skip None modules
- Pass criteria skipped dcr on non-50 steps — always require dcr when min_depth_ratio>0

---

## Open (production)

### P0 — Q5 metadata collapse on imputation (default NB head)
- **Evidence:** E01 dcr=1.30, R01 dcr=1.0, M01 imp_p=0.12
- **Fix validated in diagnostics:** E29 depth-offset with `depth_center≈24`
- **M08:** Q5 passes on imputed bins (dcr≈4.24); prior `y_depth_dcr_masked_assays≈1.8` was probe dilution
- **Not promoted** to v2 default per user request

### P2 — Spatial sine hard at L=768
- **Evidence:** E03/E04 pearson≈0; E15 spatial P5 fails despite dcr≈4
- **Priority:** research / capacity, not Q5 blocker

### P2 — Late-training divergence
- **Evidence:** ~step 6300 on long P2/P3 default runs
- **Mitigation:** best-checkpoint restore in harness

### P3 — Meta/film LR asymmetry unstable
- **Evidence:** E13 rel_mae>700 at 10× meta LR
- **Action:** do not use in production without careful tuning

---

## Ablation-validated non-fixes (default head)

| Knob | Result |
|------|--------|
| lr=1e-3 / 3e-3 | Fail to converge (E09/E10) |
| AdamW wd=1e-4 | Good fit, dcr=1.71 (E11) |
| clip_norm=1 | Good fit, dcr=1.0 (E18) |
| Decoder/encoder FiLM | Insufficient without offset (E01 vs E06–E08) |

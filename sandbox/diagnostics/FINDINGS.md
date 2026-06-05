# Synthetic Overfit Diagnosis — Findings & Recommendations

Date: 2026-05-28  
Harness: synthetic (`run_experiments.py`) + real chr19 (`run_real_experiments.py`, `real_train_smoke.py`)

See **`REPORT.md`** (synthetic), **`REAL_EXPERIMENTS.md`** (real plan).

## Executive Summary

**Synthetic:** E29 depth-offset fixes Q5 at toy depth scale (dcr≈4); FiLM optional.

**Real chr19:** Q5 reproduces on default v2 (dcr≈1). Raw `2^d` offset fails; **`depth_center=24`** (μ = 2^(d−24)·exp(η)) fixes Q5 on single-batch overfit, assay-only mask, count_peak, and **3-epoch iterable training** (R20).

## Real chr19 — all batches

| ID | Result | dcr | Notes |
|----|--------|-----|-------|
| R01 | FAIL Q5 | 1.001 | loss fits, imp_p=0.51 |
| R02 | FAIL | 1.000 | raw offset |
| R15–R17 | **PASS** | ~4.0 | depth_center=24 |
| R18 | partial | 4.008 | 8-batch cycle: dcr OK, loss diverges |
| R19 | FAIL Q5 | 0.996 | count_peak default |
| R19b | **PASS** | 4.017 | count_peak + centered offset |
| R20 default | — | 1.000 | 3 epochs × 50 batches |
| R20 offset | **PASS** | 3.999 | dcr≈4 from epoch 0 |

## Recommended next step (production)

Promote E29 with **depth centering** (or median size-factor) into v2 NB head; stage sandbox E29 run with `depth_center` from batch median log2 depth (~24 on EIC).

## Commands

```bash
python -m sandbox.diagnostics.run_real_experiments --batch 3
python -m sandbox.diagnostics.real_train_smoke --depth-offset --depth-center 24
python -m sandbox.diagnostics.run_meta_diagnostics
python -m sandbox.diagnostics.run_meta_followups
python -m sandbox.diagnostics.autoresearch.train   # Karpathy-style head search loop
```

See **`META_CONDITIONING.md`** for imputation + x_meta/y_meta probe results (M01–M10).

**Autoresearch** (`sandbox/diagnostics/autoresearch/`): agent edits `autoresearch/train.py` only; all changes confined to `sandbox/diagnostics/`. See `autoresearch/program.md`.

## M08–M10 (follow-ups)

- **M08:** Fixed masked-bin dcr probe — assay-only offset model passes Q5 on imputed bins (dcr≈4.24, not 1.8).
- **M09:** Observed x_meta ablation does not hurt imputation; y_meta/offset carries depth for masked assays.
- **M10:** Under mixed masking, y_dcr stable at 4; x_meta→Z sensitivity collapses during training; imp_p stays low (expected for loci+assay mask).

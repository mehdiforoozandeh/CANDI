# Metadata Conditioning — Findings (M01–M10)

Date: 2026-05-28  
Artifacts: `runs/meta_diagnostics.json`, `runs/meta_followups.json`

## Imputation (masked assays)

| ID | imp_p | y_dcr (all) | y_dcr (masked cols only) |
|----|-------|-------------|--------------------------|
| M01 default | **0.12** | 1.00 | 1.00 |
| M02 offset | **0.99** | 4.15 | **4.24** (M08) |
| M03 default mixed mask | 0.69 | 1.00 | 1.00 |
| M04 offset mixed | **0.99** | 4.31 | 1.68 |
| M07 offset heavy mask | **0.99** | 4.16 | **4.06** |

- Default head: imputation fails (low imp_p, dcr≈1, `y_wrong_depth_masked_mse≈0`).
- Centered offset: imputation works (imp_p≈0.99).
- **M08 (fixed probe):** `y_depth_dcr_masked_assays` ≈1.8 was a **probe aggregation bug** (sums observed+masked columns). Restricting to `masked_map` bins gives dcr≈**4.24**; per-assay median on masked bins ≈**4.02**. Q5 passes on imputation bins under assay-only mask.

## M08–M10 follow-ups

| ID | Key result |
|----|------------|
| **M08** | masked-bin dcr=4.24, median-per-assay=4.02; imp_p=0.98 — confirms Q5 on imputed bins |
| **M09** | Wipe observed x_meta at eval: imp_p 0.984→**0.988**, dcr unchanged; Δz=0.23 — imputation is **y_meta/offset**, not observed x_meta |
| **M10** | 400-step mixed-mask train: y_dcr≈4 from step 1; x_depth Δz collapses 0.011→~0.003; CLOZE fill Δz 0.55→~0.15; imp_p≈0 (mixed mask ≠ assay-only) |

## y_meta (prompt) → output

| Signal | Default | Offset center=24 |
|--------|---------|------------------|
| Depth (global dcr) | ≈1.0 | ≈4.0 |
| Readlen count MSE | ~0.03 | large (FiLM path; unstable at overfit) |
| Runtype count MSE | ≈0 | ≈0 |
| Wrong depth on masked y | ≈0 | >0 (small) |

- **Depth on y_meta drives counts** via explicit offset (primary) + FiLM (readlen).
- **Runtype row unused** in count head — architectural gap if paired-end matters for counts.
- Decoder FiLM off (M05): depth via offset still works (dcr≈4); readlen MSE=0 → readlen only through FiLM.

## x_meta (input) → latent Z

| Signal | Init | Final (trained) | enc FiLM off (M06) |
|--------|------|-----------------|---------------------|
| Depth +1 log2 Δz/‖z‖ | 0.016 | 0.001–0.002 | **0.0** |
| Readlen →100 Δz/‖z‖ | 0.22 | 0.02–0.05 | **0.0** |
| Fill CLOZE depth from y | 0.50 | 0.11–0.16 | **0.0** |

- **Encoder x_meta affects latent primarily through FiLM** (M06: zero sensitivity with FiLM disabled).
- After overfit, x_meta sensitivity **shrinks** (model may rely on signal not meta for reconstruction).
- CLOZE fill with y_meta depth moves latent strongly at init → mask_token + metadata path is wired.

## Architectural recommendations

1. **Count head:** ship E29 with `depth_center≈24`; depth on y_meta is load-bearing for imputation.
2. **Decoder FiLM:** needed for read_length (and possibly assay_id); not needed for depth once offset present.
3. **Encoder FiLM:** only path for x_meta to affect Z; verify not collapsed in long training (x_delta→0 after overfit).
4. **Runtype:** no count effect — confirm intentional or add conditioning if needed.
5. **Imputation eval:** use `y_depth_dcr_on_masked_bins` + imp_pearson, not `y_depth_dcr_masked_assays` (diluted).

## Commands

```bash
python -m sandbox.diagnostics.run_meta_diagnostics
python -m sandbox.diagnostics.run_meta_followups   # M08–M10
```

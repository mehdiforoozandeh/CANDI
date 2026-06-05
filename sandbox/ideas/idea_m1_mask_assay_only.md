# M1 — Full assay masking only (Tier A)

Status: idea  
Parent: B8 (`baseline_E7_E13` reference stack)  
Run name: TBD (e.g. `M1_mask_assay_only`)  
Checklist entry: [EXPERIMENTS.md](EXPERIMENTS.md#masking-sweep-m1-m3-tier-a)

## Problem Statement

Isolate **cross-assay imputation** with **metadata cloze**: `DataMasker._mask_full_assay` masks data, per-assay metadata, and availability for a random subset of assays on each batch. No synchronous spatial holes (full loci) and no per-assay chunk noise.

## Idea / Hypothesis

Assay-only training (see historical B3 under the old stack) may emphasize assay completion and FiLM metadata conditioning without BERT-style locus corruption; under E7+E13 defaults it should be compared cleanly to B8’s stochastic mixture (`p_assay=0.8`, `p_loci=0.5`).

## Planned Intervention

- Config: `training.masking.p_full_assay=1.0`, `training.masking.p_full_loci=0.0`, `training.masking.p_chunks=0.0`.
- All other knobs match B8: `sandbox/configs/default.yaml` + type1_chr19, multi-head, same optimizer/LR/clip unless explicitly varied elsewhere.

## Verifiables

- Compare to B8 on cornerstone metrics (`quality_score`, `imp_*` Pearson, peak AUROC), branch losses, and pval_imp stability.
- Watch `depth_count_ratio` (F1): assay-only may still not fix metadata collapse.

## Run Links

- Run directory: TBD  
- Resolved config / metrics / SLURM: TBD

## Findings

- Observed: TBD  
- Interpretation: TBD  
- Decision: TBD

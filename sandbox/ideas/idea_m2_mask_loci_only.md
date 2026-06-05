# M2 — Full loci masking only (Tier A)

Status: idea  
Parent: B8 (`baseline_E7_E13` reference stack)  
Run name: TBD (e.g. `M2_mask_loci_only`)  
Checklist entry: [EXPERIMENTS.md](EXPERIMENTS.md#masking-sweep-m1-m3-tier-a)

## Problem Statement

Isolate **synchronous spatial masking**: the same locus chunks are masked across all **available** assays. `DataMasker` does **not** mask metadata for this strategy—column-level covariates remain visible for every assay.

## Idea / Hypothesis

Full-loci-only stresses **locus imputation with shared geometry** across tracks (plus context from unmasked bins) while keeping full assay metadata in the encoder; may improve spatial coherence metrics relative to B8 if locus-level structure is under-trained today.

## Planned Intervention

- Config: `training.masking.p_full_assay=0.0`, `training.masking.p_full_loci=1.0`, `training.masking.p_chunks=0.0`.
- Other settings match B8.

## Verifiables

- Compare imputation/denoising metrics on pval/count/peak vs B8; check whether obs vs imp losses show a different split from assay-only (M1).

## Run Links

- Run directory: TBD  
- Resolved config / metrics / SLURM: TBD

## Findings

- Observed: TBD  
- Interpretation: TBD  
- Decision: TBD

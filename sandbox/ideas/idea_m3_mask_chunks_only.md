# M3 — Per-assay chunk masking only (Tier A)

Status: idea  
Parent: B8 (`baseline_E7_E13` reference stack)  
Run name: TBD (e.g. `M3_mask_chunks_only`)  
Checklist entry: [EXPERIMENTS.md](EXPERIMENTS.md#masking-sweep-m1-m3-tier-a)

## Problem Statement

Isolate **independent spatial corruption per assay**: each available track gets its own random non-overlapping chunk pattern. Metadata columns are **not** masked (same as full loci). This is the strongest “denoise with cross-assay side information at unmasked positions” flavour among the three modes.

## Idea / Hypothesis

Chunk-only may sharpen **per-assay denoising** and reduce pressure on whole-assay completion relative to M1, or may weaken cross-assay transfer if the model relies on aligned holes; either outcome is informative vs B8.

## Planned Intervention

- Config: `training.masking.p_full_assay=0.0`, `training.masking.p_full_loci=0.0`, `training.masking.p_chunks=1.0`.
- Other settings match B8. Uses existing `mask_fraction` and `chunk_size` from config (same as B8 unless sweep is opened later).

## Verifiables

- Compare denoising (obs) vs imputation (imp) metrics vs M1/M2 and B8; grad-norm / clip diagnostics if instability appears.

## Run Links

- Run directory: TBD  
- Resolved config / metrics / SLURM: TBD

## Findings

- Observed: TBD  
- Interpretation: TBD  
- Decision: TBD

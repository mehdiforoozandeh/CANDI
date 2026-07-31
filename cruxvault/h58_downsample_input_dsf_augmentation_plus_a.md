---
id: h58
type: idea
title: Downsample-input DSF augmentation plus a thinning-consistency term trains the untrained upward-depth regime
parent: q20
status: idea
verdict:
metric:
created: 2026-07-24T11:12:18
updated: 2026-07-24T11:12:18
---

# h58 — Downsample-input DSF augmentation plus a thinning-consistency term trains the untrained upward-depth regime

Parent:: [[q20_how_should_candi_architecture_and_traini]]

## Problem Statement

B6: DSF only down-samples, so training never covers told>observed depth, yet 7/12 eval targets sit ABOVE the per-assay training ceiling. The offset makes the MEAN extrapolate exactly (B2) but eta and dispersion n are FiLM-driven and untrained at large upward told-depth. Add the mirror of the existing upsample_only mode (y_dsf <= x_dsf: input downsampled, target at full dsf1 depth, no re-bake) so training densely covers upward-scaling up to ~3 log2; add a binomial-thinning consistency term (predict at told d and d+delta, penalize deviation from the exact 2^d thinning law, n preserved) as a physics-exact upward prior needing no deep GT. HONEST LIMIT: within-biosample DSF cannot exceed a biosample's natural depth, so HD2 trains the upward OPERATION but must pair with h57 (deeper biosamples) to bracket every eval target.

## Idea / Hypothesis

Downsample-input DSF augmentation plus a thinning-consistency term trains the untrained upward-depth regime

## Verifiables

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
- [ ] the 7/12 above-ceiling targets scored SEPARATELY (oracle-scale decomposed): CRPS_oracle_scaled improves or dispersion calibration (PIT-ECE at high told-depth) improves; if unchanged, the offset already covered it and HD2 is inert (pre-registered null)
- [ ] overall macro CRPS <= wd0_on 1.341 with the gain concentrated in the extrapolation targets; total told-depth slope stays ~1
- [ ] thinning-consistency: predicted distributions at adjacent told-depths satisfy the 2^d law within tolerance (n-invariance held)

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Findings

_(written by the PI/agent when the case is closed)_

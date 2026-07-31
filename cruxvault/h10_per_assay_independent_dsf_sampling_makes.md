---
id: h10
type: idea
title: Per-assay independent DSF sampling makes depth a necessary signal at production scale
parent: q4
status: idea
verdict:
metric:
created: 2026-07-02T20:37:11
updated: 2026-07-02T20:37:11
---

# h10 — Per-assay independent DSF sampling makes depth a necessary signal at production scale

Parent:: [[q4_can_candi_s_counts_be_made_depth_control]]

## Problem Statement

The dual-conditioning testbed (q16) showed the depth/metadata pathway only receives gradient when the target is NOT inferable from the input: steering emerged there because f_x≠f_y made the target a non-copyable function of the metadata. Production's default recipe is the opposite — shared-DSF, reconstruct-the-same-assay — so the target depth is redundant with the input signal magnitude and the metadata depth channel gets ~zero gradient, which (together with h8's free-mean head) is why production DCR collapses to ~1. Per-assay independent DSF sampling is the training-side lever that makes depth a *necessary*, non-redundant signal at production scale — the analogue of the testbed's non-copyable task. Default is currently OFF (`enable_per_assay_dsf_sampling=False`, data_h5.py). Pairs with h9's offset head (the structural side of the same fix).

## Idea / Hypothesis

Per-assay independent DSF sampling makes depth a necessary signal at production scale

## Verifiables

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
- [ ] per-assay DSF in the default training recipe improves DCR and robustness versus shared DSF

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Findings

_(written by the PI/agent when the case is closed)_

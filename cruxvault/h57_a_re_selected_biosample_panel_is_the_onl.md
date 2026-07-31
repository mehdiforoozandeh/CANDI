---
id: h57
type: idea
title: A re-selected biosample panel is the only lever that makes run_type (and depth-vs-read_length attribution) learnable at all
parent: q20
status: idea
verdict:
metric:
created: 2026-07-24T11:12:18
updated: 2026-07-24T11:12:18
---

# h57 — A re-selected biosample panel is the only lever that makes run_type (and depth-vs-read_length attribution) learnable at all

Parent:: [[q20_how_should_candi_architecture_and_traini]]

## Problem Statement

B1 is a DATA property, not a model failure: on the T_ slice H(run_type|assay,read_length)=0.000 bits, so run_type is a deterministic function of the other two and h47's 'run_type alive-but-weak (frac_dir 0.559~chance)' cannot be fixed by any FiLM/head/optimizer change. The full EIC metadata HAS the contrasts (DNase-seq has single+paired at read_length 36 AND 76; H3K4me1/H3K36me3 at 76). Offline-select 5-8 T_ biosamples maximizing H(run_type|assay,read_length) with >=1 matched-rl single+paired assay and >=2 assays carrying a within-assay depth ladder at fixed read_length, preserving the T_/V_/B_ pairing so the ~12-target eval survives; re-bake. Production MERGED retains 0.551 bits run_type after conditioning on assay, so a positive here validates that the fork null was the 5-biosample slice.

## Idea / Hypothesis

A re-selected biosample panel is the only lever that makes run_type (and depth-vs-read_length attribution) learnable at all

## Verifiables

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
- [ ] PRE-CONDITION GATE: the new panel has H(run_type|assay,read_length) > 0 (reported before training)
- [ ] real-z run_type metadata-ablation degradation > 0 with clustered bootstrap over the NEW eval targets (report n_clusters; expect wide CIs at small n) -- moves from chance to genuinely directional BECAUSE the signal now exists
- [ ] macro CRPS does not regress below 1.341 on the re-baked panel; the de-confounded depth axis (B5) neutral-to-better

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Findings

_(written by the PI/agent when the case is closed)_

---
id: h32
type: idea
title: Invertibility sets difficulty; inverting f_x (input) is harder than applying f_y (output)
parent: q15
status: staged
verdict: 
metric: 
created: "2026-07-03T01:59:54"
updated: "2026-07-03T02:51:37"
---

# h32 — Invertibility sets difficulty; inverting f_x (input) is harder than applying f_y (output)

Parent:: [[q15_can_candi_learn_dual_metadata_conditioni]]

## Problem Statement

Is learnability graded by transform family - invertible (x-h, +h, power) easy vs non-invertible/information-losing (binomial thinning, saturating cap, count-log) hard - and is the INPUT side (encoder must invert/undo f_x) harder than the OUTPUT side (decoder applies f_y forward)?

## Idea / Hypothesis

Invertibility sets difficulty; inverting f_x (input) is harder than applying f_y (output)

## Verifiables

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
- [ ] per-family ranking: invertible families reach M1 >= 0.9 * identity-cell ceiling; non-invertible families show a gap >= 0.10; flag any input-side transform that is unlearnable.
- [ ] M3 invariance degrades (within/between cos-dist ratio rises) specifically for non-invertible INPUT transforms - the encoder cannot fully undo information loss (thinning/cap).
- [ ] matched-family asymmetry: for the same family, input-side difficulty (measured via M3 encoder invariance) exceeds output-side difficulty (measured via M2 steering) - invert-harder-than-apply.

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

- sandbox/diagnostics/dual_conditioning (impl complete, CPU-gated; awaiting GPU sweep)

## Findings

_(written by the PI/agent when the case is closed)_

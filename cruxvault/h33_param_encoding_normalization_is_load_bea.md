---
id: h33
type: idea
title: Param-encoding normalization is load-bearing for reading the augmentation covariate
parent: q15
status: staged
verdict: 
metric: 
created: "2026-07-03T02:21:15"
updated: "2026-07-03T02:51:37"
---

# h33 — Param-encoding normalization is load-bearing for reading the augmentation covariate

Parent:: [[q15_can_candi_learn_dual_metadata_conditioni]]

## Problem Statement

aug_param spans very different per-family scales (mult 0.25-4, add 2-20, power 0.5-2, thin 0.2-0.8, cap 2-20, clog 1-8). Fed raw into the param embedder, magnitudes collide across families and the model may under-read the param, depressing steering. Does normalizing the param materially improve conditioning quality? Test THREE arms - none / per-family z-score / global log-scale - by training the full h30 matrix under each. Doubles as a guard so a param-encoding artifact is not mistaken for a real conditioning limit in h30/h32, and picks the normalization that h31/h32 then use.

## Idea / Hypothesis

Param-encoding normalization is load-bearing for reading the augmentation covariate

## Verifiables

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
- [ ] rank the three arms by M2 param-steering R2 (fix family, sweep its 4 param values): load-bearing if per-family z-score exceeds no-normalization by delta-M2 >= 0.10; report the full ordering (z-score vs log-scale vs none).
- [ ] M1 reconstruction (ceiling gap) is not degraded by the winning normalization vs no-normalization (normalization must not cost accuracy).
- [ ] mechanistic: the M2 gain concentrates in wide-param-range families (mult, add, cap) where scale-collision bites hardest; report per-family M2 deltas across the three arms.

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

- sandbox/diagnostics/dual_conditioning (impl complete, CPU-gated; awaiting GPU sweep)

## Findings

_(written by the PI/agent when the case is closed)_

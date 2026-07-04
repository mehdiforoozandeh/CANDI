---
id: h14
type: idea
title: log1p input scaling and SGD remove late divergence, but not depth collapse
parent: q7
status: done
verdict: partial
metric: 
created: "2026-07-02T20:37:11"
updated: "2026-07-02T20:37:11"
---

# h14 — log1p input scaling and SGD remove late divergence, but not depth collapse

Parent:: [[q7_why_do_sandbox_training_runs_diverge_lat]]

## Problem Statement

_(why this is worth testing)_

## Idea / Hypothesis

log1p input scaling and SGD remove late divergence, but not depth collapse

## Verifiables

- [x] log1p B7 and SGD-lr1e-4 B4 show no divergence and improve imputation over the raw B1 baseline   (found: both accepted for stability)
- [ ] depth collapse is also fixed by these   (found: depth collapse persists, needs the size-factor fix)

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Findings

Input scaling and optimizer choice control divergence but not depth collapse — the two failure modes are separate.

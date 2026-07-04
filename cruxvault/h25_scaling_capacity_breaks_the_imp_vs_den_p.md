---
id: h25
type: idea
title: Scaling capacity breaks the imp-vs-den Pareto frontier
parent: q13
status: done
verdict: refuted
metric: 
created: "2026-07-02T20:37:12"
updated: "2026-07-02T20:37:13"
---

# h25 — Scaling capacity breaks the imp-vs-den Pareto frontier

Parent:: [[q13_what_architecture_changes_move_held_out_]]

## Problem Statement

_(why this is worth testing)_

## Idea / Hypothesis

Scaling capacity breaks the imp-vs-den Pareto frontier

## Verifiables

- [ ] capacity scaling improves the combined score   (found: decoder capacity gives den_r2 +0.256 but wrecks imputation; transformer capacity gives best imp but wrecks denoising)

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Findings

Under-convergence is real but capacity cannot beat the score — the frontier is fundamental to the shared backbone. Breaking it needs a regime change (loss weights toward the score, or larger compute/data for a dual backbone).

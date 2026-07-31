---
id: h38
type: idea
title: Foreground/background loss reweighting improves held-out reconstruction and steering
parent: q17
status: idea
verdict:
metric:
created: 2026-07-09T19:58:09
updated: 2026-07-09T19:58:09
---

# h38 — Foreground/background loss reweighting improves held-out reconstruction and steering

Parent:: [[q17_is_candi_training_biased_by_foreground_b]]

## Problem Statement

Upweight foreground positions (top-k% by base count, per assay) in the NBNLL loss so the sparse foreground is not drowned by background gradient. Sweep reweight strength lambda to trace the FG<->BG frontier. Same best-config as q15 2a (norm=none / per-assay / offset-on); baseline = flat-loss model already run in the 2a sweep. Eval on the fixed natural eval set (chr21 + chr19/21). Escape move if the tradeoff is hard: focal/residual weighting (upweight foreground only where the model underfits it).

## Idea / Hypothesis

Foreground/background loss reweighting improves held-out reconstruction and steering

## Verifiables

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
- [ ] Foreground reconstruction improves: foreground CRPS/Spearman (top-k% by base count, per assay) improves by a meaningful margin vs the flat-loss baseline, with aggregate CRPS not collapsing (guardrail within tolerance of baseline)
- [ ] Steering improves: per-assay M2 rises vs the flat-loss baseline (moves toward the 0.6 bar h30 fell short of)
- [ ] Pareto: sweeping reweight strength lambda traces a monotone FG<->BG frontier; either a point improves foreground without aggregate collapse, or the hard tradeoff is confirmed and focal/residual weighting pushes the frontier outward

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Findings

_(written by the PI/agent when the case is closed)_

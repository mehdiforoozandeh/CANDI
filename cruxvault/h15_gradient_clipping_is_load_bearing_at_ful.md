---
id: h15
type: idea
title: Gradient clipping is load-bearing at full LR
parent: q7
status: done
verdict: supported
metric: 
created: "2026-07-02T20:37:11"
updated: "2026-07-02T20:37:12"
---

# h15 — Gradient clipping is load-bearing at full LR

Parent:: [[q7_why_do_sandbox_training_runs_diverge_lat]]

## Problem Statement

_(why this is worth testing)_

## Idea / Hypothesis

Gradient clipping is load-bearing at full LR

## Verifiables

- [x] removing clip at default LR trails the clipped baseline   (found: E0b stable but below B8; clipping load-bearing)
- [x] a durable clip-active-fraction pressure metric is logged   (found: E10 implemented in metrics.jsonl)

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Findings

Clipping is required at full LR, not just a safety net; removing it underperforms even when training stays stable.

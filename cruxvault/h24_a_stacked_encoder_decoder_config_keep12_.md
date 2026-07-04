---
id: h24
type: idea
title: A stacked encoder/decoder config (KEEP12) is the single-knob optimum
parent: q13
status: done
verdict: partial
metric: 
created: "2026-07-02T20:37:12"
updated: "2026-07-02T20:37:12"
---

# h24 — A stacked encoder/decoder config (KEEP12) is the single-knob optimum

Parent:: [[q13_what_architecture_changes_move_held_out_]]

## Problem Statement

_(why this is worth testing)_

## Idea / Hypothesis

A stacked encoder/decoder config (KEEP12) is the single-knob optimum

## Verifiables

- [x] KEEP12 reaches primary ~ -0.4438   (found: +0.026 from decoder GroupNorm and +0.0038 from output_rms_norm over KEEP9)
- [ ] single-knob search yields further real gains   (found: exhausted; noise floor ~0.002; every untested knob is locked or toxic)

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Findings

KEEP12 is locked as best. imp-vs-den is a hard Pareto frontier rooted in the shared transformer backbone; the NB/depth head and transformer internals are structurally immutable.

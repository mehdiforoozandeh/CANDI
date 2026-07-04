---
id: h20
type: idea
title: gaussian_var_min = 0.1 stops pval obs/imp divergence
parent: q10
status: done
verdict: inconclusive
metric: 
created: "2026-07-02T20:37:12"
updated: "2026-07-02T20:37:12"
---

# h20 — gaussian_var_min = 0.1 stops pval obs/imp divergence

Parent:: [[q10_does_the_gaussianlayer_variance_floor_pr]]

## Problem Statement

_(why this is worth testing)_

## Idea / Hypothesis

gaussian_var_min = 0.1 stops pval obs/imp divergence

## Verifiables

- [x] the variance floor mitigates F7 in pval-only isolation   (found: E13 accepted)
- [-] it is confirmed in full multi-head training   (found: validation still pending)

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Findings

The floor mitigates the collapse; multi-head confirmation is still needed.

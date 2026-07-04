---
id: h16
type: idea
title: Isolating the pval head (pval-only training) improves pval learning
parent: q8
status: done
verdict: refuted
metric: 
created: "2026-07-02T20:37:12"
updated: "2026-07-02T20:37:12"
---

# h16 — Isolating the pval head (pval-only training) improves pval learning

Parent:: [[q8_do_the_count_pval_peak_heads_cooperate_o]]

## Problem Statement

_(why this is worth testing)_

## Idea / Hypothesis

Isolating the pval head (pval-only training) improves pval learning

## Verifiables

- [ ] pval-only training improves pval   (found: E3 rejected — variance collapse on obs, pval_imp explodes, root cause F7)

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Findings

The pval head is the source of instability and motivated the Gaussian variance floor.

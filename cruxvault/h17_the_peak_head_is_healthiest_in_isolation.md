---
id: h17
type: idea
title: The peak head is healthiest in isolation while counts are capacity-limited
parent: q8
status: done
verdict: partial
metric: 
created: "2026-07-02T20:37:12"
updated: "2026-07-02T20:37:12"
---

# h17 — The peak head is healthiest in isolation while counts are capacity-limited

Parent:: [[q8_do_the_count_pval_peak_heads_cooperate_o]]

## Problem Statement

_(why this is worth testing)_

## Idea / Hypothesis

The peak head is healthiest in isolation while counts are capacity-limited

## Verifiables

- [x] peak-only is the healthiest head with a strong AUROC ceiling and no divergence   (found: E4 accepted)
- [ ] count-only reaches good imputed counts   (found: E2 count_imp plateaus ~1.92; count+peak best for counts but peak still needs pval gradients)

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Findings

The heads have asymmetric health — peak robust, counts capacity-limited, pval fragile; full multi-head is a compromise.

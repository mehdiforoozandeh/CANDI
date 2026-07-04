---
id: h1
type: idea
title: CANDI beats EIC-challenge competitors on rank correlation at ~42M params with no cell-type embeddings
parent: q1
status: done
verdict: partial
metric: 
created: "2026-07-02T20:37:11"
updated: "2026-07-02T20:37:11"
---

# h1 — CANDI beats EIC-challenge competitors on rank correlation at ~42M params with no cell-type embeddings

Parent:: [[q1_does_a_raw_count_distribution_output_ssl]]

## Problem Statement

_(why this is worth testing)_

## Idea / Hypothesis

CANDI beats EIC-challenge competitors on rank correlation at ~42M params with no cell-type embeddings

## Verifiables

- [x] Spearman >= top EIC competitor on most assays   (found: SOTA Spearman across most assays, zero-shot via covariates)
- [ ] Pearson >= competitors   (found: Pearson lags — compressed dynamic range, high-signal magnitude underestimated)

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Findings

Structural/rank imputation is SOTA and cell-type-agnostic, but absolute magnitude of high-signal regions is underestimated. This rank-vs-magnitude gap is the recurring theme of all later work.

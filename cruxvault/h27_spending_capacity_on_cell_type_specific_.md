---
id: h27
type: idea
title: Spending capacity on cell-type-specific deviation, not relearning the shared average, is the only thing that crosses the marginal baseline
parent: q14
status: done
verdict: supported
metric: 
created: "2026-07-02T20:37:13"
updated: "2026-07-02T20:37:13"
---

# h27 — Spending capacity on cell-type-specific deviation, not relearning the shared average, is the only thing that crosses the marginal baseline

Parent:: [[q14_can_a_from_scratch_era_searched_candi_v3]]

## Problem Statement

_(why this is worth testing)_

## Idea / Hypothesis

Spending capacity on cell-type-specific deviation, not relearning the shared average, is the only thing that crosses the marginal baseline

## Verifiables

- [x] a zero-init residual head on a leak-free average-reference plus a deviation-correlation loss crosses the baseline   (found: ERA best node 267, S_A = +0.0183 > 0 over ~281 candidates)

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Findings

Confirms the ENCODE-challenge lesson — the per-position average reference is brutally strong, and skill is the cell-type deviation on top of it. This is v3's central mechanism.

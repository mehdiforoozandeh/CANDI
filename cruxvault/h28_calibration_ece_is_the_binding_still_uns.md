---
id: h28
type: idea
title: Calibration (ECE) is the binding, still-unsolved constraint
parent: q14
status: done
verdict: partial
metric: 
created: "2026-07-02T20:37:13"
updated: "2026-07-02T20:37:13"
---

# h28 — Calibration (ECE) is the binding, still-unsolved constraint

Parent:: [[q14_can_a_from_scratch_era_searched_candi_v3]]

## Problem Statement

_(why this is worth testing)_

## Idea / Hypothesis

Calibration (ECE) is the binding, still-unsolved constraint

## Verifiables

- [ ] NB count calibration meets the ECE floor from NLL alone   (found: ~78% of candidates fail ECE, NB counts systematically over-confident)
- [x] explicit second-moment / CRPS / dispersion-cap terms recover coverage   (found: moment-matching and CRPS lift calibration where NLL cannot)

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Findings

NLL does not yield coverage; explicit calibration terms are required. This is the honest "calibration is the hard part" framing for the paper.

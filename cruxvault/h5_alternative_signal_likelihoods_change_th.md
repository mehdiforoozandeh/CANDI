---
id: h5
type: idea
title: Alternative signal likelihoods change the calibration trade-off
parent: q2
status: done
verdict: inconclusive
metric: 
created: "2026-07-02T20:37:11"
updated: "2026-07-02T20:37:11"
---

# h5 — Alternative signal likelihoods change the calibration trade-off

Parent:: [[q2_are_candi_s_aleatoric_uncertainty_estima]]

## Problem Statement

_(why this is worth testing)_

## Idea / Hypothesis

Alternative signal likelihoods change the calibration trade-off

## Verifiables

- [x] Gaussian/Laplace/Student-t/Gamma/const-var heads implemented, ablated at ~52M params on EIC chr19 with calibration curves   (found: dist_report.md, calibration_imputed_log_normal/laplace.svg)
- [-] a default likelihood chosen for the paper   (found: winner not recorded — needs confirm)

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Findings

The signal head is pluggable and the distributional assumption was actually tested; which to present as the default is still unconfirmed.

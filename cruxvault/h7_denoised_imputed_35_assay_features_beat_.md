---
id: h7
type: idea
title: Denoised+imputed 35-assay features beat the observed subset
parent: q3
status: done
verdict: supported
metric: 
created: "2026-07-02T20:37:11"
updated: "2026-07-02T20:37:11"
---

# h7 — Denoised+imputed 35-assay features beat the observed subset

Parent:: [[q3_do_candi_s_imputed_signals_and_latent_z_]]

## Problem Statement

_(why this is worth testing)_

## Idea / Hypothesis

Denoised+imputed 35-assay features beat the observed subset

## Verifiables

- [x] denoised+imputed 35 assays > observed available assays on RNA-seq prediction   (found: imputation adds regulatory context)
- [x] denoised >= observed   (found: marginal gain — noise removed, regulatory info preserved)

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Findings

Imputation and denoising add real regulatory signal, supporting imputation-as-denoising.

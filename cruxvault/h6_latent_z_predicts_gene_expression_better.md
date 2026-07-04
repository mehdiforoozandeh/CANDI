---
id: h6
type: idea
title: Latent Z predicts gene expression better than observed/denoised signal and is robust to input sparsity
parent: q3
status: done
verdict: supported
metric: 
created: "2026-07-02T20:37:11"
updated: "2026-07-02T20:37:11"
---

# h6 — Latent Z predicts gene expression better than observed/denoised signal and is robust to input sparsity

Parent:: [[q3_do_candi_s_imputed_signals_and_latent_z_]]

## Problem Statement

_(why this is worth testing)_

## Idea / Hypothesis

Latent Z predicts gene expression better than observed/denoised signal and is robust to input sparsity

## Verifiables

- [x] Z beats observed and denoised+imputed features on RNA-seq log-TPM via nested CV   (found: Z is the strongest predictor)
- [x] Z and imputed+denoised performance are robust to the number of available input assays   (found: robust to sparsity unlike observed)

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Findings

The latent encodes higher-order regulatory information beyond the decoded tracks; strongest and most sparsity-robust RNA-seq predictor. Core biological-validation claim.

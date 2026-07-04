---
id: h30
type: idea
title: Dual conditioning is learnable when the full f_x x f_y matrix is seen in training
parent: q15
status: staged
verdict: 
metric: 
created: "2026-07-03T01:59:53"
updated: "2026-07-03T02:51:37"
---

# h30 — Dual conditioning is learnable when the full f_x x f_y matrix is seen in training

Parent:: [[q15_can_candi_learn_dual_metadata_conditioni]]

## Problem Statement

With every (input-transform, output-transform) cell present in training, can CANDIv2 normalize the covariate-transformed count input (encoder, x_meta) and steer the NB count output (decoder, y_meta) per the two independent covariates? This is the base capability check: is dual conditioning learnable at all through the FiLM/metadata pathway (the pathway q9/h19 found collapses depth).

## Idea / Hypothesis

Dual conditioning is learnable when the full f_x x f_y matrix is seen in training

## Verifiables

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
- [ ] M1 (end-to-end recon): per-cell NB-mean R2 >= identity-cell ceiling - 0.05 (median across matrix cells); i.e. conditioning adds ~no error beyond the encoder-decoder bottleneck. Weight off-diagonal (h_x != h_y) cells since diagonal cells are near-trivial in denoising-only.
- [ ] M2 (output steering sensitivity): sweeping h_y moves the predicted NB mean in the direction/magnitude f_y dictates, R2(delta_pred, delta_target) >= 0.8 on invertible families, and >> an h_y-ignoring baseline (delta~0). This is the direct counter-test to the q9/h19 metadata collapse.
- [ ] M3 (encoder input invariance): mean within-base transform latent cos-dist / mean between-base cos-dist <= 0.3 (encoder normalizes f_x yet stays discriminative of biology), paired with M1>0 to exclude the trivial-collapse solution.

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

- sandbox/diagnostics/dual_conditioning (impl complete, CPU-gated; awaiting GPU sweep)

## Findings

_(written by the PI/agent when the case is closed)_

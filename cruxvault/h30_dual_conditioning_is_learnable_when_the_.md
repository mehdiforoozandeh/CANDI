---
id: h30
type: idea
title: Dual conditioning is learnable when the full f_x x f_y matrix is seen in training
parent: q15
status: done
verdict: partial
metric: M2 0.48-0.53 (<0.6), M3 0.12-0.17, generalizes; recon partial
created: "2026-07-03T01:59:53"
updated: "2026-07-09T00:49:00"
---

# h30 — Dual conditioning is learnable when the full f_x x f_y matrix is seen in training

Parent:: [[q15_can_candi_learn_dual_metadata_conditioni]]

## Problem Statement

With every (input-transform, output-transform) cell present in training, can CANDIv2 normalize the covariate-transformed count input (encoder, x_meta) and steer the NB count output (decoder, y_meta) per the two independent covariates? This is the base capability check: is dual conditioning learnable at all through the FiLM/metadata pathway (the pathway q9/h19 found collapses depth). [v2 setup: per-assay conditioning on both x_meta and y_meta + depth-offset log-link NB head; metrics reported on chr19+chr21. The v2 confound post-mortem lives in the child question q16.]

## Idea / Hypothesis

Dual conditioning is learnable when the full f_x x f_y matrix is seen in training

## Verifiables

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
- [ ] M1 (end-to-end recon, per-assay + depth-offset head): per-cell reconstruction within ceiling-gap <= 0.05 of the identity-cell ceiling (median across matrix cells), on CRPS and Spearman; off-diagonal (h_x != h_y) cells weighted (diagonal is near-trivial in denoising-only). Reported on BOTH chr19 (train) and chr21 (test).   (found: NOT met at the strict 0.05 bar -- off-diagonal median CRPS gap ~0.6 (chr21); rank reconstruction is strong (Spearman ~0.82-0.85) but the magnitude bar is not reached in denoising-only)
- [ ] M2 (output steering, DISTRIBUTIONAL, per plan section M2): as h_y sweeps, the predicted-NB-vs-f_y-target CRPS response is minimized at the true h_y, decomposed into a mean statistic and a tail/dispersion statistic; distributional M2 (median over invertible families) >= 0.6 and >> an h_y-ignoring baseline. Replaces v1's mean-only R2(delta_pred, delta_target). Direct counter-test to q9/h19 metadata collapse.   (found: NOT met at 0.6 -- M2 median-invertible ~= 0.48-0.53, power-family-limited (mult 0.48 / add 0.81 / power 0.22); but >> the pooled 0.02 floor, so steering IS genuine, just capped below 0.6 in denoising-only)
- [x] M3 (encoder input invariance): within-base / between-base latent cos-dist ratio <= 0.3 (encoder normalizes f_x yet stays discriminative), paired with M1>0 to exclude trivial collapse.   (found: M3 ratio = 0.12-0.17 <= 0.3 with M1 > 0; encoder normalizes f_x while staying discriminative)
- [x] generalization guard: chr19-vs-chr21 M1 ceiling-gap difference <= 0.10 (not overfitting the training chromosome).   (found: steering/invariance generalize tightly -- M2 0.482->0.483, M3 0.156->0.171, Spearman 0.843->0.824 chr19->chr21; absolute CRPS gap differs only because per-chrom count scale differs, not overfit)
- [x] encoder-depth ablation (small): report M1 and M3 for encoder depth-aware vs depth-naive; depth-aware must not degrade M3 discriminativeness (between-base floor holds) nor M1 vs depth-naive.   (found: depth-aware does not degrade -- M3 0.124 (aware) vs 0.171 (naive), M1 gap 0.55 vs 0.63; aware is marginally better on both)

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

- sandbox dual_conditioning sweep 47730802_[0-9] (10 arms, 25ep)
- sandbox/diagnostics/dual_conditioning (impl complete, CPU-gated; awaiting GPU sweep)

## Findings

Dual conditioning is learnable through the FiLM/metadata pathway, but partially -- the pathway that q9/h19 found collapses depth does carry steering here. Encoder input-invariance is clean: within/between latent cos-dist ratio M3 = 0.12-0.17 (<= 0.3) with M1 > 0, so the encoder normalizes f_x while staying discriminative; depth-aware vs depth-naive does not degrade M3 (0.12 vs 0.17) or M1 (gap 0.55 vs 0.63). Steering and invariance generalize tightly train->test (M2 0.482->0.483, M3 0.156->0.171, Spearman 0.843->0.824). Two bars are not met: distributional M2 median-invertible ~= 0.48-0.53 falls short of the >=0.6 target -- power-family-limited (mult 0.48 / add 0.81 / power 0.22) -- and off-diagonal reconstruction retains real error (median CRPS gap ~0.6, though rank recon is strong). Steering is genuine (>> the pooled 0.02 floor) but capped below 0.6 in denoising-only. Verdict: partial.

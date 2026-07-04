---
id: q15
type: question
title: "Can CANDI learn dual metadata conditioning: normalize a covariate-transformed count input and re-emit the target under an independent output covariate, in a controlled synthetic augmentation testbed?"
parent: root
status: open
stale: false
created: "2026-07-03T01:59:20"
updated: "2026-07-03T01:59:20"
---

# q15 — Can CANDI learn dual metadata conditioning: normalize a covariate-transformed count input and re-emit the target under an independent output covariate, in a controlled synthetic augmentation testbed?

Parent:: [[candi]]

## Question

Controlled dual-conditioning testbed. Base x=y=counts_dsf1 on sandbox.h5 (8 assays), CANDIv2 model, chr19 train -> chr21 test, DENOISING-ONLY (no cloze masking), DSF OFF (fixed depth). The ONLY metadata are two per-assay augmentation covariates h_x (input transform) and h_y (output transform): the real 4 metadata rows (depth/assay_id/read_length/run_type) are discarded and x_meta/y_meta are rebuilt as [B,2,F] = (aug_family, aug_param), so any metadata sensitivity is provably from the controllable knob. Transform menu maps count->non-negative integer, applied to both f_x and f_y drawn independently per assay; identity is always included as the M1/M3 reference. Invertible/easy: x-h (multiplicative, depth-rescaling), +h (additive, clamp>=0), power y^h. Non-invertible/hard: binomial thinning (deterministic-seeded), saturating cap min(y,cap), count-log round(c*log1p(y)). Input x'=f_x(counts,h_x) fed as arcsinh count channel; output y'=f_y(counts,h_y) is the NB target trained with NBNLL. Matrix = f_x x f_y; off-diagonal cells (h_x != h_y) are the counterfactual-steering test. Three metrics decompose the pathway: M1 end-to-end reconstruction relative to the identity-cell ceiling; M2 decoder/output steering sensitivity (sweep h_y, does NB mean move as f_y dictates, vs an h_y-ignoring baseline); M3 encoder/input invariance (within-base transform latent cos-dist << between-base cos-dist, paired with M1>0 to exclude collapse). Motivation: covariate_probes established the precondition (covariate fingerprint recoverable from count shape) and q9/h19 found the REAL y_meta pathway collapses depth (DCR~1); this isolates whether the FiLM conditioning mechanism can learn dual input/output covariate steering at all under full control. See-also [[q9]] (FiLM/metadata routing), [[q4]] (depth-controllable supertrack).

## Answer so far

_(interpretation — written by the PI/agent; auto-flagged stale when new evidence lands)_

<!-- crux:ledger:start -->
**4 children** · ideas 0/4 done (supported 0, partial 0, refuted 0, inconclusive 0)

- `h30` [[h30_dual_conditioning_is_learnable_when_the_|Dual conditioning is learnable when the full f_x x f_y matrix is seen in training]] — *staged*
- `h31` [[h31_the_model_composes_to_unseen_f_x_f_y_mat|The model composes to unseen (f_x, f_y) matrix cells]] — *staged*
- `h32` [[h32_invertibility_sets_difficulty_inverting_|Invertibility sets difficulty; inverting f_x (input) is harder than applying f_y (output)]] — *staged*
- `h33` [[h33_param_encoding_normalization_is_load_bea|Param-encoding normalization is load-bearing for reading the augmentation covariate]] — *staged*
<!-- crux:ledger:end -->

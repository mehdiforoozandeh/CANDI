---
id: q15
type: question
title: "Can CANDI learn dual metadata conditioning: normalize a covariate-transformed count input and re-emit the target under an independent output covariate, in a controlled synthetic augmentation testbed?"
parent: root
status: open
stale: true
created: "2026-07-03T01:59:20"
updated: "2026-07-03T01:59:20"
---

# q15 — Can CANDI learn dual metadata conditioning: normalize a covariate-transformed count input and re-emit the target under an independent output covariate, in a controlled synthetic augmentation testbed?

Parent:: [[candi]]

## Question

Controlled dual-conditioning testbed. Base x=y=counts_dsf1 on sandbox.h5 (8 assays), CANDIv2 model, chr19 train -> chr21 test, DENOISING-ONLY (no cloze masking), DSF OFF (fixed depth). The ONLY metadata are two per-assay augmentation covariates h_x (input transform) and h_y (output transform): the real 4 metadata rows (depth/assay_id/read_length/run_type) are discarded and x_meta/y_meta are rebuilt as [B,2,F] = (aug_family, aug_param), so any metadata sensitivity is provably from the controllable knob. Transform menu maps count->non-negative integer, applied to both f_x and f_y drawn independently per assay; identity is always included as the M1/M3 reference. Invertible/easy: x-h (multiplicative, depth-rescaling), +h (additive, clamp>=0), power y^h. Non-invertible/hard: binomial thinning (deterministic-seeded), saturating cap min(y,cap), count-log round(c*log1p(y)). Input x'=f_x(counts,h_x) fed as arcsinh count channel; output y'=f_y(counts,h_y) is the NB target trained with NBNLL. Matrix = f_x x f_y; off-diagonal cells (h_x != h_y) are the counterfactual-steering test. Three metrics decompose the pathway: M1 end-to-end reconstruction relative to the identity-cell ceiling; M2 decoder/output steering sensitivity (sweep h_y, does NB mean move as f_y dictates, vs an h_y-ignoring baseline); M3 encoder/input invariance (within-base transform latent cos-dist << between-base cos-dist, paired with M1>0 to exclude collapse). Motivation: covariate_probes established the precondition (covariate fingerprint recoverable from count shape) and q9/h19 found the REAL y_meta pathway collapses depth (DCR~1); this isolates whether the FiLM conditioning mechanism can learn dual input/output covariate steering at all under full control. See-also [[q9]] (FiLM/metadata routing), [[q4]] (depth-controllable supertrack).

## Answer so far

**Answered on the testbed — dual conditioning IS learnable, composes to unseen combinations, and has a characterized (though not invertibility-clean) difficulty structure; the whole capability map is now run.** (h30, partial) the encoder cleanly normalizes f_x while staying discriminative (M3 0.12-0.17) and output steering is genuine and generalizes train->test (M2 median-invertible ~0.50 >> the pooled 0.02 floor), capped below the 0.6 bar in denoising-only and power-family-limited. (h33, refuted) param-encoding normalization is NOT load-bearing — raw ranks best (none 0.515 > z 0.483 > log 0.448). (h31, partial→positive) **composition is nearly free**: withholding 15-45% of f_x×f_y pairings from training barely moves held-out recon (median M1 gen-gap 0.007-0.08, M2 gap ~0) and the model reads h_y on novel pairings (beats a memorized-f_y' baseline 86-100%) — the factorization of input-understanding from output-steering holds. (h32, partial) **inverting an INPUT transform is genuinely harder than applying an OUTPUT one** (lossy-input M1 gap 0.211 vs lossy-output -0.015; M2 steering 0.442→~0.35 under any lossy input) — but *invertibility* is not the difficulty axis: the encoder handles thin/cap/clog fine and instead struggles with `add` (an additive background shift), and steering lives in the tail for every family. The nested post-mortem **q16 (resolved)** settled *why* v1 read null: across-assay decoder **pooling** (~25x artifact), other candidates ruled out.

Still open: the new child **q18** — do these testbed findings translate to production CANDI with real metadata/covariates? The synthetic capability is established; production translation (linked to q4/h9/h10) is the remaining unknown, so q15 stays open pending q18. Emergent cross-cut: additive/background structure — not information loss — is where the model's difficulty concentrates (h32 `add`, and q17's foreground/background-imbalance question).

<!-- crux:ledger:start -->
**6 children** · ideas 4/4 done (supported 0, partial 3, refuted 1, inconclusive 0) · sub-questions 1/2 resolved

- `h30` [[h30_dual_conditioning_is_learnable_when_the_|Dual conditioning is learnable when the full f_x x f_y matrix is seen in training]] — *done* — verdict **partial**, metric `M2 0.48-0.53 (<0.6), M3 0.12-0.17, generalizes; recon partial`
- `h31` [[h31_the_model_composes_to_unseen_f_x_f_y_mat|The model composes to unseen (f_x, f_y) matrix cells]] — *done* — verdict **partial**, metric `composition near-FREE: M1 gen-gap 0.081/0.007/0.017 @rho .15/.3/.45, M2 gap ~0; beats memorization 100/89/86%; 'unmet' monotone box = null penalty not failure`
- `h32` [[h32_invertibility_sets_difficulty_inverting_|Invertibility sets difficulty; inverting f_x (input) is harder than applying f_y (output)]] — *done* — verdict **partial**, metric `invert-input harder than apply-output STRONG (M1 lossy-in 0.211 vs lossy-out -0.015; M2 0.442->~0.35); but invertibility!=difficulty (add is encoder-hard, not thin/cap/clog); tail-locus universal`
- `h33` [[h33_param_encoding_normalization_is_load_bea|Param-encoding normalization is load-bearing for reading the augmentation covariate]] — *done* — verdict **refuted**, metric `M2 none 0.515 > z 0.483 > log 0.448 (normalization not load-bearing)`
- `q16` _(Q)_ [[q16_was_the_v1_output_steering_null_an_artif|Was the v1 output-steering null an artifact of the testbed confounds, and what makes steering emerge?]] — *resolved*
- `q18` _(Q)_ [[q18_do_the_dual_conditioning_testbed_finding|Do the dual-conditioning testbed findings translate to production CANDI with real metadata and real covariates?]] — *open*
<!-- crux:ledger:end -->

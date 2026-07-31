---
id: q18
type: question
title: Do the dual-conditioning testbed findings translate to production CANDI with real metadata and real covariates?
parent: q15
status: open
stale: true
created: "2026-07-10T12:58:22"
updated: "2026-07-10T12:58:22"
---

# q18 — Do the dual-conditioning testbed findings translate to production CANDI with real metadata and real covariates?

Parent:: [[q15_can_candi_learn_dual_metadata_conditioni]]

## Question

The dual-conditioning testbed (q15/q16) proved a MECHANISM under controlled conditions: per-assay FiLM conditioning can steer the decoder output, and across-assay decoder POOLING is a ~25x artifact that nulls it (q16/h34). Open question: does this translate to production CANDI, which conditions on the real 4 covariates (log2 depth, assay_id, read_length, run_type) rather than a clean synthetic knob? Reasons it may NOT: (1) production model.py is ALREADY per-assay -- pooling is not production's bug; production's own metadata collapse (q4/h8 DCR~1, q9/h19) has a DISTINCT two-part cause (free-mean NB count head + a reconstruct-same-assay task the model can copy off the input), which the testbed removed by construction. (2) the testbed's steering is largely DISTRIBUTIONAL -- the tail tracks strongly (tail-Pearson ~0.98) while the predicted MEAN is steered only modestly (mean-Pearson ~0.5-0.8) and separates weakly from the pooled model (mult 0.48 vs 0.40); a big share of the headline CRPS gap (0.53 vs 0.02) is distribution-shape, not mean, yet production imputation cares about the mean signal. (3) clean deterministic invertible transforms with exact per-position ground truth vs real covariates acting through noisy biology + batch effects (weaker, noisier, possibly non-monotonic mapping). (4) the task was engineered non-copyable (f_x != f_y per assay) to force gradient into the metadata pathway; production's copyable reconstruct-same-assay task may starve it regardless of per-assay wiring. METHOD REQUIREMENT: any production metadata-sensitivity / steering readout MUST include CRPS (a proper distributional score), not only mean-steering metrics (Pearson/R2 of the predicted mean) -- CANDI predicts probability distributions, so CRPS is the correct metric, and a mean-only readout would UNDER-measure the steering that lives in the distribution's shape/tail (exactly what the testbed shows). See-also [[q15_can_candi_learn_dual_metadata_conditioni]], [[q4_can_candi_s_counts_be_made_depth_control]], [[q9]], [[q16_was_the_v1_output_steering_null_an_artif]].

## Answer so far

_(interpretation — written by the PI/agent; auto-flagged stale when new evidence lands)_

<!-- crux:ledger:start -->
**2 children** · ideas 0/0 done (supported 0, partial 0, refuted 0, inconclusive 0) · sub-questions 1/2 resolved

- `q19` _(Q)_ [[q19_can_we_make_dual_conditioning_work_on_re|Can we make dual conditioning work on real CANDI sandbox data before production?]] — *resolved*
- `q20` _(Q)_ [[q20_how_should_candi_architecture_and_traini|How should CANDI architecture and training condition on experimental metadata to improve imputation magnitude AND genuine metadata use]] — *open*
<!-- crux:ledger:end -->

---
id: q4
type: question
title: Can CANDI's counts be made depth-controllable to denoise toward a canonical "supertrack"?
parent: root
status: open
stale: true
created: "2026-07-02T20:37:11"
updated: "2026-08-01T18:18:11"
---

# q4 — Can CANDI's counts be made depth-controllable to denoise toward a canonical "supertrack"?

Parent:: [[candi]]
Literature:: [[wiki/digest-depth-as-covariate-vs-divisor]], [[wiki/sequencing-depth-and-coverage]], [[wiki/count-distributions-for-sequencing-data]]

## Question

Can CANDI's counts be made depth-controllable to denoise toward a canonical "supertrack"?

## Answer so far

**Yes in principle, via a structural head change — and the dual-conditioning testbed (q15/q16) has now isolated the full mechanism.** The chain: (h8, refuted) the free-mean NB head is depth-blind — asking for higher depth does nothing, DCR~1; (h9, partial) a depth-centered size-factor reparam `mu = 2^(d-center)*exp(eta)` restores DCR~4 from epoch 0 on the 8-assay diagnostic while the raw `2^d` offset fails — production-scale confirmation still open; (h10, idea) per-assay independent DSF sampling to make depth a *necessary* training signal.

**New (from q16/h34-h36):** the production collapse is now understood as **two independent causes, and per-assay conditioning is NOT one of them** — production `model.py` is already per-assay (per-assay `FiLMLayer` + grouped deconv, no `meta.mean`); it is the *sandbox candi_v2* rewrite that regressed to across-assay pooling, and q16/h34 proves pooling is what nulls steering (~25x, M2 0.50→0.02). Production's own DCR~1 is instead caused by (1) the **free-mean head** (h8's finding — no depth anchor) and (2) a **reconstruct-same-assay task** where the target depth is readable off the input, so the metadata pathway gets no gradient. The testbed steers precisely because it removed both — a **depth-centered offset head** (structural DCR guarantee, corroborating h9) **and a non-copyable dual task** with f_x≠f_y (which supplies the missing gradient, the rationale for h10) — while keeping per-assay. Actionable: the production fix is **h9 (offset head) + h10 (per-assay DSF)**, not any change to conditioning topology. See [[q16_was_the_v1_output_steering_null_an_artif]] for the controlled evidence.

<!-- crux:ledger:start -->
**3 children** · ideas 2/3 done (supported 0, partial 1, refuted 1, inconclusive 0)

- `h8` [[h8_the_v1_nb_count_head_is_depth_controllab|The v1 NB count head is depth-controllable via the output prompt]] — *done* — verdict **refuted**
- `h9` [[h9_a_depth_centered_size_factor_reparam_res|A depth-centered size-factor reparam restores depth sensitivity to DCR ~ 4]] — *done* — verdict **partial**
- `h10` [[h10_per_assay_independent_dsf_sampling_makes|Per-assay independent DSF sampling makes depth a necessary signal at production scale]] — *idea*
<!-- crux:ledger:end -->

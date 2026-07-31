---
id: q16
type: question
title: Was the v1 output-steering null an artifact of the testbed confounds, and what makes steering emerge?
parent: q15
status: resolved
stale: false
created: "2026-07-08T12:36:20"
updated: "2026-07-09T00:54:14"
---

# q16 — Was the v1 output-steering null an artifact of the testbed confounds, and what makes steering emerge?

Parent:: [[q15_can_candi_learn_dual_metadata_conditioni]]

## Question

v1 measured output-steering (M2) near zero. Four candidate causes are tested by this question's hypotheses: across-assay metadata pooling (h34), the input denoising-shortcut (h35), the count-head offset/parameterization (h36), and whole-chromosome background domination where the steering signal lives in the sparse foreground (h37). Nested under q15; this is the v2 mechanism/confound post-mortem, distinct from q15's h30-h33 capability map.

## Answer so far

**Yes — the v1 output-steering null was a testbed artifact, and its single cause is across-assay metadata POOLING in the decoder.** Of the four candidates, one is the culprit and the other three are ruled out as the cause:
- **Pooling (h34, supported):** the faithful v1 reproduction (`pool_meta`, mean over the assay axis) collapses M2 to 0.022 while per-assay conditioning reaches ~0.50 — a ~25x lift (+0.48), with pooling also hurting reconstruction. This is the first controlled, ground-truth isolation of per-assay-vs-pooled decoder conditioning as the causal lever. Refinement: it is *pooling* specifically, not uniform-per-batch *sampling* (that arm keeps M2=0.53).
- **Input-denoising shortcut (h35, supported but not fatal):** real and dose-dependent (h_y-reliance rises where the input cannot approximate the target), yet steering is strong in the isolated forced-identity regime (M2=0.62) — the shortcut co-exists with genuine steering, it does not null it.
- **Count-head offset (h36, supported=unconditional):** steering is present offset-OFF too (0.46 ~ 0.48) — the depth-offset is not required for relative steering (it buys calibrated absolute depth, the h9 DCR fix, separately).
- **Background domination (h37, partial):** does not broadly suppress steering; it only shapes *which* per-family signal reads out in the aggregate (power foreground-localized +0.12, add background-inflated -0.35).

**Bearing on production / q9-h19 / q4 (the metadata-collapse origin).** Crucially, *pooling is not production's problem* — production `model.py` is already per-assay (per-assay `FiLMLayer` + grouped deconv, no `meta.mean`). The pooling regression lives in the **sandbox candi_v2 rewrite** (`decoder.py` `meta.mean(dim=1)`), which was derived from JEPA, not production, and propagated to all autoresearch (june3, menu x7). So the q9/h19 "y_meta collapses depth" observation was measured on candi_v2's *pooled* decoder. Production's own supertrack collapse (q4/h8, DCR~1) has a **different, two-part cause** that this testbed also isolates: (1) a **free-mean NB count head** (`softplus(Wx)`, no depth anchor — the h8-refuted / h9 "raw offset fails" config), and (2) a **reconstruct-same-assay task** where the target depth/identity is readable off the input, so the metadata pathway never gets gradient. The testbed steers because it removed *both* (depth-centered offset head + a non-copyable dual task with f_x != f_y) **and kept per-assay**. Net: per-assay is necessary but not sufficient; production already has per-assay, so its fixes are the h9 offset head + h10 per-assay DSF, not "add per-assay conditioning."

Resolved on the mechanism; two follow-ups tracked elsewhere: production-scale confirmation of the offset+per-assay-DSF fix (h9/h10, under q4) and composition/invertibility (h31/h32, phase-2c, under q15).

<!-- crux:ledger:start -->
**4 children** · ideas 4/4 done (supported 3, partial 1, refuted 0, inconclusive 0)

- `h34` [[h34_per_assay_conditioning_is_necessary_for_|Per-assay conditioning is necessary for output-steering; the v1 null was an across-assay pooling artifact]] — *done* — verdict **supported**, metric `per-assay M2 ~0.50 vs pooled 0.02 (chr21); lift +0.48`
- `h35` [[h35_output_steering_is_achievable_in_the_iso|Output-steering is achievable in the isolated regime (positive control), and h_y-reliance falls where the input can approximate the target (shortcut)]] — *done* — verdict **supported**, metric `forced-identity M2 0.62; reliance power 0.19<mult 0.31<add 7.59`
- `h36` [[h36_output_steering_once_present_is_either_u|Output-steering, once present, is either unconditional or requires the depth-offset preconditioning]] — *done* — verdict **supported**, metric `offset-off M2 0.46 ~ offset-on 0.48 -> unconditional`
- `h37` [[h37_whole_chromosome_background_domination_s|Whole-chromosome background domination suppresses steering: the metadata signal lives in the sparse foreground the per-position loss under-weights]] — *done* — verdict **partial**, metric `fg-agg gap power +0.12 / add -0.35 (sign-specific, magnitudes <0.2)`
<!-- crux:ledger:end -->

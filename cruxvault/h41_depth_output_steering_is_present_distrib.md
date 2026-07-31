---
id: h41
type: idea
title: Depth output-steering is present, distributional, and offset-independent (M2, depth)
parent: q19
status: done
verdict: partial
metric: depth eta-slope ~0 offset-on (arithmetic) vs 0.88 offset-off; dir 0.43-0.57 CI-excl-0
created: "2026-07-15T23:35:46"
updated: "2026-07-16T09:20:13"
---

# h41 — Depth output-steering is present, distributional, and offset-independent (M2, depth)

Parent:: [[q19_can_we_make_dual_conditioning_work_on_re]]

## Problem Statement

Depth is the augmentation-rich, continuous covariate with a materializable per-position counterfactual (per-assay independent DSF). The depth-offset log-link head moves the NB mean by 2^Delta arithmetically, so a genuine steering claim must rest on offset-INDEPENDENT signal (residual eta, tail/dispersion) and a shuffled-prompt null, not on DCR/mean alone.

## Idea / Hypothesis

Depth output-steering is present, distributional, and offset-independent (M2, depth)

## Verifiables

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
- [x] responsiveness: CRPS(pred@true-depth vs pred@wrong-depth) > 0 and increases monotonically away from the true depth (per-assay depth sweep over meta_dsf levels within achievable support) (found: CRPS-vs-told-depth min at true depth in 76-77% of targets; prediction moves with told-depth)
- [x] direction: CRPS(true-depth prompt, GT) < CRPS(wrong-depth prompt, GT), bootstrap CI excludes 0, on foreground/high-count positions (found: dir 0.432/0.568, bootstrap CI excludes 0, foreground fg_frac=0.02)
- [ ] offset-independent: residual eta and/or tail-dispersion statistic tracks true depth (not merely the 2^Delta offset arithmetic) (UNMET on the winning recipe: median eta-slope ≈ 0 (-0.000/0.000) — depth response is the 2^(d-center) offset arithmetic, NOT learned. Offset-OFF control offoff_s0 shows the learned pathway EXISTS: eta-slope 0.880)
- [x] null: depth-shuffle prompt collapses the direction effect to <= 0.05 of the true-prompt effect (found: shuffled-prompt null delta 0.000)

## Planned Intervention

On the trained h40 model, for each eval unit (biosample × target assay), **sweep the told `y_meta` depth** over the achievable `meta_dsf` levels {1,2,4,8} (depths `d_k`), holding the applied input fixed. Run on both the masked imputation targets and unmasked DSF tracks (depth has a materializable counterfactual via `counts_dsf{k}`).
- **Responsiveness**: pairwise `ΔCRPS(pred@d_k, pred@d_true) > 0`, monotone increasing as `|d_k − d_true|` grows.
- **Direction**: `CRPS(pred@true-depth, GT@true-depth) < CRPS(pred@wrong-depth, GT)`; bootstrap CI over windows/positions excludes 0; **foreground** positions (top-`fg_frac` by base count).
- **Offset-independent (the honest lever)**: regress `eta` and an upper-quantile / log-dispersion statistic on the told depth; require they track `d_true` (significant slope) — i.e. steering survives with the `2^(d−center)` offset arithmetic partialled out. This is the guard against "DCR≈4 for free."
- **Null**: shuffle the depth prompt across positions/assays → direction effect ≤ 0.05 of the true-prompt effect.

Reuse the **primitives** `nb_crps`, `_steering_index` (NOT the testbed's `eval_M2`, which is bound to synthetic transforms — see q19 §Implementation guide); drive the sweep off `meta_dsf`/`counts_dsf`. Expected the **strongest** of the M2 axes (augmentation-rich, continuous).

**Architectural prerequisite:** for the offset-independent `eta` response to be measurable at all, **depth (real meta row 0) must be a decoder-FiLM covariate**, not offset-only. If depth flows only to the `2^(d−center)` offset, `eta` cannot respond to told-depth and this verifiable is vacuous. The `RealMetaEmbedder` (q19 §Implementation guide, new-code item 1) feeds depth to the FiLM; the offset reads depth separately.

**Tests (pre-GPU) — `tests/test_metrics_real.py::test_M2_depth`** (synthetic-model controls, see q19 §Validation):
- **offset-only** control → CRPS-curve **min at true depth** + responsiveness > 0 **but eta-slope ≈ 0** (the readout correctly flags "arithmetic, not learned" — the key confound guard).
- **eta = k·depth** control → offset-independent eta-slope ≈ k (positive control for the regression).
- **depth-ignoring** control → flat curve, responsiveness ≈ 0 (negative control).
- **shuffle-prompt null** collapses the direction effect on a responsive synthetic model.
- sweep grid stays within achievable support `{base_d − log2(dsf)}`; foreground mask = top `fg_frac=0.02` by GT count; bootstrap CI excludes 0 on a separated pair, includes 0 on identical arrays.

## Run Links

- SLURM 49274497 (dual_conditioning_real sweep, sampled, EP=25)
- SLURM 49277527 (dual_conditioning_real FULL-COVERAGE sweep — the definitive run)

## Findings

**Partial — depth output-steering is present and distributional, but on the winning recipe it is ARITHMETIC, not learned.** With the offset head on (main_s0/s1), told-depth moves the predicted NB (responsiveness present; CRPS-vs-told-depth minimum at the true depth in 76-77% of targets), the CRPS direction excludes 0 on foreground positions (0.43/0.57), and the depth-shuffle null collapses to 0 — the mean/CRPS steering criteria pass. **But the honest offset-independent lever is FLAT: median eta-slope ≈ 0 (−0.000 / 0.000).** So the entire depth response is the hardwired `2^(d−center)` offset arithmetic, not a learned FiLM/eta response — exactly the confound this hypothesis was pre-registered to catch ("DCR≈4 for free"). The offset-OFF control (offoff_s0_full) proves the learned pathway *exists* but is *suppressed* by the offset: with no arithmetic shortcut, eta learns to carry depth (**slope 0.880**, offset-independent). Verdict partial: direction + responsiveness + null met, offset-independent unmet on the winning recipe. The mechanism — the depth-offset head taking the free lunch and starving the learned metadata gradient — is empirically confirmed; the learned-vs-arithmetic tradeoff is formalized in the follow-up [[h45_removing_the_depth_offset_head_recovers_|h45]].

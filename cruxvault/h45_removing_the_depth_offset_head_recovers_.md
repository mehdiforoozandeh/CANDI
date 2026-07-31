---
id: h45
type: idea
title: Removing the depth-offset head recovers learned steering at a scale-calibration cost a hybrid can recover
parent: q19
status: done
verdict: refuted
metric: "Pareto premise refuted by h47: no hybrid needed; wd=0 gets magnitude AND steering. offset-off 'learned steering' was an eta_slope artifact (true total slope 0.775<1.000); run_type CI crosses 0 under clustering (B1)"
created: "2026-07-16T09:20:34"
updated: "2026-07-29T02:08:40"
---

# h45 — Removing the depth-offset head recovers learned steering at a scale-calibration cost a hybrid can recover

Parent:: [[q19_can_we_make_dual_conditioning_work_on_re]]

> **SUPERSEDED (2026-07-24).** [[h47_the_offset_on_steering_null_is_a_weight_|h47]] showed the offset-ON "steering null" this node was built around was a **weight-decay artifact** (wd=0 revives the pathway AND beats the anchor on magnitude+shape), so the offset-off/hybrid Pareto framing here is retired. The live question is now [[q20_how_should_candi_architecture_and_traini|q20]]; this node's "learned-scale-head" idea lives on as [[h49_read_length_as_a_fixed_coefficient_physi|h49]] (fixed read_length exposure) + [[h50_an_explicit_per_assay_output_factor_loca|h50]] (learned per-assay scale).

## Problem Statement

The depth-offset head does **two coupled jobs**: (1) it supplies the correct depth→count scale via `2^(d−center)` — genuinely useful (good CRPS/MSE) — and (2) by fitting the mean for free it **zeroes the gradient into `η`**, so the metadata path is never forced to learn scale (no learned steering: [[h41_depth_output_steering_is_present_distrib|h41]]/[[h42_counterfactual_prompt_flip_true_run_type|h42]] found η-slope ~0 and run_type ignored with offset ON). offset-OFF removes the shortcut and recovers genuine learned steering (η-slope 0.88, run_type dir 0.69) but forfeits job (1). [[h46_the_offset_off_imputation_gap_is_scale_m|h46]] localized the resulting "imputation cost" to **absolute-scale calibration, not biology**: within-assay shape ties offset-on (macro Spearman 0.465 ≈ 0.505, offset-off wins 6/8) while magnitude fails (macro CRPS 1.50→1.90, beats the per-assay marginal on 3/8 vs offset-on's 8/8). So neither anchor is production-ready — offset-ON = scale✓ / steering✗; offset-OFF = steering✓ / scale✗. **h45 asks whether a hybrid can DECOUPLE the two roles and get BOTH.**

## Idea / Hypothesis

Removing the depth-offset recovers genuinely-learned depth and run_type steering; its cost (per [[h46_the_offset_off_imputation_gap_is_scale_m|h46]]) is absolute-scale calibration, not lost biology — and a **hybrid** (offset warmup→anneal-off / attenuated offset / learned metadata-driven scale head) can **restore magnitude while keeping the learned steering**, achieving what neither anchor does.

## Verifiables

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
- [ ] offset-off recovers offset-independent **depth steering**: median η-slope ≥ 0.5 vs ~0 with offset on (established: offoff_s0_full 0.88 — [[h41_depth_output_steering_is_present_distrib|h41]]) — UNMET (refuted by [[h47_the_offset_on_steering_null_is_a_weight_|h47]]): the η-slope 0.88 was a MEASUREMENT ARTIFACT. eta_slope scores the offset-FREE residual, which is ~0 by construction under a correct offset; on the CORRECT total told-depth slope, offset-OFF is 0.775 — WORSE than offset-ON's 1.000. offset-off did not "recover" better depth steering; it lost the exact arithmetic.
- [ ] offset-off recovers real **run_type steering**: direction-frac ≥ 0.6, bootstrap CI excl 0 on BOTH single & paired (established: 0.69, resp 1.83 — [[h42_counterfactual_prompt_flip_true_run_type|h42]]) — UNMET (refuted): the 0.69 direction-frac does NOT survive target-clustered CIs (audit S4: [+0.066,+0.074] → [−0.022,+0.181], crosses 0), and run_type is analytically UNIDENTIFIABLE on this panel (H(run_type|assay,read_length)=0.000 bits, bound B1). The "recovered run_type steering" was position-level overconfidence, not signal.
- [ ] the offset-off cost is **SCALE-calibration, not shape**: within-assay shape ties (macro Spearman gap ≤ seed noise, offset-off wins ≥6/8) while magnitude fails (macro CRPS +~0.4; beats per-assay marginal on ≤3/8 vs offset-on 8/8) (established by [[h46_the_offset_off_imputation_gap_is_scale_m|h46]]) — UNMET as h45 framed it (the bare scale-not-shape fact still stands in [[h46_the_offset_off_imputation_gap_is_scale_m|h46]], but h45's ACTIONABLE claim — that this is a Pareto cost a HYBRID must recover — is refuted: h47 shows the "cost" is a weight-decay artifact, removed for free by wd=0, not a frontier to hybridize).
- [ ] **HYBRID recovers BOTH** (the open experiment): retains steering (η-slope ≥ 0.7, run_type dir ≥ 0.6 CI-excl-0) **AND** restores magnitude (per-assay macro CRPS ≤ ~1.57 = within 5% of the 1.495 offset-on anchor; beats per-assay marginal → 8/8) **AND** keeps shape (macro Spearman ≥ 0.50) — i.e. achieves what neither anchor does (ON scale✓/steer✗, OFF steer✓/scale✗) — UNMET / OBVIATED: no hybrid was ever needed. **SUPERSEDED 2026-07-28 (h48/F2), retained as the pre-h48 record:** ~~h47 (offset-ON, wd=0) achieves BOTH — magnitude (macro CRPS 1.341 < 1.495) AND functional assay steering (Δη 0.833) — with no anneal, no attenuation, no hybrid. The premise that these trade off along a Pareto is false.~~ **Corrected:** the Δη **0.833 is a MISSING-sentinel ARTIFACT** ([[h48_h0_fix_the_broken_q19_instruments_and_re|h48]]/F2) — the whole-row `assay_id` permute slid the MISSING(−1) sentinel across UNAVAILABLE slots. Sentinel-free real→real for `wd0_on` is **0.0023** (H48:L92), **43× BELOW** h47's own ≥0.10 functional bar, against **4.1772** (`offoff`) / **9.7144** (`wd0_off`) on the identical probe. The bare `macro CRPS ≤ 1.341` bar is likewise retired; the successor axis is oracle-scale-decomposed capability, on which `wd0_on` reads **1.3077** ± the ~0.09 target-clustered noise floor (H48:L269-270). So h47 is shown to achieve the **magnitude leg only**, not BOTH.

## Planned Intervention

Same `sandbox/diagnostics/dual_conditioning_real/` harness + the per-assay scorecard from [[h46_the_offset_off_imputation_gap_is_scale_m|h46]] (macro CRPS + beats-marginal + `median_mu`/`median_target` scale-bias), full-coverage, seed 0, scored against the two **fixed anchors**: `main_s0` (offset ON = scale✓/steer✗) and `offoff_s0` (offset OFF = steer✓/scale✗). Decouple the offset's two roles:

- **Primary — offset warmup → anneal-off.** Train offset ON for the first K epochs (stable recon; η learns assay-level + shape against a correct-scale target), then linearly anneal the offset coefficient β: 1→0 over the next epochs, progressively handing the `(d−center)` residual to η while shape is already in place. Sweep the anneal window.
- **Secondary — α-attenuated offset.** Fix β ∈ {0.25, 0.5, 0.75}: the model gets part of the scale for free and must learn the rest — traces the steering↔scale trade curve.
- **Architecture alt — learned metadata-driven scale head.** Replace the hardwired `2^(d−center)` with a small MLP on the depth embedding (init ≈ identity) — a learned pathway that is magnitude-correct AND metadata-responsive by construction (gradient-trained, so it need not collapse back to pure arithmetic).

Score every arm on BOTH the per-assay scorecard (magnitude) and the M2 steering readouts (η-slope, run_type flip). Success = the HYBRID verifiable bar. Checkpoint saving is now on (`train_and_eval(ckpt_path=…)`), so eval-only iteration is cheap. **Needs a new run — PI go before launch (crux leash + `--gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1`).**

## Run Links

_(none yet)_

## Findings

**REFUTED — superseded by [[h47_the_offset_on_steering_null_is_a_weight_|h47]] and the q19 metadata audit.** h45's central premise was that offset-ON scale✓/steer✗ and offset-OFF steer✓/scale✗ define a real Pareto that only a *hybrid* (anneal / attenuate / learned-scale-head) could cross. Every leg of that premise fell:
1. offset-OFF's "recovered learned depth steering" (η-slope 0.88) was a **measurement artifact** — `eta_slope` scores the offset-free residual, ~0 by construction under a correct offset; on the correct *total* told-depth slope offset-OFF is **0.775, worse** than offset-ON's 1.000.
2. offset-OFF's "run_type steering" (0.69) **does not survive target-clustered CIs** (crosses 0) and run_type is analytically unidentifiable on this panel (B1).
3. The offset-OFF "scale cost" is not a Pareto to hybridize — [[h47_the_offset_on_steering_null_is_a_weight_|h47]] showed it is a **weight-decay artifact**: **SUPERSEDED 2026-07-28 (h48/F2), retained as the pre-h48 record:** ~~`weight_decay=0` on the offset-ON model beats the anchor on magnitude AND recovers functional assay steering (Δη 0.833), **with no hybrid**.~~ **Corrected:** `weight_decay=0` on the offset-ON model beats the anchor on magnitude, but it does **not** recover functional assay steering — the Δη 0.833 was a MISSING-sentinel artifact; sentinel-free real→real `wd0_on` is **0.0023** (H48:L92), 43× below h47's own ≥0.10 bar, versus 4.1772 (`offoff`) / 9.7144 (`wd0_off`). The magnitude axis is now oracle-scale-decomposed capability (`wd0_on` **1.3077** ± ~0.09 target-clustered noise floor, H48:L269-270), not the retired bare `macro CRPS ≤ 1.341`.

So the hybrid experiment was never needed. The live line of work is now [[q20_how_should_candi_architecture_and_traini|q20]]; the one genuinely useful idea seeded here — a learned/explicit scale pathway — survives as [[h49_read_length_as_a_fixed_coefficient_physi|h49]] and [[h50_an_explicit_per_assay_output_factor_loca|h50]]. (h45's V3 restated a fact from [[h46_the_offset_off_imputation_gap_is_scale_m|h46]] that still stands on its own; it is only the h45 *hybrid framing* that is refuted.)

**FLAG 2026-07-28 — h45's refutation basis is now under review:** the "h47 achieves BOTH" leg of this refutation rested on the Δη 0.833 that [[h48_h0_fix_the_broken_q19_instruments_and_re|h48]]/F2 retracted as a MISSING-sentinel artifact (sentinel-free `wd0_on` **0.0023**, H48:L92), so whether the Pareto premise still falls is a live scientific question for the PI; the verdict, status and metric recorded here are deliberately left unchanged pending that call.

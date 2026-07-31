---
id: h47
type: idea
title: "The offset-ON steering null is a weight-decay artifact: removing weight decay revives the decoder metadata pathway at no magnitude cost"
parent: q19
status: done
verdict: partial
metric: "M1 stands (macro CRPS 1.495->1.341, imp-Sp 0.533->0.637, ECE 0.062->0.053); on oracle-scale-decomposed capability wd0_on 1.3077 is best of four, and offoff-wd0_on +0.093 [+0.004,+0.217] is the ONLY pairwise lead surviving target clustering (four-arm reordering NOT established); embedder weights alive 6e-41->2.79 BUT at random-init statistics (never destroyed, never trained); V2 UNMET: sentinel-free assay steering d_eta 0.0023, 43x below its own 0.10 bar (h48/F2)"
created: "2026-07-23T19:46:33"
updated: "2026-07-28T18:54:35"
---

# h47 — The offset-ON steering null is a weight-decay artifact: removing weight decay revives the decoder metadata pathway at no magnitude cost

Parent:: [[q19_can_we_make_dual_conditioning_work_on_re]]

## Problem Statement

q19 concluded an offset-ON/offset-OFF Pareto: offset ON = magnitude/CRPS good but 'no learned steering'; offset OFF = learned steering at a magnitude cost. Checkpoint forensics overturn the premise. Under offset ON the decoder's metadata embedder is ANNIHILATED (assay_embedding absmax 6.30e-41, runtype_embedding 6.23e-41, depth_proj 1.56e-04) and the trained decoder is BIT-EXACTLY blind to assay identity (permuting all 8 assay IDs changes eta by exactly 0.000e+00) and to run_type. The mechanism is zero-task-gradient x coupled-L2: adaLN-zero makes dL/d(memb) exactly 0 at step 0, and thereafter the task gradient (~1e-6 to 3e-7) stays 1-2 orders BELOW wd*|w| (~3e-5), so torch.optim.Adam's L2-coupled decay wins for all ~47,625 updates. Corroboration: the exact-zero run_type null appears ONLY in full-coverage runs; the three short (steps_per_epoch=300) offset-ON runs show live responses (frac_direction 0.310/0.707/0.743). So the comparison was a healthy model vs a model whose conditioning input was DELETED by the optimizer -- not two working models on a frontier. This is the cheapest decisive discriminator between 'real Pareto' and 'optimizer artifact', and it has never been run.

## Idea / Hypothesis

The offset-ON steering null is a weight-decay artifact: removing weight decay revives the decoder metadata pathway at no magnitude cost

## Verifiables

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
- [x] PATHWAY SURVIVES: in the offset-ON wd=0 checkpoint, decoder.meta_embedding assay_embedding.weight and runtype_embedding.weight absmax >= 1e-2 (anchors at wd=1e-4: 6.30e-41 and 6.23e-41); depth_proj.weight absmax >= 1e-2 (anchor 1.56e-04) — MET (found: assay 2.79, runtype 3.32, depth_proj 1.04; the 6e-41 denormals are gone. z-independent weight check, faithful.) **AMENDED 2026-07-28 (h48/F2, the report's "V2a"):** still MET at the weight level — `main`'s table is annihilated (~1e-40) and bit-exactly blind from the fusion's first `Linear` onward, while `wd0_on`'s is full-rank and injective (effective rank 6.8/8). But **"revived" is the wrong word**: `wd0_on`'s assay table sits at **random-init statistics** (element std 0.94 vs 0.97 for a fresh N(0,1) table; cosine **0.988** against `wd0_off`'s independently-trained table) — the table was **never destroyed and never trained**.
- [ ] CONDITIONING IS FUNCTIONAL, not merely non-zero: decoder-only probe on a real batch with z fixed -- permuting assay_id gives max|d_eta| >= 0.10 and a run_type flip gives max|d_eta| >= 0.05 (anchors: BOTH exactly 0.000e+00); realised (gamma,beta) effective rank >= 2.0 (anchor 1.016, offset-OFF 2.451) — **SUPERSEDED 2026-07-28 (h48/F2), retained as the pre-h48 record:** ~~MET for assay (UPGRADED from could-not-evaluate by the q19 gap-fill REAL-z re-probe: assay-permute max|d_eta| = 0.833 >> 0.10, vs the synthetic-z 2.6e-4 that undercounted 14x; (gamma,beta) eff-rank ~2). The synthetic-z probe I first ran was unfaithful (it undercounted the offset-OFF total depth slope 14x: 0.075 vs real ~1.06). run_type stays weak (real-z d_eta 0.003 < 0.05) but that is the B1 DATA-identifiability bound (H(run_type|assay,read_length)=0.000 bits), NOT a model failure — carved out and now owned by [[h57_a_re_selected_biosample_panel_is_the_onl|h57]]. So wd=0 delivered FUNCTIONAL assay conditioning, not merely non-zero weights. Gap-fill also explains the modest response: assay identity is routed structurally through the encoder z (~93% of z independent of x_meta assay_id), so the decoder assay-FiLM is functional-but-redundant.~~ **UNMET at the function level, AMENDED 2026-07-28 (h48/F2).** The 0.833 that upgraded this box is a **MISSING-sentinel ARTIFACT**: the protocol permuted the whole `assay_id` row across all 8 slots, sliding the MISSING(−1) sentinel onto and off **unavailable** slots whose prompt columns are fully (−1,−1,−1,−1) — 99.95% of the measured effect is sentinel. Sentinel-free, at true full coverage (608 units × all 7 shifts × all positions), `wd0_on`'s real→real assay-permute max|Δη| is **0.0023** — **43× below this box's own ≥0.10 bar** — against `offoff` **4.1772** and `wd0_off` **9.7144** on the identical probe (positive control fires at 1816–4224×; for both offset-OFF arms the permuted and real→real values agree bit-for-bit). The same contaminated protocol re-run at true full coverage reads **4.6686** (H48:L92); the report does not attribute h47's originally-recorded 0.833 to prefix truncation, so do not. Mechanism: not the embedder but **fusion-LayerNorm scale** — `wd0_on`'s pre-LN activation norm is 396 vs 8.9, so LN divides the assay perturbation by ~70 instead of ~1.6 (post-LN 0.0091 vs 1.41), and `film_proj` attenuates a further ~18×; end-to-end deficit ~4 orders. This is **not "refuted"** (the weight-level claim in V1 is real and the literal h48 inequality holds, 0.0023 > 0.0000) and **not "inconclusive"** (full coverage, three concordant sentinel-free probes, a positive control and a mechanism) — the reading that keeps both facts is **unmet-for-assay**. The earlier gloss "functional-but-redundant" is withdrawn: the decoder assay-FiLM is near-silent on the prompt pathway. **Confound:** `assay_id ≡ slot index` here and the trunk emits a per-slot channel block, so this bounds the **prompt pathway**, not assay-awareness. run_type is unchanged (still the B1 data bound, owned by h57).
- [x] MAGNITUDE IS NOT PAID FOR IT: per-assay macro CRPS <= 1.57 (within 5% of the 1.495 offset-ON anchor) AND pooled imp-count Spearman >= 0.50 (anchor main_s0 0.533) — MET AND EXCEEDED (found: wd0_ON macro CRPS 1.341 < 1.495 anchor; imp-Spearman 0.637 > 0.533; macro-Spearman 0.565 > 0.505; ECE 0.053 < 0.062. Removing wd IMPROVED magnitude, not merely preserved it.)
- [x] DEPTH ARITHMETIC PRESERVED: total told-depth response |d<log2_mu>/d(depth) - 1| <= 0.10 (anchors: offset-ON 1.0000, offset-OFF 0.775). NOTE: eta_slope is deliberately NOT a verifiable here -- under a correct offset the offset-free residual slope is ~0 BY CONSTRUCTION, and scoring it was the h41/h45 measurement error — MET (found: total slope 1.0000; median_eta_slope 1.9e-5 on real data confirms eta stays depth-flat, so the slope-1 is faithful for the ON arm regardless of z.)
- [x] REVIVAL IS SPECIFIC, not general regularization: in the 2x2 control arm (offset OFF x wd=0) the magnitude gap does NOT also close (macro CRPS still > 1.57). If it does close, the mechanism is general regularization rather than pathway revival and this box is unmet — MET (found: wd0_OFF macro CRPS 2.056 > 1.57, slightly WORSE than the offoff wd=1e-4 anchor 1.902; the gain is specific to reviving the offset-ON pathway.)

## Planned Intervention

2x2 factorial vs the two recorded wd=1e-4 anchors (`main_s0_perassay` = offset ON, `offoff_s0_perassay` = offset OFF): add two arms that change ONLY `--weight-decay 0.0`, everything else identical (seed 0, full-coverage, 25 ep, uniform DSF). `wd0_on_s0` is the test (does removing decay revive the offset-ON pathway?); `wd0_off_s0` is the control (does wd=0 just help everything, or is the gain offset-ON-specific?). Score: (M1) macro/pooled imp CRPS + Spearman + beats-marginal + ECE from the results JSON; (M2) run_type frac_direction / responsiveness / natural_variance_insufficient; (checkpoint probe, CPU) decoder.meta_embedding absmax per field + decoder-only d_eta under assay-permute / run_type-flip + total told-depth slope. Job: `sandbox/diagnostics/dual_conditioning_real/jobs/wd0.sh` (`--gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1`).

## Run Links

- SLURM 50372613 (array 0-1) — sandbox/diagnostics/dual_conditioning_real/jobs/wd0.sh; arms wd0_on_s0 / wd0_off_s0

## Findings

**Core claim SUPPORTED, and stronger than "no cost": removing weight decay makes the offset-ON model strictly BETTER, and the offset-OFF control confirms the gain is pathway-specific. The ambitious rider (revival yields strong functional assay/run_type steering) is NOT yet shown — the response is alive-but-weak, and the probe that was meant to measure it is unfaithful. Proposed verdict: SUPPORTED on the core (V1/V3/V4/V5), with the functional-steering sub-claim (V2) INCONCLUSIVE pending a real-z re-probe. PI to close.**

### The 2x2 (SLURM 50372613, seed 0, full-coverage, 25 ep)

| arm | imp Sp (pool) | imp CRPS | macro Sp | **macro CRPS** | beats-marg | ECE |
|---|---|---|---|---|---|---|
| offset-ON, wd=1e-4 *(anchor `main_s0`)* | 0.533 | 1.617 | 0.505 | 1.495 | 8/8 | 0.062 |
| offset-OFF, wd=1e-4 *(anchor `offoff_s0`)* | 0.401 | 2.060 | 0.465 | 1.902 | 3/8 | 0.097 |
| **offset-ON, wd=0 (`wd0_on_s0`)** | **0.637** | **1.448** | **0.565** | **1.341** | 8/8 | 0.053 |
| offset-OFF, wd=0 (`wd0_off_s0`) | 0.380 | 2.305 | 0.464 | 2.056 | 5/8 | 0.078 |

**wd0_ON beats the offset-ON anchor on every M1 axis** — magnitude (macro CRPS 1.341 < 1.495) AND shape (macro Sp 0.565 > 0.505) AND calibration (ECE 0.053 < 0.062). This is not "recover steering at a magnitude cost"; it is strictly better imputation, which is the PI's thesis: a model that keeps its metadata pathway alive should win, not trade off. The offset-OFF control does NOT improve (2.056 ≥ 1.902), so the gain is specific to reviving the offset-ON pathway, not a general effect of dropping decay.

### Mechanism confirmed (checkpoint forensics, wd0_on vs main anchor)

Decoder `meta_embedding` absmax: assay_embedding **6.30e-41 → 2.79**, runtype_embedding **6.23e-41 → 3.32**, depth_proj **1.56e-04 → 1.04**. The weight-decay annihilation (zero-task-gradient × coupled-L2 over ~47,625 updates) is confirmed as the cause of the q19 offset-ON steering null, and `weight_decay=0` fully reverses the weight death. This overturns the h41/h42/h45 "offset starves the learned metadata gradient" reading of the same runs: the pathway was not under-trained, it was DELETED by the optimizer, and the deletion is removable at zero magnitude cost.

### What did NOT change — the honest limit

Reviving the weights did NOT produce strong functional steering. run_type M2 moved from bit-exact dead (frac_dir 0.000 / resp 0.000 / natural_variance_insufficient=True) to alive-but-weak (frac_dir 0.559 ≈ chance / resp 6e-4 / flag=False). Depth `eta` stays correctly flat (median_eta_slope 1.9e-5) because the offset does the depth math — as designed. The decoder-only probe reported near-zero assay/run_type η-sensitivity (d_assay 2.6e-4, d_rt 2.0e-3) but is UNFAITHFUL: on a synthetic random z it undercounts the offset-OFF total depth slope 14× (0.075 vs real ~1.06), so those numbers are lower bounds, not measurements. V2 is therefore could-not-evaluate, not refuted.

### New follow-up questions this raises (for the restructure)

1. **Why is functional steering still weak after full weight revival?** Two live explanations: (a) run_type is near-unidentifiable on the T_ 5-biosample slice (H(run_type|assay,read_length)=0.000 bits — an audit identifiability bound, unfixable by architecture), and (b) the decoder may route assay identity through the encoder-side z (which already encodes it) rather than the y_meta FiLM path, leaving the decoder's assay-FiLM redundant and near-flat even when its weights are alive. (b) is testable and new.
2. **Does the +0.10 macro-CRPS / +0.10 imp-Spearman gain from wd=0 replicate across seeds** and hold under the corrected instruments (real foreground, clustered CIs, honest marginal, fixed assay labels — audit S3/S4/S5/S6)? The current numbers use the same instruments the audit flagged.
3. **Is a no-decay param GROUP (AdamW, decay on the trunk only) as good as global wd=0**, keeping the trunk regularized while protecting the conditioning pathway? Cheaper-to-defend production recipe.

See-also the metadata-conditioning audit (`scratchpad/CONSOLIDATED.md`): this is the register's §7.1 "highest-value, never-run" experiment, now run.

### AMENDMENT 2026-07-28 — post-[[h48_h0_fix_the_broken_q19_instruments_and_re|h48]] (verdict supported → partial)

The **M1 half of this node stands unchanged**: h48's re-score reproduces every anchor to 4 decimals (1.4950 / 1.9023 / 1.3413 / 2.0561) — the h47 M1 numbers were never wrong; what was wrong was what they were compared against and what was concluded from them.

Two corrections. **(1) V2 is UNMET, not met.** The Δη 0.833 that upgraded it is a MISSING-sentinel artifact; sentinel-free the value is 0.0023, 43× below this node's own 0.10 bar, while the two offset-OFF arms read 4.1772 / 9.7144 on the identical probe. So `wd=0` did **not** deliver functional assay steering on the prompt pathway. **(2) "Revived" is the wrong word even at the weight level.** `wd0_on`'s assay table is at random-init statistics (element std 0.94 vs 0.97 fresh; cosine 0.988 with an untrained table): never destroyed *and* never trained. The mechanism blocking steering is downstream of the embedder — the fusion LayerNorm (pre-LN norm 396 vs 8.9) and `film_proj` — which is what q20's FiLM arms must actually target.

**Qualification carried onto the magnitude claim (report §4):** *"h47's V1 / V3 / V4 / V5 are untouched, with **V3 qualified**: the magnitude margin is 0.079 on capability rather than 0.561 raw, and within the target-clustered noise floor for the three trailing arms."* Recorded as the report words it, against V3. (Arithmetically the pair is `offoff` − `wd0_on`: raw 1.902 − 1.341 = 0.561, capability 1.3871 − 1.3077 = 0.0794 — noted for traceability, not re-mapped to another verifiable.) The standing statement: `wd0_on` is the best arm on capability and its lead over `offoff` is the only pairwise difference that survives target-clustered bootstrap (+0.093 [+0.004, +0.217]); `main`, `offoff` and `wd0_off` are statistically indistinguishable from one another.

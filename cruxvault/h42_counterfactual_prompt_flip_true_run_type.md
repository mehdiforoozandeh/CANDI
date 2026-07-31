---
id: h42
type: idea
title: "Counterfactual-prompt flip: true run_type imputes better than the wrong one (M2, run_type; read_length secondary)"
parent: q19
status: done
verdict: partial
metric: run_type dir 0.00/resp 0 offset-on vs 0.69/1.83 CI-excl-0 both single&paired offset-off
created: "2026-07-15T23:35:47"
updated: "2026-07-16T09:20:13"
---

# h42 — Counterfactual-prompt flip: true run_type imputes better than the wrong one (M2, run_type; read_length secondary)

Parent:: [[q19_can_we_make_dual_conditioning_work_on_re]]

## Problem Statement

The ENCODE Imputation Challenge headline problem: single-end vs paired-end train/test mismatch biased imputations (dedup -> read-start counts; non-monotonic, worst at peaks). If CANDI reads the run_type prompt correctly, telling it the TRUE run_type should impute the held-out target better than telling it the wrong one -- a cheap flip test needing no counterfactual data (12 held-out targets = 3 single + 9 paired).

## Idea / Hypothesis

Counterfactual-prompt flip: true run_type imputes better than the wrong one (M2, run_type; read_length secondary)

## Verifiables

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
- [ ] responsiveness: flipping run_type in the imputation prompt changes the predicted NB (CRPS(pred_true vs pred_flip) > 0) on the 12 held-out imputation targets (UNMET on the winning recipe: responsiveness 0.000 — the offset-on model ignores the run_type prompt entirely. Offset-OFF control offoff_s0: responsiveness 1.83)
- [ ] direction: CRPS(true run_type prompt, GT) < CRPS(flipped prompt, GT), bootstrap CI excludes 0; power reported split single(3)/paired(9) (UNMET on the winning recipe: dir-frac 0.00, CI includes 0. Offset-OFF control: dir-frac 0.69, bootstrap CI excludes 0 for BOTH single AND paired)
- [-] read_length (secondary): same true-beats-flipped CRPS direction when flipping read_length (could-not-evaluate cleanly: CI excludes 0 on 3.7M points but responsiveness ≈ 0 on the winning recipe — a negligible effect)
- [x] honest null recorded: if Delta ~ 0 under current natural-variance training, documented as 'natural variance insufficient -> defer to path-(c) paired->single augmentation', spawning a follow-up rather than a design failure (found: metric correctly took natural_variance_insufficient=True; BUT the offoff control refutes that interpretation — the variance IS sufficient, the offset head suppresses its use; path-(c) NOT needed)

## Planned Intervention

On the trained h40 model, over the **12 held-out imputation targets** (target assay absent from the `T_` input ⇒ the prompt is the only channel carrying its covariates): predict under the **true** `y_meta_imp` (real run_type) → `CRPS(pred_true, GT)`; then **flip run_type** (0↔1) in the prompt → `CRPS(pred_flip, GT)`. Extend `sandbox/eval.py::prompt_sensitivity_runtype_mse` from `MSE(mu)` to **CRPS vs GT**.
- **Responsiveness**: `ΔCRPS(pred_true, pred_flip) > 0`.
- **Direction**: `CRPS(pred_true, GT) < CRPS(pred_flip, GT)`, bootstrap CI excludes 0; **report split by run_type power: paired (9 targets) vs single (3)** — expect more power on paired (majority-single training prior means lying "single" on a paired target pulls toward the prior and away from truth).
- **read_length (secondary)**: flip read_length to the nearest other observed value; same true-beats-flipped CRPS direction.
- **Honest null**: if `Δ ≈ 0`, record as *natural-variance-insufficient* and spawn a follow-up to **path-(c) paired→single training augmentation** (re-download FASTQ for the paired sandbox cells, reprocess to single-end, re-bake both tracks — the run_type analog of DSF). **Do not mark refuted** without that augmentation check.

**Confound to state, not fix**: run_type is a proxy for the whole single/paired *processing pipeline* (dedup → read-start counts), so a positive flip = the model uses the pipeline covariate correctly (the ENCODE-challenge goal), not necessarily biological causality. First re-verify each target's baked run_type against its source experiment (provenance-drift caveat).

**Tests (pre-GPU) — `tests/test_metrics_real.py::test_M2_runtype`** (see q19 §Validation):
- flip 0↔1 **changes** pred on a run_type-responsive synthetic model, **does not** on an ignoring one.
- direction `CRPS(true) < CRPS(flip)` computed against the correct GT; **single/paired split == 3/9** (fixture assertion).
- read_length flip goes to the **nearest observed** value ({30,36,76,100,101}); direction test runs.
- the **honest-null path returns the "natural-variance-insufficient" flag (NOT "refuted")** when responsiveness ≈ 0 — assert the code takes that branch rather than recording a false negative.

## Run Links

- SLURM 49274497 (dual_conditioning_real sweep, sampled, EP=25)
- SLURM 49277527 (dual_conditioning_real FULL-COVERAGE sweep — the definitive run)

## Findings

**Partial — run_type IS readable on real data, but the winning-recipe offset head suppresses the model's use of it (not a data-variance problem).** On the winning recipe (offset on), flipping run_type in the imputation prompt changes the predicted NB by nothing (responsiveness 0.000, direction-frac 0.00, CI includes 0), and the readout correctly took the honest-null path (`natural_variance_insufficient=True`). Pre-registration said that outcome should defer to path-(c) paired→single FASTQ augmentation — **but the offset-OFF control (offoff_s0_full) overturns that reading:** with no `2^d` shortcut the model reads the run_type prompt and imputes the held-out target better with the true run_type than the flipped one — **direction-frac 0.69, responsiveness 1.83, bootstrap CI excludes 0 for BOTH single(3) and paired(9) targets.** So the natural run_type variance is *sufficient*; the depth-offset head is what starves the run_type gradient (same mechanism as h41). Path-(c) augmentation is therefore **not** the needed fix — attenuating/removing the offset is. read_length (secondary) is a negligible effect on the winning recipe (CI excludes 0 only by virtue of 3.7M points; responsiveness ≈ 0) → could-not-evaluate. Verdict partial; the production-relevant lever is formalized in [[h45_removing_the_depth_offset_head_recovers_|h45]].

---
id: h40
type: idea
title: Winning-recipe CANDI reconstructs and imputes sensibly on real sandbox data (M1 health, counts-only)
parent: q19
status: done
verdict: supported
metric: imp-Spearman 0.53-0.59 (den 0.71); imp-CRPS 1.5-1.6<2.21 marginal; ECE 0.03-0.06; eff-rank 52
created: "2026-07-15T23:35:45"
updated: "2026-07-16T09:20:12"
---

# h40 — Winning-recipe CANDI reconstructs and imputes sensibly on real sandbox data (M1 health, counts-only)

Parent:: [[q19_can_we_make_dual_conditioning_work_on_re]]

## Problem Statement

M1 is a HEALTH gate, not a SOTA bar: the marginal average-reference baseline (Q_imp 0.4857) is essentially unbeaten at sandbox scale (best autoresearch 0.450, candi_v2 base 0.377), so q19 only needs to show the NB-counts-only model reconstructs/imputes sensibly and non-degenerately before its steering is trusted.

## Idea / Hypothesis

Winning-recipe CANDI reconstructs and imputes sensibly on real sandbox data (M1 health, counts-only)

## Verifiables

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
- [x] imp-count Spearman in the healthy band (>= ~0.38, candi_v2-base level; NOT required to beat the 0.4857 marginal baseline) and imp-count Pearson > 0, on chr21 held-out V/B targets (found: imp-Spearman 0.533/0.589 seed0/1, imp-Pearson 0.537/0.608 — above the 0.38 band, competitive with the 0.4857 marginal)
- [x] health gate: den-count Spearman >= imp-count Spearman (denoising no harder than imputation) (found: den-Spearman 0.709/0.710 >= imp 0.533/0.589)
- [x] NB CRPS finite and <= a depth-marginal reference; PIT-ECE reasonable (<= ~0.10; baseline tau_cal 0.073) (found: imp-CRPS 1.62/1.50 < depth-marginal 2.21; PIT-ECE 0.062/0.026)
- [x] latent not collapsed: encoder eff-rank > 1 with recon > 0 (found: encoder eff-rank 52.1/52.0, recon > 0)

## Planned Intervention

Train the **golden-reference model** (q19 PRD §Architecture: per-assay `per_conv` encoder + `DualCondDecoder` depth-offset **NB counts-only** head) on `T_*` chr19 with **per-assay independent DSF ON** (`dsf_sampling="uniform"`) + cloze masking (q19 PRD §Training). Then eval on **chr21**:
- **imp-count** Spearman + Pearson on the 12 held-out `V_/B_` targets; **den-count** Spearman + Pearson on unmasked `T_` assays; **NB CRPS** (closed-form, `p` from true `mu` in float64); **PIT-ECE** (non-randomized); **encoder eff-rank**.
- **Reference points** (grounding the "healthy band," not a bar to beat): candi_v2 base Q_imp 0.377 (wins on count-Spearman rank, count-Pearson only 0.165); marginal average-reference baseline Q_imp 0.4857 (count-Pearson 0.52); best menu-AR loop 0.450. So imp-count Spearman ≈ candi_v2-base level and imp-count Pearson > 0 = healthy; beating 0.4857 is explicitly **not** required.
- Tick verifiables from these numbers. **h40 passing is the precondition for trusting M2/M3** — a collapsed or degenerate model (eff-rank ≤ 1, recon ≈ 0) invalidates any steering claim, so this is the gate, run first.

Reuse `dual_conditioning/metrics.py` (nb_crps, PIT) and `sandbox/eval.py` correlation utilities; no `candi_v2` edits (import-and-swap).

**Tests (pre-GPU) — `tests/test_metrics_real.py::test_M1`** (see q19 §Validation & Test Plan):
- imp gathered from the V/B targets, den from unmasked T assays — **correct index selection, target never leaks into den**.
- a **predict-the-marginal baseline reproduces the ~0.4857 reference** on the same eval data (the metric matches the known number).
- den ≥ imp health-gate logic exercised on a fixture.
- **eff-rank** = effective rank of encoder `Z` via SVD; trips on a constant-`Z` fixture (collapse detector). Runs on the overfit-fixture model from `test_training.py`. This hypothesis + the block tests are the pre-GPU gate for everything downstream.

## Run Links

- SLURM 49274497 (dual_conditioning_real sweep, 4 arms, EP=25 — sampled: 300 steps/ep, chr21 eval capped at 150 batches)
- SLURM 49277527 (dual_conditioning_real FULL-COVERAGE sweep — whole chr19 × all 5 T_ biosamples/epoch, whole chr21 eval; the definitive run; `results/{main_s0,main_s1,offoff_s0,copyable_s0}_full.json`)

## Findings

**Supported.** The winning-recipe model (offset-on, per-assay independent DSF), trained on the whole of chr19 (all 5 T_ biosamples, every window per epoch, 25 ep) and evaluated on the whole of chr21 (608 units, 3.73M imputation points, no subsampling), reconstructs and imputes healthily and non-degenerately. imp-count Spearman **0.533 (seed0) / 0.589 (seed1)** — above the candi_v2-base band (~0.38) and competitive with the marginal average-reference baseline (0.4857); imp-count Pearson 0.54/0.61. den-count Spearman **0.71 ≥ imp** — the den≥imp health gate holds (denoising no harder than imputation). imp NB-CRPS **1.62/1.50 beats the depth-marginal reference (2.21)**; PIT-ECE **0.062/0.026 ≤ 0.10** (well-calibrated). Encoder eff-rank **52** with recon>0 — no collapse. M1 health is established on the real sandbox data, so the downstream M2/M3 steering readouts (h41–h43) are trustworthy. (Both seeds concordant; the x_eq_y arm is even higher at imp-Spearman 0.639, den 0.763.)

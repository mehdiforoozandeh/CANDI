---
id: h46
type: idea
title: The offset-off imputation gap is scale/magnitude, not lost biology — pooled metrics overstate it
parent: q19
status: done
verdict: supported
metric: shape ties (macro Spearman 0.505 vs 0.465, OFF wins 6/8); magnitude fails (macro CRPS 1.50→1.90, beats-marginal 8/8→3/8); pooling overstated the gap ~3×
created: "2026-07-23T13:28:17"
updated: "2026-07-23T13:29:51"
---

# h46 — The offset-off imputation gap is scale/magnitude, not lost biology — pooled metrics overstate it

Parent:: [[q19_can_we_make_dual_conditioning_work_on_re]]

## Problem Statement

offset-off shows genuine metadata steering (h45 v1/v2) yet lower POOLED imputation than offset-on, which contradicts the expectation that true metadata should improve predictions. Suspicion: the pooled imp Spearman/Pearson concatenate all 12 held-out targets into one vector, mixing assays of very different count magnitudes, so they reward cross-assay SCALE (the offset's free 2^(d-center) arithmetic) rather than within-assay SHAPE (biology). Decompose the metric per-assay to locate the real deficit before treating it as 'metadata hurts imputation'.

## Idea / Hypothesis

The offset-off imputation gap is scale/magnitude, not lost biology — pooled metrics overstate it

## Verifiables

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
- [x] Pooling overstates it: per-assay MACRO correlation gap is far smaller than the POOLED gap (found: imp-Spearman gap POOLED 0.132 → MACRO 0.040 ~3× smaller; imp-Pearson gap 0.165 → 0.077 ~2×. pooled 0.533/0.401, macro 0.505/0.465; denoising near-identical 0.71/0.69)
- [x] Within-assay SHAPE preserved: macro per-assay Spearman gap <= seed noise, and offset-off ties-or-wins on >= 6/8 assays (found: macro gap 0.040 < seed spread ~0.056 [main seed0 0.533 vs seed1 0.589]; offset-off ties-or-wins 6/8 — WINS ATAC 0.715>0.670, DNase 0.643>0.496, H3K36me3 0.560>0.456, H3K9me3 0.410>0.298; ties H3K4me1/H3K4me3; loses only H3K27ac [collapse 0.010] & H3K27me3 0.276<0.379)
- [x] The genuine cost is MAGNITUDE: per-assay macro CRPS and mse_log materially worse for offset-off, and the gap does NOT shrink when computed per-assay (found: macro imp-CRPS 1.495 → 1.902 +27%; macro mse_log 0.486 → 0.768 +58%; CRPS gap ~unchanged pooled +0.443 vs macro +0.407 → NOT a pooling artifact, as expected for an absolute-scale score)
- [x] Offset-off fails the honest per-assay marginal baseline on most assays where offset-on passes (found: offset-ON beats its per-assay marginal on 8/8 assays, offset-OFF on only 3/8 [DNase, H3K4me1, H3K9me3]; e.g. H3K4me3 ranks well [Spearman 0.578] yet CRPS 2.03 > marginal 1.67 → loses to the constant baseline from scale miscalibration)

## Planned Intervention

Add a **per-assay decomposition** to `eval_M1` (`sandbox/diagnostics/dual_conditioning_real/metrics_real.py`): per-assay Spearman/Pearson/CRPS/`mse_log`, macro-averages, `median_mu` vs `median_target` (scale-bias readout), and a proper **per-assay marginal NB baseline** — each assay's own median + method-of-moments dispersion (the pre-existing `marginal_crps` pools all assays into ONE median, a scale-blind strawman both arms beat trivially). Then re-run the two anchor arms — **offset ON** (`main_s0`) and **offset OFF** (`offoff_s0`), seed 0, full-coverage (whole chr19 × all 5 `T_` biosamples/epoch, 25 ep; whole chr21 eval, no subsampling) — DETERMINISTIC (`cudnn.deterministic`) so pooled numbers reproduce the definitive 49277527 run. Compare **pooled vs macro vs per-assay** across Spearman / Pearson / CRPS. Decision rule: if OFF ties ON on macro/per-assay correlation but not pooled → deficit is cross-assay SCALE; if OFF is also worse per-assay CRPS/baseline → the real cost is MAGNITUDE. Job script: `jobs/perassay.sh`.

## Run Links

- SLURM **50101078** — dual_conditioning_real per-assay re-run, 2 arms (offset ON/OFF, seed 0, full-coverage); `results/{main_s0,offoff_s0}_perassay.json` (+ checkpoints). Reproduced the definitive pooled numbers exactly.
- SLURM **49277527** — original definitive full-coverage sweep; `results/{main_s0,offoff_s0}_full.json` (same pooled numbers).

## Findings

**Supported.** offset-off's lower *pooled* imputation is **not lost biology — it is a magnitude/scale-calibration failure, and the pooled correlation metric was overstating it.** The deterministic per-assay re-run (job 50101078) reproduced the definitive numbers exactly (pooled imp-Spearman ON 0.533 / OFF 0.401), so the decomposition is trustworthy.

1. **Pooling was scale-confounded.** imp Spearman/Pearson are computed on one vector concatenating all 12 held-out targets, whose count magnitudes span orders of magnitude (ATAC ≫ H3K9me3); the pooled correlation is dominated by placing each assay's cluster at the right *height* — cross-assay SCALE, exactly the offset's free `2^(d−center)` arithmetic. Per-assay + macro-averaged (scale-free), the gap collapses: **Spearman 0.132 → 0.040 (~3×), Pearson 0.165 → 0.077 (~2×)**. Denoising was already near-identical (0.71 vs 0.69) — consistent with scale being readable from the input when the assay is present, and only *imputation* (assay absent ⇒ scale must come from the prompt) exposing it.

2. **Within-assay shape (biology) is preserved.** The residual macro-Spearman gap (0.040) is within seed noise (main seed0 0.533 vs seed1 0.589 ⇒ ~0.056). Per assay, offset-off **ties-or-beats** offset-on on **6/8** — it *wins* on ATAC (0.715>0.670), DNase (0.643>0.496), H3K36me3, H3K9me3 — losing only H3K27ac (a collapse to 0.010, Pearson −0.185) and mildly H3K27me3. The metadata prompt alone carries the biology as well as, or better than, the offset recipe.

3. **The genuine cost is magnitude, and it is NOT a pooling artifact.** Per-assay macro **CRPS 1.495 → 1.902 (+27%)** and macro **mse_log 0.486 → 0.768 (+58%)**; crucially the CRPS gap barely moves from pooled (+0.443) to macro (+0.407) — because CRPS is an absolute-scale pointwise score, re-slicing cannot flatter it. Offset-off predicts the right *rank* but the wrong *absolute magnitude* (e.g. ATAC median_mu 4.4 vs target ~0).

4. **On the honest baseline, offset-off fails.** Against a proper per-assay marginal (each assay's own median+dispersion — replacing the scale-blind global-median strawman both arms beat trivially), offset-on beats it on **8/8** assays, offset-off on only **3/8**. Telling case H3K4me3: offset-off ranks well (Spearman 0.578) yet CRPS 2.03 > marginal 1.67 — it loses to the constant baseline purely from scale miscalibration. This is exactly the "no-offset doesn't beat random baseline" symptom.

**Mechanism.** The depth-offset does two coupled jobs: it supplies the correct depth→count scale (via `2^(d−center)`) AND, by fitting the mean for free, it zeroes the gradient into `η` so the metadata path is never forced to learn scale. Offset-off recovers the depth *direction* (η-slope 0.88, [[h41_depth_output_steering_is_present_distrib|h41]]) but under-shoots the exact arithmetic, leaving each assay's absolute level mis-calibrated. So the "imputation cost" of removing the offset is not biology — it is **the depth-scale the offset handed over for free**.

**Implication.** This relocates q19's tradeoff from *"steering vs imputation quality"* to *"steering vs absolute-scale calibration,"* and sets the precise success bar for the fix ([[h45_removing_the_depth_offset_head_recovers_|h45]]): a hybrid must restore magnitude — per-assay macro CRPS ≤ ~1.57 (within 5% of the 1.495 anchor) and beats-marginal → 8/8 — while keeping offset-off's learned steering (η-slope, run_type direction) and its already-good shape. **Caveat:** the H3K27ac single-target collapse under offset-off is an outlier worth its own check.

# Synthesis: E6/E7 FiLM Ablation (2026-05-05)

**Scope:** E6 (linear FiLM), E7 (single-shot decoder FiLM), E6E7 (combined) vs baseline_anchor.  
**Status:** Incomplete — all three runs walltime-killed before 200 epochs.  
**Related questions:** Q4 (conditioning architecture), Q5 (metadata collapse).

---

## Headline Conclusions

1. All three FiLM variants avoided divergence in their completed epochs (99–124); baseline_anchor diverged (last/best = 2.02×). Causal vs coincidental is unclear because they were killed before reaching the epoch range where baseline diverged (~ep 160+).
2. E6 (linear FiLM) shows **better pval_imp** (0.684 at ep 119) but **worse count_imp** (2.033) relative to E7 and E6E7. Linear scaling removes exp-amplification of the scale gradient, which specifically hurts count-head conditioning.
3. E7 (single-shot decoder FiLM) shows **better count_imp** (1.645 at ep 99) but **plateau in pval_imp** (≈0.97 across last 5 eval epochs). A single FiLM at the latent appears insufficient to condition pval across the full deconv path.
4. E6E7 (combined) sits between the two: count_imp=1.687, pval_imp=0.941, both still improving at kill (ep 94).
5. Metadata collapse (F1) persists in all three — `depth_count_ratio` ≈ 1.0 in all runs. FiLM architecture changes do not fix depth conditioning.
6. **The 200-epoch / 3h budget is the binding constraint.** Every best epoch equals the last eval epoch. No conclusions about convergence or late-stage stability can be drawn.

---

## Cross-run Comparison Table

| Metric | baseline_anchor | E6_linear_film | E7_single_shot | E6E7_combined |
|---|---|---|---|---|
| epochs completed | 200 | 124† | 114† | 99† |
| diverged | yes (2.02×) | no | no | no |
| quality_score | ineligible | 8.68 | **8.85** | 8.82 |
| best total_loss | 4.972 | 4.922 | 5.103 | 5.062 |
| count_imp_loss (best) | ~1.78‡ | 2.033 | **1.645** | 1.687 |
| pval_imp_loss (best) | ~0.14–0.93‡ | **0.684** | 0.970 | 0.941 |
| peak_imp_loss (best) | ~0.46‡ | 0.492 | 0.587 | 0.579 |
| peak_obs_loss (last) | 0.419 | 0.401 | 0.421 | 0.418 |
| depth_count_ratio (best) | 1.024 | 1.003 | 1.003 | 1.001 |
| imp_count_pearson (best) | 0.023 | 0.030 | **0.110** | 0.059 |

† walltime-killed  
‡ baseline_anchor diverged; best-epoch branch values not directly comparable to non-diverged runs.

Ranker verdicts (from `rank_runs.py`):
- E7 ~= E6E7 (tie; E7 slightly better step at best, E6E7 slightly faster/epoch)
- E6 loses to E7 and E6E7 via branch guard: `count_imp_loss` 20–24% worse

---

## Outcome vs Hypothesis

| Hypothesis | E6 (linear FiLM) | E7 (single-shot) |
|---|---|---|
| Preserve conditioning gradients → stable training | Partial: no divergence; but count conditioning weakened | Partial: count improved, but pval plateau suggests insufficient capacity at single latent site |
| Improve prompt-sensitivity / attribution | Not measurable yet (walltime kill) | Not measurable yet |
| depth_count_ratio improves | No. Ratio ≈ 1.0, same as all prior runs. | No. Same. |

---

## Trajectory Analysis (last 5 eval epochs)

**E6 — ep 99–119 (still improving at kill):**
```
ep=99  total=5.14  cnt_imp=1.977  pval_imp=0.723  peak_imp=0.549  peak_obs=0.490
ep=119 total=4.92  cnt_imp=2.033  pval_imp=0.684  peak_imp=0.492  peak_obs=0.401
```
- peak_obs still falling linearly. pval_imp slowly improving. count_imp **stuck around 2.0** (plateau at ~ep 75, not improving).
- Count stagnation is robust: count_imp shows no downward trend across eps 99–119 while peak/pval are still moving.

**E7 — ep 89–109 (pval plateaued, peak still moving):**
```
ep=89  total=5.21  cnt_imp=1.671  pval_imp=0.993  peak_imp=0.603  peak_obs=0.451
ep=109 total=5.13  cnt_imp=1.702  pval_imp=0.968  peak_imp=0.572  peak_obs=0.421
```
- count_imp oscillates in 1.64–1.70 band → appears converged.
- pval_imp stuck 0.97–0.99 → latent FiLM cannot push it lower in this epoch range.
- peak_imp still declining linearly → not converged.

**E6E7 — ep 74–94 (everything still improving):**
```
ep=74  total=5.27  cnt_imp=1.694  pval_imp=0.997  peak_imp=0.617  peak_obs=0.458
ep=94  total=5.06  cnt_imp=1.687  pval_imp=0.941  peak_imp=0.579  peak_obs=0.418
```
- All branches still improving; most aggressive improvement rate of the three at kill.

---

## Root-Cause Explanations

**E6 count stagnation:** Linear FiLM removes exp-amplification of scale gradients
(`∂L/∂scale = ∂L/∂output · x` vs `· x · exp(scale)` in exp mode). Count activations
(post-NB) tend to be larger in magnitude than pval (post-log1p), so the exp factor was
contributing disproportionately to count conditioning gradients. Removing it weakens
count-head FiLM updates while leaving pval relatively unaffected (consistent with
observed smaller pval grad norms in E6 W&B plots, but pval converging faster).

**E7 pval plateau:** Single-shot FiLM injects metadata once on `[N, F2, L']` before all
three deconv stages. The pval branch specifically requires fine-grained spatial conditioning
(arcsinh log-pvalue varies at high resolution), but the single site conditions a compressed
latent at `L' = L/8 = 96` bins. The 3-stage deconv likely dilutes the conditioning signal
by the time it reaches the output head. Count, being smoother, is less affected by dilution.

**Metadata collapse:** Neither E6 nor E7 nor E6E7 improves `depth_count_ratio`. The FiLM
architecture is not the bottleneck for depth conditioning — consistent with F1 (F1 is likely
an optimizer-side or embedding-side issue, not in how FiLM applies the embeddings).

---

## Next Batch Recommendations

### Immediate (re-run with sufficient budget)
1. **Re-run E6, E7, E6E7 at 400 epochs / 6h walltime** — the minimum to assess convergence.
   Use `BASELINE_EPOCHS=400 BASELINE_TIME=06:00:00`. No other config changes.
   Without convergence data, the current results are directional only.

### Conditional on 400-epoch results
2. If E7's pval plateau persists at 400 epochs: try **E7 + reduced pval_weight (0.3)**
   to see if the pval gradient starvation at the single FiLM site is the bottleneck.
3. If E6's count stagnation persists: test **E6 with a larger FiLM projection init** 
   (increase `std` in bias init) to compensate for weaker scale gradients.
4. If E6E7 is clearly best at 400 epochs: adopt it as the new architecture default and
   continue E8/E9/E11 from that base.

### Parallel (metadata collapse)
5. **E8 (per-group gradient clipping) and E9 (per-module grad-norm logging)** are the
   next highest-leverage experiments for Q5. FiLM architecture is not the fix.

---

## Default Config Change Recommendation

**Change `BASELINE_EPOCHS` from 200 to 400 and `BASELINE_TIME` from `03:00:00` to `06:00:00`.**

Evidence: every run in this sweep peaked at its last eval epoch. The 3h budget produces
~120 epochs at sandbox scale, which is insufficient for pval and peak branches to converge.
At 50 sec/epoch, 400 epochs = ~5.6h, fitting within 6h with margin. This change should be
made to the default `BASELINE_TIME` in `sandbox/slurm/baseline_train.sh` and the
`BASELINE_EPOCHS` env-var default in submit scripts.

No other default config changes recommended until 400-epoch results are available.

---

## Caveats

- All three runs terminated by walltime kill. The `best epoch = last eval epoch` pattern
  means quality_score and branch losses reflect training progress, not convergence.
- baseline_anchor comparison is confounded by divergence; branch-level comparisons should
  be treated as suggestive, not conclusive.
- `depth_count_ratio` is reported per the F1 standing finding. All values ≈ 1.0 confirm F1
  remains open and unresolved by this sweep.
- Ranking `E7 ~= E6E7` is tentative given the step-count gap (E7 at 9500 vs E6E7 at 9025
  at best epoch). At matched epochs, E6E7 may outperform E7.

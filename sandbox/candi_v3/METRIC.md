# CANDI v3 — ERA_SCORE (comprehensive, FROZEN)

*Designed collaboratively 2026-06-18 (supersedes the single-Spearman v1). One clear main term
(imputation), everything else a do-no-harm floor/gate. All terms on the chr21 eval set; train =
chr19, so no leakage. Frozen for a search round.*

---

## Reconstruction composites (uniform mean of 4 correlations, all ∈ [−1,1])

```
Q_imp = mean(imp_pval_spearman, imp_pval_pearson, imp_count_spearman, imp_count_pearson)
Q_den = mean(den_pval_spearman, den_pval_pearson, den_count_spearman, den_count_pearson)
```
- **Imputation** (`Q_imp`): predicted vs GT on the reserved **V/B held-out assays** (chr21).
- **Denoising** (`Q_den`): low-depth (dsf8) T_ input → predict high-depth (dsf1) target, on the
  **observed T_ assays** (chr21). Easier task → healthy models have `Q_den > Q_imp`.
- MSE and NB-NLL deliberately excluded (scale issues / NLL is the training loss → circular,
  mean-match-gameable, distribution-type-dependent).

## The score

```
S_A = Q_imp − Q_imp_baseline                              # PRIMARY: imputation skill (single maximand)

ERA_SCORE = S_A
          − w_den  · max(0, Q_imp − Q_den)                # denoising ≥ imputation gate
          + w_cal  · min(0, τ_cal − ECE)                  # calibration floor (ECE of count dist)
          + w_cidx · min(0, c_index − cidx_baseline)      # uncertainty-discrimination floor (C-index)
          + w_peak · min(0, peak_auroc − auroc_baseline)  # peak-prediction floor (held-out V/B peaks)
          + w_dcr  · (min(0,DCR−3) + min(0,5−DCR))        # depth-calibration band [3,5]
          + (−1e9 if structurally degenerate)
```

- **Denoising gate** (the subtle one): RAW comparison, `max(0, Q_imp − Q_den)`. Zero when
  denoising ≥ imputation; penalty when denoising falls below; **capped at 0** so a perfect /
  identity denoiser earns no bonus. Encodes "denoising must be at least as good as imputation,
  but isn't the goal." (Identity also can't help imputation, so the copy shortcut earns nothing.)
- **Calibration** = ECE floor + C-index (distributional ranking, sample-based). **No naked
  sharpness** — rewarding it incentivises overconfidence (the ECE≈0.5 failure seen in testing).
- **Peaks**: floor at the baseline AUROC; a candidate that omits `peak_prob` is scored at chance
  (0.5) → penalised below baseline, so peaks are incentivised though optional in the contract.
- **DCR**: physics-absolute band [3,5] (~4.0 = +2 log2 ⇒ 4× depth), two-sided.

## Constants (FROZEN — `freeze_constants.py`, from the marginal average-reference baseline)

| const | value | meaning |
|---|---|---|
| `Q_imp_baseline` | 0.4857 | S_A zero-point (mean of 4 imp correlations) |
| `τ_cal` | 0.0734 | calibration floor (baseline ECE) |
| `cidx_baseline` | 0.4985 | C-index floor (baseline ≈ chance — gentle) |
| `auroc_baseline` | 0.7161 | peak-AUROC floor (baseline avg-peak track) |
| `dcr_lo, dcr_hi` | 3.0, 5.0 | physics-absolute DCR band |
| `w_den, w_cal, w_cidx, w_peak, w_dcr` | 0.5, 0.4, 0.4, 0.4, 0.02 | interpretable Spearman-equivalence weights |

**Baseline ERA_SCORE = −0.04** (every term at baseline → 0, except the depth-blind marginal
predictor failing the DCR band). A real candidate beats it by raising `Q_imp` while keeping
denoising ≥ imputation and the calibration / C-index / peak / DCR floors satisfied.

## Properties
- One clear main term (imputation) → ERA-compliant; can't win by maxing a secondary or
  sacrificing imputation.
- Denoising: penalised if worse than imputation, no reward if better (no identity cheat).
- All metrics on chr21 eval, never training data.

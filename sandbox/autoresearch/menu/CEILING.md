# Menu-AR — consolidation + data-regime ceiling test

Follow-on to `REPORT.md`. Tests the two live recommendations: **(1) consolidate the cross-loop wins**
into one candidate, and **(3) the data regime is the real lever** (Q_imp capped below the marginal
baseline by chr19-only overfit). Run 2026-07-01. Artifacts in `_ceiling/`.

## Design
- **Consolidated candidate** = crps champion (CRPS proper score + LOO-ref deviation-corr on
  pval/count/dispersion) **+** single_lambda's decoder GroupNorm (`cfg.decoder.norm="group"`). The two
  champions edited train.py only; candi_model is byte-identical across base/crps/single_lambda, so the
  merge is exactly those two edits.
- **Data regimes:** `chr19` (frozen, 3053 train windows) vs `augmented` = chr19 ∪ type2 loci
  (region_type≠255; +2000 cCRE/non-cCRE windows on other chromosomes). Eval is **always chr21** and the
  marginal baseline (Q_imp 0.4857) is chr21-derived → **ERA_SCORE stays comparable across all cells**.
  The 2000 type2 windows are **leak-free wrt chr21** (0 overlap, verified). Env-gated in
  `data_v3._win_indices` (`MENU_AUGMENT=1`); the frozen default is preserved — **`base_chr19` reproduced
  the frozen base bit-exactly (−0.12607312720564523)**, proving the gate inert by default and the
  `_ceiling` scaffold sound.
- **Budget** held constant in **gradient steps** (19100) to separate data-diversity from compute; plus
  an **epochs=5 variant on augmented** (31600 steps) for the compute-uncapped ceiling. 6 cells.

## Results

| model | data | steps | epochs | ERA | S_A | **Q_imp** | imp_count_Pe | imp_pval_Pe | ECE |
|---|---|---|---|---|---|---|---|---|---|
| base | chr19 | 19100 | 5.0 | −0.1261 | −0.109 | 0.377 | 0.165 | 0.300 | 0.117 |
| base | augmented | 19100 | 3.0 | **−0.1115** | −0.092 | 0.394 | 0.197 | 0.368 | 0.122 |
| base | augmented | 31600 | 5.0 | −0.1246 | −0.116 | 0.370 | 0.193 | 0.370 | 0.095 |
| consol | chr19 | 19100 | 5.0 | −0.0663 | −0.063 | 0.423 | 0.236 | 0.385 | 0.083 |
| consol | augmented | 19100 | 3.0 | −0.0901 | −0.086 | 0.400 | 0.206 | 0.366 | 0.084 |
| consol | augmented | 31600 | 5.0 | **−0.0540** | −0.050 | 0.436 | 0.244 | 0.406 | 0.085 |
| *crps alone* | *chr19* | *19100* | *5.0* | *−0.0408* | *~−0.045* | *0.450* | *0.292* | *0.086* |

Baseline reference: Q_imp 0.4857, ECE 0.0734, c_index 0.4985, AUROC 0.7161 (S_A = Q_imp − 0.4857).

## Findings

1. **Consolidation is NOT additive — GroupNorm anti-synergizes with crps.** `consol_chr19` (−0.0663,
   Q_imp 0.423) is **worse than crps alone** (−0.0408, Q_imp 0.450). GroupNorm lifted single_lambda from
   *its* base, but on top of crps's already-recalibrated CRPS/deviation heads it regresses the model.
   Attribution is clean (base reproduced bit-exact; the only delta from crps-best is the one GroupNorm
   line). **Rec #1 falsified: naive "stack the wins" underperforms the best single loop.**

2. **The data-regime lever is directionally real but too small to break the baseline.** At matched
   steps, the +2000 type2 windows raise base Q_imp 0.377→0.394 (+0.017) — diversity regularizes and
   lifts imputation, as the ceiling hypothesis predicted. But the lift is ~+0.02, not the +0.11 needed
   to reach 0.4857. **No cell crossed S_A ≥ 0; best Q_imp 0.436 < 0.4857.** The chr19-regime ceiling
   holds even with the extra sandbox data. **Rec #3 confirmed in direction, refuted in magnitude** at
   this data scale — breaking the baseline needs far more data (the full MERGED panel), not the 2000
   sandbox loci.

3. **The ceiling is an overfitting phenomenon, and regularization changes who benefits from data/compute.**
   - *base* (weakly regularized) peaks at **augmented @ 3 epochs** (Q_imp 0.394); pushing to 5 full
     epochs re-overfits back to chr19 level (0.370). It wants *fewer passes on more data*.
   - *consol* (GroupNorm + CRPS) does the **opposite**: augmented 3ep→5ep *improves* (0.400→0.436). Its
     regularization resists overfitting, so it can exploit 5 full epochs on diverse data. It wants *more
     passes*. → **The best full-model cell is `consol_aug5ep` (−0.0540, Q_imp 0.436)**, but still below
     crps alone.
   - Consistent with the earlier budget result (candi_v2-B imputation peaks ≤5ep on chr19); adding data
     shifts the overfit point but does not remove it.

4. **Calibration is robust across all consol cells** (ECE 0.083–0.085 vs base's 0.095–0.122) and peak
   AUROC rose with augmentation everywhere (0.776→0.80–0.84) — CRPS + more data help the do-no-harm
   floors even where they don't lift Q_imp enough.

## Verdict
- **crps alone (−0.0408, Q_imp 0.450) remains the best candidate.** Neither consolidation nor the
  2000-window augmentation beat it at the frozen judge.
- The ceiling is confirmed **data-limited and overfitting-driven**, not architecture-limited. The only
  lever with real headroom is a **much larger training panel** (multi-chromosome / MERGED), which is a
  production-data change, not a sandbox tweak.
- **Judge state:** the augment gate in `_judge/data_v3.py` is env-gated and inert by default (proven by
  the bit-exact base reproduction). Leave it for future data experiments, or delete the `MENU_AUGMENT`
  branch to re-freeze verbatim — user's call.

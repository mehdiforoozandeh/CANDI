# A/B Encoder × Dropout Comparison: CANDI vs Fresh, dropout 0.1 vs 0.01

Status: synthesis (read-only)
Parents: not pre-registered in EXPERIMENTS.md (submitted directly as `ab_encoder_compare` / `ab_encoder_compare_dropout001`)
Linked from: [`EXPERIMENTS.md`](EXPERIMENTS.md), [`config_promotions.md`](config_promotions.md)
Date: 2026-05-19

---

## Headline conclusions

1. **Dropout=0.01 consistently improves combined_loss (~2.5%) but degrades biological signal and geometry** across both encoder types. The gain in prediction quality is offset by a 30–36% drop in runtype sensitivity and a 60–70% worse covariance condition number. Confidence: **High** (consistent across both encoder types in same training run).

2. **CANDI encoder retains 2.4× better runtype sensitivity than fresh encoder (0.627 vs 0.256 at dropout=0.1)**. This is fully consistent with FJ15: the encoder architecture is the primary determinant of biological sensitivity, not the predictor. Confidence: **High** — but note the fresh encoder comparison is confounded (see caveat below).

3. **Fresh encoder achieves better combined_loss and geometry at both dropout levels**. fresh_encoder (dropout=0.1) is the only run with cov_cond_last < 50 (47.6 — barely under threshold), and has the highest enc_er_last (31.6). Confidence: **Medium** (confounded by simultaneous film_mode and transformer_type differences; fresh runs used old default `per_conv` not current `per_conv_and_transformer`).

4. **Higher dropout (0.1) acts as an isotropy regularizer**: the delta cov_cond from dropout=0.1 to dropout=0.01 is +33.3 for fresh (47.6→80.9) and +50.1 for original (81.2→131.3). Both worsen substantially. Confidence: **High**.

5. **All 4 runs fail the v2 geometry gate** due to pred_slope > 0 and sigreg_converged = 0 at the last step. These are only 107-step runs (~170 effective epochs); SIGReg has not converged in any run. All quantitative comparisons are from the best checkpoint, not the terminal state. Confidence: **High** (walltime / step count is the primary gating failure mode here, not structural collapse).

6. **The comparison between encoder types is confounded**: fresh encoder runs use `film_mode=per_conv` (old default) while original encoder uses `per_conv_and_transformer` (current E23.5 default). E23 batch 1 showed `per_conv_and_transformer` achieves 2.3× better runtype retention than `per_conv` — part of the original encoder's runtype advantage may be attributable to film_mode, not model_type alone. Confidence: **High** (this is a design confound, not a data uncertainty).

---

## Cross-run quantitative table

| Run | combined_loss_best | pred_loss_best | cov_cond_last | enc_er_last | runtype_best | runtype_last | dropout | model_type | film_mode | xfm_type |
|---|---|---|---|---|---|---|---|---|---|---|
| **fresh_encoder_dropout001** | **0.7034** | **0.0193** | 80.95 | 18.18 | 0.297 | 0.183 | 0.01 | fresh | per_conv | production_dual |
| original_encoder_dropout001 | 0.7087 | 0.0203 | 131.25 | 16.87 | 0.449 | 0.404 | 0.01 | candi | per_conv_and_transformer | xtransformers |
| fresh_encoder | 0.7218 | 0.0317 | **47.63** | **31.60** | 0.464 | 0.256 | 0.1 | fresh | per_conv | production_dual |
| original_encoder | 0.7277 | 0.0505 | 81.20 | 26.44 | **0.751** | **0.627** | 0.1 | candi | per_conv_and_transformer | xtransformers |

Bold = best value per column. Ranking by combined_loss_best (paper-grounded v2).

---

## Per-run grad / stability table

| Run | pred_slope_last | sigreg_converged | cov_cond_first | cov_cond_last | enc_er_first | enc_er_last | enc_er_delta | v2 gate |
|---|---|---|---|---|---|---|---|---|
| fresh_encoder_dropout001 | +0.143 | 0.0 | 35.0 | 80.95 | 14.6 | 18.18 | +0.188 | FAIL (slope + conv + cov) |
| original_encoder_dropout001 | +0.429 | 0.0 | 46.2 | 131.25 | 18.6 | 16.87 | +0.281 | FAIL (slope + conv + cov) |
| fresh_encoder | +0.429 | 0.0 | 27.2 | 47.63 | 30.5 | 31.60 | +0.235 | FAIL (slope + conv) |
| original_encoder | +0.143 | 0.0 | 20.7 | 81.20 | 38.5 | 26.44 | +0.118 | FAIL (slope + conv + cov) |

Key observation: `cov_cond` diverges upward in all runs (geometry worsening with time), but fresh_encoder stays under 50 throughout. This is the healthiest geometry trajectory.

---

## Per-experiment outcome vs hypothesis

| Run | Hypothesis | Outcome | Confidence |
|---|---|---|---|
| original_encoder | CANDI encoder with fresh xfm predictor and per_conv_and_transformer FiLM is the E23.5 default baseline | Partial — best runtype (0.627), worst combined_loss (0.7277). Geometry borderline (cov=81). Confirms biological advantage but not combined_loss advantage. | Medium |
| original_encoder_dropout001 | Reducing dropout to 0.01 improves combined_loss without harming geometry | Rejected — combined_loss improves 2.6% but cov_cond worsens from 81→131 (+62%), runtype drops 36% (0.627→0.404). The tradeoff is unfavorable. | High |
| fresh_encoder | Fresh encoder with per_conv FiLM (old default) under new E23.5 defaults produces clean geometry | Partial — best geometry (cov=47.6, enc_er=31.6), best combined_loss with dropout=0.1, but runtype=0.256 (weak biology). Note fresh runs used stale film_mode. | Medium |
| fresh_encoder_dropout001 | Fresh encoder + dropout=0.01 yields best combined_loss | Confirmed — lowest combined_loss 0.7034, but worst runtype (0.183) and degraded geometry (cov=81). Win on primary metric at cost of biological signal. | High |

---

## Implications for next batch

1. **Clean encoder type A/B (highest priority)**: Rerun both encoder types under the SAME configuration — use E23.5 default `film_mode=per_conv_and_transformer` and `transformer_type=xtransformers` for both. Currently the fresh encoder runs used `per_conv + production_dual` (old defaults). Until this confound is resolved, the biological advantage of the CANDI encoder over the fresh encoder cannot be separated from the film_mode advantage. Predicted metric: if fresh + `per_conv_and_transformer` matches original's runtype ≥ 0.5, the encoder type advantage is smaller than it appears. 1 A/B pair, ~2 GPU-hours.

2. **Longer runs for convergence gate (needed before any promotion)**: All 4 runs fail v2 gate due to 107-step walltime limit (pred_slope > 0, sigreg_converged = 0). A 3× longer run (~300 steps) on the best config (original_encoder, dropout=0.1 for biology, or fresh_encoder for combined_loss) is needed to confirm whether SIGReg converges and geometry stabilizes. 1 run, ~3 GPU-hours.

3. **Dropout sensitivity sweep on CANDI encoder only**: A 3-point sweep (dropout ∈ {0.05, 0.1, 0.2}) would localize the optimal regularization point. With dropout=0.1 as the current winner on runtype+geometry and dropout=0.01 winning on combined_loss, the crossover point is somewhere in between. 2 runs, ~2 GPU-hours.

4. **E23.5-H1/H2/H4 runs (already staged)** supersede these as the primary candidates for Stage 2 selection — the ab_encoder_compare runs are not pre-registered in E23.5 and should not replace H1/H2/H4.

---

## Standing findings (carried forward)

| Finding | Status | What this synthesis adds |
|---|---|---|
| FJ5 (enc_er peaks at random init and collapses) | open | Partially extended: fresh_encoder shows a different pattern — enc_er starts at 30.5 and stays high (31.6 at end), suggesting the `per_conv + production_dual` fresh config may behave differently from fresh configs tested in E23. |
| FJ7 (meta_tgt conditioning is dominant runtype lever) | open | Not directly tested here (all runs use meta_tgt conditioning). Consistent with high runtype values when candi encoder is used. |
| FJ15 (encoder is root cause of runtype collapse) | open | **Extended**: candi encoder achieves runtype_best=0.751 at dropout=0.1; fresh encoder achieves 0.464 at same dropout. Row effect (encoder type) ≈ 1.6× on runtype_best and 2.4× on runtype_last — consistent with FJ15. Effect may be partially attributable to film_mode confound (see caveat). |
| FJ12 (fresh encoder candidates for redesign) | partially resolved | `production_dual` transformer type tested in these fresh encoder runs; this is a different config from the `xtransformers` tested in E23. Combined_loss 0.7034–0.7218 for `production_dual`; E23 showed `xtransformers` gave −17% pred_loss. |
| FJ9 (optimization pressure accelerates collapse) | open | Lower dropout (0.01) is analogous to reduced regularization pressure; cov_cond degrades more aggressively (+60–70%) — consistent with FJ9. |

---

## Caveats and limits

- **Short training**: 107 steps (~170 epochs) only. No run achieves sigreg_converged = 1.0 at the final step. All v2 gate failures are convergence failures, not structural collapses. Results are directional only.
- **Film_mode confound (major)**: The two "fresh encoder" runs use `film_mode=per_conv` (old E23 default), not the E23.5 promoted `per_conv_and_transformer`. The E23.5 defaults table explicitly promotes `per_conv_and_transformer` as the best metadata-retention option. This confound means the encoder-type comparison is NOT a clean A/B — at least 1.6× of the runtype_sens gap could be attributable to film_mode.
- **Single seed**: All 4 runs use the same random seed; no replicate noise estimate.
- **Combined_loss range is narrow**: The spread is 0.7034–0.7277 (3.4% range). With no geometry gate passing, differences may not be stable across seeds or longer training.
- **Stage 2 relevance**: These runs were not evaluated on Stage 2 (frozen encoder → decoder) metrics. The runtype sensitivity is a proxy for biological quality; the actual Stage 2 reconstruction metrics may tell a different story.

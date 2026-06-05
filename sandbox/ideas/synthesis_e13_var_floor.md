# E13 Variance Floor Ablation: gaussian_var_min 1e-6 vs 0.1

Status: synthesis (read-only)
Parents: [idea_e13_uncertainty_logvar_clamp.md](idea_e13_uncertainty_logvar_clamp.md)
Linked from: EXPERIMENTS.md
Date: 2026-05-06

## Headline conclusions

1. **Raising the GaussianLayer variance floor from 1e-6 to 0.1 eliminates pval_imp divergence** (F7 mitigated). The control run (var_min=1e-6) reproduced the full variance-collapse failure: `pval_obs_loss` reached −0.111 at epoch 64 and `pval_imp_loss` exploded to 45.5 at epoch 334. The treatment run (var_min=0.1) kept both losses bounded through 356 epochs — `pval_obs_loss` best=0.315, last=0.423; `pval_imp_loss` best=0.476, last=0.671. **Confidence: High.** (Source: epoch rows in each run's `metrics.jsonl`.)

2. **The variance floor does not materially hurt peak-epoch imputation quality.** Best `imp_pval_pearson_gw`: control=0.278 (epoch 64), treatment=0.306 (epoch 84), a +10% improvement despite the hard floor. Best `den_pval_pearson_gw`: control=0.444 vs treatment=0.445 (negligible difference). The floor's main effect is to prevent variance collapse, not to suppress genuine signal. **Confidence: High.**

3. **The control run's "better" late obs loss is degenerate.** `pval_obs_loss` reaching −0.111 in the control represents overconfident predictions (variance → 0, NLL → −∞), not higher quality. The treatment's bounded obs loss (best=0.315) reflects calibrated prediction. Interpreting obs NLL as a quality metric is only valid when variance has not collapsed. **Confidence: High.**

4. **Post-peak gentle degradation persists in the treatment.** After epoch 84, `imp_pval_pearson_gw` gradually declines from 0.306 to 0.209 at epoch 354 (−32%), and `pval_imp_loss` slowly climbs from 0.476 to 0.671. This is a benign obs/imp generalization gap — not the catastrophic divergence seen in F7 — but it indicates the model could benefit from early stopping or a lower final LR. **Confidence: Medium** (single seed, pval-only isolation run).

5. **var_min=0.1 is sufficient; whether 0.01 would yield comparable stability is untested.** The planned range in the idea file started at 0.01. At 0.1 the model cannot express variance below 0.1, which is aggressive (prevents high-confidence predictions even when warranted). A follow-up sweep at var_min=0.01 is recommended before promoting the floor as a global default. **Confidence: Low.**

6. **Depth metadata collapse (F1) persists.** Both runs show `depth_count_ratio` ≈ 1.0 throughout. The variance floor has no effect on Q5. **Confidence: High.**

## Cross-run quantitative table

| run | var_min | epochs_run | pval_obs_loss (best) | pval_imp_loss (best) | pval_imp_loss (last) | diverged | imp_pval_pearson (best) | den_pval_pearson (best) | den_pval_spearman (best) |
|---|---|---|---|---|---|---|---|---|---|
| E13_ctrl_var_floor | 1e-6 | 338 | **−0.111** (degenerate) | **0.287** | 45.54 | **YES** | 0.278 | 0.444 | 0.420 |
| E13_var_floor | 0.1 | 356 | 0.315 | 0.476 | **0.671** | **NO** | **0.306** | **0.445** | 0.388 |

Bold = best value per column. Note: best obs loss in control is degenerate (variance collapse); treatment best obs loss is the valid quantity.

## Per-run grad / stability table

| run | grad_pre_clip (median) | grad_pre_clip (p95) | clip_fraction | pval_obs grad (median) | pval_imp grad (median) |
|---|---|---|---|---|---|
| E13_ctrl_var_floor | 6.58 | 17.95 | 0.705 | 4.166 | 5.186 |
| E13_var_floor | **2.78** | **8.80** | **0.598** | **1.656** | **2.176** |

The treatment run has ~2.4× smaller median pre-clip grad norm and ~15% lower clip fraction. The bounded variance prevents the near-zero variance regime that produces anomalously large gradients from the `(y-μ)²/var` term when `var → 0`.

## Per-experiment outcome vs hypothesis

| run | hypothesis | outcome | confidence |
|---|---|---|---|
| E13_ctrl_var_floor (var_min=1e-6) | Reproduce F7 variance-collapse divergence under pval-only isolation | **Confirmed** — `pval_imp_loss` diverges to 45.5; `pval_obs_loss` reaches −0.111 at ep 64 | High |
| E13_var_floor (var_min=0.1) | Raising the floor prevents variance collapse and keeps pval_imp bounded | **Confirmed** — not diverged (last/best ratio=1.40 < 1.5); imp_pval_pearson +10% vs control | High |

## Implications for next batch

1. **E13+E7 multi-head run** (highest priority, 1 GPU × 6h): Run the E7 default architecture (`single_shot_decoder_film=True`) with `gaussian_var_min=0.1`. The key question is whether pval_imp divergence in multi-head training (onset ep94 in E7) is also eliminated. Predicted: `imp_pval_pearson` best epoch improves over E7's 0.277 and the divergence onset is delayed or eliminated.

2. **var_min=0.01 sanity check** (1 GPU × 6h, pval-only): Test whether the weaker floor also prevents divergence. If yes, 0.01 is preferable — it allows the model more expressivity for high-confidence predictions while still preventing the degenerate near-zero regime. Predicted: `pval_obs_loss` best will be lower than 0.1-floor run; divergence may or may not be prevented.

3. **Early stopping in E13-style runs** (no run needed, config change): The treatment's imp_pval_pearson peaks at epoch 84 then degrades to 0.209 by epoch 354. Enabling `training.early_stop_enabled=True` with `early_stop_patience=10` (on `pval_imp_loss`) would save ~5× GPU time and avoid the post-peak generalization gap for future pval-focused runs.

4. **Promote `gaussian_var_min=0.1` as the new default after step 1**: Only promote after confirming it works in the multi-head context. Pval-only isolation is a necessary but not sufficient test.

## Standing findings (carried forward)

- **F1 — Depth metadata ignored**: open. Both E13 runs show `depth_count_ratio` ≈ 1.0. Unaffected by this intervention.
- **F7 — Pval Gaussian NLL variance collapse**: **mitigated** by `gaussian_var_min=0.1` in pval-only isolation (E13_var_floor, 2026-05-06). Remains open until confirmed in multi-head training (E7+E13 run).
- **F8 — E7 best multi-head architecture**: open. E13 synthesis does not test multi-head; E7 remains the best multi-head reference but may be superseded once E7+E13 completes.

## Caveats and limits

- Both runs are **pval-only isolation** (`count_weight=0`, `peak_weight=0`). Count and peak heads are absent; this removes the gradient competition that dominates multi-head training. Results may not transfer directly to multi-head context.
- Single seed (42) for both runs. No error bars on any metric.
- Both runs used the H100 MIG 1g.10gb slice; the `pval_imp_loss` trajectory is walltime-limited (TIMEOUT at 6h, ~338–356 epochs of 400). The treatment run's gentle post-peak degradation may continue beyond epoch 356.
- `den_pval_spearman` is slightly lower in the treatment (0.388 vs 0.420 best). This could be a real cost of the floor or a random-seed artifact. Not interpretable at N=1.

---
name: jepa-observability
description: >
  Analyze CANDI JEPA runs. Triggers on: lejepa/* metrics, encoder_eff_rank,
  cos_sim_ctx_tgt, sigreg_loss, adaLN_gamma_norm, meta_sens_runtype, SIGReg lambda,
  pred_hidden bottleneck, UMAP/PCA encoder geometry, encoder collapse, trivial solution,
  metadata sensitivity probe, Stage 2 checkpoint selection, hi/lo pred_loss ratio,
  geometry gate, /jepa-observability.
---

# JEPA Observability (v2 — paper-grounded)

JEPA runs are evaluated from `metrics.jsonl` `kind="training_step"` rows under the `lejepa/` prefix.
Use spike filtering (`mask_frac < 0.05`) for spike-sensitive metrics.

## Theoretical Foundation

- LeJEPA Theorem 1: isotropic Gaussian latent distribution is the optimal target.
- SIGReg is not only anti-collapse; it is the distributional quality objective.
- Primary discriminator across runs is the combined objective:
  `pred_loss + lambda * sigreg_loss` (LeJEPA Eq.10; reported with scaling in this repo as `combined_loss_scaled`).

## Scripts

```bash
python .cursor/skills/jepa-observability/scripts/extract_jepa_metrics.py sandbox/runs/<a> ...
python .cursor/skills/jepa-observability/scripts/plot_jepa_trajectories.py sandbox/runs/<a> ... --out /tmp/out.png
python .cursor/skills/log-observability/scripts/compare_configs.py sandbox/runs/<a>/resolved_config.yaml ...
```

## Metrics Reference — Primary

- `combined_loss_scaled`: top-level run discriminator. Lower is better.
- `pred_loss`: prediction error in projection space. Should trend down.
- `sigreg_loss`: isotropy loss. Should drop then stabilize.
- `encoder_eff_rank`: encoder dimensional usage. Large sustained decline indicates isotropy failure.
- `cov_condition_number`: isotropy anisotropy indicator from singular values. Lower is better.
- `embedding_mean_norm`: distance from zero-mean embedding target. Lower is better.
- `per_dim_variance_cv`: isotropy across dimensions. Near zero is better.
- `sigreg_projection_std`: spread of per-projection SIGReg statistics; high values imply unstable projection geometry.
- `adaLN_gamma_norm`: conditioning activity indicator (secondary in gate, useful for diagnosis).
- `meta_sens_runtype`: CANDI-specific biological sensitivity signal.

## Metrics Reference — Secondary (demoted)

Keep logged for continuity, but do not use as primary gate/ranking:

- `cos_sim_ctx_tgt`
- `sigreg_loss_pred_loss_ratio`
- `sigreg_to_pred_ratio`
- `pred_loss_hi_lo_ratio`
- `latent_std_mean`, `latent_n_dead` (superseded by condition-number and variance-CV metrics)
- `pred_loss_slope` and `sigreg_converged` are support diagnostics for gate dynamics.

## Geometry Gate (v2)

PASS requires all:

1. `pred_loss_slope <= 0` (combined objective not drifting up via prediction branch)
2. `sigreg_converged == 1`
3. `encoder_eff_rank_last >= 15`
4. `cov_condition_number_last < 50`

Legacy gate (`cos_sim + enc_er + gamma`) stays available only for backward compatibility.

## Collapse Cheat Sheet

- `sigreg_loss` rises late: isotropy objective failing; usually reduce predictor pressure or increase `lambda_sigreg`.
- `cov_condition_number > 100`: dimensional collapse despite training progress.
- `embedding_mean_norm` drifts up: distribution centering failure.
- `pred_loss_slope > 0`: prediction branch destabilizing; check LR/lambda tradeoff.
- `enc_er_delta < -0.5`: early collapse onset.
- `runtype_last / runtype_best < 0.4`: biological signal degrading.

## Run Ranking (paper-grounded)

Rank only with explicit citation of `combined_loss_scaled`.

Primary key:
1. Lower `combined_loss_scaled` (best)

Tiebreakers:
2. Lower `cov_condition_number_last`
3. Higher `encoder_eff_rank_last`
4. Higher `meta_sens_runtype_last`

Do not rank primarily by cosine similarity or loss ratios.

## Stage 2 Checkpoint Selection (v2)

Select checkpoint at minimum `combined_loss_scaled` among steps/epochs where `sigreg_converged == 1`.
If none converge, select minimum `combined_loss_scaled` and flag as unstable.

## Lambda Tuning Guide

- LeJEPA default around `lambda=0.05`.
- LeWM reports robust behavior in a moderate range and collapse for overly large lambda.
- In CANDI JEPA runs, prefer bisection-style search around current best value.
- Diagnose lambda with joint signals:
  - Under-regularized: `sigreg_loss` high, condition number rising.
  - Over-regularized: `pred_loss` stalls high while isotropy improves.

## Implementation Audit Notes (2026-05-15)

- SIGReg implementation uses LeWM-style setup (`knots=17`, `t in [0, 3]`, Gaussian window `exp(-t^2/2)`, trapezoidal weights).
- Characteristic-function mismatch uses both cosine and sine terms (equivalent to real+imaginary formulation).
- SIGReg is applied step-wise on `[L2, 2B, D]` via `sigreg(proj_all.transpose(0, 1))`.
- Prediction loss uses direct MSE between predicted and target projections with no stop-gradient on target projection.
- Effective-rank computation remains entropy-on-singular-values `exp(H(p))`.
- AdaLN gamma norm is computed on expanded `[B*L2, hidden]` scale for comparability.

## Workflow

1. Run extraction script and inspect v2 gate plus warnings.
2. Confirm config comparability (`compare_configs.py`) before claims.
3. Use trajectory plot panels to verify convergence shape, not only terminal values.
4. Rank candidates by `combined_loss_scaled` + v2 tiebreakers.
5. Cross-check UMAP/PCA only as supporting geometry evidence, never as sole rank criterion.

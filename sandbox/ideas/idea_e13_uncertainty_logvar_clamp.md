# E13 - GaussianLayer predicted variance floor

Status: done  
Parent: pval divergence investigation (F7)  
Run name: E13_var_floor (treatment) + E13_ctrl_var_floor (control)  
Checklist entry: [EXPERIMENTS.md](EXPERIMENTS.md#experiment-sweep-e6-e16)

## Problem Statement

The `GaussianLayer` in `model.py` predicts a per-position, per-assay variance `var` for the pval (arcsinh log-pvalue) head using `var = softplus(linear_var(x)) + 1e-6`. The `1e-6` epsilon is effectively zero — it does not prevent the model from collapsing `var` to near-zero on training assays.

When this happens, the Gaussian NLL `= 0.5 * [log(2π * var) + (y - mu)² / var]` becomes very negative on observed assays (the model is "rewarded" for being extremely confident), but catastrophically large on masked (imputed) assays where `mu` is also wrong and `(y - mu)² / var` explodes because `var ≈ 0`. This is the root cause of the obs/imp split and all late-stage pval divergence observed in F7.

Note: this is **not** about the loss uncertainty-weighting scalars (`logvar_count/pval/peak` in `CANDI_LOSS`) — those are a separate mechanism (see E14). This is about the **model output** — the predicted aleatoric variance for each position.

## Idea / Hypothesis

Raising the minimum predicted variance from `1e-6` to a meaningful floor (e.g. `0.01` or `0.1`) prevents variance collapse. The model can still predict low uncertainty on easy positions but cannot reach the degenerate near-zero regime that causes `pval_obs_loss → -∞` and `pval_imp_loss → ∞`.

## Planned Intervention

- File: `model.py`, class `GaussianLayer.forward` (~line 3315)
- Submit/config path: new config `sandbox/configs/e13_var_floor.yaml` (single key override if added to `ModelConfig`, or hard-coded change tested as default)
- Run name: TBD (e.g. `E13_gaussian_var_floor`)
- Parent run or idea: E7 (single-shot FiLM, current best multi-head baseline)

Code change in `GaussianLayer.forward` — `var_min` is now a constructor parameter:
```python
# Before (allows variance collapse):
var = F.softplus(var_logits) + 1e-6

# After (control run — same as before, explicit):
var = F.softplus(var_logits) + 1e-6   # gaussian_var_min=1e-6

# After (treatment run):
var = F.softplus(var_logits) + 0.1    # gaussian_var_min=0.1
```

Exposed as `model.gaussian_var_min: float = 1e-6` in `ModelConfig` (default preserves old behaviour). Set via YAML config (see `sandbox/configs/e13_ctrl.yaml` and `sandbox/configs/e13_var_floor.yaml`).

## Verifiables

- Validate if: `pval_imp_loss` no longer diverges (stays finite and < 5.0 throughout training); `pval_obs_loss` remains bounded above −2.0; `imp_pval_pearson` improves vs E7 baseline at 400 epochs.
- Disvalidate if: `pval_obs_loss` regresses significantly (clamp hurts denoising quality); `pval_imp` still diverges (floor too low); `imp_count_pearson` and `imp_peak_auroc` are unaffected (count and peak are unaffected by GaussianLayer — they use NB and BCE).
- **This change only affects the pval head** — count uses `NegativeBinomialLayer`, peak uses BCE. No cross-head effects expected.
- Required artifacts: `resolved_config.yaml`, `metrics.jsonl`, SLURM logs.

## Risks / Watch-outs

- A floor that is too high (e.g. `0.5`) will prevent the model from expressing high confidence even when warranted, potentially capping denoising quality.
- The `softplus + const` parametrization is well-established; this is a safe change. The production model's `GaussianLayer` should receive the same fix before any production runs that rely on calibrated pval uncertainty.
- Does NOT affect the loss uncertainty-weighting logvars (E14) — keep those separate.

## Run Links

- Run directory: `sandbox/runs/E13_var_floor/` (treatment), `sandbox/runs/E13_ctrl_var_floor/` (control)
- SLURM jobs: 38945712 (E13_var_floor), 38945711 (E13_ctrl_var_floor)
- Submit script: `sandbox/slurm/submit_experiments_e13.sh`
- Config files: `sandbox/configs/e13_var_floor.yaml`, `sandbox/configs/e13_ctrl.yaml`
- Resolved config: TBD (post-run)
- Metrics: TBD (post-run)
- W&B run: TBD (post-run)

## Findings

Completed 2026-05-06. See full cross-run analysis in [`synthesis_e13_var_floor.md`](synthesis_e13_var_floor.md).

- **Control (var_min=1e-6)**: Reproduced F7 exactly. `pval_obs_loss` → −0.111 at epoch 64 (variance collapse); `pval_imp_loss` → 45.5 at epoch 334. Diverged: YES.
- **Treatment (var_min=0.1)**: F7 mitigated. `pval_obs_loss` best=0.315, last=0.423 (never goes negative). `pval_imp_loss` best=0.476, last=0.671 (last/best ratio=1.40 < 1.5). Diverged: NO.
- **Quality**: Treatment `imp_pval_pearson_gw` best=0.306 (ep84) vs control 0.278 (ep64) — +10% improvement. `den_pval_pearson_gw` best: 0.445 vs 0.444 (negligible difference).
- **Gradient health**: Treatment has ~2.4× lower median pre-clip grad norm (2.78 vs 6.58) and lower clip fraction (0.598 vs 0.705). Bounded variance prevents anomalously large `(y-μ)²/var` gradients when `var → 0`.
- **Post-peak degradation**: Treatment `imp_pval_pearson_gw` declines from 0.306 at ep84 to 0.209 at ep354. Gentle, not catastrophic. Suggests early stopping would save GPU time.
- **Decision**: `gaussian_var_min=0.1` confirmed as effective in pval-only isolation. Must validate in multi-head training (E7+E13 run) before promoting as default. The 0.01 floor level is untested and may offer better expressivity with similar stability.

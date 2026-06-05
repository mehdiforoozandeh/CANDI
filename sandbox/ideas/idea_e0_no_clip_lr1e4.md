# E0 — No gradient clipping, lr=1e-4

Status: running  
Parent: B8 (baseline_E7_E13)  
Run name: E0_no_clip_lr1e4  
Checklist entry: [EXPERIMENTS.md](EXPERIMENTS.md#experiment-sweep-e0)

## Problem Statement

Current default uses `clip_norm=2.0` with Adamax `lr=1e-3`. Clip fraction across all recent runs is 0.60–0.71, meaning clipping fires on the majority of optimizer steps. When `clip_cap > 0` and the raw gradient norm exceeds the cap, **every parameter's gradient is scaled by the same factor** (`clip_cap / raw_norm`). This modifies the relative gradient direction across parameter groups — parameters in low-norm modules (e.g. FiLM/metadata path) are compressed proportionally to high-norm modules (e.g. decoder). This is a systematic bias whose magnitude is proportional to `1 - clip_cap/raw_norm` on clipped steps.

The high clip fraction (0.60–0.71) may be an artifact of the old pval variance-collapse driving gradient spikes. With `gaussian_var_min=0.1` now the default (E13), gradient norms may be substantially lower.

## Idea / Hypothesis

Disabling gradient clipping entirely (`clip_norm=0`) and using a smaller base LR (`lr=1e-4`, matching the old cosine LR floor = `lr × min_lr_ratio = 1e-3 × 0.1`) preserves the true gradient direction on every step, at the cost of slower convergence. If clipping was introducing meaningful bias in the metadata/FiLM path, removing it should improve `depth_count_ratio` or per-branch quality relative to B8.

Predicted direction: convergence slower than B8 (10× lower LR), but more stable late-stage behaviour (no direction-distortion); possibly better count/pval imputation if FiLM gradients were being suppressed by decoder-dominated clipping.

## Planned Intervention

- File/config: dotted CLI overrides only (no new YAML).
  - `--training.grad.clip_norm 0` — disables clipping (`clip_cap > 0.0` gate in `train.py`).
  - `--training.optimizer.adamax.lr 1e-4` — 10× lower than default.
- All other settings from `default.yaml` (E7 FiLM, E13 var floor, etc.).
- Regime: `sandbox/configs/type1_chr19.yaml`.
- Submit script: `sandbox/slurm/submit_experiments_b8_e0.sh`.

## Verifiables

- Validate if: training is stable without clipping (no NaN/Inf through 400 epochs); `imp_count_pearson` or `imp_pval_pearson` improves over B8 best epoch; or clip-related diagnostic metrics (post-run) show lower gradient distortion.
- Disvalidate if: loss diverges (gradient explosion without the clip safety net); or all metrics are uniformly worse than B8 (slower LR + no clip gives no quality benefit vs just slower convergence).
- Key diagnostic: does `training_stats/grad_pre_clip_norm` (logged even when clipping disabled) show lower median norms than B8? If yes, E13's variance floor was doing most of the work and clipping was mostly benign. If norms are similarly high, clipping was load-bearing.
- Required artifacts: `resolved_config.yaml`, `metrics.jsonl`, SLURM logs.

## Risks / Watch-outs

- Without a clipping safety net, a single gradient spike could send the model to a very poor region. Risk is reduced but not zero under `gaussian_var_min=0.1`.
- `lr=1e-4` combined with the cosine schedule means the final LR is `1e-4 × min_lr_ratio=0.1 = 1e-5`. At 400 epochs the model may not converge as far as B8. The comparison is best done at best epoch, not last epoch.
- This experiment confounds two changes (no clip + lower LR). The directional interpretation is therefore limited — if it helps, a follow-up single-axis test (no clip at lr=1e-3, or clip at lr=1e-4) is needed to attribute the cause.
- `depth_count_ratio` likely remains ≈ 1.0 (F1 is a model-level issue, not optimizer-level — but we'll check).

## Run Links

- Run directory: `sandbox/runs/E0_no_clip_lr1e4/`
- SLURM job: 39036547
- Submit script: `sandbox/slurm/submit_experiments_b8_e0.sh`
- Resolved config: TBD (post-run)
- Metrics: TBD (post-run)
- W&B run: TBD (post-run)

## Findings

- Observed: Walltime-killed at epoch 289 (SLURM 39036547). Best `eval_losses/total_loss=4.1463`; underperformed B8 (`3.04` @ ep149) and B1 (`4.97` @ ep59). `depth_count_ratio` ≈ 1.007 @ best.
- Interpretation: Removing clip while also lowering LR 10× confounds two knobs; run is stable but weak on reconstruction. Does not beat clipped B8.
- Competing explanations: Low LR may limit convergence within the 289-epoch budget; no-clip may allow noisier but more informative gradients that still fail to move imputation metrics.
- Decision: Rejected as a default change. Clean single-axis follow-up is E0b (no clip, `lr=1e-3`). See [`grad_clipping_summary.md`](grad_clipping_summary.md).

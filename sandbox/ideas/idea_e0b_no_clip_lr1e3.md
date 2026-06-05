# E0b — No gradient clipping, lr=1e-3 (default)

Status: running  
Parent: B8 (baseline_E7_E13)  
Run name: E0b_no_clip_lr1e3  
Checklist entry: [EXPERIMENTS.md](EXPERIMENTS.md#experiment-sweep-e0)

## Problem Statement

E0 (no clip + lr=1e-4) confounds two changes: removing clipping AND lowering the LR 10×. If E0 differs from B8, we cannot attribute the cause. E0b isolates the clipping effect by keeping `lr=1e-3` (default) and only disabling clipping, making it a clean single-axis test of whether gradient direction distortion from clipping is hurting training quality.

## Idea / Hypothesis

If gradient clipping at `clip_norm=2.0` is introducing meaningful directional bias (especially for low-norm modules like FiLM/metadata), removing it while keeping the same LR should improve quality metrics relative to B8. If it makes no difference or hurts (gradient explosion), clipping is load-bearing and the B8 defaults are correct.

Predicted direction: training may be less stable than E0 (no LR safety net) but more informative — any quality difference vs B8 is attributable to clipping alone.

## Planned Intervention

- Config: `--training.grad.clip_norm 0` only; all other defaults from `default.yaml`.
- `lr=1e-3` (unchanged — same as B8).
- Regime: `sandbox/configs/type1_chr19.yaml`.

## Verifiables

- Validate if: stable training without clipping at full LR; quality metrics comparable or better than B8.
- Disvalidate if: loss diverges or NaN/Inf occurs (gradient explosion without clip safety net at lr=1e-3).
- Key comparison: B8 (clip=2.0, lr=1e-3) vs E0b (no clip, lr=1e-3) — isolates clipping. E0 (no clip, lr=1e-4) vs E0b (no clip, lr=1e-3) — isolates LR under no-clip regime.
- Required artifacts: `resolved_config.yaml`, `metrics.jsonl`, SLURM logs.

## Risks / Watch-outs

- Higher risk of gradient explosion than E0 (no LR damping). The E13 variance floor reduces spike probability but does not eliminate it.
- If it diverges early (epoch < 20), the result is still informative: clipping at `clip_norm=2.0` is genuinely necessary at `lr=1e-3`.

## Run Links

- Run directory: `sandbox/runs/E0b_no_clip_lr1e3/`
- SLURM job: 39039621
- Submit script: `sandbox/slurm/submit_experiments_b8_e0.sh` (ad-hoc sbatch)
- Resolved config: TBD (post-run)
- Metrics: TBD (post-run)
- W&B run: TBD (post-run)

## Findings

- Observed: Best `eval_losses/total_loss=3.21` @ ep114; stable through cutoff but behind clipped B8 (`3.04` @ ep149, stable to ep289). `depth_count_ratio` ≈ 0.996 @ best.
- Interpretation: Clipping at `clip_norm=2.0` is load-bearing at full LR for matching B8 quality; disabling clip does not improve metadata sensitivity (F1 unchanged).
- Competing explanations: Residual gap vs B8 may reflect early-stop at best epoch 114 vs B8's longer stable basin, not a fundamental clip benefit.
- Decision: Keep clipped B8 defaults. See [`grad_clipping_summary.md`](grad_clipping_summary.md).

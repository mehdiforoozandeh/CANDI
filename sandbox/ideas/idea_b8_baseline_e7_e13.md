# B8 — Baseline: E7 + E13 defaults (multi-head reference)

Status: running  
Parent: baseline_anchor (B7-era), E7 (single-shot FiLM), E13 (gaussian_var_min=0.1)  
Run name: baseline_E7_E13  
Checklist entry: [EXPERIMENTS.md](EXPERIMENTS.md#baseline-sweep-b1-b8)

## Problem Statement

All runs since E6 use new defaults (`lr=1e-3`, `clip_norm=2.0`, `single_shot_decoder_film=True`) but no prior multi-head run has combined **both** promoted architectural fixes simultaneously in isolation:
- E7 used the single-shot FiLM but pre-dated the E13 variance floor.
- E13 tested the variance floor but in pval-only isolation.

A clean multi-head reference run under the current `default.yaml` is needed to:
1. Measure whether pval_imp divergence (F7) is eliminated in the multi-head context.
2. Establish the new cornerstone reference point for all future ablations (superseding baseline_400ep).
3. Quantify the combined E7+E13 gain vs `baseline_400ep` (which used old defaults and no variance floor).

## Idea / Hypothesis

A multi-head run with `single_shot_decoder_film=True` and `gaussian_var_min=0.1` should combine the delayed-divergence benefit of E7 (pval_imp onset ep94 vs ep29 for old baseline) with the variance-collapse prevention of E13 (pval_imp never diverged in pval-only isolation). The result should be the most stable and highest-quality multi-head run to date.

## Planned Intervention

- Config: `sandbox/configs/default.yaml` only (no extra overrides — this IS the new default).
- Regime: `sandbox/configs/type1_chr19.yaml`.
- Submit script: `sandbox/slurm/submit_experiments_b8_e0.sh`.

## Verifiables

- Validate if: no pval_imp divergence through 400 epochs; `imp_peak_auroc` and `imp_count_pearson` match or exceed E7 best epoch; `quality_score` computable (multi-head → total_loss present).
- Disvalidate if: pval_imp diverges before epoch 100 (suggests multi-head gradient competition overwhelms the variance floor).
- Key comparison: E7 best epoch (ep84): `imp_peak_auroc=0.765`, `imp_count_pearson=0.339`, `imp_pval_pearson=0.277`.
- Required artifacts: `resolved_config.yaml`, `metrics.jsonl`, SLURM logs.

## Risks / Watch-outs

- If pval_imp still diverges, it confirms multi-head gradient competition is a separate issue from the variance floor, and E14 (six logvars) or `pval_weight` soft-mute becomes the next priority.
- `depth_count_ratio` expected to remain ≈ 1.0 (F1 is unresolved and not targeted here).

## Run Links

- Run directory: `sandbox/runs/baseline_E7_E13/`
- SLURM job: 39036546
- Submit script: `sandbox/slurm/submit_experiments_b8_e0.sh`
- Resolved config: TBD (post-run)
- Metrics: TBD (post-run)
- W&B run: TBD (post-run)

## Findings

- Observed: Walltime-killed at epoch 289 (SLURM 39036546). Best `eval_losses/total_loss=3.3066` (ep259), last 3.4214. No late pval explosion (`pval_imp_loss` ~0.51–0.64). Below E7 400ep reference on peak/count/pval at best epoch.
- Interpretation: E7+E13 defaults stabilize multi-head training through 289 epochs but do not yet match E7 standalone best-epoch quality.
- Competing explanations: Shorter budget vs E7 400ep run; multi-head gradient competition still limits pval/count peaks despite variance floor.
- Decision: **Partial success** — use as interim multi-head reference; B8 run incomplete at 400ep target.

# E2 - Count head only

Status: done  
Parent: baseline_anchor  
Run name: `E2_head_count_only`  
Checklist entry: [EXPERIMENTS.md](EXPERIMENTS.md#experiment-sweep-e1-e5)

## Problem Statement

Isolate whether the count branch can learn when pval and peak losses are muted.

## Idea / Hypothesis

If multi-head competition is hurting counts, count-only training should improve count losses and count Pearson metrics.

## Planned Intervention

- Submit/config path: [`../slurm/submit_experiments_e1_e5.sh`](../slurm/submit_experiments_e1_e5.sh)
- Run directory: [`../runs/E2_head_count_only/`](../runs/E2_head_count_only/)
- Parent run or idea: `baseline_anchor`
- Config/code/data deltas: log1p type1 run with `pval_weight=0.0` and `peak_weight=0.0`; parent set to `baseline_anchor`.

## Verifiables

- Validate if: The active branch metrics move in the predicted direction and the run produces complete, comparable artifacts for that experiment type.
- Disvalidate if: The active branch metrics do not improve, artifacts are incomplete, or disabled-head behavior prevents the intended comparison.
- Specific checks: use count branch metrics only; aggregate `eval_losses/total_loss` and `quality_score` are not emitted for this head-isolation run.
- Required artifacts: `resolved_config.yaml`, `metrics.jsonl`, SLURM logs, and `hpo_graph.json` when available.

## Risks / Watch-outs

- Disabled heads emit NaN in training branch fields, so branch-specific interpretation is required.
- Untrained pval/peak metrics are diagnostic only, not success criteria.
- Graph node exists but has no aggregate result fields.

## Run Links

- Run directory: [`../runs/E2_head_count_only/`](../runs/E2_head_count_only/)
- Resolved config: [`../runs/E2_head_count_only/resolved_config.yaml`](../runs/E2_head_count_only/resolved_config.yaml)
- Metrics: [`../runs/E2_head_count_only/metrics.jsonl`](../runs/E2_head_count_only/metrics.jsonl)
- SLURM logs: `../slurm_logs/baseline_sbx_e2_count_37595464.*`
- HPO graph: [`../hpo_graph.json`](../hpo_graph.json)
- W&B run name: `E2_head_count_only`

## Findings

Cross-run rollup: see [`synthesis_e1_e5_head_interference.md`](synthesis_e1_e5_head_interference.md).

### Run 1 (200 epochs, old defaults, SLURM job 37595464)

count_imp_loss best=1.737 @ ep59, imp_count_pearson peak=0.115 (~2× B7). No divergence. Marked done.

### Run 2 (400 epochs, new defaults lr=1e-3, clip=2.0, SLURM job 38829773) — **current**

- Walltime-killed at epoch 354 of 400 (~89% budget). No divergence (count head only — no total_loss in ranker).
- `count_obs_loss`: first=1.240, last=0.799 @ ep354 — **still declining at kill**, not converged.
- `count_imp_loss`: first=1.909, last=1.920 — **flat throughout all 354 epochs**. Plateaued near 1.92 from ep~50.
- `imp_count_pearson`: best=0.319 @ ep354 (last epoch, still improving slightly). `den_count_pearson` best=0.558 @ ep354.
- Grad-norm: pre-clip median=4.38, p95=19.7, max=45.1. clip_fraction=0.83.
- `depth_count_ratio` last=0.982 — metadata collapse persists (F1).
- Interpretation: The obs/imp split is the dominant signal. `count_obs` (denoising on T_*) continues to improve monotonically while `count_imp` (imputation of V/B masked assays) plateaus at ~1.92 from ep50 onward. This is a fundamental generalization gap: the model can increasingly well denoise observed counts but has a ceiling on cross-assay imputation. The imp Pearson still improving slightly at ep354 (0.319 vs 0.282 at ep304) suggests convergence hasn't been reached for the Pearson metric even as the NLL has plateaued.
- Decision: **Count head is the most stable but hardest to improve on imputation.** The imputation ceiling (count_imp ~1.92) requires architectural or objective changes, not just more compute. Keep E2 as the isolated-count ceiling reference. Follow-ups: (a) soft-mute pval+peak at weight 0.3 in multi-head setting; (b) investigate whether the count NB distribution is miscalibrated on masked assays.

# B6 - Clip by value

Status: done  
Parent: none  
Run name: `baseline_clip_value_type1`  
Checklist entry: [EXPERIMENTS.md](EXPERIMENTS.md#baseline-sweep-b1-b7)

## Problem Statement

Test whether gradient clipping mode affects best-epoch quality and divergence under the raw-input type1 baseline.

## Idea / Hypothesis

Value clipping may control branch gradients differently than norm clipping and improve best-epoch losses.

## Planned Intervention

- Submit/config path: [`../slurm/submit_baselines.sh`](../slurm/submit_baselines.sh)
- Run directory: [`../runs/baseline_clip_value_type1/`](../runs/baseline_clip_value_type1/)
- Parent run or idea: `none`
- Config/code/data deltas: B1 plus `training.grad.clip_mode=value`.

## Verifiables

- Validate if: The changed knob improves best-epoch quality or stability versus the intended comparator without introducing NaN/Inf or a major branch regression.
- Disvalidate if: The run diverges, worsens the composite score, or improves only one branch while degrading the core imputation losses.
- Specific checks: compare against B1 on `quality_score`, per-branch losses, divergence, and grad/clip diagnostics when needed.
- Required artifacts: `resolved_config.yaml`, `metrics.jsonl`, SLURM logs, and `hpo_graph.json` when available.

## Risks / Watch-outs

- Value clipping may produce better early losses while hiding unstable dynamics.
- Depth metadata still appears ignored.
- Needs grad-norm/clip inspection before causal claims about clipping.

## Run Links

- Run directory: [`../runs/baseline_clip_value_type1/`](../runs/baseline_clip_value_type1/)
- Resolved config: [`../runs/baseline_clip_value_type1/resolved_config.yaml`](../runs/baseline_clip_value_type1/resolved_config.yaml)
- Metrics: [`../runs/baseline_clip_value_type1/metrics.jsonl`](../runs/baseline_clip_value_type1/metrics.jsonl)
- SLURM logs: `../slurm_logs/baseline_sbx_b6_clipval_*.out`
- HPO graph: [`../hpo_graph.json`](../hpo_graph.json)
- W&B run name: `baseline_clip_value_type1`

## Findings

- Observed: Best B-sweep composite: `quality_score=8.018915` and best `eval_losses/total_loss=4.573696` at epoch 49, with improved `pval_imp_loss=0.591870` and `count_imp_loss=1.732927`; nevertheless it diverged to last `eval_losses/total_loss=12.002621` and `depth_count_ratio=0.990147` failed.
- Interpretation: Promising quality signal but not stable enough; use as a follow-up seed for stabilization rather than as a keeper.
- Competing explanations: metric movement may reflect the changed data/control knob, stochastic run noise, or known sandbox limitations such as weak depth metadata response.
- Decision: Promising quality signal but not stable enough; use as a follow-up seed for stabilization rather than as a keeper.

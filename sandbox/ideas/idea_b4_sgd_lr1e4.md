# B4 - SGD lr 1e-4

Status: done  
Parent: none  
Run name: `baseline_sgd_lr1e4`  
Checklist entry: [EXPERIMENTS.md](EXPERIMENTS.md#baseline-sweep-b1-b7)

## Problem Statement

Check whether Adamax is a source of late divergence by replacing it with a conservative SGD optimizer.

## Idea / Hypothesis

SGD at `lr=1e-4` should be slower but more stable if optimizer dynamics are driving divergence.

## Planned Intervention

- Submit/config path: [`../slurm/submit_baselines.sh`](../slurm/submit_baselines.sh)
- Run directory: [`../runs/baseline_sgd_lr1e4/`](../runs/baseline_sgd_lr1e4/)
- Parent run or idea: `none`
- Config/code/data deltas: B1 plus `training.optimizer.name=sgd` and `training.optimizer.sgd.lr=1e-4`.

## Verifiables

- Validate if: The changed knob improves best-epoch quality or stability versus the intended comparator without introducing NaN/Inf or a major branch regression.
- Disvalidate if: The run diverges, worsens the composite score, or improves only one branch while degrading the core imputation losses.
- Specific checks: look for no divergence, acceptable `quality_score`, and whether stability comes at unacceptable pval/count degradation.
- Required artifacts: `resolved_config.yaml`, `metrics.jsonl`, SLURM logs, and `hpo_graph.json` when available.

## Risks / Watch-outs

- Slow optimizer may mask representational issues rather than fix them.
- Pval degradation is large enough to matter despite stable total loss.
- Best epoch near the end means walltime and convergence rate should be monitored.

## Run Links

- Run directory: [`../runs/baseline_sgd_lr1e4/`](../runs/baseline_sgd_lr1e4/)
- Resolved config: [`../runs/baseline_sgd_lr1e4/resolved_config.yaml`](../runs/baseline_sgd_lr1e4/resolved_config.yaml)
- Metrics: [`../runs/baseline_sgd_lr1e4/metrics.jsonl`](../runs/baseline_sgd_lr1e4/metrics.jsonl)
- SLURM logs: `../slurm_logs/baseline_sbx_b4_sgd_*.out`
- HPO graph: [`../hpo_graph.json`](../hpo_graph.json)
- W&B run name: `baseline_sgd_lr1e4`

## Findings

- Observed: Completed 200 epochs without divergence: best `eval_losses/total_loss=4.899788` at epoch 194 and last `5.098948`; best `quality_score=8.705782`. Peak losses improved (`peak_imp_loss=0.403254`), but pval imputation was weak (`pval_imp_loss=1.025943`). `depth_count_ratio=0.996209` still failed.
- Interpretation: Useful stability control, but not a direct winner because pval quality regressed and depth conditioning remained weak.
- Competing explanations: metric movement may reflect the changed data/control knob, stochastic run noise, or known sandbox limitations such as weak depth metadata response.
- Decision: Useful stability control, but not a direct winner because pval quality regressed and depth conditioning remained weak.

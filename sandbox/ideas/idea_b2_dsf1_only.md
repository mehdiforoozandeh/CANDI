# B2 - DSF sampling off

Status: done  
Parent: none  
Run name: `baseline_dsf1_only`  
Checklist entry: [EXPERIMENTS.md](EXPERIMENTS.md#baseline-sweep-b1-b7)

## Problem Statement

Determine whether dynamic DSF sampling is helping or destabilizing the type1 baseline.

## Idea / Hypothesis

Forcing DSF sampling off isolates whether DSF variation contributes to loss instability or weak metadata conditioning.

## Planned Intervention

- Submit/config path: [`../slurm/submit_baselines.sh`](../slurm/submit_baselines.sh)
- Run directory: [`../runs/baseline_dsf1_only/`](../runs/baseline_dsf1_only/)
- Parent run or idea: `none`
- Config/code/data deltas: B1 plus `training.dsf.sampling=off`, keeping raw input and type1 chr19.

## Verifiables

- Validate if: The changed knob improves best-epoch quality or stability versus the intended comparator without introducing NaN/Inf or a major branch regression.
- Disvalidate if: The run diverges, worsens the composite score, or improves only one branch while degrading the core imputation losses.
- Specific checks: compare against B1 on `quality_score`, per-branch imputation losses, divergence, and `depth_count_ratio`.
- Required artifacts: `resolved_config.yaml`, `metrics.jsonl`, SLURM logs, and `hpo_graph.json` when available.

## Risks / Watch-outs

- The improved best epoch may be a transient before severe divergence.
- DSF-off changes the data distribution and may not generalize to intended DSF behavior.
- Depth conditioning still appears ignored.

## Run Links

- Run directory: [`../runs/baseline_dsf1_only/`](../runs/baseline_dsf1_only/)
- Resolved config: [`../runs/baseline_dsf1_only/resolved_config.yaml`](../runs/baseline_dsf1_only/resolved_config.yaml)
- Metrics: [`../runs/baseline_dsf1_only/metrics.jsonl`](../runs/baseline_dsf1_only/metrics.jsonl)
- SLURM logs: `../slurm_logs/baseline_sbx_b2_dsf1_*.out`
- HPO graph: [`../hpo_graph.json`](../hpo_graph.json)
- W&B run name: `baseline_dsf1_only`

## Findings

- Observed: Best `quality_score=8.185813` improved versus B1 `8.752951`, mostly through lower `count_imp_loss=1.617025` and `peak_imp_loss=0.584265`; however it diverged badly (`eval_losses/total_loss` 4.677017 best to 53.648513 last) and `depth_count_ratio=0.994405` remained failed.
- Interpretation: Not a keeper as-is; useful evidence that DSF-off can improve best-epoch losses but worsens late instability.
- Competing explanations: metric movement may reflect the changed data/control knob, stochastic run noise, or known sandbox limitations such as weak depth metadata response.
- Decision: Not a keeper as-is; useful evidence that DSF-off can improve best-epoch losses but worsens late instability.

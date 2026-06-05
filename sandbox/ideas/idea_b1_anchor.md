# B1 - Anchor: type1 chr19, raw input

Status: done  
Parent: none  
Run name: `baseline_anchor`  
Checklist entry: [EXPERIMENTS.md](EXPERIMENTS.md#baseline-sweep-b1-b7)

## Problem Statement

Establish the raw-input type1 chr19 reference point before changing one knob at a time.

## Idea / Hypothesis

The untransformed input baseline reveals whether the sandbox training loop is stable enough to compare later ablations.

## Planned Intervention

- Submit/config path: [`../slurm/submit_baselines.sh`](../slurm/submit_baselines.sh)
- Run directory: [`../runs/baseline_anchor/`](../runs/baseline_anchor/)
- Parent run or idea: `none`
- Config/code/data deltas: type1 chr19, `model.encode_input_transform=none`, default Adamax, default masking, default DSF sampling.

## Verifiables

- Validate if: The changed knob improves best-epoch quality or stability versus the intended comparator without introducing NaN/Inf or a major branch regression.
- Disvalidate if: The run diverges, worsens the composite score, or improves only one branch while degrading the core imputation losses.
- Specific checks: completed 200 epochs; inspect `eval_losses/total_loss` divergence, best-epoch `quality_score`, `training_metadata_probes/depth_count_ratio`, and NaN/Inf count.
- Required artifacts: `resolved_config.yaml`, `metrics.jsonl`, SLURM logs, and `hpo_graph.json` when available.

## Risks / Watch-outs

- Raw input may make count scale harder than later log1p runs.
- Depth metadata failure (`depth_count_ratio` near 1.0) may dominate conclusions.
- Divergence means best-epoch metrics should not be confused with stable convergence.

## Run Links

- Run directory: [`../runs/baseline_anchor/`](../runs/baseline_anchor/)
- Resolved config: [`../runs/baseline_anchor/resolved_config.yaml`](../runs/baseline_anchor/resolved_config.yaml)
- Metrics: [`../runs/baseline_anchor/metrics.jsonl`](../runs/baseline_anchor/metrics.jsonl)
- SLURM logs: `../slurm_logs/baseline_sbx_b1_anchor_*.out`
- HPO graph: [`../hpo_graph.json`](../hpo_graph.json)
- W&B run name: `baseline_anchor`

## Findings

- Observed: Completed 200 epochs with no NaN/Inf in eval families, but diverged: best `eval_losses/total_loss=4.972334` at epoch 59 versus last `10.012297`; best `quality_score=8.752951`; `depth_count_ratio=1.008944` shows weak depth response.
- Interpretation: Use as the raw-input anchor only; not a stable keeper because last total loss is >1.5x best.
- Competing explanations: metric movement may reflect the changed data/control knob, stochastic run noise, or known sandbox limitations such as weak depth metadata response.
- Decision: Use as the raw-input anchor only; not a stable keeper because last total loss is >1.5x best.

# B5 - Type2 loci only

Status: done  
Parent: none  
Run name: `baseline_type2_loci_only`  
Checklist entry: [EXPERIMENTS.md](EXPERIMENTS.md#baseline-sweep-b1-b7)

## Problem Statement

Measure whether the type2 loci regime behaves differently from type1 chr19 under otherwise similar baseline settings.

## Idea / Hypothesis

The type2 loci overlay may expose regime-specific failures or improve locus-centered learning relative to type1 tiling.

## Planned Intervention

- Submit/config path: [`../slurm/submit_baselines.sh`](../slurm/submit_baselines.sh)
- Run directory: [`../runs/baseline_type2_loci_only/`](../runs/baseline_type2_loci_only/)
- Parent run or idea: `none`
- Config/code/data deltas: B1 with `sandbox/configs/type2_loci.yaml` and raw input.

## Verifiables

- Validate if: The changed knob improves best-epoch quality or stability versus the intended comparator without introducing NaN/Inf or a major branch regression.
- Disvalidate if: The run diverges, worsens the composite score, or improves only one branch while degrading the core imputation losses.
- Specific checks: compare total loss, imputation losses, peak AUROC, divergence, and whether type2 creates distinct failure modes.
- Required artifacts: `resolved_config.yaml`, `metrics.jsonl`, SLURM logs, and `hpo_graph.json` when available.

## Risks / Watch-outs

- Type2 is not directly comparable to type1 because data regime changes.
- Early best epoch plus severe divergence suggests the schedule/optimizer may be mismatched.
- Raw input may be especially problematic in the type2 regime.

## Run Links

- Run directory: [`../runs/baseline_type2_loci_only/`](../runs/baseline_type2_loci_only/)
- Resolved config: [`../runs/baseline_type2_loci_only/resolved_config.yaml`](../runs/baseline_type2_loci_only/resolved_config.yaml)
- Metrics: [`../runs/baseline_type2_loci_only/metrics.jsonl`](../runs/baseline_type2_loci_only/metrics.jsonl)
- SLURM logs: `../slurm_logs/baseline_sbx_b5_type2_*.out`
- HPO graph: [`../hpo_graph.json`](../hpo_graph.json)
- W&B run name: `baseline_type2_loci_only`

## Findings

- Observed: This was the weakest B sweep run by composite: best `quality_score=10.368331`, best `eval_losses/total_loss=6.095254` at epoch 14, and last `73.201348` with divergence. `imp_peak_auroc_gw=0.439698` and `imp_count_pearson_gw=-0.006819` were poor at the best epoch.
- Interpretation: Do not use this raw-input type2 setup as a keeper; treat it as evidence that type2 needs separate tuning.
- Competing explanations: metric movement may reflect the changed data/control knob, stochastic run noise, or known sandbox limitations such as weak depth metadata response.
- Decision: Do not use this raw-input type2 setup as a keeper; treat it as evidence that type2 needs separate tuning.

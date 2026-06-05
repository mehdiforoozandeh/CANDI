# B7 - Log1p type1 baseline

Status: done  
Parent: none  
Run name: `baseline_log1p_type1`  
Checklist entry: [EXPERIMENTS.md](EXPERIMENTS.md#baseline-sweep-b1-b7)

## Problem Statement

Test whether log1p input scaling stabilizes the type1 baseline compared with raw input.

## Idea / Hypothesis

`model.encode_input_transform=log1p` should reduce count-scale pathologies and improve stability without large quality loss.

## Planned Intervention

- Submit/config path: [`../slurm/submit_baselines.sh`](../slurm/submit_baselines.sh)
- Run directory: [`../runs/baseline_log1p_type1/`](../runs/baseline_log1p_type1/)
- Parent run or idea: `none`
- Config/code/data deltas: B1 plus `model.encode_input_transform=log1p`.

## Verifiables

- Validate if: The changed knob improves best-epoch quality or stability versus the intended comparator without introducing NaN/Inf or a major branch regression.
- Disvalidate if: The run diverges, worsens the composite score, or improves only one branch while degrading the core imputation losses.
- Specific checks: look for no divergence, competitive `quality_score`, improved count/peak metrics, and unchanged NaN/Inf health.
- Required artifacts: `resolved_config.yaml`, `metrics.jsonl`, SLURM logs, and `hpo_graph.json` when available.

## Risks / Watch-outs

- Depth metadata failure remains unresolved.
- Quality still trails unstable B6 best epoch.
- Log1p changes input scale, so comparisons to raw-input B1-B6 should call out that confound.

## Run Links

- Run directory: [`../runs/baseline_log1p_type1/`](../runs/baseline_log1p_type1/)
- Resolved config: [`../runs/baseline_log1p_type1/resolved_config.yaml`](../runs/baseline_log1p_type1/resolved_config.yaml)
- Metrics: [`../runs/baseline_log1p_type1/metrics.jsonl`](../runs/baseline_log1p_type1/metrics.jsonl)
- SLURM logs: `../slurm_logs/baseline_sbx_b7_log1p_*.out`
- HPO graph: [`../hpo_graph.json`](../hpo_graph.json)
- W&B run name: `baseline_log1p_type1`

## Findings

- Observed: Completed 200 epochs without divergence: best `eval_losses/total_loss=4.860360`, last `5.708084`, best `quality_score=8.501605`. It improved over B1 on `pval_imp_loss` (0.617486 versus 0.710964), `imp_count_pearson_gw` (0.053366 versus 0.008429), and `imp_peak_auroc_gw` (0.521821 versus 0.502402), but `depth_count_ratio=0.996246` still failed.
- Interpretation: Best stable type1 baseline among B runs; use as a stronger parent than raw B1 unless a future graph lineage requires B1.
- Competing explanations: metric movement may reflect the changed data/control knob, stochastic run noise, or known sandbox limitations such as weak depth metadata response.
- Decision: Best stable type1 baseline among B runs; use as a stronger parent than raw B1 unless a future graph lineage requires B1.

# B3 - Assay masking only

Status: done  
Parent: none  
Run name: `baseline_assay_mask_only`  
Checklist entry: [EXPERIMENTS.md](EXPERIMENTS.md#baseline-sweep-b1-b7)

## Problem Statement

Test whether full-locus masking is too hard/noisy compared with assay-only masking for the current sandbox scale.

## Idea / Hypothesis

`p_full_assay=1.0` and `p_full_loci=0.0` may improve assay-level denoising/imputation by removing locus-level corruption pressure.

## Planned Intervention

- Submit/config path: [`../slurm/submit_baselines.sh`](../slurm/submit_baselines.sh)
- Run directory: [`../runs/baseline_assay_mask_only/`](../runs/baseline_assay_mask_only/)
- Parent run or idea: `none`
- Config/code/data deltas: B1 plus `training.masking.p_full_assay=1.0` and `training.masking.p_full_loci=0.0`.

## Verifiables

- Validate if: The changed knob improves best-epoch quality or stability versus the intended comparator without introducing NaN/Inf or a major branch regression.
- Disvalidate if: The run diverges, worsens the composite score, or improves only one branch while degrading the core imputation losses.
- Specific checks: compare against B1 on imputation losses, peak AUROC, divergence, and whether gains are branch-specific or broad.
- Required artifacts: `resolved_config.yaml`, `metrics.jsonl`, SLURM logs, and `hpo_graph.json` when available.

## Risks / Watch-outs

- A small peak AUROC gain may not offset composite loss degradation.
- Divergence limits confidence in late-epoch behavior.
- Assay-only masking may under-test locus imputation behavior.

## Run Links

- Run directory: [`../runs/baseline_assay_mask_only/`](../runs/baseline_assay_mask_only/)
- Resolved config: [`../runs/baseline_assay_mask_only/resolved_config.yaml`](../runs/baseline_assay_mask_only/resolved_config.yaml)
- Metrics: [`../runs/baseline_assay_mask_only/metrics.jsonl`](../runs/baseline_assay_mask_only/metrics.jsonl)
- SLURM logs: `../slurm_logs/baseline_sbx_b3_assay_*.out`
- HPO graph: [`../hpo_graph.json`](../hpo_graph.json)
- W&B run name: `baseline_assay_mask_only`

## Findings

- Observed: Best `quality_score=8.771452` was slightly worse than B1 `8.752951`; it diverged (`eval_losses/total_loss` 4.988122 best to 12.412429 last). Peak AUROC improved slightly (`imp_peak_auroc_gw=0.512034` versus B1 0.502402), but the composite did not improve.
- Interpretation: Do not promote assay-only masking as a broad baseline replacement; keep as branch-specific evidence.
- Competing explanations: metric movement may reflect the changed data/control knob, stochastic run noise, or known sandbox limitations such as weak depth metadata response.
- Decision: Do not promote assay-only masking as a broad baseline replacement; keep as branch-specific evidence.

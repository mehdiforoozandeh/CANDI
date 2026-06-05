# E1 - Lower LR floor

Status: incomplete  
Parent: baseline_anchor  
Run name: `E1_lrfloor_low`  
Checklist entry: [EXPERIMENTS.md](EXPERIMENTS.md#experiment-sweep-e1-e5)

## Problem Statement

Test whether the cosine LR floor is too high and contributes to late-stage divergence.

## Idea / Hypothesis

Lowering `training.schedule.min_lr_ratio` from 0.1 to 0.01 should reduce late training pressure while preserving B7-like quality.

## Planned Intervention

- Submit/config path: [`../slurm/submit_experiments_e1_e5.sh`](../slurm/submit_experiments_e1_e5.sh)
- Run directory: [`../runs/E1_lrfloor_low/`](../runs/E1_lrfloor_low/)
- Parent run or idea: `baseline_anchor`
- Config/code/data deltas: log1p type1 run with `training.schedule.min_lr_ratio=0.01`, submitted as parent `baseline_anchor` in the HPO settings.

## Verifiables

- Validate if: The active branch metrics move in the predicted direction and the run produces complete, comparable artifacts for that experiment type.
- Disvalidate if: The active branch metrics do not improve, artifacts are incomplete, or disabled-head behavior prevents the intended comparison.
- Specific checks: compare with B7 on divergence, best/last total loss, quality score, and walltime completion.
- Required artifacts: `resolved_config.yaml`, `metrics.jsonl`, SLURM logs, and `hpo_graph.json` when available.

## Risks / Watch-outs

- Incomplete run is not a full 200-epoch comparison.
- Parent is recorded as `baseline_anchor` despite log1p behavior matching B7 more closely.
- Graph metadata is missing, so lineage must be documented manually.

## Run Links

- Run directory: [`../runs/E1_lrfloor_low/`](../runs/E1_lrfloor_low/)
- Resolved config: [`../runs/E1_lrfloor_low/resolved_config.yaml`](../runs/E1_lrfloor_low/resolved_config.yaml)
- Metrics: [`../runs/E1_lrfloor_low/metrics.jsonl`](../runs/E1_lrfloor_low/metrics.jsonl)
- SLURM logs: `../slurm_logs/baseline_sbx_e1_lrfloor_37595463.*`
- HPO graph: [`../hpo_graph.json`](../hpo_graph.json)
- W&B run name: `E1_lrfloor_low`

## Findings

Cross-run rollup: see [`synthesis_e1_e5_head_interference.md`](synthesis_e1_e5_head_interference.md).

- Observed: Ran 165 epochs (walltime-killed; `baseline_sbx_e1_lrfloor_37595463.err: CANCELLED ... DUE TO TIME LIMIT`). Did not diverge: `eval_losses/total_loss` first=6.1941, best=4.8657 @ epoch 74, last=5.4974. Cornerstone `quality_score=8.5099` is essentially tied with B7's `8.5016` (Δ=+0.10% relative; `rank_runs.py` verdict on B7 vs E1: `tie`, decided only by Tier-4 efficiency where both reach min total at `global_step=7125`). Per-branch deltas vs B7: `pval_imp_loss` 0.6192 vs 0.6175; `count_imp_loss` 1.7972 vs 1.7972 (same epoch 14); `peak_imp_loss` 0.4895 vs 0.4530. Pre-clip grad-norm trajectory matches B7 (median 32.8 vs 33.4; p95 220.8 vs 220.9 from `inspect_training_steps.py`). `imp_pval_pearson_gw` peak 0.1729 vs B7 0.1730. `depth_count_ratio` last=0.9976 (F1 still open). Backfilled HPO node now present.
- Interpretation: Lowering `min_lr_ratio` from 0.1 to 0.01 produced **no measurable change** under log1p input transform — neither stability (B7 already non-divergent under log1p) nor metric ceiling moved. **F3-as-LR-floor hypothesis is rejected for log1p baselines.** F3's late-stage divergence pattern was specific to runs *without* log1p (B1/B3/B6), so log1p alone appears to cover F3's stability concern, leaving min_lr_ratio inert at this scale.
- Competing explanations: (a) chr19 / 200-epoch sandbox is too short for the LR-floor effect to manifest — would need a ≥500-epoch or larger-data setting to detect; (b) the cosine schedule's bottom 10% of training is small enough at 200 epochs (~20 epochs) that the difference between 0.1 and 0.01 of base LR is dominated by gradient noise; (c) min_lr_ratio matters only for non-log1p inputs and is masked by the dominant log1p stabilization.
- Decision: Drop `min_lr_ratio=0.01` from the active search. Treat F3 as resolved-by-log1p for chr19-scale runs and re-test only when a non-log1p or longer-horizon run shows late divergence again. No follow-up E1' rerun.

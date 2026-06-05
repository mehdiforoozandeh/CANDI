# E5 - Count plus peak heads

Status: incomplete  
Parent: baseline_anchor  
Run name: `E5_head_count_peak`  
Checklist entry: [EXPERIMENTS.md](EXPERIMENTS.md#experiment-sweep-e1-e5)

## Problem Statement

Test whether muting pval while keeping count and peak improves the cleaner count/peak objectives.

## Idea / Hypothesis

If pval is noisy or dominant, count+peak training should improve count and peak branch behavior relative to the full multi-head run.

## Planned Intervention

- Submit/config path: [`../slurm/submit_experiments_e1_e5.sh`](../slurm/submit_experiments_e1_e5.sh)
- Run directory: [`../runs/E5_head_count_peak/`](../runs/E5_head_count_peak/)
- Parent run or idea: `baseline_anchor`
- Config/code/data deltas: log1p type1 run with `pval_weight=0.0`; parent set to `baseline_anchor`.

## Verifiables

- Validate if: The active branch metrics move in the predicted direction and the run produces complete, comparable artifacts for that experiment type.
- Disvalidate if: The active branch metrics do not improve, artifacts are incomplete, or disabled-head behavior prevents the intended comparison.
- Specific checks: use count and peak branch metrics; aggregate total loss/quality are not emitted, and walltime completion matters.
- Required artifacts: `resolved_config.yaml`, `metrics.jsonl`, SLURM logs, and `hpo_graph.json` when available.

## Risks / Watch-outs

- Timed out before 200 epochs.
- No aggregate total loss/quality score for standard ranking.
- Missing graph node requires manual lineage documentation.

## Run Links

- Run directory: [`../runs/E5_head_count_peak/`](../runs/E5_head_count_peak/)
- Resolved config: [`../runs/E5_head_count_peak/resolved_config.yaml`](../runs/E5_head_count_peak/resolved_config.yaml)
- Metrics: [`../runs/E5_head_count_peak/metrics.jsonl`](../runs/E5_head_count_peak/metrics.jsonl)
- SLURM logs: `../slurm_logs/baseline_sbx_e5_countpeak_37595467.*`
- HPO graph: [`../hpo_graph.json`](../hpo_graph.json)
- W&B run name: `E5_head_count_peak`

## Findings

Cross-run rollup: see [`synthesis_e1_e5_head_interference.md`](synthesis_e1_e5_head_interference.md).

- Observed: Ran 189 epochs (walltime-killed; `CANCELLED ... DUE TO TIME LIMIT`). Count + peak branches only. `count_imp_loss` best=1.7177 @ epoch 99 vs B7 1.7972, vs E2 1.7367 — **best of any run on this metric** (4.4% improvement vs B7, 1.1% improvement vs E2-count-only). `count_obs_loss` best=1.0791 @ epoch 84 vs B7 1.1262 vs E2 1.0491. `peak_imp_loss` best=0.4591 @ epoch 184 vs B7 0.4530 vs E4 0.5539 — essentially **tied with B7** (1.3% regression) and **17% better than E4-peak-only**. `peak_obs_loss` best=0.2692 @ epoch 179 vs B7 0.4256 vs E4 0.3218 — substantial improvement. `imp_peak_auroc_gw` peak=0.4861 vs B7 0.5530 (12% regression — peak imputation generalization slightly worse than multi-head). `imp_count_pearson_gw` peak=0.0508 vs B7 0.0592 vs E2 0.1154. Pre-clip grad-norm: median=7.89, p95=25.7, max=82.0 — between E2 (5/13/20) and B7 (33/221/501); `clip_fraction.running_mean=0.986`. Backfilled HPO node now present.
- Interpretation: **Strongest evidence yet that pval head is interfering with count+peak training.** Compared to B7 (full multi-head), removing only pval lifts count_imp by 4.4% and `peak_obs` by 37% while keeping peak_imp essentially tied. Compared to E2 (count-only), adding peak back **does not regress count_imp** (1.7177 vs 1.7367 — actually slightly better), suggesting peak is a benign or even helpful auxiliary for the count head. Compared to E4 (peak-only), keeping count alongside peak **lifts peak_imp_loss by 17%** and `peak_obs_loss` by 16%, confirming E4's reading that peak depends on auxiliary gradients. The one weakness — `imp_peak_auroc_gw` being below B7 — suggests pval gradients carry information specifically useful for the peak imputation decision boundary that count cannot fully replace.
- Competing explanations: (a) the AUROC gap may close at 200 epochs (E5 was cut at 189) — the trajectory is still improving; (b) `imp_count_pearson_gw` regressing below E2 (0.0508 vs 0.1154) suggests peak co-training pulls some encoder capacity away from count's correlation structure even while reducing `count_imp_loss`; (c) the sandbox's 200-epoch budget may not be enough to see whether count+peak overtakes B7 on absolute AUROC if given more time.
- Decision: **Adopt count+peak (pval-muted) as a strong candidate setting** for further development. Highest-priority follow-up: rerun E5 with 4 h walltime to complete 200 epochs and confirm trends; then sweep `pval_weight ∈ {0.0, 0.1, 0.3, 1.0}` while keeping count+peak active to find the smallest pval contribution that recovers B7's AUROC ceiling without sacrificing E5's count_imp gains.

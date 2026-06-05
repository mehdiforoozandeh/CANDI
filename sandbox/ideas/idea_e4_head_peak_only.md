# E4 - Peak head only

Status: done  
Parent: baseline_anchor  
Run name: `E4_head_peak_only`  
Checklist entry: [EXPERIMENTS.md](EXPERIMENTS.md#experiment-sweep-e1-e5)

## Problem Statement

Isolate peak head behavior without count and pval loss competition.

## Idea / Hypothesis

Peak-only training should clarify whether peak AUROC/loss is limited by multi-head interference or by the peak objective itself.

## Planned Intervention

- Submit/config path: [`../slurm/submit_experiments_e1_e5.sh`](../slurm/submit_experiments_e1_e5.sh)
- Run directory: [`../runs/E4_head_peak_only/`](../runs/E4_head_peak_only/)
- Parent run or idea: `baseline_anchor`
- Config/code/data deltas: log1p type1 run with `count_weight=0.0` and `pval_weight=0.0`; parent set to `baseline_anchor`.

## Verifiables

- Validate if: The active branch metrics move in the predicted direction and the run produces complete, comparable artifacts for that experiment type.
- Disvalidate if: The active branch metrics do not improve, artifacts are incomplete, or disabled-head behavior prevents the intended comparison.
- Specific checks: use peak branch losses/AUROC only; aggregate total loss and composite quality are not available.
- Required artifacts: `resolved_config.yaml`, `metrics.jsonl`, SLURM logs, and `hpo_graph.json` when available.

## Risks / Watch-outs

- Aggregate quality score is absent by design.
- Untrained pval/count metrics should not be interpreted as failures of trained heads.
- HPO graph node exists but lacks aggregate result fields.

## Run Links

- Run directory: [`../runs/E4_head_peak_only/`](../runs/E4_head_peak_only/)
- Resolved config: [`../runs/E4_head_peak_only/resolved_config.yaml`](../runs/E4_head_peak_only/resolved_config.yaml)
- Metrics: [`../runs/E4_head_peak_only/metrics.jsonl`](../runs/E4_head_peak_only/metrics.jsonl)
- SLURM logs: `../slurm_logs/baseline_sbx_e4_peak_37595466.*`
- HPO graph: [`../hpo_graph.json`](../hpo_graph.json)
- W&B run name: `E4_head_peak_only`

## Findings

Cross-run rollup: see [`synthesis_e1_e5_head_interference.md`](synthesis_e1_e5_head_interference.md).

### Run 1 (200 epochs, old defaults, SLURM job 37595466)

Peak-only. imp_peak_auroc peak=0.483 (near chance). Confirmed peak depends on auxiliaries. Marked done.

### Run 2 (400 epochs, new defaults lr=1e-3, clip=2.0, SLURM job 38829775) — **current**

- Walltime-killed at epoch 364 of 400 (~91% budget). No total_loss in ranker (peak-only).
- `peak_obs_loss`: first=0.622, last=0.105 — monotonically declining throughout all 364 epochs. **Never diverges.**
- `peak_imp_loss`: first=0.700, last=0.138 — also monotonically declining. **Never diverges.**
- `imp_peak_auroc`: best=**0.834** @ ep154, then mild degradation to 0.795 @ ep364.
- Grad-norm: pre-clip median=1.07, p95=7.35, max=76.9. clip_fraction=**0.23** — dramatically lower than any other run (multi-head: 0.77–0.88). Peak head has the most benign gradient dynamics in the entire sweep.
- `depth_count_ratio` last=0.990 — metadata collapse persists (F1).
- **Key observations:**
  1. **Peak head is uniquely stable**: both obs and imp losses decline monotonically without any divergence across 364 epochs. This contrasts sharply with pval (catastrophic obs/imp split) and count (imp plateau at ep~50).
  2. **imp_peak_auroc=0.834** is the best peak imputation metric ever recorded for a single head. This sets the isolation ceiling for comparison.
  3. The mild degradation after ep154 (0.834 → 0.795) is a small overfitting signal but nothing like pval's explosion. Peak is intrinsically better-behaved for both denoising and imputation.
  4. The low clip_fraction (0.23) is diagnostic: peak's BCE gradients are naturally small and well-scaled, unlike the Gaussian NLL gradients of pval (which can become arbitrarily large as variance collapses).
- Verdict vs multi-head: E7 achieves imp_peak_auroc=0.796 in multi-head mode — nearly matching E4's isolation ceiling of 0.834. This means E7's single-shot decoder FiLM architecture recovers most of the peak performance that was previously lost to head interference.
- Decision: **E4 serves as the peak-only ceiling reference.** The 0.038 gap between E4 (0.834) and E7 (0.796) quantifies the remaining cost of multi-head sharing. Closing this gap is a secondary goal after fixing pval divergence.

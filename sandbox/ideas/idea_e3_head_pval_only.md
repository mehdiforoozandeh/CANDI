# E3 - Pval head only

Status: done  
Parent: baseline_anchor  
Run name: `E3_head_pval_only`  
Checklist entry: [EXPERIMENTS.md](EXPERIMENTS.md#experiment-sweep-e1-e5)

## Problem Statement

Isolate pval head behavior without count and peak loss competition.

## Idea / Hypothesis

If pval gradients are diluted by other heads, pval-only training should improve pval losses or pval Pearson.

## Planned Intervention

- Submit/config path: [`../slurm/submit_experiments_e1_e5.sh`](../slurm/submit_experiments_e1_e5.sh)
- Run directory: [`../runs/E3_head_pval_only/`](../runs/E3_head_pval_only/)
- Parent run or idea: `baseline_anchor`
- Config/code/data deltas: log1p type1 run with `count_weight=0.0` and `peak_weight=0.0`; parent set to `baseline_anchor`.

## Verifiables

- Validate if: The active branch metrics move in the predicted direction and the run produces complete, comparable artifacts for that experiment type.
- Disvalidate if: The active branch metrics do not improve, artifacts are incomplete, or disabled-head behavior prevents the intended comparison.
- Specific checks: use pval branch metrics only; aggregate total loss and composite quality are not available.
- Required artifacts: `resolved_config.yaml`, `metrics.jsonl`, SLURM logs, and `hpo_graph.json` when available.

## Risks / Watch-outs

- No aggregate quality score means no standard ranking.
- Disabled count/peak branches make their metrics non-success criteria.
- Training total reached 0.0 at the last logged step, so branch loss logging should be checked before deeper conclusions.

## Run Links

- Run directory: [`../runs/E3_head_pval_only/`](../runs/E3_head_pval_only/)
- Resolved config: [`../runs/E3_head_pval_only/resolved_config.yaml`](../runs/E3_head_pval_only/resolved_config.yaml)
- Metrics: [`../runs/E3_head_pval_only/metrics.jsonl`](../runs/E3_head_pval_only/metrics.jsonl)
- SLURM logs: `../slurm_logs/baseline_sbx_e3_pval_37595465.*`
- HPO graph: [`../hpo_graph.json`](../hpo_graph.json)
- W&B run name: `E3_head_pval_only`

## Findings

Cross-run rollup: see [`synthesis_e1_e5_head_interference.md`](synthesis_e1_e5_head_interference.md).

### Run 1 (200 epochs, old defaults, SLURM job 37595465)

pval-only. Degenerate end-state at step 13200. imp_pval_pearson peak=0.117 (−32% vs B7). Marked done.

### Run 2 (400 epochs, new defaults lr=1e-3, clip=2.0, SLURM job 38829774) — **current**

- Walltime-killed at epoch 369 of 400 (~92% budget). No total_loss in ranker (pval-only).
- `pval_obs_loss`: first=0.942, last=0.606. Improved monotonically, reaching **negative values** by ep104 (−0.148, meaning Gaussian NLL is negative → model is highly confident on observed assays). Best=−0.201 @ ep149.
- `pval_imp_loss`: first=0.871, last=**20.84** — catastrophic divergence from ep~100 onward. Best=−2.04 @ ep149 (only briefly negative), then explodes.
- `imp_pval_pearson`: best=0.319 @ ep54, then degrades. Last=0.233.
- `imp_pval_spearman`: best=0.477 @ ep119, last=0.403.
- Grad-norm: pre-clip median=33.6, p95=342, max=844. clip_fraction=0.71. Pval head is the **largest-gradient head** in the whole sweep.
- `depth_count_ratio` last=1.005 (best of all runs — notably non-collapsed; pval training has the healthiest depth probe despite its other problems).
- **Critical finding (F7):** The obs/imp split for pval is catastrophic and structural. `pval_obs` reaches negative Gaussian NLL (model collapses variance → effectively certain predictions on training assays) while `pval_imp` diverges to 20+. This is **variance collapse / overconfidence**: the model learns to predict with zero uncertainty on seen assay types but this extreme confidence is catastrophically wrong on masked assays. This is NOT a gradient competition effect (co-training helps pval stability) — it is intrinsic to the unbounded Gaussian NLL + the obs/imp masking regime.
- Decision: **Pval divergence requires a distributional fix.** The Gaussian NLL must be constrained: either clamp logvar (E13), add σ² ≥ σ²_min, or switch to a bounded distribution (laplace, student_t). Retrying pval-only at any epoch budget will not resolve this. The correct approach is multi-head training (pval needs count/peak as a stabilizer) plus Gaussian hardening.

# E14 - Six loss uncertainty-weighting logvars

Status: idea  
Parent: uncertainty-weighting review  
Run name: TBD  
Checklist entry: [EXPERIMENTS.md](EXPERIMENTS.md#experiment-sweep-e6-e16)

## Problem Statement

**Context — what these logvars are:** `CANDI_LOSS` contains three learned scalar parameters `logvar_count`, `logvar_pval`, `logvar_peak` that implement Kendall & Gal (2017) multi-task uncertainty weighting. When `enable_uncertainty_weighting=True`, each head's loss contribution to the total is:

```
total += exp(-logvar_head) * head_loss + 0.5 * logvar_head
```

This is a **loss-reweighting mechanism**, not a prediction. The model automatically learns how much weight to give each head's loss relative to the others. A high `logvar` suppresses a head's contribution; a low `logvar` amplifies it.

Note: these are completely separate from the `GaussianLayer` predicted variance (E13). E13 fixes the model's output; E14 refines the loss aggregation.

**The problem:** Each logvar currently ties the observed (denoising) and imputed (masked assay) branches of the same head to a single weight. But imputed supervision is structurally harder and noisier than observed supervision — the two branches have different natural noise scales and may benefit from different effective weights.

## Idea / Hypothesis

Replacing 3 logvars with 6 (one per obs/imp × count/pval/peak branch) lets the optimizer independently rebalance observed vs imputed difficulty per head. This could prevent scenarios where a hard imputed branch silently drives up its shared logvar and suppresses the more tractable observed branch (or vice versa).

## Planned Intervention

- File: `candi_loss.py`, `CANDI_LOSS.__init__` — replace `logvar_count/pval/peak` with `logvar_count_obs`, `logvar_count_imp`, `logvar_pval_obs`, `logvar_pval_imp`, `logvar_peak_obs`, `logvar_peak_imp`.
- Apply separately in `_apply_uncertainty` for obs and imp branches.
- Prerequisite: `enable_uncertainty_weighting=True` must be the active default before this experiment is meaningful. E13 should be run first.
- Submit/config path: TBD
- Run name: TBD
- Parent: E13 (gaussian var floor) + E7 (single-shot FiLM) as base

## Verifiables

- Validate if: the 6 learned logvars are logged per epoch; obs and imp branch weights diverge (i.e., the optimizer actually exploits the extra DOF); `imp_pval_pearson` improves vs E13 baseline.
- Disvalidate if: all 6 logvars converge to the same value (extra DOF unused); branch losses become less stable; added degrees of freedom increase noise without improving metrics.
- Required artifacts: `resolved_config.yaml`, `metrics.jsonl`, SLURM logs.

## Risks / Watch-outs

- E14 only makes sense **after** `enable_uncertainty_weighting=True` is established as stable. Running it with the default (uncertainty weighting off) is a no-op.
- Six learned weights increase the risk of overfitting the rebalancing to the sandbox's small assay set (8 assays).
- The fix for pval divergence (E13) is architectural and must come first — E14 cannot substitute for it.

## Run Links

- Run directory: TBD
- Resolved config: TBD
- Metrics: TBD
- SLURM logs: TBD
- HPO graph node: TBD
- W&B run: TBD

## Findings

- Observed: TBD
- Interpretation: TBD
- Competing explanations: TBD
- Decision: TBD

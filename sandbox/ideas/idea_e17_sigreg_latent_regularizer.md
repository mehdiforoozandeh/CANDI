# E17 - SIGReg latent regularizer

Status: idea  
Parent: TBD  
Run name: TBD  
Checklist entry: [EXPERIMENTS.md](EXPERIMENTS.md#experiment-sweep-e17-e20)

## Problem Statement

CANDI's encoder latent may be poorly conditioned or anisotropic when trained only through noisy raw reconstruction losses.

## Idea / Hypothesis

Add a LeJEPA/SIGReg-faithful auxiliary regularizer to the CANDI encoder latent while keeping the current masked reconstruction objective and inference path intact. The hypothesis is that isotropic Gaussian latent pressure will improve representation conditioning, reduce collapse or dominant latent axes, and improve downstream imputation stability without hurting raw likelihood performance.

## Planned Intervention

- Submit/config path: TBD
- Run name: TBD
- Parent run or idea: likely strongest current full-head sandbox baseline, TBD.
- Config/code/data deltas: add a separate SIGReg projection head on the pre-decoder encoder latent; keep the existing decoder path unchanged; add `lambda_sigreg * SIGReg(sigreg_head(z))` to the current CANDI loss; do not apply SIGReg after a LayerNorm-normalized latent.
- Faithfulness constraint: implement SIGReg as random one-dimensional projections with characteristic-function matching to `N(0, I)`, as in LeJEPA/SIGReg; use a BatchNorm-compatible projection head rather than a LayerNorm-terminated head.

## Verifiables

- Validate if: SIGReg loss decreases to a stable range, latent covariance/effective-rank diagnostics improve, and imputation metrics are tied or better than the matched baseline without extra inference cost.
- Disvalidate if: CANDI losses or imputation metrics regress beyond run noise, SIGReg dominates gradients, or latent diagnostics improve while biological reconstruction metrics degrade.
- Specific checks: compare `eval_losses/total_loss`, branch losses, `quality_score` when available, imputation Pearson/AUROC metrics, gradient norms, SIGReg loss curve, latent mean/std/covariance/effective-rank diagnostics if logged.
- Required artifacts: `resolved_config.yaml`, `metrics.jsonl`, SLURM logs, and graph/W&B metadata when available.

## Risks / Watch-outs

- A too-large `lambda_sigreg` can force Gaussian geometry at the expense of assay-specific predictive detail.
- Applying SIGReg after LayerNorm would test the wrong mechanism because LayerNorm removes per-token moment information SIGReg is meant to shape.
- Improved latent geometry alone is not sufficient; the run must preserve or improve reconstruction and imputation metrics.
- If latent diagnostics are not logged yet, this idea may require a small observability patch before the result is interpretable.

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

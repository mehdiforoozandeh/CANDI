# E18 - Joint JEPA assay-mask prediction

Status: idea  
Parent: E17 or matched full-head baseline, TBD  
Run name: TBD  
Checklist entry: [EXPERIMENTS.md](EXPERIMENTS.md#experiment-sweep-e17-e20)

## Problem Statement

Reconstruction-only training may let raw likelihood gradients dominate representation learning before the encoder learns assay-completion structure.

## Idea / Hypothesis

Train CANDI with a joint objective that combines the existing decoder likelihood loss with a LeJEPA/LeWM-style latent prediction loss and SIGReg. The hypothesis is that predicting the full-input latent from a full-assay-masked context latent will force the encoder to learn cross-assay biological structure that improves imputation beyond raw reconstruction alone.

## Planned Intervention

- Submit/config path: TBD
- Run name: TBD
- Parent run or idea: E19 Stage 1 (encoder-only JEPA) — E18 adds the CANDI decoder loss on top.
- Config/code/data deltas: reuse `sandbox/train_jepa.py` harness from E19; add `L_candi` (full CANDI_LOSS) alongside the JEPA objective; train with `L_candi + L_pred + lambda_sigreg * L_sigreg`.
- Faithfulness constraint: NO stop-gradient on the target latent — this is the core LeJEPA/LeWM design principle; SIGReg is the sole anti-collapse mechanism. See [spec_e19_jepa_encoder_harness.md](spec_e19_jepa_encoder_harness.md) for full architecture details.

## Verifiables

- Validate if: latent prediction loss and SIGReg loss are stable, reconstruction/imputation metrics improve over the matched baseline, and the predictor does not destabilize decoder training.
- Disvalidate if: two-pass training causes instability, CANDI reconstruction metrics regress, or the predictor learns an easy shortcut that does not improve masked-assay imputation.
- Specific checks: compare `eval_losses/total_loss`, branch losses, `quality_score` when available, imputation Pearson/AUROC metrics, `L_pred`, `L_sigreg`, predictor gradient norms, and walltime/memory overhead.
- Required artifacts: `resolved_config.yaml`, `metrics.jsonl`, SLURM logs, and graph/W&B metadata when available.

## Risks / Watch-outs

- This is less clean than encoder-only LeJEPA because decoder likelihood gradients and latent prediction gradients hit the encoder simultaneously.
- The extra target encoder pass roughly doubles encoder compute and may require a smaller batch size or shorter context.
- Full-assay masking means the predictor condition is assay-set-level, not locus-position-level; using patch-style query metadata would test the wrong adaptation.
- Loss-weight tuning for `beta` and `lambda_sigreg` may dominate the result.

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

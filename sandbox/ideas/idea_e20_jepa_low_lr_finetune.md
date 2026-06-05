# E20 - JEPA encoder then low-LR fine-tuning

Status: idea  
Parent: E19  
Run name: TBD  
Checklist entry: [EXPERIMENTS.md](EXPERIMENTS.md#experiment-sweep-e17-e20)

## Problem Statement

A frozen JEPA encoder may cap raw CANDI likelihood performance if the pretrained latent omits decoder-specific signal details.

## Idea / Hypothesis

Start from the encoder-only JEPA/SIGReg pretrained checkpoint, attach the CANDI decoder, and fine-tune the encoder with a much smaller learning rate than the decoder. The hypothesis is that low-LR encoder adaptation will recover raw reconstruction detail while preserving most of the pretrained SIGReg/JEPA latent geometry.

## Planned Intervention

- Submit/config path: TBD
- Run name: TBD
- Parent run or idea: E19 pretrained encoder checkpoint.
- Config/code/data deltas: initialize from the Stage 1 JEPA/SIGReg encoder; train decoder/output heads with standard `L_candi`; unfreeze the encoder after an optional decoder warmup; use `encoder_lr = gamma * decoder_lr` with `gamma` initially in `[0.01, 0.1]`; optionally keep a weak SIGReg anchor during fine-tuning.
- Faithfulness constraint: keep the LeJEPA-style encoder-only pretraining stage intact, then treat fine-tuning as a downstream adaptation step rather than mixing raw reconstruction into the initial representation objective.

## Verifiables

- Validate if: low-LR fine-tuning improves over E19 frozen decoding while preserving stable latent diagnostics and not regressing relative to the matched current CANDI baseline.
- Disvalidate if: fine-tuning erases the pretrained latent geometry, destabilizes training, or fails to improve over the frozen-decoder result.
- Specific checks: compare to E19 on `eval_losses/total_loss`, branch losses, `quality_score` when available, imputation Pearson/AUROC metrics, encoder gradient norms, `L_sigreg` anchor if enabled, and latent diagnostics before/after fine-tuning.
- Required artifacts: `resolved_config.yaml`, `metrics.jsonl`, SLURM logs, and graph/W&B metadata when available.

## Risks / Watch-outs

- A too-large encoder learning rate can overwrite the JEPA/SIGReg representation and collapse this into ordinary CANDI training.
- A too-small encoder learning rate may behave indistinguishably from E19 frozen decoding.
- The optional SIGReg anchor weight must be weak enough not to fight decoder-specific adaptation.
- This result depends on the quality of the E19 pretrained encoder; poor Stage 1 pretraining makes the fine-tuning comparison hard to interpret.

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

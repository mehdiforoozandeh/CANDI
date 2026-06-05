# E15 - Uncertainty optimizer-scheduler hardening

Status: idea  
Parent: uncertainty-weighting review  
Run name: TBD  
Checklist entry: [EXPERIMENTS.md](EXPERIMENTS.md#experiment-sweep-e6-e16)

## Problem Statement

Uncertainty parameters must be attached to the optimizer before any scheduler captures parameter-group base learning rates; otherwise their LR behavior can silently drift.

## Idea / Hypothesis

Adding explicit ordering checks and clear optimizer-group construction should prevent uncertainty-weighting parameters from being missed by the scheduler.

## Planned Intervention

- Submit/config path: TBD
- Run name: TBD
- Parent run or idea: uncertainty-weighting review
- Config/code/data deltas: assert uncertainty parameters are present in optimizer parameter groups before scheduler construction and document the ordering constraint.

## Verifiables

- Validate if: scheduler base LR count matches optimizer group count and uncertainty parameter groups show expected LR values during training.
- Disvalidate if: uncertainty params are absent from scheduled groups, LR logs diverge unexpectedly, or optimizer setup becomes order-dependent.
- Required artifacts: `resolved_config.yaml`, `metrics.jsonl`, SLURM logs, and graph/W&B metadata when available.

## Risks / Watch-outs

- The current order appears safe, but fragile to future refactors.
- This is a guardrail, not expected to change metrics.
- Pair with E14 if six uncertainty logvars introduce additional criterion parameters.

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

# E8 - Per-group gradient clipping

Status: idea  
Parent: E9 / E11 diagnostics  
Run name: TBD  
Checklist entry: [EXPERIMENTS.md](EXPERIMENTS.md#experiment-sweep-e6-e16)

## Problem Statement

Global gradient clipping fires on every step and may let decoder-heavy gradients consume the shared norm budget, shrinking metadata-branch gradients even when those gradients are already small.

## Idea / Hypothesis

Clipping metadata and non-metadata optimizer groups separately should preserve metadata-branch updates while still preventing decoder and head gradient spikes from destabilizing training.

## Planned Intervention

- Submit/config path: TBD
- Run name: TBD
- Parent run or idea: E9 and E11 diagnostics
- Config/code/data deltas: define optimizer groups for metadata-conditioning parameters and everything else; apply `clip_grad_norm_` separately per group.

## Verifiables

- Validate if: metadata-group post-clip norms are no longer uniformly shrunk by decoder-group spikes and prompt-sensitivity metrics improve or become less flat.
- Disvalidate if: metadata-group norms remain near zero, training destabilizes, or prompt-sensitivity metrics do not move relative to the same architecture with global clipping.
- Required artifacts: `resolved_config.yaml`, `metrics.jsonl`, SLURM logs, and graph/W&B metadata when available.

## Risks / Watch-outs

- Per-group clipping does not create gradient where the architecture gives none.
- Group membership must be explicit and logged so run comparisons are reproducible.
- Compare against E9/E10/E11 instrumentation before claiming causality.

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

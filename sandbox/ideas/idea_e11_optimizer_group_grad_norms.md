# E11 - Optimizer-group grad-norm logging

Status: idea  
Parent: E8 per-group clipping  
Run name: TBD  
Checklist entry: [EXPERIMENTS.md](EXPERIMENTS.md#experiment-sweep-e6-e16)

## Problem Statement

Per-group clipping needs direct evidence at the optimizer-group level, not just module-level diagnostics.

## Idea / Hypothesis

Logging pre-clip and post-clip grad norms per optimizer group will verify whether metadata groups are protected from decoder-dominated clipping and whether each group hits its own cap.

## Planned Intervention

- Submit/config path: TBD
- Run name: TBD
- Parent run or idea: E8 per-group clipping
- Config/code/data deltas: name optimizer groups and log each group's pre-clip norm, post-clip norm, clip threshold, and clip-active flag.

## Verifiables

- Validate if: artifacts show group-level norms for every optimizer group and clarify whether per-group clipping changes metadata vs decoder update pressure.
- Disvalidate if: optimizer groups are unnamed, metrics are missing, or pre/post values do not match the configured clipping behavior.
- Required artifacts: `resolved_config.yaml`, `metrics.jsonl`, SLURM logs, and graph/W&B metadata when available.

## Risks / Watch-outs

- Group names must be stable across runs to support comparison.
- Post-clip norm should be measured after clipping, not inferred.
- Do not confuse LR groups with grad-norm groups unless they are intentionally identical.

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

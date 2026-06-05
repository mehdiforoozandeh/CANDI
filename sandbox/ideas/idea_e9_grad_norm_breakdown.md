# E9 - Per-module grad-norm logging

Status: idea  
Parent: prompt-collapse investigation  
Run name: TBD  
Checklist entry: [EXPERIMENTS.md](EXPERIMENTS.md#experiment-sweep-e6-e16)

## Problem Statement

The current global gradient norm hides whether metadata encoders, FiLM layers, decoder heads, or the backbone dominate clipping and training dynamics.

## Idea / Hypothesis

Logging per-module gradient norms will reveal whether metadata-conditioning parameters are starved relative to decoder and backbone parameters during sandbox training.

## Planned Intervention

- Submit/config path: TBD
- Run name: TBD
- Parent run or idea: prompt-collapse investigation
- Config/code/data deltas: record pre-clip grad norms for key module families including metadata encoders, FiLM, decoder heads, DNA stem, and transformer/backbone blocks.

## Verifiables

- Validate if: `metrics.jsonl` or equivalent logs include stable module-level grad-norm keys and identify which modules dominate the global clipping norm.
- Disvalidate if: logged values are missing, too noisy to interpret, or cannot be mapped to model modules consistently.
- Required artifacts: `resolved_config.yaml`, `metrics.jsonl`, SLURM logs, and graph/W&B metadata when available.

## Risks / Watch-outs

- Too many metric keys can make logs noisy; keep module groups coarse and named consistently.
- Grad norms should be computed before clipping for diagnosis and optionally after clipping for interventions.
- This is instrumentation, not a behavior change, so evaluate it by observability quality.

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

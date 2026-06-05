# E10 - Clip-active fraction logging

Status: done  
Parent: prompt-collapse investigation  
Run name: TBD  
Checklist entry: [EXPERIMENTS.md](EXPERIMENTS.md#experiment-sweep-e6-e16)

## Problem Statement

Clipping appears to happen on every step, but this needs to be logged as a durable run metric rather than inferred from ad hoc inspection.

## Idea / Hypothesis

Logging clip-active fraction will make clipping pressure comparable across sandbox runs and show whether architectural or optimizer changes reduce always-on clipping.

## Planned Intervention

- Submit/config path: TBD
- Run name: TBD
- Parent run or idea: prompt-collapse investigation
- Config/code/data deltas: log whether pre-clip grad norm exceeds the configured clip threshold, plus a rolling or epoch-level fraction.

## Verifiables

- Validate if: run artifacts contain clip-active metrics that reproduce the observed always-on clipping baseline and vary under clipping interventions.
- Disvalidate if: the metric is missing, inconsistent with pre-clip norms, or cannot distinguish global vs per-group clipping.
- Required artifacts: `resolved_config.yaml`, `metrics.jsonl`, SLURM logs, and graph/W&B metadata when available.

## Risks / Watch-outs

- With per-group clipping, log clip-active fraction per group as well as globally.
- Do not interpret high clip fraction as bad by itself; pair it with losses and grad-norm breakdowns.
- This metric is diagnostic only.

## Run Links

- Run directory: TBD
- Resolved config: TBD
- Metrics: TBD
- SLURM logs: TBD
- HPO graph node: TBD
- W&B run: TBD

## Findings

- Observed: Already implemented (2026-05-06). Every `training_step` row in `metrics.jsonl` contains five clip-related fields:
  - `training_stats/grad_pre_clip_norm` — global pre-clip gradient norm
  - `training_stats/grad_clip_cap` — the configured cap value
  - `training_stats/grad_clipped` — 1 if clipped this step, 0 otherwise
  - `training_stats/grad_clipped_frac_running` — running mean of clipped fraction since epoch start
  - `training_stats/grad_clipped_frac_window` — windowed mean over the last `clip_log_window` steps
- These metrics have been used throughout all log-observability analyses (e.g. clip_fraction=0.88 for baseline, 0.23 for E4 peak-only). The `inspect_training_steps.py` script reads and reports them as `clip_fraction.running_mean`.
- Decision: No implementation needed. Marked done.

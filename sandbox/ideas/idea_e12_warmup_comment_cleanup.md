# E12 - Scheduler warmup comment cleanup

Status: done  
Parent: sandbox training hygiene  
Run name: TBD  
Checklist entry: [EXPERIMENTS.md](EXPERIMENTS.md#experiment-sweep-e6-e16)

## Problem Statement

The scheduler code uses 10% warmup while the nearby comment says 20%, which creates ambiguity before optimizer and scheduler experiments.

## Idea / Hypothesis

Aligning the comment with the implemented value should reduce interpretation mistakes without changing behavior if the code remains at 10%.

## Planned Intervention

- Submit/config path: TBD
- Run name: TBD
- Parent run or idea: sandbox training hygiene
- Config/code/data deltas: update the comment to match the code, or deliberately change the code to match the intended 20% after a separate decision.

## Verifiables

- Validate if: the comment and code agree and the resolved config or logs still show the expected scheduler behavior.
- Disvalidate if: the change accidentally alters scheduler behavior without being staged as a real scheduler experiment.
- Required artifacts: `resolved_config.yaml`, `metrics.jsonl`, SLURM logs, and graph/W&B metadata when available.

## Risks / Watch-outs

- Treat code behavior changes as a separate experiment from comment cleanup.
- This is a hygiene change, not expected to affect metrics.
- Keep it isolated from optimizer experiments when possible.

## Run Links

- Run directory: TBD
- Resolved config: TBD
- Metrics: TBD
- SLURM logs: TBD
- HPO graph node: TBD
- W&B run: TBD

## Findings

- Observed: Two comment mismatches found and fixed (2026-05-06):
  1. `sandbox/config_types.py` line 124: `warmup_frac` comment said "fraction of **epochs**" — changed to "fraction of total training **steps**" (the scheduler uses `warmup_steps = int(round(total_steps * wf))`, operating on steps, not epochs).
  2. `sandbox/train.py` `build_scheduler`: the `start_factor=0.2` in `LinearLR` was undocumented and easily mistaken for a "20% warmup duration" — added a docstring clarifying that `0.2` is the **starting LR fraction** (LR begins at 20% of peak and ramps to 100%), which is independent of `warmup_frac` (the **duration** of the warmup phase, default 10% of total steps).
- Decision: No code behavior changed. No run needed. Marked done.

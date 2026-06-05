# E16 - Uncertainty variant documentation

Status: idea  
Parent: uncertainty-weighting review  
Run name: TBD  
Checklist entry: [EXPERIMENTS.md](EXPERIMENTS.md#experiment-sweep-e6-e16)

## Problem Statement

The uncertainty weighting path currently uses one Kendall-Gal convention across regression-like heads and classification-like peak loss, which makes the learned logvars harder to interpret.

## Idea / Hypothesis

Explicitly documenting the convention, or separating regression and classification variants, should make uncertainty-weighting experiments reproducible and easier to compare.

## Planned Intervention

- Submit/config path: TBD
- Run name: TBD
- Parent run or idea: uncertainty-weighting review
- Config/code/data deltas: either document the current `exp(-s) * L + 0.5 * s` convention for all heads or use the regression variant for count/signal and classification variant for peak.

## Verifiables

- Validate if: the chosen formula is documented in code/config and logged results can be interpreted without guessing which convention was used.
- Disvalidate if: formula changes are mixed with unrelated changes or learned logvars remain ambiguous across runs.
- Required artifacts: `resolved_config.yaml`, `metrics.jsonl`, SLURM logs, and graph/W&B metadata when available.

## Risks / Watch-outs

- Documentation-only change should not affect metrics.
- Formula change should be staged as behavior-changing and compared to the documented baseline.
- Keep this separate from E14 if possible to isolate effects.

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

---
name: log-observability
description: Analyze sandbox.train run logs deterministically and act as autoresearch co-researcher on EpiDenoise sandbox runs. Use when the user asks to inspect, compare, rank, or diagnose sandbox runs, mentions metrics.jsonl / resolved_config.yaml / SLURM job IDs / W&B runs / training divergence / grad norms, or invokes /log-observability.
---

# Log Observability

Deterministic, no-hallucination analysis of `sandbox.train` runs in EpiDenoise. Pairs a slim
workflow with bundled scripts that do the heavy lifting.

## Quick start

Given one or more SLURM job IDs or run dirs, run the pipeline in order and stop early once a step
gives a definitive answer:

```bash
# 1. Map SLURM job IDs to run dirs (skip if user gave run dirs directly).
bash .cursor/skills/log-observability/scripts/find_runs_for_jobs.sh <jobid> [<jobid> ...]

# 2. Confirm controlled experiment (only diffing leaf keys).
python .cursor/skills/log-observability/scripts/compare_configs.py \
  sandbox/runs/<a>/resolved_config.yaml sandbox/runs/<b>/resolved_config.yaml ...

# 3. Per-run summary (epochs, sec/epoch, NaN/Inf, divergence, depth_count_ratio).
python .cursor/skills/log-observability/scripts/summarize_runs.py sandbox/runs/<a> sandbox/runs/<b>

# 4. Grad-norm / clip diagnosis (only on divergence or quality collapse).
python .cursor/skills/log-observability/scripts/inspect_training_steps.py sandbox/runs/<a>

# 5. Cornerstone ranking (the only place "winner" verdicts come from).
python .cursor/skills/log-observability/scripts/rank_runs.py sandbox/runs/<a> sandbox/runs/<b>
```

Always cite the exact metric key and value when stating a finding — no claim without evidence.

## Required inputs (per run)

1. `resolved_config.yaml` — what the run actually executed.
2. `metrics.jsonl` — source of truth (rows are `kind="epoch"` or `kind="training_step"`).
3. SLURM stdout/err under `sandbox/slurm_logs/baseline_<name>_<jobid>.{out,err}` for walltime / OOM.
4. Optional W&B history (only as fallback when (2) lacks `training_step` rows).

If two sources disagree, treat `metrics.jsonl` as authoritative and report the conflict explicitly.

## Workflow checklist

For every analysis request:

- [ ] Step 1 — map jobs → run dirs; flag walltime-kill / OOM rows immediately.
- [ ] Step 2 — diff configs; if a key the user didn't mean to vary differs, stop and flag.
- [ ] Step 3 — `summarize_runs.py`; record divergence flag and depth_count_ratio per run.
- [ ] Step 4 — only if divergence or sudden quality collapse: `inspect_training_steps.py`.
- [ ] Step 5 — `rank_runs.py` for any A vs B verdict; never invent your own ranking metric.
- [ ] Apply No-Hallucination Rules (REFERENCE.md) to every claim.
- [ ] Carry every unresolved Standing Finding (FINDINGS.md) into the report.
- [ ] Format each insight with the Insight Extraction Template (REFERENCE.md).

## Autoresearch loop protocol

One loop iteration:

1. Diagnose from current logs using the pipeline above.
2. Form **one** hypothesis explaining the dominant signal.
3. Propose **one** bounded config change.
4. Predict the direction of `quality_score` and at least one per-branch imp loss
   (no new ranking metrics; see Cornerstone Decision Rule in REFERENCE.md).
5. Run, then re-rank with `rank_runs.py` against the previous best.
6. Keep / revert based on a stop condition tied to Tier 2 eligibility.

Never run multiple independent hypotheses in one step unless explicitly requested.

## Quantitative interpretation (cheat sheet)

Full table and decision rule live in [REFERENCE.md](REFERENCE.md). The most-used thresholds:

- `training_metadata_probes/depth_count_ratio`: healthy 3.0–5.0; failure < 1.5 or > 8.0.
- `eval_losses/total_loss` divergence: `last > 1.5 × best` ⇒ diverged.
- Walltime utilization (`Σ epoch_seconds / SLURM walltime`): healthy ≥ 70%; failure < 40%.
- NaN/Inf in any `eval_*` family ⇒ disqualifying.

## Forbidden conclusions

- Any `*_r2_1obs` metric in success criteria (sparse target → meaningless R2; use Spearman).
- Any reconstruction/eval MSE/RMSE family (removed in 2026-04 refactor; if present the run is
  pre-refactor — flag and downgrade confidence).
- Grad-norm or clip-fraction claims without `kind="training_step"` rows in `metrics.jsonl`.
- A "winner" declaration from any metric not on the cornerstone list (REFERENCE.md).

## Files

- [REFERENCE.md](REFERENCE.md) — metric language contract, Cornerstone Decision Rule with full
  tier definitions and field lists, no-hallucination rules, confidence rubric, Insight Extraction
  Template, quantitative interpretation table.
- [FINDINGS.md](FINDINGS.md) — Standing Findings (F1…Fn). Append-only; mark resolved, never delete.
- `scripts/find_runs_for_jobs.sh` — SLURM job ID → run dir mapping.
- `scripts/compare_configs.py` — leaf-level diff of `resolved_config.yaml` files.
- `scripts/summarize_runs.py` — per-run progress + health summary from `metrics.jsonl`.
- `scripts/inspect_training_steps.py` — grad-norm / clip / per-branch loss trajectory.
- `scripts/rank_runs.py` — deterministic implementation of the Cornerstone Decision Rule.

## Updating this skill

This skill lives inside the autoresearch loop. Update on these triggers:

- New diagnostic script proves useful → add under `scripts/` with a top docstring; reference it
  in the Quick start and Files sections of this file.
- New quantitative threshold established → add a row to the table in REFERENCE.md, summarize
  here only if it belongs in the cheat sheet.
- New recurring observation → append to FINDINGS.md with a fresh `Fn` tag. Never delete; mark
  `resolved (run <id>, <date>)` instead.
- `metrics.jsonl` schema changes → update REFERENCE.md "Metric Language Contract".

Keep this file under ~120 lines. Anything longer belongs in REFERENCE.md or FINDINGS.md.

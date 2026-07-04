---
name: sandbox-synthesis
description: Produce cross-run synthesis docs for sandbox experiments. Use whenever rolling up findings across multiple sandbox runs (sweeps, ablations, head-isolation comparisons, longitudinal F* findings), creating or editing files matching `sandbox/ideas/synthesis_*.md`, or asked to write a "synthesis", "rollup", "summary report", "comparison report", or "post-sweep analysis".
---

# Sandbox Synthesis

Standardised convention for cross-run synthesis docs in `sandbox/ideas/`. Use this skill whenever you compare two or more runs and the comparison itself is a deliverable. Single-run write-ups belong in their `idea_*.md` file, not here.

## Trigger

Use this skill when you are about to:

- Aggregate findings from a sweep (e.g. B1–B7, E1–E5, a `pval_weight` sweep).
- Compare two or more runs against each other or against a baseline.
- Update or extend a Standing Finding (`F*`) using evidence from multiple runs.
- Write any "headline conclusions", leaderboard, or rollup table that spans more than one run.

If the deliverable is about a single run, stop and use `sandbox-idea-hub` instead — write the finding into the run's `idea_*.md` file.

## File naming

- Filename: `sandbox/ideas/synthesis_<scope>.md` — e.g. `synthesis_e1_e5_head_interference.md`, `synthesis_b_sweep.md`, `synthesis_pval_weight_sweep.md`.
- Prefix is always `synthesis_` (with the underscore) so synthesis docs sort and search distinctly from `idea_*.md` files.
- Choose a scope tag that names the runs or the question, not the date. Date the document inside the file, not in the filename.
- One file per scope. If a scope grows, append; do not split into `synthesis_e1_e5_v2.md`.

## Required structure (use this exact section order)

```
# <Title — runs or question being synthesised>

Status: synthesis (read-only)
Parents: <linked idea files for every run included>
Linked from: EXPERIMENTS.md  (only if the index has a one-line pointer to this file)
Date: YYYY-MM-DD

## Headline conclusions
## Cross-run quantitative table
## Per-run grad / stability table  (omit when irrelevant)
## Per-experiment outcome vs hypothesis
## Implications for next batch
## Standing findings (carried forward)
## Caveats and limits
```

Section rules:

- **Headline conclusions** — 3 to 6 numbered bullets. Each bullet ends with a confidence level (`High` / `Medium` / `Low`) per the log-observability rubric.
- **Cross-run quantitative table** — one row per run, columns chosen from the cornerstone metric list in `.cursor/skills/log-observability/REFERENCE.md`. Bold the best value in each column. Show `—` when a metric is not emitted (e.g. head-isolation runs without `eval_losses/total_loss`).
- **Per-run grad / stability table** — only when divergence, clipping, or grad-norm differences are part of the conclusion. Pull values from `inspect_training_steps.py`.
- **Per-experiment outcome vs hypothesis** — table with columns: `run | hypothesis | outcome | confidence`. Outcome must be one of `Confirmed`, `Rejected`, `Partial`, `Inconclusive` followed by a one-clause reason.
- **Implications for next batch** — prioritised list of bounded experiments. Each item: name, one-axis change, predicted metric move, cost in runs/GPU-hours.
- **Standing findings (carried forward)** — list every `F*` from `.cursor/skills/log-observability/FINDINGS.md` that any run in this synthesis touches. Note status (`open`, `mitigated`, `resolved`) and what this synthesis adds.
- **Caveats and limits** — walltime cuts, missing aggregate metrics, single-seed disclaimers, anything that downgrades confidence.

Do not invent extra sections. If something doesn't fit, it probably belongs in an `idea_*.md` instead.

## Evidence rules

- Every numeric claim cites a metric key + value, sourced from `metrics.jsonl` (primary) or scripts in `.cursor/skills/log-observability/scripts/`.
- Never produce a "winner" verdict outside the cornerstone decision rule (`rank_runs.py`). For head-isolation runs without an aggregate `quality_score`, use per-branch comparisons and say so explicitly.
- Quote at least one competing explanation for each headline conclusion (matches log-observability's no-hallucination rule #2).
- If you carry a Standing Finding forward, link to its `F*` tag in `FINDINGS.md` and report whether the synthesis runs change its status.

## Hygiene

- The synthesis file owns the rollup. Each contributing `idea_*.md` should reference the synthesis from its Findings section (one-line link), but should not duplicate the cross-run tables.
- `EXPERIMENTS.md` must NOT inline a synthesis. The index optionally references the synthesis as a follow-up link from a single experiment's findings sentence (never as its own section).
- When a Standing Finding's status changes due to this synthesis, update `.cursor/skills/log-observability/FINDINGS.md` in the same change set. Never delete an `F*`; mark it mitigated/resolved with a date and the synthesis filename.
- Update the synthesis in place when new runs in the same scope finish. Append a dated note inside the relevant section; do not bump filenames.
- If the synthesis is superseded by a more comprehensive one, mark `Status: superseded by synthesis_<new>.md` at the top and keep the file.

## META.md upkeep (mandatory)

`sandbox/ideas/META.md` is a **navigation index only — headlines + links, no finding text** (see [`sandbox-idea-hub`](../sandbox-idea-hub/SKILL.md) "META.md upkeep" for the full rules). In the **same change set** that creates or updates a `synthesis_*.md`:

- Identify which `Q*` block(s) the synthesis informs (most touch 1–3).
- Under each affected `Q*` toggle, add a **one-line link** to the synthesis (`synthesis_<scope>.md`) if it isn't already linked. No dated bullets, no finding prose, no metric tables — all of that lives in the synthesis file.
- If the synthesis changes a `Q*`'s status, edit only the `Q*` **headline** tag (`open` → `partially resolved` → `resolved (synthesis_<file>.md, YYYY-MM-DD)`). Keep the block; never delete a `Q*`.
- If a Standing Finding (`F*`) changed status, update `log-observability/FINDINGS.md`; reflect it in META only as a `Q*` headline status change, never as bullet text.
- If the synthesis exposes a question no existing `Q*` covers, open a new `Q*` (next number, never reuse retired ones).

Never duplicate synthesis content into META — META links to the synthesis for all technical detail.

## Skill update triggers

Update this skill when:

- The required structure changes (a section is added or removed).
- A new naming convention is agreed (only the user can change the `synthesis_` prefix).
- A new evidence source becomes mandatory (e.g. a new log-observability script becomes a default citation).
- Update `sandbox-idea-hub` in lockstep when these conventions affect cross-skill behaviour.

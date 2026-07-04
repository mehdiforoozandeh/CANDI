---
name: sandbox-idea-hub
description: Maintain the EpiDenoise/CANDI sandbox experiment idea hub. Use when planning, staging, running, documenting, or interpreting sandbox experiments, hypotheses, B/E/I experiment IDs, idea_*.md files, EXPERIMENTS.md, META.md, or sandbox run findings.
---

# Sandbox Idea Hub

Use this skill whenever work touches sandbox experiment ideation or documentation.

## Quick Start

1. Read `sandbox/ideas/META.md` first to orient on the open research questions, then `sandbox/ideas/EXPERIMENTS.md` for the per-run checklist.
2. For each experiment or idea, maintain exactly one detailed `sandbox/ideas/idea_*.md` file.
3. Keep `EXPERIMENTS.md` minimal: status, ID + linked title, one-line problem, one-line hypothesis, one-line finding (**Accepted/Rejected/Partial/Not run** + why). Link to idea/synthesis for detail. Config promotions go in `config_promotions.md`.
4. Put detailed problem statements, hypotheses, verifiables, risks, artifact links (run dir, resolved_config, metrics.jsonl, SLURM logs, HPO node, W&B run name), and findings in the linked idea file.
5. Preserve existing IDs (`B1`-`B7`, `E1`-`E5`, future `E*` or `I*`, `Q*` for META questions) unless the user explicitly asks to rename them.

## EXPERIMENTS.md Hygiene (strict)

`EXPERIMENTS.md` is a **navigation index only**. Each entry is exactly:

- Status checkbox + ID + linked title + status tag
- Problem: one line
- Hypothesis: one line
- Findings: one line starting with **Accepted** / **Rejected** / **Partial** / **Not run**, plus one-sentence why; link to `idea_*.md` or `synthesis_*.md` for detail

It must NOT contain:

- Artifact paths or links (run dirs, `metrics.jsonl`, SLURM job IDs, W&B names) — those live in the linked `idea_*.md` only
- Config promotion tables or default-change chronology — those live in [`config_promotions.md`](sandbox/ideas/config_promotions.md)
- Cross-experiment synthesis sections, summary tables, leaderboards, or multi-paragraph findings
- "Verifiables" / "Required artifacts" lines — those live in the idea file
- Numeric metric dumps in findings — one qualitative sentence only; numbers belong in idea/synthesis files

If you catch yourself adding any of the above to `EXPERIMENTS.md`, stop and put it in the linked idea file, a `synthesis_*.md`, or `config_promotions.md` instead.

## When to write a synthesis (vs an idea)

- Single-run write-up → goes in that run's `idea_*.md`. Use this skill.
- Two-or-more-run rollup, sweep summary, comparison table, leaderboard, or `F*`-finding update spanning multiple runs → goes in `sandbox/ideas/synthesis_<scope>.md`. Use the `sandbox-synthesis` skill, which defines the required structure (headline conclusions / cross-run table / outcome-vs-hypothesis / next batch / standing findings / caveats).
- Each `idea_*.md` linked to a synthesis should reference the synthesis from its Findings section (one-line link), but should not duplicate the cross-run tables.

## META.md upkeep (mandatory on every change)

`sandbox/ideas/META.md` is a **navigation index only**: open research questions (`Q*`) as nested toggle lists (`<details>`/`<summary>`), with experiment IDs and one-line links to `idea_*.md` / `synthesis_*.md`. No dated finding bullets, no metric tables, no "why it matters" prose, no artifact paths, no chronology log. Detail and history live in idea and synthesis files; standing findings in `FINDINGS.md`. META must stay in lockstep with `EXPERIMENTS.md`, idea files, and synthesis files. Update it in the **same change set** as the underlying file change — never as a follow-up cleanup.

Trigger events:

- New idea / new `idea_*.md` → add its ID under the matching `Q*` toggle (link only). If no existing `Q*` fits, open a new `Q*` (next number, never reuse retired ones).
- Status change in `EXPERIMENTS.md` (`idea` → `staged` → `running` → `done` / `incomplete` / `superseded`) → no META change unless the change produces a finding worth a status tag on the `Q*` headline (e.g. `partially resolved`, `resolved`).
- A run finishes with an evidence-backed finding → update the linked `idea_*.md` or `synthesis_*.md` first; META gets at most a one-word status change on the `Q*` headline, not the finding text.
- A new `synthesis_*.md` is written or updated → add a link under the relevant `Q*` toggle.
- A Standing Finding (`F*`) is added or its status changes in `log-observability/FINDINGS.md` → reflect only in the `Q*` headline status if it closes or partially closes the question (e.g. `partial fix on v2`).
- A `Q*` is fully answered → mark it in the toggle headline: `resolved (synthesis_<file>.md, YYYY-MM-DD)`; keep the toggle (do not delete).

Writing rules for META.md:

- **Toggle-list format.** Each `Q*` is a `<details>` block; sub-areas (e.g. JEPA Stage 1 vs fresh encoder) may nest child toggles. Inside each toggle: experiment IDs + file links only.
- **Headlines only.** The `<summary>` line is the question title + optional status (`open`, `partially resolved`, `resolved`). No finding prose inside toggles.
- Never delete a `Q*`. New questions get the next `Q*` number; numbers are never reused.
- Every claim must be derivable from an `idea_*.md` or `synthesis_*.md`. Write it there first; META only links.
- If a `Q*` toggle would need more than ~5 link lines, factor detail into a `synthesis_*.md` and link that synthesis instead of many idea files.
- Cross-experiment synthesis material lives in `synthesis_*.md` and is linked from META, never duplicated.

## Lifecycle Rules

- Before staging a run: add/update the checklist entry and idea file, including verifiables.
- When submitting: record run name, parent run, submit script/config, SLURM job ID when known, and artifact paths.
- After completion: update findings from concrete evidence only.
- When superseded: keep the old entry, mark it `superseded`, and link to the successor.
- Never delete historical ideas just because they failed.

## Evidence Rules

Use `sandbox/runs/<run>/metrics.jsonl` as the primary metric source and `resolved_config.yaml` as the executed-config source. SLURM logs explain walltime/OOM/failure status. `sandbox/hpo_graph.json` is useful for lineage but may be incomplete; say so when nodes or result fields are missing.

For run analysis, prefer the `log-observability` skill and scripts. Do not claim a winner without the standard ranking evidence. If a head-isolation run lacks aggregate `eval_losses/total_loss` or `quality_score`, document branch-specific findings instead of forcing a composite comparison.

## File Roles

- `sandbox/ideas/META.md` - open research questions (`Q*`) as nested toggle lists (headlines + file links only). Updated on every relevant change.
- `sandbox/ideas/EXPERIMENTS.md` - central checklist and navigation index. No artifacts, no synthesis, no config tables.
- `sandbox/ideas/config_promotions.md` - promoted default config changes (tables and evidence links).
- `sandbox/ideas/idea_template.md` - template for new ideas.
- `sandbox/ideas/idea_*.md` - per-run detailed hypothesis, intervention, verifiables, risks, artifacts, and findings.
- `sandbox/ideas/synthesis_*.md` - cross-run rollups (multi-run findings, sweep summaries). Owned by the `sandbox-synthesis` skill.
- `.cursor/skills/sandbox-synthesis/` - required structure and conventions for synthesis docs.
- `.cursor/skills/log-observability/` - deterministic run-log analysis helpers (cited from idea and synthesis files).

Keep this skill concise. Add detailed conventions to the hub docs, not here.

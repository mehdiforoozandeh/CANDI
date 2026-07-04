---
name: candi-autoresearch
description: >-
  Design and run Karpathy-style autoresearch harnesses in CANDI/EpiDenoise.
  Use when planning FAFO loops, writing prepare.py/train.py/program.md,
  scope-fenced agent experiment search, or sandbox/autoresearch/* tags.
  Not for full sandbox production runs — those follow sandbox-idea-hub.
---

# Autoresearch — design and operation

Autonomous experiment search: an agent edits one file, trains under a **fixed budget**, scores against an **immutable eval harness**, keeps improvements via git ratchet, reverts failures. Based on [Karpathy autoresearch](https://github.com/karpathy/autoresearch) ([program.md](https://github.com/karpathy/autoresearch/blob/master/program.md)); overview in [DataCamp guide](https://www.datacamp.com/tutorial/guide-to-autoresearch).

**Not** Optuna/Ray (predefined search space). **Not** open-ended vibe coding. The LLM proposes structural changes; `prepare.py` is the neutral judge neither side edits mid-loop.

---

## When to use

- Hypothesis has **many coupled knobs** (architecture + loss + hparams) and you want FAFO, not a grid.
- You need **fast iteration** on a pinned subset (single batch, short wall time) before a full sandbox A/B.
- You can state **one scalar objective** and a small set of guard-rails.

**Skip autoresearch when:** the question is a single known config diff (run one sandbox job); you need multi-GPU or full-data stats; the bug is in eval/data pipeline (fix directly, then autoresearch).

**See also `candi-era`:** when this greedy single-trajectory loop **plateaus in a local optimum**, or you want a *population* tree search over whole-program candidates with a PUCT bandit and a diverse portfolio of winners, switch to the ERA/FUTS skill (`candi-era`).

---

## Three-file contract

| File | Who edits | Role |
|------|-----------|------|
| **`prepare.py`** | Human at setup only | Data pin, model shell, eval metric(s), OOM/timeout guards, parseable run footer. **Frozen during loop.** |
| **`train.py`** | Agent every iteration | Model, optimizer, loss weights, training loop — everything that fits the budget. |
| **`program.md`** | Human (steering) | Research agenda, keep/discard rules, priors, failure policy, **"NEVER STOP"** loop instruction. |

Division of labor (Karpathy): human = research advisor via `program.md`; agent = executor on `train.py`; `prepare.py` = yardstick.

Optional harness files (human setup): `scope.py`, `agent_step.py`, `loop.sh`, `.gitignore` for `results.tsv` / `run.log`.

---

## Designing a new harness

### 1. Write the design doc first

Before code, capture in `sandbox/ideas/autoresearch_<tag>_*.md`:

- Problem, primary metric (maximize or minimize), guard-rail metrics
- Fixed vs tunable split with rationale
- Known hypotheses / axes to FAFO (prioritized)
- Success criteria and promotion path to a real sandbox experiment
- Register in `META.md` + `EXPERIMENTS.md`

### 2. Pick location and scope fence

```
sandbox/autoresearch/<tag>/     # preferred for new harnesses
```

**Hard rule:** during the loop, git changes stay inside the tag directory. Import production code read-only; copy/wrap/fork eval or model code *inside the tag dir* if behavior must differ — never edit `sandbox/candi_v2/`, `sandbox/train.py`, or data mid-loop.

Implement `scope.py` that fails if staged paths fall outside the tag prefix.

### 3. Fix the comparison budget

Karpathy uses **fixed wall-clock** (5 min train) so faster and slower configs compete fairly. Alternatives: fixed step count, fixed token count — pick one and never mix.

Every run must report budget used (`training_seconds`, `num_steps`, etc.) in the footer for debugging.

### 4. Design the metric

**Primary:** one scalar in `program.md` (e.g. minimize `val_bpb`, maximize `imp_count_r2_gw`).

**Guard-rails:** metrics that must not regress (e.g. Pearson ≥ baseline − ε, VRAM ≤ cap). A primary win with guard-rail failure = **discard**.

**Properties of a good primary metric:**

- Computable in `prepare.py` without agent cooperation
- Stable on the pinned eval set (avoid near-zero-variance slices — see F4 in log-observability FINDINGS)
- Aligned with what you will promote to full training

Print a **machine-parseable footer** every run (grep-friendly):

```
---
primary_score:    <float>
peak_vram_mb:     <float>
status:           ok|crash|oom
...diagnostic keys...
---
```

State in `program.md` whether lower or higher `primary_score` is better.

### 5. Wire `prepare.py`

- Pin: dataset slice, seed, batch, model preset, eval batches
- Train by importing `TrainConfig` from `train.py` (agent edits config/constants there)
- Eval after train; compute primary + guard-rails
- Enforce VRAM ceiling and run timeout; mark `peak_vram_ok`
- First run = **baseline**; optionally write `baseline.json` for VRAM reference

Reference implementation patterns: `sandbox/diagnostics/autoresearch/` (May 2026 count-head run) — copy ideas into new tag dir, do not depend on it at runtime unless intentional.

### 6. Wire `train.py`

Expose tunables as a **`TrainConfig` dataclass** or top-level constants — one obvious place for the agent to edit. Document defaults matching a sensible baseline.

Keep the file self-contained enough that the agent can rewrite architecture without touching imports across the repo.

### 7. Write `program.md`

Include explicitly (Karpathy checklist):

- Run tag, branch name (`autoresearch/<tag>`)
- **Goal** and primary metric direction
- **CAN** / **CANNOT** lists (mirror three-file contract)
- Baseline numbers to beat (fill after first run)
- Exact run command and `grep` patterns for scores
- **Keep rule:** e.g. primary improved AND guard-rails pass AND `peak_vram_ok`
- **Simplicity criterion:** small gain + ugly complexity → discard; equal/better after deleting code → keep
- **Crash policy:** fix typos once; fundamental breakage → log `crash`, revert, move on
- **Timeout:** kill runs exceeding 2× budget → crash
- **NEVER STOP** after loop starts — no asking human to continue; ~12 runs/hour at 5 min each
- Domain priors: what usually fails, what axes to try when stuck (from design doc)

### 8. Baseline and loop

1. Run unmodified `train.py` → record row 0 in `results.tsv`
2. Agent loop: one hypothesis → edit `train.py` → scope check → commit → train → parse footer → append TSV → keep or `git reset --hard HEAD~1`
3. `results.tsv`: tab-separated, gitignored — `commit`, primary score, memory, status, description

---

## Ratchet loop (9 steps)

1. Read `program.md`, recent `results.tsv`, current `train.py`
2. One hypothesis
3. Edit `train.py` only
4. Commit (after scope check)
5. Run train → `run.log` (redirect stdout; don't flood context)
6. Parse footer; on empty grep → `tail run.log`, fix once or crash
7. Append `results.tsv`
8. **Keep** if improved + guard-rails; else **reset** to last keep
9. Repeat without pausing

Git history = validated lineage. Failures leave no permanent diff.

---

## `program.md` vs design doc

| Design doc (`sandbox/ideas/…`) | `program.md` (in tag dir) |
|-------------------------------|---------------------------|
| Full hypothesis space, code refs, success criteria | Actionable agent instructions |
| Permanent experiment record | Living steering for one loop session |
| Human + implementer audience | Agent audience |

After the loop: write `synthesis_<exp>_*.md`, update META/EXPERIMENTS. **Promoting** a winning recipe → separate controlled sandbox run (single-knob A/B), not merge autoresearch commits into production.

---

## Anti-patterns

- Editing `prepare.py` mid-loop (invalidates cross-run comparison)
- Multiple files edited per iteration (confounds cause)
- Composite primary metric without a clear main term (agent optimizes noise)
- W&B as source of truth instead of local footer/TSV
- Declaring victory on a proxy metric when the design doc targets something else
- No scope fence → agent patches production and breaks teammates
- Skipping baseline run

---

## CANDI-specific defaults

- GPU: `--gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1` on SLURM
- Env: `conda activate candi`, `source candi_venv/bin/activate`, `module load samtools`, `PYTHONPATH=$PWD`
- Full training validation still uses `sandbox/train_candi_v2.py` + `metrics.jsonl` — autoresearch only **hypothesizes** configs

---

## Quick checklist (implementer)

- [ ] Design doc + META + EXPERIMENTS entry
- [ ] `sandbox/autoresearch/<tag>/` with prepare / train / program / scope
- [ ] Fixed budget + primary metric + guard-rails documented
- [ ] Parseable footer + `results.tsv` schema
- [ ] Baseline run recorded
- [ ] `program.md` includes NEVER STOP + simplicity criterion
- [ ] Promotion path defined (out of scope for loop)

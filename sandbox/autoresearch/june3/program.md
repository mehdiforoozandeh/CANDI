# E34 june3 — architecture autoresearch

**Branch:** `autoresearch/june3`  
**Spec:** `sandbox/ideas/autoresearch_june3_arch.md`

## Setup

```bash
cd ~/projects/def-maxwl/mforooz/EpiDenoise
conda activate candi && source candi_venv/bin/activate && module load samtools
export PYTHONPATH=$PWD
python -m sandbox.autoresearch.june3.validate_parity
python -m sandbox.autoresearch.june3.train 2>&1 | tee sandbox/autoresearch/june3/run.log
```

## Scope

Loop: edit files under `sandbox/autoresearch/june3/` **except** `prepare.py`, `pins.py`, `ar_fixed.yaml`, `pin_manifest.json`, `validate_parity.py`, `program.md`, `README.md`.

```bash
python -m sandbox.autoresearch.june3.scope --staged
git add sandbox/autoresearch/june3/
git commit -m "autoresearch/june3: <description>"
```

## CAN

- Change `candi_v2/` encoder, decoder, fusion, FiLM, transformer stack
- Change `train.py::get_config()` structural hyperparameters (width, depth, dropout, …)
- Add new modules under `june3/`

## CANNOT

- Edit `prepare.py`, `pins.py`, eval or train loop
- Change count head type, `depth_center`, loss weights, epochs, pin manifest
- Touch files outside `june3/`

## Primary (higher is better)

`primary_score` from footer — see spec.

## Keep

`primary_score` improved vs `results.tsv` best AND guard-rails pass AND `peak_vram_ok` AND `param_ok` (params ≤ **5×** baseline; VRAM ≤ 9500 MB).

**Guard-rails:** `imp_count_r2_gw > 0`, `den_count_r2_gw > 0`, `depth_count_ratio ∈ [3, 5]`, finite count losses.

**Simplicity:** small gain + large complexity → discard.

**Crash:** fix typo once; else revert (`git reset --hard HEAD~1`), log, continue.

**NEVER STOP** after loop starts.

---

## Overnight loop (~5 h interactive session)

**You are ready** when all of these pass on the GPU node:

```bash
python -m sandbox.autoresearch.june3.validate_parity
python -m sandbox.autoresearch.june3.validate_data_frac
# baseline row 0 already in results.tsv from vendored CANDI v2
```

Budget: **~4 min/train** × 20 epochs → plan **~15–25 experiments** in 5 h (agent overhead dominates).

### Option A — Cursor Agent (recommended)

1. SSH to your interactive node (`squeue -u $USER`), `cd` repo, `conda activate candi`, `source candi_venv/bin/activate`, `module load samtools`, `export PYTHONPATH=$PWD`.
2. Checkout `autoresearch/june3`. Open **Agent** mode (not Ask). Enable **auto-run / YOLO** so shell commands do not wait for approval.
3. Paste the **startup prompt** below as the first message. Do **not** start a second agent on the same branch.
4. Optional watchdog (separate terminal, does **not** edit code — only useful if you wire an external mutator): `bash sandbox/autoresearch/june3/loop.sh` — **not** a substitute for the Cursor agent.

**Hard rules for the agent**

- Run until **wall clock ≥ 5 h from loop start** OR the interactive job ends — whichever comes first. Check with `date` and `squeue -j $SLURM_JOB_ID -o %L` when available.
- **Do not** ask the user questions mid-loop; **do not** post long summaries until time is up or the job dies.
- Each iteration: one architectural hypothesis → edit scope-allowed files → `scope --staged` → commit → `agent_step` → keep or `git reset --hard HEAD~1` → immediately next hypothesis.
- Read `results.tsv` and `run.log` footer each time; ratchet only on `keep` rows.

### Startup prompt (copy verbatim)

```
E34 june3 architecture autoresearch — unattended ~5h loop.

Read sandbox/autoresearch/june3/program.md and sandbox/ideas/autoresearch_june3_arch.md.
Branch autoresearch/june3. Env: conda candi, candi_venv, module samtools, PYTHONPATH=$PWD.
Read sandbox/autoresearch/june3/AGENT_SYSTEM_PROMPT.md first.
NEVER edit sandbox/train.py or any file outside sandbox/autoresearch/june3/.
Work ONLY under sandbox/autoresearch/june3/ except frozen harness files in AGENT_SYSTEM_PROMPT.md.

NEVER STOP until 5 hours after your first agent_step OR the SLURM interactive job ends.
Do not ask me questions. Do not pause for approval summaries.

Loop forever:
1. Propose one architecture change (encoder/decoder/fusion/transformer/FiLM in candi_v2/ or train.py get_config).
2. python -m sandbox.autoresearch.june3.scope && python -m sandbox.autoresearch.june3.scope --staged && git add sandbox/autoresearch/june3/ && git commit -m "autoresearch/june3: <short desc>"
3. python -m sandbox.autoresearch.june3.agent_step --description "<short desc>"
4. If run.log shows keep=keep: stay on branch. Else: git reset --hard HEAD~1
5. On crash: one typo fix retry; else reset and continue.
6. Go to step 1.

Maximize primary_score. Guards: imp_count_r2_gw>0, den_count_r2_gw>0, dcr in [3,5], param_count≤5× baseline, peak_vram_ok.
Baseline primary_score is in results.tsv row 0 (~-2.8); beat the best kept row only.

Log each experiment in one line to sandbox/autoresearch/june3/agent_notes.txt (commit, primary, keep/discard, one-line lesson).
```

### Option B — shell watchdog only

`loop.sh` re-runs `agent_step` without new commits — **useless** for architecture search unless the Cursor agent is also committing. Ignore unless testing harness stability.

### If the agent stops early

Resume with: "Continue E34 june3 AR from results.tsv best row; NEVER STOP; same rules as program.md; do not re-baseline."

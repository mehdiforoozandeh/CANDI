# E32 autoresearch harness (May 31)

Karpathy-style FAFO loop for imp count R² calibration (Q11 / E32).

Full specification: [`sandbox/ideas/autoresearch_may31_r2vscorr_disparity.md`](../../ideas/autoresearch_may31_r2vscorr_disparity.md)

Reference structure (copy only): `sandbox/diagnostics/autoresearch/`

## Quick start

```bash
cd ~/projects/def-maxwl/mforooz/EpiDenoise
conda activate candi && source candi_venv/bin/activate && module load samtools
export PYTHONPATH=$PWD

# Baseline (row 0)
python -m sandbox.autoresearch.may31.train 2>&1 | tee sandbox/autoresearch/may31/run.log

# Scope check before loop commits
python -m sandbox.autoresearch.may31.scope --staged

# Agent loop step
python -m sandbox.autoresearch.may31.agent_step --description "..."

# Optional A1 ablation (row 0b)
python -m sandbox.autoresearch.may31.validate_a1
```

**Branch:** `autoresearch/may31`  
**Scope:** edit only `sandbox/autoresearch/may31/` during setup; **`train.py` only** during loop.

Agent playbook: [`program.md`](program.md) (Session 2 — imp phase, `DEN_GATE=0.28`)

## Session 2 quick start

After pulling harness updates, one verification run (appends 4 VB train pins + confirms imp phase):

```bash
python -m sandbox.autoresearch.may31.train 2>&1 | tee sandbox/autoresearch/may31/run.log
# expect metric_phase: imp with exp23-class config
```

Then paste the agent prompt from `program.md` into Cursor.  
Architecture context (read first): [`AGENT_ARCHITECTURE_CONTEXT.md`](AGENT_ARCHITECTURE_CONTEXT.md)

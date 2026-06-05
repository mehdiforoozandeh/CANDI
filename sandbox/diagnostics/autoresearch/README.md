# Autoresearch — CANDI v2 diagnostics

Karpathy-style autonomous loop on real chr19 (`sandbox/data/sandbox.h5`).

**Scope:** autoresearch may only **edit** files under `sandbox/diagnostics/`. During the experiment loop, only `autoresearch/train.py` changes. Production code (`sandbox/candi_v2/`, repo-root `train.py`, etc.) is read-only.

Verify before commit:
```bash
python -m sandbox.diagnostics.autoresearch.scope --staged   # after git add train.py
```

## Quick start

```bash
# Compute node (4 h SLURM)
cd ~/projects/def-maxwl/mforooz/EpiDenoise
conda activate candi && source candi_venv/bin/activate && module load samtools
export PYTHONPATH=$PWD

python -m sandbox.diagnostics.autoresearch.train
```

## Files

| File | Role |
|------|------|
| `prepare.py` | Fixed harness (do not edit) |
| `train.py` | **Agent edits** — count head, optimizer, loss weights |
| `program.md` | Agent playbook |
| `agent_step.py` | One run + append TSV |
| `scope.py` | Scope guard — diagnostics-only edits |

## Fixed vs tunable

**Fixed (prepare.py):** sandbox CANDIv2 shell (~0.3M params), chr19 data, 1500 steps, batch=4, assay-only mask (`p_full_assay=1`, `p_full_loci=0`, `p_chunks=0`).

**Tunable (train.py):** count head / depth offset, optimizer name + hparams, loss weights.

## Objectives (composite score)

- Imputation: assay-only mask, `imp_pearson`, `y_depth_dcr_on_masked_bins`
- Denoising: DSF input, reconstruction MAE
- Metadata: x_meta latent deltas via probe battery

## OOM safety

10 GB H100 slice — cap 9500 MB peak; baseline +10% reject; head-only edits.

## Session

4 h wall → ~200 experiments. Chain sessions via `autoresearch/<tag>` branch + `results.tsv`.

See `../FINDINGS.md` and `../META_CONDITIONING.md` for diagnostic context.

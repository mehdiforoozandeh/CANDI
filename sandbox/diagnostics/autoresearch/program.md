# CANDI diagnostics autoresearch

Autonomous experiment loop for CANDI v2 count-head research on real chr19 data.

## Setup

1. **Run tag**: propose `may28` (or today's date). Branch `autoresearch/<tag>` must not exist.
2. **SLURM** (4 h max):
   ```bash
   srun --account=def-maxwl --cpus-per-task=2 \
     --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1 \
     --mem=14G --time=4:00:00 --pty bash
   ```
3. **Env** on compute node:
   ```bash
   cd ~/projects/def-maxwl/mforooz/EpiDenoise
   conda activate candi && source candi_venv/bin/activate && module load samtools
   export PYTHONPATH=$PWD
   ```
4. **Read** (full context): `README.md`, `prepare.py`, `train.py`, `../FINDINGS.md`, `../META_CONDITIONING.md`
5. **Branch**: `git checkout -b autoresearch/<tag>`
6. **Init** `results.tsv` header only (agent_step creates it on first run)
7. **Baseline**: run unmodified `train.py` once

## Scope boundary (hard rule)

**All autoresearch work stays inside `sandbox/diagnostics/`.** Nothing else in the repo may be edited, staged, or committed.

| Allowed | Forbidden |
|---------|-----------|
| Read/import `sandbox/candi_v2/*`, `sandbox/batch.py`, etc. | Edit `train.py` (repo root), `sandbox/candi_v2/*`, `sandbox/train.py`, configs, data |
| Edit **`sandbox/diagnostics/autoresearch/train.py`** during the loop | Edit `prepare.py`, `../FINDINGS.md`, or any file outside `sandbox/diagnostics/` mid-loop |
| Append gitignored artifacts (`results.tsv`, `run.log`) | Commit files outside `sandbox/diagnostics/` |

Before **every** commit in the loop:

```bash
python -m sandbox.diagnostics.autoresearch.scope
git add sandbox/diagnostics/autoresearch/train.py
git commit -m "autoresearch: <description>"
```

If scope check fails: `git checkout -- .` outside diagnostics (or `git reset --hard`), fix, retry.

## What you CAN do

- Modify **`sandbox/diagnostics/autoresearch/train.py` only**:
  - **Count head:** depth offset formula, `depth_center`, NB parameterization
  - **Optimizer:** `optimizer` (`adam` | `adamw` | `adamax` | `sgd`), `lr`, `weight_decay`, `beta1`, `beta2`, `eps`, `sgd_momentum`, `clip_norm`
  - **Loss weights:** `obs_weight`, `imp_weight`, `count_weight`

## What is fixed in `prepare.py` (not tunable)

- **Model shell:** sandbox CANDIv2 ~**0.3M params** (L=768, 8 assays, 2 transformer layers — not production ~42M CANDI)
- **Data:** `sandbox/data/sandbox.h5`, chr19, batch_size=4, pinned single batch
- **Steps:** 1500
- **Masking:** `p_full_assay=1`, `p_full_loci=0`, `p_chunks=0` (assay-only imputation)
- Eval suite, composite score weights, OOM guards

## What you CANNOT do

- Modify any file **outside** `sandbox/diagnostics/`
- Modify `prepare.py` (data, eval, composite score, OOM guards, architecture shell)
- Edit `sandbox/candi_v2/*` or promote changes to production
- Add new dependencies or widen encoder/decoder trunk
- Add `nn.Module` children beyond the count NB head swap
- Increase batch size or model width (frozen in prepare)
- Install packages

## Goal

Minimize **`composite_score`** (lower is better). Weighted over:

1. **Imputation** — `imp_pearson`, `dcr_masked_bins` (masked-bin probe, not diluted assay sum)
2. **Denoising** — `denoise_rel_mae` on DSF-corrupted input
3. **Metadata** — `x_depth_latent_delta`, `x_readlen_latent_delta`

## Run command

```bash
python -m sandbox.diagnostics.autoresearch.train > sandbox/diagnostics/autoresearch/run.log 2>&1
grep "^composite_score:\|^peak_vram_mb:\|^peak_vram_ok:" sandbox/diagnostics/autoresearch/run.log
```

## Output format

```
---
composite_score:  0.142300
imp_pearson:      0.984100
dcr_masked_bins:  4.241000
denoise_rel_mae:  0.061200
x_depth_latent_delta: 0.002100
peak_vram_mb:     8234.100000
peak_vram_ok:     True
status:           ok
---
```

## results.tsv

Tab-separated:

```
commit  composite_score  memory_gb  peak_vram_ok  status  description
```

Status: `keep` | `discard` | `crash`

**Keep rule:** `composite_score` strictly decreased vs best so far **AND** `peak_vram_ok: true` **AND** status = ok

## Experiment loop — NEVER STOP

Run on compute node for the **full 4 h SLURM window**. ~200 runs at ~60 s each.

LOOP until SLURM kills the job or human interrupts:

1. Note current branch/commit
2. Edit **`sandbox/diagnostics/autoresearch/train.py`** with one experimental idea
3. `git add sandbox/diagnostics/autoresearch/train.py`
4. `python -m sandbox.diagnostics.autoresearch.scope --staged` — must print `scope ok`
5. `git commit -m "autoresearch: …"`
6. Run train (redirect to run.log)
6. Parse summary from log
7. Append `results.tsv`
8. If improved (keep rule): stay on commit
9. Else: `git reset --hard HEAD~1` (or reset to last kept)
10. **Immediately** start next experiment — do NOT pause, summarize, or ask the human

### On crash / OOM

- Log `status=crash`, score=9.999
- `git reset --hard` to last kept commit
- Continue loop (prepare cleans GPU memory)

### VRAM discipline (10 GB slice)

- Never add layers beyond count head
- Prefer simpler changes
- If `peak_vram_ok: false`, discard even if score improved

### Diagnostic priors (from completed runs)

- Default NB head fails Q5 (dcr≈1) — keep offset unless strong alternative
- Raw `2^d` without center fails on real data — use `depth_center≈24`
- Imputation eval: trust `dcr_masked_bins`, not `y_depth_dcr_masked_assays`
- Runtype unused in count head — low priority
- Encoder FiLM is x_meta path — frozen; focus on head/loss

### Out of ideas

Re-read `results.tsv`, combine near-misses, sweep `depth_center`, `obs_weight`/`imp_weight`, offset formula variants. **Do not stop.**

## Resume next 4 h session

Same branch, append `results.tsv`, continue from last kept commit.

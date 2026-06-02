# E32 autoresearch — Session 2 (imp R² phase)

**Branch:** `autoresearch/may31`  
**Spec:** `sandbox/ideas/autoresearch_may31_r2vscorr_disparity.md`  
**Architecture:** `AGENT_ARCHITECTURE_CONTEXT.md`

Session 1 (31 exps): den_r2 **0.335** max; **`lambda_mse_imp` never tried**; never entered imp phase (gate was 0.35).

Session 2 changes (human, in `prepare.py`):
- **`DEN_GATE = 0.28`** (was 0.35; 0.28 after +4 VB train pins → imp-phase search)
- **+4 train pins** with V/B metadata (`session2_vb_train_added` in manifest)
- Footer: **`imp_r2_task_gap`** = cloze_T − vb imp R²

**Seed config in `train.py`:** exp23 (`dsf=off`, `dc=23`, `obs=3.5`, `imp=0.5`, `count=2.0`, `mse_obs=0.2`).

---

## Setup (compute node)

```bash
cd ~/projects/def-maxwl/mforooz/EpiDenoise
conda activate candi && source candi_venv/bin/activate && module load samtools
export PYTHONPATH=$PWD
git checkout autoresearch/may31
git pull   # if needed
```

First run after session-2 update appends 4 VB-meta train batches to `pin_manifest.json` once.

---

## Scope

During the loop: edit **`train.py` only**.

```bash
python -m sandbox.autoresearch.may31.scope --staged
git add sandbox/autoresearch/may31/train.py
git commit -m "autoresearch/may31 s2: <description>"
```

---

## Goal (Session 2)

**Maximize `primary_score`** with **`DEN_GATE=0.28`**:

- **`metric_phase=imp`:** `primary_score = imp_count_r2_gw + 1.0` (when den ≥ 0.28 and DCR in band)
- Scientific targets: **`imp_count_r2_gw > 0`**, then **`> 0.15`**; keep **`den_count_r2_gw ≥ 0.28`**

---

## Mandatory experiment queue (first ~12 commits)

Do these **before** any `obs_weight` / `count_weight` micro-sweeps:

1. **`lambda_mse_imp`** ∈ {0.1, 0.2, 0.5, 1.0} (keep exp23 base)
2. **`lambda_mse_imp=0.2`** + **`imp_weight`** ∈ {0.75, 1.0, 1.5}
3. **`calib_loss="log2"`** + **`lambda_mse_imp=0.2`**
4. **`use_vb_meta_on_masked=True`** + **`lambda_mse_imp=0.2`**
5. **`signal_transform`** ∈ {`none`, `arcsinh`} (one at a time)

---

## Banned (session 2)

- May28 `imp=8, obs=0.5`
- `lr < 1e-3`
- `mse_on_log1p` (use **`calib_loss`** instead)
- Re-sweeping `dsf_sampling` / `depth_center` unless **den_r2 < 0.25**
- More than 2 consecutive `obs_weight` tweaks without `lambda_mse_imp` change

---

## Keep rule

ALL must pass:

```
primary_score > best_primary_score
depth_count_ratio in [3.25, 4.75]
peak_vram_ok == True
status == ok
```

When **`metric_phase=imp`** also:

```
den_count_r2_gw >= 0.28
imp_count_pearson_gw >= max(baseline_pearson - 0.05, 0.38)   # baseline_pearson from row 0 / session 1
```

**Tie-break** (primary within 1e-4): higher **`imp_count_r2_gw`**, then **`den_count_r2_gw`**.

**Optional keep:** if `imp_count_r2_gw` improves by **≥ 0.05** vs best imp, keep even if primary ties.

On failure: `git reset --hard HEAD~1`

---

## Run commands

```bash
python -m sandbox.autoresearch.may31.train 2>&1 | tee sandbox/autoresearch/may31/run.log
python -m sandbox.autoresearch.may31.agent_step --description "s2 expN ..."
grep -E 'primary_score|metric_phase|imp_count_r2|den_count_r2|imp_r2_task' sandbox/autoresearch/may31/run.log
tail -5 sandbox/autoresearch/may31/results.tsv
```

---

## Agent prompt (paste into Cursor)

```
Run E32 autoresearch SESSION 2 per sandbox/autoresearch/may31/program.md.

Read program.md and AGENT_ARCHITECTURE_CONTEXT.md. Session 1 results in results.tsv (31 rows).

Session 2 is IMP PHASE: DEN_GATE=0.28 in prepare.py. Starting train.py = exp23 seed.
Edit train.py ONLY. NEVER edit prepare.py.

First 8 experiments MUST sweep lambda_mse_imp (0.1, 0.2, 0.5, 1.0) before obs/count weight tweaks.
Then: imp_weight, calib_loss=log2, use_vb_meta_on_masked=True, signal_transform.

Keep if primary_score improves + guard-rails pass. Tie-break on imp_count_r2_gw.
Target: imp_count_r2_gw > 0, den_count_r2_gw >= 0.28, DCR ~4.
Watch imp_r2_task_gap in footer — if cloze_T >> vb after mse_imp, flag task-gap.

NEVER STOP until SLURM wall or I say stop. ~2.5 min/run.
```

---

## NEVER STOP

~20–24 GPU runs/hour (~2.5 min each). Stop only on SLURM wall or human interrupt.

If eval harness seems wrong, **stop and ask human** — do not edit `prepare.py` mid-loop.

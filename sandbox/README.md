# CANDI sandbox

Fast, self-contained prototype training on a small HDF5 slice of the EIC panel (8 assays, top biosamples from `sandbox/selection.py`).

## Setup

Use the same Conda env as production CANDI (e.g. `conda activate candi`), repo root on `PYTHONPATH`, and **PyYAML** for config files (`pip install pyyaml` if `import yaml` fails).

## One-time data prep

1. **Selection (Gate A)** — writes `sandbox/data/selection.json`:

   ```bash
   python -m sandbox.selection --gate-a
   ```

2. **Bake HDF5** — builds `sandbox/data/sandbox.h5` (or a smoke file):

   ```bash
   python -m sandbox.prepare_h5 bake \
     --eic-data /path/to/DATA_CANDI_EIC \
     --selection sandbox/data/selection.json \
     --out sandbox/data/sandbox.h5
   ```

   Quick smoke (few windows): add `--max-windows 32`.

3. **Parity (Gate B)** — compares baked tensors to `reference_tensors` online:

   ```bash
   python -m sandbox.prepare_h5 validate-parity \
     --eic-data /path/to/DATA_CANDI_EIC \
     --h5 sandbox/data/sandbox.h5 \
     --parity-ok sandbox/data/parity.ok
   ```

   CI-friendly: `--parity-fast`.

4. **Overfit sanity (Gate)** — 200 optimization steps on one batch; fails if loss does not drop:

   ```bash
   python -m sandbox.prepare_h5 overfit-sanity --h5 sandbox/data/sandbox_smoke.h5
   ```

   Uses `--regime auto` by default (tries `type1_chr19`, then `type2_loci`).

## Training

**Type-1 (chr19 tiling)** — preset overlay:

```bash
python -m sandbox.train \
  --config sandbox/configs/default.yaml \
  --config sandbox/configs/type1_chr19.yaml \
  --h5 sandbox/data/sandbox.h5 \
  --epochs 1 \
  --no-wandb
```

**Type-2 (cCRE / non-cCRE loci)**:

```bash
python -m sandbox.train \
  --config sandbox/configs/default.yaml \
  --config sandbox/configs/type2_loci.yaml \
  --h5 sandbox/data/sandbox.h5 \
  --epochs 1 \
  --no-wandb
```

Merge order for hyperparameters: **dataclass defaults** → packaged `sandbox/configs/default.yaml` (if present) → each `--config` YAML (deep-merge in order) → shortcut flags (`--h5`, `--epochs`, …) → dotted overrides such as `--data.regime type2_loci` or `--training.optimizer.name adamw`. Unknown keys in YAML or CLI raise a clear error.

Inspect the fully merged config without training:

```bash
python -m sandbox.train --print-config
python -m sandbox.train --dry-run
```

**Weights & Biases** — add `--wandb` (and omit `--no-wandb`); project name defaults to `candi_sandbox`.

**Optional diagnostic / safety flags** (all OFF by default — sandbox is for diagnostics, not production training):

- `--save-checkpoint` — write `<run_dir>/checkpoint_last.pt` at end of training.
- `--early-stop` — stop when `eval_losses/total_loss` is strictly above its best for `--early-stop-patience` consecutive eval points. Triggers only on rising loss (divergence/overfitting), never on plateau.
- `--early-stop-patience N` — tighten/loosen the patience (default 5 eval points = 5 × `eval_every_n_epochs` epochs).

**Unified entrypoint** (forwards to submodules):

```bash
python -m sandbox.cli train --h5 sandbox/data/sandbox.h5 --epochs 1 --no-wandb
python -m sandbox.cli prepare-h5 bake --eic-data /path/to/EIC --out sandbox/data/x.h5
python -m sandbox.cli gates all    # pytest sandbox/tests
python -m sandbox.cli gates c      # model forward/backward smoke
python -m sandbox.cli gates d      # config + CLI + optimizer routing
```

## Six-gate validation protocol (plan §14)

| Gate | What it checks | How to rerun |
|------|----------------|--------------|
| **A** | `selection.json`: 5 biosamples, each has ≥1 `T_*` assay, union of `V_*`/`B_*` non-empty | `python -m sandbox.cli gates a` |
| **B** | Baked H5 tensors match `data.py` reference | `python -m sandbox.prepare_h5 validate-parity --eic-data … --h5 …` |
| **C** | Model shapes, finite outputs, non-zero grads, toggle smoke | `python -m sandbox.cli gates c` |
| **D** | Nested config + YAML rejection + `--dry-run` / optimizers | `python -m sandbox.cli gates d` |
| **E** | Overfit one batch (200 steps) on GPU; strict head-wise loss drops | `python -m sandbox.cli gates e` (auto-sbatch; set `SANDBOX_USE_SBATCH=0` to run locally). Add `--relax-gate-e` for debugging only |
| **F** | 3-epoch `type2_loci` run (online wandb); validator checks monotonic loss, Pearson ≥ 0.30, AUROC ≥ 0.70, finite probes | `python -m sandbox.cli gates f`; validator: `python -m sandbox.validate_gate_f sandbox/runs/gate_f_attempt1` |
| **G** | Two 1-epoch runs (`type1_chr19`, `type2_loci`) under 90 min wall each; `resolved_config.yaml` round-trips; `parity.ok` present | `python -m sandbox.cli gates g`; validator: `python -m sandbox.validate_gate_g sandbox/runs/gate_g_type2_loci` |

`sandbox.cli gates {b,e,f,g}` submit via `sbatch` by default (scripts in `sandbox/slurm/`). Set `SANDBOX_USE_SBATCH=0` to run inline (e.g. inside an allocated `salloc`), and `SANDBOX_WAIT=0` to submit without `--wait`. Gate F retry attempts use `SANDBOX_F_ATTEMPT=1|2|3` to separate run dirs (`sandbox/runs/gate_f_attempt${n}`).

## Layout

| Path | Role |
|------|------|
| `sandbox/selection.py` | Top biosamples + Gate A |
| `sandbox/reference_sample.py` | EIC handler + tensor packing for bake/parity |
| `sandbox/prepare_h5.py` | `bake`, `validate-parity`, `overfit-sanity` |
| `sandbox/data.py` | `SandboxH5Dataset` (`IterableDataset`) |
| `sandbox/batch.py` | `DataMasker` wiring + `prepare_masked_batch` |
| `sandbox/model.py` | `build_sandbox_candi` (thin `CANDI` preset) |
| `sandbox/losses.py` | `CANDI_LOSS` + `SandboxCompositeLoss` (optional contrastive on `z`) |
| `sandbox/train.py` | Single-GPU loop, cosine LR per epoch, optional eval + prompt probes |
| `sandbox/eval.py` | MSE / RMSE / Pearson / Spearman / AUROC + prompt-sensitivity helpers |
| `sandbox/configs/*.yaml` | Defaults + regime overlays |
| `sandbox/config_types.py` | `SandboxConfig` nested dataclass schema |
| `sandbox/config.py` | Merge, dotted argparse, `config_from_dict` validation |
| `sandbox/gates.py` | Launcher for gates A/C/D (pytest) + B/E/F/G (sbatch wrappers) |
| `sandbox/slurm/*.sh` | SLURM templates for Gates B, E, F, G (Narval, `h100:1`) |
| `sandbox/validate_gate_f.py` | Gate F validator: reads `metrics.jsonl`, enforces Pearson/AUROC floors |
| `sandbox/validate_gate_g.py` | Gate G validator: wall-clock ≤90 min, `resolved_config.yaml` round-trip, `parity.ok` present |

## Ablation / knobs

- **Masking**: `p_full_assay`, `p_full_loci`, `p_chunks`, `mask_fraction`, `chunk_size` (YAML or extend `train.py` CLI).
- **DSF sampling** (train iterator): `dsf_sampling` — `uniform`, `x_eq_y`, `upsample_only`, `off`.
- **Contrastive regularizer** (on pooled latent): set `contrastive_weight` > 0 and trainer will call `forward(..., return_z=True)` (spread-style penalty; not full InfoNCE with paired views).
- **Aux metadata heads**: reserved on `SandboxCompositeLoss` (`aux_md_weight`); not wired on the base `CANDI` stub.

## Prompt sensitivity (logged when `--eval-each-epoch`)

- **`training_metadata_probes/depth_count_ratio`**: ratio of expected total NB counts at `probe_depth_hi` vs `probe_depth_lo` (defaults: 24 vs 22 in log2 space → 4× depth swing). **Healthy target ≈ 4.0; ≈1.0 means the model is ignoring the depth token (failure mode).**
- **`training_metadata_probes/readlen_mse`**, **`training_metadata_probes/runtype_mse`**: squared mean change in `mu` when perturbing read length (row 2) or run type (row 3).

Interpretation:
- `depth_count_ratio` is *quantitative* — it has a known target of ~4.0.
- `readlen_mse` and `runtype_mse` are *qualitative* — larger values mean the decoder responds more strongly to that metadata channel on the probe batch.

## Tests

From repo root with a working Torch install:

```bash
python -m pytest sandbox/tests
```

## Per-run artifacts

Every `sandbox.train` run writes to `cfg.training.run_dir`:

- `resolved_config.yaml` — fully-merged config (Gate G round-trip uses this).
- `metrics.jsonl` — append-only newline-delimited JSON. Two record kinds (discriminated by `"kind"`):
  - `kind: "epoch"` — written at end of each epoch. Fields: `epoch`, `global_step`, `epoch_seconds`, optional grouped families:
    - `eval_metrics` (Pearson/Spearman/R2/AUROC only)
    - `eval_losses` (unweighted branch losses + total weighted loss)
    - `training_metadata_probes` (`depth_count_ratio`, `runtype_mse`, `readlen_mse`, plus sensitivity probe MSEs)
    - `early_stop_triggered` / `early_stop_best_total_loss` / `early_stop_strikes` (only when early stop fires).
  - `kind: "training_step"` — written every `training.training_stats_jsonl_every_n_steps` steps (default 200, 0 disables). Fields: `epoch`, `global_step`, `training_stats` (step/lr/total_loss/grad-norm/clip), `training_losses` (per-branch raw losses), `training_grad_norms` (per-branch grad-norm). This is the offline source of truth for grad/clip diagnostics.
- `elapsed.txt` — total seconds (Gate G budget check).
- `checkpoint_last.pt` — only if `--save-checkpoint` (or `training.save_checkpoint: true`). Default OFF.

## Wandb layout

- `training_stats/*` (step/lr/total_loss and gradient clipping health)
- `training_losses/*` (unweighted branch losses + per-branch gradient norms)
- `training_metadata_probes/*` (`depth_count_ratio`, `runtype_mse`, `readlen_mse`, then sensitivity MSE keys)
- `eval_metrics/*` (Pearson/Spearman/R2/AUROC only, no MSE/RMSE)
- `eval_losses/*` (eval branch losses, no MSE/RMSE)
- `run/elapsed_seconds`, `epoch`
- Runs log to project `candi_sandbox` (online by default via `WANDB_MODE=online`).

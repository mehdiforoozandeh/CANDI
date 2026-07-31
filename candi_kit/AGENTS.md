# AGENTS.md — machine-actionable spec for `candi_kit/`

Audience: an LLM coding agent (and its human) working inside this directory with no prior context.
Read this file top to bottom before editing anything. Every technical claim below cites `file:line`
inside this kit or inside the research repo it was vendored from.

Companion human-facing docs in this directory: `README.md` (quickstart), `RECIPE.md` (architecture and
training recipe), `DATA.md` (input layout + h5 schema), `TRADEOFF.md` (the offset ON/OFF result),
`EXTENSION_HOOKS.md` (adding the p-value / peak heads), `.BUILD_PLAN.md` (the full manifest, knobs,
metric split, validation plan, and RISKS — the most detailed source of truth for *why* each file
looks the way it does).

---

## 1. ORIENTATION

`candi_kit/` is a **self-contained, vendored copy of one CANDI recipe**: the "q19 dual-conditioning"
model — a counts-only Negative-Binomial epigenome imputer/denoiser conditioned on 4 experimental
covariates on both the input and the output side. It contains everything needed to go from an
ENCODE-style directory of per-biosample count files to a trained checkpoint and a scored results JSON:
ingestion/bake (`src/candi_kit/prep/`), dataset + masking, model, training driver, evaluation stack,
tests, and SLURM scripts. It has **no imports back into the parent EpiDenoise repo** and runs from any
working directory once `pip install -e .` has been run.

What it is **not**:

* **Not a pretrained model.** No weights ship in this kit. `src/candi_kit/goldens/*.pt` are *forward
  output tensors* (`p, n, eta, log2_mu, mu`, each `[2,768,8]` float32 — verified by loading
  `goldens/wd0_on_s0.pt`), used only to prove the vendored code reproduces the original numerics. The
  historical `.ckpt` weight files live in the research repo, are referenced only by the optional
  Gate-A checkpoint test, and are **not** part of the kit. Training is from scratch.
* **Not full CANDI.** This is the counts-only NB head. There is no Gaussian p-value head and no
  Bernoulli peak head (`src/candi_kit/model.py:43`); therefore no peak precision/recall/AUROC and no
  p-value track. `y_pval` / `y_peaks` are carried through the loader and batch untouched
  (`src/candi_kit/batch.py:51-52,104-105`) purely as the extension hook.
* **Not a general framework.** It is one recipe with a frozen construction order and a frozen
  optimizer/loss/step order. Most "improvements" an agent might reflexively make (reordering module
  construction, renaming state_dict keys, "cleaning up" unreachable branches, normalizing covariates)
  are regressions here. See §3.

---

## 2. FILE MAP

Paths are relative to the kit root (`candi_kit/`). "DO NOT TOUCH" = editing changes recorded numerics
or breaks the Gate-A copy proof; change only with an explicit instruction and a re-run of §5.

### 2.1 `src/candi_kit/` — training path (importable without the `[prepare]` extra)

| file | lines | responsibility | do not touch |
|---|---|---|---|
| `__init__.py` | 22 | Exports only `EncoderConfig`, `V2Encoder`, `RealDualCondModel`, `build_real_model`, `forward_full`, `nb_nll`; defines `__version__`. | Do not add re-exports that pull the bake path or a loss stack — a bare `import candi_kit` must stay at ~5 third-party top-levels. |
| `_vendored.py` | 344 | The severance module: `MISSING=-1`, `CLOZE=-2`, `exponential_linspace_int`, `DataMasker` (verbatim from repo `_utils.py:36-353`). | `DataMasker` is RNG-load-bearing: `apply_mask` (`_vendored.py:238`) draws from the **global** torch RNG every step. Do not reorder, do not add draws. |
| `config.py` | 65 | `EncoderConfig` dataclass only (all encoder knobs, incl. `d_model=0` auto). | Do not add/remove fields or change defaults. |
| `encoder.py` | 974 | `MetadataEmbedding`, `SignalConvTower`, `MaskTokenInjector`, `DNAConvTower`, `LinearFusion`, `V2Encoder`. Contains added asserts at `:142`, `:162`, `:184` (4-row / assay_id / run_type bounds), `:876` (`transformer_layer_drop` must be 0.0), and the DNA-length check in `encode` (`:920-925`). | Blocks marked `# UNREACHABLE on the shipped path` (`:51`, `:302`, `:327`, `:571`, `:642`, `:654`) are kept **verbatim on purpose** — deleting them is a refactor with checkpoint-compat risk and zero benefit. |
| `decoder.py` | 185 | `DeconvBlock`, `DeconvTower`, `_build_deconv_channel_schedule`, `DecoderTrunk`. `norm='layer'` only. | The trunk is ~91% of model parameters; its channel schedule fixes state_dict shapes. |
| `model.py` | 277 | `_LegacyPlaceholderMetaEmbedder`, `RealMetaEmbedder` (alias of `MetadataEmbedding`), `DualCondDecoder`, `RealDualCondModel`, `build_real_model`, and the NB helpers `forward_counts / forward_full / nb_mean / nb_nll / encode_latent`. | **FROZEN CONSTRUCTION ORDER** — banner at `model.py:1-16` enumerates the six changes that silently invalidate every recorded anchor. Read it before any edit. |
| `batch.py` | 148 | `make_masker` + `prepare_masked_batch`: masking, obs/imp maps, control append, availability cross-check. Verbatim from repo `sandbox/batch.py:15-148`. | Batch-key contract must stay byte-identical (that is what keeps `--compat-q19` valid). Control is concatenated **after** masking at `batch.py:121-123`. |
| `dataset.py` | 367 | `CandiKitH5Dataset` (iterable, HDF5-backed) + `h5_depth_center`. Reads scale from `h5.attrs`; refuses schema v1 (`dataset.py:131-135`). | `__iter__`'s emitted dict keys/shapes (`dataset.py:320-339`) must not move. |
| `metrics.py` | 137 | Numeric primitives only: `nb_crps`, `nb_quantile`, `calibration_pit_curve`, `ece`, `spearman`, `pearson`, `r2`, `_cos_dist`, `_steering_index`, `P_EPS`, `PIT_GRID`. | `nb_crps` is a closed-form NB CRPS (validated to ~2.4e-5 relative against exact discrete sums by `tests/test_metrics_primitives.py:40`). Do not "simplify". |
| `eval.py` | 1109 | The measurement stack: `build_eval_units`, `eval_M1`, `eval_M2`, `eval_M3`, `_dsf_counterfactual` (S14), the clustered bootstrap `_cluster_bootstrap_ci`, `evaluate()` (`:1025`) and a checkpoint-scoring CLI (`:1052`). Owns `DEPRECATED_VERDICTS` (`:49`). | Assay labels are threaded in as an argument, never declared. Deprecated keys must keep their verdict strings. |
| `train.py` | 347 | Training driver + CLI: NB loss, cosine warmup, `_train_step`, full-coverage round-robin, `train_and_eval`, config echo, results JSON. | `train.py:78-87` (step order), `:98` (`Adam`, coupled L2, positional lr), `:228-231` (cudnn determinism pair) are FROZEN. Never add `torch.use_deterministic_algorithms` — it crashes this encoder. |
| `compat.py` | 150 | Gate A: `build_compat_q19`, `state_dict_sha1`, the frozen `synthetic_batch`, golden comparison, `verify`, CLI. | `synthetic_batch` (`:53`) is the frozen Gate-A input; changing its draw order invalidates every golden and the forward digests. |
| `report.py` | 350 | Regenerates a markdown scorecard + matplotlib figures from a results JSON alone (`[report]` extra, needs `MPLBACKEND=Agg`). CLI: `python -m candi_kit.report <results.json> [--outdir DIR]`. | Assay labels come from the results JSON `config.assays` (`report.py:40`), never a literal. |
| `goldens/*.pt` | 6 files | Frozen forward outputs (`init_on`, `init_off`, and four named after historical checkpoints). Shipped as package data (`pyproject.toml` `[tool.setuptools.package-data]`). | **Never regenerate.** `compat._check_golden` (`compat.py:86-98`) silently *writes* a golden if the file is missing — deleting one turns the copy proof into self-certification. |

### 2.2 `src/candi_kit/prep/` — bake path (needs the `[prepare]` extra: pysam, intervaltree, pandas)

| file | lines | responsibility | do not touch |
|---|---|---|---|
| `panel.py` | 57 | `Panel` dataclass + `load_panel` with strict unknown-key rejection; validates ≥2 `T_` biosamples, `context_bins % 8 == 0`, square `resolution`, disjoint train/eval chroms. | The `%8` and perfect-square rules encode the encoder's pool/block arithmetic; loosening them produces shape errors deep in the tower. |
| `paths.py` | 42 | `SideFiles(chrom_sizes, fasta, blacklist=None, ccres=None)` + `validate()`; loud warning when the blacklist is empty or absent. | — |
| `handler.py` | 1993 | `CANDIDataHandler`, vendored verbatim from repo `data.py` with 5 surgical edits (E1 side files/panel injection, E2 explicit `includes`, E3 DSF fallback now raises, E4 int16 overflow check, E5 unparseable `run_type` raises). Sole definition of assay→id ordering, covariate assembly, NPZ/FASTA/blacklist IO. | Do not reformat and do not "modernize". Edits E1 must stay inside `__init__` before `_load_blacklist`. |
| `reference_sample.py` | 191 | `make_handler` (`:20`) — builds the handler and runs the **panel/alias bijection gate** (`:46-52`); `resolve_column_order` (`:56`), `assert_panel_bijection` (`:61`), `reference_tensors`, `fixed_dsf_pair_maps`. | The bijection gate is the fix for the assay-order landmine. Never bypass it. |
| `bake.py` | 480 | ENCODE-style dir → HDF5 schema v2. Window tiling, optional type2 cCRE loci, per-(bios,chrom) bake cache, attrs, and the post-bake `_verify` gate (`:398-446`, checks F3 zero-filled meta / F4 DSF depth ladder / F7 raw counts / F15 control availability) which runs **before** the temp file is renamed into place (`:392-394`). CLI at `:446`. | `_enable_bake_cache` is load-bearing for runtime (minutes vs hours). `-1` (never `0`) is the sentinel written for an absent DSF level. |

### 2.3 Everything else

| path | responsibility |
|---|---|
| `tests/test_compat_q19.py` (172) | Gate A as pytest: param count, state_dict sha1, forward digests, construction-order source guards, placeholder size, adaLN-zero init, x-transformers pin. The 4 historical-checkpoint cases `pytest.skip` when the ckpt dir is absent (`:102`). |
| `tests/test_model.py` (238) | Synthetic-tensor model tests: sentinel handling, offset reads row 0, `offset=off ⇒ log2_mu == eta`, per-assay locality, latent invariance to `y_meta`, out-of-range `assay_id`/`run_type` raise, scale axes (`num_assays ∈ {3,8,16}` × `context ∈ {384,768,1536}`), `--d-model` decoupling, DNA-length mismatch raises. No h5 needed. |
| `tests/test_bake_gates.py` (371) | Ingestion landmines on a synthetic ENCODE-layout fixture in `tmp_path`: bijection error naming, `-1` vs `0` meta, DSF-ladder forgery detection, int16 overflow, unparseable run_type, 3-assay/384-bin round-trip into the model, raw-integer counts out of the loader. |
| `tests/test_metrics_primitives.py` (171) | `nb_crps` vs exact sums and Monte Carlo, PIT uniformity, ECE sensitivity, correlations vs scipy. Data-free. |
| `slurm/bake.sh` (60) | Bake job. Pure CPU *work*, but it still requests `--gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1` (`slurm/bake.sh:20`) — deliberately, to route through the GPU account; the reason is in the header comment at `:7-13`. |
| `slurm/train.sh` (56) | 2 arms × 3 seeds array (`--array=0-5`). `--gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1`. |
| `slurm/gate.sh` (69) | Fail-fast pre-flight: pytest → compat-q19 → CPU smoke → GPU smoke, propagating the first non-zero rc. |
| `configs/panel.q19.json` | The exact 8-assay / 10-biosample q19 panel. |
| `configs/panel.example.json` | Minimal 3-assay template. |
| `pyproject.toml` / `requirements-fir.txt` | Portable pins / literal Alliance-Canada (`+computecanada`) rebuild list. |
| `research/` (5 files) | The primary sources every result in this kit cites, copied from the research repo and frozen. **`research/README.md` carries a decoder ring for the `h<N>`/`q<N>` tracker ids those documents use** — the tracker itself does not ship, so without it those ids are dead links. |
| `configs/panel.gatec.json` | 5-assay / 512-bin panel used by Gate C to prove the scale knobs are real. |
| `slurm/gatec.sh` (44) | Gate C: bake + 3-epoch train at a non-q19 scale. |

The directory is named `slurm/`, **not** `jobs/`, because the parent repo's `.gitignore:15` has an
unanchored `jobs/` rule that would silently swallow the sbatch scripts. Avoid `models/` and `logs/`
for the same reason.

---

## 3. INVARIANTS — check yourself against this list after every edit

1. **Frozen construction order.** `RealDualCondModel.__init__` builds `V2Encoder` **first**, then
   replaces `encoder.metadata_embedding`, then builds `DualCondDecoder`
   (`model.py:214-223`). `DualCondDecoder.__init__` builds trunk → `_LegacyPlaceholderMetaEmbedder`
   → `film_proj` (constructed **then** zeroed) → `head_eta` → `head_n` → replace `meta_embedding`
   (`model.py:141-157`). All of these draw from the global torch RNG. Reordering, skipping the
   placeholder, or creating `film_proj` as a zero tensor changes the weight stream and breaks Gate A.
2. **The placeholder embedder is fixed at `8 × embed_dim`** (`model.py:83`). It is never used in
   forward; it exists solely to keep the RNG stream aligned with the historical construction. It must
   **not** scale with `num_assays`.
3. **No imports reaching back into the parent repo.** Nothing under `src/` may contain `from sandbox`,
   `from model import`, `from _utils import`, or `SANDBOX_ASSAYS`. Import-time third-party top-levels
   must stay within `{torch, numpy, h5py, x_transformers, einops, einx, loguru, sympy, scipy}` plus
   the `[prepare]`/`[report]` extras in `prep/`/`report.py`.
4. **The assay order is derived and asserted, never declared.** There is **no assay-name string
   literal anywhere under `src/`**. Bake derives the order from the handler's filtered
   `experiment_aliases` keys (`prep/reference_sample.py:56-58`), asserts a bijection against the
   requested panel (`:61-65`), asserts `assay_to_id == range(A)` and `control_assay_id == A`
   (`prep/reference_sample.py:48-52`), and records it in `h5.attrs['assays']`. Everything downstream
   reads it back (`dataset.py:135`) and threads it as an argument (`eval.py`, `report.py`). Adding a
   hard-coded list of assay names anywhere is a defect.
5. **`arcsinh` lives in the model, not the loader.** The loader yields **raw counts**; the transform is
   applied inside `V2Encoder.encode` via `_apply_signal_transform` (`encoder.py:629-639`,
   `signal_transform="arcsinh"`, set at `model.py:209`). Never transform in `dataset.py` or `batch.py`
   — that is a silent double-arcsinh.
6. **Counts stay raw non-negative integers as NB targets.** `y_data` is the NB target
   (`train.py:47`). The bake writes int16 raw counts and `_verify`'s F7 check rejects an already
   transformed range (`prep/bake.py:427-433`); `tests/test_bake_gates.py:359` guards the loader side.
7. **The control channel is never masked.** `prepare_masked_batch` masks first, then concatenates the
   control column at index `A` (`batch.py:121-123`), so it is structurally unmaskable. Model inputs
   are therefore `F = num_assays + 1` wide on the input side and `num_assays` wide on the target side.
8. **`obs` / `imp` mean unmasked / masked**, not biological observed/imputed. `observed_map` =
   unmasked-and-available, `masked_map` = cloze (`batch.py:68-77`); the loss adds the two means
   unweighted (`train.py:45-51`).
9. **Covariates stay raw.** The 4 rows are `[log2_depth, assay_id, read_length, run_type]` with **no
   normalization** (parameter-encoding normalization was tested and refuted upstream). Sentinels are
   `MISSING=-1` and `CLOZE=-2` and are kept distinct everywhere
   (`_vendored.py`, consumed at `encoder.py:147-190`). `assay_id == num_assays` is the control slot;
   the embedding table is `Embedding(num_assays + 3)` = real ids + control + MISSING + CLOZE. Do not
   "fix" the `+3` to `+2`.
10. **Scale is a property of the baked file, never a CLI flag.** `num_assays`, `context_bins`,
    `resolution`, `dsf_list`, assay order and train/eval chromosomes come from `h5.attrs`
    (`dataset.py:135-145`) and are passed into `build_real_model` from the dataset (`train.py:232-236`).
    Never add flags for them.
11. **`d_model=0` means auto** `= (num_assays+1) * expansion**n_cnn_layers` — i.e. transformer width
    silently tracks panel size (8 assays → 72). The value is printed at every build
    (`model.py:224-226`). When `num_assays != 8`, set `--d-model` explicitly.
12. **Determinism pattern:** `cudnn.deterministic=True`, `cudnn.benchmark=False`, then
    `torch.manual_seed(seed)` immediately before `build_real_model` with nothing touching the RNG in
    between (`train.py:228-236`). `transformer_layer_drop` must remain 0.0 (`encoder.py:876-879`) — it
    consumes global RNG inside forward.
13. **SLURM GPU spec is fixed:** every `#SBATCH --gres` line in this kit reads
    `--gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1`, and **all three** scripts carry one — `train.sh:13`,
    `gate.sh:13`, and `bake.sh:20`. The bake does no GPU compute; it requests the smallest MIG slice
    only to route through the GPU account (rationale at `bake.sh:7-13`). Never write any other `--gres`
    spec on any line.
14. **Never write a `0`-filled metadata row.** An absent DSF level or unavailable assay is `-1`
    (`prep/bake.py` B6 edit; availability test is `float(xm[0]) != -1.0`, `dataset.py:306,314`). A
    zero-filled row marks every assay available at `log2(depth)=0` with all-zero counts: loss
    descends, metrics are garbage. `dataset._check_available_columns_nonzero` (`dataset.py:169`) and bake `_verify`
    F3 both guard this.


**Never infer what a tracker id means.** `h34`, `h45`, `q19` and friends appear throughout `research/`
and in the prose of the main docs. The tracker is not shipped. `research/README.md` has the decoder
ring; if an id is not in it, report it as unresolvable rather than reconstructing it from context.
Confidently-interpolated provenance is how the retracted `Δη = 0.833` reached three separate records.

---

## 4. COMMON TASKS

Set once (adapt to your machine):

```bash
export KIT=/project/6014832/mforooz/EpiDenoise/candi_kit
export VENV=/project/6014832/mforooz/EpiDenoise/candi_venv
source "$VENV/bin/activate"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1; unset PYTHONPATH || true
export MPLBACKEND=Agg WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### 4.1 Install

```bash
pip install -e "$KIT"                 # train-only deps
pip install -e "$KIT[prepare,report,test]"   # + bake, figures, pytest
```

On Alliance Canada, build the venv from `requirements-fir.txt` first (it documents the exact
`module load python/3.10.13` → `--no-index` → `pip install x-transformers==2.11.23` sequence), then
`pip install -e "$KIT" --no-deps`.

### 4.2 Run the tests

```bash
cd /tmp && python -m pytest "$KIT/tests" -q
```

Run from a directory that is **not** the parent repo — that is part of the proof that the kit is
cwd-independent. Current status in this repo: **63 passed**, CPU only (wall time is machine-dependent:
11 s on a warm login node, ~50 s cold). On a machine without the
historical checkpoints, the 4 `test_historical_checkpoint_loads_strict` cases skip; point
`CANDI_KIT_CKPT_DIR` at a directory of `.ckpt` files to enable them.

### 4.3 Bake a new panel

1. Write a panel JSON (copy `configs/panel.example.json`). Every key is consumed; unknown keys are
   rejected (`prep/panel.py:51-57`). Required: ≥2 `T_` biosamples, `context_bins % 8 == 0`,
   `resolution` a perfect square, `train_chroms` disjoint from `eval_chroms`.
2. Bake:

```bash
python -m candi_kit.prep.bake \
  --root /project/6014832/mforooz/DATA_CANDI_EIC \
  --panel "$KIT/configs/panel.q19.json" \
  --out /scratch/$USER/candi_kit/q19.h5 \
  --fasta /project/6014832/mforooz/EpiDenoise/data/hg38.fa \
  --chrom-sizes /project/6014832/mforooz/EpiDenoise/data/hg38.chrom.sizes \
  --type2-ccre 0 --type2-non 0 --seed 42
```

Optional: `--blacklist BED`, `--ccres BED` (required iff `--type2-ccre`/`--type2-non` > 0),
`--max-tile-per-chrom N` (smoke bakes), `--allow-missing-control`. `--out` has no default; point it at
scratch. Note `--type2-ccre` / `--type2-non` / `--seed` are **bake CLI flags, not panel keys** —
`load_panel` rejects unknown keys (`prep/panel.py:51-57`). On SLURM: `sbatch "$KIT/slurm/bake.sh"` (no
GPU compute, but the script does request the MIG slice — see invariant 13).

Roughly ~0.16 GB and ~1.5 min per biosample at 8 assays / 7,485 windows (`.BUILD_PLAN.md` RISKS).

### 4.4 Train an arm

```bash
python -m candi_kit.train \
  --h5 /scratch/$USER/candi_kit/q19.h5 \
  --out-dir /scratch/$USER/candi_kit/runs \
  --offset on --seed 0 --tag kit_on_s0 \
  --weight-decay 0.0 --dsf-sampling uniform --epochs 25 --batch-size 8 --full-coverage \
  --eval-batch-size 4 --eval-max-batches 0 --eval-budget 50000000 --m3-regions 40 \
  --fg-frac 0.02 --n-boot 1000
```

`--offset off` (alias `--arm offset_off`) trains the other arm. Writes `{out_dir}/{tag}.json` and
`{out_dir}/{tag}.ckpt`. Both arms × 3 seeds as one array: `sbatch "$KIT/slurm/train.sh"`
(~85 min wall per arm on one `1g.10gb` MIG slice; host RSS 15–18 GiB is the binding constraint).

`--compat-q19` pins the q19 architecture knobs (`embed_dim=32, dropout=0.1, n_transformer_layers=2,
feat_per_assay=16, depth_center=25.1, d_model=0, nhead=4`) and refuses an h5 that is not 8-assay /
768-bin (`train.py:220-222`).

### 4.5 Evaluate an existing checkpoint

```bash
python -m candi_kit.eval \
  --h5 /scratch/$USER/candi_kit/q19.h5 \
  --ckpt /scratch/$USER/candi_kit/runs/kit_on_s0.ckpt \
  --out /scratch/$USER/candi_kit/runs/kit_on_s0_rescore.json \
  --offset on --m3-regions 40 --n-boot 1000 --eval-budget 50000000
```

The architecture flags must match those the checkpoint was trained with, or the `strict=True` load
fails loudly (`eval.py:1094`). Add `--include-deprecated` to also emit the audited-and-rejected metric
keys, each carrying its verdict string. Figures + markdown from a results JSON:
`python -m candi_kit.report <results.json> --outdir DIR`.

### 4.6 Add a knob

1. Add the argparse flag in `train.py:270-307` (and/or `eval.py:1062-1085` if it affects scoring).
2. Thread it through `train_and_eval` (`train.py:209-215`) into `build_real_model` / `train` /
   `evaluate` — do not read globals. (The eval-only CLI's flags live at `eval.py:1062-1082`.)
3. Add it to the `config` echo dict (`train.py:249-265`) so every results JSON is self-describing.
4. If it changes model construction, it **must** default to the q19 value, and `--compat-q19` must
   pin it (`train.py:309-313`). Then re-run §5 Gate A.
5. If it is scale (`num_assays`, `context_bins`, `resolution`, `dsf_list`, chroms), it is **not** a
   knob — it belongs in the panel and the h5 attrs (invariant 10).

### 4.7 Change the panel size

Panel size flows: `panel.json` → handler → derived assay order → `h5.attrs` → dataset →
`build_real_model`. To go from 8 assays to N:

1. Edit `assays` / `biosamples` in the panel JSON and re-bake (§4.3). Nothing in `src/` changes.
2. Train with an explicit `--d-model` (invariant 11). Attention inner dim is `nhead * 64` regardless
   of `d_model`, so capacity does not scale with the panel unless `--nhead` also rises.
3. Do **not** use `--compat-q19`, and expect the recorded anchors in §5 to be meaningless for the new
   panel — they are anchors for the 8-assay q19 build only.
4. Re-run `tests/test_model.py::test_scale_axes_independent` and `tests/test_bake_gates.py` — they
   already cover 3/8/16 assays and 384/768/1536 bins.

---

## 5. VALIDATION GATES

### Gate A — the copy proof (CPU, ~5 min, no SLURM, no data)

**What it proves:** that this kit's model layer is a *bit-exact copy* of the original research
implementation, not a reimplementation that merely looks similar. It is the reason the recorded
science numbers may be attached to this code at all. It does **not** prove anything about data,
training quality, or the science claims.

```bash
cd /tmp
python -m candi_kit.compat        # A1 + A3 only (no checkpoints needed) -> "[compat] OK"
python -m candi_kit.compat --ckpt-dir /project/6014832/mforooz/EpiDenoise/sandbox/diagnostics/dual_conditioning_real/results   # + A2
python -m pytest "$KIT/tests/test_compat_q19.py" -q
```

⚠ **`--ckpt-dir` is not skip-safe.** If the directory exists but holds no `main_s0_perassay.ckpt` /
`offoff_s0_perassay.ckpt` / `wd0_on_s0.ckpt` / `wd0_off_s0.ckpt`, `compat.main` raises
`SystemExit("no q19 checkpoints found under ...")` and **exits 1** (`compat.py:144-145`) — verified.
`slurm/gate.sh:35` passes `--ckpt-dir "$CKPT_DIR"` unconditionally, so on any machine without the
historical checkpoints **gate tier 2 fails**. Drop that argument (A1+A3 are the parts that prove the
copy) or run the tiers by hand. The pytest path is skip-safe (`tests/test_compat_q19.py:101-102`).

Exact expected values — **no tolerance, do not relax them**:

| id | check | expected |
|---|---|---|
| A1 | `build_compat_q19(use_offset=True)` under `torch.manual_seed(0)` | **3,103,194** parameters, state_dict sha1 **`fd0e9493ac92a15f`** (`compat.py:27-28`); identical for both arms (`use_offset` holds no parameters) |
| A1b | pinned dependency | `importlib.metadata.version("x-transformers") == "2.11.23"` (`compat.py:29`) — the package has no `__version__`, so this is the only check |
| A3 | frozen synthetic forward, single-threaded CPU | digest **`8c9271f29299b2e0`** (offset on) / **`cd18d437a13c041f`** (offset off); `mu.mean()` 2.604400634765625 / 1.1116695404052734; `eta.std()` 0.15498559176921844 (`tests/test_compat_q19.py:31-33`) |
| A3b | goldens | all 5 output tensors equal to the stored `goldens/*.pt` at **0 ULP** (`compat.py:86-98`) |
| A2 | historical checkpoints (optional) | all four load with `strict=True`, zero missing / zero unexpected keys; skipped when the ckpt dir is absent |
| — | source guards | construction order of `RealDualCondModel` / `DualCondDecoder`, placeholder table fixed at `8 × E`, `film_proj` zero at init (`tests/test_compat_q19.py:147-172`) |

`torch.set_num_threads(1)` is load-bearing for the 0-ULP comparison — CPU float reductions
re-associate with thread count (`compat.py:76-80`).

**If A1/A3 fail, STOP and do not edit the anchors.** The cause is almost always construction-order
drift in `model.py`; walk the banner at `model.py:1-16` item by item.

### Gate B — from-scratch reproduction (SLURM, 2 arms × 3 seeds, ~6 GPU-hours)

```bash
GATE=$(sbatch --parsable "$KIT/slurm/gate.sh")
sbatch --dependency=afterok:$GATE "$KIT/slurm/train.sh"
```

Acceptance bands are set **at the noise floor on purpose** (§7): mean over 3 seeds of the offset-ON
arm within ±0.09 macro `crps_oracle_scaled` of 1.3077, ±0.06 pooled imputation Spearman of 0.6372,
±0.035 ECE of 0.0533. (An M3 band of ±0.05 around 0.2154 appears in `.BUILD_PLAN.md`, but **do not gate
on it**: M3 was never re-scored under h48 — `H48_REPORT.md` §3 — and the kit fixed the `between`-pool
same-region contamination at `eval.py:917-932`, so the kit's M3 is *not* bit-comparable to that anchor.)
The **structural** sign pattern is the sharper test
and should be treated as primary: offset ON → `median_eta_slope ≈ 0` and run_type direction ≈ 0;
offset OFF → `median_eta_slope ≥ 0.7` and a positive run_type direction; offset OFF → macro
`scale_error ≥ 0.4` while `crps_oracle_scaled` stays within 0.09 of the ON arm. Exact structural nulls:
`_metadata_ablation(mode="within_batch")` must read exactly 0, and `n_sentinel_skipped == 0` on a
fully populated panel. Full detail: `.BUILD_PLAN.md` VALIDATION_PLAN Gate B.

### Gate C — generalization smoke (proves the knobs are real)

Bake a different panel (3 assays, 2 `T_` biosamples, `context_bins 384`, other chromosomes) and train
2 epochs. Acceptance is **build correctness, not quality** — do not read a smoke run's CRPS as
evidence of anything. Checks: bijection passes at 3 assays; `assay_ids == [0,1,2]`,
`control_assay_id == 3`; `_verify` passes; the dataset reports `num_assays=3, context_bins=384` and
`build_real_model` consumes them without a flag; `--d-model` overrides the auto width; loss decreases
with no NaN; per-assay entries carry the 3 derived names; the negative controls in §6 all raise.
Full spec: `.BUILD_PLAN.md` VALIDATION_PLAN Gate C.

---

## 6. FAILURE MODES AND THEIR SIGNATURES

| symptom | likely cause | where to look |
|---|---|---|
| Gate A A1 fails: wrong param count or sha1 | construction-order drift; placeholder skipped or scaled with `num_assays`; `film_proj` created as zeros | `model.py:1-16` banner, `model.py:141-157`, `model.py:214-223` |
| Gate A passes A1 but A3 golden mismatch at ~1e-6 | thread count not pinned, or a different CPU/BLAS | `compat.py:76-80`; run with `torch.set_num_threads(1)` / `OMP_NUM_THREADS=1` |
| A2: `load_state_dict` reports unexpected `encoder.transformer_blocks.*` keys | x-transformers version drift | `pyproject.toml` pin; `compat.py:103-106` |
| `ValueError: ... is schema v1; candi_kit requires v2` | h5 baked by the old repo pipeline | re-bake with `candi_kit.prep.bake` (`dataset.py:131-135`) |
| `ValueError: panel/alias mismatch: missing=[...]` | a panel assay is absent from the data root, or the alias filter dropped it | `prep/reference_sample.py:46,61-65`; check the assay name spelling against the directory |
| Loss descends smoothly, metrics are garbage, everything "available" | zero-filled metadata rows (absent DSF level written as 0 rather than -1) | `dataset.py:169`; bake `_verify` F3 (`prep/bake.py:411-415`) |
| `AssertionError: F4 ... not the downsampled data they claim to be` | a `counts_dsf{k}` dataset holds DSF-1 data | `prep/bake.py:417-425`; handler edit E3 (an assay missing from the DSF map now raises instead of silently loading DSF-1) |
| `ValueError: count ... exceeds int16` | counts wider than int16 in the source NPZ | handler edit E4; re-bake with a wider dtype |
| `ValueError: unparseable run_type ...` | a metadata value outside the expected vocabulary (previously inherited the previous assay's value silently) | handler edit E5 |
| `ValueError: metadata must be 4 rows ...` / `assay_id ... exceeds table bound` / `run_type ... exceeds table bound` | a hand-built or externally-supplied metadata tensor | `encoder.py:142-190`; these are deliberate — previously they aliased onto the MISSING/CLOZE slots |
| `ValueError: DNA length G != context L x resolution` | panel `resolution`/`context_bins` mismatch between h5 and model | `encoder.py:920-925` |
| `ValueError: Availability/supervision mismatch` | masking config other than assay-only (`p_full_loci`/`p_chunks` nonzero) under `missing_data_mode='mask_token'` | `batch.py:80-85`, `encoder.py:890-895` |
| `ValueError: no training windows: regime=...` | `type2_loci` regime on an h5 baked with `--type2-ccre 0 --type2-non 0` | `dataset.py:159-162` |
| `[data] WARNING: single-DSF ladder` | `dsf_list` has one level; per-assay independent DSF sampling is inert and the depth-steering signal is absent | `dataset.py:163-165`, `prep/panel.py:44-48` |
| `[train] WARNING: --mask-fraction is INERT` | `--mask-fraction` set while `--p-full-assay 1.0`; `DataMasker._mask_full_assay` never reads it | `train.py:93-95`, `_vendored.py:114-143` |
| Training runs but imputation loss is ~0 | biosamples with ≤1 available assay are skipped by the masker, so no cloze is produced | `_vendored.py:133-137`; the panel gate requires ≥2 `T_` biosamples but not ≥2 assays each |
| Host OOM at ~N×1.6 GB | the shared RAM buffer was not shared across the per-biosample datasets | `train.py:106-113` (`ds._ram_buf = shared`) |
| `raise ValueError("missing_data_mode='mask_stem' not shipped ...")` / `transformer_type='production_dual' not shipped` / `norm=... not shipped` | a config branch that depends on parent-repo modules | intentional; use `mask_token` / `xtransformers` / `layer` |
| Run-to-run numbers differ on the same seed | `torch.use_deterministic_algorithms` added, `cudnn.benchmark` re-enabled, or something touched the RNG between `manual_seed` and `build_real_model` | `train.py:228-236` |
| `MPLBACKEND` / display errors from `report.py` | headless node without `MPLBACKEND=Agg` | §4 env block |

---

## 7. HOW TO READ THE SCIENCE

Authority order: `sandbox/diagnostics/dual_conditioning_real/H48_REPORT.md` (2026-07-24/25,
post-adversarial-verification) > `research/H48_SCORECARD.md` > `METADATA_AUDIT.md` > everything else.
Where an older document disagrees, H48_REPORT wins.

### 7.1 The recorded four-arm results (single seed 0, full chr21 coverage)

| arm | macro CRPS | oracle-scaled (capability) | scale_error | macro Sp | pooled imp Sp | ECE | beats honest marginal | M3 ratio |
|---|---|---|---|---|---|---|---|---|
| `wd0_on_s0` (offset ON, wd=0) | 1.3413 | 1.3077 | 0.0336 | 0.5653 | 0.6372 | 0.0533 | 7/8 | 0.2154 |
| `main_s0_perassay` (offset ON, wd=1e-4) | 1.4950 | 1.4210 | 0.0740 | 0.5051 | 0.5327 | 0.0615 | 5/8 | 0.2443 |
| `offoff_s0_perassay` (offset OFF) | 1.9023 | 1.3871 | 0.5152 | 0.4647 | 0.4007 | 0.0968 | 2/8 | 0.1974 |
| `wd0_off_s0` (offset OFF, wd=0) | 2.0561 | 1.4026 | 0.6535 | 0.4641 | 0.3800 | 0.0782 | 1/8 | 0.2185 |

M1 columns are from `research/H48_SCORECARD.md` §M1. The **M3 ratio column is carried forward from a
pre-fix run** (`.BUILD_PLAN.md` anchor table): `H48_REPORT.md` §3 records that M3 was never re-scored,
so it still uses the old permuted labels and a `between` pool that admits same-region pairs. The kit
fixes that pool (`eval.py:917-932`), so **your M3 will not be comparable to this column** — quote it
only with that caveat, or not at all.

Steering, sentinel-free and target-clustered (H48_SCORECARD §M2, H48_REPORT F2):

| arm | total told-depth slope | assay real→real max\|Δη\| | run_type clustered CI (n=12) | supports direction | sign-test p |
|---|---|---|---|---|---|
| `wd0_on_s0` | 1.0000 | **0.0023** | [−0.00066, +0.000087] | no | 1.000 |
| `main_s0_perassay` | 1.0000 | 0.0000 | [0, 0] | no | n/a (all ties) |
| `offoff_s0_perassay` | 0.8869 | **4.1772** | [+0.1179, +2.1804] | YES | 0.039 |
| `wd0_off_s0` | 1.0325 | **9.7144** | [−0.2326, +9.4084] | no | 0.039 |

### 7.2 Rules an agent must follow when quoting any of this

1. **Never present the four-arm ordering as established.** H48_REPORT F1 says so explicitly. Under the
   oracle per-assay scale the spread compresses 0.7148 → 0.1133 (84%), and only
   "`wd0_on` is best on capability" survives inference (`offoff − wd0_on = +0.093`, 95% CI
   [+0.004, +0.217]). The other three are statistically indistinguishable; the target-level sign test
   `main` vs `offoff` is 7+/5−, p = 0.77, and P(`main` worst of four) = 0.54. The published 4-dp
   ordering is the modal bootstrap ordering at only 45% of replicates.
2. **Quote the noise floor with every number.** Effective replication is **12 held-out targets / 5
   biosample pairs / 4 cell types**, with `(T_RWPE2, B_RWPE2)` supplying 7 of the 12. Target-clustered
   bootstrap noise floor on macro CRPS is **~0.09**; per-comparison uncertainty ±0.13. A single **seed**
   change moves pooled imputation CRPS by 0.1195, Spearman by 0.0562, ECE by 0.0354, M3 ratio by
   0.0479 — comparable to or larger than several between-arm gaps. Sign-test resolution at n=12 is
   quantized: 10/12 → p=0.039, 11/12 → 0.0063, 12/12 → 0.00049.
3. **The offset head is a real Pareto, not a solved problem.** Offset ON gives the best imputation
   (macro CRPS 1.3413, macro Spearman 0.5653, ECE 0.0533, beating an honest per-assay marginal on 7/8
   assays) but its covariate steering is **functionally null**: sentinel-free real→real assay ablation
   **0.0023**, 43× below its own pre-registered 0.10 bar, and a run_type clustered CI of
   [−0.00066, +0.000087]. Its depth response is **arithmetically exact, not learned**: told-depth
   slope 1.0000 because `log2_mu = (depth − depth_center) + eta` (`model.py:178-179`) is a closed-form
   thinning identity — NB is closed under thinning with `n` preserved. Offset OFF gives genuine
   learned steering (assay real→real 4.1772; run_type CI [+0.118, +2.180], sign-test p=0.039) at +42%
   macro CRPS (1.9023) and 2/8 beats-marginal. **State both sides whenever you state either.**
4. **Read the clamp tail next to any slope.** `wd0_on`'s 1.0000 is read partly through the saturating
   `log2_mu` clamp on 16.9% of targets (p90 clamp fraction 0.475). Always emit
   `frac_targets_any_clamp` / `p90` / `max` alongside a slope.
5. **Known bounds — do not design around them without re-selecting the panel.**
   * `run_type` is **analytically unidentifiable** on the shipped 8-assay panel:
     `H(run_type | assay_id, read_length) = 0.000 bits` (n=26 `T_` records). The full EIC panel retains
     0.551 bits. A run_type steering demonstration is **impossible** here; it needs a re-selected
     biosample panel, not an architecture change.
   * DSF only **down**-samples, so the upward-depth regime is never trained, and **7 of 12** eval
     targets sit above their per-assay training depth ceiling (worst +1.43 log2). Depth steering on
     those targets is extrapolation into an untrained direction.
   * In this dataset `assay_id ≡ slot index`, and the decoder trunk emits a dedicated channel block per
     slot, so assay identity is carried structurally. Any assay-steering verdict bounds the **prompt
     pathway**, not assay-awareness.
   * Eval runs `dsf_sampling='off'` with `apply_mask=False`, so `x_data == y_data` for available
     assays and "denoising" is autoencoding. Nothing in the suite measures denoising as distinct from
     autoencoding.
   * `crps_oracle_scaled` is an **in-sample upper bound** — `c*` is fitted on the same 12 targets it
     scores — and `scale_error` can be slightly negative (−0.0008 observed).
6. **Capability of the model, not of this kit's outputs:** the model is counts-only NB. Do not claim
   peak calling, peak precision/recall/AUROC, or p-value tracks.
7. **Metric hygiene.** Only the default-emitted metrics may back a claim. Keys behind
   `--include-deprecated` ship with a verdict string in `eval.py:49` and must never be quoted as
   evidence: the `read_length` flip arm (7/12 flips are out of training support), the shuffled-depth
   `null` (a mathematical no-op — one `[4,F]` tensor broadcast over the batch, so the permutation is
   bitwise identity), `frac_min_at_true` (0.25 is not chance; it is the deterministic value of
   argmin-always-at-told=1, and a perfect model caps at ~0.73), all position-level `_bootstrap_ci`
   outputs (~24× too narrow), and `median_eta_slope` / `offset_independent` (decided by the sign of
   ~1e-17 float noise under offset-ON). The condition-recoverability probe (S23) is **withdrawn** and
   is not shipped in any mode: its ordering is inverted against every other instrument. `pearson_log1p`
   and `spearman_raw` are computed in different spaces and must never be quoted as a pair.
8. **Float resolution.** GPU-vs-CPU on identical weights reproduces macro CRPS to 4 decimals; per-assay
   values move 2e-6…3.3e-4, and `nb_crps` itself is accurate to ~2.4e-5 relative. Anything at 1e-7 or
   below in `eta`/`mu` is float noise.
9. **Per-assay labels.** Earlier documents in the research repo label per-assay results with a
   *permuted* assay list. The corrected order is the one derived from the handler aliases and stored in
   `h5.attrs['assays']` (38/38 vs 5/38 on a metadata join). Under corrected labels, the largest CRPS
   contributor is **DNase-seq** (not ATAC-seq) and the collapse outlier is **H3K9me3** (not H3K27ac).
10. **Hybrid recovery was not demonstrated.** The upstream hypothesis that a hybrid (annealed /
    α-attenuated offset / learned scale head) recovers both magnitude and steering is recorded
    **REFUTED** in the project's research tracker (`h45`; see the decoder ring in
    `research/README.md`) — but refuted on its premises, with **no hybrid arm ever trained**, and the
    entry carries a 2026-07-28 flag that its refutation basis is itself under review. Do not present a hybrid
    as either available or ruled out by experiment.

### 7.3 Provenance of the numbers in §7.1

They are **single-seed** scores of four historical checkpoints trained in the research repo and
re-scored on CPU with the corrected instruments (608 eval units, 1215 target-records, 12 held-out
targets). They are anchors for validating that this kit reproduces the same computation — they are not
results this kit has produced, and no weights backing them ship here.

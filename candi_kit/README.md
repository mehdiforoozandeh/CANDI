# candi_kit — CANDI q19 dual-conditioning recipe

A self-contained, vendored copy of one CANDI training recipe. **It ships code, configuration and
validation gates — no trained weights.** You bake your own HDF5 from ENCODE-style data and train from
scratch.

---

## 1. What CANDI is, and what this directory is

**CANDI** (Confidence-Aware Neural Denoising Imputer) is a self-supervised model for epigenome
**imputation** (predict an assay that was never run in a biosample) and **denoising** (clean up an assay
that was run shallowly). It consumes **raw sequencing read counts** plus the experimental covariates that
describe how those counts were produced, rather than pre-normalized signal tracks. Per assay and per
genomic bin it emits a **probability distribution**, not a point estimate, so predictions carry
calibrated uncertainty.

The design consequence: batch/exposure effects are **conditioned on** instead of normalized away. The
covariates — sequencing depth, assay identity, read length, run type — enter the model as inputs, so at
prediction time you can *ask for* a track as it would look under a different exposure, instead of
post-hoc rescaling one. Full CANDI additionally has Gaussian p-value and Bernoulli
peak heads; **this kit is counts-only, Negative Binomial only** (§6).

**This directory** is the `q19` "dual metadata conditioning" recipe extracted from the research repo:
~3.1 M parameters, a convolutional + transformer encoder over counts and DNA, a deconvolutional decoder,
and one NB head. It is vendored — nothing outside `candi_kit/` is imported, and the third-party
import surface is 5 hard packages instead of 43 (`.BUILD_PLAN.md` § DEPENDENCIES). It is a **recipe with
recorded anchors and honest error bars, not a finished product**.

## 2. What "dual conditioning" means

Two independent metadata pathways, two separate (untied) embedders
(`ARCHITECTURE_HANDOFF.md:23-24`, verified at runtime):

| side | tensor | question it answers | where it acts |
|---|---|---|---|
| **input-side conditioning** | `x_meta` `[B,4,A+1]` | "what *were* the tracks I am reading?" | FiLM after **every** one of the 3 encoder conv layers |
| **output-side steering** | `y_meta` `[B,4,A]` | "what track should I *produce*?" | **one** FiLM after the whole decoder trunk, plus an arithmetic depth offset |

Both tensors carry the same 4 rows, in this order:
`[log2(sequencing_depth), assay_id, read_length, run_type]`, float32, no normalization. Sentinels:
`MISSING = -1`, `CLOZE = -2`. `y_meta` has no control column and never carries `CLOZE`.

Concrete example. You hold a shallow H3K27ac track from cell type X and want H3K4me3 in the same cell
type. **Input side:** you tell the encoder that the track it is reading is H3K27ac at log2-depth 23.1,
read length 36, single-ended — so it can interpret sparse counts as *shallow*, not as *absent signal*.
**Output side:** you set `y_meta` for the H3K4me3 slot to log2-depth 26, read length 100, paired-end —
and the model should return the distribution that H3K4me3 *would* have had if it had been sequenced that
deeply, with that protocol. Changing only that prompt and getting a correspondingly different prediction
is what "steering" means, and it is the property this kit was built to measure. The whole point of
§3 is that the two shipped arms trade steering against accuracy.

## 3. The headline tradeoff — read `TRADEOFF.md` before quoting anything

`--offset on` adds a closed-form depth term to the NB mean:
`log2_mu = (told_depth − depth_center) + eta` (`ARCHITECTURE_HANDOFF.md:238-243`). It is the arm with the
best imputation and **functionally null covariate steering**. `--offset off` uses `log2_mu = eta` and
steers, at a 42 % macro-CRPS cost. Both are first-class; neither dominates. **This is a real Pareto, not
a solved problem.** A hybrid was hypothesized (internal node h45) and recorded **refuted, 0/4
verifiables met** — but refuted *on its premises*: **no hybrid arm was ever trained**, and the node
carries a 2026-07-28 flag that its own refutation basis is under review. Treat the hybrid as **neither
available nor ruled out by experiment** (`TRADEOFF.md` § 4).

Recorded seed-0 anchors from the original q19 runs (M1/M2 columns from `research/H48_SCORECARD.md`; the
sentinel-free real→real assay column from `H48_REPORT.md` § F2; CRPS and Spearman are macro over
8 assays; lower CRPS better):

| checkpoint | offset | wd | macro CRPS | oracle-scaled (capability) | macro Sp | ECE | beats honest marginal | assay steering, sentinel-free max\|Δη\| | run_type clustered CI (n=12) |
|---|---|---|---|---|---|---|---|---|---|
| `wd0_on_s0` | ON | 0 | **1.3413** | **1.3077** | **0.5653** | **0.0533** | **7/8** | 0.0023 | [−0.0007, +0.0001] |
| `main_s0_perassay` | ON | 1e-4 | 1.4950 | 1.4210 | 0.5051 | 0.0615 | 5/8 | 0.0000 | [+0.0000, +0.0000] |
| `offoff_s0_perassay` | OFF | 1e-4 | 1.9023 | 1.3871 | 0.4647 | 0.0968 | 2/8 | **4.1772** | **[+0.118, +2.180]**, p = 0.039 |
| `wd0_off_s0` | OFF | 0 | 2.0561 | 1.4026 | 0.4641 | 0.0782 | 1/8 | **9.7144** | [−0.233, +9.408] |

Four things you must carry with those numbers:

1. **Noise floor.** Effective replication is **12 held-out targets / 5 biosample pairs / 4 cell types**,
   with `(T_RWPE2, B_RWPE2)` supplying 7 of the 12. Target-clustered bootstrap noise floor on
   oracle-scaled macro CRPS is **~0.09**; per-comparison uncertainty **±0.13**. A single **seed** change
   moves pooled imputation CRPS by **0.1195** and Spearman by **0.0562**. Position-level CIs (n ≈ 893 k)
   in the result JSONs are fictions.
2. **The 4-arm ordering is NOT established.** Under the oracle per-assay scale the four arms compress
   into a **0.113** band (84 % compression from 0.7148 raw) and only "`wd0_on` is best on capability"
   survives inference (`offoff − wd0_on` = +0.093, CI [+0.004, +0.217]); the other three are pairwise
   indistinguishable. `H48_REPORT.md:47-72` says the reordering is not established — do not present it as
   a ranking.
3. **Offset-ON's depth response is arithmetic, not learned.** Told-depth slope is exactly **1.0000**
   because `log2_mu = (depth − depth_center) + eta` is the closed-form NB thinning identity. That is
   arithmetic. Its *learned* conditioning is null: sentinel-free real→real assay ablation **0.0023**,
   **43× below** the pre-registered 0.10 bar; run_type CI [−0.00066, +0.000087]. An earlier 0.833 figure
   was a MISSING-sentinel artifact (`H48_REPORT.md:80-123`).
4. **Offset-OFF's raw deficit is mostly calibration, not capability** (`scale_error` 0.5152 / 0.6535 vs
   0.0336) — and its depth-counterfactual failure is a *level* failure: correcting one per-target
   constant flips all four arms to passing (`H48_REPORT.md:125-144`).

## 4. Quickstart (Compute Canada / Fir)

You need **two directories**: this kit, and an ENCODE-style data directory. Plus a reference genome,
which is public and which you download once — see step 0b.

### 4.0 Environment

```bash
module load python/3.10.13
virtualenv --no-download ~/candi_venv && source ~/candi_venv/bin/activate
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1; unset PYTHONPATH || true
export MPLBACKEND=Agg WANDB_MODE=disabled

KIT=/path/to/candi_kit                       # <- the kit directory
pip install --no-index -r $KIT/requirements-fir.txt   # CVMFS wheelhouse
pip install -r $KIT/requirements-pypi.txt             # PyPI (login node; needs internet)
pip install --no-deps $KIT                            # the kit itself
```

**Use a non-editable install (`pip install --no-deps $KIT`), not `pip install -e`.** An editable
install writes `src/*.egg-info` into the kit's own tree, which fails when the kit is a shared or
read-only copy you do not own. Non-editable is correct here and keeps the golden tensors, which
`pyproject.toml` ships as package data.

Off-cluster: any Python 3.10 venv; portable pins are in `pyproject.toml`. `x-transformers==2.11.23` is
pinned **exactly** — a minor bump changes state_dict key names and init order, and breaks `compat`.

### 4.0b Reference genome (one-time download)

The genome is **not** in this kit and **not** in the data directory. Get it from UCSC —
**[`DATA.md` §5.1](DATA.md) has the exact `wget` block.** In short, for the default chr19-train /
chr21-eval panels you only need two chromosomes (~107 MB), not all of hg38 (3.27 GB):

```bash
mkdir -p ~/side && cd ~/side
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.chrom.sizes
for c in chr19 chr21; do
  wget -qO- https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/$c.fa.gz | zcat
done > hg38_subset.fa
grep -E '^(chr19|chr21)\s' hg38.chrom.sizes > hg38_subset.chrom.sizes
```

### 4.1 Verify the install (CPU, no data, ~1 min)

```bash
pytest $KIT/tests -q             # expect: 63 passed
python -m candi_kit.compat       # expect: params=3103194 sha1=fd0e9493ac92a15f ... [compat] OK
```

`compat` reproduces a byte-level claim from this README in ~20 s, so you can confirm the install is
sound before spending anything. It needs no data and no GPU.

### 4.2 Smallest run that actually works — start here

Before the full panel, prove the whole pipeline end to end in **~15 minutes**. This uses
`configs/panel.example.json`: 3 assays × 5 biosamples, two held-out imputation targets.

```bash
DATA=/path/to/DATA_CANDI_EIC
SIDE=~/side

python -m candi_kit.prep.bake \
  --root $DATA --panel $KIT/configs/panel.example.json \
  --out /scratch/$USER/candi_kit/example.h5 \
  --fasta $SIDE/hg38_subset.fa --chrom-sizes $SIDE/hg38_subset.chrom.sizes \
  --type2-ccre 0 --type2-non 0 --allow-missing-control --seed 42

python -m candi_kit.train \
  --h5 /scratch/$USER/candi_kit/example.h5 --out-dir /scratch/$USER/candi_kit/runs_example \
  --offset on --seed 0 --tag example --epochs 3 --batch-size 8 --full-coverage \
  --eval-batch-size 4 --eval-max-batches 0 --m3-regions 10 --n-boot 100

python -m candi_kit.report /scratch/$USER/candi_kit/runs_example/example.json
```

Or as one SLURM job (the three commands above, with the `#SBATCH` header already set):

```bash
mkdir -p slurm-logs
VENV=~/candi_venv KIT=$KIT ROOT=$DATA SIDE=$SIDE sbatch $KIT/slurm/example.sh
```

**What a good result looks like.** Observed on this panel at 3 epochs, offset ON (~7 min on one MIG
slice). Yours will differ — this is the shape, not a target:

| field | observed | what it tells you |
|---|---|---|
| `imp_beats_marginal_n` | **2/2** | beats a per-assay marginal baseline on every held-out assay — the honest sanity check |
| `imp_macro_spearman_raw` | ~0.46 | rank correlation on held-out assays |
| `imp.ece` | ~0.007 | calibration; well under the ~0.05 bar |
| `M2.depth.median_total_slope` | **1.0000** | exact, because offset ON makes this closed-form arithmetic |
| `M3.ratio` | ~0.13 | **< 0.30 ⇒ `invariance_ok`** — the encoder grouped by biology, not by sequencing depth |
| `M2.ablation.{assay_id,run_type,depth}` | real values | the steering instruments have something to measure |

`M3 ratio < 0.30` and `M2 total_slope ≈ 1.000` are the two that should hold on any healthy offset-ON
run. `imp_macro_crps` is **only comparable to another run on the same bake** — never to the 8-assay
numbers in `TRADEOFF.md`. See `TRADEOFF.md` §2b for what every field means.

> **`M2.ablation.read_length` comes back `nan` on this panel, and that is correct.** An ablation needs
> the covariate to *vary* across held-out targets; here every target shares a read length, so there is
> nothing to ablate and `n_targets` is 0. This is an identifiability property of the panel, not a broken
> metric — the same class of limit as bound `B1` for `run_type` on the reference panel
> (`research/METADATA_AUDIT.md`). If you need a covariate's steering measured, choose biosamples that
> differ in it. A small panel will usually leave at least one ablation undefined.

**Outputs are overwritten, not versioned.** Re-running with the same `--tag` replaces the previous
`.ckpt`, `.json` and report in place. Change `--tag` (or `--out-dir`) per run if you want to keep both.

Two flags worth understanding before you change them:

- **`--allow-missing-control`** — some biosamples carry only accessibility assays (ATAC/DNase), which
  need no ChIP input control, so they have no control track at all. The bake stops on that by default
  because for a ChIP panel a missing control *is* a silent capability loss. Here it is the real shape of
  the data. See the comment block in `slurm/bake.sh`.
- **`--eval-max-batches 0`** means *evaluate everything*. A small non-zero value takes a **prefix** of
  the eval chromosome, which on chr21 is telomeric and all-zero — you will get
  `Spearman undefined -- target is constant` and `nan` metrics. That is a property of the eval slice, not
  a broken model. Use `0` unless you know why you want otherwise.

### 4.3 The reference panel (8 assays, the configuration the recorded numbers come from)

```bash
python -m candi_kit.prep.bake \
  --root $DATA --panel $KIT/configs/panel.q19.json \
  --out /scratch/$USER/candi_kit/q19.h5 \
  --fasta $SIDE/hg38_subset.fa --chrom-sizes $SIDE/hg38_subset.chrom.sizes \
  --type2-ccre 0 --type2-non 0 --allow-missing-control --seed 42
# ~5 min, ~1.35 GB. `panel.q19.json` includes B_DND-41, which has no control track --
# hence --allow-missing-control. Without it the bake aborts AFTER doing all the work.

sbatch $KIT/slurm/gate.sh        # pytest -> compat -> CPU smoke -> GPU smoke, ~5 min

H5=/scratch/$USER/candi_kit/q19.h5 OUT=/scratch/$USER/candi_kit/runs \
  sbatch $KIT/slurm/train.sh     # 2 arms x 3 seeds, ~85 min/arm, ~6 GPU-hours
```

> **To reproduce the recorded anchors exactly** you also need the 2,000 type2 loci
> (`--type2-ccre 1000 --type2-non 1000 --ccres <GRCh38-cCREs.bed>`), which requires a **whole-genome**
> FASTA because those loci are sampled from every chromosome except your train/eval ones. With
> `--type2-* 0` the pipeline is identical and the run is entirely valid — it is simply not
> numerically comparable to `TRADEOFF.md`'s table. `DATA.md` §5.1 has the cCRE download.

SLURM logs go to `./slurm-logs/` **relative to wherever you run `sbatch` from**, so `mkdir -p slurm-logs`
first, or pass your own `--output`/`--error`. Every script takes `KIT`, `VENV`, `ROOT`, `SIDE`, `PANEL`,
`H5` and `OUT` as environment overrides.

Single run, no SLURM array (this is exactly what `slurm/train.sh` executes per task):

```bash
python -m candi_kit.train \
  --h5 /scratch/$USER/candi_kit/q19.h5 --out-dir /scratch/$USER/candi_kit/runs \
  --offset on --seed 0 --tag kit_on_s0 --weight-decay 0.0 \
  --dsf-sampling uniform --epochs 25 --batch-size 8 --full-coverage \
  --eval-batch-size 4 --eval-max-batches 0 --eval-budget 50000000 --m3-regions 40 \
  --fg-frac 0.02 --n-boot 1000
```

Writes `<out-dir>/<tag>.ckpt` and `<out-dir>/<tag>.json` (all metrics + the resolved config). Re-score an
existing checkpoint without retraining:

```bash
python -m candi_kit.eval --h5 /scratch/$USER/candi_kit/q19.h5 \
  --ckpt /scratch/$USER/candi_kit/runs/kit_on_s0.ckpt \
  --out /scratch/$USER/candi_kit/runs/kit_on_s0_rescore.json \
  --offset on --max-batches 0 --n-boot 1000 --m3-regions 40
```

Optional markdown/plot summary (needs the `[report]` extra and `MPLBACKEND=Agg`):
`python -m candi_kit.report <results.json> --outdir <dir>`.

**Wall time and resources**, from 10 recorded arms: ~85 min per arm (~72 min for 25 epochs × ~1,910
batches at batch size 8 ≈ 47,750 updates, plus ~13 min for the full chr21 eval). All 10 fit the 9.75 GiB
usable `nvidia_h100_80gb_hbm3_1g.10gb` MIG slice at 3.10 M params / `d_model` 72 / context 768. The
binding constraint is **host RSS, 15–18 GiB**, because the HDF5 is slurped into one shared buffer.
On this cluster every GPU job must use `--gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1`.

**Scale is declared once, in the bake panel.** `num_assays`, `context_bins`, `resolution`, `dsf_list`,
assay order and train/eval chromosomes are written into the HDF5 attrs and read back by the trainer —
they are **never** training flags, so you cannot silently train an 8-assay model on a 12-assay file. If
`num_assays ≠ 8` you must set `--d-model` explicitly, or transformer width tracks panel size
(auto = `(num_assays+1) · 2^3`); the kit prints the auto value at every build.

## 5. What is in the box

| path | what it is |
|---|---|
| `src/candi_kit/_vendored.py` | `DataMasker`, `MISSING`/`CLOZE`, `exponential_linspace_int` — the only repo-root code copied in |
| `src/candi_kit/config.py` | `EncoderConfig` dataclass (encoder knobs; `d_model=0` means auto) |
| `src/candi_kit/encoder.py` | `V2Encoder`, `MetadataEmbedding`, conv towers, mask-token injector, input-side FiLM |
| `src/candi_kit/decoder.py` | `DeconvBlock`/`DeconvTower`/`DecoderTrunk` — the metadata-blind deconv trunk |
| `src/candi_kit/model.py` | `RealDualCondModel`, `DualCondDecoder`, `build_real_model`, NB helpers. **Construction order is frozen** (it fixes the RNG stream) |
| `src/candi_kit/batch.py` | `make_masker`, `prepare_masked_batch`; control column appended at index `A` *after* masking, so it is structurally unmaskable |
| `src/candi_kit/dataset.py` | `CandiKitH5Dataset` (iterable) + `h5_depth_center`; reads all scale from HDF5 attrs |
| `src/candi_kit/train.py` | training driver + CLI (`python -m candi_kit.train`) |
| `src/candi_kit/eval.py` | `evaluate()` + CLI: M1 magnitude/calibration, M2 steering, M3 latent, S14 depth counterfactual |
| `src/candi_kit/metrics.py` | numeric primitives: `nb_crps`, `nb_quantile`, PIT/ECE, spearman/pearson/r2 |
| `src/candi_kit/compat.py` | bit-exact reconstruction gate: seed-0 build = 3,103,194 params, state_dict sha1 `fd0e9493ac92a15f` |
| `src/candi_kit/report.py` | markdown/plot summary from a results JSON alone |
| `src/candi_kit/goldens/*.pt` | forward-**output** tensors (~248 KB each) used by the compat gate. **Not weights** |
| `src/candi_kit/prep/{panel,paths}.py` | `Panel` + `SideFiles` — the only two config objects |
| `src/candi_kit/prep/handler.py` | `CANDIDataHandler`: ENCODE-directory reader, sole definition of assay-id ordering |
| `src/candi_kit/prep/reference_sample.py` | handler construction + the panel↔alias **bijection assert** |
| `src/candi_kit/prep/bake.py` | directory → HDF5 bake + post-bake verification gate |
| `configs/panel.q19.json` | the exact 8-assay / 10-biosample q19 panel |
| `configs/panel.example.json` | 3-assay template to copy |
| `slurm/{bake,gate,train}.sh` | ready-to-`sbatch` jobs (directory is `slurm/`, not `jobs/`, because an unanchored `jobs/` .gitignore rule would swallow them) |
| `tests/` | compat, model scale-invariance, metric primitives, bake gates |
| `.BUILD_PLAN.md` | provenance: every file's origin, every edit, the validation plan and the risk register |

## 6. What this kit does not do

- **No checkpoints.** Nothing pretrained ships here. You train from scratch. (Historical checkpoints
  exist in the research repo and were used only as an internal correctness gate. The pytest tier skips
  those four cases cleanly when the checkpoints are absent — `tests/test_compat_q19.py:101-102` — but
  `slurm/gate.sh` tier 2 runs `candi_kit.compat` WITHOUT `--ckpt-dir` by default, so it verifies
  the parameter count, state_dict hash and golden forward outputs — the parts that actually prove the
  port is faithful — and skips the historical-checkpoint cases you do not have. Set `CKPT_DIR=...`
  only if you were given those files.)
- **Counts only.** One Negative Binomial head. **No Gaussian p-value head, no Bernoulli peak head** — so
  no peak precision/recall/AUROC and no p-value track. `pval` and `peaks` are baked into the HDF5 and
  carried through the batch dict but never supervised. See `EXTENSION_HOOKS.md` to add them back.
- **`run_type` steering cannot be demonstrated on the shipped panel.** On the 8-assay q19 panel
  `H(run_type | assay_id, read_length) = 0.000 bits` — it is a deterministic function of the other two
  covariates, so no architecture can make it identifiable. (The full EIC panel retains 0.551 bits.) A
  `run_type` demo needs a **re-selected biosample panel**, not a model change.
- **Upward depth extrapolation is untrained.** Downsampling (DSF) only ever removes reads, so training
  support is `[natural_min − 3, natural_max]`; **7/12 eval targets sit above their per-assay training
  depth ceiling** (worst +1.43 log2). Depth steering on those is extrapolation in the untrained
  direction.
- **Covariate attribution is not identified.** Depth, read length and run type are mutually collinear at
  n=38, so "covariate X carries Y % of the signal" is unsupportable on this data.
- **One configuration has recorded anchors**: 8 of the 35 ENCODE assays, train on chr19, evaluate on
  chr21, 10 biosamples. Other panels/chromosomes are supported by the code (scale is data-derived and
  gated) but have **no reference numbers**.
- **Eval measures autoencoding, not denoising**, as configured: eval runs with `dsf_sampling='off'` and
  masking off, so `x_data == y_data` for available assays.
- **No fast masking regression test.** The masking-invariant tests depend on a real-data fixture and are
  not ported; the end-to-end path is covered only by a full bake+train smoke.
- **Not measured:** peak GPU memory at panel sizes beyond 8 assays / context 768. Scaling to 35 assays or
  context 1536 has no measured memory envelope.

## 7. Where to go next

- [`research/`](research/) — the primary sources behind every number quoted here, frozen copies from the
  research repo. Start at [`research/README.md`](research/README.md): it says which document wins when
  two disagree, flags which numbers were later retracted, and carries the **decoder ring** for the
  `h<N>`/`q<N>` tracker ids the research documents use. Those documents were written for a colleague
  already inside the project — the decoder ring is what makes them readable from outside it.

| doc | read it for |
|---|---|
| `RECIPE.md` | architecture, the training loop, which hyperparameters are load-bearing vs free |
| `DATA.md` | expected ENCODE-style directory layout, HDF5 v2 schema, side files, disk/wall scaling |
| `TRADEOFF.md` | the offset ON/OFF arms in full, with every caveat attached to every number |
| `EXTENSION_HOOKS.md` | adding the Gaussian p-value and Bernoulli peak heads back |
| `AGENTS.md` | orientation for an LLM coding agent working in this tree |
| `.BUILD_PLAN.md` | provenance, knob inventory, validation gates A/B/C, risk register |

**The evidence behind every number, shipped in [`research/`](research/).** Everything quoted in §3
traces to four internal research documents. They are included here as frozen copies so you can audit any
claim yourself, each with a reader preamble explaining what it is and how to read it from outside the
project. Start at [`research/README.md`](research/README.md) — it also carries the **decoder ring** that
translates the `h<N>`/`q<N>` tracker ids those documents use.

| file | what it is |
|---|---|
| [`research/H48_REPORT.md`](research/H48_REPORT.md) | **the authority** — post-adversarial-verification re-score; wins over any older document |
| [`research/H48_SCORECARD.md`](research/H48_SCORECARD.md) | the 4-arm results table §3 is drawn from |
| [`research/METADATA_AUDIT.md`](research/METADATA_AUDIT.md) | `S1`–`S27` defect register and `B1`–`B9` bounds — what this data *cannot* answer |
| [`research/ARCHITECTURE_HANDOFF.md`](research/ARCHITECTURE_HANDOFF.md) | runtime-verified map of how metadata reaches the model |

# Covariate Decodability Probes — Plan

## Scientific question
Is there a **run_type** and/or **read_length** fingerprint recoverable from the *shape* of a
read-count track (and DNA sequence), **independent of sequencing depth**, on a **per-assay** basis?
This is the precondition for CANDI's metadata-conditioning / supertrack goal: if a covariate leaves a
recoverable fingerprint, the encoder can factor it out and the decoder can steer it; if not, the
counterfactual is moot.

Mechanism hypothesis: any fingerprint comes largely from **mappability**, which the DNA branch
supplies. The signal-only vs signal+DNA *lift* tests this directly.

## Data (MERGED — single-experiment tracks, not pooled)
- Root: `/home/mforooz/projects/def-maxwl/mforooz/DATA_CANDI_MERGED`
- Layout: `<biosample>/<assay>/signal_DSF1_res25/chr{19,21}.npz`
  - npz key `arr_0`: `int64`, per-25bp-bin **raw read counts**, full depth (DSF1). chr19 ≈ 2,344,705 bins.
  - `signal_DSF1_res25/metadata.json`: `{depth, coverage, dsf}` — `depth` = genome-wide library depth.
  - `<biosample>/<assay>/file_metadata.json`: `run_type` ("single-ended"/"paired-ended"), `read_length`.
  - (DSF2/4/8 = depth-downsampled copies; `signal_BW_res25` = float p-value signal; `peaks_res25`.
    **Not used here** — reserved for later depth self-supervision work.)
- DNA: `data/hg38.fa` (+ `.fai`) via pysam, mirroring `data.py:_get_DNA_sequence`. Blacklist:
  `data/hg38_blacklist_v2.bed`.
- Metadata CSV (`data/merged_metadata.csv`) `biosample_name` == directory name (incl. `_grp*_rep*`/`_nonrep`).
  **Labels read from the per-instance JSON** (ground truth); CSV used only for the viability census.

## Instances, grouping, labels
- **Instance** = one `(biosample_dir, assay)` track = one experiment.
- **Independent group** (for splitting, to avoid leakage): **base cell type** (LOCKED) =
  `strip _(grp\d+_)?(rep\d+|nonrep)$`. Technical reps / groups of the same cell type are near-duplicates
  → all variants of a base cell type stay on the **same** side of the split.
- Labels per instance: `run_type` ∈ {single, paired}; `read_length` ∈ ℝ (numeric);
  `depth` = log10(genome-wide depth) from the DSF1 sidecar.

## Depth control — THREE normalization modes (key design point)
Total-count *scaling* turned out **not** to remove depth (the residual-leak meter caught it: depth
R²≈0.90 after scaling). Reason: scaling removes magnitude but not the depth *signature* in sparsity /
Poisson-noise (low-depth tracks are spiky-sparse, high-depth smooth). So we keep all three modes — the
comparison is itself a reported finding:
- **raw** — full-depth counts. Depth fully present (validator, R² high).
- **scaled** — sum-normalized so every instance's chr19+chr21 total == C (median total). **FINDING:
  depth survives (R²≈0.90).** Bins untouched; `arcsinh` for input.
- **thin** — binomial downsampling of every instance to the common minimum depth T (removes magnitude
  *and* noise/sparsity signature). Depth removed (R²≈0). **This is the input for the run_type /
  read_length probes** (the genuinely depth-controlled condition).

## Splits — double holdout
Per assay, per target:
1. Partition **independent groups** into train/test, **stratified by the target label**
   (single/paired for run_type; read-length quantile bins for read_length).
2. **Train = chr19 windows of train-groups; Test = chr21 windows of test-groups.**
   ⇒ a test window is from an **unseen cell type** AND an **unseen chromosome** (no biosample fingerprinting,
   no DNA memorization).
3. Repeat with **k=5** stratified group splits → report mean ± CI (single splits too noisy at these n).

## Windowing & input tensors
- Window = **768 bins** (= 19,200 bp). Non-overlapping tiling of chr19 (train) / chr21 (test).
- Drop windows overlapping the ENCODE blacklist; drop all-zero windows.
- Cap **N_win per instance** (e.g. 512, sampled) to balance instances and bound compute.
- Inputs:
  - count channel `[1, 768]` arcsinh(counts).
  - DNA channel (signal+DNA arm) `[4, 19200]` one-hot (A/C/G/T; N→zeros), same coords as the count window.
- **No x_meta. No control channel.**

## Probes / models (no shared backbone — separate model each)
| probe          | input (norm)   | arms                       | head                  | metric            |
|----------------|----------------|----------------------------|-----------------------|-------------------|
| depth_raw      | raw            | signal-only                | regression (log-depth)| R² — high (sanity)|
| depth_scaled   | scaled         | signal-only                | regression            | R² — FINDING (~0.9)|
| depth_thin     | thin           | signal-only                | regression            | R² — should ≈ 0   |
| run_type       | thin           | signal-only **+** signal+DNA | binary (BCE)        | AUROC, AUPRC      |
| read_length    | thin           | signal-only **+** signal+DNA | regression (MSE)    | MAE, R²           |

- depth probes are the **validators + the normalization finding**: raw high (depth present), scaled
  still high (sum-norm fails to remove depth), thin ≈ 0 (thinning removes it). run_type/read_length use
  the thin input so any signal is genuinely covariate, not residual depth.

## Architecture (simplified CANDI encoder, ~1M params)
- Count tower: `Conv1d(1→64,k7)→GELU→Conv1d(64→128,k7)→GELU` (length stays 768).
- DNA tower (when on): one-hot `[4,19200]` → two stride-5 convs (÷25) → `[128,768]`.
- Fuse: concat channels → 1×1 conv → `[128,768]`.
- Transformer: 2 layers, d_model=128, 4 heads, sinusoidal positions.
- Mean-pool over 768 positions → `[128]` → linear head.

## Aggregation (window → biosample)
The label is a property of the **instance/biosample**, not a window. Report two levels:
- **Per-window**: each test window scored independently → AUROC/R² over all windows. High statistical
  power, but windows of one instance are correlated and share a label (optimistic; effective n ≈ #groups).
- **Per-biosample (headline)**: average a model's per-window predictions within each held-out instance →
  **one prediction per instance** → AUROC/R² over held-out instances. Honest; n = #test groups.
Headline = per-biosample; per-window reported as supplementary (within-instance consistency / power).

## Validation / correctness tests
1. **Normalization invariant**: every instance's normalized total over chr19+chr21 == C (±tol). assert.
2. **Split integrity**: train/test group sets disjoint; train coords ⊂ chr19, test ⊂ chr21; stratified
   class balance preserved (±tol). assert.
3. **depth-prenorm R² ≈ 1** (positive control: depth is trivially in raw counts) and
   **depth-postnorm R² ≈ 0** (normalization works). Built-in self-check.
4. **Label-shuffle negative control**: permute labels across groups, retrain one assay → AUROC → ~0.5.
   A surviving positive ⇒ leakage/bug.
5. **DNA-only negative control**: DNA branch alone (no counts) → AUROC ~0.5 *by construction* (reference
   DNA is identical across biosamples for a given window → cannot separate a per-biosample label).
   A positive ⇒ window/coord or split bug.
6. **Count↔DNA coord alignment**: assert DNA window == counts window coords; `len(count)=768`,
   `DNA=[4,19200]`; one-hot column sums ∈ {0,1}.
7. **Overfit-tiny**: model drives train AUROC→1 on a handful of windows (optimization wired & capable).
8. **Determinism**: fixed seed ⇒ identical split & metrics.

## Model count (base, MERGED, base-cell-type grouping)
- run_type-viable assays (min independent-group class ≥ 6): **7** — DNase-seq, H3K36me3, H3K27ac,
  H3K9me3, H3K4me1, H3K27me3, H3K4me3 (CTCF=5, H2AFZ=4, POLR2A=4 marginal, optional).
- read_length-viable assays (≥3 distinct lengths, std≥10bp): ~**15**.
- Counts: run_type 7×2 arms + read_length 15×2 arms + depth ~16×2 (pre/post) = **~76 base models**,
  × k=5 splits ≈ **~380 tiny fits** (minutes each on a MIG slice).

## Outputs & visualization
- `results.tsv`: per (assay, target, arm, split) → metric, n_train/n_test groups & class balance.
- Fig 1 (headline): per-assay grouped bars — run_type AUROC (signal-only vs +DNA) w/ chance line;
  same panel for read_length R². The **DNA-lift** bar = mappability test.
- Fig 2 (validators): per-assay prenorm vs postnorm depth R² (visual proof norm stripped depth).
- Fig 3: read_length pred-vs-true scatter per assay.

## Deliverable: report.md (after all runs)
`sandbox/diagnostics/covariate_probes/report.md` — embeds Fig 1–3 and **briefly** describes: the data
(MERGED, paths, DSF1 counts + hg38 DNA), what was done (per-assay depth-normalized probes, double
holdout, signal vs +DNA arms), what was tested (run_type / read_length decodability beyond depth;
mappability lift), the controls run (label-shuffle, DNA-only, depth pre/postnorm), and the headline
per-biosample results. Keep it short and scannable.

## Resource profiling (avoid wasted SLURM, fast allocation)
During test/validation, **measure** the minimum allocation per run rather than guessing:
- Profile a single (assay × target × arm × 1 split) on the MIG slice with peak-GPU-mem
  (`torch.cuda.max_memory_allocated`), peak host RAM, wallclock, and dataloader CPU usage.
- Each model is ~1M params → GPU mem expected small (≈1–2 GB on the 10 GB MIG slice); the real cost is
  **npz/FASTA I/O + windowing**, so tune `n_cpu` (dataloader workers) and host mem to that.
- Set sbatch `--time`, `--mem`, `--cpus-per-task` from measured peak × safety margin (~1.5×).
- Run unit: **job array over (assay × target)** combos (~16–22 jobs); each job runs both arms × 5 splits
  sequentially. Small jobs ⇒ fast MIG allocation. Report the chosen per-job resources in report.md.
- GPU (hard constraint, every job): `--gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1`.

## How to run
- Module: `sandbox/diagnostics/covariate_probes/` — `data.py` (read/window/normalize/split),
  `model.py` (encoder), `run.py` (assay×target×arm×split sweep), `plot.py`.
- SLURM, GPU **`--gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1`** (hard constraint). Sequential sweep or
  small job-array.

## Resolved decisions
- Grouping: **base cell type** (no leakage).
- Input counts: **DSF1 only** (full depth); DSF2/4/8 reserved.
- Windows: non-overlapping, **≤512 windows/instance** (sampled).
- read_length target: **raw bp**, stratified by quantile bins for the split.
- Per-job SLURM resources: **measured** via profiling (see Resource profiling), not guessed.

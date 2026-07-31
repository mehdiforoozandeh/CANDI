<!-- candi_kit-reader-preamble -->

> [!NOTE]
> **Reader preamble — added for this kit. Everything below the horizontal rule is the
> original document, unedited.**

**How to read this from outside the project.** This is a frozen, verbatim copy of an internal audit. It
is a *register*, not a narrative — you look things up in it, you do not read it front to back.

**What it is, in plain terms:** three things, and it is worth knowing which you want before you open it.
**(1) Checkpoint forensics** — what the trained weights actually contained, measured directly.
**(2) The `S1`–`S27` defect register** — every known bug or weakness in the measurement code, each with a
file and line. **(3) The `B1`–`B9` bounds** — things this dataset *cannot* answer, no matter how good the
model is. The `B` bounds are the most useful part for a newcomer: `B1` in particular proves that
`run_type` conditioning is unmeasurable on the shipped panel, which will save you from chasing a result
that is not obtainable without re-selecting the data.

Every `S` and `B` item defines itself where it appears, so you can jump straight to one.

### The `h<N>` / `q<N>` ids

Those are entries in the project's research tracker (`h` = a testable hypothesis, `q` = an open
question). The tracker is not shipped. **[`README.md` in this folder](README.md) has a decoder ring
translating every id used anywhere in this kit into a plain-language claim and its verdict.** Look an id
up there rather than inferring it from context.

### Code paths named in this document

It cites files by their path **in the research repository**, which is not what this kit calls them.
Translation:

| cited as | in this kit |
|---|---|
| `sandbox/candi_v2/encoder.py` | `src/candi_kit/encoder.py` |
| `sandbox/candi_v2/decoder.py` | `src/candi_kit/decoder.py` |
| `sandbox/candi_v2/config.py` | `src/candi_kit/config.py` |
| `sandbox/batch.py` | `src/candi_kit/batch.py` |
| `sandbox/data.py` | `src/candi_kit/dataset.py` |
| `_utils.py` | `src/candi_kit/_vendored.py` |
| `sandbox/diagnostics/dual_conditioning/model.py`, `.../dual_conditioning_real/model_real.py` | `src/candi_kit/model.py` (the two were merged) |
| `sandbox/diagnostics/dual_conditioning/metrics.py` | `src/candi_kit/metrics.py` |
| `sandbox/diagnostics/dual_conditioning_real/metrics_real.py` | `src/candi_kit/eval.py` |
| `sandbox/diagnostics/dual_conditioning_real/run_real.py` | `src/candi_kit/train.py` |
| `sandbox/prepare_h5.py` | `src/candi_kit/prep/bake.py` |
| `data.py` (repo root) | `src/candi_kit/prep/handler.py` |
| `sandbox/reference_sample.py` | `src/candi_kit/prep/reference_sample.py` |
| `sandbox/data/sandbox.h5` | the h5 **you** bake — `slurm/bake.sh` |
| `.../results/*.ckpt` | **not shipped.** You train your own; see `README.md` §4 |

Line numbers in citations refer to the research-repo originals and will not line up with this kit.

---

# q19 Metadata-Conditioning Audit — Consolidated Fact Base & Suboptimality Register

Eight auditors + adversarial verification, de-duplicated. Items marked **[RV]** were independently re-verified during consolidation (commands and outputs in §8). Items marked **[×N]** were reached independently by N auditors — treat that as corroboration signal.

---

## 1. CHECKPOINT FORENSICS — the hardest evidence

All from `sandbox/diagnostics/dual_conditioning_real/results/{main,offoff}_s0_perassay.ckpt`. **[RV]**

### 1.1 Decoder metadata weight magnitudes (absmax)

| tensor | main (offset ON) | offoff (offset OFF) | status |
|---|---|---|---|
| `assay_embedding.weight` | **6.2990e-41** | 3.0688e-01 | annihilated ON |
| `runtype_embedding.weight` | **6.2255e-41** | 3.7985e-01 | annihilated ON |
| `depth_proj.weight` | **1.5550e-04** | 4.6770e-01 | 3000× suppressed ON |
| `read_length_proj.weight` | 5.2258e-02 | 4.0363e-01 | **alive in both** |
| `fusion.0.weight` | 5.7118e-02 | 3.4703e-01 | 6× suppressed ON |
| `film_proj.weight` | 3.4682e-01 | 3.9365e-01 | healthy in both |
| `depth_missing_emb` | 5.9564e-41 | **5.9564e-41** | denormal in **BOTH**, bit-identical |
| `readlen_missing_emb` | 6.2617e-41 | **6.2617e-41** | denormal in **BOTH**, bit-identical |
| `depth_cloze_emb` | 4.2984e-02 | **4.2984e-02** | frozen at init in **BOTH**, bit-identical |
| `readlen_cloze_emb` | 5.2004e-02 | **5.2004e-02** | frozen at init in **BOTH**, bit-identical |

**The cloze/missing pair is a within-checkpoint controlled experiment, and it is the single most important forensic fact in this audit.** Three regimes, all present in one file:

- `grad is None` (CLOZE never appears in `y_meta`; `sandbox/batch.py:66` masks `x_meta` only) → `torch.optim.Adam._init_group` skips the param entirely → **frozen at init, bit-identical across arms**.
- `grad == 0.0 exactly` (MISSING is reached but excluded from loss) → Adam still runs, `_single_tensor_adam` does `grad = grad.add(param, alpha=weight_decay)` → **annihilated to denormal in BOTH arms**.
- `grad small but nonzero` → survives, scaled by how load-bearing it is.

**Consequence: denormal weights do NOT predict dead steering** — the offoff arm has denormals too and steers fine. The discriminating variable is **zero task-gradient**, not decay per se. Both the original handoff §9 headline and the "it's just a weight-decay artifact" rebuttal point at the wrong conjunct.

### 1.2 Behavioural probe of the trained decoders **[RV]**

Decoder-only forward, fixed `z`, sweeping one covariate at a time:

| probe | main (ON) | offoff (OFF) |
|---|---|---|
| **total** d⟨log2_mu⟩/d(depth) | **0.999967** | **0.774588** |
| d⟨eta⟩/d(depth) | −1.12e-08 | +0.774594 |
| dispersion `n` over depth 22→28 | 5.833017 → 5.833017 (**rel 0.000e+00**) | 2.802627 → 3.210263 (**rel +14.5%**) |
| run_type flip: max\|Δeta\| | **0.000e+00** | 8.455e-01 |
| run_type flip: max\|Δn\| | **0.000e+00** | 8.619e-01 |
| assay_id permute: max\|Δeta\| | **0.000e+00** | 3.802e+00 |
| read_length 36→101: max\|Δeta\| | 3.691e-03 | 7.974e+00 |

Four load-bearing readings:

1. **offset-ON's total depth response is arithmetically perfect (1.0000); offset-OFF's is wrong by 22% (0.775).** On the behaviourally correct statistic, the "steering-dead" arm wins and the "steering-alive" arm fails. Corroborated on real data by `frac_min_at_true` 0.7588 (ON) vs 0.7597 (OFF) — a statistical tie — and by the wrong-depth CRPS penalty being **5.2× larger** for ON (0.432 vs 0.083). **[×3]**
2. **offset-ON's decoder is bit-exactly blind to assay identity.** Permuting all 8 assay IDs changes the output by exactly 0.0. No thinning argument excuses this: the decoder cannot tell DNase from H3K9me3 except by which 16 of 128 mixed channels each column happens to read. This is the real pathology, and it is invisible to every metric in the suite.
3. **`n` is depth-conditioned by construction** (`raw_n = head_n(feat)`, and `feat` is FiLM-modulated by `y_meta` whose row 0 is depth; `model_real.py:117-124`). It is depth-frozen in the ON arm *only because the embedder died*. The claim "not conditioning `n` on depth is correct" is circular — it uses the collapse as evidence the collapse is correct.
4. **`read_length_proj` is the sole surviving decoder metadata input under offset-ON** — exactly the covariate §2.3 shows still carries un-modelled exposure signal. A natural positive control for the starvation mechanism.

### 1.3 Recorded metric values **[RV]**

```
main   eta_slope: median -9.096e-17, min -1.049e-06, max 8.583e-07, frac|·|<1e-6 = 0.9992
offoff eta_slope: median  0.8799,    min -0.5071,    max 1.3845
main   run_type: frac_direction 0.0,    mean_responsiveness 0.0        (exact ties, 1215/1215)
offoff run_type: frac_direction 0.6872, mean_responsiveness 1.8289
both   depth null: {mean 0.0, lo 0.0, hi 0.0, n 893528}
```

`metrics_real.py:436` decides `offset_independent` from `median(eta_slope) > 0.0` — under offset-ON that is the sign of float32 noise at 1e-17.

---

## 2. VERIFIED FACT BASE

### 2.1 Architecture (q19 fork)

- `log2_mu = (depth − 25.1) + eta` when `use_offset`, else `log2_mu = eta`; clamped to (−15, 30); `n = softplus(raw_n)`; `p = n/(n+mu)`. `model_real.py:126-137`, `depth_row=0`, `depth_center=25.1` at `:101`. **[RV]**
- Decoder FiLM is **adaLN-zero**: `nn.init.zeros_` on both `film_proj.weight` and `.bias` (`dual_conditioning/model.py:143-144`), inherited unchanged by `RealDualCondDecoder` via `super().__init__` (`model_real.py:103-105`). Bit-exact identity at step 0 in both arms.
- **At init, 16 of 18 decoder metadata tensors receive no gradient** in BOTH arms — 14 exactly 0.0, 2 `None` (the cloze embeddings). Only `film_proj.weight/bias` get gradient. This is an algebraic consequence of `W=0 ⇒ dL/d(memb) = JᵀW = 0`, not an empirical property. **It lasts exactly one optimizer step** (step 1: `depth_proj.weight` 1.383e-05, `assay_embedding.weight` 3.188e-07, `fusion.0.weight` 1.381e-04). The load-bearing quantity is the **grad-vs-decay ratio** — for `assay_embedding`, task grad ~1e-6 vs `wd·|w|` ~3e-5, i.e. decay is ~30× larger — not the zero-at-init transient.
- Optimizer: `torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)`, `eps` default 1e-8 (`run_real.py:92,95`). Coupled L2. All recorded runs confirm `weight_decay: 0.0001`.
- Trunk is **ungrouped** (`grouped=False`, `dual_conditioning/model.py:138`); the `[B,L,A,C]` "per-assay" view is an arbitrary reshape of a fully-mixed 128-channel vector. Heads are weight-shared `Linear(16,16)+GELU+Linear(16,1)` with a **single scalar output bias**. **The FiLM is therefore the only explicit per-assay parameter in the decoder** — and under offset-ON it is metadata-invariant to ~1e-7.
- FiLM `(gamma,beta)` are broadcast over all 768 positions (`model_real.py:121`). Empirically the depth response is 93–95% a position-constant shift even in the working arm.
- Parameter split: trunk 2,830,848 (91.2%); `film_proj` 1,056 (**0.034%**); `meta_embedding` 5,984. Decoder total 2,838,466; metadata pathway 0.248%.
- **Full-coverage is the discriminating condition, not the offset alone.** The exact-zero run_type null appears in all four `full_coverage=True` runs; the three short runs (`steps_per_epoch=300`) with `use_offset=True` show live responses (frac_direction 0.310 / 0.707 / 0.743). Annihilation is a training-length phenomenon — corroborating the decay mechanism.

### 2.2 Data — the DSF axis

- DSF within-(biosample,assay) slope of log2(mean count) on `depth_log2`: **n=38, mean 0.9999, sd 0.0009, min 0.9978, max 1.0023.** **[RV]** **[×3]**
- Depth bookkeeping: `meta_dsf{k}[0] − meta_dsf1[0] = −log2(k)` to max deviation 2.44e-03 (12.3% of levels exceed 1e-3 — the recorded depth is the *actual* subsampled BAM read count, not the ideal). **[RV]**
- DSF is genuine Bernoulli read-retention (`data_utils.py:1135`), and the conditional is exactly Binomial(c₁, 1/k).
- **The DSF levels are NOT a nested ladder.** `dsf2 ≤ dsf1` is True but `dsf4 ≤ dsf2` is **False** and `dsf8 ≤ dsf4` is **False** — independent draws from the full BAM. **[RV]** A training pair with `x_dsf=2, y_dsf=4` is therefore not a subsample relation; "counts ∝ 1/dsf" holds only in expectation, with independent extra Poisson noise.
- NB is closed under binomial thinning with `n` preserved (simulated: thinned mean 25.006/var 181.22 vs NB(n, q·mu) 25.005/181.07).

### 2.3 Data — the natural (cross-experiment) depth axis

Full-data regression, all 7,485 windows, 38 experiments, assay fixed effects: **[RV]**

| window set | subset | depth-only | +log2(read_length) | read_length coef |
|---|---|---|---|---|
| ALL | ALL (n=38) | 1.274 ± 0.062 | **0.975 ± 0.064** | 0.588 |
| ALL | T_ (n=26) | 1.258 ± 0.087 | **1.007 ± 0.076** | 0.588 |
| chr19 (train) | T_ | 1.235 ± 0.105 | **0.976 ± 0.108** | 0.606 |
| chr21 (eval) | T_ | 1.297 ± 0.062 | **1.094 ± 0.039** | 0.476 |

**This settles the biggest inter-auditor contradiction (§6.1).** The marginal excess slope over 1 is real, but it is **not a depth exposure-exponent effect** — it is read_length. Once `log2(read_length)` enters, the depth coefficient is statistically indistinguishable from 1. Mechanistically expected: a length-R read at 25 bp resolution touches ~R/25+1 bins, predicting a coefficient ~0.68 over the observed range vs the fitted 0.48–0.61.

Caveat, and it is severe: assay-centered `corr(depth, log2rl) = 0.763`, `corr(depth, run_type) = 0.590`, `corr(log2rl, run_type) = 0.697`, with n=38 from **only 10 biosample views / 5 cell types**. **[RV]** Cluster-bootstrapping over biosamples puts the full-model depth coefficient in [−0.19, +0.20] and read_length-alone R² in [0.41, 0.89]. The split between depth and read_length is directionally clear but not sharply identified. Also: given assay FE, `log2(read_length)` is itself 73% predicted by biosample identity — a large share of the "read_length effect" is indistinguishable from cell-type/batch coverage. After assay+biosample FE it retains partial corr 0.689 and slope 0.543, so it is not *pure* batch proxy.

**read_length is not in the offset at all** (`model_real.py:126-130` reads `y_meta[:, 0, :]` only); it reaches the decoder solely through `MetadataEmbedding.read_length_proj` (`candi_v2/encoder.py:83,131,138`) → `film_proj`.

### 2.4 Data — covariate joint distribution

- **Assay column order in `sandbox.h5` is NOT `SANDBOX_ASSAYS`.** Full-triple (depth, read_length, run_type) match against `data/eic_metadata.csv`: **38/38 under handler order, 5/38 under `SANDBOX_ASSAYS`.** **[RV]** **[×2]**
  True order: `[DNase-seq, H3K4me3, H3K36me3, H3K27ac, H3K9me3, H3K27me3, H3K4me1, ATAC-seq]`.
  Mechanism: `reference_sample.py:47-48` returns `handler.aliases['experiment_aliases'].keys()` (ordered by descending biosample availability, `data.py:295-301`); `prepare_h5.py:352` writes in that order; `metrics_real.py:244,336,420` label with `SANDBOX_ASSAYS[a]`. **[RV]**
  Only index 5 (H3K27me3) coincides. Display→truth map: `ATAC-seq→DNase-seq, DNase-seq→H3K4me3, H3K4me3→H3K36me3, H3K4me1→H3K27ac, H3K27ac→H3K9me3, H3K27me3→H3K27me3, H3K36me3→H3K4me1, H3K9me3→ATAC-seq`.
- **run_type conditional entropy in T_ (n=26): H(rt)=0.7063, H(rt|assay)=0.1248, H(rt|assay,read_length)=0.0000.** **[RV]** **[×2]** Exactly one assay (col3 = H3K27ac) has within-assay run_type variance, and its single paired record is also the unique read_length=100 record and the deepest — run_type, read_length and depth are perfectly confounded on that one contrast.
- Told-depth distribution actually seen in training (T_ × uniform DSF{1,2,4,8}): **mean 23.227, sd 1.841, range [19.20, 28.13]**. `depth_center=25.1` is the mean over **all views at dsf1 only** (25.117). **[RV]** **[×2]** The offset term has mean ≈ −1.9 rather than 0, contradicting the head's own docstring rationale.
- DSF only ever **down**-samples, so the training depth support for assay *a* is [natural_min − 3, natural_max]. **7/12 held-out targets sit above the per-assay training ceiling** (worst +1.43 log2). Every eval extrapolation is in the untrained direction.
- **9/12 held-out targets are OOD in (assay × read_length)** — worse than the run_type OOD (5/12) and unreported anywhere.
- There is exactly **one natural depth per (biosample, assay)**: `meta_dsf{k}` is a single `[4,8]` array per biosample with no window axis. DSF is the only paired same-position depth counterfactual in the data.
- `T_RWPE2` has exactly one available assay; `_utils.py:152-154` skips `num_available <= 1`, so it contributes **zero imputation supervision** while occupying ~20% (381/1905) of full-coverage steps — a pure copy task, the failure mode the offset was introduced to patch.
- Two complete, non-degenerate, unused covariates exist in `data/eic_metadata.csv` and `data/merged_metadata.csv`: `sequencing_platform` (9–10 levels) and `lab` (6–16 levels). In the T_ slice: H(platform | assay, read_length) = 0.443 bits, H(lab | assay, read_length) = 0.212 bits — vs run_type's 0.000. Control depth is **already in the h5** as `control_meta[:,0,0]`, varies within assay across biosamples, needs no re-bake.

### 2.5 Metrics instruments

- **`_foreground_mask` degenerates.** `thr = quantile(target, 0.98); target >= thr` on 36–67%-zero targets. Requested 2%, **actual mean 23.94% selected; 262/1215 records (21.56%) select 100%**, and those supply **90.08% of the pooled bootstrap sample**. **[RV]** In every saturated case the 98th percentile is exactly 0 (`frac(saturated & thr>0) = 0.0`); the `fg.sum()<5` fallback never fires. 221/1215 records have an identically-zero target and alone supply 76% of the pooled mass. Among non-saturated records the selector is near-nominal (median 2.51%).
- **The bootstrap resamples positions, not targets.** `metrics_real.py:73-82` resamples 893,528 pooled per-position deltas. Those are 12 distinct (imp_biosample, assay) targets × ~101 chr21 windows from **5 biosample pairs / 4 cell types**, with `(T_RWPE2, B_RWPE2)` supplying 7 of 12. n_fg-weighted cluster bootstrap over the 12 targets: offoff run_type goes from **[+0.0655, +0.0739] to [−0.0215, +0.1810]** (lo<0 in 200/200 seeds); read_length to [−0.373, +0.115]. **The h42 "CI excludes 0" result does not survive correct clustering.** **[×2]**
- **`excludes_zero` is sign-blind** (`metrics_real.py:82`: `lo > 0.0 or hi < 0.0`) while `delta = crps_flip − crps_true` (`:328`). Three laundering instances in shipped results, all rendered "✓" by `report_all.py:332-333`: offoff run_type `single` = [−0.01644, −0.01466] (**anti**-steering); offoff read_length overall = [−0.1365, −0.1149] (**anti**); copyable read_length overall = mean **6.73e-09** with CI [5.45e-09, 8.14e-09] (magnitude-blind too — no minimum-effect gate).
- **The shuffled-depth null is a mathematical no-op.** `y_meta_imp` is one `[4,F]` tensor broadcast across the batch (`data.py:351`), so `base_d[perm] == base_d`; and `told[0] = base_d − log2(1)` is bitwise identical to `base_d`. Null = {0.0, 0.0, 0.0} in all 10 result files, all 1215 targets. **[RV]** Its unit test uses the same constant-depth fixture (`tests/test_metrics_real.py:70`) — mutation-tested: deleting the permutation entirely leaves the test **passing**.
- **The per-assay marginal baseline is the constant-zero forecast for 4/8 assays.** `mmu = median(tgt) + 1e-6` (`metrics_real.py:225`). 4 assays have median target 0 → mu₀=1e-6, p below `P_EPS` → point mass at 0 → `marg_crps` = mean(y) exactly. **[RV]** Confirmed: recorded marg_crps 5.1894/0.7454/1.2889/1.1980 equal the target means to 12 decimals. Against a properly centred unconditional NB the macro bar is 1.837 → mean-matched 1.677 (−8.7%) → CRPS-optimal 1.564 (−14.9%). Re-scored: **offset-ON 8/8 → 5/8; offset-OFF 3/8 → 2/8.** The global pooled `marginal_crps = 2.2062` is degenerate identically. `nb_crps` itself is sound (agrees with exact discrete sum / 60-digit mpmath to ≤2.4e-5 across the whole q19 hull, and is exact in the degenerate limit).
- **The "denoise" family is autoencoding.** Eval uses `dsf_sampling='off'` (`metrics_real.py:117`) → `x_dsf = y_dsf = 1` (`data.py:36-37`) and `apply_mask=False` (`:125`), so `x_data == y_data` for every available assay. den CRPS 0.305/0.332 vs imp 1.617/2.060; `health_gate_den_ge_imp` is near-automatic.
- `natural_variance_insufficient` is computed **entirely from the model's own mean response** (`mean|mu_true − mu_flip| < 1e-6`, `metrics_real.py:335,362-363`). Proven model-side, not data-side: main and offoff evaluate the *identical* 1215 targets with identical covariate composition (304 single / 911 paired), yet the flag is True for main and False for offoff. It is rendered as "honest null" in `report.py:154` and was recorded as such in h42 before an external control overturned it.
- `--eval-max-batches` truncates a deterministic spatially-ordered iteration (`shuffle=False`), so any truncated eval is a biased genomic-region sample: the first 150 of 608 units are ~8× lower-signal.
- Minor: `frac_direction` uses strict `>` on exact ties, so "100% ties" reports as "0% correct"; `_agg` writes NaN over per-split responsiveness; `OBSERVED_READLENS` is hardcoded (`:42`); `encoder_eff_rank` names two different quantities in M1 (`:193`, per-position) and M3 (`:484`, mean-over-length); M3's "between" pool admits same-region pairs.

### 2.6 Production topology (bearing on transfer)

- Production decoder (`DecoderConfig` defaults, signal_dim=8, encoder_d_model=72) totals **26,464 params, metadata pathway 10,672 (40.3%)** — vs q19's 2,838,466 with 0.248%. q19's decoder is 107× larger with a 162× smaller metadata share.
- **Production's only active decoder FiLM pools metadata across assays**: `pooled = meta_embed.mean(dim=1)` (`candi_v2/decoder.py:208`, `:490`). q16/h34 already established across-assay pooling as the cause of the v1 output-steering null. q19 is per-assay (`pool_meta=False`) and is therefore **stronger** on the axis that governs steering.
- Production FiLM is xavier + N(0,0.1) bias, **live at step 0** (measured: W std 0.1114, `torch.equal(FiLM(z), z) == False`, mean|Δz| 0.2517). `model.py:2390-2402` carries an explicit comment that this was a deliberate fix ("Previously used near-identity init … which allowed the model to ignore prompts"). The zero-init is the q19 exception, not the house style.
- Production NB head is a **single dense `Linear(8,8)`** on an 8-channel trunk output, with nonzero off-diagonals — assays are linearly mixed, no nonlinearity, no per-assay feature vector. Tighter bottleneck than q19's.
- Effective production `depth_center` is **22.5** (`config.py:120`, `candi_v2_default.yaml:25`), not the 24.0 class kwarg.
- Production has **no `log2_mu` clamp** (`decoder.py:172-174`) — live overflow hazard in the code path any fix must land in.

---

## 3. SUBOPTIMALITY REGISTER — ranked by severity × confidence × bearing on the Pareto

### Tier 0 — the Pareto is partly an artifact of these

**S1. `eta_slope` measures *where* the depth response is implemented, not *whether* the model steers — and the h45 bar mechanically forbids the hybrid it exists to find.** `severity: blocking · confidence: certain · bearing: direct` **[×3]**
`metrics_real.py:406-408,436` regresses the **offset-free residual**. Under offset-ON the correct residual is 0 and the measured total response is 1.0000 **[RV]**. For any hybrid `log2_mu = β(d−c) + η` the total is `β + eta_slope`; requiring `eta_slope ≥ 0.7` while keeping total ≈ 1 forces `β ≤ 0.3`, disqualifying the β=0.5 and 0.75 arms **before they run**. The recorded evidence already contradicts the "ON = steering dead" reading (`frac_min_at_true` 0.759 vs 0.760; wrong-depth CRPS penalty 5.2× *larger* for ON). `offset_independent` is decided by the sign of a 1e-17 float.
→ Re-pre-register on **total slope** `|d log2_mu/d(told depth) − 1| ≤ tol`, keep `eta_slope` as a labelled attribution diagnostic, and add a **metadata-ablation score** (randomise `y_meta`, measure CRPS/Spearman degradation) as the covariate-agnostic "does it use metadata at all" number. Do this before launching h45.

**S2. The decoder metadata pathway is deleted by zero-task-gradient × coupled-L2, and the FiLM slot is then repurposed as an unconditional affine.** `severity: blocking · confidence: certain · bearing: direct` **[×3]**
adaLN-zero gives exactly-zero gradient at step 0; thereafter the task gradient stays 2–3 orders below `wd·|w|` and coupled L2 wins. Endpoint: `assay_embedding` 6.30e-41, `runtype_embedding` 6.23e-41 **[RV]**, and the decoder is **bit-exactly blind to assay identity and run_type** **[RV]**. Critically, `film_proj` is *not* healthy in the useful sense: its metadata sensitivity is ~1e-7 while its output reaches max|γ|=4.30 — the optimizer converted the conditioning projection into a **fitted, load-bearing, metadata-blind per-channel gain**. Reviving the embedder therefore faces a *second* barrier the handoff §9 framing does not anticipate. Measured conditioning capacity over the 152 real metadata states: `memb` effective rank **1.008** (ON) vs 3.752 (OFF); realised (γ,β) manifold **1.016** vs 2.451 out of 32 dims — even the working arm expresses ~2.5 independent directions where assay identity alone needs ~7.
→ (a) `AdamW` + a **no-decay param group** for all embeddings / LayerNorm affines / biases / `film_proj` (GPT-3, minGPT, LLaMA split; DiT — the source of adaLN-zero — trains with **no weight decay** at all, so the current combination is a recipe violation). (b) Replace adaLN-zero with production's xavier + N(0,0.1). (c) **Split the affine**: give the decoder an explicit unconditional per-channel scale/bias (or LayerNorm) so FiLM cannot be co-opted as free bias capacity, and drive FiLM strictly from `(memb − memb̄)` with no unconditional component. (d) Log per-field **absmax** per epoch — not `nnz`, the dead weights are denormal-nonzero.

**S3. Foreground selection degenerates to "all positions", so the steering tests are 90% background.** `severity: blocking · confidence: certain · bearing: direct` **[×2]**
Requested 2%, actual 23.9%; 21.6% of records at 100%; those supply 90.1% of the bootstrap mass **[RV]**. Direct power consequence measured: a 0.68–2.20 log2 (up to 4.6×) shift in μ moves CRPS by −0.00003 on those positions, *in the wrong direction*, because they are background. h37's entire premise is defeated inside the instrument built to enforce it.
→ Select by rank (`argsort(target)[-k:]`) or require `target >= max(thr, 1)`; emit realised fg fraction per record; weight the pooled statistic **by target, not by position count**.

**S4. Every M2 significance claim is ~24× over-confident, and the headline h42 result does not survive clustering.** `severity: blocking · confidence: certain · bearing: direct` **[×2]**
Effective n is 12 targets / 5 biosample pairs / 4 cell types, not 893,528 positions. offoff run_type: [+0.0655,+0.0739] → **[−0.0215,+0.1810]**, lo<0 in 200/200 seeds. The `single` subgroup has n=3 clusters, not 304.
→ Aggregate to one number per target, then bootstrap/permute over those 12 (or the 5 pairs for the conservative bar). Report `n_clusters` beside every CI; make a target-level sign test primary. *Method note:* use **n_fg-weighted** cluster means — the unweighted variant is dominated by a handful of blown-up records and flips run_type to "significant" with a point estimate 13× the reported mean.

**S5. The sole magnitude pillar ("beats per-assay marginal 8/8 vs 3/8") is measured against a constant-zero forecast on half the assays.** `severity: blocking · confidence: certain · bearing: direct`
4/8 assays have median target 0 → the "NB marginal" is a point mass at 0 and `marg_crps` = mean(y) exactly **[RV]**. Against a properly centred marginal: **ON 8/8 → 5/8, OFF 3/8 → 2/8**. The gap narrows but does not close — the ON arm still clears the CRPS-optimal bar, by 4.4% instead of 18.6%.
→ Mean-match or CRPS-fit the marginal. Add the strictly stronger arbiter: **oracle per-assay multiplicative scale** — `CRPS_oracle_scaled` is the capability number, `CRPS − CRPS_oracle_scaled` is the calibration number. That decomposition is what turns "Pareto frontier" into "one arm has a fixable per-assay scale error".

**S6. Every per-assay label in the q19 reporting stack is permuted.** `severity: blocking · confidence: certain · bearing: interpretive, and it corrupts the two headline per-assay arguments` **[×2] [RV]**
Training is unaffected (integer assay_id only), so the **numbers stand** — but every biological reading is misattributed. Concretely: the "ATAC-seq contributes 56% of the CRPS gap (3.727→5.555)" row is column 0 = **DNase-seq**; the "excluding H3K27ac, offset-OFF wins macro Spearman" outlier (0.625→0.010) is column 4 = **H3K9me3**. The latter flips the story from "an active-enhancer mark inexplicably collapses" to "the broad low-SNR heterochromatin mark collapses" — a far more coherent account. `build_canonical_meta` (`data.py:84-100`) has the same bug and is **live** in `sandbox/train.py:560` where `use_canonical_missing_meta=True` is the default (inert in q19: `use_canonical=False` + `pool_meta=False`).
→ Do **not** re-bake. Add `H5_ASSAY_ORDER` next to `SANDBOX_ASSAYS`, use it at `metrics_real.py:244,336,420` + `report*.py` + `chr21_umap.py:49`, pass it to `build_canonical_meta`, add a bake-time assertion. Re-label existing JSONs by permutation.

### Tier 1 — real design defects that plausibly move the frontier

**S7. The exposure term hardwires depth alone; read_length is the missing physics.** `severity: major · confidence: high (direction), medium (magnitude) · bearing: direct` **[×2]**
`log2_mu = (d − 25.1) + η` with no read_length term, while read_length carries a coefficient of ~0.48–0.61 on log2 mean count and spans 30–101 bp (1.75 log2 units) **[RV]**. It is left for a 1,056-param starved pathway to discover — and `read_length_proj` is the one decoder metadata input that survives annihilation precisely because it still has a task gradient. Confounding caveat from §2.3 applies: n=38, `corr(depth, log2rl)=0.763`, and read_length is 73% predicted by biosample given assay FE.
→ `log2_mu = (d + λ_rl·log2(read_length/25 + 1) + λ_rt·run_type − c) + η` with `λ_*` learned scalars init at the fitted values (or at the physical value 1 for the bin-coverage term). Keeps the arithmetic-shortcut benefit for magnitude while making read_length load-bearing instead of orphaned.

**S8. Under offset-ON the decoder has zero explicit per-assay conditioning capacity.** `severity: major · confidence: certain · bearing: direct`
Ungrouped trunk + weight-shared head with a single scalar bias ⇒ FiLM is the only per-assay parameter ⇒ and it is assay-invariant to 8e-08 (bit-exactly 0 response to an assay-ID permutation **[RV]**). Peak width, background level and dynamic range differ enormously across the panel; a position-constant affine on a shared 16-channel slice cannot express that. Even the working arm has only ~2.5 effective conditioning dimensions.
→ Widen `feat_per_assay` (16→32/64); add per-assay head bias/scale (8×2 params); apply FiLM at more than one trunk depth. Report **(γ,β) effective rank as a first-class training diagnostic** — it is cheap and would have flagged the ON-arm death inside the first epoch.

**S9. Decoder conditioning is applied once, at 0.034% of params, after the entire 91%-of-model trunk.** `severity: major · confidence: high · bearing: moderate`
The trunk computes the whole spatial profile with no knowledge of what measurement it is producing; `y_meta` only rescales the 16 channels it already committed to. The encoder already does FiLM after every conv; the decoder does not. `DecoderTrunk.forward` already accepts `film_layers` and `pooled_meta` (`candi_v2/decoder.py:265-277`) and `model_real.py:113` passes neither — this is a forward-call change, not surgery.
→ **Read the §5 caveat before proposing this**: the pooled variant was already tested and rejected. The untested variant is per-deconv-layer FiLM with **per-assay** metadata.

**S10. FiLM is position-constant, so metadata cannot express the operation depth and run_type actually perform on a coverage track.** `severity: major · confidence: high · bearing: moderate`
Measured: 93–95% pure shift even in the working arm (sd across positions 0.04–0.12 vs mean 0.72–1.11). For depth this is *correct* — thinning scales the mean uniformly and the offset gets it exactly right (sd across positions exactly 0). The inadequacy is for assay identity, and for run_type (paired-end changes fragment-length smoothing, a purely positional effect).
→ SPADE-lite: make `(γ,β)` a function of `(memb, local feature summary)`, or gate two FiLM parameter sets by a learned foreground/background soft mask.

**S11. The dispersion head has no metadata route the mean offset cannot shortcut.** `severity: major · confidence: high (revised) · bearing: direct`
Under offset-ON, `n` has **bit-exactly zero** response to told depth **[RV]** — it cannot express that an OOD prompt should widen the interval, or that imputation uncertainty differs from denoising uncertainty. Under offset-OFF the same head learns a 14.5% response **[RV]**, so the capacity exists and offset-ON throws it away. Note the correction to the original framing: `n` is *not* "depth-blind by construction" — it reads the FiLM'd features (§6.2).
→ `log n = n₀(feat) + g(y_meta)` with `g` a small MLP on `memb`. **This is the cleanest available demonstration of the PI's thesis**: a pathway with no competing hardwired-correct arithmetic path, where the mean stays offset-anchored. It is a candidate h45 arm the current plan does not contain.

**S12. The offset is a boolean, foreclosing the hybrid the PI wants.** `severity: major · confidence: certain · bearing: direct`
`(d−c)+η` vs `η` admits no middle. Note the empirical stakes: offset-ON's total slope is 1.0000 and offset-OFF's is 0.775 **[RV]** — a learned-but-anchored coefficient is exactly the untested region.
→ `log2_mu = β·(d − c) + η` with `β` a learned scalar (or per-assay), init 1.0, **excluded from weight decay**. β=1 reproduces ON exactly, β=0 reproduces OFF exactly, everything between is h45 arm (b) for free. The fitted β is scientifically interesting on its own. **Expect β ≈ 1** given §2.3 — treat a large deviation as a red flag for the read_length confound, not as a discovery.

**S13. 20% of training steps carry zero imputation supervision and are a pure copy task.** `severity: major · confidence: certain · bearing: moderate`
`T_RWPE2` (1 available assay) is never masked but gets an equal share of the full-coverage round-robin — 381/1905 updates train only the observed branch on a single assay, i.e. reconstruct-with-a-depth-rescale, which the arithmetic offset solves for free.
→ Weight the round-robin by maskable-assay count. **Do not simply drop it** — it is the only source of within-assay run_type variance (§4.1).

**S14. The one covariate with a real counterfactual ground truth is never scored against it.** `severity: major · confidence: certain · bearing: direct`
`counts_dsf{2,4,8}` + matching `meta_dsf{k}` exist and are already used by M3, but M2's depth sweep scores every told depth against the **fixed dsf1 target** (`metrics_real.py:380,383`). Telling a lower depth is wrong by construction, so `dir_mean_delta > 0` is guaranteed for any model whose μ decreases with told depth — which the offset does arithmetically. `frac_min_at_true` is 0.7588 vs 0.7597: **zero discriminative power between arms** **[RV]**.
→ Cache `z` from the dsf1 input; for k ∈ {1,2,4,8} decode with the dsf-k prompt and score against `counts_dsf{k}`; require `CRPS(told=k, GT=k) < CRPS(told=1, GT=k)`. Real null, not satisfiable by the offset unless the offset is correct, identical for both arms. **Constraint:** covers the down-sampling direction only.

**S15. Nothing in training ever penalizes ignoring the metadata.** `severity: major · confidence: high · bearing: direct`
Training passes the honest `y_meta` on every step, so the loss is minimized identically by a model that reads it and one that infers everything from `z`. Combined with S2, "no dedicated gradient" becomes "deleted".
→ **Conditioning dropout**: with p≈0.1–0.2 replace `y_meta` rows with the learned MISSING sentinel during training. **Zero architecture change — the sentinel embeddings already exist**, and this simultaneously revives `depth_missing_emb`/`readlen_missing_emb`, currently annihilated in both arms (§1.1), which is a clean testable side-prediction. Gives a free inference-time guidance dial; do the extrapolation in `log2_mu` space and select `w` on a validation CRPS curve.

**S16. `excludes_zero` is sign-blind and magnitude-blind.** `severity: major · confidence: certain · bearing: direct`
→ Emit `sign` and `supports_direction = lo > 0` separately; make the pre-registration reference `supports_direction`; add a minimum-effect-size gate.

**S17. `natural_variance_insufficient` is an unfalsifiability hatch.** `severity: major · confidence: certain · bearing: direct`
It converts the *strongest possible refutation signal* (bit-exact zero response) into a non-refutation, and it is a model statistic wearing a data statistic's name.
→ Rename the model-side quantity `model_unresponsive`. Compute a genuine data-side flag (within-assay covariate entropy in the training pool + fraction of eval targets whose true/flipped value is out of support) and let **only that** excuse a null.

**S18. The magnitude/shape mixture manufactures the Pareto.** `severity: major · confidence: high · bearing: direct`
Macro CRPS is location-dominated (4× μ error = +84%; 4× dispersion error = +9%) and PIT-ECE cannot separate them (0.187 vs 0.127). Offset-ON wins the scale component, offset-OFF wins the shape component, and summing them produces a frontier. Borzoi and BPNet independently factorize into Poisson-on-total + multinomial-over-positions for exactly this reason. **Caveat before importing**: CANDI emits a per-position `n`, so independent NBs factorize as NB-on-sum × **Dirichlet**-multinomial, not multinomial — the clean factorization needs `p` shared across positions.
→ At minimum, report the split. Emit per assay: oracle scale error `c* = argmin_c CRPS(μ·2^c)`; shape score after removing `c*`; dispersion error vs an oracle `n` scale; and `CRPS_oracle_scaled` as the headline capability number.

**S19. `depth_center = 25.1` is ~1.9 log2 off the training prompt mean (23.227), and continuous covariates enter `Linear(1,32)` unnormalized.** `severity: major · confidence: certain · bearing: moderate` **[×2] [RV]**
Depth spans [19.20, 28.13] and read_length {30..101} fed raw, after which the fusion LayerNorm pins ‖memb‖ to √32 — the entire depth range moves the embedding a small fraction of its radius while a read_length flip moves it several times more. `depth_proj.weight` is consequently the smallest live weight in the ON checkpoint (1.56e-04). **The frozen golden testbed this harness forked from already had `norm='zscore'` as the default** (`dual_conditioning/model.py:45,76-78`) and q19 dropped it.
→ Standardize both continuous covariates as registered buffers; set `depth_center` to the measured prompt mean (or estimate it from epoch 1). Cheapest candidate explanation for a steering deficit that has nothing to do with the offset.

### Tier 2 — real but lower leverage

**S20.** The "denoise" family is autoencoding (`dsf_sampling='off'`, `apply_mask=False`); `health_gate_den_ge_imp` is near-trivial. → score `x` at a higher DSF than `y`, or cloze-mask the assay. `major · certain · low`
**S21.** No noise ceiling anywhere; with 36–67% exact zeros it is unknown whether macro Spearman 0.505 is near-ceiling. → binomial-thinning ceiling (Spearman between two half-depth splits). `major · certain · low`
**S22.** M1 emits point estimates with no CI, no stratification, no seed replication, while h45's bar is written against them. The handoff's own seed delta (0.120 CRPS) is comparable to some reported gaps. `major · certain · moderate`
**S23.** No condition-recoverability diagnostic. The standard conditional-generation probe (can an auxiliary model recover `c` from the output?) needs no dose, no counterfactual GT, no natural variance — and would have caught the 6e-41 embedder without checkpoint forensics. `major · high · moderate`
**S24.** Truncated evals are region-biased (`shuffle=False`); first 150 of 608 units are ~8× sparser. `minor · certain · low`
**S25.** `control_avail` uses `(control_data != 0).any()`, so B_DND-41's all-(−1) control reads as available (`data.py:285`). Inert for q19 (B_ groups are ground-truth only) but live the moment a B_ biosample is an encoder input. `cosmetic · certain · none`
**S26.** Production has no `log2_mu` clamp — a live overflow hazard in the code path any fix must land in, and it binds any arm raising the exposure coefficient. `minor · certain · low`
**S27.** Cosmetics: NaN'd per-split responsiveness; dead `crps_wrong` shape ternaries; `encoder_eff_rank` key collision; hardcoded `OBSERVED_READLENS`; M3 "between" contamination; `frac_direction` strict-`>` on ties.

---

## 4. IDENTIFIABILITY BOUNDS — no architecture can fix these

**This is the honesty rail for the next phase. Any h45 arm whose success criterion lands here is unwinnable by construction.**

**B1. run_type is analytically unidentifiable in the T_ training set.** `H(run_type | assay_id, read_length) = 0.0000 bits`, n=26 **[RV]**. It is a deterministic function of the other two covariates, so any function of run_type is reproducible at zero loss cost and weight decay has a strictly free ride on the embedding. h42's "run_type is ignored" is a **data property**, not an optimization or architecture failure. Only one assay (col3 = H3K27ac) has within-assay variance, and its single paired record is simultaneously the unique read_length=100 record and the deepest — the entire learnable signal rests on **n=1**. *(The full EIC panel retains 0.551 bits after conditioning on assay, so this is an artifact of the 5-biosample slice, not of the field.)*
→ **No architecture change can make run_type identifiable on this split.** Either re-select the panel with an explicit constraint, or drop run_type as a verifiable.

**B2. On the DSF axis the offset is the exact and complete generative model, so η has analytically nothing to learn about depth.** Slope 0.9999 ± 0.0009 **[RV]**; NB is closed under thinning with `n` preserved. `eta_slope ≈ 0` under offset-ON is the **arithmetically correct answer**, not a failure. **Scope this precisely**: it bounds the DSF axis only. §2.3 shows the cross-experiment axis has real residual structure — it just turns out to be read_length rather than a depth-exponent deviation.

**B3. There is exactly one natural depth per (biosample, assay).** `meta_dsf{k}` is `[4,8]` per biosample with no window axis, so a *within-batch* depth permutation is structurally the identity. The shuffled-prompt depth null cannot be made informative without permuting across targets/assays or drawing from another target's empirical depth.

**B4. Effective replication is 12 targets / 5 biosample pairs / 4 cell types.** 7/12 targets come from one pair. Every position-level CI in the suite is a fiction; the `single` subgroup is n=3 clusters. No metric change recovers statistical power that the panel does not contain.

**B5. Depth, read_length and run_type are mutually collinear at n=38.** Assay-centered `corr(d, log2rl)=0.763`, `corr(d, rt)=0.590`, `corr(log2rl, rt)=0.697` **[RV]**; cluster-bootstrapped over 10 biosamples the full-model depth coefficient spans [−0.19, +0.20]. **Attribution among the three exposure covariates is not identified on this dataset** — you can fit them jointly, you cannot cleanly credit them. Any claim of the form "covariate X carries Y% of the exposure signal" is unsupportable here.

**B6. DSF only down-samples, so upward depth extrapolation is never trained.** Training support is [natural_min − 3, natural_max]; 7/12 eval targets sit **above** the per-assay ceiling (worst +1.43 log2). The magnitude verifiable is being scored in a regime the training distribution never covers, and the error direction is systematically negative by construction.

**B7. 9/12 eval targets are OOD in (assay × read_length)**, and the read_length flip lands outside the per-assay training support in 7/12 — often comparing two out-of-support prompts. read_length is the strongest additional magnitude covariate *and* the one the eval extrapolates on for 75% of targets.

**B8. `T_RWPE2` cannot supply imputation supervision** (1 available assay; masker skips `num_available ≤ 1`), yet it is the only within-assay run_type contrast. Fixing the sampling weight and preserving run_type variance are in direct tension.

**B9. Eval `x_data == y_data`**, so nothing in the current suite measures denoising as distinct from autoencoding. Not fixable by architecture — it is an eval-configuration property (`dsf_sampling='off'`, `apply_mask=False`).

---

## 5. ALREADY-SETTLED — do not re-propose

**A1. Across-assay pooling of decoder metadata is settled-bad.** q16/h34: pooling costs ~25× steering; per-assay is necessary. q19 is already per-assay (`pool_meta=False`). Do not re-litigate. **Corollary that IS still open:** production still pools (`candi_v2/decoder.py:208,490`), so any q19 fix must be ported together with de-pooling or it will not transfer.

**A2. `decoder.film_mode = per_deconv_layer` was trained and rejected.** `sandbox/autoresearch/june3/ar_loop.py:35` sets it programmatically; `toexplore.md:335` records `no_gain (-0.788, +1008 params): over-conditions decoder`; `toexplore.md:38` records the resulting lock. Zero YAML in the repo selects it.
**⚠ Read the scope before treating this as closed:** it was measured **once**, on the **shared trunk**, with **pooled** metadata, on a **single scalar**. Per-deconv-layer FiLM with *per-assay* metadata is a different intervention and is **not** settled. Do not cite −0.788 against it. *(Correction to the handoff, which states at :66-67 and :250 that production builds and passes `PerDeconvLayerFiLM` — it does not under any shipped config.)*

**A3. The offset ON/OFF binary itself.** Both arms are run and recorded at full coverage. Re-running the same two arms adds nothing; the informative arms are the ones *between* them (S12) and the ones that change the optimizer/init (S2).

**A4. The v1 output-steering null is explained** (pooling, ~25×; q16 RESOLVED). Not an open mystery.

**A5. `nb_crps` is exonerated.** Verified to ≤2.4e-5 relative against exact discrete sums and 60-digit mpmath across the full q19 (n, μ, y) hull, and exact in the degenerate baseline limit. If the readout distorts the story, the distortion is upstream of the primitive. *(Two latent caveats, both inactive: the formula mis-indexes at non-integer `y` — up to 20% error — safe only because h5 counts are int16; and `np.maximum(y, 0)` at `metrics.py:55` would silently score a −1 sentinel as 0, currently prevented by upstream filtering.)*

**A6. Production is not running the q19 optimizer.** The real production trainer is `train.py`, defaulting to Adamax with `weight_decay=0.0`. **But do not conclude the mechanism is absent there**: `--optimizer adamw` resolves to `wd=0.01` (decoupled, strictly more annihilating), and `wd=0` does not produce a *live* pathway — it produces one **frozen bit-exactly at random init**, which is equally non-functional. Production also carries the zero-task-gradient half: `candi_v2/config.py:119` defaults `count_head="depth_offset"` and `decoder.py:120` is the same `(d − depth_center) + eta`. Production swaps annihilation-to-denormal for freezing-at-init. *(Also: production Adamax passes no `eps`, so it runs at 1e-8, not the 1e-3 sandbox dataclass default.)*

---

## 6. AUDITOR CONTRADICTIONS — adjudicated

**6.1 Is the offset coefficient misspecified (1.24–1.31) or correct (1.0)?**
*decoder-path + data-identifiability:* between-biosample slope is 1.24–1.31, so slope-1 is misspecified by 25–30%; recommend a free β.
*verifier + my measurement:* depth-only 1.274 ± 0.062, but **0.975 ± 0.064 once `log2(read_length)` enters**; T_-only 1.007 ± 0.076; chr19-train 0.976 ± 0.108 **[RV]**.
→ **Verifier wins decisively.** The marginal excess is a read_length confound, not a depth-exponent effect. **The lever is S7 (add read_length to the exposure term), not "learn β on depth".** Keep S12's free β as a cheap safety valve and a continuous ON↔OFF interpolation, but expect β ≈ 1 and do not sell it as the fix. Also note the original supporting statistics (residual sd 0.471, R² 0.653, corr 0.799) do not reproduce on full data (0.427, 0.741, 0.861) and appear to come from a small window subsample; and the headline "65% by read_length alone" mixed a univariate corr with a 3-predictor R².

**6.2 Is the dispersion head depth-blind by construction?**
*decoder-path:* yes, and correctly so — thinning preserves `n`.
*prior-art verifier + my probe:* no. `raw_n = head_n(feat)` and `feat` is FiLM-modulated by `y_meta` row 0 = depth. Measured: offoff `n` moves +14.5% over depth 22→28; main is frozen at **exactly 0.000e+00** **[RV]**.
→ **Verifier wins.** `n` *is* depth-conditioned architecturally; it is frozen in the ON arm only because the embedder died. The thinning argument (`n` invariant under thinning) remains correct for the **DSF** component and is why S11 is framed as *adding a route the offset cannot shortcut*, not as *fixing a bug*.

**6.3 Is the obs:imp loss up-weighting 3–6× (handoff §10.4) or ~1.07×?**
→ **Metrics/data auditors win** (measured: `num_to_mask ~ U{1..n−1}` ⇒ E[obs]=2.70, E[imp]=2.50; simulated pooled ratio 1.071). The handoff's 3–6× is wrong. The consequential fact in that neighbourhood is S13 (20% of batches have imp=0).

**6.4 Is DSF a nested thinning ladder?**
*decoder-path:* yes — `dsf8 ≤ dsf1` elementwise.
*verifier + my check:* the cited test is vacuous (dsf1 is the full set, so *any* subsample passes). `dsf4 ≤ dsf2` is **False** and `dsf8 ≤ dsf4` is **False** for all 10 groups **[RV]**.
→ **Verifier wins.** Independent draws, not a nested ladder. Matters for any pair with `x_dsf ≠ 1`: the relation holds only in expectation, with extra independent Poisson noise.

**6.5 Does offset-OFF beat offset-ON on macro CRPS?**
*metrics auditor:* on the first 150 of 608 units, offoff 0.3184 < main 0.3265, and 7/8 vs 2/8 against the best marginal; one oracle scale per assay collapses the difference (0.2413 vs 0.2436).
*counter:* `shuffle=False` makes that a spatially-contiguous, ~8× lower-signal region block, not a random subsample. Full eval: main 1.4950 vs offoff 1.9023.
→ **Unresolved and currently the weakest load-bearing claim in the audit.** The *oracle-scale decomposition* is the genuinely important idea (S5/S18) and should be re-run on the **full** eval on a compute node before any conclusion is drawn from it. Do not cite the reversal as established.

**6.6 Is production's decoder "strictly weaker" than q19's (handoff §1)?**
→ **Neither. They are different decoders.** Production is stronger on FiLM placement (pre-trunk), on relative metadata capacity (40.3% vs 0.248%), and on init (live vs adaLN-zero); q19 is stronger on granularity (per-assay vs pooled — the axis h34 identified as load-bearing). Production's head is a *tighter* bottleneck (`Linear(8,8)` mixing assays, one channel per assay) than q19's per-assay MLP. Any q19 finding about head capacity is **optimistic** relative to production.

**6.7 Is `read_length` steering absent in both arms?**
*decoder-path:* yes — `frac_direction` 0.472 (ON) / 0.412 (OFF), both at chance.
*my probe:* `read_length_proj` is alive in both checkpoints, and a 36→101 flip moves η by 3.7e-03 (ON) / 7.97 (OFF) **[RV]**.
→ **Not a contradiction — a metric failure.** The weights are live; the M2 read_length test is broken by OOD flips (7/12), sign-blind CIs (offoff overall is significantly *anti*-directional), and foreground saturation. Do not read 0.472/0.412 as evidence about the model.

---

## 7. RESIDUAL OPEN QUESTIONS, ranked by (decisiveness / cost)

1. **Does offset-ON with `weight_decay=0` (or AdamW + no-decay group on the metadata pathway) keep the decoder embedder alive — and if so, does run_type/assay responsiveness return without costing CRPS?** One 85-min run, one CLI change. This single control discriminates "unbreakable Pareto frontier" from "optimizer artifact" and has never been run. **Highest value in the register.** Pre-commit to logging per-field **absmax** (not nnz) per epoch and (γ,β) effective rank.
2. **Full-eval oracle-per-assay-scale decomposition** (§6.5). Needs a compute node; settles whether the magnitude gap is a fixable 1-dof-per-assay scale error or a genuine capability gap.
3. **Does a learned β (init 1.0, no decay) converge to ~1.0 and hold?** If yes, the frontier is optimization/extrapolation. If it converges toward the data's marginal 1.25 *and* improves CRPS, re-check against S7 first — it is probably absorbing read_length.
4. **Does adding read_length to the exposure term improve out-of-sample CRPS on the 12 targets?** Only in-sample dof-adjusted fit was measured (0.263 → 0.170). With 4 labs / 7 platforms in a 26-record training set, held-out records may land in n=1 cells.
5. **Does conditioning dropout revive `depth_missing_emb` / `readlen_missing_emb`** (currently denormal in *both* arms)? A clean, cheap, falsifiable side-prediction of S15.
6. **Does the production topology exhibit the annihilation at all?** No production checkpoint was inspected by anyone. If production's decoder embedder is healthy, the q19 finding is a fork artifact and the h45 scope question resolves differently.
7. **Is a re-selected 5–8 biosample panel available** that breaks run_type ⊥ (assay, read_length), gives ≥2 assays with >2 log2 within-assay depth range at matched read_length, and preserves the 12-target structure? A small combinatorial search over 363 experiments / 89 biosamples. **Highest-leverage data-side fix; requires no architecture change.** It is the only thing that can lift B1.
8. **Has the assay permutation ever affected a numeric result** rather than only labels? Training is index-only so probably not, but `chr21_umap.py:49`, any per-assay loss weighting, and any name-selected prior/threshold were not audited.
9. **Whether `--dsf-sampling x_eq_y` is reachable from the CLI without other code changes** — it would remove the slope-1 attractor and isolate what the model learns about natural depth.

---

## 8. RE-VERIFICATION LOG (this consolidation pass)

Environment: `cd /project/6014832/mforooz/EpiDenoise && source candi_venv/bin/activate && export PYTHONNOUSERSITE=1`, CPU only. Read-only on the repo; no files created, edited or deleted; no SLURM/GPU.

| # | check | result |
|---|---|---|
| 1 | `torch.load` both `*_perassay.ckpt`, absmax of all `decoder.meta_embedding.*` / `film_proj.*` | §1.1 table reproduced exactly, incl. bit-identical cloze/missing across arms |
| 2 | h5 ↔ `data/eic_metadata.csv` full-triple join under both assay orders | **38/38 handler, 5/38 SANDBOX_ASSAYS** |
| 3 | `grep -n SANDBOX_ASSAYS metrics_real.py` | labelling at `:244, :336, :420`; import at `:31` |
| 4 | `n_fg` distribution over 1215 M2 records | min 62, med 90, max 3072; mean frac 0.2394; frac saturated 0.2156; saturated mass 0.9008 |
| 5 | `eta_slope` / `frac_min_at_true` / `null` / run_type from both JSONs | §1.3 reproduced exactly |
| 6 | full-data assay-FE regression of log2(mean count) on depth ± log2(read_length) ± run_type, 3 window sets × 2 subsets | §2.3 table; depth → 0.975/1.007 with read_length |
| 7 | assay-centered covariate correlations | 0.763 / 0.590 / 0.697 |
| 8 | DSF within-unit slope + depth bookkeeping + nesting, all 38 units | 0.9999 ± 0.0009; max dev 2.44e-03; `dsf4≤dsf2` **False** |
| 9 | told-depth distribution T_ × all DSF | mean 23.227, sd 1.841, [19.20, 28.13]; dsf1-all-views 25.117 |
| 10 | run_type conditional entropies, T_ n=26 | 0.7063 / 0.1248 / **0.0000**; only col3 has both levels |
| 11 | decoder-only probe, both trained arms, per-covariate sweeps | §1.2 table |
| 12 | `_marginal_crps` source + per-assay `median_target` | 4/8 assays median 0 → marg_crps = target mean |

---

**Bottom line for the critique phase.** The recorded ON/OFF result is not a Pareto frontier between two working models. It is: (i) a decoder whose metadata pathway was deleted by a known optimizer/init interaction, leaving it **bit-exactly blind to assay identity and run_type** while its depth response remains arithmetically perfect; measured against (ii) a metric that scores *where* the depth response lives rather than whether it is correct, on a foreground mask that degenerated to background, with position-level CIs that do not survive clustering, against a baseline that is "predict zero" for half the assays, under per-assay labels that are permuted. The PI's thesis is not refuted by this evidence — it has not yet been tested. The cheapest decisive test is §7.1.
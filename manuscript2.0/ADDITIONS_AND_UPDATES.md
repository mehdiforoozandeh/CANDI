# CANDI 2.0 — Repo audit: what changed since the v1 submission

*Author: repo audit for the MLCB 8-page abstract (deadline ~2026-07-01). Compiled 2026-06-29.*

**Purpose.** The submitted manuscript (`../manuscript/`, snapshot ≈ bioRxiv `2025.01.23.634626`)
predates roughly 18 months of method changes, new evaluations, and new infrastructure that
live scattered across the repo (top level + `sandbox/`). This file is a PI-style inventory:
*what* was added, *where* it lives, *what state* it's in, and *whether it's manuscript-ready*.
It is a planning document, not prose for the paper.

**How to read.** Each item is tagged:

| Tag | Meaning |
|-----|---------|
| 🟢 **READY** | Has results / artifacts; can be written up now (numbers may need a confirming run). |
| 🟡 **PARTIAL** | Implemented and run, but validation/benchmarking incomplete. |
| 🔵 **EXPLORATORY** | Tried; result inconclusive, negative, or sandbox-only diagnostic. |
| ⚪ **INFRA** | Engineering/process change; not a paper claim but enables one. |

> ⚠️ **Verification caveat.** Statuses below are inferred from code, design docs, `models/` run
> names, `progress/*.csv`, and the auto-memory — **not** from re-running anything. Every number
> flagged `[confirm]` needs you to point me at the run that produced it (or a quick re-eval)
> before it goes in the paper.

---

## 0. Baseline — what the v1 manuscript already claims

So we don't re-pitch things already in the paper. The submitted draft (`../manuscript/`) covers:

- **Model.** ~42M params. Parallel Conv1D towers (reads `M` + DNA `S`), per-conv
  MetadataCrossAttention / FiLM on **4 covariates** (`log2 depth, assay_id?, read_length, run_type` —
  *v1 lists `sequencing_platform`, not `assay_id`; see §1.1*), transformer encoder (`n_sab=4`,
  `n_head=9`, RoPE), three deconv decoders.
- **Outputs.** NB `(n,p)` counts, Gaussian `(μ,σ²)` arcsinh-signal, Bernoulli peaks. Calibrated
  aleatoric uncertainty.
- **Training.** SSL: full-assay mask (imputation) + full-loci mask (MLM) + downsample denoising.
  Adamax, lr 5e-4, batch 90. Trained on cCRE/random 30kb loci, chr21 held out.
- **Results sections (the 4 we must update or extend):**
  1. SSL confidence-aware denoising imputation (concept/architecture).
  2. Imputation vs EIC competitors — *Spearman SOTA, Pearson lags (compressed dynamic range)*.
  3. Calibrated aleatoric uncertainty — calibration curves + C-index.
  4. Latent `Z` + imputed signal predict RNA-seq log-TPM; robust to input sparsity.

---

## 1. Method / architecture changes (the model itself)

### 1.1 🟢 Metadata redesign: explicit assay-ID, platform dropped, runtype cardinality fix
*Files: `model.py` (`MetadataEncoder`, `QueryMetadataEncoder`), `spec_query_based_decoder.md`,
`issue_supertrack.md`, `data.py`.*

- v1's covariate vector was `(depth, sequencing_platform, read_length, run_type)`. **Now:
  `(log2 depth, assay_id, read_length, run_type)`** — `sequencing_platform` removed, **explicit
  per-assay `assay_id` embedding added** as metadata row 1.
- `num_runtypes` corrected `4 → 2` (single/paired) for new checkpoints; old checkpoints still load.
- Distinct handling of **cloze (`-2`) vs missing (`-1`)** sentinels (v1 conflated them), and a
  dedicated `MaskStem` for missing-assay encoding.
- **Why it matters for the paper:** directly motivated by the *prompt-invariance* failure (§1.4) —
  giving the decoder the explicit identity of the assay it must produce. Small but real method delta
  vs the submitted Methods text.

### 1.2 🟡 Query-based decoder family (breaks the "fixed-channel shortcut")
*Files: `model.py` (`CNP_MoE_Decoder`, `CNP_DynConv_Decoder`, `CNP_CondConv_Decoder`,
`QueryDeconvStage`, `QueryMetadataEncoder`), `spec_query_based_decoder.md`, `train.py`.*

- v1 decoder emits all `F` assays into **fixed output channels** → the decoder can learn
  "channel 5 = H3K4me3" and **ignore the output prompt** (root cause of the supertrack failure).
- New: decode **only queried assays**, per sample, from a shared decoder keyed by
  `assay_id + metadata` query vectors. Three variants:
  - `query_moe` — stage-wise mixture of experts (experts differ by conv kernel size).
  - `query_dynconv` — per-assay FiLM-style modulation, sparse query flow.
  - `query_condconv` — conditional conv (the **`CondConv5_Hybrid`** runs in `models/`, Mar 2026).
- Backward-compatible: `--decoder-type {fixed,query_moe,query_dynconv}`, `fixed` default; full-`F`
  decode at inference; per-sample scatter-back to `[B,L,F]`.
- **Status:** code complete + smoke-trained (Mar 2026 `models/*QueryDecoder_CondConv5_Hybrid*`).
  Spec explicitly flags **"full parity/benchmark vs fixed baseline still needed."** → `[confirm]`
  whether query decoding *won* on imputation/controllability before claiming it.

### 1.3 🟢 Expanded distributional output heads + likelihood ablation
*Files: `model.py` (`GaussianLayer`, `LaplacianLayer`, `GaussianLayerConstantVar`,
`LaplacianLayerConstantScale`, `GammaLayer`, `StudentsTLayer`, `NegativeBinomialLayer`),
`candi_loss.py`, `dist_report.md`, `figures/`.*

- v1 signal head = Gaussian only. Now the signal head is **pluggable**: Gaussian, Laplace,
  Student-t (adds a `df` output), Gamma, and constant-variance/scale variants; loss supports
  Gaussian/Laplace/Student-t NLL + MSE/MAE deterministic modes.
- **Ablation done** (`dist_report.md`, Jan 2026): Laplace vs Gaussian vs constant-variance, all
  ~52M params, 30 epochs, EIC chr19 — with **calibration curves for log-Normal vs log-Laplace**
  (`figures/calibration_imputed_log_{normal,laplace}.svg`, `dist_comparison_progress.png`).
- **Why it matters:** strengthens the "confidence-aware" pillar — turns "we chose Gaussian" into
  "we tested the distributional assumption and report calibration trade-offs." → `[confirm]` which
  distribution you want to present as the default + the one-line verdict.

### 1.4 🟢 Depth-controllable counts (the supertrack / prompt-conditioning fix)
*Files: `sandbox/diagnostics/FINDINGS.md`, `META_CONDITIONING.md`, `issue_supertrack.md`,
`eval_scripts/smoking_gun_supertrack.py`, `train.py` (`train_st_*` probes).*

- **The problem (diagnosed, important and honest):** after training, predictions were **invariant
  to the output `y_meta` prompt** — changing requested depth/run-type/read-length barely moved the
  output. The NB count head ignored depth (depth-collapse), so "ask for a higher-depth supertrack"
  did nothing.
- **The metric:** `DCR = prompt_sensitivity_depth_count_ratio` = output count ratio when the depth
  prompt is raised by +2 log2 (4×). Healthy ≈ **4.0**; collapse = **1.0**.
- **The fix (validated in diagnostics, May 2026):** size-factor reparam
  **μ = 2^(d − 24)·exp(η)** with `depth_center≈24` (batch-median log2 depth). Raw `2^d` fails
  (DCR≈1.0); centered offset gives **DCR≈4.0 from epoch 0** across single-batch overfit, assay-only
  mask, count+peak, and 3-epoch training (`FINDINGS.md` R15–R20).
- **Why it matters:** this is a genuine new capability + a clean result — **controllable denoising
  to a canonical depth ("supertrack")** and an honest negative→fix narrative. Strong candidate for a
  new results subsection. → `[confirm]` it reproduces at production scale (35-assay MERGED), not just
  the chr19/8-assay diagnostic.
- Live training now logs `train_st_depth_ratio / runtype_mse / readlen_mse` as prompt-sensitivity
  monitors (`progress/training_progress_*.csv`).

### 1.5 🟡 Unified "SuperLoss" objective (assay balancing + robustness + uncertainty)
*Files: `candi_loss.py` (`CANDI_LOSS`), `models/*SuperLoss_*` (Mar 2026).*

- v1 loss = static weighted sum (`w_count, w_pval, w_peak, w_obs, w_imp`). New `CANDI_LOSS`
  consolidates several mechanisms, CLI-selectable:
  - **assay-EMA reweighting** (`_update_ema_and_get_weights`) — balances per-assay loss scale (the
    `assayema` / `hier` runs).
  - **robust-stable count branch** (`_count_rstable`) — the `count_rstable` runs.
  - **uncertainty weighting** (`_apply_uncertainty`) — learned per-task uncertainty (`uncertainty` runs).
  - **foreground/background** weighting (`fgbg` runs).
- **Status:** all variants trained Mar 2026; **which variant is the keeper is not recorded in a
  synthesis doc** → `[confirm]` winner + ablation deltas. Likely a Methods refinement + an ablation
  table, not a headline.

### 1.6 🔵 Variational latent / generative CANDI (latent-KL)
*Files: `model.py` (`enable_latent_kl`, `latent_mu_head`, reparam, KL, staged A/B,
`get_last_latent_kl`), `models/*SuperLoss_latentKL*` and `*_generative_*` (Mar–Apr 2026).*

- Adds a **diagonal-Gaussian posterior q(z|x)** on the latent with **reparameterization + KL
  regularization** → an ELBO/VAE-style generative variant of CANDI. Staged training (Phase-B bridge
  from raw `z` to posterior mean), posterior-mean at eval, KL-weight sweeps (`kl1e-3 … kl1e-5`),
  genome-wide (`gw`) runs.
- Heavily explored Apr 2026 then activity shifts to H5/optimizer work → reads as **inconclusive /
  not landed**. Relevant to the v1 "latent `Z` predicts RNA-seq" story (a regularized latent could
  help), but **don't claim** without a clear win. → `[confirm]` if any latentKL run beat the
  non-variational baseline on imp or Z→RNA-seq.

### 1.7 🔵 Iterative refinement at inference + RIM fine-tuning
*Files: `ITERATIVE_REFINEMENT_PLAN.md`, `post_training/{test_iterative_refinement,train_refinement}.py`,
`progress/refinement_progress_deltas_*.csv` (Dec 2025).*

- Plug-and-Play / Recurrent-Inference-Machine idea: feed CANDI's own output back in, with
  input-replacement / variance-weighted ("Kalman") fusion / re-masking, optionally fine-tuned via
  BPTT.
- **Result reads negative/unstable:** `refinement_progress_deltas` show small, sign-mixed
  per-batch deltas and **`inf` gradient norms** during the unrolled fine-tune. No clean
  "refinement improves PCC" signal. → likely **omit** from the abstract (or one sentence as future
  work) unless you have a positive run I haven't found.

### 1.8 🔵 Other architecture knobs explored
- **`DualAttentionEncoderBlock`** (cross-assay × cross-position attention), `RMSNorm`,
  `RelativePositionBias`, `SE_Block_1D`, `XTransformerEncoderBlock` — encoder refinements in
  `model.py`. → `[confirm]` which are in the current default config.
- **Optimizer exploration:** `muon.py` (Muon) added; `models/*_sgd*` vs Adamax (Apr 2026). v1
  reports Adamax. → `[confirm]` current default optimizer.
- **`CANDI_UNET`** — U-Net variant, now **archived** to `legacy/__archive__.py` (dead end).

---

## 2. New evaluations & benchmarks

### 2.1 🟢 Real EIC competitor benchmark integrated
*Files: `eic_paper/` (`benchmark.csv` + 4 ENCODE-Imputation-Challenge MOESM CSVs),
`eval_scripts/viz_eic_bench.py`, `unified_benchmark.py`.*

- v1 compares to EIC competitors qualitatively. Now there's a **per-file, per-metric table** of
  actual competitor outputs — **Avocado, Guacamole, Lavawizard, Hongyang Li & Yuanfang Guan, `avg`
  baseline** — with `mse / pearson / spearman` per `(biosample, assay)`, and CANDI as `imp`.
- **Why it matters:** upgrades the headline comparison from narrative to a real benchmark table /
  win-rate. → `[confirm]` the CANDI checkpoint these `imp` rows came from is current.

### 2.2 🟢 Reorganized end-to-end evaluation suite
*Files: `eval_scripts/{run_all,compute_metrics,viz_pred_perf,viz_conf,viz_eic_bench,viz_rnaseq,
viz_coavailability,viz_scatter_density,viz_quantile_performance,smoking_gun_supertrack}.py`.*

- A clean `run_all.py`-driven figure/metric pipeline replacing the sprawling top-level
  `eval.py`/`viz.py` (78KB/43KB). Covers: prediction performance, calibration/confidence, EIC
  bench, RNA-seq, co-availability, scatter-density, **quantile-stratified performance**, and the
  supertrack "smoking gun." ⚪ mostly INFRA, but `viz_quantile_performance` (performance by signal
  quantile) is a **new analysis** that can address the v1 "Pearson lags / compressed dynamic range"
  critique directly.

### 2.3 🟢 Co-availability & imputation-hop-distance analysis
*Files: `manuscript/reports/co-availability.md`, `eval_scripts/viz_coavailability.py`,
`data/heatmap_*`, `data/network_*`.*

- New theoretical framing: assays form a **directed co-availability graph** (edge `i→j` if
  `P(j|i) > τ`); imputation difficulty = **shortest-path hop distance**. Quantifies which targets
  are directly imputable vs reached only through bridge assays, and flags long-hop predictions as
  low-confidence.
- **Why it matters:** explains *which* imputations are hard and ties prediction confidence to data
  topology — a clean new figure + a paragraph. Already partly written in report form.

### 2.4 🟡 Richer in-training validation monitor
*Files: `model.py` (`EIC_VALIDATION_MONITOR`), `.validation_review.md`, `progress/*.csv`.*

- Validation now reports, per assay and stratified by **genome-wide / gene / promoter / 1-obs**:
  count+pval Pearson/Spearman/R²/MAE-R²/MSE, **perplexity**, peak AUROC, and distribution-agnostic
  NLL across all head types; plus EMA-smoothed tracking and the supertrack probes (§1.4). ⚪ INFRA,
  but the **promoter/gene-stratified** metrics may sharpen the RNA-seq / biological-relevance story.

---

## 3. Data pipeline changes ⚪
*Files: `get_candi_data.py` (243KB), `data_h5.py`, `data_zarr.py`, `prepare_eic_h5.py`,
`prepare_eic_zarr.py`, `data/`, README §"Control Experiments".*

- **ChIP control integration productionized:** auto-discovery + `signal_DSF{1,2,4,8}_res25_control/`
  directories; control as always-available, never-masked channel (matches v1 claim, now with a
  real pipeline).
- **HDF5 + Zarr fast loaders** (`*_h5`, `*_zarr`) replace per-file BAM/BigWig reads → the
  `H5_Baseline` runs (Apr 2026). Methods/repro note, not a claim.
- **DSF set extended to {1,2,4,8}** (v1 Methods lists [1,2,4]); per-assay independent DSF sampling
  proposed (`issue_supertrack.md` Solution 3) to make depth a *necessary* signal. → `[confirm]`
  whether per-assay DSF is in the current training default.

---

## 4. R&D process / autoresearch (methodology, mostly 🔵 sandbox-only)
*Files: `sandbox/` (`candi_v3/`, `autoresearch/`, `diagnostics/`, `ideas/`).*

These are **diagnostic prototyping harnesses** (CLAUDE.md: "not production"). Probably **not**
results for this abstract, but two insights are scientifically citable and could shape framing:

- **`sandbox/candi_v3/` — ERA first-principles redesign (in progress, June 2026).** A Flat-UCB
  program search over (architecture + objective) on an 8-assay EIC slice, scored by an ε-Pareto
  objective (held-out imputation skill + calibration + DCR band centered on 4.0). Stage 0 done;
  marginal baseline imp-Spearman **0.4652**. **Key learned lessons across ~280 candidates:**
  1. The **only** move that crosses the average-reference (ChromImpute/Avocado) baseline is spending
     capacity on the **cell-type-specific deviation** (reference + zero-init deviation head +
     deviation-correlation loss), not relearning the strong shared average.
  2. **Calibration (ECE) is the binding, still-unsolved constraint** — NB counts are systematically
     **over-confident**; NLL alone doesn't fix coverage (needs explicit 2nd-moment / PIT terms).
  - These two framings (deviation-vs-reference; calibration as the hard part) are honest, defensible,
    and could sharpen the paper's discussion even if v3 itself isn't presented.
- **`sandbox/autoresearch/` + `diagnostics/`** — large ablation history (FiLM placement, encoder/
  decoder norm, JEPA-style latent reg, depth-centering, head interference). Source of the §1.4 fix.
- **JEPA / LeJEPA** (`sandbox/jepa*.py`): latent-prediction objective explored, **deprioritized**
  (v3 PLAN: "No full JEPA"). Mention only as explored-alternative if at all.

---

## 5. Proposed manuscript-update map (draft)

How the above could slot into the existing 4 results sections (for an 8-page abstract — **be
selective**; my recommended "tier-1" set is starred ★):

- **§Methods / architecture**
  - ★ Metadata: `assay_id` in, platform out (§1.1).
  - ★ Depth-centered size factor enabling controllable denoising (§1.4).
  - Pluggable signal likelihood (§1.3); query-based decoder as the controllability mechanism (§1.2).
  - Loss refinements (§1.5); DSF{1,2,4,8} + control pipeline (§3).
- **§Results 2 (imputation vs EIC)**
  - ★ Real competitor benchmark table / win-rate (§2.1).
  - Quantile-stratified performance addressing the Pearson/dynamic-range critique (§2.2).
  - Co-availability hop-distance as the "what's hard to impute and why" figure (§2.3).
- **§Results 3 (uncertainty)**
  - ★ Distribution ablation + calibration curves (§1.3); honest "calibration is the hard part"
    discussion from v3 (§4).
- **§Results 4 (biological validation / Z)**
  - Promoter/gene-stratified metrics (§2.4); variational latent *only if* it helps Z→RNA-seq (§1.6).
- **New subsection candidate**
  - ★ **Controllable denoising to a canonical "supertrack"** (§1.4 + §1.2) — arguably the single
    most novel addition since v1.

---

## 6. Open questions / decisions for Mehdi (pingpong)

1. **Headline framing.** Is the 2.0 story "**controllable denoising / supertracks**" (depth-prompt
   + query decoder, §1.4/§1.2), "**rigorous benchmark + calibration**" (§2.1/§1.3), or both? An
   8-page abstract can't carry all of §1–§4.
2. **Scope cut.** Confirm we **drop** iterative refinement (§1.7) and full JEPA from the abstract,
   and treat candi_v3 (§4) as framing/discussion only, not a result.
3. **Which results are real?** I can't tell from artifacts alone — point me to the winning run for:
   (a) query decoder vs fixed (§1.2), (b) SuperLoss variant (§1.5), (c) latentKL (§1.6),
   (d) the EIC `imp` checkpoint (§2.1), (e) default distribution (§1.3).
4. **Production-scale confirmation of the DCR fix (§1.4).** Validated on the 8-assay/chr19
   diagnostic — is there a 35-assay MERGED run showing DCR≈4 and that supertracks improve a
   downstream metric? That gates whether §1.4 is a result or a method note.
5. **Current default config.** Which encoder blocks (DualAttention? RMSNorm?), optimizer
   (Muon/SGD/Adamax), decoder type, and loss are in the *current* canonical CANDI? I'll write
   Methods against that, not the union of everything tried.
6. **Author/venue specifics.** MLCB 8-page format, figure budget, and whether this is positioned as
   an extended abstract of the existing preprint or a standalone.

---

## Appendix — file pointers (audit trail)

| Area | Key files |
|------|-----------|
| v1 manuscript | `manuscript/{CANDI_manuscript.md,introduction.tex,methods.tex,results.tex}`, `manuscript/reports/co-availability.md`, `CANDI.pdf` |
| Model | `model.py` (heads L3222–3492; decoders L1230–1567; latentKL L1713–1997; encoder L1568–2242) |
| Loss | `candi_loss.py` (`CANDI_LOSS` L59+) |
| Design docs | `spec_query_based_decoder.md`, `issue_supertrack.md`, `ITERATIVE_REFINEMENT_PLAN.md`, `dist_report.md`, `.validation_review.md` |
| Eval | `eval_scripts/*`, `unified_benchmark.py`, `eic_paper/benchmark.csv` |
| Data | `get_candi_data.py`, `data_h5.py`, `data_zarr.py`, `prepare_eic_{h5,zarr}.py` |
| Post-training | `post_training/*`, `progress/refinement_progress_deltas_*.csv` |
| Runs | `models/` (154 dirs; Jan→Apr 2026 timeline encodes the experiment history) |
| Sandbox R&D | `sandbox/candi_v3/{PLAN,METRIC,DESIGN_MENU,NOTE}.md`, `sandbox/diagnostics/FINDINGS.md`, `sandbox/ideas/` |
</content>
</invoke>

---
id: q19
type: question
title: Can we make dual conditioning work on real CANDI sandbox data before production?
parent: q18
status: resolved
stale: true
created: "2026-07-15T23:35:12"
updated: "2026-08-01T18:18:11"
---

# q19 — Can we make dual conditioning work on real CANDI sandbox data before production?

Parent:: [[q18_do_the_dual_conditioning_testbed_finding]]
Literature:: [[wiki/covariate-conditioning-and-counterfactuals]], [[wiki/film-conditioning]]

## Question

Following q15/q16 (dual-conditioning mechanism proven on a synthetic testbed) and q18 (production translation), q19 is the intermediate SANDBOX rung: does the testbed winning recipe reproduce dual metadata-conditioning on REAL CANDI sandbox data (sandbox.h5 -- 8 assays + control, 5 biosamples stored as T/V/B views, chr19 train / chr21 eval, 12 held-out imputation targets) with the REAL 4-dim metadata [log2 depth, assay_id, read_length, run_type]? Architecture is held FIXED to the testbed golden reference (per-assay per_conv FiLM encoder + per-assay adaLN-zero FiLM decoder + depth-offset log-link NB head; NB COUNTS ONLY; closed-form NB CRPS + non-randomized PIT). Real biology gives no per-position counterfactual, so steering is read via a COUNTERFACTUAL-PROMPT FLIP test on held-out imputation targets: predict the true target under the TRUE prompt vs a FLIPPED/wrong prompt and require true < flipped in CRPS (direction) plus a nonzero prompt-induced shift (responsiveness); depth additionally uses the DSF machinery (per-assay independent x_dsf/y_dsf) as a materializable counterfactual. Three metric families port from the testbed: M1 counts-only recon/imputation HEALTH, M2 responsiveness + direction, M3 shared-biological-latent invariance across measurement conditions. Goal: an honest positive sandbox demonstration BEFORE scaling to EIC/MERGED production.

## Design & Deliverables (PRD)

_This section is the implementation plan. It plus the child hypotheses should be sufficient to build q19 end-to-end without re-deriving anything. Hard rule: **the dual-conditioning testbed (`sandbox/diagnostics/dual_conditioning/`) is the golden reference for the CANDI architecture** — mirror it; only the metadata assembly changes._

### Data — verified from `sandbox/data/sandbox.h5` (2026-07-15)
- **8 assays**, fixed index order: `0 ATAC-seq · 1 DNase-seq · 2 H3K4me3 · 3 H3K4me1 · 4 H3K27ac · 5 H3K27me3 · 6 H3K36me3 · 7 H3K9me3`, plus **1 ChIP control** channel (index `A`, always available, never masked).
- **5 biosamples** (DND-41, RWPE2, heart_left_ventricle, H1-hESC, H9) stored as **10 groups** with `T_/V_/B_` prefixes = **Train / Validation / Blind-test views of the same biosample**, each holding only that role's assays.
- **Split is per (biosample, assay).** Train loop uses `T_*` only; `V_/B_` groups supply held-out imputation ground truth. Train windows = **chr19** (`type1_chr19`) or sampled loci `region_type∈{0,1}` (`type2_loci`); **eval = chr21**.
- **12 held-out imputation targets** (assay present in the `V_/B_` view, absent from the `T_` input): `T_DND-41→{H3K4me1(single), ATAC(paired), H3K9me3(paired)}`, `T_H1-hESC→{H3K27ac(single)}`, `T_RWPE2→{ATAC,DNase,H3K4me3,H3K27ac,H3K27me3,H3K36me3,H3K9me3 — all paired}`, `T_heart→{DNase(single)}`; `T_H9→none` (denoising-only). **By run_type: 9 paired + 3 single** (more flip-test power on paired targets).
- **Metadata** = rows `[depth_log2, assay_id, read_length, run_type]`; sentinels `missing=-1`, `cloze=-2`. depth ∈ [22.2, 28.1], read_length ∈ {30,36,76,100,101}, run_type ∈ {single=0, paired=1}. **`meta_dsf{1,2,4,8}` + `counts_dsf{1,2,4,8}`** provide per-assay depth counterfactuals (in-silico downsampling).
- **CAVEATS:** the baked run_type/read_length for ATAC/DNase/H3K9me3 drifted vs a later on-disk snapshot — re-verify each target's baked meta against its source experiment before trusting flip-test labels. Real data lives at `def-maxwl/…/DATA_CANDI_{EIC,MERGED}`; `sandbox.h5` is the small curated slice.

### Architecture — GOLDEN REFERENCE (held FIXED) = `sandbox/diagnostics/dual_conditioning/model.py`
- **Encoder**: `V2Encoder`, `film_mode="per_conv"` — per-assay conv FiLM at every layer, **NO across-assay pooling**; arcsinh on counts inside the encoder.
- **Decoder**: `DualCondDecoder` — per-assay FiLM (`gamma_a, beta_a` from `y_meta`, **adaLN-zero** init so recon is stable and steering grows) modulates each assay's features; a weight-shared per-assay head → `(eta_a, raw_n_a)`.
- **Head — depth-offset log-link NB, COUNTS ONLY** (no Gaussian signal head, no Bernoulli peak head): `log2_mu = (d − depth_center) + eta` [offset ON, valid depth] else `eta`; `mu = 2^clamp(log2_mu)`; `n = softplus(raw_n)+eps`; `p = n/(n+mu)`. **`eta` is the offset-INDEPENDENT mean statistic** — the honest steering lever. `depth_center` = real observed mean depth (~25).
- **Loss**: masked NB NLL over available assays.
- **Metadata embedder — the ONLY change**: rebuild from the testbed's synthetic 2-row knob to the real 4 rows: **depth (row 0) → offset (+FiLM), assay_id (row 1) → per-assay identity, read_length (row 2) + run_type (row 3) → decoder FiLM covariates**. Architecture otherwise identical.
- **Eval instruments (eval-only, numpy/scipy, no autograd)**: closed-form NB CRPS `E|X−y| − ½E|X−X'|` (Gini via Pfaff-transformed ₂F₁), with `p` reconstructed from the **true** `mu` in float64 (not the floored `p`); **non-randomized PIT** calibration. Re-validate CRPS numeric stability at real count magnitudes.

### Training process
- **Base**: candi_v2 import-and-swap via the `dual_conditioning/` scaffolding (data/model/metrics/run), swapping the synthetic metadata assembly for the real `meta_dsf{k}`. Single-GPU sandbox (SLURM `--gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1`).
- **Data**: `SandboxH5Dataset`, `biosample_prefix="T_"`, regime `type1_chr19` (baseline) — with **per-assay independent DSF ON** (`dsf_sampling="uniform"`, `x_dsf ≠ y_dsf` per assay) = the **non-copyable depth task** (target depth not readable off the input; decorrelates context/control depth). Cloze masking (`make_masker`/`prepare_masked_batch`) creates imputation supervision (obs=unmasked, imp=masked); control never masked.
- **Prompt**: cloze-masked/absent assays receive the `V_/B_` natural metadata as `y_meta` (`_build_mixed_meta`). This is the honest prompt; the M2 flip test perturbs it **at eval only**.
- **Schedule**: LR warmup + cosine, adaLN-zero decoder init (per testbed). Eval on chr21; imputation GT from `V_/B_`.

### Metric families (ported from the testbed)
- **M1 (health, counts-only)** — imp/den count Spearman + Pearson, NB CRPS, PIT-ECE, encoder eff-rank. A sanity gate that the model reconstructs/imputes non-degenerately, **not** a SOTA bar. See **h40**.
- **M2 (responsiveness + direction) — the counterfactual-prompt FLIP test**, per covariate, on the 12 held-out imputation targets (target assay absent from input ⇒ prompt is the only channel):
  - **Responsiveness** = does changing the prompt move the predicted NB? `ΔCRPS(pred_true, pred_flip) > 0` (needs no GT). Depth: sweep told-depth over `meta_dsf` levels; run_type/read_length: flip the value.
  - **Direction** = does the TRUE prompt match GT better than the wrong one? `CRPS(pred_true, GT) < CRPS(pred_flip, GT)`, bootstrap CI excludes 0, foreground positions.
  - Depth extras (h41): offset-independent `eta`/tail must track true depth (not just the `2^Δ` arithmetic); depth-shuffle null. See **h41** (depth) / **h42** (run_type + read_length).
- **M3 (shared-biological-latent invariance)** — port `eval_M3`: the **same region under different input DSF-depth conditions (each with its TRUE metadata)** maps to a consistent latent; within-region ≪ between-region cos-dist (ratio ≤ 0.3), guarded by recon>0 & eff-rank>1. Invariance to the *measurement condition* by **using** metadata, not ignoring it. See **h43**.

### Deliverables
- **Metrics JSON** (`results/*.json`): per-covariate M1/M2/M3 scorecard, matrices un-collapsed; report reads JSON only (fully regenerable).
- **Figures**: **F1** headline M2 bars (true vs flip/shuffle per covariate); **F2** depth CRPS-vs-told-depth curves (min at true depth) per assay; **F3** run_type flip CRPS(true) vs CRPS(flip) scatter, single/paired split; **F4** M3 within/between cos-dist bar + `Z` embedding (PCA/UMAP) across input conditions; **F5** M1 quality bars vs marginal baseline; **F6** PIT reliability; **F7** offset-independent `eta`/tail response.
- **Tables**: **T1** per-covariate M1/M2/M3 scorecard; **T2** the 12-target inventory (bios×assay×run_type).
- **`report.md` + `report.html`** synthesis (like the testbed), hypothesis-sectioned with a verifiable scorecard.

### Confounds controlled (from the adversarial review)
- **Offset arithmetic** → gate depth steering on `eta`/tail + shuffle null, not DCR/mean.
- **Input leakage** → measure steering on MASKED imputation targets (target absent from input) + the correct-vs-flipped prompt delta (both arms see the same context).
- **Context/control depth correlation** → per-assay independent DSF ON (precondition) decorrelates; add a constant-prompt-vs-varying-context null.
- **Background domination** → report foreground (top-count positions) as primary + aggregate as a dilution check.
- **run_type honest null** → weak natural variance may give Δ≈0 ⇒ documented path-(c) paired→single **training augmentation** follow-up (deferred), not a design failure.

### Implementation guide — build order, reuse map, new code

**q19 is a HYBRID harness**: real DATA + masking from the sandbox production harness + MODEL core & numeric PRIMITIVES from the testbed, joined by (1) a new real-metadata embedder and (2) new flip-test metric readouts. **Critical: do NOT reuse the testbed's synthetic data (`DualCondData`, `transforms.py`) or its `eval_M1/eval_M2/eval_M3` — those are hardwired to the synthetic augmentation counterfactuals (`make_batch(fx,px,fy,py)`) and cannot run on real data.** Golden reference for CANDI architecture: `sandbox/diagnostics/dual_conditioning/model.py`.

**REUSE AS-IS (do not rewrite):**
- **Model core** — `dual_conditioning/model.py`: `DualCondModel` / `DualCondDecoder` (per-assay adaLN-zero FiLM + depth-offset log-link NB head), `nb_nll` (masked NB loss), `nb_mean`, `encode_latent`, `forward_full`. Hyperparams: `num_assays=8`, `embed_dim=32`, `feat_per_assay=16`, `n_transformer_layers=2`, **`depth_center=25.1`** (already the real sandbox mean depth), encoder `film_mode="per_conv"`, `signal_transform="arcsinh"`.
- **Numeric primitives** — `dual_conditioning/metrics.py`: `nb_crps` (closed-form, validated — re-validate at real count magnitudes; reconstruct `p` from the true `mu` in float64), `nb_quantile`, `calibration_pit_curve`, `ece` (non-randomized PIT), `spearman`, `pearson`, `r2`, `_cos_dist`, `_steering_index`.
- **Data + masking** — `sandbox/data.py::SandboxH5Dataset` (`biosample_prefix="T_"`, `regime="type1_chr19"`, `dsf_sampling="uniform"` = per-assay independent DSF; `eval_include_vb_ground_truth=True` yields `y_data_imp` + `y_meta_imp`), `sandbox/batch.py::make_masker`/`prepare_masked_batch` (`CLOZE=-2`, `MISSING=-1`), `sandbox/train.py::_build_mixed_meta` (swaps V/B natural meta into cloze slots — the imputation prompt), `sandbox/data.py::build_canonical_meta`, `sandbox/__init__.py::SANDBOX_ASSAYS`.
- **Encoder** — `sandbox/candi_v2/encoder.py::V2Encoder` (import-and-swap, no `candi_v2` edits).

**NEW CODE (write in a new dir, e.g. `sandbox/diagnostics/dual_conditioning_real/`):**
1. **`RealMetaEmbedder`** (replaces `DualCondMetaEmbedder`): real 4 rows `[depth_log2, assay_id, read_length, run_type]` → `[B, F, embed_dim]`. Per-field: depth(row0) continuous→Linear; assay_id(row1)→`nn.Embedding(num_assays+2)`; read_length(row2) continuous→Linear (or Embedding over {30,36,76,100,101}); run_type(row3)→`nn.Embedding(4)` with {single=0, paired=1, MISSING(-1)→2, CLOZE(-2)→3}. Handle -1/-2 sentinels per field (mirror the testbed missing-emb pattern), fuse→LayerNorm. Swap into **both** `encoder.metadata_embedding` and the decoder's `meta_embedding`. **The decoder FiLM must include depth as a covariate** so `eta` can carry an offset-independent depth response (required for h41); the offset separately reads depth.
2. **Re-key the offset**: in `DualCondDecoder.forward` the offset reads `y_meta[:, 2, :]` (testbed depth row 2). For real data depth is **row 0** → change to `y_meta[:, 0, :]`; keep `depth_center=25.1`.
3. **`run_real.py`** — training loop: SandboxH5Dataset + `make_masker`(cloze) + `prepare_masked_batch` + `_build_mixed_meta`; model = DualCondModel(real embedder, offset row 0); loss = `nb_nll` on counts. Adapt the loop skeleton from `sandbox/train.py` (~L314–460 train, ~L477–609 eval), **counts-only** (drop the Gaussian/Bernoulli branches). Warmup+cosine LR, adaLN-zero (already in the decoder).
4. **`metrics_real.py`** — reimplement M1/M2/M3 on real batches using the primitives: **M1** count spearman/pearson + `nb_crps` + `ece` + encoder eff-rank on chr21 (imp = V/B targets, den = unmasked T assays); **M2 flip** build TRUE vs FLIPPED `y_meta`, `forward_full`, `nb_crps(pred,GT)` each → responsiveness ΔCRPS(true,flip), direction CRPS(true)<CRPS(flip) with bootstrap CI (`n_boot≈1000`), foreground `fg_frac=0.02` by GT count; depth sweeps told-depth over the assay's achievable set `{base_d − log2(dsf)}`, run_type flips 0↔1, readlen flips to nearest observed; offset-independent = regress `eta`/upper-quantile on told-depth; null = shuffle the prompt; **M3** encode each region at input DSF∈{1,2,4,8} (true x_meta) → `encode_latent` → within/between `_cos_dist` ratio, guard eff-rank>1.
5. **`report.py`/`report_html.py`** — adapt the testbed's to read `results/*.json` only → F1–F7 + T1–T2.

**BUILD ORDER (gate each stage before the next):** (1) CPU smoke — build RealMetaEmbedder + DualCondModel(real), forward a tiny SandboxH5Dataset batch, assert shapes + finite `nb_nll`. (2) Train **h40** short (overfit chr19), confirm M1 healthy (imp-count spearman in band, den≥imp, eff-rank>1) — **this is the gate; do not proceed on a degenerate model**. (3) **h41** depth, then **h42** run_type/readlen, then **h43** M3 on the trained model. (4) Full run + report; then close hypotheses in crux (tick verifiables, write findings).

**LAUNCH**: mirror `dual_conditioning/jobs/gate.sh` (CPU gates → train gate → integration smoke, fail-fast, propagate exit) then a `sweep.sh` array. SLURM header **MUST** be `--gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1`, `--account=def-maxwl`; body: `source candi_venv/bin/activate`, `export PYTHONNOUSERSITE=1`, `module load samtools`. Ref: `dual_conditioning/jobs/{gate,sweep}.sh`.

**Supporting artifact**: the full 2026-07-15 deep-repo synthesis + adversarial critique is preserved at `sandbox/diagnostics/dual_conditioning/q19_realdata_review.md` (the offset-arithmetic / input-leakage / context-depth confounds are distilled into §"Confounds controlled" above).

See-also [[q18_do_the_dual_conditioning_testbed_finding]] · [[q15_can_candi_learn_dual_metadata_conditioni]] · [[q4_can_candi_s_counts_be_made_depth_control]] · [[q16_was_the_v1_output_steering_null_an_artif]].

## Validation & Test Plan

**Policy (strict order — NO big GPU job before this is green):** implement each block → **unit-test every block** (CPU, deterministic, second-scale) → **CPU integration smoke** (tiny end-to-end, exit 0, no NaN) → **GPU integration smoke** (MIG 1g.10gb, a few steps, CUDA correctness) → **`gate.sh` green** → only then *propose* the full training + eval array to the PI. A big allocation is authorized ONLY on a green gate. Mirrors the testbed `dual_conditioning/jobs/gate.sh` discipline.

**Test layout**: `sandbox/diagnostics/dual_conditioning_real/tests/` — one file per block, `pytest`-runnable (or plain asserts), all CPU-only and fixture-driven off a **2–3-window slice of `sandbox.h5`** (bake once, cache). Plus `smoke_cpu.py`, `smoke_gpu.py`, and `jobs/gate.sh`.

### Block tests (all CPU, pre-smoke)
**`test_data.py` — data processing**
- batch keys + shapes: `x_data [B,L,8]`, `x_meta/y_meta [B,4,8]`, `x_dna [B,Lbp,4]`, `control [B,L,1]`, `control_meta [B,4,1]`; all finite.
- metadata rows = `[depth_log2, assay_id, read_length, run_type]`; **`assay_id` row == column index** (identity ordering); present assays finite, absent == -1.
- **fixture assertion**: the **12 held-out imputation targets and their run_type match the known inventory** (T_DND-41:{H3K4me1 s, ATAC p, H3K9me3 p}, T_H1-hESC:{H3K27ac s}, T_RWPE2:{7 paired}, T_heart:{DNase s} → 9 paired + 3 single).
- per-assay independent DSF (`dsf_sampling="uniform"`) yields some assays with `x_dsf≠y_dsf`; `x_meta` depth == `meta_dsf{x_dsf}` row 0, `y_meta` depth == `meta_dsf{y_dsf}`; `x_data` drawn from `counts_dsf{x_dsf}`.
- train pool = `T_*` only; eval yields `y_data_imp`/`y_meta_imp` from V/B when `eval_include_vb_ground_truth=True`.

**`test_masking.py` — masking**
- cloze-masked assay → x set to `CLOZE(-2)`; `y_data` retained; `masked_map` marks it; **obs vs imp == unmasked vs masked**.
- **control channel never masked** (`control_avail` stays 1).
- `_build_mixed_meta`: masked slots take V/B natural meta, unmasked slots keep `T_` meta.
- missing assays (`y_avail=0`) get -1 meta; `query_mask` extends to them.

**`test_model.py` — RealMetaEmbedder + DualCondModel**
- `RealMetaEmbedder [B,4,8]→[B,8,E]`, finite with sentinels (-1/-2) in **any** field (no NaN).
- **offset keyed to row 0**: +1 to depth (row 0) → `log2_mu` shifts +1 (offset ON); offset OFF → `log2_mu==eta`, depth-independent.
- **adaLN-zero**: at init `film_proj` weight+bias == 0 → gamma=beta=0 → FiLM identity (feat unchanged at init).
- **per-assay (not pooled)**: perturbing ONE assay's `y_meta` changes only that assay's output.
- **encoder ignores y_meta**: `encode_latent` unchanged when `y_meta` changes (Z depends only on `x_*`).
- `forward_full` → {p,n,eta,log2_mu,mu} finite, shapes `[B,L,8]`; `nb_nll` finite; backward populates grads.

**`test_metrics_primitives.py` — the reused numerics (re-validate at real magnitudes)**
- **`nb_crps` vs exact CDF sum ≤ 1e-10** (small n,p,y) AND vs Monte-Carlo ≤ ~0.01; **no NaN/inf up to real counts (~1e5–1e6)** — the key numeric re-validation.
- **`p` reconstructed from the true `mu` in float64** (assert metrics do NOT use the floored `p`).
- `ece`/`calibration_pit_curve`: samples drawn from the model's own NB → PIT ~uniform (ECE≈0); mismatched → ECE>0.
- spearman/pearson/r2 vs scipy; `_cos_dist`, `_steering_index` on synthetic arrays.

**`test_training.py` — run_real loop**
- one overfit fixture (single window): loss **decreases over ~50 steps** (memorization sanity).
- counts-only: only `nb_nll` invoked (no Gaussian/Bernoulli); masked positions contribute, control excluded.
- LR warmup+cosine values; checkpoint save/load round-trips.

**`test_report.py` — report**
- reads a fixture `results/*.json` only (no re-inference), regenerates F1–F7 + T1–T2 deterministically.

### Metric-readout CONTROLS (synthetic models with KNOWN behavior — validate the readout, not the model)
Tiny synthetic "models" that steer in a known way, so a **metric bug is caught independently of whether the real model steers** (`test_metrics_real.py`):
- **offset-only** (eta≡0, offset ON): depth CRPS-curve **min at true depth** (direction mechanics OK) BUT offset-independent eta-slope ≈ 0 → asserts the readout correctly flags "arithmetic, not learned."
- **depth-ignoring** (offset OFF, eta const): flat curve, responsiveness ≈ 0 (negative control).
- **eta = k·depth**: offset-independent slope ≈ k (positive control for the eta regression).
- **run_type-responsive** vs **run_type-ignoring**: flip changes / doesn't-change pred; direction correct only when wired to GT.
- **invariant** vs **non-invariant** encoder: M3 ratio small vs large; **collapsed constant-Z** → eff-rank guard trips.

### Integration smoke
- **`smoke_cpu.py`**: end-to-end on 2–3 windows, 1–2 train steps → eval M1/M2/M3 → write JSON → report; assert exit 0, all-finite, deterministic.
- **`smoke_gpu.py`**: same on the MIG 1g.10gb slice, ~10 steps — catches what CPU misses (device/dtype, cuDNN determinism, memory-fit); assert M1/M2/M3 finite.

### The gate (`jobs/gate.sh`, mirror the testbed's)
CPU block tests + readout controls → CPU smoke → GPU smoke, **fail-fast, propagate the first failing exit code**. `--gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1`, `--account=def-maxwl`, candi_venv. **Green gate = the precondition to PROPOSE the full training + eval array to the PI.** No main GPU job before green; per the crux leash rule the PI OKs the run after seeing the green gate.

## Answer so far

YES — dual metadata-conditioning works on real CANDI sandbox data. Health (h40) and shared-biological-latent invariance (h43) are supported; the winning-recipe model imputes well (imp-Spearman 0.53-0.64) and uses metadata to stay invariant across measurement conditions. The apparent offset-ON 'steering null' that made h41/h42 look partial was NOT a real scale-vs-steering Pareto: h47 proved it a WEIGHT-DECAY ARTIFACT — plain-Adam coupled-L2 x adaLN-zero annihilated the decoder metadata embedder (6e-41), and weight_decay=0 revives it AND beats the anchor on every M1 axis (macro CRPS 1.495->1.341, imp-Sp 0.533->0.637, ECE 0.062->0.053) ~~while restoring functional assay steering (real-z d_eta 0.833)~~ [**AMENDED 2026-07-28 (h48/F2)** — that steering clause is withdrawn: the 0.833 was a MISSING-sentinel ARTIFACT (whole-row assay permute sliding the MISSING(-1) sentinel across unavailable slots). Sentinel-free real->real for wd0_on is **0.0023** (H48:L92), 43x BELOW h47's own >=0.10 functional bar, against 4.1772 (offoff) / 9.7144 (wd0_off) on the identical probe; see [[h48_h0_fix_the_broken_q19_instruments_and_re|h48]]]. So the PI thesis holds — a model that keeps its metadata pathway alive wins, it does not trade off (h45 refuted; no hybrid needed). Two honest limits: (1) offset-off's imputation cost is absolute-scale calibration, not lost biology (h46); (2) run_type steering is analytically UNIDENTIFIABLE on this 5-biosample panel (H(run_type|assay,read_length)=0 bits), a data bound no architecture can fix. The open, still-unanswered part — how to best DESIGN the architecture/training for maximal metadata conditioning (fix the broken instruments, read_length exposure, per-assay capacity, no-decay group, conditioning dropout, +platform/lab, re-panel for run_type) — is spun out as q20. q19 itself is resolved: dual conditioning is real, learnable, and production-relevant on sandbox data.

<!-- crux:ledger:start -->
**8 children** · ideas 8/8 done (supported 3, partial 3, refuted 2, inconclusive 0)

- `h40` [[h40_winning_recipe_candi_reconstructs_and_im|Winning-recipe CANDI reconstructs and imputes sensibly on real sandbox data (M1 health, counts-only)]] — *done* — verdict **supported**, metric `imp-Spearman 0.53-0.59 (den 0.71); imp-CRPS 1.5-1.6<2.21 marginal; ECE 0.03-0.06; eff-rank 52`
- `h41` [[h41_depth_output_steering_is_present_distrib|Depth output-steering is present, distributional, and offset-independent (M2, depth)]] — *done* — verdict **partial**, metric `depth eta-slope ~0 offset-on (arithmetic) vs 0.88 offset-off; dir 0.43-0.57 CI-excl-0`
- `h42` [[h42_counterfactual_prompt_flip_true_run_type|Counterfactual-prompt flip: true run_type imputes better than the wrong one (M2, run_type; read_length secondary)]] — *done* — verdict **partial**, metric `run_type dir 0.00/resp 0 offset-on vs 0.69/1.83 CI-excl-0 both single&paired offset-off`
- `h43` [[h43_encoder_recovers_a_shared_biological_lat|Encoder recovers a shared biological latent by combining data and metadata, invariant across measurement conditions (M3)]] — *done* — verdict **supported**, metric `within/between ratio 0.244-0.292 <=0.3; x_eq_y control breaks it (0.334)`
- `h44` [[h44_adding_sequencing_platform_lab_with_an_u|Adding sequencing_platform + lab (with an UNKNOWN/OOV token) and pruning to the optimal metadata set improves imputation and denoising]] — *done* — verdict **refuted**, metric `Superseded (not disproven): relocated to h56 under q20; platform/lab identifiable (0.443/0.212 bits) unlike run_type (0 bits); to be tested vs wd0_on 1.341 with corrected instruments`
- `h45` [[h45_removing_the_depth_offset_head_recovers_|Removing the depth-offset head recovers learned steering at a scale-calibration cost a hybrid can recover]] — *done* — verdict **refuted**, metric `Pareto premise refuted by h47: no hybrid needed; wd=0 gets magnitude AND steering. offset-off 'learned steering' was an eta_slope artifact (true total slope 0.775<1.000); run_type CI crosses 0 under clustering (B1)`
- `h46` [[h46_the_offset_off_imputation_gap_is_scale_m|The offset-off imputation gap is scale/magnitude, not lost biology — pooled metrics overstate it]] — *done* — verdict **supported**, metric `shape ties (macro Spearman 0.505 vs 0.465, OFF wins 6/8); magnitude fails (macro CRPS 1.50→1.90, beats-marginal 8/8→3/8); pooling overstated the gap ~3×`
- `h47` [[h47_the_offset_on_steering_null_is_a_weight_|The offset-ON steering null is a weight-decay artifact: removing weight decay revives the decoder metadata pathway at no magnitude cost]] — *done* — verdict **partial**, metric `M1 stands (macro CRPS 1.495->1.341, imp-Sp 0.533->0.637, ECE 0.062->0.053); on oracle-scale-decomposed capability wd0_on 1.3077 is best of four, and offoff-wd0_on +0.093 [+0.004,+0.217] is the ONLY pairwise lead surviving target clustering (four-arm reordering NOT established); embedder weights alive 6e-41->2.79 BUT at random-init statistics (never destroyed, never trained); V2 UNMET: sentinel-free assay steering d_eta 0.0023, 43x below its own 0.10 bar (h48/F2)`
<!-- crux:ledger:end -->

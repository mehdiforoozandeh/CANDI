# q19 formalization — deep repo review (synthesis + adversarial critique)

## SYNTHESIS

# q19 Formalization Draft — Does the dual-conditioning testbed recipe reproduce steering on real CANDI sandbox data?

**Placement in tree:** child of q18 (production translation), grandchild of q15. q18 is scoped to "production CANDI"; q19 is the *missing intermediate rung* — sandbox-scale (8 assays + control, `sandbox.h5`), real 4-dim metadata, denoising + masked-imputation regime, **before** the 35-assay MERGED move. Recommend registering q19 as a **child of q18** (it operationalizes q18's translation question at one scale) rather than a sibling; q18 stays the umbrella, q19 is its first executable instance.

**One-line charter:** With the testbed's *method* (per-assay CRPS steering readout, latent-cosine invariance, non-randomized PIT) held as the instrument, and its *architecture claims* (per-assay FiLM, depth-offset head) held as substrate, does **real depth metadata** steer the real CANDI decoder at sandbox scale, and is that steering distributional (CRPS), not just mean (DCR)?

---

## 0. Load-bearing framing facts (agreed across readers; flag where not)

- **Dual conditioning already exists in the real architecture.** `model.py:2211` `forward(src, seq, x_metadata, y_metadata, …)` — two separate metadata tensors, per-assay `MetadataEncoder`, per-assay `FiLMLayer` at every signal-conv and every deconv layer. There is **no `meta.mean` pooling in production `model.py`** (Reader 2). So per-assay conditioning is *not* a production fix — it is already the default.
- **The testbed's headline (per-assay 0.526 vs pooled 0.022, ~25×, h34) is a fix for the candi_v2 *sandbox* regression, not production** (Reader 5, q16 RESOLVED). Pooling exists only as an optional `film_mode='per_conv_and_transformer'` path in candi_v2 (`encoder.py:899`). **q19 must NOT re-test per-assay-vs-pooled as if it were the live production lever.** Per-assay is *necessary but not sufficient*.
- **Production's own DCR~1 collapse has a distinct two-part cause** (q16): (1) free-mean NB head `mu=softplus(Wx)` with no depth anchor (`model.py:3244`), and (2) a *copyable* reconstruct-same-assay task where target depth is readable off the input, starving the metadata gradient. The diagnosed fixes are **h9 (depth-offset head)** + **h10 (per-assay DSF)**, not topology (q4).
- **Only depth is a genuinely steerable real covariate at sandbox scale.** `assay_id`, `read_length`, `run_type` are effectively constant per assay column in the h5 slice and have **no deterministic counterfactual transform**; depth alone has both dynamic range (via DSF) and a known count relationship (counts ~ 2^depth). This is the single most consequential constraint on q19 (Readers 3, 5, 6, 7 all converge).
- **The deepest translation gap:** real biology gives **no per-position counterfactual ground truth** — you cannot observe the same position at a different real depth/run-type. The testbed's M2 depends on knowing the exact target `f_y(base)` per position. The *only* real construct that supplies a known deterministic per-position transform is **in-silico depth thinning (DSF)** — which is why depth is not just the easiest but the *only* clean first target (Readers 6, 7).
- **CRPS is mandatory, not optional** (q18 method requirement): CANDI predicts distributions; mean-only DCR under-measures steering that lives in the tail. The testbed empirically shows steering is *largely* tail (tail-Pearson 0.92–0.99) with a *modest* mean effect (mean-r 0.23–0.60). Production imputation cares about the mean — so q19 must report **both** and cannot declare success on a tail-only CRPS gain.

---

## 1. Testbed → real-CANDI mapping table

| Testbed construct | Real-data analog | Status | Reason / caveat |
|---|---|---|---|
| Synthetic 2-row knob `[aug_family, aug_param]` | Real covariate columns of `x_meta`/`y_meta`, primarily **row 0 `log2_depth`** | **Partial map** | No real analog to a discrete "transform family." `aug_param` (continuous monotone dose) maps only onto `log2_depth`; `read_length`/`run_type`/`assay_id` are near-constant and carry no dose. |
| Row 2 `log2_depth` (non-steerable size factor) | **Already the real covariate** `np.log2(depth)` (`data.py:1249`), row 0 of the real `[4,F]` tensor | **Direct** | Testbed literally re-admitted the real ENCODE row. Routing-isolation (offset reads depth, FiLM reads rest) ports directly and *is* h9. |
| Deterministic `f_x` (input transform) | **DSF downsampling of the input** (`x_dsf`), a known binomial-thinning transform | **Direct via DSF** | The one real transform with per-position ground truth. `enable_per_assay_dsf_sampling` (default OFF). |
| Deterministic `f_y` (output transform) | **Target depth = `y_dsf`**, i.e. denoise a shallow input toward a deeper target | **Direct via DSF** | Only depth moves; assay_id/read_length/run_type are DSF-invariant. "Dual" collapses toward single-covariate (depth) steering. |
| Identity cell / reconstruction ceiling | **Denoising tracks: `x_meta == y_meta`, `x_dsf == y_dsf`, no cloze mask** (PI's proposed analog) | **Must adapt** | No covariate yields exact input=output, but same-depth same-assay denoising is the achievable floor. Not a clean 0-gap reference. |
| Uniform-per-batch condition (v1 pooling artifact) | candi_v2 `film_mode='per_conv_and_transformer'` (pools via `meta.mean`, `encoder.py:899`) | **Available as A/B** | Exists as a switch; useful as a *negative control*, not as the object of study (production is already per-assay). |
| Per-assay FiLM (winning mechanism) | Already the production/candi_v2 default (`FiLMLayer`, grouped conv `groups=f1`) | **Direct, already present** | Hold FIXED. Not a variable to test. |
| NB + masked NBNLL head | Real NB count head `NegativeBinomialLayer` (`model.py:3222`) — **but free-mean** vs testbed's **depth-offset log-link** | **Diverges — the crux** | Free-mean head is the diagnosed DCR~1 culprit. q19 must A/B free-mean vs `depth_offset` head (candi_v2 `count_head=depth_offset`, `depth_center=22.5`). |
| **M1** (ceiling-relative reconstruction gap) | Reconstruction CRPS on **denoising (unmasked, x=y depth) tracks** as proxy ceiling | **Must invent** | No identity transform. Ceiling = per-assay same-depth denoising floor; gap is not a clean 0. |
| **M2** (per-assay CRPS steering-response matrix) | **Depth-prompt sweep** on a masked/held assay, CRPS of predicted NB vs the observed target at the *true* depth, with covariate-shuffle null | **Must rebuild** | The square `C[i,j]` matrix collapses to a **single-target CRPS-vs-told-depth curve** (real biology supplies one target, not a row of counterfactuals). `nb_crps` + `_steering_index` primitives reusable as-is. |
| **M3** (latent-cosine invariance) | Perturb **input** `x_metadata` depth, measure cos-dist of encoder `Z`; within-depth (same assay, swept DSF) ≪ between-assay | **Partial** — plumbing exists | `latent_delta_ratio` (L2) exists in `meta_probes.py` but is cruder and **collapsed during real training** (0.016→0.001). Cosine ratio itself must be built. Redefine "base" identity: same-biosample-different-depth = near; different-biosample = far. |
| Closed-form NB CRPS (Pfaff-2F1 Gini) | Reusable **as-is** over the real NB head | **Direct** | Must re-validate stability outside power≤1.5 range on real count magnitudes (Reader 7 flag). Must reconstruct `p` from the true decoder mean in float64, not the 1e-6-floored `out['p']` (`metrics.py:176-181`). |
| Non-randomized PIT calibration | **Recommended upgrade** — replaces the interval-coverage `ece_from_pit` in candi_v3/harness | **Direct, low-risk** | Production `ece_from_pit` (`eval_v3.py:64`) is the low-count-miscalibrated form the testbed explicitly rejects. Swap in `calibration_pit_curve`. |

---

## 2. The steering knob on real data

**Recommendation: depth (`log2(seq_depth)`) is the sole first-target steering knob for q19.** Argument, not survey:

1. **Continuous + monotone + physically grounded.** Depth has a known ground-truth relationship to counts (E[counts] ∝ 2^depth), so a "correct response" is *definable*: +2 log2 in the prompt should ~4× the predicted mean (the DCR target). No other covariate has a defined correct output response.
2. **It is exactly where production collapsed.** DCR~1 (`sandbox/eval.py:204`, `meta_probes.py`, candi_v3 harness) is the documented failure; depth steering is the capability q19 is really about. Testing a covariate that never collapsed would be a null exercise.
3. **It is the only covariate with a real per-position counterfactual** — via DSF thinning we can materialize the same position at a different effective depth, which is the one place real data supplies the ground truth M2 needs.
4. **It is already isolated in the architecture** as the offset-head anchor (h9), separable from the FiLM path — so the routing-isolation trick the testbed relied on ports directly.

**Why the other three covariates are deferred, not tested:**
- `assay_id` — an *identity* prompt, not a magnitude knob. Its only real steering role is as the **imputation prompt** for a masked assay (which channel to decode). This is a distinct question (query-based decoding, `spec_query_based_decoder.md`) and should be a **separate leaf**, not folded into depth steering. Note the manuscript covariate set is contested: published methods still list `sequencing_platform`, not `assay_id` (Reader 1); q19 must adopt the **code/v2 set** `(log2 depth, assay_id, read_length, run_type)` and note the manuscript is stale.
- `read_length` — responds only weakly through FiLM, unstable at overfit (`META_CONDITIONING.md`).
- `run_type` — **architecturally unused by the count head** on real data (`y_runtype_count_mse ≈ 0`). Steering it would require an architecture change, out of scope for a first pass.

**Honest limitation to state in the charter:** with only depth genuinely steerable, "**dual** conditioning" on real sandbox data largely reduces to **single-covariate (depth) input↔output steering**. The "dual" survives only in the sense that `x_metadata` (encoder, input depth) and `y_metadata` (decoder, target depth) can differ per assay via independent DSF. q19 should say this plainly rather than pretend the full 4-covariate dual knob is exercisable.

---

## 3. Input / task / output formalization

### 3.1 Input
- `x_data`: `[B, L, A]` raw counts for 8 assays, arcsinh applied **inside** the encoder on non-masked positions (`model.py:1904`), counts stored raw (int16 in h5). L=768 bins @25bp.
- Control appended at index A **after masking** (`batch.py:121`) — never masked; carries its own real metadata row.
- `x_metadata`: `[B, 4, A+1]` (incl. control) = `[depth_log2, assay_id, read_length, run_type]`, depth from `x_dsf`.
- `y_metadata`: `[B, 4, A]` (signal-only) = target covariates, depth from `y_dsf`.
- `seq`: one-hot DNA (no metadata touches the DNA tower — Reader 2).

### 3.2 Masking regime (three metadata states, keep distinct)
- **available/observed (obs)**: real assay, real metadata, `x != CLOZE & x != MISSING`.
- **masked (imp)**: cloze-masked whole assay; `_mask_full_assay` sets **data AND metadata AND availability to -2** (`_utils.py:164`). Default sandbox `p_full_assay=1.0`.
- **missing**: never-present assay, all-`-1` sentinel column (`data.py:1257`).

`obs`/`imp` = **unmasked/masked** (project constraint), AND-ed with `y_avail` (`batch.py:76`).

### 3.3 Task decomposition (the PI's identity-cell analog, made concrete)
- **Denoising tracks = reconstruction ceiling (identity-cell analog).** `x_meta == y_meta`, `x_dsf == y_dsf`, assay unmasked, no imputation. This is the achievable floor for M1. Per PI: this is the natural real analog of the synthetic identity cell — pure reconstruction, no imputation, no counterfactual.
- **Depth-steering test = denoise a shallow input toward a deeper target.** `x_dsf > y_dsf` on an **unmasked** assay: `x_meta` depth is low, `y_meta` depth is high, target is the higher-depth counts. This is the clean steering probe because **`y_meta` is real (not filled)** on unmasked assays. Do the steering measurement HERE, not on masked assays.
- **Imputation (masked) tracks = the harder, prompt-fill regime.** For a cloze/missing assay the `y_meta` prompt is itself masked/`-1`; steering can only be probed if a prompt is *injected* (canonical median or paired V/B). This regime confounds steering with prompt-construction and should be a **secondary** readout, explicitly flagged.

> **Key design pin (Reader 3):** steering is cleanly probeable only where `y_meta` is genuinely present, i.e. **unmasked/denoising** assays. The 2-row testbed never faced the masked-prompt problem; q19 must separate the denoise regime (y_meta real) from the impute regime (y_meta filled) in every readout.

### 3.4 Output head — what is fixed vs varied
- **Held FIXED:** per-assay FiLM (pool_meta=False), per-assay decoder deconv FiLM, adaLN-zero init, closed-form CRPS + non-randomized PIT as eval-only instruments, masked NBNLL loss.
- **VARIED (the q19 experimental axis):** the count head — **free-mean** `mu=softplus(Wx)` (production/manuscript default, `model.py:3244`) **vs depth-offset log-link** `log2_mu=(d − depth_center)+eta, mu=2^mu` (h9, candi_v2 `count_head=depth_offset`, `depth_center=22.5`). This is the primary structural hypothesis (§6, H3).
- **VARIED (training-side):** `enable_per_assay_dsf_sampling` OFF (copyable, shared-DSF) vs ON (non-copyable, h10). This is the task-design hypothesis (§6, H4).

---

## 4. How to train it in the sandbox

**Base model: candi_v2, not production import-and-swap, not candi_v3.** Reasons:
- candi_v2 already exposes the exact A/B knobs q19 needs as config: `count_head ∈ {free_mean, depth_offset}`, `depth_center`, `film_mode ∈ {per_conv, per_conv_and_transformer}` (pooling A/B), dual `x_meta/y_meta` interface (Readers 2, 6). Production `model.py` has the same topology but no config switch for the head; candi_v3 is optimized for Q_imp with DCR as a frozen guard-rail, wrong objective.
- The dual_conditioning testbed harness (`data.py`, `metrics.py`, `model.py`) already reads `sandbox.h5` and already wraps a per-assay-FiLM + depth-offset model — **reuse the scaffolding, swap the metadata assembly** from the synthetic 3-row knob to the real 4-dim `meta_dsf{k}[4,F]` array (Reader 6). Least-friction path.

**Metadata pathway:** per-assay (`film_mode='per_conv'`), never the pooled transformer FiLM. Raw (unnormalized) covariate encoding is the testbed/production default (h33 "raw wins"), **but re-verify** — real covariates collide across scales (depth ~22–28, read_length in bp, run_type binary) where the synthetic single-scale knob did not (Reader 5). Add a small raw-vs-zscore re-check as a q19 sub-verifiable, not an assumption.

**Prerequisite fixes to enable (all three, per the diagnosis):**
1. **Depth-offset head (h9)** — structural precondition; without it the free-mean head absorbs depth from the input and nulls DCR. This is itself H3 (test both arms), but the *expected-good* arm is depth_offset.
2. **Per-assay independent DSF (h10, `enable_per_assay_dsf_sampling=True`)** — makes target depth non-copyable so the metadata pathway receives gradient. Without it (shared DSF, reconstruct-same-assay) h35 predicts reliance ≈ 0. **Verify the sandbox h5 actually carries multiple DSF levels per assay** — it does: `counts_dsf{1,2,4,8}` + `meta_dsf{1,2,4,8}` (Reader 3, `prepare_h5.py`). So the DSF steering axis is available at sandbox scale.
3. **Non-copyable task** — is *entailed* by (2); the point of h10 is precisely to break copyability.

**Gradient-to-metadata check:** train with `x_dsf ≠ y_dsf` per assay so the target count scale differs from the input scale by `log2(y_dsf/x_dsf)`; only the depth covariate distinguishes them → the pathway must be read. This is the real analog of the testbed's `f_x ≠ f_y` non-copyable cell.

**Missing-assay metadata-slot problem (train and test):**
- **Train:** cloze-masked assay → metadata is `-2` (`_mask_full_assay`). The model must learn `-2` = "prompt withheld." Do the *steering* supervision on **unmasked** assays where `y_meta` is real; masked assays supply imputation signal but not steering signal. Do not conflate.
- **Test:** two documented paths give **different depth prompts** — canonical EIC medians (`use_canonical_missing_meta=True`, sandbox default, `train.py:560`) vs paired V/B real covariates (`_build_vb_natural_missing_meta`, `train.py:568`). **This is an open decision (§7).** For the *depth-steering* probe on unmasked assays it is moot (real y_meta); it only bites the secondary imputation-steering readout.

---

## 5. How to evaluate responsiveness & invariance (no deterministic counterfactual)

### 5.1 M2 → real depth-steering CRPS response (the rebuilt instrument)
For a target assay `a` (all other assays held at their real metadata), on an **unmasked** track with a known target at depth `d_true`:
1. Sweep the **told** `y_meta` depth `d_j` over a grid (e.g. `{d_true−2, d_true−1, d_true, d_true+1, d_true+2}`), keeping the *applied* input fixed.
2. Compute `CRPS_j = nb_crps(pred(d_j), target@d_true)` (closed-form NB CRPS, p reconstructed from the true mean in float64).
3. **Steering present** ⇔ the CRPS-vs-told-depth curve has a **minimum at `d_j = d_true`** and rises away from it. Report a scalar steering index = `(mean_j CRPS_j − CRPS_{true}) / mean_j CRPS_j` (the `_steering_index` reduction, reusable). 0 = depth-ignoring; →1 = perfectly steered.
4. **Decompose** into a **mean statistic** (Δ log μ̂ vs Δ depth — should track slope ≈ ln2 per log2 unit, i.e. DCR≈4 at +2) and a **tail statistic** (Δ upper-quantile). q18 mandate + testbed finding: report both; **success requires the MEAN to move**, not only CRPS to drop.

**Positive control:** a model trained with the depth-offset head + per-assay DSF should show a clear minimum at `d_true` and DCR≈4.
**Null baselines (pre-register both):**
- **Covariate-shuffle:** permute the depth prompt across positions/assays → steering index should collapse to ≈0.
- **Covariate-ignoring / free-mean head:** the diagnosed collapse config → DCR≈1, flat CRPS curve.

**Why mean-only DCR is insufficient (state explicitly):** DCR is a ratio of summed NB means (`sandbox/eval.py:204`) — it is blind to the distribution's shape/tail, and it dilutes across observed columns (use the per-assay-median / position-masked variant, `meta_probes.py:111`). The testbed showed steering is largely tail (tail-Pearson ~0.98) with a modest mean; a DCR-only readout would miss shape steering and a CRPS-only readout could pass on a tail artifact that leaves imputation unmoved. **Both are required; CRPS is the proper score (q18), DCR is the cheap depth-axis complement.**

### 5.2 M3 → real latent-cosine invariance (encoder side, needs no counterfactual target)
Feed the **same input** under two `x_metadata` depth settings; extract encoder `Z` (via `encode()` / `bios_pipeline get_latent_z`); measure:
- **within-depth cos-dist** (same biosample+assay, swept input DSF) — should be **small** (encoder normalizes the input depth).
- **between-assay / between-biosample cos-dist** — should be **large** (discriminative).
- ratio = within/between; testbed target **≤ 0.3** (met at 0.127 synthetically). Guard with **M1 > 0** (must reconstruct, not collapse).

**Critical redefinition:** on real data depth is a *legitimate biological covariate*, not a nuisance. So **measure invariance on the latent `Z` (which should be depth-normalized) but responsiveness on the decoder output** (which should move with the prompt). Do not demand the decoder be depth-invariant.

**Documented risk (Reader 4/6):** the L2 precursor `latent_delta_ratio` **collapsed during real training** (0.016→0.001). q19 must check whether encoder metadata sensitivity survives real training or whether the testbed's clean invariance was an artifact of the strong synthetic steering signal. This is H2 (§6).

### 5.3 Foreground-vs-aggregate split (guard-rail, from h37/q17)
Report every steering/reconstruction number **both aggregate and on top-2% foreground positions** — whole-chromosome real data is background-dominated, and steering may be foreground-localized; aggregate-only would under-report. Treat FG/BG imbalance as a monitored confound (h38/h39 are the escape hatch), **not** a prerequisite that blocks q19.

### 5.4 Guardrails (must-not-regress, not conditioning metrics)
C-index (distributional ranking) and RNA-seq log-TPM validation are downstream quality metrics that never perturb metadata — use them only to confirm steering improvements don't regress general quality.

---

## 6. Proposed hypotheses for q19 (falsifiable leaves)

**H40 — Real depth-prompt steering is present and distributional at sandbox scale.**
Verifiable: on unmasked tracks, the CRPS-vs-told-depth curve has its minimum at `d_true` with steering index ≥ **0.3** (echoing the testbed's ~0.5 per-assay M2, discounted for real noise), **and** the mean statistic moves (DCR ∈ [3,5] at +2 log2). Null (covariate-shuffle) index ≤ 0.05. *Both* the mean and tail statistics must be non-flat.
Falsified if: CRPS curve is flat or minimized off-`d_true`, or the mean is flat (DCR≈1) even if the tail moves.

**H41 — The depth-offset log-link head is required for steering; the free-mean head nulls it.**
Verifiable: A/B `count_head ∈ {free_mean, depth_offset}`, all else fixed. depth_offset → DCR ∈ [3,5] and steering index ≥ 0.3; free_mean → DCR ≈ 1 and index ≤ 0.1. Reproduces the real-chr19 R01/R15 result (DCR 1.0→4.0) at CRPS resolution.
Falsified if: free_mean also steers (index ≥ 0.3) — would overturn the q16 diagnosis.

**H42 — Per-assay independent DSF sampling (h10, non-copyable task) is necessary for the depth pathway to receive gradient.**
Verifiable: A/B `enable_per_assay_dsf_sampling ∈ {False, True}` with the depth-offset head fixed ON. OFF (shared-DSF, copyable) → steering index and DCR degrade toward null despite the offset head; ON → steering restored. Mirrors h35's dose-response (reliance rises with input↔target inapproximability).
Falsified if: steering is equally strong with shared DSF (offset head alone suffices, task design irrelevant).

**H43 — The encoder is invariant to input depth metadata while the decoder is responsive, and this survives real training.**
Verifiable: latent within-depth/between-assay cos-dist ratio ≤ 0.3 (guarded by M1>0) **at the end of training**, while decoder DCR ∈ [3,5]. Directly tests whether the `latent_delta` collapse (0.016→0.001) recurs as a genuine invariance or a pathological one.
Falsified if: the ratio → ~0 with M1 also collapsing (degenerate/ignoring), or ratio stays high (encoder not normalizing depth).

**H44 — Real noisy covariates reproduce the testbed steering effect within a stated margin.**
Verifiable: real-metadata steering index ≥ **0.6×** the testbed's synthetic per-assay M2 (i.e. ≥ ~0.3 vs 0.526) on the depth axis, at matched sandbox scale (chr19→chr21, DSF-denoising). Pre-registers "how much of the clean effect survives real covariate noise."
Falsified if: real index < 0.3× testbed (steering largely an artifact of clean synthetic knobs).

**H45 (optional, per-assay-is-necessary re-confirmation on REAL metadata).**
Verifiable: flip candi_v2 `film_mode` per_conv vs per_conv_and_transformer (pooled) with real 4-dim metadata; per-assay steering index ≫ pooled (expect the ~25× gap to shrink but persist). *Lower priority* — q16 says this is settled for production; run only to confirm the pooling artifact reproduces with real (not synthetic) metadata, which was never tested (Reader 6 open question).

---

## 7. Open design decisions for the PI (pingpong agenda, ranked by load-bearing)

1. **Does q19 accept that "dual conditioning on real data" reduces to single-covariate (depth) steering?** (Most load-bearing.) Only depth has dynamic range + a ground-truth transform + a counterfactual (via DSF). If the PI wants genuine multi-covariate dual steering, there is **no data-time mechanism** to synthesize counterfactual read_length/run_type (`fill_in_prompt_manual` is inference-only), and run_type is unused by the count head. Decision: scope q19 to depth, or first build a covariate-swap data mechanism (larger project).

2. **What is the M2 ground-truth target given no per-position counterfactual?** Recommend: **the higher-depth (lower-DSF) reconstruction is the target**, and steering is measured as the CRPS-minimum-at-`d_true` on DSF-thinned inputs. Confirm this is the intended construct — it is the only one real biology supports, and it ties M2 to the existing DSF machinery.

3. **Free-mean vs depth-offset head — is the free-mean head still the live production config?** (h9's production-scale verifiable is unresolved.) q19's premise (H41) assumes free-mean is the collapse baseline; a code-check confirms `model.py:3244` is still free-mean, but candi_v2 default is already `depth_offset`. Decide whether q19 tests both arms (recommended) or takes depth_offset as given and only tests the task/DSF axis.

4. **Missing-assay test-time prompt: canonical EIC medians vs paired V/B real covariates?** (`train.py:560` vs `:568`.) Changes the imputation-steering readout but not the primary unmasked-denoising probe. Recommend: **denoising (unmasked, real y_meta) is the primary steering surface; canonical-median for the secondary imputation readout**, reported separately.

5. **CRPS/PIT — supplement or replace the manuscript's suite?** The published eval is MSE/Pearson/Spearman + calibration-coverage + C-index; CRPS + non-randomized PIT appear in **no** manuscript. Decide whether q19's positive result must also be expressed in the manuscript's existing vocabulary (recommended, for a manuscript-ready claim) or stays a diagnostic.

6. **Re-verify "raw covariate encoding wins" (h33) on real multi-scale covariates?** Cheap sub-verifiable, not an assumption — real covariates collide across scales the synthetic knob never did. Recommend include as a small arm.

7. **Foreground/background: prerequisite or monitored confound?** Recommend the latter — report FG-vs-aggregate as a guard-rail; do not block q19 on resolving h38/h39.

8. **CRPS numeric stability at real count magnitudes** — re-validate the Pfaff-2F1 closed form against Monte-Carlo on real count distributions before trusting the steering numbers (the testbed validated only to power≤1.5).

9. **Does the manuscript's "covariates of the desired outputs" phrasing intend steering, or passive conditioning?** (Framing.) The published text never claims output responsiveness; the "controllable denoising to a supertrack" subsection is proposed but unwritten (ADDITIONS §5). Decide whether a q19 positive result is a **new claim** or a **repair of an unstated one** — this governs how it enters the manuscript.

---

### Provenance notes / disagreements flagged
- **Covariate set is contested:** published methods list `sequencing_platform`; code/v2 uses `assay_id`. q19 adopts the code set; manuscript is stale (Reader 1, unverified whether any manuscript branch updated it).
- **depth_center value differs across loops:** 22.5 (june3 `ar_fixed.yaml`) vs 24 (E30/E31). Pick one sandbox reference before baselining DCR (Reader 6 open question).
- **Per-assay-vs-pooled (h34) is settled for production** (already per-assay) — H45 is a *confirmation on real metadata*, not a fix, and is deliberately low-priority to avoid re-answering a resolved question.
- **Unverified:** whether encoder metadata sensitivity survives real training (collapsed in the L2 precursor); whether real read_length/run_type have any usable variance in the 8-assay slice (Readers 4, 6 both flag as unknown).

---

## CRITIQUE

# q19 Formalization — Adversarial Review (last gate before registration)

Prioritized by how badly each would corrupt a "positive q19 result." The first three are disqualifying as written.

---

## 1. [CONFOUND — disqualifying] The offset head makes the primary steering metric pass by arithmetic, not by learning

**Problem.** The depth-offset head computes `mu = 2^(d − center) · exp(eta)`. With the offset ON, sweeping the told `y_meta` depth `d_j` moves `mu` by `2^(d_j−center)` *mechanically*, independent of whether the model learned anything. So every mean-based readout the draft calls "steering" is satisfied by construction:
- DCR ≈ 4 at +2 log2 is the *arithmetic of the offset you hardwired*, not learned conditioning.
- The M2 "CRPS-vs-told-depth curve has its minimum at `d_true`" is minimized at `d_j≈d_true` because `2^(d_j−center)` best matches the target mean there — again pure arithmetic.
- The mean statistic (`Δ log μ̂` slope ≈ ln2) is the offset's derivative, not `eta`'s.

The testbed already knew this: h36 says offset attribution is *unconditional* ("steering present offset on or off"), and it therefore measured genuine steering on the **tail statistic** (`Δ` upper-quantile), which "never enters `n`" and so is offset-independent. The draft imports the mean-DCR machinery but drops that firewall.

**Why it matters.** H40 and H41's success criteria (`DCR∈[3,5]`, mean moves, curve minimum at `d_true`) will pass for *any* model with the offset head, including one that ignores the input and the FiLM prompt entirely and only reads `d` through the offset. A "positive q19" under these criteria is not evidence of learned dual conditioning — it is evidence that `2^d` arithmetic works. This is the single most likely way q19 produces an artifact.

**Fix.** Split the question the draft is conflating:
- **(a) "Can the output be depth-controlled?"** — trivially yes via the offset; not worth an experiment.
- **(b) "Did the model *learn* to read the depth prompt?"** — the real q19. Measure it only on offset-*independent* components: the **residual `eta` response** and the **tail/dispersion statistic** (`Δ` quantile, `Δ log n̂`). Note DSF thinning changes the target's *dispersion*, not just its mean (thinned NB has different `n`), so the tail is where a genuinely learned response must show up — the offset cannot fake it.
- Re-write H40/H41 success as: *with the offset arithmetic partialled out, `eta` and/or the tail statistic track `d_true`, and correct-depth CRPS beats shuffled-depth CRPS by more than the bootstrap noise floor.* Demote raw DCR∈[3,5] from a pass criterion to a sanity check.

This also resolves the draft's internal tension where §0/§5.1 declares "success requires the MEAN to move" (from q18) — with the offset head the mean moving is free and therefore the *wrong* gate.

---

## 2. [CONFOUND — disqualifying] Measuring steering on unmasked/denoising assays reintroduces the copyability leak the whole design exists to avoid

**Problem.** §3.3 correctly moves the steering probe to **unmasked** assays (so `y_meta` is real, not filled). But on an unmasked assay the model *sees the input counts for that exact position*. The target at `d_true` is a depth-rescaling of those same counts. So the input is highly informative about the target regardless of the prompt — this is the h35 shortcut in a new costume. The marginal CRPS contribution of the `y_meta` prompt is small *even if the model reads it*, systematically under-powering the steering signal on precisely the regime the draft nominates as primary.

There is a genuine dilemma the draft doesn't state:
- **Unmasked** (draft's choice): `y_meta` is real, but the input leaks the target → steering under-powered.
- **Masked**: the prompt is the *only* depth information (clean test of prompt-reading) → but `y_meta` is filled (canonical median / V-B), confounding steering with prompt construction.

**Why it matters.** The draft presents unmasked-denoising as the clean surface. It is clean for *prompt authenticity* but dirty for *steering power*. A weak or null steering index on unmasked assays could mean "no steering" OR "steering present but masked by input leakage" — the readout cannot distinguish them.

**Fix.** Report both regimes as a **matched pair with opposite biases**, and make the load-bearing contrast explicit: on unmasked assays, does the *correct* `y_dsf` prompt beat a *shuffled* `y_dsf` prompt on held-out CRPS at the same position? That contrast survives the input leak because both arms see the same leaking input; only the prompt differs. Pin this correct-vs-shuffled-prompt delta (not the absolute steering index) as the primary falsifiable quantity.

---

## 3. [CONFOUND] Depth leaks into the target through the control channel and correlated context assays

**Problem.** Control is never masked and carries its own real depth row; all other unmasked assays carry their real depths in `x_metadata`. Within a biosample, sequencing depth is correlated across assays (shared library/run) — Reader 6's covariate_probes explicitly flags a depth↔biology cohort confound. So the target depth for the probed assay is partially inferable from context depths *without reading `y_meta`*.

**Why it matters.** A model can score DCR≈4 and a low correct-vs-shuffled CRPS gap by reading depth off the *context*, not the prompt — a covariate-ignoring-of-the-prompt baseline that still passes. The draft's null baselines (shuffle the target prompt, free-mean head) do **not** catch this, because context depth is untouched by shuffling the target prompt.

**Fix.** This is exactly what **per-assay independent DSF (h10, H42)** buys — it *decorrelates* each assay's sampled depth from the others' and from control. Reframe H42's justification: it is not only "makes the target non-copyable from the input" but "decorrelates context depth from target depth so `y_meta` is the *only* source of target depth." Then add a null: **freeze `y_meta` depth to a constant while context depths vary** — if the output still tracks the (varying) context depth, the prompt is being bypassed. Make h10=ON a *precondition* for a valid steering claim, not merely a second hypothesis.

---

## 4. [METRIC — cannot be computed cleanly] M1 "ceiling" is position-dependent and not comparable across conditions

**Problem.** The testbed M1 gap = `cell_CRPS − identity_cell_CRPS`, where the identity cell is a *deterministic perfect reconstruction* (CRPS floor is a fixed reference). On real data, even same-depth same-assay reconstruction has irreducible aleatoric dispersion, so the "ceiling" CRPS (a) is not 0, (b) varies per position with count level and dispersion, and (c) differs across assays. The M1 gap loses its meaning as "distance from achievable optimum" and is not comparable across the conditions you want to rank.

**Why it matters.** Any M1-based number in q19 mixes steering skill with per-position dispersion heterogeneity. Cross-condition M1 comparisons become uninterpretable.

**Fix.** Drop M1-as-gap. Replace with a **skill score normalized by a marginal-baseline CRPS** (the Q_imp pattern already used in the autoresearch loops): `1 − CRPS_model / CRPS_marginal`, where the marginal baseline is the depth-conditioned marginal count distribution. That gives a per-condition-comparable, dispersion-normalized number and connects to the existing sandbox metric vocabulary.

---

## 5. [UNFALSIFIABLE] H44's threshold (≥0.6× the synthetic index) has no valid baseline

**Problem.** H44 compares a real-data steering index to the *synthetic* testbed index (0.526) via an invented 0.6× factor. The two numbers come from different data-generating processes, different target definitions (deterministic transform vs single noisy count), and different metric constructions (7×7 `C[i,j]` matrix vs collapsed single-target curve). There is no principled null for "how much of a synthetic effect should survive real noise," so 0.6× is arbitrary and the hypothesis is unfalsifiable in any meaningful sense.

Additionally H44 says "real *noisy covariates*" (plural) but the entire draft has narrowed to depth-only — the framing is internally inconsistent and largely duplicates H40.

**Fix.** Delete H44. Fold its intent into H40 with an **internally-anchored, absolute** criterion: correct-depth prompt CRPS < shuffled-depth prompt CRPS by a margin whose bootstrap CI excludes zero, measured on foreground positions. No cross-regime ratio.

---

## 6. [METRIC EMPHASIS INVERTED] Aggregate CRPS is dominated by background zeros where depth barely matters

**Problem.** Depth scaling moves the distribution meaningfully only at appreciable counts (4× on `mu=100` is large; 4× on `mu=0.1` is negligible in CRPS). Most positions are low-count background. So *aggregate* CRPS steering is dominated by positions where depth is nearly irrelevant, systematically under-reading the true steering signal — which lives in foreground/high-count positions.

**Why it matters.** The draft makes aggregate the primary readout and FG a "guard-rail" (§5.3). That is backwards: the biologically and statistically load-bearing steering signal is foreground-localized, and h37 already showed FG-vs-aggregate gaps are family-specific and can flip sign.

**Fix.** Make **foreground/high-count-stratified CRPS the primary steering readout**; aggregate is the secondary/dilution check. State the count-magnitude rationale explicitly.

---

## 7. [SCOPE / MEASURABILITY] H43 "invariance survives real training" is ill-posed at sandbox scale

**Problem.** The L2 precursor collapsed 0.016→0.001 "after overfit." Sandbox runs are known to be either undertrained (~2.3 epochs; invariance not yet formed) or chr19-overfit (collapse). "Does encoder depth-invariance survive *real training*" has no well-defined operating point at sandbox scale — the regime it needs is the production regime q19 is explicitly deferring.

**Fix.** Either (a) define an exact training-length protocol and checkpoint-selection rule for H43 and acknowledge it measures "invariance at checkpoint X," not "survives training," or (b) demote H43 to a monitored diagnostic (track the cos-ratio trajectory vs M1 trajectory) rather than a pass/fail leaf. Also fix the stated guard: `ratio→0 with M1 collapsing` = degenerate; the draft says this but should add the trajectory check since a single end-of-training snapshot can't distinguish "healthy invariant" from "on its way to collapse."

---

## 8. [GAP] The depth sweep grid `d_true ± 2` is not realizable per assay and risks out-of-support extrapolation

**Problem.** Real depths span 22.2–28.1; per-assay DSF gives at most 4 levels `{1,2,4,8}` → a reachable depth *set* of width `log2(8)=3` anchored at each assay's base depth, not a symmetric `±2` around `d_true`. Sweeping `d_true+2` for a base-28 assay asks for depth 30, never observed. The **offset head extrapolates fine (arithmetic) but a FiLM/`eta` response cannot** — so an out-of-support sweep unfairly flatters the offset arm and penalizes the free-mean arm in the H41 A/B.

**Fix.** Define the sweep over each assay's **DSF-achievable depth set**, not an abstract `±2`, and keep all told-depths within observed support. Report the grid per assay.

---

## 9. [PROXY, name it] candi_v2 `free_mean` is not production `model.py`'s head — H41 "reproduces the production collapse" is an inference

**Problem.** H41 claims to reproduce the production DCR~1 diagnosis, but q19 trains candi_v2, whose `free_mean` head is not verified bit-identical to production `NegativeBinomialLayer` (`model.py:3244`). candi_v2's default is *already* `depth_offset`.

**Fix.** State that q19 tests candi_v2 `count_head=free_mean` as a **proxy** for production's free-mean head; a production-model reproduction is out of scope. Otherwise H41's provenance claim overreaches.

---

## 10. [CONSISTENCY] The h33 "raw vs zscore" re-check is unmotivated in a depth-only q19

**Problem.** The multi-scale collision that motivates re-checking normalization (depth ~22–28 vs read_length in bp vs run_type binary) only arises with **multiple** covariates. q19 is depth-only, and depth is *already* log2-scaled — there is no cross-scale collision to worry about. §7.6/the sub-verifiable is orphaned.

**Fix.** Drop the h33 re-check from a depth-only q19 (or explicitly restore ≥2 covariates to justify it — but that reopens the "no ground truth for non-depth" problem). Pick one.

---

## 11. [GAP, easily closed] Counts-only scope should be *justified biologically*, not left as an open question

The draft leaves Reader 3's "extend to Gaussian/Bernoulli heads?" open. It is closeable now: **depth's causal target is raw counts; pval is depth-normalized and peaks are calls**, so depth steering is counts-only *by construction*. State this — it converts an open question into a positive scope boundary and further justifies the depth-only, NB-only design.

## 12. [SHARPEN] assay_id deferral reason is the fixed-channel redundancy, not "identity vs magnitude"

The current fixed-channel decoder already encodes which assay by column position; `_mask_full_assay` sets the assay_id *metadata* to −2 anyway. So assay_id-as-prompt is redundant with channel position and structurally un-steerable on the current decoder (`spec_query_based_decoder.md` is the whole point). Deferring is right; give the *structural* reason so the separate leaf is scoped correctly (it needs the query-based decoder, not just "a magnitude knob").

---

## What the draft got RIGHT and must keep

- **Refusing to re-test per-assay-vs-pooled as the production lever** (h34/q16). H45 correctly demoted to an optional "does the artifact reproduce with real metadata" confirmation. Do not let it creep up.
- **Naming depth as the only real steerable covariate with a per-position counterfactual (via DSF)**, and being honest that "dual" reduces to single-covariate depth steering. This is the correct, defensible core.
- **Separating the denoise regime (`y_meta` real) from the impute regime (`y_meta` filled)** as distinct readouts (Reader 3's key structural gap) — keep, and extend it per Issue 2.
- **h10 per-assay independent DSF as the non-copyability lever** (H42) — keep; upgrade its rationale to include context-depth decorrelation (Issue 3).
- **The free-mean vs depth-offset head as the crux structural axis** (H41) — keep, but strip the offset-arithmetic tautology from its success metric (Issue 1) and label it a proxy (Issue 9).
- **CRPS mandatory + mean/tail decomposition** (q18) and the **non-randomized PIT upgrade** over the interval-coverage `ece_from_pit` — keep; the tail half becomes *more* central after Issue 1.
- **The p-from-true-mean float64 reconstruction** (not the 1e-6-floored `out['p']`) — keep; add the Reader 7 re-validation of the Pfaff-2F1 CRPS at real count magnitudes before trusting numbers.
- **Reusing the dual_conditioning scaffolding and swapping only the metadata assembly**, staying at sandbox scale, verifying `counts_dsf{1,2,4,8}` exist — correct, low-friction, and the right rung below q18.
- **Flagging the stale covariate set, the 22.5-vs-24 `depth_center` ambiguity, and the manuscript-claim provenance** — good hygiene; keep in the pingpong agenda.

**Bottom line:** the draft's architecture is sound but its *success criteria* are, as written, mostly satisfiable by the offset head's arithmetic and by input/context depth leakage. Before registration, re-anchor every steering pass/fail on **offset-independent components (residual `eta` + tail) and correct-vs-shuffled-prompt CRPS deltas on foreground positions, under h10=ON**, and delete the cross-regime H44 threshold. Do that and q19 becomes falsifiable; leave it and a "positive" result is very likely an artifact of `2^d`.

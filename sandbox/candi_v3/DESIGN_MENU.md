# CANDI v3 — design menu + hard constraints (→ ERA `problem.description`)

This document is the source for the ERA `Problem.description`. It states (1) the problem &
interface, (2) the HARD CONSTRAINTS every candidate must obey, (3) a MENU of high-value
directions distilled from `sandbox/CRITIQUE.md` that candidates *may* explore, and (4) the
scoring contract. ERA chooses freely among the menu; the constraints and contract are
non-negotiable.

---

## 1. Problem & interface

Build a self-supervised model for **epigenome imputation + denoising** from raw sequencing
read counts + 4 covariates + DNA, on the 8-assay sandbox HDF5 slice. The candidate is a
whole program defining `model.py` + `objective.py`; the frozen harness supplies data,
trains for a fixed budget, evaluates, and emits the score. Keep CANDI's genuine strengths:
raw counts with a size-factor offset, distinct MISSING vs CLOZE tokens, distributional
outputs, no RNA-seq leakage.

---

## 2. HARD CONSTRAINTS (violation → degeneracy gate → −1e9)

1. **Native DNA tower only.** No external pretrained models (Borzoi/Enformer/genomic-LM)
   and no imported external weights. Optimize a native DNA tower instead.
2. **Memory-efficient context.** Longer context is welcome but the quadratic attention
   memory is the binding constraint — any long-range mechanism must be memory-efficient
   (dilated / hierarchical / linear-/flash-style / two-scale). No dense O(L²) blow-up.
3. **Control-optional.** Must run with OR without a ChIP control track. Use control if
   present; never require it. (Control channel = index `A` in `M`, never masked when present.)
4. **Fixed decoder first.** Begin with a fixed decoder. A query-based decoder is an
   optional later exploration, not the initial design.
5. **No full JEPA initially -- only try it if we hit deadends elsewhere.** Latent regularization only via the light routes: Gaussian prior + ELBO,
   or SIGReg on the latent. No full JEPA training.
6. **Covariate semantics.** 4 covariates = `log2(seq_depth), assay_id, read_length,
   run_type`. `obs`/`imp` = unmasked/masked (NOT biological observed/imputed).

---

## 3. MENU — high-value directions to try (from the critique)

Candidates may adopt any subset; none is mandatory.

**Targets & likelihood (§2, §3, §9.1–2)**
- Predict ONE latent enrichment field `λ(assay,pos)`; derive the three views instead of
  three independent heads: counts ~ NB(λ·size_factor, φ) as the only learned likelihood;
  pval as a fixed calibrated transform of (λ vs local-control λ); peaks as `P(λ>τ)` under
  the count posterior (automatically calibrated). Collapses 3 heads / 3 likelihoods / 6
  weights.
- Make control an **explicit local background** (differentiable MACS-style local-lambda) so
  the network predicts depth/background-invariant *enrichment* and control stops being
  double-counted — **but keep it optional** (constraint 3).

**Corruption / SSL objective (§4, §9.5)**
- Unify corruption into ONE noise process: compose per-(assay,pos) dropout at a sampled rate
  ⊕ depth downsampling, drawn from a schedule → a single posterior `p(clean|corrupted)`.
  "Impute a whole assay" and "denoise a shallow track" become two points on one continuum,
  one branch, one weight — dissolving the obs/imp adversarial frontier.
- Or, if kept separate, **factor the representation by task**, not the loss.
- Make masking a clean **vectorized parameterized distribution** (not nested Python loops),
  and match the corruption distribution to deployment (high-mask-ratio tail / curriculum).

**Architecture (§5, §9.3–7)**
- **Assays as tokens / a set**: axial or factorized attention over (assay × position), or a
  set-transformer over assays at each latent position → true cross-assay attention + native
  handling of variable/novel assay sets (the zero-shot premise). Replace the single linear
  fusion.
- **Condition once per covariate role** to kill DCR fragility: depth enters ONLY through the
  size-factor offset (its physics is multiplicative library size); assay identity enters ONLY
  as a query/embedding; run-type/read-length as a small additive bias. Fewer FiLM pathways →
  less gradient competition.
- **Native DNA tower** that preserves motif-scale (6–20 bp) information rather than pooling
  it away before fusion (constraint 1).
- **Longer, memory-efficient context** to reach enhancer–promoter / TAD-scale dependencies
  (constraint 2).
- (Optional, later) **query-based decoder**: requested (assay,pos) cross-attends into the
  latent; assay identity is the query (constraint 4 — start fixed).

**Losses & training (§6, §7, §8)**
- Fewer heads → fewer weights; principled uncertainty weighting for any remaining trade-off.
- An **explicit calibration objective** (coverage penalty / CRPS) — calibration is scored,
  do not merely hope it emerges from NLL.
- Focal / class-balanced loss for the (derived) peak readout.
- Representation-first instinct: a strong latent with light, internally-consistent readout
  heads (the latent `Z` already beats the outputs at RNA-seq) — pursued via the light latent
  regularizers (constraint 5), not full JEPA.

---

## 3.5 Proven prior knowledge (priors from ~hundreds of GPU-hours — NOT constraints)

Empirical observations distilled from prior CANDI v1/v2 ablations, the May/June-2026
autoresearch loops, and the first 100 ERA nodes. **These are priors, not rules** — adopt,
adapt, or ignore any of them freely; only §2 HARD CONSTRAINTS are non-negotiable. They exist
to stop you re-discovering known dead-ends and to point at where the real headroom is. Many
diverse designs can use (or reject) each of these.

**Latest lessons (ERA rounds ~100–180, after the budget was raised to 5 epochs)**
- **The marginal baseline IS now beatable.** Several candidates crossed it on imputation skill
  (Q_imp up to ~0.50, S_A > 0; population median Q_imp ~0.37 → ~0.44) — the earlier
  "plateau below baseline" was largely a training-budget artifact. The move that first crossed
  it: **predict the cell-type-specific DEVIATION on top of a leak-free average-reference**
  (a zero-initialised deviation head over `reference + CF`, and/or an explicit
  `(pred − ref)·(target − ref)` deviation-correlation loss) — spend capacity on the residual,
  not on relearning the reference (§3.6a, confirmed).
- **Multi-resolution regional context helped magnitude + peak shape** — an Avocado-style
  pyramid of average-pools (~200 bp / 800 bp / 3.2 kb) concatenated into the readout lifted the
  pval-Pearson term (median ~0.20 → ~0.34).
- **Calibration (ECE) is now THE binding constraint, and it is still UNSOLVED.** ~70% of recent
  candidates violate the ECE floor, and **zero** of the baseline-beating candidates also satisfy
  it — so every skill gain is currently taxed by miscalibration. Many tried (per-assay
  dispersion, variance floors); none cracked it. The single highest-value win now is a candidate
  that pairs baseline-beating skill **with** a satisfied ECE floor.
- **The count-mean (NB) Pearson is the lagging correlation** — pval-Pearson rose but
  count-Pearson barely moved (~0.21 → ~0.23): the NB mean is shaped by NLL, not for magnitude
  correlation. Shaping the count-mean for magnitude/deviation (as already done for the signal
  head) is open headroom.

**Where the headroom actually is**
- The held-out score's ceiling is the **magnitude correlations (Pearson / R²), not rank
  (Spearman)**: across prior runs rank correlations reach ~0.45–0.65 while Pearson sits
  ~0.20–0.30. Effort spent *purely* on rank/Spearman surrogates repeatedly **regressed** real
  skill; absolute-level accuracy (per-assay scale/offset alignment, cross-track level placement,
  getting the magnitude right) is the under-served lever.
- **Calibration is a scored floor most candidates miss**: ~78% of the first 100 nodes violated
  the ECE floor (over-confident count distribution). Calibration rarely "falls out" of NLL
  alone — designing for it (a dispersion/variance floor, per-assay dispersion, or a *light*
  explicit coverage term) is high-value. A *heavy* coverage/aux term, by contrast, has been
  neutral-to-harmful: the models are **capacity-limited, not gradient-limited**.

**Satisfying the DCR floor (depth) — a near-solved mechanism**
- Letting **depth enter only as a multiplicative size-factor on the count mean** —
  `μ = 2^(d − c)·exp(η)` with `d` = target log2-depth, `c` ≈ the data's median log-depth (~22.5),
  the network predicting depth-free enrichment `η` — holds DCR in band by construction (only
  2/97 prior nodes fell out of [3,5]). A learnable center/slope is fine. The NB/size-factor head
  is **fragile**: changing its *structure* (grouped/diagonal/quadratic dispersion, extra competing
  depth params) has repeatedly collapsed DCR/denoising — a stable component to innovate *around*.

**Inputs / tokens (small, repeatedly-confirmed wins)**
- **log1p / arcsinh the counts** before the network — raw counts drive late divergence; the
  transform fixes it at no quality cost.
- **Per-assay mask/query tokens** (a distinct learned vector per held-out assay) beat a single
  shared mask token, which aliases all held-out assays to identical features.
- If you attend across assays, **mask the attention to genuinely-present assays** (don't let a
  query attend to zero-filled absent/held-out slots).

**Loss weighting for generalization (not memorization)**
- For *held-out* generalization, **do not heavily up-weight masked/imputation positions** —
  moderate-to-slightly-down-weighted imputation loss (≈ the observed weight or below) generalized
  best; aggressive up-weighting (3–8×) helped only single-batch memorization and hurt real
  held-out skill.

**Architecture priors worth trying (from CANDI v2; medium transfer to this cold-start backbone —
options to widen the search, none required)**
- **GroupNorm** inside the decoder/deconv path was the single biggest v2 architecture win (large
  denoising gain); other norms there were worse.
- A **single-shot pre-decoder FiLM** conditioned on *target* metadata (telling a fixed decoder
  *what* to reconstruct) beat per-layer decoder FiLM, which over-conditions.
- A **2-layer ("deep") linear signal+DNA fusion with a LayerNorm after it** beat 1-layer;
  **gated** fusion was consistently catastrophic — keep fusion linear.
- Transformer hygiene that helped: **RoPE positions, QK-norm, d_head ≈ 8–9, a small dropout floor
  (≥0.02)** (dropout here behaved like cross-assay augmentation, not just regularization).
- A **cross-assay consistency term** (imputed-assay per-locus means ≈ observed-assay per-locus
  means, observed side detached) gave a small, direct imputation gain.

**Known dead-ends (deprioritize — each cost multiple prior runs)**
- **Rank-only / soft-Spearman as the dominant loss** — regressed real held-out skill almost every
  time (see "headroom" above).
- **Standalone extra heads** (e.g. a separate peak head) — gradient competition that hurt the
  main objective; derive secondary views (peaks) from the enrichment field instead of a competing
  head.
- **MACS-style local-background ratio as the regression *target*** — discards magnitude and
  regressed both pval correlations (it can still be a useful *feature*).
- **Heavy auxiliary MSE / coverage terms** — neutral-to-harmful; the bottleneck is model capacity
  within the budget, not gradient signal.

**A known structural tension (an opening, not a wall)**
- Imputation and denoising trade off along a **Pareto frontier when they share one backbone** —
  pushing capacity into one has repeatedly wrecked the other. Candidates that **decouple the
  imputation and denoising pathways** (separate heads/paths/representations, within the memory and
  fixed-decoder constraints) are an under-explored direction for escaping it.

---

## 3.6 Priors from the literature

CANDI's task is FIXED and not up for redesign: **DNA + observed signal tracks in → impute &
denoise the signal tracks out**. DNA is always a conditioning input, **never a prediction
target**; do not add sequence/DNA-prediction or language-modeling objectives. The items below
are **directions and idea-pointers, not blueprints** — read them for transferable
architectural/methodological choices, not to replicate any single model. Each links the
official repo; you MAY fetch a repo to study the code, but take only what serves CANDI's task.
§2 HARD CONSTRAINTS still bind (native DNA tower, no external weights).

**(a) Where the imputation task is known to be hard — directions to attack** *(ENCODE Imputation
Challenge assessment, https://pmc.ncbi.nlm.nih.gov/articles/PMC10111747/)*
- **The per-position average-reference baseline is brutally strong** — in the official challenge
  it placed 3rd overall and won 5 of 9 metrics. The only thing that beats it is the
  **cell-type-specific *deviation* from the cross-cell average**, not genome-wide shape. Model
  that residual; do not re-predict the average.
- **Cell-type specificity is the blind spot.** Genome-wide correlations can look fine while
  missing *which* cell types a peak is active in. The hardest, most valuable targets are
  **peaks active in some-but-not-all cell types, and low-abundance peaks**.
- **Magnitude at correctly-located peaks is the dominant error** — models find peaks but
  mispredict the precise value inside them (over-smoothing / regression-to-mean); in H3K4me3
  ~99% of residual error sat at correctly-predicted peaks. Getting peak *magnitude* right (not
  just rank/location) is the headroom.
- **Accuracy has three regimes** vs signal level: easy at low signal, easy at very high signal,
  **sharp failure at intermediate/ambiguous signal** — the mid-range is where skill is won.
- **Local peak *shape* is under-served** — accessible loci largely keep peak shape across cell
  types, yet few methods beat the baseline on shape; within-peak local correlation matters, not
  just global rank.
- **Treat peaks and background differently** — normalization/distributional confounds dominate
  genome-wide metrics; separate handling of peak vs background regions was the principled fix.
- **No single architecture wins** — factorization and sequence-based methods beat
  similarity/KNN/plain-CNN; winners were diverse (supports the factorization idea below *and*
  keeping the search diverse).

**(b) Architectural/methodological ideas worth borrowing** *(high-level only — read the repos for
specifics; take only what fits CANDI's fixed task)*
- **Low-rank tensor factorization** — represent the data as separate learned embeddings for
  *cell-type × assay × genomic-position* (with **multi-resolution position factors**, e.g.
  fine + coarse scales, for cheap long-range context), combined by a small network. **Net-new
  for CANDI** (its search used attention-based borrowing, never explicit factorization).
  *Avocado:* https://github.com/jmschrei/avocado
- **Contextualised cross-cell / cross-assay attention embeddings** — attention over the
  *observed* (cell, assay) set to build context embeddings that impute the missing tracks.
  *eDICE:* https://github.com/alex-hh/eDICE
- **Observed-signal-conditioned sequence pathway** — let a cell-type-specific signal (e.g.
  accessibility) *condition* the sequence encoder so one DNA backbone generalizes across cell
  types (in CANDI, the observed assays play that conditioning role).
  *EPCOT:* https://github.com/liu-bioinfo-lab/EPCOT
- **Memory-efficient long-range DNA tower** — dilated-conv + attention/soft pooling (Enformer)
  and multiscale conv U-net (Borzoi) reach long range cheaply while preserving motif scale.
  **Mechanism only — weights forbidden (§2.1);** this is CANDI's DNA *input* pathway, not the
  whole model. *Enformer:* https://github.com/google-deepmind/deepmind-research/tree/master/enformer
  · *Borzoi:* https://github.com/calico/borzoi
- **One masking interface over the signal tracks** — a single masking scheme over the observed
  tracks unifies imputation, denoising, and context-aware local correction (DNA stays an unmasked
  input). CANDI already does a version of this; the framing is the takeaway.
  *Nona:* https://www.biorxiv.org/content/10.1101/2025.11.06.687036v1

---

## 4. Scoring contract

Run under the fixed budget; print exactly one line `ERA_SCORE: <float>` (higher better).
Crash / OOM / timeout / missing footer / constraint violation / structural degeneracy →
`−1e9`. Self-contained: write only inside the candidate's own temp workdir; never edit the
harness or shared files. The objective (frozen, see `METRIC.md`):
`ERA_SCORE = S_A + w_cal·min(0,τ_cal−ECE) + w_dcr·(min(0,DCR−DCR_lo)+min(0,DCR_hi−DCR)) + (−∞ if degenerate)`.

**Imputation skill (S_A) is the REAL zero-shot metric:** train masks/denoises only the
biosample's T_ assays; the reserved **V_/B_ assays of the same biosample are never seen in
training** and are the imputation targets (`imp_*` on `imp_eval_map`). Denoising = `den_*` on
T_ at higher depth via DSF; DCR target ≈ 4.0 (band). See PLAN §2.5.

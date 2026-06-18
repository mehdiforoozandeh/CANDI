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

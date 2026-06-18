# CANDI — A First-Principles Critique

*Scope: the scientific and design choices behind CANDI, read off `sandbox/candi_v2/`
(canonical clean reference) and the production code (`model.py`, `candi_loss.py`,
`_utils.py::DataMasker`, `sandbox/data.py`, `sandbox/batch.py`). This is a "if I were
designing it from scratch, knowing what we now know" document, not a bug list. Each
section ends with **→ what I'd change**.*

---

## 0. Thesis

CANDI is a sound idea executed as **three loosely-coupled regression problems bolted
onto one shared CNN→transformer→deconv backbone**, conditioned by FiLM in a dozen
places. Most of the pain we've seen empirically — the imputation↔denoising Pareto
frontier, the extreme fragility of the depth-calibration gradient path (DCR), the
endless hand-tuning of six loss weights, the fact that the *latent* `Z` predicts
RNA-seq better than the model's own *outputs* — are not independent bugs. They are
symptoms of three first-principles design decisions:

1. **Redundant targets.** Counts (NB), arcsinh-pval (Gaussian), and peaks (Bernoulli)
   are three deterministic views of *one* underlying enrichment field, modeled as three
   independent heads with three incompatible likelihoods and six loss weights.
2. **Two adversarial self-supervised tasks on one representation.** "Imputation"
   (whole-assay masking) and "denoising" (DSF depth downsampling) are different
   corruption processes with *different invariances*, trained jointly on a single
   backbone. The Pareto frontier the autoresearch loop spent weeks mapping is the direct
   consequence.
3. **Shallow, fixed-axis cross-assay reasoning.** The hardest part of imputation —
   inferring an unseen assay from seen ones — is handled by a *single linear fusion
   layer plus position-wise FFNs*. There is no attention over the assay axis, and the
   assay axis is a fixed integer dimension, not a set.

Everything below elaborates. I'll flag what is genuinely well-designed first, so the
critique is calibrated.

---

## 1. What is right (keep these)

- **Raw counts in, with an explicit library-size offset.** `DepthOffsetNegativeBinomialLayer`
  ([decoder.py:111](candi_v2/decoder.py#L111)) reconstructs `log2(mu) = (d − c) + eta`
  with `d = log2(seq_depth)` as a *fixed* offset. This is the DESeq2/edgeR/scVI size-factor
  convention and is the single most principled thing in the model: it makes `eta` learn
  enrichment, not library size, and gives the model a clean place to be depth-invariant.
- **Distinct MISSING vs CLOZE tokens** in metadata and signal
  ([encoder.py:57](candi_v2/encoder.py#L57)). "Data absent" and "predict this" are
  genuinely different conditioning signals; collapsing them would be wrong.
- **Distributional outputs.** Predicting `(n̂, p̂)` rather than a point estimate is the
  right substrate for the calibration / C-index deliverables.
- **Self-supervision with no RNA-seq leakage.** The biological-validation story (latent
  `Z` → RNA-seq, never trained on it) is a strong, honest evaluation.

The problems are in how these good primitives are *composed*.

---

## 2. Targets & likelihoods — the redundancy at the core

CANDI predicts, per (assay, position):
- NB `(n̂, p̂)` for raw counts,
- Gaussian `(μ̂, σ̂²)` for `arcsinh(-log10 pval)` signal,
- Bernoulli for peaks.

But these are **not three observations** — they are one latent intensity seen three
ways. `-log10 pval` is (essentially) a MACS2 deterministic function of the same counts
versus the local control; peaks are a *threshold* on that pval. We are asking the network
to predict a quantity, a smooth monotone transform of that quantity, and a step function
of the transform — with three separate decoder trunks (`trunk="separate"` in KEEP9), three
likelihoods, and six obs/imp loss weights to balance them.

Consequences that show up in the code and the empirics:
- **Conflicting gradients.** The NB head wants depth-sensitive raw-count calibration; the
  Gaussian-pval head wants a depth-*normalized* enrichment. They share the encoder. This is
  a second axis of the same task-conflict problem as imp/den.
- **The Gaussian is a known-bad fit** for arcsinh-pval (zero-inflated, heavy-tailed,
  bounded below). The codebase carries Laplace, Student-t, Gamma, and constant-variance
  variants ([model.py:3293-3463](model.py#L3293)) — an inventory of escape hatches is
  evidence the base likelihood was never right.
- **Peaks via plain BCE mean** ([candi_loss.py:423](candi_loss.py#L423)) on an extremely
  imbalanced label (most positions are background). fg/bg balancing exists but is *off* by
  default in v2. A model can minimize this by predicting "background everywhere."

**→ What I'd change.** Predict **one** latent enrichment field — a rate `λ(assay, pos)` —
and *derive* the three views:
- counts ~ NB(λ · size_factor, dispersion) (the only learned likelihood),
- pval = a fixed, calibrated transform of (λ vs local control λ) — not a separately
  regressed head,
- peaks = a learned-threshold / calibrated readout of the *same* λ posterior (so peak
  probability is literally `P(λ > τ)` under the predicted distribution, which is also
  automatically calibrated).

This collapses three heads, three likelihoods, and six weights into one generative model
with internally-consistent uncertainty, and makes the peak probability a *property of the
count posterior* instead of a fourth thing to balance.

---

## 3. The control channel is double-counted

The ChIP control is fed as an input track (index `A`, never masked) **and** is already
baked into the pval/peak targets (pval is enrichment over local control λ). So control
information enters twice, and the NB count head never uses it the way peak-calling actually
does — as a *local* background rate. The model has to re-learn "counts relative to control"
implicitly.

**→ What I'd change.** Make the control explicit in the count generative model:
`λ_signal = enrichment × λ_control_local`, where `λ_control_local` is a smoothed function
of the control track (MACS's local-lambda idea, but differentiable). Then the network only
predicts *enrichment*, which is exactly the depth- and background-invariant quantity we
want for cross-cell-type zero-shot transfer — and the control stops being double-counted.

---

## 4. Corruption / masking / the SSL objective

This is where the biggest structural problem lives.

CANDI runs **two distinct corruption processes** and supervises them as **two loss
branches**:
- **Imputation** = whole-assay masking (`p_full_assay=1.0`, v2 forbids position-cloze;
  see `validate_v2_config` [config.py:178](candi_v2/config.py#L178)). Mask entire assay
  columns; predict them from the other assays → the `imp` (masked) branch.
- **Denoising** = DSF downsampling (`_sample_xy_dsf` [data.py:28](data.py#L28)): input at
  depth `xd`, target at higher depth `yd`; reconstruct the cleaner track → the `obs`
  (unmasked) branch.

The autoresearch loop **proved** (project memory, KEEP12 study) that these two objectives
trade off on a Pareto frontier rooted in the shared encoder/transformer: decoder capacity
helps denoising and wrecks imputation; transformer capacity helps imputation and wrecks
denoising/DCR. That is not a tuning accident — it's inherent. The two tasks have
**different invariances**:
- denoising must *preserve* within-track intensity and be depth-sensitive;
- imputation must *transfer across* tracks and be depth-invariant / assay-identity-sensitive.

Asking one representation to be simultaneously depth-sensitive and depth-invariant is a
contradiction the optimizer resolves by sitting on a compromise frontier.

Two more issues with the masking *distribution*:
- **The corruption distribution doesn't match deployment.** `_mask_full_assay`
  ([_utils.py:131](_utils.py#L131)) masks `uniform(1, N−1)` assays. Zero-shot deployment is
  "predict all 35 from a handful," i.e. the high-mask-ratio tail the model rarely trains on.
  The training corruption should match the test-time corruption (or be curriculum-scheduled
  toward it).
- **The masker is a Python double `for`-loop over batch and assays** with rejection
  sampling for non-overlapping chunks ([_utils.py:223](_utils.py#L223)). It's both slow and
  hand-tuned. Masking policy is a *modeling decision* and deserves to be a clean,
  vectorized, parameterized distribution, not nested loops.

**→ What I'd change.** Unify corruption into **one noise process** so the model learns a
*single* posterior `p(clean | corrupted)`. Concretely: a corruption operator that composes
(a) per-(assay,position) dropout at a sampled rate and (b) depth downsampling, drawn from a
schedule. Then "impute a whole assay" and "denoise a shallow track" are just two points on
one corruption continuum, supervised by one branch with one weight. This is the MAE/diffusion
framing and it dissolves the obs/imp weight-mismatch problem (loss `obs_weight=3.5 ≫
imp_weight=0.59` while the eval score weights imputation **higher** — a mismatch the AR
explicitly flagged as un-fixable under the frozen regime). If denoising and imputation must
stay separate, then the representation itself should be **task-factored** (Section 6.4), not
the loss.

---

## 5. Architecture

### 5.1 Cross-assay reasoning is too shallow and the assay axis is fixed
Grouped convs keep each assay independent through the whole signal tower
([encoder.py:357](candi_v2/encoder.py#L357)); mask tokens are injected per-assay
([encoder.py:312](candi_v2/encoder.py#L312)); then **one** `LinearFusion`
([encoder.py:499](candi_v2/encoder.py#L499)) concatenates all assay channels with DNA and
projects to `d_model`. From there the transformer attends **only over genomic position**
(L2≈96 tokens); assays mix solely through that one linear layer and the position-wise FFNs.

So the central inferential act of imputation — "given H3K4me3 and CTCF, infer H3K27ac" — is
performed by a dense channel MLP, with no attention over the assay set and **no permutation
structure**. The assay axis is a fixed integer dimension: adding an assay means changing
tensor shapes and retraining. Avocado and eDICE got this right years ago by *factorizing*
into (cell-type, assay, position) embeddings; CANDI threw that away for conv towers and
recovers cross-assay structure only weakly.

**→ What I'd change.** Make assays first-class tokens. Either axial/factorized attention over
(assay × position), or a set-transformer over assays at each latent position. This gives
true cross-assay attention *and* graceful handling of variable/novel assay sets — which is
the entire zero-shot premise.

### 5.2 The decoder's "which assay do I predict" query is weak
Target conditioning is FiLM from the **mean-pooled** target metadata
(`PreDecoderFiLM`, [decoder.py:185](candi_v2/decoder.py#L185); `pooled = meta_embed.mean(dim=1)`).
Pooling over assays *destroys per-assay query specificity* before it ever modulates the
latent; per-assay distinctness only returns at the final linear head, which outputs **all**
assays at once and then you index the one you asked for. That is a very blunt query mechanism
for a model whose headline feature is "ask for any assay."

**→ What I'd change.** A **query-based decoder**: each requested (assay, position) is a query
that cross-attends into the latent (Perceiver-IO / DETR style). The assay-identity embedding
*is* the query, so zero-shot querying of a novel assay is the native operation, and you never
compute outputs for assays you didn't ask for.

### 5.3 FiLM is everywhere, and that is why DCR is brittle
Four covariates are injected via FiLM at: every conv layer, every transformer layer
(`per_conv_and_transformer`), the pre-decoder, and optionally every deconv layer. Depth in
particular flows through **both** the explicit size-factor offset *and* this FiLM web. The
autoresearch loop's most reproducible finding is that *any* new parameter sharing the
depth-calibration gradient path collapses DCR (depth-calibration ratio) — `pre_transformer_bottleneck`,
`learnable_depth_quadratic`, even at σ=0.01 init. That extreme fragility is the signature of
an **over-coupled conditioning design**: many pathways competing to explain the same covariate,
so gradients fight and the calibration sits on a knife-edge.

**→ What I'd change.** Condition *once, cleanly, per role*: depth enters **only** through the
size-factor offset (it's a multiplicative library effect — that's its physics); assay identity
enters **only** as the decoder query; run-type/read-length enter as a small additive bias.
Fewer pathways → less gradient competition → DCR stops being a tightrope, and the whole
config-space stops being a minefield where 18/18 single-knob changes break something.

### 5.4 The DNA tower is under-powered and discards motif-scale information
`DNAConvTower` ([encoder.py:438](candi_v2/encoder.py#L438)) pools 30,000 bp → ~96 positions
with large pooling steps, on raw one-hot, in a shallow stack. Motifs (6–20 bp) — the actual
mechanistic link from sequence to chromatin — are pooled away *before* fusion. Meanwhile
Enformer/Borzoi spend most of their parameters and 100kb+ context exactly here, because
sequence→chromatin is the hard direction.

**→ What I'd change.** Replace the toy DNA tower with frozen/fine-tuned embeddings from a
pretrained genomic model (Borzoi/Enformer trunk, or a DNA LM), fused at a resolution that
preserves motif structure. This is likely the single biggest *ceiling*-raising change for the
sequence-driven assays.

### 5.5 Context length caps the biology
30 kb / 1,200 bins / L2≈96 latent tokens. Enhancer–promoter and TAD-scale dependencies
(10 kb–Mb) are simply out of receptive field, and a 96-token transformer is tiny. Whatever
the heads do, long-range regulatory structure is unreachable.

**→ What I'd change.** Longer context with hierarchical/dilated attention, or a two-scale
model (fine local + coarse long-range), so the architecture isn't structurally blind to the
regulatory distances that matter.

---

## 6. Losses & optimization

- **Six static weights** (count/pval/peak × obs/imp) plus optional Kendall uncertainty
  weighting plus optional EMA assay balancing plus an R_stable objective
  ([candi_loss.py](candi_loss.py)). This is a lot of machinery to balance heads that
  (Section 2) shouldn't all exist. The frozen weights don't match the eval score — a known,
  load-bearing mismatch.
- **The obs branch is near-trivial and weighted 3.5×.** Reconstructing the *visible*
  tracks is close to an identity/denoise-in-place task and can leak; pouring 3.5× of the
  gradient budget there optimizes mostly the easy half.
- **No calibration term despite calibration being a headline deliverable.** Calibration and
  C-index are *hoped to emerge* from NLL, not optimized. Proper scoring with an explicit
  calibration/sharpness penalty (or CRPS) would target the actual objective.
- **NB NLL rewards mean-matching.** For the denoising (super-resolution) target, NB NLL can
  be minimized by matching the conditional mean while getting the *noise structure* wrong —
  which is exactly the thing uncertainty estimates are supposed to capture.

**→ What I'd change.** Fewer heads (Section 2) → fewer weights; learn the remaining trade-off
via principled uncertainty weighting; add an explicit calibration objective; use focal/balanced
loss for the (now-derived) peak readout.

---

## 7. Training loop & model selection

- **Selection on eval `total_loss`** ([train_candi_v2.py](candi_v2/../train_candi_v2.py),
  `best_eval_total_loss`), which is the obs-dominated weighted sum — *not* the downstream
  score (imputation R² / biological validation). We are early-stopping and checkpointing on
  a proxy that the AR memory shows is misaligned with the score by construction.
- **Regime caution.** The sandbox is 8 assays, `batch_size=4`, ~10–40 epochs, on a 10GB MIG
  slice. The model overfits past ~epoch 20 and capacity experiments under-converge at 10. It
  is entirely possible that the "fundamental" Pareto frontier is partly an artifact of this
  tiny data/compute regime and would shift at production scale (35 assays, 3,064 biosamples).
  Architectural conclusions from the sandbox should be re-validated before being treated as
  laws.

**→ What I'd change.** Select checkpoints on the actual downstream metric; treat sandbox
findings as hypotheses to confirm at scale, not as settled physics; add weight EMA/SWA given
how quickly the small model overfits.

---

## 8. The `Z` paradox — the decoders are a bottleneck

The most telling result in the project guide: **latent `Z` is the strongest, most
sparsity-robust predictor of RNA-seq — stronger than the model's own denoised+imputed
outputs.** That means the generative heads are *discarding* biology that the representation
already contains. The reconstruction decoders are an information bottleneck, and we are
selecting/optimizing the model on reconstruction loss while the *useful* product is the
latent.

**→ What I'd change.** Lean into representation learning. The JEPA work in `lejepa/`/
`sandbox/jepa_*` is the right instinct: optimize the latent directly (predict masked-assay
*representations*, not pixels), and treat the generative heads as lightweight,
internally-consistent readouts of a strong `Z` rather than the main event. If `Z` already
beats the outputs, the outputs are not where the value is.

---

## 9. A first-principles redesign sketch

If I started over, keeping CANDI's genuine strengths (raw counts, size-factor offset,
distinct missing/cloze, distributional uncertainty, no-RNA-leak evaluation):

1. **One latent enrichment field, one likelihood.** Predict `λ(assay,pos)`; counts ~
   NB(λ·size_factor, φ); pval and peaks are *calibrated deterministic readouts* of the λ
   posterior and the local control λ. One learned likelihood, internally consistent
   uncertainty, peak probability = `P(λ>τ)`.
2. **Control as explicit local background**, so the network predicts depth/background-invariant
   *enrichment* — the right zero-shot quantity — and control stops being double-counted.
3. **Assays as tokens.** Factorized/axial attention over (assay × position); set-structured so
   novel assays are native, not a reshape-and-retrain.
4. **Query decoder.** Requested (assay,pos) cross-attends into `Z`; assay identity is the query.
5. **One unified corruption process** (dropout ⊕ depth-downsample on a schedule) →
   *one* denoising posterior, eliminating the imp/den adversarial frontier; or, if kept
   separate, **factor the representation by task**, not the loss.
6. **Condition once per covariate role** (depth→offset only; assay→query only) to kill the DCR
   fragility at the root.
7. **Real sequence model** (pretrained DNA trunk) and **longer context** to raise the biology
   ceiling.
8. **Representation-first training** (JEPA-style latent prediction) with the generative heads as
   calibrated readouts; **select on downstream metrics**, not obs-weighted reconstruction loss.

The unifying move: stop training one shared backbone to satisfy three redundant likelihoods and
two adversarial corruption tasks under a dozen FiLM pathways. Predict **one** quantity, condition
**once**, reason over **assays as a set**, and read everything else out consistently from a strong
latent.

---

## 10. Caveats

- This critiques *design*, not correctness; the implementation is careful (sentinel-safe
  transforms, NB numerical guards, availability/supervision consistency checks).
- Several recommendations (pretrained DNA trunk, longer context, factorized attention) raise
  compute well beyond the sandbox; they're production-regime bets, not sandbox knobs.
- The Pareto-frontier and DCR-fragility findings are robust *within* the frozen sandbox
  regime; whether they survive at production scale is itself the first experiment I'd run.
</content>

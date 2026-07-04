# Dual metadata-conditioning — implementation plan

Tracked in crux as **q15** with hypotheses **h30** (learnable, full matrix), **h31** (composes to
unseen cells), **h32** (invertibility sets difficulty; invert-input harder than apply-output).
This doc is the implementation contract; nothing here is launched until the PI greenlights.

## Scientific question (recap)
Can CANDI (i) **normalize** a covariate-transformed count INPUT (encoder, `x_meta`) and (ii)
**steer** the count OUTPUT under an independent covariate (decoder, `y_meta`), when the transform
"type" is given as metadata? Controlled synthetic testbed — we own every transform, so we have exact
ground truth. Motivated by covariate_probes (precondition: covariate fingerprint recoverable) and
q9/h19 (the REAL `y_meta` pathway collapses depth, DCR~1).

## Data & base signal
- `sandbox/data/sandbox.h5`, **`counts_dsf1`** [W,768,8] as the base signal `x=y`. 8 assays
  (ATAC, DNase, H3K4me3, H3K4me1, H3K27ac, H3K27me3, H3K36me3, H3K9me3).
- **chr19 (3053 win) → train · chr21 (2432 win) → test.** DSF **off** (fixed depth, DSF=1 only).
- Availability per `(biosample, assay)` from `meta_dsf1[0] != -1`. Pool **all 10 biosamples'** available
  tracks (T_/V_/B_ all carry `counts_dsf1`) → ~38 `(biosample,assay)` tracks. Cell-type is NOT a
  holdout axis here (only 5 T_); the generalization axes are **chromosome** and (h31) **matrix cell**.
- Unavailable assays excluded from loss via an **explicit availability mask** (never the in-band −1
  sentinel — see collision note below).

## Transforms (count → non-negative integer)
Applied to both `f_x` (input) and `f_y` (output), drawn **independently per assay**. Identity always
included as the M1/M3 reference. `family_id`: 0=identity, 1=mult, 2=add, 3=power, 4=thin, 5=cap, 6=clog.

| family | def | param(s) | class |
|--------|-----|----------|-------|
| identity | `y` | — (ref) | ref |
| mult `×h` | `round(y·h)` | h∈{0.25,0.5,2,4} | invertible (depth-rescale) |
| add `+h` | `max(0, round(y+h))` | h∈{2,5,10,20} | invertible (bg shift) |
| power | `round(y**h)` | h∈{0.5,0.75,1.5,2} | invertible (dyn-range) |
| thin | `Binomial(y,p)` | p∈{0.2,0.4,0.6,0.8} | **non-invertible**, det-seeded |
| cap | `min(y, c)` | c∈{2,5,10,20} | **non-invertible** (censor) |
| clog | `round(a·log1p(y))` | a∈{1,2,4,8} | non-invertible (compress) |

**Granularity RESOLVED: 4 param values per non-identity family** → 6×4=24 +identity = **25 conditions
per side**; the family-level matrix stays **7×7 = 49 cells** (params aggregated within a family for
h30/h31/h32 reporting; the 4-value grid gives M2 its steering points).

- **Deterministic seeding** for `thin`: RNG keyed by `(biosample, assay, window_start, family, param)`
  so the target is bit-identical across epochs (confirmed reproducible in validate.py).
- **Param encoding is itself a hypothesis (h33), not a fixed default.** `aug_param` is a scalar the
  model reads via `Linear(1→d)`; its raw scale varies wildly across families (mult 0.25–4 vs add/cap up
  to 20), so magnitudes collide. We run the full h30 matrix under **three arms — none / per-family
  z-score / global log-scale** — and h33 reads off the comparison. `aug_family` disambiguates semantics
  in all arms. Whichever arm wins is the encoding h31/h32 then use.

## Metadata schema (the isolation choice)
`x_meta`, `y_meta` reduced to **[B, 2, F]** = row0 `aug_family` (categorical), row1 `aug_param`
(continuous). The real 4 rows (depth/assay_id/read_length/run_type) are **discarded** so any metadata
sensitivity is provably from the augmentation knob. New module `DualCondMetaEmbedder`:
`Embedding(7 + {missing}) ⊕ Linear(1→d)` (param, with a missing/sentinel vector) → fuse → [B,F,d].
Consequence: no `assay_id` — acceptable ONLY because denoising-only (input carries the assay's signal;
output is positional). Re-adding imputation later would need assay_id back.

## Model (CANDIv2, denoising-only, NB + NBNLL)
- **Base = CANDIv2** (`sandbox/candi_v2/`): `encode(x_data, x_dna, x_meta) → z`; `decoder(z, y_meta) → out`.
  Winning single-shot `y_meta` `PreDecoderFiLM` (h18), exposed latent `z` (needed for M3).
- **Integration (first impl task):** audit how tightly `V2Encoder`/`V2Decoder` couple to the 4-row
  `MetadataEmbedding`. Prefer **import-and-swap** (reuse ConvTower/FiLM/PreDecoderFiLM/NB head, replace
  only the embedder); fall back to a **minimal local fork** if coupling is too tight. Do NOT edit
  `candi_v2/` in place (protects v2/v3 work) unless the swap requires a tiny, additive config hook.
- **Output head = NB `(p, n)`**, trained with **NBNLL** on `y'` (integer counts). Gaussian/peak heads off.
- **Input:** feed `arcsinh(f_x(counts))` as the count channel, model's own `encode_input_transform`
  set to `none` (arcsinh tolerates the negatives `f_x` can produce; avoids the −1 sentinel collision).
- **Control channel & DNA:** kept for architectural fidelity but inert here (control neutral; DNA is
  constant across transforms so cannot leak transform info). Revisit if it complicates the swap.

## Task & batch construction
- **Denoising-only**: no cloze masking; all available assays visible in input, all supervised.
- Per batch window × biosample: load `counts_dsf1` for available assays; per available assay draw
  `(f_x,h_x)`, `(f_y,h_y)` independently from the allowed cell set; build `x'`, `y'`, `x_meta`, `y_meta`.
- Loss = NBNLL over available (observed) positions only. No imputation branch.
- **Off-diagonal cells (`f_x ≠ f_y`) are the real test** — the model must undo `f_x` and apply `f_y`.

## Splits
- Global: chr19 train / chr21 test.
- **h30 (hA):** all 49 cells trainable; evaluate per-cell on chr21.
- **h31 (hB):** hold out a set of cells at **intersections of seen rows & seen columns** (every family
  still appears on both sides, only specific pairings hidden); evaluate held-out vs seen on chr21.

## Metrics
- **M1 (end-to-end):** per cell `(f_x,f_y)`, R² of predicted NB-mean vs target `y'` over chr21
  positions; headline = per-cell R² and the **gap to the identity-cell ceiling** (`R²(identity,identity)`).
  Also NBNLL-vs-ceiling. Off-diagonal weighted (diagonal is near-trivial).
- **M2 (decoder/output steering):** fix inputs `(x',h_x)`; sweep `h_y` over a family; `Δpred = NBmean(h_y=j) −
  NBmean(ref)`, `Δtarget = f_j(base) − f_ref(base)`; **M2 = R²(Δpred, Δtarget)** pooled, per family.
  Baseline = h_y-ignoring (Δpred≈0 → R²≤0). Generalizes `sandbox/eval.py::prompt_sensitivity_*`.
- **M3 (encoder/input invariance):** `z_id = Enc(arcsinh(base), h_x=id)`, `z_aug = Enc(arcsinh(f_x(base)), h_x)`;
  within-base = mean over transforms `cos_dist(z_aug, z_id)`; between-base = mean over different bases
  `cos_dist(z_id^a, z_id^b)`; **M3 = within/between**, want ≪1. Guard: require between-base above a
  floor (not collapsed) AND M1>0.

## Validation (`validate.py` — HARD gates block the sweep; SOFT are reported)
Correctness/leakage (HARD):
1. **Analytic metric oracles:** true-target-as-pred → M1 == ceiling (gap≈0); correct-`f_y` oracle → M2≈1;
   base-invariant encoder → M3≈0. Pins metric + data plumbing without the model.
2. **Metadata causally wired:** perturbing `h_y` changes output; perturbing `h_x` changes `z`
   (nonzero grad of output w.r.t. meta embedding; `out(h_y=A)≠out(h_y=B)`).
3. **No crosswiring:** `h_y` must not reach the encoder, `h_x` must not reach the decoder (perturbation test).
4. **Shuffled-`h_y` control:** randomize output covariate, keep true target → M1 & M2 collapse. A
   surviving positive ⇒ leakage / trivial-copy. **Shuffled-`h_x` control:** M3 collapses.
5. **Transform unit tests:** each family matches hand-computed values; identity exact; outputs valid
   non-neg ints; −1 sentinel never collides; per-assay `h` independence.
6. **Determinism:** fixed seed → identical split & metrics; `thin` bit-identical across epochs.

Capability (HARD unless noted):
7. **Overfit-tiny:** few windows, full matrix → M1→ceiling, M2 high (NBNLL optimization capable).
8. **Identity-only run:** `f_x=f_y=id` everywhere → plain denoising; M1 hits the model's normal
   reconstruction ceiling (this DEFINES the ceiling). SOFT (informational baseline).
9. **NBNLL sanity:** loss ↓, NB-mean tracks counts, no NaNs.

Honesty: report per-cell spread; for h31 held-out cells state effective n and lean on the
memorization-baseline contrast rather than raw thresholds (avoids the optimistic-CI trap seen in
covariate_probes).

## Training protocol & resources
- Custom lightweight loop (covariate_probes-style; GPU-resident where feasible) — cleaner than the
  full `train.py` for a single-head NB denoising task with bespoke M1/M2/M3.
- Optimizer/clip: sandbox defaults (adamax, clip_norm 1.0). Epochs set from a `converge.py` plateau
  check (start 15–30). Profile a single cell×split for GPU mem / wall before the sweep.
- **SLURM GPU (hard constraint): `--gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1`** on every job.

## Module layout
```
sandbox/diagnostics/dual_conditioning/
  plan.md          this
  transforms.py    6 families + h-encoding + deterministic seeding
  data.py          base-count loader (wraps SandboxH5Dataset) → transformed views + 2-row meta + splits
  model.py         DualCondMetaEmbedder + CANDIv2 wiring (denoising-only NB)
  metrics.py       M1, M2, M3
  validate.py      HARD/SOFT correctness + negative controls
  converge.py      epoch-plateau check
  run.py           train/eval sweep for h30/h31/h32
  plot.py report.py
  jobs/  results/
```

## Open implementation decisions (my defaults; redline before impl)
1. **Model integration:** import-and-swap v2 blocks (default) vs minimal fork — decided after the coupling audit.
2. ~~Param normalization~~ **RESOLVED → promoted to hypothesis h33** (test all three: none / per-family
   z-score / log-scale; h30 matrix run under each arm).
3. ~~Matrix param granularity~~ **RESOLVED → 4 param values per non-identity family** (25 conditions/side; 49 family cells).
4. **h31 holdout cells:** exact set (intersection design) — pick after hA sizing.
5. **Keep vs drop DNA/control:** keep inert (default) vs drop for simplicity.

## Mapping to crux
h30 ← M1+M2+M3 on the full seen matrix · h31 ← held-out cells + memorization baseline ·
h32 ← per-family difficulty ranking + M3(input) vs M2(output) asymmetry · **h33 ← the 3-arm param-
normalization comparison (none / z-score / log), read off the h30 runs**. `close` each by ticking its
pre-registered verifiables once the validated sweep completes.

## Implementation order (when greenlit)
1. `transforms.py` + unit tests. 2. `data.py` + shape/leakage asserts. 3. coupling audit → `model.py`
(2-row embedder + NB). 4. `metrics.py` + analytic-oracle tests. 5. `validate.py` (all gates). 6. tiny
smoke (CPU) → profile (1 GPU cell). 7. `converge.py`. 8. sweep h30 → h31 → h32.

## Implementation notes (resolved during the build — 2026-07-03)
- **Integration = import-and-swap (confirmed).** With `heads="count_only"` + `count_head="plain"`, the
  decoder reads `y_meta` ONLY through its embedder (the depth-offset path that indexes row 0 is off),
  and the encoder's only raw metadata use is a shape-only `ones_like(meta[:,0,:])`. So `model.py` builds
  CANDIv2 normally and replaces `encoder.metadata_embedding` + `decoder.decoder_meta_embedding` with
  `DualCondMetaEmbedder`. **No `candi_v2/` edits.**
- **Input:** feed RAW transformed counts `x'` (>=0 int; -1 for missing assays) as the count channel with
  encoder `signal_transform="arcsinh"` (all 6 families output >=0, so no sentinel collision; the encoder
  handles compression). Availability via `mask_token` needs signal- and meta-availability to AGREE, so
  missing assays are -1 in both, and the control channel's meta is set to -1 where the h5 stores -1 control.
- **Param stored RAW-positive; normalized INSIDE the embedder** (h33 arm = a model knob) — a normalized
  param can never hit the -1/-2 sentinels.
- **`make_batch` decouples the applied transform from the metadata covariate** (`fam_ym`/`par_ym` etc.),
  so the shuffle leakage control feeds a WRONG covariate with the TRUE target (a self-consistent
  permutation would otherwise be trivially learnable and not break the covariate->target link).
- Model is ~0.30M params; `z` is [B, 96, 72] (d_model=72). All 21 CPU pre-flight gates pass.
- **Conditions are UNIFORM per batch (one matrix cell/batch), NOT per-assay.** First gate run exposed why:
  the fixed v2 decoder's `PreDecoderFiLM` pools `y_meta` across assays (`meta_embed.mean(dim=1)`) into a
  single global scale/shift, so per-assay-independent `h_y` averages out and the decoder cannot steer
  per assay (M2≈0). This is the same limitation that motivated CANDI's query decoders ([[q5]]). So the
  testbed conditions the whole track by one cell; the shuffle control feeds an independently-drawn WRONG
  cell (not a permutation). Design finding, not just a fix — output conditioning here is track-level.

## Findings so far (2026-07-03, converge/debug phase — pre-sweep)
The correctness pipeline is fully validated (21 CPU gates pass). Getting the model to actually train
surfaced three real issues, each fixed, and then a genuine scientific result:

1. **Decoder pools y_meta** (fixed v2 `PreDecoderFiLM`/`per_deconv` use `meta.mean(dim=1)`) → per-assay
   conditioning impossible → switched to **uniform-per-batch conditions** (one matrix cell/batch).
2. **Training instability**: base counts are heavy-tailed (max ~10k) and `power^2` pushed NB targets to
   ~1e8 → NBNLL diverged, M1 degraded with training. Fixed: **winsorize counts at 128**, drop the
   squaring exponent (`power` max 1.5), **LR warmup+cosine @5e-4**. Reconstruction then stable/positive.
3. **Decoder conditioning init**: single-shot FiLM → M2≈0; per-layer FiLM (xavier+bias) → M2 up slightly
   but M1 re-destabilized; **adaLN-zero** (zero-init decoder FiLM) → stable M1 AND cleanest conditioning
   activation (h_y influence 0→0.30 in 30 steps).

**Result (robust across all three decoder configs):**
- **Encoder input-understanding WORKS** — M3 within/between latent ratio ~0.04–0.10 (encoder normalizes
  the input transform while staying discriminative). Reconstruction M1 ceiling reaches ~0.20–0.23.
- **Decoder output-steering does NOT emerge** — M2 (identity input, sweep h_y) stays ~0.01–0.03 and does
  not rise with training, even as M1 climbs. adaLN-zero trajectory M2_inv: 0.020→0.029→0.021→0.008 (flat).
- **Mechanism:** in denoising-only the input f_x(base) is correlated with the target f_y(base), so the
  model reconstructs transformed cells via the INPUT rather than by STEERING from the output covariate.
  When the input can't help (M2's identity-input sweep), steering collapses. This is the same
  "metadata pathway collapse" seen in real CANDI ([[q9]]/h19) — reproduced under full control.

**Open fork (needs PI decision):**
- (A) **Positive control** — force `f_x=identity` in training so the output covariate is NECESSARY; if M2
  then rises sharply, the finding is "steering is learnable but suppressed by the denoising shortcut."
- (B) Run the 4-task sweep as-is to document the input/output asymmetry across norm arms + holdout.
- (C) Redesign toward a regime that forces output-covariate use (masking/imputation, input-decorrelation).

# RECIPE — architecture, training procedure, and why each choice is what it is

This is the scientific core of the handoff. It documents the model in `candi_kit` module by module with
exact tensor shapes, states which choices are **load-bearing** (changing them changes the result) versus
**free** (tune at will), and records what has already been tried and refuted so you do not repeat it.

**Read this first:**

- **No checkpoints ship with this kit.** You train from scratch. The `.pt` files under
  `src/candi_kit/goldens/` are ~248 KB *forward-output* tensors used by the bit-exactness gate
  (`python -m candi_kit.compat`), not model weights. Historical checkpoints exist only in the research
  repo and were used solely as an internal correctness gate.
- **The model is counts-only Negative Binomial.** There is no Gaussian p-value head and no Bernoulli
  peak head, therefore no peak precision/recall/AUROC and no p-value track. `y_pval` and `y_peaks` are
  carried through the batch dict (`src/candi_kit/batch.py:144-146`) but never supervised. See
  `EXTENSION_HOOKS.md` to add them.
- **Every number in this document carries a noise floor.** Effective replication in the recorded
  experiments is **12 held-out targets / 5 biosample pairs / 4 cell types**, with `T_RWPE2`/`B_RWPE2`
  supplying 7 of the 12. The target-clustered bootstrap noise floor on macro CRPS is **~0.09**;
  per-comparison uncertainty is **±0.13**; a **single seed change** moves pooled imputation CRPS by
  **0.1195** and Spearman by **0.0562**. Do not read 4-decimal orderings as rankings.

---

## 1. Input contract

One training sample is one genomic window. With the shipped q19 panel (`configs/panel.q19.json`:
8 assays, `context_bins=768`, `resolution=25`):

| tensor | shape | dtype | meaning |
|---|---|---|---|
| `x_data` | `[B, 768, 9]` | float32 | raw integer counts, input side. Columns `0..7` = assays, column `8` = ChIP control (appended at `batch.py:121`) |
| `x_dna` | `[B, 19200, 4]` | float32 | one-hot DNA, `context_bins × resolution` bp. The encoder accepts `[B,4,G]` or `[B,G,4]` (`encoder.py:516-523`) |
| `x_meta` | `[B, 4, 9]` | float32 | covariates, input side, incl. the control column |
| `y_meta` | `[B, 4, 8]` | float32 | covariates, **target** side. Control-free by construction — the control column is appended to `x_meta` only (`batch.py:122`) |
| `y_data` | `[B, 768, 8]` | float32 | raw integer counts, target side |
| `observed_map` / `masked_map` | `[B, 768, 8]` | bool | loss masks. **`obs` = unmasked, `imp` = masked (cloze)** — these are *not* biological observed/imputed |

The 4 covariate rows are fixed by the ingestion contract and are **not** configurable:

| row | covariate | type | range in the q19 panel |
|---|---|---|---|
| 0 | `log2(sequencing_depth)` | continuous | natural (DSF-1) 22.20 – 28.13, mean **25.117**; **as actually prompted during training** (T_ × uniform DSF{1,2,4,8}) mean **23.227**, sd 1.841, range **[19.20, 28.13]** (`METADATA_AUDIT.md:114`) |
| 1 | `assay_id` | categorical | `0..num_assays-1` real assays; `num_assays` = ChIP control |
| 2 | `read_length` | continuous | {30, 36, 76, 100, 101} bp |
| 3 | `run_type` | categorical | {0 = single, 1 = paired} |

Two sentinels, kept distinct everywhere (`src/candi_kit/_vendored.py`): **`MISSING = -1`** (assay does
not exist for this biosample) and **`CLOZE = -2`** (exists, masked, must be imputed).

`assay_id` embedding indices are `Embedding(num_assays + 3)`: `0..A-1` real, `A` = control, `A+1` =
MISSING, `A+2` = CLOZE (`encoder.py:104`, `encoder.py:169-178`). The `+3` is correct; the comment at
`encoder.py:103` omits the control slot and is wrong — do not "fix" it to `+2`.

`MetadataEmbedding.forward` now **raises** rather than silently aliasing on a malformed input: a
non-4-row tensor (`encoder.py:142-146`), `assay_id > num_assays` (`encoder.py:163-168`), and
`run_type >= num_runtypes` (`encoder.py:183-188`).

---

## 2. Architecture walkthrough

```
x_meta[B,4,9] ──► MetadataEmbedding #1 ──► meta_embed[B,9,32] ─┬─► FiLM after conv0
                                                               ├─► FiLM after conv1
                                                               └─► FiLM after conv2
x_data[B,768,9] ─► zero masked chans ─► SignalConvTower ─► [B,96,72] ─► MaskTokenInjector ─┐
x_dna[B,19200,4] ────────────────► DNAConvTower (NO metadata) ─► [B,96,72] ───────────────┤
                                                                                          ▼
                                                              LinearFusion ─► [B,96,72] ─► 2× x-transformers (RoPE, NO FiLM)
                                                                                          ▼
                                                                                     z[B,96,72]
                                                                                          ▼
z ─► DecoderTrunk (3× deconv, NO metadata) ─► feat[B,768,128] ─► view[B,768,8,16]
y_meta[B,4,8] ─► MetadataEmbedding #2 (SEPARATE, UNTIED) ─► memb[B,8,32] ─► film_proj (adaLN-zero) ─► γ,β [B,8,16]
                                                                                          ▼
                                                 feat·(1+γ)+β ─► head_eta / head_n (weight-shared) ─► η, raw_n  [B,768,8]
y_meta row 0 ────────────────► arithmetic depth offset (bypasses the embedder) ─────────► log2_mu ─► μ, n, p
```

**Two disjoint metadata pathways, two separate untied embedders.** The encoder embeds `x_meta`
("what the input measurement was"); the decoder embeds `y_meta` ("what measurement to produce"). They
are different `nn.Module` instances with independent weights (`model.py:217-218` vs `model.py:156-157`).

### 2.1 Metadata embedder (`encoder.py:70-202`)

`[B, 4, F] → [B, F, 32]`. Per-field lift, then late fusion:

| field | branch | line |
|---|---|---|
| `log2_depth` | `nn.Linear(1, 32)` | `encoder.py:95` |
| `read_length` | `nn.Linear(1, 32)` | `encoder.py:96` |
| `assay_id` | `nn.Embedding(A+3, 32)` | `encoder.py:104` |
| `run_type` | `nn.Embedding(num_runtypes+2, 32)` | `encoder.py:106` |

then `concat(4×32) → Linear(128,32) → GELU → Linear(32,32) → LayerNorm(32)` (`encoder.py:109-116`;
concat order at `encoder.py:201` is depth, assay, readlen, runtype).

Sentinels are handled **per field, independently**. Continuous fields project everything and then
*overwrite* at sentinel positions with four learned vectors initialised `randn(32)*0.02`
(`encoder.py:118-133`). Categorical fields remap the sentinel to a reserved row before lookup. This is
what makes the imputation prompt work: you can hand the decoder a target's `assay_id`, `depth` and
`run_type` while its *signal* is absent.

The wasted `proj(-1)` / `proj(-2)` FLOPs do **not** leak gradient: with all-sentinel depth,
`depth_proj.weight.grad.norm() == 0.0` while `depth_missing_emb.grad.norm() == 0.194`. The in-place
`emb[mask] = ...` does not break autograd (`index_put_` has a defined backward).

### 2.2 Signal conv tower — grouped, per-assay, FiLM after every conv (`encoder.py:391-469`)

`groups = num_tracks = A+1 = 9`, so each assay's channels are processed independently. Channel schedule
with `expansion_factor=2`, `n_cnn_layers=3`, `pool_size=2`:

| conv layer | in→out channels | per-assay C | sequence length | FiLM projection |
|---|---|---|---|---|
| 0 | 9 → 18 | 2 | 768 → 384 | `Linear(32 → 4)` |
| 1 | 18 → 36 | 4 | 384 → 192 | `Linear(32 → 8)` |
| 2 | 36 → 72 | 8 | 192 → 96 | `Linear(32 → 16)` |

Each `ConvTower` is `Conv1d(same) → LayerNorm → + 1×1 residual → GELU → MaxPool`
(`encoder.py:249-270`). One **independently learned** `FiLMLayer` is applied after *each* block
(`encoder.py:445-449`, `encoder.py:462-465`); because each block pools, layer 2 conditions at 8× the
receptive field of layer 0. `pre_film` and `post_film` are `None` under `film_mode="per_conv"`.

FiLM is `x ← x·(1+scale) + shift` where `(scale, shift) = proj(meta_embed)` — **metadata-only** (the
activations are only ever the operand) and **broadcast over all positions**
(`encoder.py:290-299`). There is zero per-position specificity. This is textbook FiLM (Perez et al.).

> **Known caveat, not a bug.** `conv_norm="layer"` normalises over the *full* channel axis, mixing all
> 9 tracks' statistics at every layer — a measured **~37% cross-assay leak**. `conv_norm="group"`
> removes it exactly (0.0000 delta on non-modulated assays) **but also cancels much of the per-assay
> FiLM's own effect** (relative change 0.117 → 0.073). It is a tradeoff, not a free fix.

### 2.3 Mask token injection (`encoder.py:346-384`)

After the conv tower and **before** DNA fusion, each assay flagged CLOZE or MISSING has its
`d_per_assay=8`-wide channel slice replaced by a learned per-assay vector
(`nn.Parameter(randn(9, 8)*0.02)`). Availability is derived from **all four** metadata rows with `.any`
(`encoder.py:609-616`), and `_prepare_signal` raises if metadata- and signal-derived availability
disagree (`encoder.py:890-896`) — this is the assertion an ad-hoc synthetic batch will trip.

### 2.4 DNA tower (`encoder.py:472-526`)

Ungrouped, no metadata. `n_cnn_layers + 2 = 5` `ConvTower` blocks; channel schedule
`[4] + exponential_linspace_int(4, 72, 5)`. Under `dna_pool_order="late"` the first 3 blocks pool by
`pool_size=2` and the last 2 by `dna_pool_size=5`, giving a total stride of `8 × 25 = 200`:
`19200 → 96`. Output `[B, 96, 72]`, matching the signal tower's length.

There is **no counts-only path that skips DNA** — `V2Encoder` always builds and calls `dna_tower`.

### 2.5 Fusion and transformer (`encoder.py:533-568`, `encoder.py:841-852`)

`LinearFusion`: `concat([signal 72, dna 72]) → Linear(144, d_model) → GELU → Identity → Dropout(0.1)`.
(`fusion_norm` defaults to `"none"`, so the norm is `nn.Identity`; `fusion_deep=False`, so
`hidden_projs` is an empty `ModuleList`.)

Transformer: **2 independent x-transformers `Encoder` blocks**, each `depth=1, heads=4,
rotary_pos_emb=True, ff_mult=4, pre_norm=True`, `attn_dropout = ff_dropout = 0.1`. **No FiLM on the
transformer** — `transformer_film_layers` is built only under `film_mode="per_conv_and_transformer"`
and is `None` here. `pooled_meta = meta_embed.mean(dim=1)` is computed at `encoder.py:950` but consumed
only by that dead branch. `output_norm` is `nn.Identity` (`output_rms_norm=False`).

Output `z: [B, 96, 72]`.

> x-transformers `dim_head` is 64, so attention inner dim is `nhead × 64` **regardless of `d_model`**.
> Capacity does not scale with panel size unless you raise `--nhead`.

### 2.6 Decoder trunk (`decoder.py:96-185`)

`signal_dim = A × feat_per_assay = 8 × 16 = 128`;
`decoder_input_dim = signal_dim × expansion_factor**n_cnn_layers = 128 × 8 = 1024`.

`input_proj = Linear(72, 1024)`, then 3 `DeconvTower` blocks (`ConvTranspose1d(stride=2) → LayerNorm`,
plus a `1×1` transposed residual, then GELU) over the channel schedule `[1024, 512, 256, 128]`. Length
`96 → 192 → 384 → 768`.

Output `feat: [B, 768, 128]`, viewed as `[B, 768, 8, 16]`.

**The trunk is metadata-blind.** Note *why*: `film_layers` and `pooled_meta` are `forward()` arguments
(`decoder.py:166-171`), and the kit simply never passes them. Enabling per-deconv FiLM is a
forward-call change, not architecture surgery. Production `V2Decoder` does construct and pass them —
this kit's decoder is deliberately the simpler one.

### 2.7 Decoder FiLM — per-assay, adaLN-zero (`model.py:149-151`, `model.py:166-170`)

```python
memb = self.meta_embedding(y_meta.float())          # [B, 8, 32]  SEPARATE untied embedder
gamma, beta = self.film_proj(memb).chunk(2, dim=-1) # [B, 8, 16] each
feat = feat * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)   # broadcast over all 768 positions
```

`film_proj = nn.Linear(32, 32)` with **weight and bias both zero-initialised** (`model.py:150-151`) —
adaLN-zero, so the decoder FiLM is a bit-exact identity at step 0 and steering grows from there.

This is the **only** output-side FiLM: exactly once, after the whole trunk, before the heads. It is
1,056 parameters — **0.034% of the model** — and it is the entire learned output-steering path.

Cross-assay leak here is **exactly 0.0**, unconditionally, because `pool_meta=False`: perturbing one
assay's `y_meta` column changes no other assay's output.

### 2.8 Heads and the NB arithmetic (`model.py:153-186`)

Two weight-shared heads applied on the last dim of `[B, 768, 8, 16]`:

```python
head_eta = Sequential(Linear(16,16), GELU(), Linear(16,1))   # 289 params
head_n   = Sequential(Linear(16,16), GELU(), Linear(16,1))   # 289 params
eta   = head_eta(feat).squeeze(-1)    # [B, 768, 8]
raw_n = head_n(feat).squeeze(-1)      # [B, 768, 8]
```

Then the depth-offset log-link, **exactly** as implemented at `model.py:175-186`:

```python
depth = y_meta[:, 0, :]                                  # [B, 8]   depth_row = 0
valid = (depth != MISSING) & (depth != CLOZE)            # [B, 8]

if use_offset:                                           # offset ON
    d_off   = (depth - depth_center).unsqueeze(1)        # [B, 1, 8]
    log2_mu = torch.where(valid.unsqueeze(1), d_off + eta, eta)
else:                                                    # offset OFF
    log2_mu = eta

log2_mu = log2_mu.clamp(-15.0, 30.0)                     # log2_mu_clamp
mu      = torch.pow(2.0, log2_mu).clamp_min(1e-6)        # mu_eps
n       = F.softplus(raw_n) + 1e-6
p       = (n / (n + mu)).clamp(1e-6, 1.0 - 1e-6)
```

Returns `dict(p, n, eta, log2_mu, mu)`, each `[B, 768, 8]`. The NB mean is `mu`; `eta` is the
offset-free mean statistic used by the steering diagnostics.

**Read the clamp carefully.** `log2_mu.clamp(-15, 30)` saturates. In the recorded runs the *median*
clamp fraction is 0 on all four arms, but a minority of targets read the depth slope through the
floor: **16.9%** of targets clamp somewhere on the offset-ON `wd=0` arm (p90 clamp fraction 0.475,
max 0.967), 15.1% on offset-OFF `wd=0`, 6.3% and 0.2% on the two `wd=1e-4` arms. Any depth-slope number
quoted for an offset-ON arm is partly read through that clamp.

### 2.9 Parameter budget (runtime-verified, q19 defaults)

| module | params | note |
|---|---|---|
| **TOTAL** | **3,103,194** | |
| encoder | 264,728 | |
| ├ `metadata_embedding` | 5,984 | |
| ├ `signal_tower` | 2,940 | of which **FiLM projections 924** |
| ├ `dna_tower` | 13,740 | |
| └ `mask_injector` | 72 | |
| decoder | 2,838,466 | |
| ├ `trunk` (deconv) | 2,830,848 | **91.2% of the whole model** |
| ├ `meta_embedding` | 5,984 | |
| ├ **`film_proj`** | **1,056** | **0.034% — the entire learned output-steering path** |
| └ `head_eta` / `head_n` | 289 / 289 | |

The remainder of the encoder budget is the fusion projection and the two transformer blocks.

---

## 3. Training recipe

One `python -m candi_kit.train` invocation = one arm: train on the h5's `train_chroms`, then evaluate
on `eval_chroms`, writing `{out_dir}/{tag}.json` and `{out_dir}/{tag}.ckpt`.

```bash
python -m candi_kit.train \
  --h5 /scratch/$USER/mypanel.h5 --out-dir /scratch/$USER/kit_out \
  --offset on --seed 0 \
  --weight-decay 0.0 --dsf-sampling uniform --epochs 25 --batch-size 8 --full-coverage \
  --eval-batch-size 4 --eval-max-batches 0 --eval-budget 50000000 --m3-regions 40 \
  --fg-frac 0.02 --n-boot 1000 --tag kit_on_s0
```

### 3.1 Masking

`make_masker(p_full_assay=1.0, p_full_loci=0.0, p_chunks=0.0)` (`train.py:96-97`) — **whole-assay cloze
only**. Per sample, a uniform random `num_to_mask ∈ [1, num_available-1]` assays are set to `CLOZE` in
data, metadata and availability (`_vendored.py:133-150`). A sample with `num_available ≤ 1` is skipped
entirely — which is why a panel needs biosamples with ≥2 available assays or the imputation loss is
identically 0.

The **control channel is never masked**: it is concatenated onto `x_meta`/`x_data` *after*
`masker.apply_mask` runs (`batch.py:66` vs `batch.py:121-123`).

**`y_meta` never carries CLOZE** — the masker touches `x_meta` only. Decoder embedding rows for CLOZE
and control are therefore unreachable during training.

**Training uses no prompt builder.** The dataset's raw `T_` `y_meta` passes straight through. The honest
imputation prompt (`V_`/`B_` natural metadata for absent targets) is an *eval*-side construction.

### 3.2 Per-assay independent DSF sampling

`_sample_xy_dsf` (`dataset.py:34-56`) draws, **independently per assay**, an input downsampling factor
`x_dsf` and a target factor `y_dsf` from the h5's `dsf_list` (default `[1,2,4,8]`). Column `fi` of
`x_data` is then loaded from `counts_dsf{x_dsf[fi]}` with `meta_dsf{x_dsf[fi]}`, and column `fi` of
`y_data` from `counts_dsf{y_dsf[fi]}` with `meta_dsf{y_dsf[fi]}` (`dataset.py:300-318`). The
metadata's depth row moves with the counts, so the model is always told the truth about what it is
looking at and what it must produce.

### 3.3 Loss

Counts only. Element-wise NB NLL, then a **sum of two separately normalised means**
(`train.py:38-51`):

```python
elem = -NegativeBinomial(total_count=n, probs=1-p).log_prob(y_data)   # [B, L, A]
loss = elem[observed_map].mean() + elem[masked_map].mean()            # obs + imp, unweighted sum
```

Because the two terms are normalised independently, cloze positions are effectively up-weighted ~3–6×
per element, and the ratio drifts with how many assays the masker happens to pick. This is a property
of the recipe you are inheriting, not an accident to "fix" casually — it changes the obs/imp gradient
balance.

### 3.4 Optimizer and schedule — FROZEN

These are constants in the code, not flags (`train.py:78-98`, `train.py:54-63`):

| item | value |
|---|---|
| optimizer | `torch.optim.Adam(params, lr, weight_decay=wd)` — **plain Adam, coupled L2, NOT AdamW** |
| lr | `5e-4` |
| weight_decay | **`0.0`** (kit default; see §4.5) |
| grad clip | global norm `1.0`, every step |
| schedule | linear warmup over `0.1 × total_steps`, then cosine to `0.1 × lr` |
| batch size | 8 |
| epochs | 25 |
| determinism | `cudnn.deterministic=True`, `cudnn.benchmark=False`, `torch.manual_seed(seed)`. **Not** `torch.use_deterministic_algorithms` — it crashes this encoder |
| `transformer_layer_drop` | must be `0.0`; the kit raises otherwise, because it consumes global RNG inside `forward` and destroys step-for-step determinism (`encoder.py:876-880`) |

### 3.5 `--full-coverage`

The default sampled path picks a random `T_` biosample per batch and runs `steps_per_epoch` batches.
`--full-coverage` (`train.py:101-140`) instead builds one dataset per `T_` biosample sharing a single
RAM buffer and pulls them **round-robin**, so every epoch deterministically visits all train windows of
all `T_` biosamples. `--steps-per-epoch` is dead under `--full-coverage`. On the recorded q19 panel this
is ~1,910 batches/epoch × 25 epochs ≈ 47,750 updates.

**Recorded runtime envelope** (8 assays, `d_model=72`, `context_bins=768`, 3.10 M params, one
`nvidia_h100_80gb_hbm3_1g.10gb` MIG slice): ~72 min train + ~13 min full-chr21 eval = **~85 min/arm**.
Host RSS 15–18 GiB is the binding constraint, not GPU memory. GPU peak was never directly measured; a
co-worker scaling to 35 assays or `context_bins=1536` has no measured envelope.

---

## 4. The load-bearing choices

For each: the choice, the evidence, and what breaks without it.

### 4.1 Per-assay conditioning, NOT pooled across assays

**Choice.** The decoder FiLM computes `(γ, β)` per assay column from `memb[B, A, 32]` with
`pool_meta=False` (`model.py:167-170`). It does **not** average over the assay axis.

**Evidence.** A controlled arm that pools (`pool_meta=True`, reproducing the v1 across-assay mean):
distributional output-steering **M2 collapses 0.50 → 0.022, a ~25× drop**, and reconstruction *also*
degrades (M1 ceiling gap 1.16 pooled vs 0.63 per-assay). A refinement worth knowing: *uniform-per-batch
sampling* alone does **not** reproduce the null (M2 = 0.53) — the causal factor is across-assay
**pooling** specifically. Recorded in the research vault as `h34` under `q16`, verdict *supported*.

**What breaks.** Output steering, and reconstruction with it. Pooling forces one conditioning vector
per batch where the task needs one per assay.

**⚠ Divergence you must not undo.** Production `candi_v2`'s shipped decoder **does** pool —
`sandbox/candi_v2/decoder.py:208` and `:490` both compute `meta_embed.mean(dim=1)` to feed
`PreDecoderFiLM` / `PerDeconvLayerFiLM`. This kit deliberately does not. `pool_meta` is reachable from
the Python API (`build_real_model(pool_meta=True)`) as a control arm; it is not a CLI flag.

### 4.2 Raw, UNNORMALIZED covariates

**Choice.** `log2_depth` (~22–28) and `read_length` (30–101) go straight into `nn.Linear(1, 32)` with no
scaling (`encoder.py:95-96`, `encoder.py:128`).

**Evidence, empirical.** Normalization was pre-registered as load-bearing and **refuted**: ranking three
arms by distributional M2 gave **none 0.515 > z-score 0.483 > log-scale 0.448** — the exact reverse of
the hypothesis, with no family showing a gain and reconstruction comparable (M1 gap 0.59 / 0.63 / 0.70).
Recorded as `h33` under `q15`, verdict *refuted*.

**Evidence, structural.** Sentinel detection is **exact equality against `-1` and `-2`**
(`encoder.py:126-127`). A normalized covariate that happens to take the value `-1` would be silently
reinterpreted as MISSING. Any normalization scheme you add must be range-guaranteed to avoid `-1`
and `-2`, or must move sentinel detection to a separate mask tensor.

**What breaks.** Normalizing costs steering (measurably), and a careless scheme can corrupt the
missing-data semantics.

*Open counter-consideration, recorded but not tested:* because `LayerNorm` pins `‖memb‖` to `√32`, the
entire observable depth range moves the embedding only ~10% of its radius while a `read_length` flip
moves it several times that. This was flagged as a possible steering-deficit mechanism independent of
normalization. It has **not** been tested.

### 4.3 Per-assay independent DSF sampling (`--dsf-sampling uniform`)

**Choice.** `x_dsf` and `y_dsf` are drawn **independently** per assay (`dataset.py:45-46`, the
`mode == "uniform"` branch), so the target depth is generally not the input depth, and the target is not
a copy of the input.

**Evidence.** The `x_eq_y` control — one DSF drawn per assay and used for *both* sides
(`dataset.py:47-49`) — makes the task copyable. It gets **higher raw imputation Spearman (0.6386 vs
0.533 on the main recipe)** but **breaks latent invariance**: the M3 within/between cos-distance ratio
is **0.3335, above the 0.3 bar**, versus 0.244–0.292 on the main recipe. Recorded as `h43` under `q19`.

**What breaks.** With `x_eq_y`, target depth is redundant with input signal magnitude, the depth
covariate receives ~zero gradient, and the encoder stops using metadata to normalise the measurement
condition. The higher Spearman is the *symptom*, not a win: it is what a copying model scores.

**`x_eq_y` is a control, not a recipe.** So is `--dsf-sampling off`. And a **single-DSF `dsf_list`
makes independent sampling inert and deletes the depth signal the whole recipe rests on** — the kit
warns loudly at bake time (`src/candi_kit/prep/panel.py:46`).

### 4.4 `depth_center` derived from the data

**Choice.** `log2_mu = (depth - depth_center) + eta`. The kit derives `depth_center` at build time as
the **median of finite `meta_dsf1[0]` over `T_` biosamples** (`dataset.py:60-74`) and prints it
(`train.py:224-226`). The historical q19 constant — still the hardcoded default in
`build_real_model` and the value `--compat-q19` pins — is **25.1**, which was the *mean* over **all**
views at DSF 1 (25.117), not a median over `T_` (`METADATA_AUDIT.md:114`). The kit's derived value is a
different statistic on a different subset: **do not assume the two coincide**; read the printed value.

**Known defect you inherit with 25.1** (audit S19, `METADATA_AUDIT.md:221-223`): because DSF only ever
*down*-samples, the depth actually prompted during training averages **23.227**, so a `depth_center` of
25.1 leaves the offset term with mean ≈ **−1.9** rather than 0 — contradicting the head's own rationale.
Deriving the centre from the h5 (the kit's default) does not fix this either, since it is also a DSF-1
statistic. Centring on the *prompt* mean is an untested one-line change.

**Evidence.** An **uncentred raw `2^d` offset FAILS** — depth-controllability ratio ~1.0, i.e. no
depth response at all — while the depth-centred size factor `μ = 2^(d - center) · exp(η)` gives
DCR 3.99–4.02 from epoch 0. Recorded as `h9` under `q4`.

**What breaks.** Without centring, `2^d` at d≈25 is ~3.4e7, so `η` must carry a ~25-log2 offset that the
head is not parameterised to produce; the run collapses to a depth-insensitive solution.

This also retires three inconsistent hardcoded defaults that exist in the research repo (24.0, 22.5,
25.1). **Derive it; do not copy 25.1 onto a different panel.** `--depth-center` overrides.

### 4.5 `weight_decay = 0.0`

**Choice.** The kit's default is `--weight-decay 0.0` (`train.py:278`), which differs from the research
driver's `1e-4`.

**Evidence.** With `weight_decay=1e-4` — **L2-coupled inside plain Adam**, applied over ~47,750 updates
to a pathway whose task gradient is ≈0 because the arithmetic offset already fits the mean — the
decoder's metadata embedder is *annihilated*: `assay_embedding` and `runtype_embedding` absmax reach
**~6e-41** (denormal), all 11 assay-embedding row norms are exactly 0, and `depth_proj.weight` is
1.6e-4. `film_proj` itself stays healthy (0.347), so the death is specifically on the embedder's **input
projections**. With adaLN-zero the entire decoder metadata pathway also receives *exactly zero* gradient
at init, so it starts dead and decay outruns anything it might later acquire.

Setting `weight_decay=0` **prevents the annihilation**: the table stays full-rank and injective
(effective rank 6.8/8). This is the weight-level claim, and it is supported.

**⚠ State the limit honestly.** `weight_decay=0` does **not** buy functional steering. On the offset-ON
`wd=0` arm the table sits at **random-init statistics** (element std 0.94 vs 0.97 for a fresh N(0,1)
table; cosine 0.988 against an independently-trained table) — it was never destroyed *and* never
trained. See §5 for the corresponding steering numbers.

**What breaks.** At `wd=1e-4` you get a decoder that is bit-exactly blind to its own target metadata
from the fusion's first `Linear` onward, which will read as an "honest null" in any steering probe.

### 4.6 adaLN-ZERO init on the decoder FiLM

**Choice.** `film_proj` weight *and* bias are zero-initialised **after construction**
(`model.py:149-151`), so the FiLM is a bit-exact identity at step 0.

**Why.** Reconstruction is stable from step 0 and steering grows from a clean identity, rather than the
model having to unlearn a random modulation.

**What it costs — know this.** Combined with the free arithmetic offset, adaLN-zero means the optimizer's
depth objective is *already satisfied* before the learned pathway is even alive. The encoder FiLM uses
the opposite init (Xavier weight, `N(0, 0.1)` bias, `encoder.py:287-288`); that asymmetry is
deliberate but it is also the mechanism behind the offset-ON steering null.

**Construction detail that is load-bearing:** `film_proj` must be **constructed then zeroed**, not
created as a zero tensor — see §6.

---

## 5. The offset head is a real Pareto — not a solved problem

`--offset {on,off}` selects between two arms. **Both are first-class; neither dominates.** Numbers are
from the recorded 4-arm re-score (full chr21 eval, 608 units, 1215 target-records, 12 held-out
targets). Read them against the noise floor stated in §0.

| | **offset ON** (`--offset on --weight-decay 0.0`) | **offset OFF** (`--offset off --weight-decay 0.0`) |
|---|---|---|
| macro CRPS | **1.3413** | 2.0561 |
| macro Spearman | **0.5653** | 0.4641 |
| pooled imputation Spearman | **0.6372** | 0.3800 |
| PIT-ECE | **0.0533** | 0.0782 |
| beats honest per-assay marginal | **7/8** | 1/8 |
| oracle-scaled CRPS (capability) | 1.3077 | 1.4026 |
| per-assay scale error | 0.0336 | 0.6535 |
| assay steering, sentinel-free real→real max\|Δη\| | **0.0023** | **9.7144** |
| run_type steering, target-clustered CI | [−0.00066, +0.000087] | [−0.2326, +9.4084] |
| told-depth total slope | 1.0000 | 1.0325 |

The `--offset off --weight-decay 1e-4` arm quoted elsewhere in this repo reads macro CRPS 1.9023
(+42% over the ON arm), assay steering 4.1772, run_type clustered CI [+0.1179, +2.1804], sign-test
p = 0.039 — the only arm whose steering direction is target-clustered supported.

**What this means, stated plainly:**

1. **offset ON is the best imputer** and the best-calibrated. Its **covariate steering is functionally
   null**: sentinel-free real→real assay ablation **0.0023**, which is **43× below** its own
   pre-registered ≥0.10 functional bar, and 1816×/4224× below the two offset-OFF arms on the *identical*
   probe. Its run_type response is bit-indistinguishable from zero.
2. **Its depth response is arithmetic, not learned.** `log2_mu = (depth − depth_center) + η` is a
   closed-form thinning identity — NB is closed under binomial thinning with `n` preserved, and DSF
   downsampling *is* exactly that generative model. A told-depth slope of **1.0000** is therefore the
   arithmetically correct answer produced by a hardwired subtraction. It is **not** evidence of learned
   conditioning. Under offset-ON, `η` has analytically nothing left to learn about depth on the DSF axis.
3. **offset OFF has genuine learned steering** at a large magnitude cost. Its raw-CRPS deficit is
   **calibration, not capability**: under an oracle per-assay scale the four arms compress from a 0.7148
   spread to **0.1133 (84% compression)**.
4. **A prior "steering 0.833" result for the offset-ON `wd=0` arm was retracted.** It was a
   MISSING-sentinel artifact: the probe permuted the whole `assay_id` row across all 8 slots, sliding
   the `-1` sentinel onto and off *unavailable* slots whose prompt columns are entirely `(-1,-1,-1,-1)`.
   Sentinel-free, the real→real value is 0.0023.
5. **`h45` — the hypothesis that a hybrid (offset warmup→anneal, α-attenuated offset, or a learned
   metadata-driven scale head) recovers both — is recorded REFUTED, 0/4 verifiables met.** Read that
   precisely: it was refuted **on its premises**, and **no hybrid arm was ever trained**. The node also
   carries a 2026-07-28 flag that the refutation basis is under review, because one leg of it rested on
   the retracted 0.833. So the hybrid is **neither demonstrated nor experimentally excluded** — do not
   re-run it as posed, and do not cite it as a negative result.

**⚠ The 4-arm ordering is NOT established.** Under oracle per-assay scale the four arms sit in a 0.113
band against a ~0.09 target-clustered noise floor. Only "the offset-ON `wd=0` arm is best on capability"
survives inference (paired bootstrap Δ = +0.093, 95% CI [+0.004, +0.217]); the other three are
statistically indistinguishable (all pairwise CIs cover 0). The published ordering is the modal
bootstrap ordering at only 45% of replicates. Full detail is in `TRADEOFF.md`.

### Bounds you inherit with the 8-assay panel

- **`run_type` is analytically unidentifiable** on the shipped panel:
  `H(run_type | assay_id, read_length) = 0.000 bits` (n=26). It is a deterministic function of the other
  two covariates, so weight decay has a free ride on its embedding and *any* function of it is
  reproducible at zero loss cost. **A run_type steering demo is impossible on this panel** — it needs a
  re-selected biosample panel. (The full EIC panel retains **0.551 bits** after conditioning on assay,
  so this is a property of the 5-biosample slice, not of the field.)
- **DSF only ever DOWN-samples**, so the upward-depth regime is never trained. Training support for
  assay *a* is `[natural_min − 3, natural_max]`, and **7/12 eval targets sit above their per-assay
  training ceiling** (worst +1.43 log2). Depth steering evaluated there is extrapolation in the
  untrained direction.
- **Depth, read_length and run_type are mutually collinear** at n=38 (assay-centred
  `corr(d, log2 rl) = 0.763`, `corr(d, rt) = 0.590`, `corr(log2 rl, rt) = 0.697`). Attribution among the
  three exposure covariates is **not identified** on this dataset. Any claim of the form "covariate X
  carries Y% of the exposure signal" is unsupportable here.
- **9/12 eval targets are OOD in (assay × read_length)**.

---

## 6. The FROZEN RNG construction order

`RealDualCondModel.__init__` and `DualCondDecoder.__init__` both draw from the **global torch RNG**.
Every historical q19 checkpoint was produced under one exact draw sequence, and
`python -m candi_kit.compat` proves the kit reproduces it bit-exactly (3,103,194 params, `state_dict`
sha1 `fd0e9493ac92a15f` under `torch.manual_seed(0)`).

**Reordering module construction in `src/candi_kit/model.py`, `encoder.py` or `decoder.py` silently
changes every initial weight.** Nothing crashes; the model just trains from different initial
conditions and no longer matches any checkpoint. The authoritative banner is
`src/candi_kit/model.py:1-16` — read it before touching that file.

The frozen sequence:

| # | draw | file:line |
|---|---|---|
| 1 | `V2Encoder(enc)` **in full** — `metadata_embedding` → `signal_tower` → `mask_injector` → `dna_tower` → `fusion` → `transformer_blocks` → `output_norm` | `encoder.py:762-872` |
| 2 | replacement encoder `RealMetaEmbedder` (**the one built at step 1 is discarded**) | `model.py:217-218` |
| 3 | decoder `DecoderTrunk` (`input_proj`, then each `DeconvTower` in schedule order) | `model.py:141-146` |
| 4 | `_LegacyPlaceholderMetaEmbedder` (**constructed, then discarded**) | `model.py:147` |
| 5 | `film_proj` — **constructed, then zeroed** | `model.py:149-151` |
| 6 | `head_eta` | `model.py:153` |
| 7 | `head_n` | `model.py:154` |
| 8 | the real decoder `RealMetaEmbedder`, overwriting slot 4 | `model.py:156-157` |

**Two placeholder embedders are deliberately constructed and thrown away** (steps 1→2 and 4→8). They
consume RNG. Deleting them is *not* a harmless cleanup.

Specific traps, each of which silently invalidates `--compat-q19`:

- moving `V2Encoder(enc)` after the `self.encoder.metadata_embedding = ...` replacement;
- skipping the `_LegacyPlaceholderMetaEmbedder` (decoder state hash `b3fe340929d15aeb` →
  `39682aaea5552718`);
- scaling that stub's embedding table with `num_assays` — it is **FIXED at 8 × embed_dim**
  (`model.py:83`) and must not track panel size;
- creating `film_proj` as a zero tensor instead of constructing-then-zeroing it;
- reordering trunk → placeholder → `film_proj` → `head_eta` → `head_n`;
- **an x-transformers version other than the pinned `2.11.23`** — its per-module init order is part of
  the RNG stream *and* it fixes the `encoder.transformer_blocks.*` `state_dict` key names. The package
  has no `__version__` attribute, so the gate checks `importlib.metadata.version`
  (`compat.py:103-106`).

The masker also draws from the global stream at every step (`_vendored.py:140-143`), so any
construction-time RNG change shifts the entire masking sequence too.

**If you intend to change the architecture, that is fine — just stop claiming compat.** Run without
`--compat-q19` and record that your run is a new lineage.

---

## 7. The full knob table

**Single source of truth rule:** scale is declared **once** in the bake panel, written into the h5
attrs, and read back by the trainer. `num_assays`, `context_length`, `resolution`, `dsf_list`, assay
order, and train/eval chromosomes are **never** train-time CLI flags. You cannot silently train an
8-assay model on a 12-assay file.

### 7.1 Bake-time (`configs/panel.json`)

| knob | type | default | status | note |
|---|---|---|---|---|
| `assays` | list[str] | required | load-bearing (via derived order) | order-**insensitive**; the resolved column order is derived from the handler's `aliases.json` and asserted bijective against this list |
| `biosamples` | list[str] | required | **LOAD-BEARING** | full names with `T_`/`V_`/`B_` prefix; ≥2 `T_` biosamples, each with ≥2 available assays |
| `dsf_list` | list[int] | `[1,2,4,8]` | **LOAD-BEARING** | a single-DSF ladder makes independent `x_dsf ≠ y_dsf` inert and deletes the depth signal |
| `resolution` | int | 25 | semi-knob | it is `dna_pool_size**2`, so only perfect squares are reachable without editing the encoder's block-count formula |
| `context_bins` | int | 768 | FREE | must be divisible by `pool_size**n_cnn_layers = 8` |
| `train_chroms` | list[str] | `["chr19"]` | FREE | |
| `eval_chroms` | list[str] | `["chr21"]` | FREE | must be disjoint from `train_chroms` |

Those **seven keys are the whole schema**. `load_panel` rejects anything else
(`prep/panel.py:51-57`), so `type2_ccre`, `type2_non` and `seed` are **not** panel fields — putting them
in `panel.json` raises `ValueError: unknown panel keys [...]`. They are **bake CLI flags**
(`prep/bake.py:456-458`): `--type2-ccre N` / `--type2-non N` (both default 0; `>0` also requires
`--ccres`) and `--seed` (default 42, type-2 locus sampling only).

### 7.2 Train-time: DERIVED from the h5, not configurable

`num_assays`, `context_length`, `resolution`, `dsf_list`, assay order, `train_chroms`, `eval_chroms` ←
`h5.attrs`.

`depth_center` ← median of finite `meta_dsf1[0]` over `T_` biosamples, **printed at build**;
`--depth-center` overrides.

### 7.3 Train-time: LOAD-BEARING flags

| flag | default | why |
|---|---|---|
| `--offset {on,off}` | `on` | the two shipped arms; §5 |
| `--dsf-sampling` | `uniform` | per-assay independent `x_dsf ≠ y_dsf`; §4.3. `x_eq_y` / `off` / `upsample_only` are controls |
| `--p-full-assay` | `1.0` | whole-assay cloze is the **only** source of imputation supervision |
| `--weight-decay` | `0.0` | §4.5 |
| `--lr` | `5e-4` | |
| `--batch-size` | `8` | |
| `--epochs` | `25` | |
| `--full-coverage` | set it | deterministic all-windows × all-`T_`-biosamples round-robin |

Frozen constants (not flags): grad-clip 1.0; cosine warmup_frac 0.1 / min_ratio 0.1; plain Adam with
coupled L2; loss = mean-NB-NLL over unmasked **+** mean over cloze, unweighted sum.

### 7.4 Train-time: FREE

| flag | default | note |
|---|---|---|
| `--d-model` | `0` = auto | **see §7.6** |
| `--nhead` | `4` | attention inner dim is `nhead × 64` regardless of `d_model` |
| `--embed-dim` | `32` | metadata embedding width, both towers |
| `--n-transformer-layers` | `2` | |
| `--feat-per-assay` | `16` | decoder `C`; `signal_dim = A × C` |
| `--dropout` | `0.1` | |
| `--seed` | `0` | but see the seed sensitivity in §0 |
| `--regime` | `type1` | or `type2_loci` |
| `--steps-per-epoch` | `200` | **dead under `--full-coverage`** |
| `--eval-batch-size` | `4` | |
| `--eval-max-batches` | `0` (=all) | |
| `--eval-budget` | `200000` | max eval points for CRPS/correlation; set very high for no subsampling |
| `--m3-regions` | `8` | |
| `--fg-frac` | `0.02` | foreground fraction for the scored positions |
| `--n-boot` | `1000` | |
| `--include-deprecated` | off | emits legacy metric keys with their verdict strings attached |
| `--compat-q19` | off | pins `embed_dim=32, dropout=0.1, n_transformer_layers=2, feat_per_assay=16, depth_center=25.1, d_model=0, nhead=4` and asserts an 8-assay / 768-bin h5 |
| `--tag` | auto | output basename |

### 7.5 INERT and NOT EXPOSED

- `--mask-fraction` (`0.2`) is **inert** under `p_full_assay=1.0` — `_mask_full_assay` never reads it.
  The trainer warns if you set a non-default (`train.py:93-95`).
- `num_metadata_rows` (4) and `num_runtypes` (2) are **not** config fields. They are fixed by the
  ingestion contract and fenced by raises in `MetadataEmbedding.forward`.
- Python-API only, no CLI: `pool_meta`, `log2_mu_clamp`, `mu_eps`, `depth_row`, `trunk_norm`,
  `n_cnn_layers`, `expansion_factor`, `pool_size`, `conv_kernel_size`, `use_layernorm`
  (`model.py:121-127`, `model.py:236-244`).
- Hardcoded into `RealDualCondModel.__init__` (`model.py:207-213`), editable only by changing that
  file: `film_mode="per_conv"`, `signal_transform="arcsinh"`, `missing_data_mode="mask_token"`. The
  remaining `EncoderConfig` fields (`conv_norm`, `dna_pool_size`, `dna_pool_order`, `fusion_mode`,
  `fusion_norm`, `fusion_deep`, `attn_qk_norm`, `output_rms_norm`, `meta_embed_layernorm`,
  `transformer_type`) take their `config.py:19-65` defaults.

### 7.6 ⚠ The `d_model` / `num_assays` coupling

With `--d-model 0` (the default), the transformer width is **derived from the panel**:

```
d_model = signal_tower.out_channels = num_tracks × expansion_factor**n_cnn_layers
        = (num_assays + 1) × 2**3
```

So 8 assays → `d_model = 72`; 3 assays → 32; 16 assays → 136; 35 assays → 288. The model prints the
auto-derived value at every build (`model.py:224-226`).

`--d-model` is an **independent override**: set it and the transformer width follows the flag, not the
panel (`encoder.py:798`). **Set it explicitly whenever `num_assays != 8`**, or your transformer capacity
silently tracks how many assays you happened to include.

**Changing the assay count changes the transformer width and therefore invalidates any checkpoint** —
plus the grouped conv schedule, the mask-token table, the decoder `signal_dim = A × C`, and the
`assay_id` embedding table size. A checkpoint is valid only against an h5 whose recorded assay **order**
matches; compare against the order stored in the checkpoint, not merely against the panel, and never
delete `aliases.json` once a bake exists.

---

## 8. What was tried and did NOT work

Do not redo these.

| idea | status | what actually happened |
|---|---|---|
| **`h45` — a hybrid recovers both magnitude and steering** (offset warmup→anneal-off; α-attenuated offset β ∈ {0.25,0.5,0.75}; learned metadata-driven scale head) | **recorded REFUTED, 0/4 verifiables met — but on its premises only: NO HYBRID ARM WAS EVER TRAINED**, and the node carries a 2026-07-28 flag that its refutation basis is itself under review | The premises that fell: offset-OFF's "recovered depth steering" (η-slope 0.88) was a measurement artifact — `eta_slope` scores the *offset-free residual*, which is ~0 by construction under a correct offset. On the correct **total** told-depth slope the h45/h47-era measurement put offset-OFF at 0.775 vs offset-ON's 1.000; **the later full-coverage h48 re-score supersedes that number** — §5's table (0.8869 `offoff` / 1.0325 `wd0_off` vs 1.0000) is the current one, and `H48_REPORT.md` wins wherever the two disagree. Its run_type steering does not survive target-clustered CIs, and run_type is analytically unidentifiable on this panel. Note the offset is a **boolean all the way down** (`--offset` → `build_real_model(use_offset=…)` → `DualCondDecoder.use_offset` → the `torch.where`), so a β-schedule arm is a 4-file change plus threading a per-step schedule into `_train_step`, not a flag. **Do not report the hybrid as tested.** |
| **`h33` — covariate normalization is load-bearing** | **REFUTED** | none 0.515 > z-score 0.483 > log 0.448; no family showed a gain. Keep raw. §4.2 |
| **Across-assay covariate pooling** (`h34` / `q16`) | **refuted as a design; supported as a diagnosis** | Pooling costs ~25× steering (M2 0.50 → 0.022) *and* hurts reconstruction (M1 gap 1.16 vs 0.63). It is retained only as the `pool_meta=True` control arm. §4.1 |
| **Uncentred raw `2^d` offset** (`h9`) | **FAILS** | depth-controllability ratio ~1.0 vs 3.99–4.02 for the depth-centred size factor. §4.4 |
| **`S23` condition-recoverability probe** | **WITHDRAWN — do not cite in either direction** | Its ordering is inverted against every other measurement: the arm carrying ~5,900× more feature energy scored *below* chance, while the near-zero-signal arm scored 2.5× higher. Leave-one-target-out nearest centroid on within-target-centred features penalises a target-*adaptive* response whose direction flips sign between targets. Reliable only as a bit-exactly-dead detector. |
| **"`wd=0` revives functional assay steering"** | **RETRACTED** | The 0.833 that supported it was a MISSING-sentinel artifact. Sentinel-free real→real is 0.0023. The *weight-level* claim (`wd=0` prevents annihilation) stands; the *function-level* claim does not. §4.5, §5 |

### Ranked but NOT yet run — the two obvious next steps

Both target the same finding: **most of the apparent ON/OFF magnitude difference is per-assay
scale-calibration error, not capability** (84% compression under an oracle per-assay scale; 0.52–0.65
macro CRPS of *fixable* per-assay scale sits on the offset-OFF arms). That compression result is the
solid one; the 4-arm reordering is not.

**`h50` — an explicit per-assay output factor.** Add metadata-**independent** per-assay `η` scale + bias
and a per-assay `log n` offset (~24 params, no decay), indexed by **slot**, not by `y_meta`.

*Why it should help:* the head is `Linear(16,16)+GELU+Linear(16,1)` **weight-shared across assays** with
a single scalar output bias, so the only per-assay knob today is a rank-~2 FiLM — while 8 assays with
very different dynamic range and dispersion need ~7 degrees of freedom. The oracle correction that
produces the 84% compression is *literally one scale per assay*. This also closes a fork-vs-production
gap: production's decoder already carries a per-assay bias.

*How to judge it:* the gain must show up in the **scale-error term** (`CRPS − CRPS_oracle_scaled`
shrinks), not in shape; hold `macro Spearman ≥ 0.5653` and `ECE ≤ 0.0533`; and beat capability 1.3077
**by more than the ~0.09 noise floor**, not by a 4-decimal margin.

**`h49` — `read_length` as a fixed-coefficient physical exposure term.** The NB head is a size-factor
GLM, but its exposure term is **incomplete**: it counts *reads*, not read *footprint*. A length-R read
at 25 bp resolution covers ~`R/25 + 1` bins, so a second exposure factor `log2(R/25 + 1)` spans ~1.2
log2 over the 30–101 bp range.

*Why it should help:* `read_length` carries a ~0.48–0.61 coefficient on log2 mean count and **is** the
"excess depth slope" — once `log2(rl)` enters, the depth coefficient returns to 0.975. Today that job is
left to a 1,056-param FiLM that cannot extrapolate, and **9/12 eval targets are OOD in
(assay × read_length)**; an arithmetic offset extrapolates by construction, and it frees the FiLM to
spend its rank on assay identity instead.

*Constraint:* fix the coefficient at the physical value **1**. Because depth, read_length and run_type
are mutually collinear at n=38, attribution among them is not identified on this data — do not fit and
then claim credit. An optional second arm may learn a no-decay coefficient and check it converges
to ~1.

---

## 9. Where to look next

| question | file |
|---|---|
| how to build the h5 from ENCODE-style data | `DATA.md` |
| the offset ON/OFF tradeoff with full statistics | `TRADEOFF.md` |
| adding the Gaussian p-value and Bernoulli peak heads | `EXTENSION_HOOKS.md` |
| quickstart and the three commands | `README.md` |
| the bit-exactness gate | `python -m candi_kit.compat`, `src/candi_kit/compat.py` |
| the frozen-construction banner | `src/candi_kit/model.py:1-16` |

# q19 Architecture Handoff — Metadata Conditioning, Embeddings & FiLM

**Purpose.** You are picking up CANDI's **q19** dual-conditioning work. This document is a complete,
verified map of **how experimental metadata enters the model** — the embedder, the FiLM layers, where
they sit, and what the trained weights actually did. It exists so you can (a) navigate the code fast and
(b) later **critically review this design** and propose fixes under hypothesis **h45**.

**Provenance.** Every claim below was re-derived from source and, where marked ⚙, verified at runtime or
read from a trained checkpoint. A 10-agent verification pass checked 123 claims (26 required correction —
those corrections are already folded in). Where something is *unverified but load-bearing*, it says so.

| substrate | value |
|---|---|
| repo | `/project/6014832/mforooz/EpiDenoise` @ `df070d62` |
| data | `sandbox/data/sandbox.h5` (1,686,890,341 B, Apr 22 10:41) |
| env | `candi_venv`, python 3.10.13, torch 2.6.0 (`source candi_venv/bin/activate; export PYTHONNOUSERSITE=1`) |
| canonical results | `results/{main_s0,offoff_s0}_perassay.json` + `.ckpt`; `results/*_full.json` (bit-identical on shared keys) |

---

## 0. TL;DR — the seven things that matter

1. **Two disjoint metadata pathways, two separate embedders** (untied). Encoder embeds `x_meta`
   ("what the input measurement was"); decoder embeds `y_meta` ("what measurement to produce").
2. **Input side is deep, output side is shallow.** Encoder: FiLM after **every** one of 3 conv layers
   (3 independent projections). Decoder: **exactly one** FiLM, after the *entire* deconv trunk.
3. **All FiLM is metadata-only and position-constant.** `(scale, shift)` never depend on the
   activations, and are broadcast over all 768 positions. Zero per-position specificity.
4. **Depth reaches the output twice**: through the embedder→FiLM→`η` (learned, nonlinear) **and** as a
   hardwired additive term in log2-μ space (`(d − 25.1)`, exactly linear). The second starves the first.
5. **⚙ THE BIG ONE — in the trained offset-ON checkpoint the decoder's metadata embedder is *dead*.**
   `assay_embedding` and `runtype_embedding` absmax ≈ **6e-41** (denormal), `depth_proj.weight` ≈ 1.6e-4.
   Training didn't just *fail to use* metadata; **weight decay deleted the pathway's input side.** With
   offset OFF all four fields are healthy. This reframes h41/h42's "starved gradient" story (§9).
6. **The decoder is never told the input depth** — only `y_meta`. Rescaling observed→told depth must be
   recovered from `z` alone.
7. **The deconv trunk is metadata-blind, but not by omission of a constructor arg** — `film_layers` is a
   **forward()** argument that q19 simply never passes. Enabling per-deconv FiLM is a *forward-call*
   change, not surgery (§7.3). This is the most obvious untried h45 lever.

---

## 1. Navigation map

### The q19 harness (`sandbox/diagnostics/dual_conditioning_real/`)
| file | what to read it for |
|---|---|
| `model_real.py` | `RealMetaEmbedder` (:70-80), `RealDualCondDecoder` (:82-140), `build_real_model` (:181+). **h45 edits go here.** |
| `run_real.py` | training driver. `nb_count_loss` (:43-52), `cosine_warmup` (:55-64), `_t_biosamples` (:71-76), `_train_step` (:79-88), `train` (:90+, `full_coverage` branch :96-135), `evaluate` (:161-175), `train_and_eval` (:209, ckpt save :219-221), CLI (:236-256) |
| `metrics_real.py` | M1/M2/M3. `eval_M1` (:176+), per-assay block (:210-249), `_decode` (:154-155), `_flip_covariate` (:315), `_depth_sweep` (:368) |
| `tests/`, `smoke_cpu.py`, `smoke_gpu.py`, `jobs/gate.sh` | the pre-GPU gate — keep green |
| `results/*.ckpt` | ⚙ trained state_dicts — **the weight audit in §9 is reproducible from these with no GPU** |

### Upstream modules it imports
| file | symbol |
|---|---|
| `sandbox/candi_v2/encoder.py` | `MetadataEmbedding` (:57-170), `FiLMLayer` (:245-267), `PerAssayFiLM` (:270-291), `TransformerFeatureFiLM` (:294-305), `MaskTokenInjector` (:312+), `SignalConvTower` (:357-435), `V2Encoder.encode` (:857+), `_infer_availability_from_meta` (:574-581) |
| `sandbox/candi_v2/decoder.py` | `DecoderTrunk` (:259+), and the **production** target: `DepthOffsetNegativeBinomialLayer` (:111-183, `depth_center` default **24.0**), `PreDecoderFiLM` (:185-211), `PerDeconvLayerFiLM` (:213-238), `V2Decoder` FiLM construction (:406-424) |
| `sandbox/candi_v2/config.py` | `EncoderConfig` (:31+): `n_cnn_layers=3` (:40), `expansion_factor=2` (:41), `metadata_embed_dim=32` (:36), `meta_embed_layernorm=True` (:37). `DecoderConfig` `meta_embed_layernorm=False` (:100) |
| `sandbox/diagnostics/dual_conditioning/model.py` | **golden reference, frozen.** `DualCondDecoder` (:102+), `nb_nll` (:238-247), `forward_full` (:231) |
| `sandbox/batch.py` | `MISSING = -1` (:11), `CLOZE = -2` (:12), `mask_value=CLOZE` (:24), masker (:66), control concat (:121-123) |
| `sandbox/data.py` | `SandboxH5Dataset`, `control_assay_id = num_assays` (:195) |
| `sandbox/__init__.py` | `SANDBOX_ASSAYS` (:5-14) — the assay↔index order every per-assay table depends on |

> **Scope note.** `RealDualCondDecoder` is a **diagnostic fork**. Production is
> `candi_v2/decoder.py::V2Decoder` + `DepthOffsetNegativeBinomialLayer`, which *does* build
> `PreDecoderFiLM` and `PerDeconvLayerFiLM`. **q19's decoder is strictly weaker than production.**
> Whether an h45 fix must land in production to count is a PI decision (§11 Q10).

---

## 2. The metadata schema

Raw metadata is a **4×F** matrix per sample — 4 covariates, one column per assay.

| row | covariate | type | observed range in `sandbox.h5` |
|---|---|---|---|
| **0** | `depth_log2` | continuous | 22.20 – 28.13 (mean **25.117**); control 22.67 – 27.12 |
| **1** | `assay_id` | categorical | 0–7 real assays, **8 = ChIP control** |
| **2** | `read_length` | continuous | {30, 36, 76, 100, 101} |
| **3** | `run_type` | categorical | {0 = single, 1 = paired} |

**Sentinels** (`sandbox/batch.py:11-12` — the sole definition): `MISSING = -1` (assay doesn't exist for
this biosample), `CLOZE = -2` (exists, masked, you must impute it). They are kept **distinct** everywhere.

**Row-index constants** — the most likely place to introduce an off-by-one:
- `metrics_real.py:41` → `DEPTH_ROW, READLEN_ROW, RUN_TYPE_ROW = 0, 2, 3`
- `model_real.py` → `depth_row=0` (q19, real data)
- golden testbed `dual_conditioning/model.py:163` → depth was **row 2** (synthetic). The base-class
  docstring still says row 2 and is **stale** for q19.

**`assay_id` index map** (both towers use `Embedding(num_assays + 3)` = `Embedding(11)`):

| index | 0–7 | 8 | 9 | 10 |
|---|---|---|---|---|
| meaning | real assays | **control** | MISSING | CLOZE |

⚙ Verified: `control_meta[:,1,0] == 8.0` for all 9 biosamples that have a control. The `+3` is exactly
control + MISSING + CLOZE. **No collision, not a bug.** ⚠ But the comment at `encoder.py:90`
(`# assay_id: [0..num_assays-1] + MISSING + CLOZE`) omits the control slot and is **wrong** — don't
"fix" the `+3` to `+2`. The convention is pinned by `tests/test_model.py:27`.

---

## 3. The metadata embedding module

`MetadataEmbedding` — `sandbox/candi_v2/encoder.py:57-170`. Maps **`[B, 4, F] → [B, F, E]`** (E = 32).
`RealMetaEmbedder` (`model_real.py:70-80`) subclasses it and **overrides nothing** (docstring only) — it
is just a named q19 handle. ⚙ Confirmed: `RealMetaEmbedder.__dict__` has no non-dunder entries.

### Design: per-field lift → late fusion

Each covariate is independently projected to E, then mixed:

| field | branch | line |
|---|---|---|
| `depth_log2` | `nn.Linear(1, E)` | :82 |
| `read_length` | `nn.Linear(1, E)` | :83 |
| `assay_id` | `nn.Embedding(num_assays+3, E)` | :91 |
| `run_type` | `nn.Embedding(num_runtypes+2, E)`, `num_runtypes=2` | :93, :73 |

then (:96-103, concat order at :169 = depth, assay, readlen, runtype):
```
concat(4E) → Linear(4E, E) → GELU → Linear(E, E) → LayerNorm(E)
```
`use_layernorm` defaults True (:74). ⚠ **q19 passes the *encoder* config's value (True) to *both*
towers** (`model_real.py:167`, `:171` via `enc.meta_embed_layernorm`, `config.py:37`). Production's
`DecoderConfig` defaults it **False** (`config.py:100`) — a silent divergence worth knowing.

### Sentinel handling — per field, independent

**Continuous fields** (:113-119) — project everything, then *overwrite* at sentinel positions:
```python
missing_mask = values == MISSING
cloze_mask   = values == CLOZE
emb = proj(values.unsqueeze(-1).float())      # computed for ALL entries...
if missing_mask.any(): emb[missing_mask] = missing_emb   # ...then discarded here
if cloze_mask.any():   emb[cloze_mask]   = cloze_emb
```
Four learned vectors, init `randn(E) * 0.02` (:84-87): `depth_missing_emb`, `depth_cloze_emb`,
`readlen_missing_emb`, `readlen_cloze_emb`.

**Categorical fields** — sentinels remapped to reserved rows *before* lookup:
`assay_id`: MISSING→`num_assays+1`, CLOZE→`num_assays+2` (:144-153);
`run_type`: MISSING→`num_runtypes`, CLOZE→`num_runtypes+1` (:156-166).
Remap order (MISSING then CLOZE, on the already-remapped tensor) is safe — `num_assays+1/+2` can never
equal −2.

⚙ **Three verified sub-facts** worth knowing:
- `proj(-1)`/`proj(-2)` **is** computed then thrown away — wasted FLOPs, **no gradient leak**
  (with all-sentinel depth, `depth_proj.weight.grad.norm() == 0.0` while `depth_missing_emb.grad.norm() == 0.194`).
- The in-place `emb[mask] = …` does **not** break autograd (`index_put_` has a defined backward; `nn.Linear`
  saves its *input*). A mixed batch yields healthy grads on both paths simultaneously.
- No NaN/inf risk from sentinels; only a genuinely non-finite input at a non-sentinel position propagates.

**Why this design matters:** because masking is **per field**, you can hand the decoder a target's
`assay_id` + `depth` + `run_type` while its *signal* is absent. That is exactly what makes the
imputation prompt work.

---

## 4. Input-side conditioning (encoder)

q19 sets **`film_mode="per_conv"`** (`model_real.py:161`).

### Placement: after **every** conv block, nowhere else
`SignalConvTower` builds one `FiLMLayer` per block (:411-415); `pre_film` and `post_film` are `None`:
```python
self.per_conv_film_layers = nn.ModuleList([
    FiLMLayer(meta_embed_dim, 2 * (ch // self.num_tracks)) for ch in out_channels_list])
```
and applies them **after** each conv (:417-435):
```python
for i, block in enumerate(self.blocks):
    x = block(x)
    x = self.per_conv_film_layers[i](x, meta_embed)
```

### Each layer gets its own projection
Same shared `meta_embed` tensor in; **independently learned, differently-sized** projections out
(`num_tracks = A+1 = 9`, `expansion_factor = 2`, `n_cnn_layers = 3`):

| conv layer | out channels | per-assay C | FiLM projection |
|---|---|---|---|
| 0 | 18 | 2 | `Linear(32 → 4)` |
| 1 | 36 | 4 | `Linear(32 → 8)` |
| 2 | 72 | 8 | `Linear(32 → 16)` |

Each block also max-pools, so layer 2 conditions at **8× the receptive field** of layer 0.

### Where metadata does NOT go on the input side
- **Transformer: no FiLM.** `transformer_film_layers` is built only under
  `film_mode == "per_conv_and_transformer"` (:824) — ⚙ it is `None`. `pooled_meta = meta_embed.mean(dim=1)`
  is computed in `encode()` but consumed only by that dead branch.
- **DNA tower: none.** `DNAConvTower.forward(x_dna)` takes no metadata argument.

### The *second*, non-FiLM metadata path — availability
Metadata also drives a **discrete, structural** path:
1. `_infer_availability_from_meta(x_meta)` — ⚠ **reads ALL FOUR rows**, not one:
   `has_cloze = (meta == CLOZE).any(dim=1)` (`encoder.py:574-581`). A track is flagged if **any**
   covariate holds the sentinel. (Row 0 appears only as a shape template.)
2. `_prepare_signal` **zeroes** masked signal channels and **raises** if meta-vs-signal availability
   disagree (this is the assertion an ad-hoc synthetic batch will trip).
3. `MaskTokenInjector` replaces masked assays' features with learned per-assay tokens **after** the conv
   tower, **before** DNA fusion.

⚙ `mask_stem` is `None` in q19; `mask_injector` is active; 2 x-transformers blocks with RoPE.

### ⚠ Per-assay independence is *not* as clean as it looks
Convs are grouped (`groups=num_tracks`, :396) and FiLM is per-assay — **but `ConvBlock`'s default
`LayerNorm` normalizes over the full channel axis**, mixing all 9 tracks' statistics at every layer.
So independence holds for the *learned parameters* but is **broken statistically**: a measured ~37%
cross-assay leak. Switching `conv_norm="group"` removes the leak exactly (0.0000 delta on
non-modulated assays) **but also cancels much of the per-assay FiLM's own effect** (rel-change 0.117
layer → 0.073 group). It is a **tradeoff, not a free fix.**

---

## 5. Output-side conditioning (decoder)

### Placement: exactly ONCE, after the whole trunk, before the heads
`RealDualCondDecoder.forward` (`model_real.py:113-140`):
```python
feat = self.trunk(z)                                   # 3-layer deconv tower — NO metadata
feat = feat.view(B, Lq, self.A, self.C)                # [B, L, 8, 16]
memb = self.meta_embedding(y_meta.float())             # [B, 8, 32]
gamma, beta = self.film_proj(memb).chunk(2, dim=-1)    # [B, 8, 16] each
feat = feat * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)   # ← the ONLY output FiLM
eta   = self.head_eta(feat).squeeze(-1)                # [B, L, 8]
raw_n = self.head_n(feat).squeeze(-1)
```
- `film_proj = nn.Linear(32, 2*C) = Linear(32, 32)`, **weight AND bias zero-init** (`model.py:142-144`)
  = **adaLN-zero** → FiLM is a bit-exact identity at step 0.
- `head_eta`/`head_n` = `Sequential(Linear(C,C), GELU, Linear(C,1))`, **weight-shared across assays**.
- `pool_meta = False` (per-assay, not the h34 across-assay-pooled baseline).

### The arithmetic offset — bypasses the embedder entirely
```python
depth = y_meta[:, self.depth_row, :]                   # depth_row = 0
valid = (depth != MISSING) & (depth != CLOZE)
log2_mu = torch.where(valid, (depth - self.depth_center) + eta, eta)   # offset ON
# offset OFF:  log2_mu = eta
mu = 2**clamp(log2_mu, -15, 30);  n = softplus(raw_n)+eps;  p = n/(n+mu)
```
`depth_center = 25.1` (q19 empirical mean) vs **24.0** in production `DepthOffsetNegativeBinomialLayer`.

### ⚠ The trunk is metadata-blind — but not for the reason you'd guess
`DecoderTrunk.__init__` has **no `film_layers` parameter at all** (`decoder.py:265-277`).
`film_layers` and `pooled_meta` are **`forward()` arguments**, and q19 calls `self.trunk(z)` passing
**neither** (`model_real.py:113`). Production `V2Decoder` *does* construct and pass `PerDeconvLayerFiLM`.
**→ Enabling per-deconv-layer output FiLM in q19 is a forward-call change, not architecture surgery.**

---

## 6. FiLM semantics: metadata-only, position-constant

**Every FiLM in this codebase computes `(scale, shift)` from the metadata embedding alone.** The
activations `x` are only ever the *operand*:

| module | line | parameter source |
|---|---|---|
| `FiLMLayer` (input, active) | :263 | `self.proj(metadata_embed)` |
| decoder `film_proj` (active) | `model_real.py:116` | `self.film_proj(memb)` |
| `PerAssayFiLM` (inactive) | :289 | `self.proj(meta_embed)` |
| `TransformerFeatureFiLM` (inactive) | :304 | `self.proj(pooled_meta)` |

Parameters are then **broadcast over the sequence** (`[B,C,1]` input side; `.unsqueeze(1)` output side).
**Consequence: every one of the 768 positions receives the identical scale and shift.** There is *zero*
per-position specificity. This is textbook FiLM (Perez et al.).

⚙ **Empirically confirmed**, with an important caveat:
- **Cross-assay leak = exactly 0.0** for all other assays when one assay's `y_meta` is perturbed — this
  is **unconditional** (holds at init *and* with a randomized `film_proj`), because `pool_meta=False`.
- The **"constant across positions / exactly Δdepth"** property holds **only while the decoder FiLM is
  at adaLN-zero init**. Once `film_proj` trains away from zero, `η` itself becomes depth-dependent and the
  shift is no longer exactly `Δdepth`. Do not over-claim this for trained models.

---

## 7. The prompt: how `y_meta` is actually built

⚠ **Two corrections to the obvious assumption:**

1. **Training uses no prompt builder at all.** q19's training path passes the dataset's raw `T_` `y_meta`
   straight through (`run_real.py` → `forward_full`). **`_build_mixed_meta` is never called by any q19
   code path** (only by the generic sandbox trainer and one unit test).
2. **Eval** uses `_build_vb_natural_missing_meta` (imported in `metrics_real.py`) — masked/absent target
   slots receive the `V_`/`B_` **natural** metadata (the honest imputation prompt).

Other verified prompt facts:
- **`y_meta` is born control-free** (`[B,4,8]` vs `x_meta` `[B,4,9]`). Nothing is "dropped" — the control
  column is *appended* to `x_meta` at `batch.py:121-123` and never added to `y_meta`.
- **The masker touches `x_meta` only** (`batch.py:66` vs `:122`) → **`y_meta` never carries CLOZE.**
  Decoder embedding rows 10 (CLOZE) and 8 (control) are therefore unreachable; row 9 (MISSING) is
  reachable but excluded from the loss. ⚙ Confirmed in the trained ckpt: assay_emb rows 8/9/10 have norm 0.
- M2's flip test perturbs the prompt at **eval only**; training always sees the honest prompt.
- `obs`/`imp` in the **loss** = unmasked/cloze. `den`/`imp` in **M1** = available/absent. **Same words,
  different sets.**
- M2 is **decoder-only**: `_decode` (`metrics_real.py:154-155`) re-runs `model.decoder(u.z, y_meta)` with
  `z` cached under the *true* `x_meta`. So every "steering is absent" statement concerns ~7k decoder
  params; **the encoder FiLM is excluded by construction and is healthy in both arms.**

---

## 8. Dataflow and shape/parameter tables

```
h5 → SandboxH5Dataset → prepare_masked_batch ─┬─→ x_data[B,768,9]  x_dna[B,4,19200]  x_meta[B,4,9]
                                              └─→ y_data[B,768,8]  y_meta[B,4,8]

ENCODER  ── "what the input measurement was"
  x_meta ─┬─ _infer_availability_from_meta (ALL 4 rows, .any) ─→ zero masked chans ─→ MaskTokenInjector
          └─ metadata_embedding #1 → meta_embed[B,9,32]
                 └→ ①FiLM after conv0  ②FiLM after conv1  ③FiLM after conv2
  x_dna → DNAConvTower (no metadata) ─┐
                    signal tower ─────┴→ fusion → transformer ×2 (RoPE, NO FiLM) → z[B,96,72]

DECODER  ── "what measurement to produce"
  z → trunk (deconv ×3, NO metadata) → feat[B,768,128] → view[B,768,8,16]
  y_meta ─ meta_embedding #2 → memb[B,8,32] → film_proj → ④FiLM (γ,β [B,8,16], broadcast over L)
                 → head_eta / head_n → eta, raw_n
  y_meta row 0 ──────────────── ⑤ arithmetic offset (bypasses the embedder) ──→ log2_mu → mu,n,p
```
**Metadata injection points: ①②③ encoder FiLM · ④ decoder FiLM · ⑤ hardwired offset.**
Absent by choice: transformer FiLM, deconv-trunk FiLM, DNA-tower metadata.

### ⚙ Parameter budget (runtime-verified, `build_real_model()` defaults)

| module | params | note |
|---|---|---|
| **TOTAL** | **3,103,194** | |
| encoder | 264,728 | |
|  ├ `metadata_embedding` | 5,984 | |
|  ├ `signal_tower` | 2,940 | of which **FiLM projections 924** |
|  ├ `dna_tower` | 13,740 | |
|  └ `mask_injector` | 72 | |
| decoder | 2,838,466 | |
|  ├ `trunk` (deconv) | 2,830,848 | **91.2% of the whole model** |
|  ├ `meta_embedding` | 5,984 | |
|  ├ **`film_proj`** | **1,056** | **0.034% — the entire learned output-steering path** |
|  └ `head_eta` / `head_n` | 289 / 289 | |

⚙ `d_model = 72` (= `num_tracks × 2³` — **changing `num_assays` silently changes transformer width**);
`transformer_film_layers = None`; `mask_stem = None`; 2 transformer blocks; trunk contains **no** FiLM
submodules; `A=8, C=16, depth_center=25.1, pool_meta=False, depth_row=0, clamps=(-15, 30)`.
Trunk channel schedule `[1024, 512, 256, 128]`, `input_proj = Linear(72, 1024)`; L 768→96→768.

---

## 9. ⚙ THE HEADLINE FINDING — the offset-ON decoder embedder is annihilated

Read directly from the trained checkpoints (no GPU needed; reproduce with `torch.load(...)`):

| `decoder.meta_embedding.*` (absmax) | **offset ON** (`main_s0_perassay.ckpt`) | **offset OFF** (`offoff_s0_perassay.ckpt`) |
|---|---|---|
| `assay_embedding.weight` | **6.30e-41** (denormal) | 3.07e-01 |
| `runtype_embedding.weight` | **6.23e-41** (denormal) | 3.80e-01 |
| `depth_proj.weight` | **1.56e-04** | 4.68e-01 |
| `depth_proj.bias` | 4.68e-06 | 1.99e-01 |
| `read_length_proj.weight` | 5.23e-02 | 4.04e-01 |
| `decoder.film_proj.weight` | 3.47e-01 (healthy) | 3.94e-01 |
| `assay_embedding` row norms | **all 11 rows = 0.0** | rows 0-7 alive; 8/9/10 = 0 |

**Interpretation.** With the offset ON, `meta_embedding(y_meta)` is effectively **bit-invariant** to
depth, assay_id and run_type. `film_proj` is healthy — so the death is specifically on the **input
projections of the embedder**, not the FiLM. The likely mechanism: `weight_decay=1e-4` is **L2-coupled
inside plain Adam**, applied over ~47,625 updates to a pathway whose task-gradient is ≈0 because the
arithmetic offset already fits the mean. Supporting detail: at init, **the entire
`decoder.meta_embedding` receives exactly zero gradient** (all 16 tensors) — a direct consequence of
adaLN-zero — so the pathway starts dead and decay outruns any signal it might later acquire.

**Why this matters:** the recorded h41/h42 conclusion ("the offset starves the learned metadata
gradient") is right in *spirit* but wrong in *mechanism* — the input was **deleted**, not merely
under-trained. And `metrics_real.py:361-363` reports `natural_variance_insufficient = True` (an "honest
null") when the true cause is a **conditioning pathway that training removed**. ⚠ This is the single
most decision-relevant fact in this document, and it suggests an **h45 arm 0**: rerun offset-ON with
`weight_decay=0` and with *decoupled* AdamW, logging per-field `absmax` (not nnz — the weights are
denormal, so `!= 0` counts them as alive) each epoch.

---

## 10. Design rationale — the "why"

1. **Why the offset head exists.** It is the fix for the production DCR≈1 collapse (free-mean NB +
   copyable reconstruct-same-assay task). Do **not** propose "just delete the offset" — that is
   `offoff_s0`, already run, and it costs scale calibration (h46).
2. **Why depth is the only genuinely steerable covariate on real data.** DSF is the only transform with
   per-position counterfactual ground truth; `assay_id` is an identity prompt, not a magnitude knob;
   `read_length`/`run_type` have no dose. ⚠ **Corollary that is easy to miss:** the DSF counterfactual is
   *exactly the generative model of the offset head* (counts ∝ 1/dsf, depth −log2(dsf)), so under
   offset-ON `η` has analytically nothing left to learn about depth. That is a **data-design** property,
   not an optimization failure — and it bounds what h45 can possibly demonstrate.
3. **Why the init is asymmetric.** Decoder FiLM is adaLN-zero (`model.py:142-144`); encoder FiLM is
   xavier + `N(0, 0.1)` bias (`encoder.py:255-256`). Consequence: decoder conditioning is a bit-exact
   no-op at step 0 while the free arithmetic path is *already correct*, so the optimizer's depth
   objective is satisfied before the learned path is even alive.
4. **Why the loss up-weights cloze.** `elem[obs].mean() + elem[msk].mean()` — a sum of two separately
   normalized means up-weights cloze positions ~3–6× per element, and the ratio drifts with how many
   assays the masker picks. Directly relevant if h45 anneals the offset (the obs/imp gradient balance
   shifts as free scale is withdrawn).
5. **Why the golden testbed is frozen.** h45 edits go in `model_real.py`. Note
   `RealDualCondDecoder.forward` is a **hand-copy** of the base `forward` with one line changed — it will
   **not** inherit upstream fixes.

---

## 11. Open questions / candidate h45 issues

Ordered by expected value. **Q1–Q3 are the ones I would open h45 with.**

- **Q1 — Weight-decay annihilation (§9).** Is the offset-ON null caused by decay deleting the embedder?
  Disambiguating run: `weight_decay=0` and decoupled AdamW; log per-field `absmax` per epoch.
- **Q2 — Three live explanations for the exactly-zero run_type null**, none yet discriminated by recorded
  evidence: (a) float underflow after `2**log2_mu`; (b) run_type **collinear with assay_id** for 7/8
  assays in the T_-only training set; (c) weight annihilation.
- **Q3 — Is h45's run_type verifiable attainable *in principle*?** 7/8 assays have zero within-assay
  run_type variance in training; all 911 "paired" M2 records come from `B_` biosamples and all 304
  "single" from `V_` (run_type ⟂ imputation-source prefix); 5/12 targets are OOD in (assay × run_type).
  **Consider amending the pre-registration before the run, not after.**
- **Q4 — The ON-vs-OFF headline is a one-assay effect.** Excluding H3K27ac, offset-OFF *wins* macro
  Spearman (0.530 vs 0.488) and Pearson (0.522 vs 0.498) while offset-ON still wins CRPS 8/8. h45's bar
  mixes both families — state which metric arbitrates.
- **Q5 — Statistical power.** n=2 seeds on one arm only; seed Δ = 0.056 (Spearman), 0.120 (CRPS). h45's
  "macro CRPS ≤ 1.57 (within 5% of 1.495)" sits **inside the seed floor.** Needs ≥3 seeds/arm or a
  paired/shared-seed design.
- **Q6 — Pseudo-replication.** 1,215 M2 records = 12 targets × ~101 chr21 windows (only 405 distinct
  units; 58% from one biosample pair). Bootstrapping over positions overstates confidence — use
  target-level (n=12) or biosample-clustered inference.
- **Q7 — Covariate scaling (cheap alternative arm).** `depth` (raw ~22–28) and `read_length` (raw 30–101)
  enter `Linear(1,32)` **unnormalized**, while assay/run_type come from `N(0,1)` embeddings; LayerNorm
  then pins ‖memb‖ to √32, so the *entire* observable depth range moves the embedding ~10% of its radius
  while a read_length flip moves it several×. The golden testbed normalized here; **q19 does not.** This
  could explain the steering deficit *without touching the offset at all*.
- **Q8 — Give the decoder Δdepth.** The decoder is never told the *input* depth (`encode(..., return_meta=False)`
  discards `meta_embed`), so rescaling observed→told depth must be recovered from `z` alone, through a
  924-param encoder FiLM that leaks ~37% across assays. Candidate hybrid: pass `Δdepth = told − observed`.
- **Q9 — Per-deconv-layer output FiLM** (§5) — a forward-arg change; the largest architecture divergence
  from production.
- **Q10 — Knobs not reachable from the CLI** that h45 likely needs plumbed first: `log2_mu_clamp`,
  `mu_eps`, `pool_meta`, `depth_center`, `meta_embed_layernorm`, `conv_norm`, `film_mode`,
  `n_transformer_layers`.
- **Q11 — Infra gaps blocking h45's own plan:** no validation-loss trajectory, no per-epoch checkpoints,
  no per-epoch chr21 eval → *"sweep the anneal window"* currently has nothing to sweep against.
- **Q12 — Scope (PI call):** must the fix land in `candi_v2/decoder.py::DepthOffsetNegativeBinomialLayer`
  to count as production-transferable, or is the diagnostic fork sufficient?

---

## 12. Known-wrong-in-code (do **not** "fix" these into bugs)

- `encoder.py:90` — comment omits the control slot; the `Embedding(num_assays+3)` is **correct**.
- `dual_conditioning/model.py:108-109` — docstring says depth is row 2; **q19 uses row 0**.
- `batch.py:123` — `x_avail_in` is dead.
- `data.py:285` — marks an all-−1 control as available (currently dead code).
- `_build_mixed_meta` (`train.py:96-115`) — would blank the prompt if masking were ever re-enabled in eval.
- `_flip_read_length` — hardcoded `OBSERVED_READLENS`.
- `metrics_real.py:352` — NaN `mean_responsiveness` copies.
- `metrics_real.py:436` — `offset_independent = median(eta_slope) > 0.0`; with offset ON the median is
  **float noise** (−9.1e-17 / +1.05e-18 / +7.45e-19 across the three ON runs), so this flag is decided by
  an arbitrary sign. Not systematically inverted — **meaningless** for offset-ON arms.
- `beats_marginal` uses non-strict `<=`; Pearson is computed in **log1p space** while Spearman is raw.

---

## 13. How to run

```bash
cd /project/6014832/mforooz/EpiDenoise
source candi_venv/bin/activate && export PYTHONNOUSERSITE=1 && module load samtools

python -m pytest sandbox/diagnostics/dual_conditioning_real/tests -q     # block tests
python -m sandbox.diagnostics.dual_conditioning_real.smoke_cpu           # CPU integration smoke
sbatch sandbox/diagnostics/dual_conditioning_real/jobs/gate.sh           # full gate (CPU→GPU)
sbatch sandbox/diagnostics/dual_conditioning_real/jobs/perassay.sh       # 2-arm template for h45
```
Outputs land in `results/{tag}.json` + `results/{tag}.ckpt`. **SLURM `--gres` MUST be
`gpu:nvidia_h100_80gb_hbm3_1g.10gb:1`** (hard project rule), `--account=def-maxwl`. A full-coverage arm
is ~85 min. **Crux leash: PI approval is required before launching any new run.**

**Training config of the recorded runs:** 5 `T_` biosamples × all chr19 windows ≈ 1,905 batches/epoch ×
25 epochs ≈ 47,625 updates (`steps_per_epoch=200` is **dead** under `--full-coverage`); bs 8; Adam
lr 5e-4, cosine + 10% warmup; **`weight_decay=1e-4`, L2-coupled inside plain Adam** (see §9); grad-clip
global-norm 1.0. Eval = 608 units / 3,732,480 positions / 12 held-out targets / 1,215 M2 records.

⚠ **Reproducibility hazard:** the discarded-then-replaced modules (`RealMetaEmbedder` on both towers, the
placeholder `DualCondMetaEmbedder` in the decoder) **consume the global torch RNG**, so *any* constructor
edit breaks bit-exact comparison with the recorded runs. The masker also uses the global RNG.

---

## 14. h45's pre-registered bar (reproduce verbatim)

Anchors: `main_s0` = scale ✓ / steering ✗ · `offoff_s0` = steering ✓ / scale ✗. A hybrid must hit
**all** of:
- **Steering retained:** median η-slope ≥ 0.7; run_type direction-frac ≥ 0.6 with bootstrap CI excluding
  0 on **both** single and paired targets.
- **Magnitude restored:** per-assay macro CRPS ≤ ~1.57 (within 5% of the 1.495 offset-ON anchor);
  beats per-assay marginal → **8/8** assays.
- **Shape not regressed:** macro Spearman ≥ 0.50.

Proposed arms: (a) offset warmup → anneal-off (β 1→0); (b) fixed attenuation β ∈ {0.25, 0.5, 0.75};
(c) learned metadata-driven scale head (init ≈ identity). **Add arm 0 from §9** (weight-decay /
decoupled-AdamW control) — it may explain the whole phenomenon more cheaply than any of (a)-(c).

⚠ Note the offset is a **boolean all the way down**: `--offset {on,off}` (`run_real.py:243`) →
`build_real_model(use_offset=…)` (:214) → `RealDualCondDecoder.use_offset` → `model_real.py:129-131`.
A β-schedule arm requires changing all four **plus** threading a per-step schedule into `_train_step` —
a 4-file change, not a flag.

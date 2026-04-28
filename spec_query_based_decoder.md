# Query-Based Decoder Spec

## Goal

Introduce query-based decoding into CANDI while preserving strict backward compatibility.

The current decoder is fixed-structure and always decodes all `F` assays.  
The new approach decodes only queried assays during training/evaluation, while keeping full-`F` decoding as the default behavior for inference and legacy workflows.

#implemented strict query decode path for non-`fixed` decoders is now in `model.py` (`CANDI.decode()`, `CNP_MoE_Decoder`, `CNP_DynConv_Decoder`).
#todo full parity/benchmark validation vs fixed baseline is still needed.

---

## Scope and Compatibility

- Add two new decoder variants:
  - `query_moe`
  - `query_dynconv`
- Keep existing decoder as:
  - `fixed` (default)
- New decoder selection is CLI/config controlled:
  - `--decoder-type {fixed,query_moe,query_dynconv}`
- Existing checkpoints must continue loading/running unchanged.
- Existing scripts must keep current behavior unless new flags are explicitly used.
- `control` track behavior remains unchanged:
  - encoder-side only
  - never queryable target output

#implemented decoder variants + CLI selection + config persistence are wired in `train.py` (`create_argument_parser`, `create_model_from_args`, config save block) and `model.py` (`CANDI.__init__`).
#notimplemented query decoders are not enabled for `CANDI_UNET`; `model.py` currently raises `NotImplementedError` for non-`fixed` UNET mode.

---

## Current Tensor Contracts (must remain stable externally)

### Trainer -> Model inputs

- `x_data_masked`: `[B, L, F+1]` (assays + control)
- `x_meta_masked`: `[B, 4, F+1]`
- `x_dna`: `[B, L*25, 4]`
- `y_meta`: `[B, 4, F]`
- `y_avail`: `[B, F]`
- `observed_map`: `[B, L, F]`
- `masked_map`: `[B, L, F]`
- `y_dsf` (optional): `[B, F]`

Metadata rows stay:
- row 0: `depth_log2`
- row 1: `assay_id`
- row 2: `read_length`
- row 3: `run_type`

### Model outputs

Externally preserved output signature:
- `p, n, mu, scale, df, peak`
- each shaped `[B, L, F]` (except `df=None` for non-StudentT)

This must hold for all decoder types.

#implemented model outputs remain `[B,L,F]` for all heads in `model.py` (`CANDI.decode()`), including scatter-back from sparse query decoding.

---

## Decoder Behavior by Mode

### Inference (default behavior)

- Decode all `F` assays by default for all decoder types.
- No required user change to existing inference entrypoints.

### Training/Evaluation (query mode for query decoders)

- For `query_moe` and `query_dynconv`, decode sparse per sample using query sets.
- Query set must be per-sample, not batch-global.

Rationale: assay availability differs across samples in the same batch.

#implemented inference defaults to full-`F` decode when query masks are omitted in `model.py` (`CANDI.decode()`).
#implemented train/eval pass query masks from `train.py` (`_process_batch`) to `model.py` (`CANDI.forward()`).

---

## Query Set Definition

For sample `b`, define assay-level supervision indicators:

- `obs_any[b, f] = observed_map[b, :, f].any()`
- `msk_any[b, f] = masked_map[b, :, f].any()`
- `sup_any[b, f] = obs_any[b, f] or msk_any[b, f]`

Primary query mask:
- `query_mask[b, f] = (y_avail[b, f] == 1) and sup_any[b, f]`

Required assertion (debug-safe, train/eval):
- `(y_avail == 1) == sup_any`

If violated, raise/assert with informative error.

### Pathological case

- If `query_mask[b]` has no `True` entries (`k_b = 0`):
  - skip sample
  - log warning
  - increment counter metric `skipped_k0_samples`

#implemented query mask + assertion + pathological skip logic are in `train.py` (`_process_batch`).

---

## Task-specific Querying

### Count heads

- Query set uses full `query_mask` above.
- Count supervision remains where valid GT exists (with explicit `y_avail` intersection).

### P-value / peak heads

- Use tighter mask:
  - `query_mask_signal[b, f] = query_mask[b, f] and (y_dsf[b, f] == 1)`
- This is used for sparse decode in pval/peak branches and corresponding losses.

Rationale: pval/peak GT exists only at `DSF=1`.

#implemented `query_mask_signal = query_mask & (y_dsf==1)` is built in `train.py` (`_process_batch`) and used in `model.py` (`CANDI.decode()`).

---

## Loss Masking Safety Rule

To guarantee GT validity under per-assay DSF sampling and possible x/y divergence:

- Intersect all training/eval supervision masks with `y_avail`.
- This includes count, pval, and peak paths.

Concretely:
- `observed_map_final = observed_map & y_avail_broadcast`
- `masked_map_final = masked_map & y_avail_broadcast`
- and DSF-gated variants for pval/peak.

This becomes required behavior regardless of decoder type.

### Sentinel-safety assertions in loss

Because sparse decode scatters only queried lanes, non-queried lanes may contain sentinel values (for example `-1`).
To prevent silent logical bugs, add explicit assertions inside the loss path:

- Any prediction position with sentinel value must be outside all active supervision masks.
- Equivalently: for every head, `(pred == -1)` and `(loss_mask == True)` must never co-occur.
- If such overlap exists, raise immediately with a descriptive error.

Recommended checks:
- Count head: assert sentinel positions are excluded by `observed_map_final` and `masked_map_final`.
- Pval/peak heads: assert sentinel positions are excluded by DSF-gated signal masks.
- Optional strict mode: assert all masked-in positions are finite and non-sentinel before calling NLL/BCE.

These assertions are mandatory guardrails for scatter-back correctness and mixed-availability safety.

#implemented `y_avail` intersection for supervision masks and sentinel assertions are in `train.py` (`_process_batch`).

---

## Query Representation

- Query vectors must come from `MetadataEncoder` output.
- No explicit DSF token in query representation.
- DSF information is implicitly represented via metadata (especially `depth_log2`) and explicit `y_dsf` gating logic where applicable.

### Metadata embedding split (encoder-side vs decoder-side)

For query-based decoding, we need an explicit per-assay query embedding interface.

Design requirement:
- encoder-side metadata path may remain fixed-structure/global as needed for compatibility
- decoder-side query path must expose per-assay vectors so each queried assay is decoded independently

Planned interface:
- keep existing metadata encoder behavior for the encoder branch (`x_metadata` path)
- add a decoder-query metadata embedding path that returns per-assay embeddings for `y_metadata`
- query decoder consumes per-assay embeddings indexed by `(pair_b, pair_f)` in sparse mode

Suggested implementation options:

1. Split-class approach (clearest):
- `EncoderMetadataEncoder` for encoder branch
- `QueryMetadataEncoder` for decoder branch (guaranteed per-assay output)

2. Single class with mode flag:
- one class with explicit mode:
  - `mode="encoder"` (existing behavior)
  - `mode="query"` (returns per-assay embeddings for decoder queries)

Minimum technical contract for query mode:
- input: `y_metadata` of shape `[B, 4, F]`
- output: per-assay embeddings `q_embed` of shape `[B, F, E]`
- sparse gather in query decoder:
  - `q_sparse = q_embed[pair_b, pair_f]` -> `[Nq, E]`

Special token handling in query mode:
- keep identical semantics for `-1` and `-2` handling used elsewhere
- ensure categorical and continuous token logic matches current conventions
- query mode must preserve assay identity strongly (assay_id embedding remains mandatory)

Compatibility constraints:
- default `fixed` decoder path should keep current metadata behavior unless query decoder is selected
- old checkpoints remain loadable without requiring metadata encoder conversion
- new checkpoints must persist metadata-encoder mode/config so loader restores exact query behavior

#implemented decoder-side query embeddings are produced as per-assay `[B,F,E]` in `model.py` (`QueryMetadataEncoder`, `CANDI.decode()`).
#implemented compatibility path keeps `fixed` behavior unchanged in `model.py` (`CANDI.decode()` fixed branch).

---

## Decoder Variants

## 1) `fixed` (existing)

- No architectural change in behavior.
- Decodes all `F` assays.
- Default mode for backward compatibility.

#implemented `fixed` remains default via `train.py` (`--decoder-type` default) and `model.py` (`CANDI.__init__` branch).

## 2) `query_moe`

- Query-based sparse decoder.
- Stage-wise MoE at each decoder deconvolution stage.
- Experts differ by kernel size only.
- Gating is query-only (no latent-context gating).
- Gating type: soft weighting across all experts (no top-k hard routing).

### MoE expert count

- Controlled by CLI:
  - `--moe-experts`

### Kernel size generation

Base motif range is odd kernels from 3 to 11.

Rules:
- If experts <= 5, use increasing odd sequence from 3.
  - e.g., 5 -> `[3,5,7,9,11]`
- If experts > 5, duplicate lower kernels while keeping upper tail to 11.
  - e.g., 6 -> `[3,3,5,7,9,11]`
  - e.g., 7 -> `[3,3,5,5,7,9,11]`
  - e.g., 8 -> `[3,3,5,5,7,7,9,11]`

Implementation note:
- Build deterministic helper for kernel list generation so behavior is stable/reproducible.

#implemented in `model.py` (`CNP_MoE_Decoder`, `_build_kernel_sizes`, stage-wise query gating in `forward`).

## 3) `query_dynconv`

- Query-based sparse decoder with same FiLM form as current feature modulation:
  - per-assay multiplicative + additive modulation on activations.
- No full dynamic kernel generation.
- Distinct from `fixed` due to sparse query-based decode flow and decoder topology/control flow, not because FiLM math changes.

#implemented strict query path in `model.py` (`CNP_DynConv_Decoder` consumed via `CANDI.decode()` sparse pair gather/scatter).

---

## Integration with CANDI Class

`query_moe` and `query_dynconv` are alternatives to `CANDI_Decoder` at CANDI level.

Must preserve support for existing higher-level mode:
- `--separate-decoders`
- `--shared-decoders`

Meaning:
- If separate decoders are enabled, instantiate query decoder per task branch.
- If shared decoder is enabled, instantiate one query decoder shared by branches.

This policy is controlled by existing `CANDI` architecture logic, not by decoder class internals.

#implemented separate/shared decoder support for query decoders is wired in `model.py` (`CANDI.__init__`, `CANDI.decode()`).
#notimplemented query decoder integration with `CANDI_UNET` is intentionally blocked for now.

---

## Sparse Decode API

Extend CANDI forward path to optionally accept query masks/indices:

- `CANDI.forward(..., query_mask=None, query_mask_signal=None, ...)`

Behavior:
- If query args are omitted: decode full `F` (default inference + backward compatibility).
- If provided and decoder type is query-based: sparse decode by sample.

Internal scatter-back:
- Sparse outputs are scattered to full `[B, L, F]` tensors.
- Non-queried lanes can carry sentinel values (e.g., `-1`) because losses/metrics are strictly masked off those lanes.

### Scatter-back details (index correctness)

Use explicit `(sample_idx, assay_idx)` pairs as the canonical sparse index map.

For one head (count/pval/peak), construct:
- `pair_b`: shape `[Nq]`, sample indices in `[0, B-1]`
- `pair_f`: shape `[Nq]`, assay indices in `[0, F-1]`
- where `Nq = sum_b k_b` and `k_b = query_mask[b].sum()`

Sparse decoder input/output convention:
- gather query embeddings as `[Nq, E]` from `y_metadata_embed[pair_b, pair_f]`
- run decoder to get sparse outputs `[Nq, L]`

Scatter-back into dense outputs:
- initialize dense tensor `out_full = fill_value * ones([B, L, F])`
- write with advanced indexing:
  - `out_full[pair_b, :, pair_f] = out_sparse`

Why this is safe:
- every sparse row carries both its sample id and assay id
- no dependence on local per-sample ordering after flattening
- exact inverse mapping of gather/scatter by reusing the same `(pair_b, pair_f)`

Required runtime checks:
- `pair_b.numel() == pair_f.numel() == Nq`
- `pair_b.min()>=0`, `pair_b.max()<B`, `pair_f.min()>=0`, `pair_f.max()<F`
- optional uniqueness check for stability:
  - `unique((pair_b, pair_f)).size == Nq`

### Per-sample query forward in mixed-availability batches

Because each sample has its own query set, do not assume batch-global assay indices.

execution strategies:

1. Direct flattened sparse execution (default recommendation):
- Build `pair_b/pair_f` from per-sample `query_mask`.
- Gather sparse query vectors `[Nq, E]` and any aligned per-query tensors.
- Repeat/select latent features `z` from `[B, L', C]` to `[Nq, L', C]` via `z[pair_b]`.
- Run query decoder in this flattened query-batch.
- Scatter back to `[B, L, F]`.

Complexity and overhead notes:
- compute-heavy path is deconvolution over `L`; sparse mode reduces work from `B*F` to `sum_b k_b`
- gather/scatter is mostly memory traffic and usually smaller than decoder compute
- worst-case fallback is naturally full decode when all `k_b == F`

Pathological handling:
- if a sample has `k_b=0`, skip that sample and increment `skipped_k0_samples`
- if all samples in batch have `k_b=0`, skip batch

#implemented optional `query_mask`/`query_mask_signal` API in `model.py` (`CANDI.forward`, `CANDI.decode`).
#implemented gather/scatter utilities in `model.py` (`_mask_to_pairs`, `_scatter_sparse_tracks`).
#implemented flattened per-sample query execution in `model.py` (`CANDI.decode()` query branch using `z[pair_b]` and `q[pair_b,pair_f]`).

---

## Trainer / Eval Changes

1. Build query masks from:
- `y_avail`
- assay-level any-supervision derived from `observed_map` and `masked_map`

2. Assert consistency:
- `(y_avail == 1) == sup_any`

3. Build task masks:
- count: `query_mask`
- pval/peak: `query_mask_signal = query_mask & (y_dsf==1)` (when `y_dsf` available)

4. Pass masks to model forward for query decoders.

5. Ensure final loss masks always intersect with `y_avail` for GT safety.

6. Add logging metric:
- `skipped_k0_samples`
- export to progress logs and W&B.

#implemented all listed trainer changes in `train.py` (`_process_batch`, `loss_dict` additions).

---

## CLI / Config Changes

### New CLI

- `--decoder-type {fixed,query_moe,query_dynconv}`  
  Default: `fixed`
- `--moe-experts <int>`  
  Used by `query_moe`.

### Existing CLI that remains

- `--separate-decoders` / `--shared-decoders`

### Removed as unnecessary for v1

- no `--query-k`
- no `--query-sampling`

Query set is determined directly from available supervision.

### Config persistence

New model configs must save decoder settings so future checkpoint loading is automatic:
- `decoder_type`
- `moe_experts` (if applicable)
- any derived MoE kernel policy fields as needed for reproducibility

#implemented CLI flags + persistence in `train.py` (`create_argument_parser`, `create_model_from_args`, model config save).
#todo persist explicit derived kernel-size list in config if exact replay across future rule changes is required.

---

## num_runtypes Correction

Current training model creation uses `num_runtypes=4`; this should be corrected to `2` for new model creation.

Compatibility requirement:
- Loader should continue reading saved checkpoint configs as-is for old checkpoints.
- Do not force-convert old metadata cardinalities.

#implemented new creation uses `num_runtypes=2` in `train.py` (`main` and `create_model_from_args` defaults).
#implemented loader reads saved `num_runtypes` with fallback in `train.py` (`CANDI_LOADER.load_CANDI`).

---

## Migration Plan (phased)

1. Implement sparse-query infrastructure and API hooks.
2. Implement `query_moe`.
3. Implement `query_dynconv`.
4. Wire CLI/config/loader compatibility.
5. Add logging counters and assertions.
6. Validate parity for `fixed` path (no behavior change).

#implemented phases 1-9 in code structure.
#todo full empirical rollout validation (training run, metric parity reports, throughput/memory comparison) still pending.

---

## Non-goals (for this spec)

- No change to control-track target behavior.
- No inference-time subset decode flag in v1 (full `F` default only).
- No DSF token added to query vectors.

#implemented DSF token not added; DSF remains supervision gating logic in `train.py` (`_process_batch`).

---

## Acceptance Criteria

1. `fixed` mode reproduces current behavior and interfaces.
2. Query modes decode sparse during train/eval and full `F` during default inference.
3. Per-sample query sets work with mixed assay availability in a batch.
4. No loss computed on invalid/missing GT lanes.
5. `k=0` samples are skipped and tracked.
6. `--decoder-type` and `--moe-experts` are persisted and restorable from saved config.
7. `num_runtypes=2` for new model creation without breaking old checkpoint loading.

#implemented criteria 1,2,3,4,5,6,7 in code paths (`train.py`, `model.py`), pending runtime verification at scale.
#todo run end-to-end training/eval smoke tests to confirm all acceptance criteria empirically.


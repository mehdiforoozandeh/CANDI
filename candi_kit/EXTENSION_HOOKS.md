# EXTENSION_HOOKS

How to extend `candi_kit`. Every hook names the exact file and line.

Paths written bare (`src/candi_kit/model.py:153`) are **inside this kit**. Paths prefixed
`RESEARCH-REPO/` are in the CANDI research repository (`EpiDenoise/`) and are **not shipped** — they
are reference implementations you must copy in yourself.

## 0. Two things that break on any extension

1. **No checkpoints ship with this kit.** You train from scratch. Nothing below is "loading a
   different head onto existing weights" — there are no existing weights.
2. **`--compat-q19` is a bit-exact construction gate, not a feature.** `src/candi_kit/model.py:1-16`
   declares a frozen module-construction order; `tests/test_compat_q19.py` asserts a fixed parameter
   count (3,103,194 for `use_offset=True`, seed 0) and a `state_dict` SHA1. Any new parameter changes
   both. Gate every new module behind a flag that defaults to *off*, and append it **after** the last
   existing construction, so the default path keeps the frozen RNG draw sequence and the gate keeps
   passing.

---

## 1. Adding the Gaussian (p-value) and Bernoulli (peak) heads

This kit is **counts-only Negative Binomial**. There is no p-value track and no peak track in the
model, the loss, or the metrics. Adding them is the single largest available extension, and most of
the plumbing is already in place.

### 1.1 What already exists (do not rebuild)

| Artifact | Where | State |
|---|---|---|
| `pval` in the h5 | `src/candi_kit/prep/bake.py:330` — `(n, L, F) float16`, gzip-1 | baked |
| `peaks` in the h5 | `src/candi_kit/prep/bake.py:331` — `(n, L, F) int64` | baked |
| Both written from the **DSF-1** realization only | `src/candi_kit/prep/bake.py:357-368` (`cmap1`) | by design |
| `pval` is arcsinh-transformed signal | `src/candi_kit/prep/reference_sample.py:128` (`load_bios_BW(..., arcsinh=True)`) | — |
| Unavailable-assay fill | `src/candi_kit/prep/bake.py:361-362`: `pval → 0` (`:361`), `peaks → -1` (`:362`) | sentinel, must be masked |
| Loaded into the batch | `src/candi_kit/dataset.py:294-295`, emitted at `:328-329` as `y_pval`, `y_peaks`, both cast to `float32`, shape `[B, L, F]`, `F = num_assays` (no control column) | live |
| Moved to device | `src/candi_kit/batch.py:51-52` | live |
| Kept in sync through the drop-invalid-samples path | `src/candi_kit/batch.py:104-105` | live |
| DSF-1 supervision masks, already computed | `src/candi_kit/batch.py:127-131` → returned as `signal_observed_map` / `signal_masked_map` at `:142-143`; also `query_mask_signal` at `:90` | **computed and never read** |
| Returned in the prep dict | `src/candi_kit/batch.py:145-146` | live |

### 1.2 Where they are discarded

`src/candi_kit/train.py:45-51` (`nb_count_loss`) reads only `out["p"]`, `out["n"]`, `prep["y_data"]`,
`prep["observed_map"]`, `prep["masked_map"]`. `y_pval`, `y_peaks`, `signal_observed_map`,
`signal_masked_map` and `query_mask_signal` enter the prep dict and are never touched again — by
`train.py`, `eval.py`, or `report.py` (verified: no reader outside `batch.py`). **That is the hook.**

### 1.3 Reference implementation in the research repo

| Symbol | Location | Notes |
|---|---|---|
| `V2Decoder` | `RESEARCH-REPO/sandbox/candi_v2/decoder.py:336-547` | supports `heads="all"`; head resolution at `:459-468`, construction at `:439-457`, dispatch at `:501-530` |
| `heads` config field | `RESEARCH-REPO/sandbox/candi_v2/config.py:87-88` — `Literal["count_peak","count_only","peak_only","all"]`, default `"count_peak"`; `gaussian_var_min: float = 0.1` at `:113` | `"all"` = count + peak + pval |
| `NegativeBinomialLayer` | `RESEARCH-REPO/model.py:3222-3258` | lineage only; this kit reimplements the NB head inline (`src/candi_kit/model.py:172-186`) because of the depth-offset arithmetic |
| `GaussianLayer` → `(mu, var)` for pval | `RESEARCH-REPO/model.py:3260-3291` | its `FeedForwardNN` dependency (`:3173-3198`) fires only when `FF=True`, which `V2Decoder` never sets |
| `PeakLayer` → sigmoid | `RESEARCH-REPO/model.py:3465-3483` | |

`V2Decoder` is *not* drop-in: it has its own FiLM and trunk topology and it does **not** implement the
depth-offset log-link this kit's `DualCondDecoder` is built around. Port the three head classes, not
the decoder.

### 1.4 Step list

1. **Add a `heads` argument** to `DualCondDecoder.__init__` (`src/candi_kit/model.py:121-127`),
   `RealDualCondModel.__init__` (`:202-205`) and `build_real_model` (`:236-239`), defaulting to
   `"count"` (current behaviour).
2. **Construct the new heads after `head_n`** (`src/candi_kit/model.py:154`), inside
   `if heads != "count":`. Mirror the existing shape: `nn.Sequential(nn.Linear(C, C), nn.GELU(),
   nn.Linear(C, 1))` per output — `head_mu`, `head_logvar` (Gaussian), `head_peak` (Bernoulli logit).
   `C = feat_per_assay` (default 16). Do not insert before `head_eta`/`head_n`; see §0.2.
3. **Emit them from `forward`** (`src/candi_kit/model.py:159-186`): apply to the same post-FiLM
   `feat` `[B, L, A, C]`, `.squeeze(-1)` to `[B, L, A]`, and add
   `mu_signal` / `var_signal` / `peak_logit` keys to the return dict at `:186`. Clamp the Gaussian
   variance at a floor (production uses `gaussian_var_min=0.1`). Leave the NB block at `:175-185`
   untouched — the offset arithmetic is count-specific.
4. **Add the loss terms** in `src/candi_kit/train.py:45-51`. Gaussian NLL on `prep["y_pval"]`,
   `binary_cross_entropy_with_logits` on `prep["y_peaks"]`. Mask with **`signal_observed_map`** and
   **`signal_masked_map`** (`src/candi_kit/batch.py:142-143`), *not* `observed_map`/`masked_map`:
   pval and peaks exist only for the DSF-1 realization (`prep/bake.py:357-368`), so supervising them
   on a downsampled `x_dsf != y_dsf` pair supervises the wrong target. That gate — `(y_dsf == 1) &
   (y_avail > 0)` at `batch.py:127-128` — is exactly why those two maps were kept.
5. **Handle the peak sentinel.** `peaks` is `-1` for unavailable assays (`prep/bake.py:362`). The
   `signal_*` maps already exclude `y_avail == 0`, but assert `y_peaks >= 0` on the masked subset
   before BCE rather than trusting it.
6. **Weight the terms.** The kit's loss is an unweighted `obs + imp` sum of NB NLL
   (`src/candi_kit/train.py:51`). Production uses `count 0.5 / pval 1.0 / peak 0.25`
   (`RESEARCH-REPO/train.py:3189-3193`, applied at `RESEARCH-REPO/candi_loss.py:457-462`).
   **These weights were never tuned in this fork** and were never run against the counts-only recipe
   or its metrics. Treat them as a starting point, not a setting.
7. **Extend evaluation.** `src/candi_kit/metrics.py` contains NB primitives only (`nb_crps`,
   `nb_quantile`, PIT/ECE against an NB CDF). There is no Gaussian CRPS, no peak precision/recall,
   no AUROC anywhere in the kit. `src/candi_kit/eval.py` M1/M2/M3 all score `out["p"], out["n"]`.
   Adding heads without adding metrics gives you two untested outputs.
8. **Re-run the gate.** `pytest candi_kit/tests` must stay green with `heads="count"`; the
   `heads="all"` path is by construction outside the compat anchor.

### 1.5 Consequence to be explicit about

Adding heads changes the parameter set. Any weights you have already trained with the counts-only
head **cannot** be loaded `strict=True` into the extended model, and the `--compat-q19` anchors
(parameter count and `state_dict` SHA1) do not apply to it. The multi-head configuration also has no
recorded result in this program — every number in `TRADEOFF.md` and `H48_SCORECARD` is counts-only.

---

## 2. Changing the assay panel or the scale

**Single source of truth:** scale is declared once in the bake panel, written into the h5 attrs, and
read back by the trainer. It is never a train-time flag.

| You edit | File | What recomputes automatically |
|---|---|---|
| `assays`, `biosamples` | your `panel.json` (schema: `src/candi_kit/prep/panel.py`) | resolved column order, `num_assays`, `control_assay_id`, the h5 attrs, `ds.assays`, every per-assay label in `eval.py`/`report.py` |
| `context_bins`, `resolution`, `dsf_list`, `train_chroms`, `eval_chroms` | same `panel.json` | h5 attrs → `src/candi_kit/dataset.py:135-145` → `build_real_model(num_assays=ds.num_assays, context_length=ds.context_bins)` |

Constraints enforced at load, not by you: `context_bins % 8 == 0`; `resolution` must be a perfect
square (it is `dna_pool_size ** 2`); required DNA length is `context_bins * resolution`;
`eval_chroms` disjoint from `train_chroms`. A single-entry `dsf_list` makes per-assay independent DSF
sampling inert and deletes the depth signal the whole recipe rests on — `src/candi_kit/dataset.py:163`
warns.

No assay-name list literal exists anywhere in the kit. The column order is *derived* from the
handler's alias order and asserted bijective against the requested panel at bake time. Do not
reintroduce a hard-coded order.

### The `d_model` coupling warning

`src/candi_kit/config.py:53` defaults `d_model = 0`, and `src/candi_kit/encoder.py:798` resolves
`d_model = signal_dim` when it is 0 — i.e. **the transformer width silently tracks the panel size**:

```
d_model_auto = (num_assays + 1) * expansion_factor ** n_cnn_layers    # 8 assays → 72; 35 assays → 288
```

`src/candi_kit/model.py:224-226` prints the resolved value on every build. **Set `--d-model`
explicitly whenever `num_assays != 8`**, or a panel change becomes an unintended capacity change and
your run is not comparable to anything. Note also that `x-transformers` uses `dim_head=64`, so the
attention inner dim is `nhead * 64` regardless of `d_model` — capacity does not actually scale with
the panel unless you also raise `--nhead`.

---

## 3. The two ranked next architecture changes

Both are proposals from the project's research tracker (`h50` and `h49` — see the decoder ring in
[`research/README.md`](research/README.md)). **Neither has ever been run: no result exists for either.**
The verifiables below are the pre-registered success criteria, not measurements.

### 3.1 h50 — explicit per-assay output factor (ranked first)

**Motivation.** Under an oracle per-assay scale `c* = argmin_c CRPS(NB(n, mu·2^c), y)`, the four-arm
macro-CRPS spread compresses **0.7148 → 0.1133 (84%)**, reproduced independently at 84.0% and 83.5%
(`RESEARCH-REPO/sandbox/diagnostics/dual_conditioning_real/H48_REPORT.md:44-52`). Most of the
apparent between-arm capability difference is a per-assay *scale* error the model has no parameter to
absorb. Against the ~0.09 macro-CRPS noise floor (§ `TRADEOFF.md`), only "`wd0_on` is best" survives
inference at all — so the target here is the scale term itself, not the arm ordering.

**Why the fork lacks it.** The head is weight-shared across assays: `head_eta` and `head_n` are
`Linear(C, C) → GELU → Linear(C, 1)` with a single scalar output bias
(`src/candi_kit/model.py:153-154`). The only per-assay degree of freedom is the adaLN-zero FiLM
(`:149-151`, `:169-170`), which is rank-limited and metadata-driven. Production's dense head already
carries a per-assay bias; **this fork does not**.

**Change.** Add ~`3 × num_assays` (= 24 at 8 assays) metadata-**independent** parameters indexed by
slot `a ∈ [0, A)`, constructed after `head_n` (`src/candi_kit/model.py:154`):

- `eta_scale: nn.Parameter(torch.ones(A))`, `eta_bias: nn.Parameter(torch.zeros(A))`
- `log_n_offset: nn.Parameter(torch.zeros(A))`

applied in `forward` (`src/candi_kit/model.py:172-173`) immediately after the heads and **before** the
offset arithmetic at `:177-182`:

```
eta   = eta * eta_scale.view(1, 1, A) + eta_bias.view(1, 1, A)
raw_n = raw_n + log_n_offset.view(1, 1, A)
```

They must be indexed by **slot, not by `y_meta`** — the point is a structural identity that does not
compete with the metadata pathway. Put them in a **no-decay** parameter group (the kit's default
`--weight-decay 0.0` already satisfies this; if you raise weight decay, exempt them explicitly).

**Pre-registered verifiables.** The arbiter `H48_REPORT.md` settled on is the **capability** term, not
raw CRPS: beat macro `crps_oracle_scaled` **1.3077 by more than the ~0.09 target-clustered noise floor**
(a bare 4-dp improvement on raw macro CRPS 1.3413 does not count), with the gain attributable to the
scale term (`crps - crps_oracle_scaled` shrinks) rather than shape; macro Spearman ≥ 0.56;
ECE ≤ 0.0533; fitted `eta_bias[a]` tracks the oracle `c*[a]`; the metadata-ablation degradation does
not fall below `wd0_on`; total told-depth slope stays ≈ 1. Quote each against the noise floor: the
target-clustered bootstrap floor is ~0.09 macro CRPS and a seed change alone moves pooled imp CRPS by
0.1195.

### 3.2 h49 — read_length as a fixed-coefficient physical exposure term

**Motivation.** The NB head is a size-factor GLM — `log2_mu = (depth - depth_center) + eta`
(`src/candi_kit/model.py:177-179`) — but its exposure term counts *reads*, not read *footprint*. A
length-`R` read at 25 bp resolution covers ~`R/25 + 1` bins, a second exposure factor spanning ~1.2
log2 units over the observed 30–101 bp range (`src/candi_kit/eval.py:44`:
`OBSERVED_READLENS = (30, 36, 76, 100, 101)`). The audit finds `read_length` carries a **0.48–0.61**
coefficient on log2 mean count and **is** the excess depth slope: once `log2(read_length)` enters the
regression, the depth slope returns from **1.258–1.274** to **0.975–1.007**.

**Change.** In `DualCondDecoder.forward` (`src/candi_kit/model.py:175-182`), add a second offset with
a **fixed** coefficient of 1:

```
rl    = y_meta[:, 2, :]                                   # read_length row; see eval.py:43 READLEN_ROW
rl_ok = (rl != MISSING) & (rl != CLOZE)                    # same sentinel gate as depth, :176
expo  = torch.log2(rl / resolution + 1.0)                  # resolution = 25 in the shipped panel
log2_mu = log2_mu + torch.where(rl_ok.unsqueeze(1), expo, torch.zeros_like(expo).unsqueeze(1))
```

Thread `resolution` in from `ds.resolution` (`src/candi_kit/dataset.py:139`) rather than hard-coding
25. Apply it inside the `use_offset` branch only if you want it to share the offset's fate; applying
it unconditionally is a separate arm and should be run as one.

**Fix the coefficient at the physical value 1.** Do not fit it and do not claim an attribution among
`depth`, `read_length` and `assay_id` — they are collinear on this panel (audit bound B5). An
optional second arm may learn a no-decay coefficient and check that it converges near 1.

**Pre-registered verifiables.** Macro CRPS < 1.3413 with the gain concentrated on the
read_length-OOD targets; total told-depth slope stays `|slope − 1| ≤ 0.10`; macro Spearman ≥ 0.56;
ECE ≤ 0.0533; the `read_length` metadata-ablation ΔCRPS becomes large and correct-signed (it is ≈ 0
today because read_length rode a starved 1056-parameter FiLM path).

**Caveat that survives the change.** DSF only *down*-samples, so the upward-depth regime is untrained
and 7 of the 12 eval targets sit above their per-assay training depth ceiling. Adding a read-length
exposure term does not fix that; it only makes the extrapolation arithmetic rather than learned.

---

## 4. Adding a covariate (sequencing platform, lab)

Both `sequencing_platform` and `lab` are **already read off disk** — they are columns of the handler's
metadata frame (`src/candi_kit/prep/handler.py:150`) — and both are deliberately excluded from the
covariate tensor, which is fixed at 4 rows `[log2_depth, assay_id, read_length, run_type]`.

### 4.1 What to expect before you start

| Covariate | Conditional entropy given `(assay_id, read_length)`, T_ slice | Implication |
|---|---|---|
| `run_type` | **0.000 bits** (full EIC panel retains 0.551) | analytically unidentifiable on the shipped 8-assay panel; a run_type steering demo is *impossible* there and needs a re-selected biosample panel |
| `sequencing_platform` (9–10 levels) | **0.443 bits** | the most identifiable unused covariate |
| `lab` (6–16 levels) | **0.212 bits** | identifiable but thin |

Source: `RESEARCH-REPO/sandbox/diagnostics/dual_conditioning_real/METADATA_AUDIT.md:119`. Also noted
there: control depth is *already* in the h5 as `control_meta[:, 0, 0]`, varies within assay across
biosamples, and needs no re-bake — the cheapest additional signal available.

With 4 labs / 7 platforms across a 26-record training set, held-out records can land in n=1 cells.
Check cell occupancy before you conclude anything from a held-out score.

### 4.2 Edit sites for a 5th row

1. `src/candi_kit/prep/handler.py:1276-1281` — the per-assay `mdtensor.append([...])` 4-vector.
   Append the new field. `:1286` — the unavailable-assay branch (`mdtensor.append([missing_value ×4])`),
   which must append the same number of `missing_value` entries. `:1355` — the control column's
   `(4, 1)` tensor.
2. `src/candi_kit/prep/bake.py:379` — the `np.full((4, F), -1.0)` fallback and `:380` the
   `meta_dsf{d}` dataset, both `(4, F)` → `(5, F)`. Keep the fill at **-1, never 0**: the dataset's
   availability test is `float(xm[0]) != -1.0` (`src/candi_kit/dataset.py:306`, `:314`), so a
   zero-filled row marks every assay available at log2(depth)=0 with all-zero counts, and training
   proceeds happily on garbage.
3. `src/candi_kit/dataset.py:280-281` — `torch.full((B, 4, F), -1.0)` for `x_meta`/`y_meta`.
4. `src/candi_kit/encoder.py:142-146` — the hard 4-row check in `MetadataEmbedding.forward`; `:147-150`
   the row unpacking; `:95-106` the per-field projections/embedding tables (a categorical covariate
   needs `nn.Embedding(num_levels + 2, embed_dim)` with the MISSING/CLOZE slots, mirroring
   `runtype_embedding` at `:106` and the bound check at `:182-188`); `:109-110` the fusion
   `nn.Linear(4 * embed_dim, embed_dim)` → `5 *`.
5. `src/candi_kit/eval.py:43` (`RUN_TYPE_ROW, DEPTH_ROW, READLEN_ROW = 3, 0, 2`) and `:736`
   (`META_ROWS`) — add the row so the metadata-ablation instrument covers it. `_metadata_ablation`
   iterates `for r in (0, 1, 2, 3)` at `:830` and `:836`.
6. A re-bake is required (the h5 `meta_dsf{d}` shape changes), and every existing checkpoint is
   invalidated (the metadata embedder gains parameters).

---

## 5. Deprecated instruments behind `--include-deprecated`

These keys are **off by default**. They are measurements that were made, audited, and found not to
support the claim they were reporting. Each is emitted with its verdict string attached
(`src/candi_kit/eval.py:49-68`, `DEPRECATED_VERDICTS`) so a reader who finds one in a results JSON
cannot mistake it for evidence. **Do not cite any of them.** The flag exists for reproducing older
result files, nothing else.

| Key | Emitted by | Why deprecated |
|---|---|---|
| `read_length` (flip arm) | `eval.py:837-841` (M2) | 7/12 flips land outside per-assay training support → measures OOD extrapolation, not read-length steering |
| `null`, `null_clustered` | `eval.py:715-732` (`_depth_sweep`) | the shuffled-depth null is a mathematical **no-op**: `y_meta_imp` is one `[4, F]` tensor broadcast over the batch, so `base_d[perm] == base_d` bitwise. Recorded as exactly 0 in all 10 historical result files. A real null must permute *across* targets/assays |
| `frac_min_at_true` | `eval.py:725` | scores every told depth against the fixed DSF-1 target, so any μ-decreasing model satisfies it (0.7588 vs 0.7597 between arms — no discriminative power). Superseded by `_dsf_counterfactual` |
| `direction`, `overall`, `single`, `paired` | `eval.py:602-607`, `:721-724` | position-level bootstrap CIs, ~24× too narrow — positions within a target are not independent draws. Use the `*_clustered` keys |
| `median_eta_slope`, `offset_independent` | `eval.py:727-730` | decided by the sign of ~1e-17 float noise under offset-ON. The identity `total_slope = beta + eta_slope` means `eta_slope ≈ 0` under a correct offset is arithmetically right, not a failure. Demoted to a labelled attribution diagnostic |
| `frac_direction` | `eval.py:605-607` | strict `>` on `mean_delta`, so exact ties report 0.0% correct rather than "no signal" |

**Deleted outright** — not emitted in any mode, and no flag brings them back: `_recoverability_probe`
(+ `_output_features`, `_leave_group_out_nearest_centroid`; its ordering was inverted against every
other instrument — the arm with the most real assay steering scored 0.0907, *below* the 0.125 chance
level); the global `marginal_crps` / `crps_beats_marginal` pair (degenerate median rule: on ≥50%-zero
pools it is a point mass at 0 and equals `mean(y)`); `health_gate_den_ge_imp` (near-automatic, because
eval runs `dsf_sampling='off'` with `apply_mask=False`, so "denoise" is autoencoding); and
`build_canonical_meta` / `--use-canonical` (the assay-order label bug at its source). If you want any
of these, re-derive them — do not resurrect them.

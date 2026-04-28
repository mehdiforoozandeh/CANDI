# Supertrack prompting issue (CANDI): problem → hypotheses → solutions → keywords

## A) Problem statement

### What CANDI is doing today (experimental covariates)

- **Metadata tensor layout**: per locus and per assay, CANDI constructs a 4-row metadata tensor:
  - row0: depth (stored/used as `log2(depth)`)
  - row1: sequencing platform id (categorical int)
  - row2: read length
  - row3: run type id (0 single / 1 paired)
  - **Code**: `data.py:L1034-L1072` (`make_bios_tensor_Counts`)

- **Encoding & Conditioning Architecture**:
  - **Embedding (`MetadataEncoder`)**:
    - **Continuous fields** (`depth`, `read_length`): projected via `nn.Linear(1, dim)`. **Note**: Sentinels (`-1`) are treated as just another float value (e.g., -1.0 vs log2(depth)≈25.0).
    - **Categorical fields** (`platform`, `run_type`): projected via `nn.Embedding`. Sentinels (`-1`) are mapped to a special "missing" index.
    - **Fusion**: All 4 embeddings are concatenated and passed through an MLP (`Linear` -> `GELU` -> `Linear` -> `LayerNorm`) to produce a single vector per assay.
    - **Code**: `model.py:L1635-L1691` (`MetadataEncoder`)
  - **Conditioning (`FiLMLayer`)**:
    - The fused metadata vector is projected to `scale` and `shift` parameters for each feature channel.
    - **Injection**: `feature = feature * exp(scale) + shift`.
    - **Initialization**: FiLM layers are initialized to near-identity (weights ~0, bias 0).
    - **Code**: `model.py:L1693+` (`FiLMLayer`)

- **Potential architectural contributors to the issue**:
  1.  **Identity Initialization**: Since FiLM starts at identity, if the decoder can solve the task using only its fixed weights (which know "channel 5 is H3K4me3"), it faces no gradient pressure to use the prompt.
  2.  **Continuous Sentinel Handling**: Projecting missing/cloze flags (`-1`/`-2`) through the same linear layer as continuous values (like depth) forces the network to learn a non-linear "sentinel detection" function inside the fusion MLP, which might be harder than using an explicit mask/embedding.
  3.  **Fixed-Channel Decoding**: The architecture outputs `F` fixed channels. The prompt is *auxiliary* info. If the architecture forced the prompt to *select* the channel (Query-based), it couldn't be ignored. As is, it's easily ignored.

- **DSF-specific depth is real**: when loading DSF versions, the loader reads DSF-specific `signal_DSF{DSF}_res{res}/metadata.json` and uses that `depth`.
  - **Code**: `data.py:L831-L873` (`load_bios_Counts` reads `md1["depth"]`)
  - (We spot-checked EIC `metadata.json` files: DSF=1/2/4 depths differ appropriately.)

- **Two-sided prompting**:
  - `x_meta` conditions the **encoder** (input side). Training concatenates control into X, so X has `F+1` assays.
    - **Code**: `train.py:L607-L609` (concat control into `x_data_masked`, `x_meta_masked`, `x_avail_masked`)
    - **Code**: `model.py:L1277-L1303` (`encode` embeds `x_metadata` and passes into encoder)
  - `y_meta` conditions the **decoder** (output side) and is intended as the "generation prompt".
    - **Code**: `model.py:L1305-L1337` (`decode` embeds `y_metadata` and FiLM-conditions decoder)

- **Conditioning mechanism (FiLM)**: metadata is embedded and then injected via FiLM at multiple layers:
  - metadata embedding:
    - **Code**: `model.py:L1635-L1691` (`MetadataEncoder`)
  - decoder FiLM injection:
    - **Code**: `model.py:L1002-L1058` (`CANDI_Decoder.forward`)
  - (encoder FiLM injection happens inside `CANDI_DNA_Encoder.forward`; see `model.py` around the `CANDI_DNA_Encoder.forward` loop that calls `self.film_layers[i](...)`.)

- **Training-time Y prompt filling**: when building the Y-side batch, the dataset may "fill in" missing `y_meta` columns using median/mode or sampling.
  - **Code**: `data.py:L1550-L1641` (`get_batch(side='y', y_prompt=True)` calls `fill_in_prompt(...)`)

- **Masking semantics (important for reliance on covariates)**:
  - full assay masking masks **data + metadata** for selected assays
  - full loci / per-assay chunks mask **data only**, leaving metadata intact
  - **Code**: `_utils.py:L36-L56` (docstring summary), `_utils.py:L131-L168` (full assay masks `metadata`), `_utils.py:L255-L309` (`apply_mask`)

### The intended behavior (your "supertrack" goal)

- You want a "golden prompt" (a canonical `y_meta`) that, at inference time, yields:
  - high-quality "supertrack" imputations, and
  - denoised/canonicalized versions of inputs.
- You also want DSF self-supervision to teach the model what "depth" means: low-depth input + its `x_meta` should map to a higher-depth target consistent with the `y_meta` prompt.

### The observed failure mode

- After training, predictions are essentially **invariant** to the `y_meta` prompt: changing depth/run_type/etc. doesn't change outputs in a meaningful way (as measured via `viz_supertrack.py`).
- Practically: the model behaves as if the decoder-side prompt is ignored (FiLM collapses to "do nothing" or becomes redundant).

---

## B) Hypotheses (what could be going wrong)

### H1) Prompt/Target mismatch teaches model to ignore prompt
- **Core idea**: Training often fills missing `y_meta` with median/mode values even when the target `y` is missing or comes from a different run. The model learns "prompt values are not reliably tied to the target" and ignores them.
- **Related code**: `fill_in_prompt` in `data.py`.

### H2) Fixed-channel decoder bypasses conditioning
- **Core idea**: The decoder outputs to fixed assay channels (e.g., channel 5 is always H3K4me3). It can learn "channel 5 → H3K4me3" without looking at `y_meta`. FiLM layers initialize to identity, so the path of least resistance is to ignore `y_meta`.
- **Related code**: `model.py` decoder structure.

### H3) Cloze (-2) confused with Missing (-1)
- **Core idea**: Training uses `-2` for cloze (masked) targets, but `model.py` currently converts `-2` to `-1` before embedding. The model cannot distinguish "please predict this masked assay" from "this assay is missing".
- **Related code**: `model.py` `encode`/`decode` methods.

### H4) Supertrack behavior is not explicitly trained
- **Core idea**: "Supertrack" is a canonical target (high depth, standardized). If training never includes an objective to "output canonical signal when requested," the model may not discover this capability just from prompts.

### H5) Prompt values effectively low-entropy/OOD
- **Core idea**: Metadata like run_type may not vary enough during training to force the model to learn a dependency.

---

## C) Solutions Strategy

We have two main architectural paths (Solution 1 vs 2), plus a set of "Must-Do" tasks that apply to both.

### MUST-DO Tasks (Prerequisites)

#### ToDo 1: Metadata Hygiene & Assay Identity
- **Goal**: Clean inputs and explicitly tell the model *which* assay it's processing (since Platform ID is being removed).
- **Action**:
  1.  **Remove** `sequencing_platform` from `data.py` loading and `model.py` `MetadataEncoder`.
  2.  **Add** `Assay Identity` (categorical embedding, e.g., H3K4me3, CTCF) to the `MetadataEncoder` input.
  3.  **Keep**: `depth`, `read_length`, `run_type`.

#### ToDo 2: Fix Prompt Filling
- **Goal**: Stop training on hallucinations.
- **Action**: Change the default behavior of `fill_in_prompt` in `data.py`. During training, if a target is missing, `y_meta` must be missing (`-1`), never filled with median/mode.

---

### Solution 1: Fixed-Structure Decoder (Implicit Prompting)
**Status**: The "Minimal Change" approach. Keeps the current multi-head architecture where each assay has a fixed output channel.

- **Mechanism**: Rely on `MetadataEncoder` + `FiLM` to modulate the fixed heads.
- **Required Changes**:
  1.  **S3 (Distinct Cloze)**: Modify `model.py` to stop converting `-2` (cloze) to `-1` (missing). The `MetadataEncoder` must learn a distinct embedding for `-2`, signaling "This assay is requested".
  2.  **S4 (Random FiLM)**: Change `FiLMLayer` initialization from identity/zero to random. This breaks the symmetry/laziness where the model ignores the prompt and relies solely on the fixed decoder weights.
- **Logic**: The model implicitly learns: "If I see `-2` in metadata, I must predict this assay from context using the prompt. If I see real values, I denoise."

### Solution 2: Query-Based Decoder (Explicit Prompting)
**Status**: The "Architectural Fix". Replaces the fixed heads with a shared Spatial CNP decoder.

- **Mechanism**: A single shared decoder function `f(GenomicContext, Query) -> Track`. The `Query` is `AssayID + Metadata`.

- **Addressing Capacity Limitation (Morphology Variance)**:
  A single shared decoder might struggle to model diverse peak shapes (e.g., sharp H3K4me3 vs. broad domains) simultaneously. To add capacity **without re-introducing shortcuts**, use:
  - **Mixture-of-Experts (MoE)**: Use \(N\) shared expert blocks (e.g., different kernel sizes). The `QueryVector` generates gating weights to mix their outputs.
  - **Hypernetworks (Dynamic Conv)**: The `QueryVector` generates a low-rank update to the convolution weights, adapting the filters to the specific assay/depth requested.

- **Implementation Strategy (Backwards Compatible)**:
  - **Internal**:
    1.  Select `K` target assays (random subset during training).
    2.  Construct `QueryVectors` for these `K` assays.
    3.  Run the shared decoder (with MoE/DynamicConv) to get `K` tracks.
  - **Interface (API Compatibility)**:
    - To plug into the existing loss/metric framework, the forward pass should still return `[B, Total_Assays, L]`.
    - Scatter the `K` predicted tracks back into their correct slots.
    - Fill the remaining `Total_Assays - K` slots with `-1` (missing).
  - **Modular Class Design (for CLI & Backward Compatibility)**:
    - Add two new classes to `model.py` to encapsulate the new decoder logic:
      1.  `CNP_MoE_Decoder`
      2.  `CNP_DynConv_Decoder`
    - Plug these into the main `CANDI` class to fit nicely with the current design, minimizing modifications to existing implementations and enabling easy switching via CLI arguments in `train.py`.

- **Why**: This guarantees prompt usage (no fixed lane shortcut) while minimizing refactoring of the training loop/loss functions.

### Solution 3: Per-assay DSF sampling + per-assay loss gating (Counts-only DSF supervision)
**Status**: The "Training/Data Fix". Minimal architecture change; increases metadata variation and forces depth usage using only the existing DSF downsampled **count** tracks.

#### Motivation
- Today `y` is effectively pinned to DSF=1 counts (because `Y_loaded_*` is only refreshed at `dsf_pointer==0`), while `x` cycles DSF. This trains "DSF=k → DSF=1" but does **not** train controllability: the decoder can ignore `y_meta.depth` because the target distribution is always the same.
- Also, DSF is applied globally across assays within a biosample. This reduces combinatorial variation in "which assays are low/high depth" and makes it easier for the model to treat depth as redundant.
- Constraint: we only have DSF downsampled files for the **count** head (DSF=2/4/8); we do **not** have downsampled signal/pval or peaks.

#### Core idea
Sample DSF independently per assay for the **count** head, for both input and target, with an upsampling constraint:
- For each assay \(a\), sample `y_dsf[a]` and `x_dsf[a]` from the available DSFs and enforce `x_dsf[a] ≥ y_dsf[a]`.
- Load `x_counts[a]` from `signal_DSF{x_dsf[a]}` and `y_counts[a]` from `signal_DSF{y_dsf[a]}`.
- Set `x_meta.depth[a]` and `y_meta.depth[a]` using the DSF-specific `metadata.json` depth (so depth is consistent with the actual downsampled counts).

This creates many more (input, target, prompt) configurations and turns depth into a necessary signal: the same biosample+assay can be trained to map to multiple valid targets depending on the requested `y_meta.depth`.

#### Multi-head handling without DSF signal files
Because pval/signal and peaks only exist at DSF=1:
- Keep `y_pval` and `y_peaks` sourced from DSF=1 as today.
- Do **not** backpropagate pval/peaks losses for assays where `y_dsf[a] != 1` (per-assay gating).

Implementation sketch (high level):
- Construct an assay mask `sig_ok[a] = (y_dsf[a] == 1)`.
- When computing pval/peaks loss, restrict the loss to indices where `sig_ok[a]` is true (and availability is true). Practically this can be done by AND-ing the training masks with a broadcasted `sig_ok` so the loss sees those entries as "not selected".
- Count loss remains computed for all assays per the normal masking/availability logic.

This preserves correctness (we never pretend to have DSF≠1 signal ground truth) while still allowing DSF supervision to shape the shared latent representation and to teach the model to respect depth prompts via the count head.

#### Why this helps prompt sensitivity
- It breaks the "always output DSF=1" shortcut by making `y` vary, and ties that variation to `y_meta.depth`.
- It increases within-sample heterogeneity (some assays high depth, some low depth), forcing the model to use metadata to interpret which context tracks are reliable.
- It aligns prompt with target (no hallucinated prompt filling), preventing the model from learning to ignore `y_meta`.

---

## D) Validation Strategy (Prompt Sensitivity Checks)

Run these checks during training validation to confirm metadata actually affects outputs.

1.  **Depth Sensitivity Ratio**
    - **Test**: Prompt with `log2(depth)=23` (Low) vs `log2(depth)=25` (High) on same input.
    - **Metric**: `Sum(Output_High) / Sum(Output_Low)`
    - **Ideal**: ~4 (Since \(2^{25} / 2^{23} = 2^2 = 4\)).
    - **Goal**: Verify output scale responds to depth prompt.

2.  **RunType Identity MSE**
    - **Test**: Prompt with `RunType=Single` vs `RunType=Paired` on same input.
    - **Metric**: MSE between the two prediction tensors.
    - **Ideal**: > 0 (Non-identical).
    - **Goal**: Verify categorical metadata changes output morphology (even slightly).

3.  **ReadLength Identity MSE**
    - **Test**: Prompt with `ReadLength=36` vs `ReadLength=100`.
    - **Metric**: MSE between the two prediction tensors.
    - **Ideal**: > 0 (Non-identical).
    - **Goal**: Verify technical covariates are not ignored.

4.  **DSF Invariance Ratio**
    - **Test**: Input `DSF=1` (High Depth) vs `DSF=4` (Low Depth), both prompted with **same** canonical target prompt (e.g., Depth=24).
    - **Metric**: `Sum(Output_DSF1) / Sum(Output_DSF4)` (or MSE).
    - **Ideal**: ~1 (Identical outputs).
    - **Goal**: Verify "Supertrack" behavior: model normalizes different inputs to the requested prompt level.

### Training-Set Supertrack Monitoring (Plan)

To monitor prompt sensitivity continuously during training (without waiting for full validation epochs), implement an on-the-fly check using training batches.

1.  **Add `_monitor_supertrack_on_batch` to `CANDI_TRAINER` in `train.py`**:
    - **Purpose**: Adapt checks 1-3 to work on the current training batch tensors.
    - **Logic**:
      - Temporarily switch model to `eval()` mode.
      - Clone the batch metadata `y_meta`.
      - **Check 1 (Depth)**: Create `y_low` (depth=23) and `y_high` (depth=25). Compute `Sum(High)/Sum(Low)` for valid prompt indices.
      - **Check 2 (RunType)**: Create `y_single` (0) and `y_paired` (1). Compute MSE.
      - **Check 3 (ReadLen)**: Create `y_short` (36) and `y_long` (100). Compute MSE.
      - Restore model to `train()` mode.
    - **Difference from Validation**: Skip Check 4 (DSF Invariance) as it requires loading paired external files not present in the batch.

2.  **Integrate into Training Loop**:
    - Call `_monitor_supertrack_on_batch` periodically (e.g., every 100 batches).
    - Log metrics (`train_st/depth_ratio`, `train_st/runtype_mse`, `train_st/readlen_mse`) to W&B/CSV.
    - This provides a real-time signal: if `depth_ratio` flatlines at 1.0, the model is ignoring prompts.

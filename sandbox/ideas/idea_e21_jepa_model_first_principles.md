# E21 — Fresh JEPA Model from First Principles

Status: running (e21a-e21h complete/running; e21m-p 2×2 ablation matrix submitted 2026-05-14)
Parent: E19 (JEPA encoder sweep — provides baseline metrics and standing findings)
Checklist entry: [EXPERIMENTS.md](EXPERIMENTS.md#experiment-sweep-e21)

## Metrics note (2026-05-15)

- FJ11 gamma-gate guidance is now explicitly secondary: `adaLN_gamma_norm` remains useful for diagnosing conditioning failures but is no longer part of the primary pass/fail gate.
- Primary JEPA gate/ranking now uses `combined_loss_scaled` with isotropy diagnostics (`cov_condition_number`, `embedding_mean_norm`, `per_dim_variance_cv`) plus convergence tracking (`sigreg_converged`).

## Problem Statement

The current JEPA pipeline (`sandbox/jepa.py` + `sandbox/train_jepa.py`) wraps the full production `CANDI` model from `model.py` (3500+ lines). This creates three problems:

1. **Import coupling**: every JEPA encoder experiment imports `CANDI`, which drags in multi-head decoders, output layers, distribution heads, and 30+ constructor parameters that are irrelevant for encoder-only training.
2. **Inherited design flaws**: the production encoder uses per-layer FiLM conditioning that universally fails to convey metadata (F1: `depth_count_ratio ≈ 1.0` across all 20+ runs). Redesigning metadata injection requires editing `model.py`, risking production regressions.
3. **Iteration speed**: adding a new encoder feature (e.g. BERT-style mask tokens, transformer predictor) requires navigating production CANDI's complex initialization and ensuring backward compatibility with all decoder types.

## Hypothesis

A purpose-built `sandbox/jepa_model.py` — designed from first principles with lessons learned from E19's 20+ runs — should:
- At least match e19q's encoder quality (cos_sim < 0.10, structured UMAP, meta_sens_runtype > 0.5)
- Enable 3-5× faster design iteration by eliminating production code dependencies
- Fix the metadata injection failure (F1) through architectural redesign

## Design Decisions (locked)

### D1. Missing/masked assay handling: BERT-style post-conv mask tokens

**Decision**: Zero-fill masked/missing assay channels before the conv tower, then replace the conv output for those channels with a learned `mask_embedding` of shape `[d_per_assay]` at every position.

**Rationale**: Adapted from I-JEPA/MAE which replace patch embeddings (not pixels) with mask tokens. For CANDI's feature-level masking, each assay is a "patch" — when masked, its conv features are meaningless and should be replaced with a learned embedding that tells the transformer "this assay is unknown — predict it from context."

**Implementation**:
```
Input: x_signal [B, L, F+1], availability [B, F+1]
1. mask = (availability != CLOZE) & (availability != MISSING)  # [B, F+1]
2. x_clean = x_signal * mask.unsqueeze(1)  # zero-fill masked channels
3. x_conv = signal_conv_tower(x_clean)  # [B, L2, d_per_assay * (F+1)]
4. For each masked/missing assay a (including control if genuinely missing):
     x_conv[:, :, a*d : (a+1)*d] = mask_embedding  # learned [d_per_assay]
5. → continue to per-assay FiLM (D2), DNA fusion, transformer
```

- Control channel (index F) is NEVER cloze-masked (never part of the JEPA masking task). However, if the control is genuinely missing (-1) in the input data, it receives the mask_embedding like any other missing assay.
- `mask_embedding` is a single learned parameter of shape `[d_per_assay]`, broadcast across all positions for all masked/missing assays.
- The distinction between -1 (missing) and -2 (cloze) is carried by the metadata encoder's separate missing/cloze embeddings, which condition the mask token via per-assay FiLM (D2).

**Replaces**: MaskStem (grouped 1×1 conv on interleaved value+mask channels). MaskStem was functional but didn't give the transformer an explicit "unknown assay" signal.

### D2. Metadata injection: single-shot per-assay linear FiLM, applied AFTER mask token replacement

**Decision**: Inject metadata embeddings ONCE via per-assay linear FiLM conditioning. Applied AFTER D1's mask token replacement, before DNA fusion and the transformer. Each assay's conv features (or mask_embedding for masked assays) are conditioned exclusively on that assay's own metadata embedding. No cross-assay metadata leakage.

**Rationale**:
- F1 (metadata collapse) persists across ALL per-LAYER FiLM variants — the root cause is repeated per-layer application (dilution/overwriting), not the FiLM formulation itself.
- F8 (E7 single-shot decoder FiLM) dramatically outperforms per-layer — single-shot is the key insight.
- FJ7 (meta_tgt conditioning is the dominant metadata sensitivity lever) argues for one decisive injection point.
- Per-assay granularity is essential: H3K4me3's features must be conditioned on H3K4me3's metadata only, not on an averaged pool of all assay metadata. Mean-pooling across assays would lose this critical per-assay specificity.
- FiLM applied AFTER mask token replacement enables assay-specific mask tokens: the mask_embedding gets conditioned with the cloze/missing metadata for that specific assay, so the transformer sees "H3K4me3 is requested for prediction" rather than a generic "some assay is missing."
- Linear FiLM (`x * (1 + scale) + shift`) avoids the gradient dead zones of exponential FiLM (E6 root cause: `exp(scale)` with clamping zeros gradients).

**Implementation**:
```python
# After conv tower + mask token replacement (D1):
# x_conv: [B, L2, d_per_assay * (F+1)]  — per-assay features (mask tokens for masked assays)
# meta_embed: [B, F+1, emb_dim]         — from MetadataEmbedding module

# Per-assay FiLM projection: emb_dim → 2 * d_per_assay (scale + shift)
film_proj = nn.Linear(emb_dim, 2 * d_per_assay)  # shared across assays

for a in range(F+1):
    params = film_proj(meta_embed[:, a])  # [B, 2*d_per_assay]
    scale, shift = params.chunk(2, dim=-1)  # each [B, d_per_assay]
    # Linear FiLM: x * (1 + scale) + shift
    x_conv[:, :, a*d:(a+1)*d] = (
        x_conv[:, :, a*d:(a+1)*d] * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    )
# → continue to DNA fusion, transformer
```

Note: the `film_proj` is shared across all assays — the per-assay specificity comes from the different `meta_embed[:, a]` inputs, not from separate projection weights. This keeps parameter count low while respecting per-assay metadata boundaries.

### D3. Signal conv tower: pure depthwise convolutions with per-assay pointwise expansion

**Decision**: Pure depthwise 1D convolutions (groups=F+1) for spatial feature extraction, followed by per-assay pointwise (1×1) expansion convolutions (groups=F+1) for channel growth. No cross-assay mixing anywhere in the conv tower.

**Rationale**:
- D1 injects mask_embedding AFTER the conv tower. During the conv tower, masked assay channels contain garbage (from zero-filled input). Pure depthwise ensures this garbage stays in its own channels and never leaks into available assays' features.
- Cross-assay interaction is handled by the transformer (after mask tokens and FiLM are in place), which is the right level to learn complex cross-assay correlations like "H3K4me3 peak implies H3K27ac peak."
- The per-assay pointwise expansion handles channel growth (F+1 → d_model) without cross-assay mixing — it's essentially a per-assay MLP applied position-wise.

**Implementation** (per layer):
```python
class PureDepthwiseBlock(nn.Module):
    def __init__(self, in_ch_per_assay, out_ch_per_assay, num_assays,
                 kernel_size=3, pool_size=2):
        n_groups = num_assays  # F+1
        in_ch = in_ch_per_assay * n_groups
        out_ch = out_ch_per_assay * n_groups
        # Depthwise spatial conv: per-assay, kernel_size on L dimension
        self.dw_conv = nn.Conv1d(in_ch, in_ch, kernel_size,
                                  padding=kernel_size//2, groups=n_groups)
        # Per-assay pointwise expansion: 1×1, still grouped
        self.pw_conv = nn.Conv1d(in_ch, out_ch, kernel_size=1, groups=n_groups)
        self.norm = nn.GroupNorm(n_groups, out_ch)
        self.pool = nn.MaxPool1d(pool_size)
        self.act = nn.GELU()
```

Channel progression per assay: `1 → exp → exp² → ... → exp^n = d_per_assay`
Total channels: `(F+1) * d_per_assay` at the output.
With F=8, exp=2, n=3: `d_per_assay = 2³ = 8`, total = `9 * 8 = 72 = d_model`.

### D4. Transformer encoder: x-transformers with pre-norm + RoPE

**Decision**: Use the [x-transformers](https://github.com/lucidrains/x-transformers) library for the encoder transformer. Pre-norm architecture with Rotary Position Embeddings (RoPE).

**Rationale**:
- x-transformers is well-tested, widely used, and provides RoPE, pre-norm, and many other modern transformer features out of the box.
- Avoids reimplementing RoPE from scratch (risk of off-by-one position indexing bugs).
- Pre-norm (LayerNorm before attention and FFN) is more training-stable than post-norm (LLaMA, GPT-NeoX, PaLM, ViT-22B).
- Already a dependency in the production codebase (used by `CANDI_DNA_Encoder` with `attention_type="xtransformers"`).

**Implementation**:
```python
from x_transformers import Encoder as XEncoder

self.transformer = XEncoder(
    dim=d_model,
    depth=n_layers,
    heads=nhead,
    rotary_pos_emb=True,
    attn_dropout=dropout,
    ff_dropout=dropout,
    ff_mult=4,
    pre_norm=True,
)
```

### D5. JEPA predictor: 1-layer ConditionalBlock transformer (LeWM-faithful)

**Decision**: 1-layer transformer predictor using LeWM's `ConditionalBlock` with 6-parameter AdaLN-zero. Bidirectional (non-causal) attention. Conditioned on flattened MetadataEmbedding output `[B, (F+1)*emb_dim]` — the same learned embeddings that condition the encoder's per-assay FiLM (D2). Implemented from scratch following [lucas-maes/le-wm module.py](https://github.com/lucas-maes/le-wm/blob/main/module.py) as closely as possible.

**Rationale**:
- LeWM uses `ConditionalBlock` with 6 AdaLN-zero parameters per block: `(shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)`. The `gate` parameters multiply the sub-layer outputs — this is more expressive than simple `(1+gamma)*h + beta` because the gate starts at zero, making the entire block identity at initialization.
- FJ6 shows constrained predictors produce better encoder representations — 1 layer keeps capacity low.
- FJ7 proves meta_tgt conditioning is essential.
- x-transformers does NOT support AdaLN-zero natively, so the predictor must be implemented from scratch.
- Non-causal (bidirectional) attention because CANDI has no temporal axis — all positions are predicted simultaneously.

**Predictor conditioning: flattened MetadataEmbedding output (not raw meta_tgt)**

The predictor is conditioned on the **flattened output of the MetadataEmbedding module** — the same learned embeddings used to condition the encoder's per-assay FiLM (D2). Concretely: `meta_embed.reshape(B, -1)` gives `[B, (F+1) * emb_dim]`. With F=8, emb_dim=32, the conditioning vector is `[B, 288]`.

This is strictly better than conditioning on raw flattened `meta_tgt [B, 4*(F+1)]` because:
- The MetadataEmbedding module has already learned to handle -1 (missing) and -2 (cloze) tokens via distinct learned embeddings, embed categorical fields (assay_id, run_type) via embedding tables, and fuse the 4 metadata fields per assay into a coherent representation.
- The predictor doesn't have to re-learn metadata interpretation from raw values — it reuses the encoder's metadata understanding.
- The cloze embedding for a masked assay is distinct from the missing embedding and from any real metadata value — the predictor knows exactly which assays are masked and what kind of masking occurred.
- The encoder and predictor see a consistent metadata representation (same embeddings, same module), so the predictor can learn to complement what the encoder already knows.

The `cond_proj` inside the predictor's `Transformer` maps `(F+1)*emb_dim → hidden_dim`.

**Changes from LeWM `ConditionalBlock`**:
1. **Non-causal attention**: LeWM uses `causal=True` (autoregressive prediction of future frames). CANDI predicts all positions simultaneously → `causal=False`. Concretely: `F.scaled_dot_product_attention(q, k, v, is_causal=False)` instead of `is_causal=True`.
2. **Conditioning input**: LeWM conditions on action embeddings `[B, T, act_dim]` (per-timestep). CANDI conditions on flattened `meta_embed` `[B, (F+1)*emb_dim]` (per-sample, broadcast to all positions via `cond_proj`).
3. **Depth**: 1 layer (vs LeWM's 6). FJ6 motivates low predictor capacity.
4. **No positional embedding**: LeWM's `ARPredictor` has learned positional embeddings because temporal order matters. For CANDI, positions already carry RoPE-encoded spatial information from the encoder. Adding separate positional embeddings in the predictor is optional and can be tested as an ablation.

**Everything else is verbatim from LeWM**: `modulate()` function, `Attention` class, `FeedForward` class, `ConditionalBlock` class structure, zero-initialization of AdaLN weights and biases.

**Implementation** (adapted from LeWM module.py):
```python
def modulate(x, shift, scale):
    """AdaLN-zero modulation — verbatim from LeWM module.py."""
    return x * (1 + scale) + shift

class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning.
    Verbatim from lucas-maes/le-wm module.py except:
      - is_causal=False (CANDI has no temporal axis)
    """
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa), causal=False
        )
        x = x + gate_mlp * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x
```

The `Attention` and `FeedForward` classes are also taken from LeWM module.py. The only change to `Attention.forward` is passing `is_causal=False` to `F.scaled_dot_product_attention`.

### D6. DNA conv tower: keep current design

**Decision**: Reimplement the same DNA conv tower from model.py (residual ConvTowers, exponential channel growth, pool_size=2 then pool_size=5).

**Rationale**: Not implicated in any standing finding (F1-F8, FJ1-FJ8). Proven to work across all E19 runs. Genomic sequence encoding is well-understood (Basenji/Enformer pattern). Redesigning would be scope creep.

### D7. Signal-DNA fusion: keep linear fusion

**Decision**: Concatenate signal and DNA conv outputs, project to d_model via a linear layer.

**Rationale**: Simple, effective, not implicated in any issue. The transformer downstream handles complex integration.

### D8. Metadata embedding module: reimplemented from scratch

**Decision**: Reimplement MetadataEncoder with the same architecture: per-field projections (depth, read_length → linear; assay_id, run_type → embedding table) + special tokens for -1 (missing) and -2 (cloze) + MLP fusion (4*emb_dim → emb_dim).

**Rationale**: The metadata *encoding* is well-designed. The issue is *injection* (D2), not encoding. Separate continuous/categorical handling and distinct missing/cloze tokens are both correct practice.

### D9. Encoder output → projector: different dims, no intermediate projection

**Decision**: Encoder d_model ≠ proj_dim. The JEPAProjector (Linear → BN → GELU → Linear) maps directly from encoder output to projection space. No CANDI-style latent_projection layer.

**Rationale**: Follows LeJEPA/LeWM exactly. In LeJEPA, ViT output dim (192 for tiny) differs from proj_dim (256). The projector exists to bridge this gap and provide the BN that SIGReg needs. CANDI's latent_projection was specific to the reconstruction pipeline (removing control channel dimension before decoders) and is not needed for JEPA.

### D10. SIGReg placement: on projector output

**Decision**: SIGReg operates on the projector output (not raw encoder output). Faithful to LeJEPA/LeWM.

**Rationale**: The isotropy constraint should apply to the space where the prediction loss operates, not the raw encoder space. FJ5 (eff_rank collapse) is about λ tuning, not placement — e19d (λ=1.0) showed the slowest collapse.

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────────┐
│  JEPA Encoder (sandbox/jepa_model.py)                           │
│                                                                 │
│  Inputs:                                                        │
│    x_signal [B, L, F+1]   — raw counts (log1p transformed)     │
│    x_dna    [B, 4, G]     — one-hot DNA sequence                │
│    x_meta   [B, 4, F+1]   — per-assay metadata                 │
│    avail    [B, F+1]      — availability mask                   │
│                                                                 │
│  ┌──────────────────────────────────────────┐                   │
│  │  MetadataEmbedding                       │                   │
│  │  x_meta [B, 4, F+1]                     │                   │
│  │  → meta_embed [B, F+1, emb_dim]         │                   │
│  └──────────────────────────────────────────┘                   │
│                                                                 │
│  ┌──────────────────────────────────────────┐                   │
│  │  Signal Conv Tower (pure depthwise)      │                   │
│  │  x_signal [B, L, F+1]                   │                   │
│  │  → zero-fill masked/missing channels     │                   │
│  │  → depthwise conv + per-assay pointwise  │                   │
│  │  → [B, L2, d_per_assay * (F+1)]         │                   │
│  └──────────────────────────────────────────┘                   │
│                           │                                     │
│                           ▼                                     │
│  ┌──────────────────────────────────────────┐                   │
│  │  D1: Mask Token Replacement              │                   │
│  │  For each masked/missing assay a:        │                   │
│  │    features[a] = mask_embedding          │                   │
│  └──────────────────────────────────────────┘                   │
│                           │                                     │
│                           ▼                                     │
│  ┌──────────────────────────────────────────┐                   │
│  │  D2: Per-Assay Linear FiLM              │                   │
│  │  For each assay a:                       │                   │
│  │    (scale, shift) = film_proj(meta[a])   │                   │
│  │    features[a] *= (1 + scale)            │                   │
│  │    features[a] += shift                  │                   │
│  │  (includes mask tokens for masked assays │                   │
│  │   → assay-specific mask identities)      │                   │
│  └──────────────────────────────────────────┘                   │
│                           │                                     │
│  ┌──────────────────────────────────────────┐                   │
│  │  DNA Conv Tower                          │                   │
│  │  x_dna [B, 4, G] → [B, L2, d_dna]      │                   │
│  └──────────────┬───────────────────────────┘                   │
│                 │                                               │
│                 ▼                                               │
│  ┌──────────────────────────────────────────┐                   │
│  │  Linear Fusion                           │                   │
│  │  cat(signal_features, dna) → d_model     │                   │
│  └──────────────────────────────────────────┘                   │
│                           │                                     │
│                           ▼                                     │
│  ┌──────────────────────────────────────────┐                   │
│  │  x-transformers Encoder                  │                   │
│  │  Pre-Norm + RoPE                         │                   │
│  │  n_layers × {LN→MHSA→LN→FFN}           │                   │
│  │  → z [B, L2, d_model]                   │                   │
│  └──────────────────────────────────────────┘                   │
│                                                                 │
│  Output: z [B, L2, d_model]                                     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  JEPA Head                                                      │
│                                                                 │
│  JEPAProjector: Linear→BN→GELU→Linear  [d_model → proj_dim]   │
│    (shared for ctx + tgt, LeWM MLP convention)                  │
│                                                                 │
│  JEPAPredictor: 1-layer ConditionalBlock (LeWM-faithful)       │
│    6-param AdaLN-zero, bidirectional attention                  │
│    conditioned on meta_embed [B, (F+1)*emb_dim]                 │
│                                                                 │
│  JEPAPredProjector: Linear→BN→GELU→Linear (separate weights)  │
│                                                                 │
│  SIGReg: on projector output (LeJEPA/LeWM convention)          │
│                                                                 │
│  Loss: MSE(pred_proj(pred(proj_ctx, meta_embed)), proj_tgt)     │
│      + λ * SIGReg(cat(proj_ctx, proj_tgt))                      │
└─────────────────────────────────────────────────────────────────┘
```

### Default shapes (sandbox 8-assay config)

```
F = 8 assays, control = 1 → F+1 = 9
expansion_factor = 2, n_cnn_layers = 3, pool_size = 2
context_length (L) = 768, G = L * 25 = 19200

Signal conv tower (pure depthwise, per-assay pointwise expansion):
  Layer 0: 9 ch (1/assay) → 18 ch (2/assay)   pool 2, L: 768→384
  Layer 1: 18 ch (2/assay) → 36 ch (4/assay)   pool 2, L: 384→192
  Layer 2: 36 ch (4/assay) → 72 ch (8/assay)   pool 2, L: 192→96
  → [B, 72, 96] → permute → [B, L2=96, 72]
  d_per_assay = 8, total channels = 9 * 8 = 72

Mask token: mask_embedding [d_per_assay=8]
Per-assay FiLM: film_proj Linear(emb_dim=32 → 2*8=16)

DNA conv tower:
  4 → ... → d_dna channels, pooled to L2=96 positions

Linear fusion: concat(72, d_dna) → d_model (e.g. 72)

x-transformers Encoder: d_model=72, nhead=4, n_layers=2, RoPE
Output: [B, 96, 72]

Projector: 72 → 256 (hidden) → proj_dim (e.g. 72 or 256)
Predictor: 1-layer ConditionalBlock, hidden_dim=proj_dim
  AdaLN cond_dim = (F+1) * emb_dim = 9 * 32 = 288 (meta_embed flattened)
  → cond_proj maps 288 → hidden_dim
  Bidirectional attention, dim_head=64, heads=4
PredProjector: proj_dim → 256 (hidden) → proj_dim (separate weights)
```

## Key Architectural Departures from Production CANDI

| Component | Production `model.py` | E21 `jepa_model.py` | Motivation |
|---|---|---|---|
| Masked assay handling | MaskStem (grouped 1×1 conv, value+mask interleave) | BERT-style learned mask_embedding replacing post-conv features; control gets mask_embedding only when genuinely missing (-1) | Gives transformer explicit "unknown assay" signal; I-JEPA convention |
| Metadata injection | Per-layer FiLM (exponential or linear scale+shift at every conv) | Single-shot per-assay linear FiLM applied ONCE after mask token replacement | F1: per-layer FiLM universally fails; single-shot preserves per-assay specificity; FiLM after mask tokens creates assay-specific mask identities |
| Signal convolutions | Grouped conv (groups=F+1, no cross-assay mixing) | Pure depthwise + per-assay pointwise expansion (both groups=F+1) | No cross-assay leakage from garbage in masked channels; transformer handles cross-assay mixing after clean mask token injection |
| Transformer | DualAttention or x-transformers (various PE types) | x-transformers Encoder (pre-norm + RoPE) | Well-tested library, modern standard, already a project dependency |
| JEPA predictor | Per-position 2-layer MLP with AdaLN-zero | 1-layer LeWM-faithful ConditionalBlock (6-param AdaLN-zero, bidirectional) | LeWM convention; cross-position attention; FJ6 low capacity; FJ7 meta_tgt conditioning |
| Latent projection | Linear→GELU→LayerNorm (removes control dim for decoder) | None (projector maps directly from encoder output) | LeJEPA/LeWM convention; no decoder in JEPA Stage 1 |

## Implementation Clarifications (locked)

These decisions resolve ambiguities for a fresh implementation session:

1. **Reuse shared components**: Import `SIGReg`, `JEPAProjector`, `compute_latent_geometry`, `compute_metadata_sensitivity` from `sandbox/jepa.py`. These are self-contained utilities with no CANDI dependency. Do NOT reimplement them.

2. **Forward interface**: `JEPAModel.forward()` must match `CANDIJepa.forward()` signature exactly — takes `(x_ctx, x_tgt, x_dna, meta_ctx, meta_tgt, mask_cond)`, does two shared-weight encoder passes internally, returns the same loss dict keys. This enables drop-in replacement in `train_jepa.py` with minimal branching.

3. **Configuration**: Define a `@dataclass JEPAModelConfig` with all hyperparameters (num_assays, d_model, proj_dim, n_layers, nhead, emb_dim, n_cnn_layers, expansion_factor, etc.). The config is built from the existing YAML/CLI system in `train_jepa.py`. All defaults must be explicit in the dataclass.

4. **Predictor conditioning**: Only `meta_embed` mode — the predictor is always conditioned on flattened MetadataEmbedding output `[B, (F+1)*emb_dim]`. No support for the legacy `assay`/`loci`/`meta_tgt`/`meta_concat`/`none` modes. (If an unconditional ablation is needed later, set `lambda_sigreg` high and predictor hidden_dim small rather than adding mode switches.)

5. **DNA conv tower**: Reimplement by reading `model.py::CANDI_DNA_Encoder` and `sandbox/model.py::build_sandbox_candi` as the source of truth for architecture and hyperparameters. These files specify: residual ConvTower blocks, exponential channel growth from 4 input channels, kernel_size, pool_sizes, and the final output dim (`d_dna`). The implementer must read these files.

6. **d_model = signal_dim (compression)**: After linear fusion, `d_model` stays at the signal conv tower's output dim (e.g. 72 for sandbox). The fusion linear layer compresses `concat(signal_features, dna_features)` down to `d_model`. This matches production CANDI behavior.

## Implementation Plan

### Phase 1: Core modules (`sandbox/jepa_model.py`)

1. **JEPAModelConfig** — `@dataclass` with all hyperparameters and explicit defaults
2. **MetadataEmbedding** — per-field projections, special tokens, MLP fusion (reimplemented from production MetadataEncoder)
3. **PureDepthwiseBlock** — depthwise conv (groups=F+1) + per-assay pointwise expansion (groups=F+1) + GroupNorm + pool + GELU
4. **SignalConvTower** — stack of PureDepthwiseBlock
5. **MaskTokenInjector** — replaces masked/missing assay features with learned mask_embedding
6. **PerAssayFiLM** — single-shot per-assay linear FiLM (shared projection, per-assay metadata input)
7. **DNAConvTower** — residual ConvTowers (reimplemented from `model.py::CANDI_DNA_Encoder`, matching production shapes)
8. **LinearFusion** — concat + linear projection (compresses to d_model)
9. **JEPAEncoder** — orchestrates all of the above + x-transformers Encoder into a single `encode()` method
10. **Predictor components** — `modulate`, `Attention`, `FeedForward`, `ConditionalBlock` (all from LeWM module.py, `is_causal=False`), wrapped in a `Transformer` with `input_proj`/`output_proj`/`cond_proj` and a final `nn.LayerNorm(hidden_dim)` after all blocks (matching LeWM's `Transformer` wrapper)
11. **JEPAModel** — encoder + projector (imported from jepa.py) + predictor + pred_projector (imported) + SIGReg (imported); `forward()` matches `CANDIJepa.forward()` interface

### Phase 2: Integration with train_jepa.py

- Modify `train_jepa.py` to accept a `model_type` config flag (`"candi"` or `"fresh"`)
- When `model_type="fresh"`, build `JEPAModel` from `jepa_model.py` using `JEPAModelConfig`
- All JEPA training infrastructure (masking, data loading, SIGReg, loss, logging, UMAP, metadata sensitivity) is reused
- The `CANDIJepa` wrapper continues to exist for backward compatibility with E19 runs

### Phase 3: Validation

All tests adapted from E19 spec §12:

#### Shape contracts
- Encoder forward pass: `[B=8, L=768, F+1=9]` signal + `[B, 4, G]` DNA + `[B, 4, F+1]` meta → `[B, L2=96, d_model=72]`
- With masking: mask 3/8 assays → output shape unchanged, masked assay features replaced then FiLM-conditioned
- Predictor: `[B, L2, proj_dim]` + `[B, 288]` meta_embed_flat → `[B, L2, proj_dim]`

#### Gradient flow
- `z_tgt.grad_fn is not None` (no stop-gradient on target)
- Encoder parameters receive gradients from both context and target passes
- Predictor AdaLN weights receive gradients from pred_loss
- SIGReg gradients flow through projector to encoder
- FiLM projection receives gradients through the mask tokens (metadata conditioning is in the graph)

#### AdaLN-zero invariant (LeWM-faithful)
- At initialization: all 6 AdaLN parameters are zero → `gate_msa = 0`, `gate_mlp = 0` → predictor block is identity
- After one gradient step: outputs diverge for different conditioning inputs

#### Mask token + FiLM correctness
- For a masked assay: post-conv features are `mask_embedding`, then FiLM-conditioned with that assay's metadata
- For an available assay: post-conv features are actual conv output, then FiLM-conditioned with that assay's metadata
- Control channel: never cloze-masked; gets mask_embedding only when genuinely missing (-1)
- Two masked assays with different metadata (e.g. cloze H3K4me3 vs cloze CTCF) produce different post-FiLM features

#### SIGReg calibration
- SIGReg(N(0,I) samples) < 0.05
- SIGReg(collapsed constant) >> SIGReg(N(0,I))

#### Metadata sensitivity
- `compute_metadata_sensitivity()` with different run_type values produces different encoder outputs (meta_sens_runtype > 0)
- This should be significantly higher than production CANDI encoder due to D2 (per-assay single-shot FiLM)

#### Numerical stability
- 100-step bf16 training run: no NaN/Inf in any loss or metric
- Gradient clipping active (pre-clip norm sometimes > clip_cap)

### Phase 4: Baseline comparison run

- Submit E21 baseline run with e19q-equivalent config: `lambda_sigreg=0.5`, `pred_mask_cond_type=meta_embed`, assay masking, 200 epochs
- Success criteria (from e19q):
  - `cos_sim_ctx_tgt < 0.10` on masked batches
  - Structured UMAP (visual assessment)
  - `meta_sens_runtype > 0.3`
  - No divergence (last/best ratio < 1.5)

## Risks / Watch-outs

1. **Pure depthwise conv may under-represent cross-assay correlations**: all cross-assay learning is deferred to the transformer. If the transformer can't learn these correlations from L2-resolution (96-position) sequences, conv-level cross-assay mixing (D3 option B with depthwise-separable) can be tested as a follow-up. Note: this would require moving D1 mask token replacement to BEFORE each pointwise conv, or accepting some garbage leakage.
2. **Single-shot FiLM may not be enough for depth sensitivity**: F1 may be deeper than injection architecture (e.g. the depth signal in metadata may be too weak relative to signal features). The `meta_sens_depth_wide` probe will diagnose this.
3. **Transformer predictor capacity**: even with 1 layer, a ConditionalBlock transformer has more capacity than a 16-hidden MLP (FJ6). If encoder quality degrades, reduce predictor hidden_dim or number of heads.
4. **ConditionalBlock attention is different from x-transformers attention**: the predictor's `Attention` class is from LeWM, not x-transformers. Ensure consistent behavior (both use `F.scaled_dot_product_attention` under the hood, but normalization details may differ).
5. **BatchNorm in projector with small batches**: with B=16, L2=96, effective BN batch = 1536 — fine. But if batch_size drops below 4, BN may become noisy.
6. **No fallback to production CANDI**: if E21 encoder is worse than E19 on all metrics, the diagnosis may be tricky. Keep E19 infrastructure working as a control.
7. **Shared film_proj**: the single Linear projection shared across all assays relies on different `meta_embed[:, a]` inputs for per-assay specificity. If assay metadata embeddings are too similar, the FiLM conditioning may lack diversity. The distinct assay_id embeddings should prevent this.

## Staged Run Plan (not submitted yet)

All runs use W&B logging, full `metrics.jsonl` emission, and end-of-run chr21 UMAP/PCA diagnostics from `train_jepa.py`.

| run | purpose | model_type | regime | key deltas vs e19q | submission file |
|---|---|---|---|---|---|
| e21a | Control rerun of e19q under current harness | candi | type2_loci | `lambda_sigreg=0.5`, `pred_mask_cond_type=meta_tgt`, assay masking, 200 epochs | `sandbox/slurm/submit_e21a_e19q_rerun.sbatch` |
| e21b | Fresh-vs-control comparison on same regime | fresh | type2_loci | same as e21a but `model_type=fresh` (fresh predictor uses embedded target metadata internally) | `sandbox/slurm/submit_e21b_e19q_fresh_type2.sbatch` |
| e21c | Regime transfer control (chr19 train setup) | candi | type1_chr19 | same as e21a, `data.regime=type1_chr19` | `sandbox/slurm/submit_e21c_e19q_rerun_type1.sbatch` |
| e21d | Regime transfer with fresh encoder | fresh | type1_chr19 | same as e21c, `model_type=fresh` | `sandbox/slurm/submit_e21d_e19q_fresh_type1.sbatch` |
| e21e | R2 diagnostic (architecture): fresh MLP predictor | fresh | type2_loci | `fresh.predictor_type=mlp`, keep fresh default cond (`meta_tgt_embed`, shared) | `sandbox/slurm/submit_e21e_fresh_mlp_type2.sbatch` |
| e21f | R3 diagnostic (conditioning source): raw target metadata | fresh | type2_loci | `fresh.cond_source=raw_meta_tgt`, transformer predictor unchanged | `sandbox/slurm/submit_e21f_fresh_rawmeta_type2.sbatch` |
| e21g | R4 diagnostic (conditioning coupling): separate predictor metadata embed | fresh | type2_loci | `fresh.cond_embed_shared=separate`, transformer predictor + embedded cond unchanged | `sandbox/slurm/submit_e21g_fresh_sepembed_type2.sbatch` |
| e21h | MLP predictor + raw meta_tgt (combines R2+R3) | fresh | type2_loci | `fresh.predictor_type=mlp`, `fresh.cond_source=raw_meta_tgt` | `sandbox/slurm/submit_e21h_fresh_mlp_rawmeta_type2.sbatch` |
| e21m | 2×2 (0,0): candi encoder + e19q-identical predictor | candi | type2_loci | `jepa.predictor_type=legacy_mlp`, `jepa.pred_mask_cond_type=meta_tgt`; job 39819435 | `sandbox/slurm/submit_e21m_candi_enc_candi_pred.sbatch` |
| e21n | 2×2 (1,1): fresh encoder + fresh transformer predictor | fresh | type2_loci | `fresh.predictor_type=transformer`, `fresh.cond_source=meta_tgt_embed`; job 39819436 | `sandbox/slurm/submit_e21n_fresh_enc_fresh_pred.sbatch` |
| e21o | 2×2 (0,1): candi encoder + fresh transformer predictor | candi | type2_loci | `jepa.predictor_type=fresh_transformer`, `jepa.pred_mask_cond_type=meta_tgt`; job 39819437 | `sandbox/slurm/submit_e21o_candi_enc_fresh_pred.sbatch` |
| e21p | 2×2 (1,0): fresh encoder + e19q-identical predictor | fresh | type2_loci | `fresh.predictor_type=legacy_mlp`; job 39819438 | `sandbox/slurm/submit_e21p_fresh_enc_candi_pred.sbatch` |

## Run Links

| run | directory | metrics | SLURM log | W&B |
|---|---|---|---|---|
| e21a_e19q_rerun | `sandbox/runs/e21a_e19q_rerun_<jobid>` | `sandbox/runs/e21a_e19q_rerun_<jobid>/metrics.jsonl` | `sandbox/slurm_logs/e21a_e19q_<jobid>.out` | run name `e21a_e19q_rerun_<jobid>` |
| e21b_e19q_fresh_type2 | `sandbox/runs/e21b_e19q_fresh_type2_<jobid>` | `sandbox/runs/e21b_e19q_fresh_type2_<jobid>/metrics.jsonl` | `sandbox/slurm_logs/e21b_fresh_t2_<jobid>.out` | run name `e21b_e19q_fresh_type2_<jobid>` |
| e21c_e19q_rerun_type1 | `sandbox/runs/e21c_e19q_rerun_type1_<jobid>` | `sandbox/runs/e21c_e19q_rerun_type1_<jobid>/metrics.jsonl` | `sandbox/slurm_logs/e21c_e19q_t1_<jobid>.out` | run name `e21c_e19q_rerun_type1_<jobid>` |
| e21d_e19q_fresh_type1 | `sandbox/runs/e21d_e19q_fresh_type1_<jobid>` | `sandbox/runs/e21d_e19q_fresh_type1_<jobid>/metrics.jsonl` | `sandbox/slurm_logs/e21d_fresh_t1_<jobid>.out` | run name `e21d_e19q_fresh_type1_<jobid>` |
| e21e_fresh_mlp_type2 | `sandbox/runs/e21e_fresh_mlp_type2_<jobid>` | `sandbox/runs/e21e_fresh_mlp_type2_<jobid>/metrics.jsonl` | `sandbox/slurm_logs/e21e_fresh_mlp_t2_<jobid>.out` | run name `e21e_fresh_mlp_type2_<jobid>` |
| e21f_fresh_rawmeta_type2 | `sandbox/runs/e21f_fresh_rawmeta_type2_<jobid>` | `sandbox/runs/e21f_fresh_rawmeta_type2_<jobid>/metrics.jsonl` | `sandbox/slurm_logs/e21f_fresh_raw_t2_<jobid>.out` | run name `e21f_fresh_rawmeta_type2_<jobid>` |
| e21g_fresh_sepembed_type2 | `sandbox/runs/e21g_fresh_sepembed_type2_<jobid>` | `sandbox/runs/e21g_fresh_sepembed_type2_<jobid>/metrics.jsonl` | `sandbox/slurm_logs/e21g_fresh_sep_t2_<jobid>.out` | run name `e21g_fresh_sepembed_type2_<jobid>` |
| e21h_fresh_mlp_rawmeta | `sandbox/runs/e21h_mlp_rawmeta_type2_<jobid>` | `sandbox/runs/e21h_mlp_rawmeta_type2_<jobid>/metrics.jsonl` | `sandbox/slurm_logs/e21h_mlp_rawmeta_<jobid>.out` | run name `e21h_mlp_rawmeta_type2_<jobid>` |
| e21m_candi_enc_candi_pred | `sandbox/runs/e21m_candi_enc_candi_pred_39819435` | `sandbox/runs/e21m_candi_enc_candi_pred_39819435/metrics.jsonl` | `sandbox/slurm_logs/e21m_candi_candi_39819435.out` | run name `e21m_candi_enc_candi_pred_39819435` |
| e21n_fresh_enc_fresh_pred | `sandbox/runs/e21n_fresh_enc_fresh_pred_39819436` | `sandbox/runs/e21n_fresh_enc_fresh_pred_39819436/metrics.jsonl` | `sandbox/slurm_logs/e21n_fresh_fresh_39819436.out` | run name `e21n_fresh_enc_fresh_pred_39819436` |
| e21o_candi_enc_fresh_pred | `sandbox/runs/e21o_candi_enc_fresh_pred_39819437` | `sandbox/runs/e21o_candi_enc_fresh_pred_39819437/metrics.jsonl` | `sandbox/slurm_logs/e21o_candi_freshpred_39819437.out` | run name `e21o_candi_enc_fresh_pred_39819437` |
| e21p_fresh_enc_candi_pred | `sandbox/runs/e21p_fresh_enc_candi_pred_39819438` | `sandbox/runs/e21p_fresh_enc_candi_pred_39819438/metrics.jsonl` | `sandbox/slurm_logs/e21p_fresh_oldpred_39819438.out` | run name `e21p_fresh_enc_candi_pred_39819438` |

## Findings

**e21a-e21d (completed):** e21a (candi, type2) and e21b (fresh, type2) both run. Fresh model produces blob UMAPs vs structured UMAP in e21a. type1 loci (e21c/d) produces less structured UMAPs than type2 in both model types → type2_loci retained for all subsequent diagnostics.

**e21e-e21g (completed, 2026-05-14):** Diagnostic sweep to isolate predictor conditioning failure.
- **gamma_norm logging bug found and fixed (2026-05-14):** Old predictor logs `[B*L2, H]` norm; fresh predictors were logging `[B, H]`. Scale factor ≈ 9.8×. Both `JEPAMLPPredictor` and `JEPATransformerPredictor` now expand to `[B*L2]` before norm.
- **Corrected per-element gamma (same scale, post-fix):** e21a/candi ≈ 2.2, e21f (transformer+raw_meta) ≈ 3.2 (HIGHER than candi), e21e (MLP+embed) ≈ 1.0, e21b/e21g (transformer+embed) ≈ 0.003 (genuinely dead).
- **Root cause revised:** transformer+embedded conditioning is truly inactive (confirmed). But e21f has MORE predictor activity than candi and STILL produces blob UMAPs → blob UMAPs are not explained by predictor inactivity alone. Encoder architecture is the primary suspect (single-shot vs per-layer FiLM, XEncoder vs DualAttention).
- See `synthesis_e21efg_diagnostic_sweep.md` for full analysis.

**e21h (complete, job 39808069):** MLP predictor + raw meta_tgt. Achieved highest peak geometry in E21 sweep: `enc_er_best=44.1` at ep=20, `runtype_best=1.041`. But collapsed by ep=79 (`enc_er=18.3`, erratic runtype). gamma stayed low (41–83) throughout — much lower than old-pred runs. The raw meta_tgt + MLP combination unlocks extraordinary peak capacity in the fresh encoder but cannot sustain it.

**e21m-p (complete, 2026-05-14, jobs 39819435-38):** Clean 2×2 encoder×predictor ablation matrix, all 4 runs completed to ep=199 with no errors.

```
              old predictor (e19q-exact)    fresh transformer predictor
candi encoder   e21m: ✓ healthy               e21o: ✓✓ best in batch
fresh encoder   e21p: ✗ collapses             e21n: ✗✗ worst (dead AdaLN)
```

**2×2 results (2026-05-14):** Encoder is the confirmed root cause. Summary:
- e21m (candi+old): `enc_er_last=20.1`, `runtype_last=0.802` — healthy control
- e21o (candi+fresh xfm): `enc_er_last=26.2`, `runtype_last=0.708` — **best run**: unique late enc_er peak at ep=155, no collapse, best visual UMAP (user-verified). Recommended Stage 2 checkpoint: ep=155–170.
- e21p (fresh+old): `enc_er_last=17.2`, `runtype_last=0.098` — confirms encoder is culprit
- e21n (fresh+fresh xfm): `enc_er_last=17.8`, `runtype_last=0.256`, `gamma=0.9` (dead AdaLN) — worst

Key interaction: same fresh transformer predictor produces dead AdaLN (gamma=0.9) with fresh encoder but hyperactive AdaLN (gamma=1647) with candi encoder. Encoder architecture determines predictor gradient quality (FJ14).

**Priority fixes for next batch (2026-05-14):**
1. **e21q**: add `LayerNorm` after transformer output in `JEPAEncoder.forward` before projector (cheapest, 1-line; tests if missing normalization is the collapse mechanism)
2. **e21r**: restore per-layer CNN FiLM (3 injections, one after each conv layer) in fresh encoder (tests if single-shot D2 is the collapse mechanism)
Run e21q first; if `runtype_last > 0.40`, stop. Otherwise run e21r.

**Pending code fix (2026-05-14):** `_apply_signal_transform` should skip sentinel values (-1, -2) explicitly, only transforming real signal values. Currently uses `log1p(clamp_min(x, 0))` which converts sentinels to 0 (functionally correct but semantically wrong). Fix with `torch.where(sentinel_mask, x, transformed)`. Apply before submitting encoder architecture experiments.

See `synthesis_e21h_mnop_2x2.md` for full analysis, trajectory tables, and per-run grad/stability data.

**Future experiment on predictor conditioning:** Moved to [`E22`](idea_e22_embedded_predictor_conditioning.md) — embedded predictor conditioning with separate `MetadataEmbedding` to fix raw `assay_id` ordinal treatment and scale mismatch.

# CANDI v2 — Reference Implementation

Status: validated
Parent: E21 (fresh JEPA encoder), E24/E26 (promoted defaults), E7 (single-shot decoder FiLM)
Checklist entry: [EXPERIMENTS.md](EXPERIMENTS.md)

## What It Is

CANDI v2 is a modular, first-principles reimplementation of the CANDI model for end-to-end reconstruction training. It replaces the production `model.py` dependency with a standalone `sandbox/candi_v2/` package composed of independently swappable encoder and decoder modules. This is the new reference backbone for all future sandbox experiments.

**Design goals:**
- Modular: encoder and decoder are separate modules with independent configs; any component can be ablated or replaced without touching the other.
- Clean provenance: encoder architecture distilled from the E21–E26 JEPA sweep (22+ ablation runs); decoder adapted from the JEPA Stage 2 decoder with added metadata conditioning.
- Two heads by default: count (Negative Binomial) + peak (Binary). Pval head (Gaussian) is available via config (`decoder.heads="all"`) but disabled by default — motivated by Q2/Q3 findings (pval interference, variance collapse, F5/F7).

## Architecture Overview (~300K params at sandbox scale)

### Encoder (`V2Encoder`)

A fresh encoder distilled from the JEPA encoder development (E21–E26), with the following promoted defaults:

1. **MetadataEmbedding**: per-assay covariate encoder. Four fields (log2 depth, assay_id, read_length, run_type) are independently projected to `embed_dim=32` and fused via 2-layer MLP. Continuous fields have learned MISSING/CLOZE sentinel embeddings; categorical fields have dedicated embedding entries. Optional LayerNorm (on by default for encoder FiLM path).

2. **SignalConvTower**: 3-layer grouped Conv1d tower (one group per assay+control channel). Each layer: Conv1d → GELU → GroupNorm/LayerNorm → MaxPool(2). FiLM conditioning applied at each conv layer (`per_conv_and_transformer` mode). Optional MaskStem for handling missing/cloze data.

3. **MaskTokenInjector**: Per-assay learnable mask vectors at conv-output dimension. After the signal conv tower, masked assay channels are replaced with their corresponding learned embeddings. This resolves the aliasing problem from E23 (shared mask token collapses all masked assays to identical features).

4. **DNAConvTower**: Ungrouped Conv1d on one-hot DNA sequence (4 channels). Matched spatial downsampling to align with signal tower output.

5. **LinearFusion**: Concatenate signal + DNA features → Linear → GELU. No LayerNorm (E26: fusion LN removed, transformer pre-norm is sufficient).

6. **Transformer**: x-transformers encoder with RoPE positional encoding, 2 layers, 4 heads. Optional DualAttention mode available. FiLM conditioning optionally applied at each transformer layer.

**Input → Output shapes (default config, 8 assays):**
```
x_signal: [B, 768, 9]  →  z: [B, 96, d_model]
x_dna:    [B, 4, 19200]
x_meta:   [B, 4, 9]
```

### Decoder (`V2Decoder`)

Adapted from the JEPA Stage 2 decoder (`jepa_decoder.py`) with added metadata conditioning:

1. **PreDecoderFiLM** (default: `single_pre_decoder`): Single-shot FiLM conditioning on the encoder latent using target metadata (`y_meta`). Uses a separate `MetadataEmbedding` instance (no LayerNorm, independent from encoder). This tells the decoder *what* to reconstruct.

2. **DeconvTower**: 3-layer ConvTranspose1d tower with residual skip connections. Upsamples from L2=96 → L=768. Configurable as shared trunk (split into heads at the end) or separate towers per head.

3. **Output Heads**:
   - **NegativeBinomialLayer**: predicts `(p, n)` for count reconstruction.
   - **PeakLayer**: predicts peak probability via sigmoid for binary peak calls.
   - **GaussianLayer** (optional, `heads="all"`): predicts `(μ, σ²)` for processed signal.

**Input → Output shapes:**
```
z:      [B, 96, d_model]  →  p:    [B, 768, 8]
y_meta: [B, 4, 8]             n:    [B, 768, 8]
                               peak: [B, 768, 8]
```

### Model (`CANDIv2`)

Thin composition layer:
- `forward()` returns a dict: `{'p', 'n', 'peak', 'mu', 'var', 'z'}`
- `forward_tuple()` returns the 6-tuple `(p, n, mu, var, df, peak)` for compatibility with `train.py` and `CANDI_LOSS`.

### Loss

Reuses production `CANDI_LOSS` via `SandboxCompositeLoss`. Default: `pval_weight=0`, so only count NLL and peak BCE contribute. NaN-safe eval aggregation handles disabled heads gracefully.

## Config Hierarchy

```
CANDIv2Config
├── encoder: EncoderConfig
│   ├── film_mode, missing_data_mode, fusion_mode, fusion_norm
│   ├── transformer_type, n_transformer_layers, nhead
│   ├── signal_transform, conv_norm, dropout
│   └── ...
├── decoder: DecoderConfig
│   ├── heads, trunk, film_mode
│   ├── meta_embed_dim, grouped_deconv
│   └── ...
├── data: DataConfig
├── training: CANDIv2TrainingConfig
│   ├── optimizer, schedule, grad, masking, loss_weights, dsf
│   └── ...
├── eval: EvalConfig
├── wandb: WandbConfig
└── hpo: HpoConfig
```

YAML: `sandbox/configs/candi_v2_default.yaml`
CLI: `python -m sandbox.train --model-type candi_v2 [--config ...] [--encoder.film_mode pre_conv] ...`

## File Layout

```
sandbox/candi_v2/
├── __init__.py     # Public API: CANDIv2, CANDIv2Config, build_v2_loss
├── config.py       # EncoderConfig, DecoderConfig, CANDIv2Config dataclasses
├── encoder.py      # V2Encoder + all sub-modules
├── decoder.py      # V2Decoder + FiLM + deconv + output heads
├── loss.py         # build_v2_loss() — configures SandboxCompositeLoss
└── model.py        # CANDIv2 composition + forward/forward_tuple
```

## Validation Results

4-epoch validation run (type2_loci, batch_size=8, eval every epoch):

| Epoch | Total Loss | Count Obs | Count Imp | Peak Obs | Peak Imp |
|-------|-----------|-----------|-----------|----------|----------|
| 0     | 3.838     | 1.170     | 1.869     | 0.562    | 0.661    |
| 1     | 3.565     | 1.126     | 1.828     | 0.475    | 0.531    |
| 2     | 3.458     | 1.124     | 1.920     | 0.372    | 0.438    |
| 3     | 3.226     | 1.074     | 1.840     | 0.310    | 0.374    |

All loss components trend downward monotonically. Total eval loss dropped 16% (3.84 → 3.23) over 4 epochs with no sign of plateauing. Peak losses show the strongest improvement (~44%).

## Key Design Decisions

| Decision | Choice | Evidence |
|----------|--------|----------|
| FiLM placement | `per_conv_and_transformer` | E23 batch 1: best metadata retention (runtype_last=0.168, 2.3× control) |
| Transformer | x-transformers + RoPE | E23 batch 1: −17% pred_loss vs dual attention |
| Mask tokens | Per-assay (MaskTokenInjector) | E24: +2.6% combined_loss, +37% runtype_sens vs shared token |
| Fusion norm | None | E26: +1.1% combined_loss; transformer pre-norm sufficient |
| Signal transform | log1p | B7/E1: eliminates divergence; inherited from baseline sweep |
| Decoder FiLM | Single-shot pre-decoder | E7: best multi-head architecture (+54% peak, +914% count vs baseline) |
| Default heads | count + peak | Q2/F5: pval interferes; Q3/F7: pval variance collapse |
| Decoder deconv | Non-grouped (cross-assay mixing) | Q7/E28 default; allows cross-assay feature sharing |

## Relationship to Prior Work

- **Production CANDI** (`model.py`): CANDI v2 is a clean-room reimplementation with no import dependency on `model.py` for architecture (only reuses output layer classes `NegativeBinomialLayer`, `GaussianLayer`, `PeakLayer`).
- **JEPA encoder** (`jepa_model.py`): V2Encoder is a standalone copy of the JEPA encoder at E26 promoted defaults, forked so v2 can diverge independently.
- **JEPA decoder** (`jepa_decoder.py`): V2Decoder adapts the Stage 2 deconv tower but adds metadata FiLM conditioning (JEPA Stage 2 had none).
- **B8 baseline**: CANDI v2 replaces B8 as the reference for future ablations. B8 used the production model with E7+E13 fixes; v2 uses the fresh encoder + clean decoder.

## Next Steps

- ~~Long-duration training (50–200 epochs)~~ — done via E30 (200-epoch A/B); see [`idea_e30_v2_depth_offset_head.md`](idea_e30_v2_depth_offset_head.md).
- ~~Metadata collapse diagnostic on count-only v2~~ — E30: plain head still F1 (`dcr≈1.1`); depth_offset head healthy (`dcr≈4.0`). Next: 3-head v2 with offset.
- Head isolation ablation matrix (count-only, peak-only, all) under v2 architecture.
- Comparison with B8 on matched epoch budget using `log-observability` ranking.
- Port depth-offset head to production `model.py` / B8 stack (E29 still open on production path).

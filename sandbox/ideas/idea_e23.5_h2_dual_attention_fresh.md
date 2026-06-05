# E23.5-H2 — DualAttention in fresh encoder (imported from production CANDI)

Status: staged
Parent: E23 (all 22 runs fail v2 gate), E21 (FJ12, FJ15)
Checklist entry: [EXPERIMENTS.md](EXPERIMENTS.md)

## Problem Statement

ALL 22 E23 fresh encoder runs fail the v2 geometry gate (cov_condition_number 108-191, threshold=50). The collapse is structural and not addressable by FiLM placement, conditioning source, predictor architecture, or lambda tuning. The 2×2 ablation (FJ15) proved the encoder is the root cause: the CANDI encoder maintains geometry while the fresh encoder collapses regardless of predictor.

The primary remaining structural hypothesis is that CANDI's `DualAttentionEncoderBlock` (sequence + channel attention with relative position bias, post-norm, FFN fusion) provides collapse resistance that the fresh encoder's x-transformers or local DualAttention reimplementation cannot match. This is the most important untested hypothesis for the fresh encoder path.

## Hypothesis

Replacing the fresh encoder's transformer blocks with the **exact production DualAttentionEncoderBlock from `model.py`** (imported, not reimplemented) will prevent or substantially delay the dimensional collapse, enabling the fresh encoder to pass the v2 geometry gate for the first time.

If this works, DualAttention is the critical missing component. If it fails (fresh encoder still collapses), the collapse is upstream of the transformer (in the conv tower, fusion, or FiLM path) and the fresh encoder architecture needs fundamental redesign.

## Planned Intervention

Config deltas vs `jepa_default.yaml`:
```yaml
model_type: fresh

fresh:
  transformer_type: production_dual    # NEW: import DualAttentionEncoderBlock from model.py
  film_mode: per_conv_and_transformer  # new default
  cond_source: meta_tgt_embed          # new default: separate embed
  cond_embed_shared: separate
  meta_embed_layernorm: true
  pred_meta_embed_layernorm: false

# All other defaults from jepa_default.yaml
```

**Required code changes:**

1. Add `transformer_type="production_dual"` option to `JEPAModelConfig` and `JEPAEncoder`.

2. In `sandbox/jepa_model.py::JEPAEncoder.__init__`, when `transformer_type == "production_dual"`:
   ```python
   from model import DualAttentionEncoderBlock as ProductionDualBlock
   self.transformer_layers = nn.ModuleList([
       ProductionDualBlock(
           d_model=self.d_model,
           num_heads=int(cfg.nhead),
           seq_length=self.l2,
           dropout=float(cfg.dropout),
       )
       for _ in range(int(cfg.n_transformer_layers))
   ])
   ```

3. In `JEPAEncoder.forward`, the forward pass for `production_dual` should match the
   existing `dual` path since the production block has the same `forward(x) -> x` signature.

4. Add `"production_dual"` to the `transformer_type` Literal type.

## Verifiables

- **Primary gate:** Does a fresh encoder run with production DualAttention pass v2 geometry gate (`cov_condition_number_last < 50`, `enc_er_last >= 15`, `sigreg_converged=1`, `pred_slope <= 0`)?
- Validate if: `cov_condition_number_last < 50` (first fresh run to pass). `runtype_last > 0.30` (meaningful biology). `enc_er_last > 18`.
- Disvalidate if: `cov_condition_number_last > 100` (collapse persists despite DualAttention). This would rule out DualAttention as the explanation and point to upstream conv/FiLM issues.
- Comparison: e23a (fresh control, cov_cond=191), e23i (best fresh config, cov_cond=158), e21o (CANDI reference, cov_cond=52.9).
- Required artifacts: `resolved_config.yaml`, `metrics.jsonl`, SLURM logs, W&B UMAP.

## Risks / Watch-outs

- **Import coupling.** Importing `DualAttentionEncoderBlock` from `model.py` reintroduces the production dependency that E21 was designed to eliminate. This is acceptable for a diagnostic experiment but should not become the permanent architecture if avoidable.
- **Signature differences.** Production `DualAttentionEncoderBlock.__init__` takes `max_distance`, `pos_encoding_type`, `max_len` parameters not present in the local version. Use the default values (`pos_encoding_type="relative"`, `max_distance=128`, `max_len=5000`).
- **Helper function dependency.** Production block calls `get_divisible_heads` from `model.py`. Ensure this is also importable.
- **If the test is positive**, follow-up is to identify WHICH component of DualAttention matters (relative bias? channel attention? post-norm FFN fusion?) and implement minimally in `jepa_model.py` without the full `model.py` import.

## Run Links

- Run directory: TBD
- Resolved config: TBD
- Metrics: TBD
- SLURM logs: TBD
- W&B run: TBD

## Findings

**Superseded** by clean_ab_fresh_enc (SLURM 40548216). Clean A/B determines whether encoder redesign is still needed before DualAttention test. See [`synthesis_clean_ab_encoder.md`](synthesis_clean_ab_encoder.md).

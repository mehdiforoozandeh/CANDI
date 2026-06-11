# E34 june3 — To-explore list for next AR attempt

Written after june3 session end (2026-06-10). Based on ~85 experiments across the full AR loop.

## Current best: KEEP6 (346a0b46), primary=-0.476069

```python
cfg.encoder.n_transformer_layers = 4
cfg.encoder.nhead = 8
cfg.encoder.conv_norm = "layer"
cfg.encoder.signal_transform = "log1p"
cfg.encoder.dropout = 0.02
cfg.encoder.dna_pool_order = "early"
cfg.encoder.fusion_deep = True      # KEEP4: 2-layer LinearFusion
cfg.encoder.fusion_norm = "layer"   # KEEP5: LayerNorm after fusion
cfg.encoder.attn_qk_norm = True     # KEEP6: QK norm in transformer
cfg.decoder.trunk = "separate"
cfg.decoder.learnable_depth_center = True
cfg.decoder.learnable_depth_slope = True
cfg.decoder.conv_kernel_size = 5
cfg.decoder.meta_embed_dim = 8
cfg.decoder.norm = "rms"
cfg.decoder.dcr_penalty_weight = 1.5
```

**Score breakdown**: imp_r2=-0.096, den_r2=+0.051, count_imp_loss=1.761, count_obs_loss=0.934
**Main bottleneck**: count_imp_loss=1.761 (imputation quality). den_r2 is now positive — denoising is nearly solved.

---

## LOCKED constraints (never change these — confirmed across many experiments)

- `encoder.n_cnn_layers=3`, `decoder.n_cnn_layers=3`
- `encoder.nhead=8` (minimum; 4 causes DCR guard_fail + quality collapse)
- `encoder.n_transformer_layers=4` (3: guard_fail; 5: catastrophic den_r2)
- `encoder.conv_kernel_size=3` (k=5 encoder causes DCR guard_fail)
- `encoder.film_mode="per_conv_and_transformer"` (removing transformer FiLM is catastrophic)
- `decoder.film_mode="single_pre_decoder"` (per_deconv_layer over-conditions)
- `fusion_mode="linear"` (gated fusion always catastrophic)
- `missing_data_mode="mask_token"` (mask_stem: catastrophic)
- `decoder.grouped_dispersion=False` (always fails DCR)
- `decoder.diagonal_eta=False` (always fails DCR)
- Post-transformer ops (output_rms_norm, PTCAS, signal_tower_output_ln) are ALL TOXIC
- `decoder.pool_size=2` (hardcoded; changing causes shape crash)
- `dna_pool_size=5` (only value that produces matching sequence lengths)

---

## Priority 1 — Tier 1 config switches (highest ROI, minimal risk)

### 1.1 `transformer_layer_drop=0.05` on KEEP6 base

**Rationale**: Tried 0.1 (no_gain on KEEP3 base, -0.599), tried 0.15 (catastrophic). The KEEP6 base has fundamentally different encoder characteristics (stable fusion_norm=layer + attn_qk_norm). On this stable base, small stochastic depth (0.05) might add useful regularization without disrupting the spatial reconstruction. Unlike 0.1, a smaller value leaves 95% of training steps using the full model.

**Expected**: +0.005 to +0.015 if it helps; –0.010 if it disrupts like 0.1 did.
**Key check**: DCR should stay ≥3.0 (layer_drop slightly raises alpha; 0.1 gave DCR=3.016).

### 1.2 `encoder.ff_glu=True` (GEGLU feedforward) on KEEP6 base

**Rationale**: Tried on arcsinh+rms+separate-trunk base (1ccab69c, no_gain -0.896, den_r2→-1.15). That base had GroupNorm and very different fusion. KEEP6 has LayerNorm throughout and stable fused features — the GEGLU gating mechanism might pair better with the normalized input. GEGLU reduces FFN information loss by gating the projection.

**Expected**: +0.005 to +0.020 if the stable base allows GEGLU to work.
**Caution**: If den_r2 drops sharply, the FFN architecture is still incompatible.

### 1.3 `encoder.attn_dropout=0.1` (if xtransformers exposes it)

**Rationale**: KEEP6 uses dropout=0.02 on the signal features but never tried attention-specific dropout. Attention dropout prevents specific heads from over-fitting to particular assay co-occurrence patterns — directly relevant to imputation quality. Check if xtransformers supports per-layer attn_dropout in the config.

**Expected**: +0.005 to +0.010 on imp_r2.
**Implementation**: Check `cfg.encoder.attn_dropout` or pass via xtransformer kwargs.

### 1.4 `decoder.meta_embed_dim=6` on KEEP6 base

**Rationale**: meta_embed_dim=8 was the sweet spot discovered early (24579dc0). But the entire architecture has transformed since then — KEEP6 has fusion_deep+fusion_norm+attn_qk_norm, which changes the representation space the decoder sees. A slightly smaller embedding (6) might reduce the FiLM over-conditioning risk while maintaining meaningful depth signal.

**Expected**: Small change (±0.005). Low risk.

### 1.5 `nhead=12` (d_head=6) or `nhead=9` (d_head=8) on KEEP6 base

**Rationale**: nhead=8 (d_head=9) has been optimal throughout. But KEEP6's stable fusion features might benefit from different attention geometry. nhead=9 gives d_head=8 (power of 2, efficient). nhead=8 was tried extensively but nhead=9 was blocked by the DCR boundary in earlier sessions. On KEEP6 with penalty=1.5, DCR compliance might be maintained.

**Key risk**: nhead changes the transformer's parameter layout. nhead=9 (d_head=8) might not divide cleanly in xtransformers.

---

## Priority 2 — Tier 2 modifications (medium ROI, moderate risk)

### 2.1 Three-layer LinearFusion on KEEP6 base ⭐ HIGH PRIORITY

**Rationale**: The BIGGEST discovery of this session was the 2-layer fusion (KEEP4) + LayerNorm (KEEP5) path. A 3-layer fusion continues the same idea: `Linear(144→72)→GELU→Linear(72→72)→GELU→Linear(72→72)→GELU→LN`. The monotone improvement from 1→2 layers suggests a 3rd layer might give further gains.

**Implementation**: Add `fusion_proj3: Optional[nn.Linear]` to LinearFusion, controlled by a `fusion_depth: int = 2` config field (2 → current KEEP5/6, 3 → new variant). Initialize same as deep_proj.

**Expected**: +0.005 to +0.020 if the trend holds. Risk: diminishing returns or training instability.
**Estimated params**: +72×72=5184 extra params (~838530+5184=843714).

### 2.2 Residual connection in LinearFusion ⭐ MEDIUM PRIORITY

**Rationale**: The 2-layer fusion currently does `out = LN(GELU(Lin2(GELU(Lin1(cat)))))`. Adding a residual connection: `out = LN(GELU(Lin2(GELU(Lin1(cat)))) + proj_res(cat))` where proj_res is a shortcut Linear(144→72), might stabilize training and allow the second layer to focus on refinements rather than full reconstruction. Standard residual networks benefit from this.

**Implementation**: Add `fusion_residual: bool = False` to EncoderConfig. In LinearFusion.forward(), add `residual = self.res_proj(concat)` and sum before LayerNorm.
**Note**: Only valid when `deep=True` since a single-layer fusion already IS the residual.

### 2.3 Bottleneck projection with LayerNorm before transformer ⭐ MEDIUM PRIORITY

**Rationale**: `da2b2044` (bottleneck MLP) was tried on arcsinh+rms base and caused NaN (no normalization). KEEP6 now has fusion_norm=layer which stabilizes the fusion output. A bottleneck with LN: `fused → LN → Linear(72→36) → GELU → Linear(36→72) → +residual` could compress the representation and force the transformer to learn more general features. The residual prevents gradient issues.

**Implementation**: In V2Encoder, add `self.pre_transformer_bottleneck` (MLP with residual) after fusion, before transformer. Config flag `encoder.pre_transformer_bottleneck: bool = False`.
**Caution**: This is "between fusion and transformer" — not "post-transformer" — so it should be DCR-safe.

### 2.4 Soft gate on fusion (temperature-gated DNA mixing) ⭐ LOW-MEDIUM

**Rationale**: Instead of `fusion = cat(signal, dna)`, try learnable soft gating where the model can dynamically emphasize signal vs DNA: `gate = sigmoid(Linear(dna)) · temperature`. Different genomic regions might have different optimal signal-DNA trade-offs. Unlike the catastrophic `fusion_mode=gated`, this keeps the linear projection as the backbone and adds a mild gating bias.

**Note**: `gated` fusion was catastrophic across all bases. This is a milder version — keep the full linear fusion but modulate the DNA contribution. Risk: medium.

---

## Priority 3 — Tier 4 new loss terms (potentially high ROI for imputation)

### 3.1 Cross-assay consistency loss ⭐⭐ HIGHEST PRIORITY (novel, directly targets imp_r2)

**Rationale**: KEEP6's main bottleneck is count_imp_loss=1.761 (imputation). The transformer must predict masked assays from context. A consistency loss forces the model's imputed predictions to have the same locus-level statistics as the observed predictions:

```python
# For each position: mean of imputed predicted counts ≈ mean of observed predicted counts
obs_mean  = (mu * observed_map).sum(dim=-1) / observed_map.sum(dim=-1).clamp(min=1)
imp_mean  = (mu * masked_map).sum(dim=-1) / masked_map.sum(dim=-1).clamp(min=1)
consistency = F.mse_loss(imp_mean, obs_mean.detach())
# Add to loss: total_weighted += consistency_weight * consistency
```

This biologically captures the idea that different assays at the same genomic locus should have correlated activity levels. The imputed assays should "feel like" the observed ones.

**Implementation**: Add `consistency_weight: float = 0.0` to DecoderConfig. In loss.py `CANDI_LOSS.forward()`, compute consistency term using the NB predicted mean (`mu` tensor). Set `consistency_weight=0.1` as first test.

**Expected**: Could give +0.020 to +0.050 on primary if it effectively guides imputation. Low risk of disrupting den_r2 since the loss uses `.detach()` on obs_mean.

### 3.2 Auxiliary log1p MSE on imputed positions ⭐ MEDIUM PRIORITY

**Rationale**: The NB NLL loss for imputed positions is shaped differently from a direct regression signal. Adding a small auxiliary MSE on `log1p(predicted_mu)` vs `log1p(true_count)` for masked positions provides a smoother, more direct gradient for imputation:

```python
mask_flat = masked_map.bool()  # [B, L2*scale, A]
pred_log = torch.log1p(mu[mask_flat])
true_log = torch.log1p(y_data[mask_flat])
aux_mse = F.mse_loss(pred_log, true_log) * aux_mse_weight
```

**Implementation**: `aux_mse_imp_weight: float = 0.0` in DecoderConfig. Start with `0.05`.
**Caution**: Must NOT regularize toward biased targets. Ensure `y_data` is the raw count (not transformed) at imputed positions.

---

## Priority 4 — Tier 3 new modules (higher complexity, high potential)

### 4.1 Position-aware observed context summary for decoder FiLM ⭐⭐ NOVEL

**Rationale**: The decoder's FiLM conditioning currently uses only target metadata (depth, assay type). But the ENCODER has information about the observed assay levels at each position. A useful signal for decoder: at each genomic position, what is the mean observed assay level?

```python
# In V2Encoder.encode() or model.py forward_tuple:
obs_mask = (1 - cloze_mask) * available_mask  # [B, L2, A]
obs_context = (fused * obs_mask.sum(-1, keepdim=True)).mean(dim=-1)  # [B, L2, 1]
# Pass to decoder as additional FiLM conditioning alongside y_meta
```

This gives the decoder a per-position "baseline signal level" that is directly informative for imputation. Biologically motivated: high-signal positions should impute to higher values regardless of assay type.

**Implementation**: Tier 3/5 — requires model.py changes to pass obs_context from encoder to decoder FiLM. Moderate crash risk. Config flag: `decoder.obs_context_film: bool = False`.

### 4.2 DNA cross-attention variant (signal queries, DNA keys/values) ⭐ MEDIUM

**Rationale**: Current fusion: `cat(signal, dna) → Linear → fusion`. Alternative: let signal features attend to DNA features (cross-attention where signal is query, DNA is key/value). This allows the model to selectively read DNA features based on the current signal content — e.g., a high-ATAC signal might selectively attend to motif-specific DNA features.

From SEARCH_SPACE.md Tier 3: `SignalToDNACrossAttention(signal [B,L2,C_sig], DNA [B,L2,C_dna])`. Was never tried. `c18a0b3e` tried DNA-to-signal cross-attention and it was catastrophic (-6.31 den_r2). But signal-to-DNA is different: signal queries DNA (selective reading) vs DNA queries signal (overwriting signal with DNA).

**Caution**: High risk of DCR disruption. Try with small num_heads=1.

---

## Priority 5 — Higher-risk explorations based on broader synthesis

### 5.1 Separate imputation / denoising decoder paths (Tier 5)

**Rationale**: From SEARCH_SPACE.md Tier 5. KEEP6 has near-perfect den_r2=+0.051 but poor count_imp_loss=1.761. These two tasks may fundamentally interfere when sharing the same decoder. Two separate decoder instances (same encoder, separate decoders for masked vs observed positions) would allow each to fully optimize without interference.

**Implementation**: 
```python
self.imp_decoder = V2Decoder(cfg.decoder, ...)   # for imputed (masked) assays
self.den_decoder = V2Decoder(cfg.decoder, ...)   # for denoised (observed) assays
```
**Risk**: High — requires model.py changes, ~doubles decoder params (~1.6M total, still <5× baseline). High potential: eliminates the fundamental imp/den trade-off.

### 5.2 Variational encoder (partial CVAE, Tier 5)

**Rationale**: The encoder produces deterministic latents. If we add a small stochastic bottleneck (mean+logvar → sample z), with KL regularization, the latent space becomes smoother and more generalizable. From SEARCH_SPACE.md Tier 5. The KEEP6 base with stable fusion makes this less likely to cause NaN than before.

**Implementation**: `self.mean_proj = Linear(72,72)`, `self.log_var_proj = Linear(72,72)`. KL weight start = 1e-4.
**Risk**: Medium-high. Parameters almost unchanged (+2×72×72=10368). May disrupt both imp and den if KL is too strong.

---

## Key lessons guiding next exploration

1. **Architecture changes interact non-linearly across components.** `attn_qk_norm` was no_gain on k=5+LN+pen base but gave KEEP6 on fusion_deep+fusion_norm base. Don't write off "failed" experiments from older bases.

2. **The fusion pipeline (signal+DNA integration) has the most unexplored potential.** 2-layer fusion and LayerNorm both gave massive wins. 3-layer fusion and residual connections are natural next steps.

3. **Imputation (count_imp_loss) is now the bottleneck, not denoising.** den_r2=+0.051 is excellent. Focus new experiments on what improves imp_r2: transformer regularization, consistency losses, richer cross-assay context.

4. **DCR compliance is fragile.** penalty_weight=1.5 is finely calibrated for the KEEP6 k=3-encoder architecture. Any change that alters gradient flow to the depth slope parameter risks DCR violation. Test DCR probe after each experiment.

5. **Post-transformer ops are UNIVERSALLY TOXIC.** Confirmed across 6+ experiments. Never add layers between transformer output and decoder input.

6. **The sweet spots (empirical)**: n_layers=4, nhead=8, dropout=0.02, meta_embed_dim=8 (decoder), meta_embed_dim=32 (encoder). These have withstood the architectural overhaul.

7. **Novel paths with highest ROI**: Cross-assay consistency loss (Tier 4, directly targets imputation), 3-layer fusion (Tier 2, continues the winning trend), transformer_layer_drop=0.05 (Tier 1, small regularization, low risk).

---

## Session update (2026-06-11, continued) — KEEP9 breakthrough and next steps

**Current best: KEEP9 (166a88d0), primary=-0.447645**

KEEP9 stack: KEEP8 + decoder.norm="group" (GroupNorm in decoder deconv blocks)

**MAJOR INSIGHT**: decoder.norm="group" gave +0.026 (biggest gain since KEEP5/KEEP6 era):
- den_r2: +0.030 → +0.121 (4× denoising improvement!)
- imp_r2: -0.081 → -0.075 (slight improvement)
- Mechanism: GroupNorm provides channel-group spatial normalization that preserves spatial
  structure better than RMSNorm for the deconv architecture

**KEY LESSON**: Results from old bases do NOT transfer. decoder.norm="group" was tried
on older bases and showed no gain or hurt. On KEEP8 base with the full stack, it suddenly
gave a massive improvement. Always re-test "failed" experiments on new KEEP bases.

### All KEEP8 experiments tried → NO GAIN:
- transformer changes (sandwich_norm, use_rmsnorm, shift_tokens): ALL collapse den_r2
- fusion_depth=3: guard_fail (DCR disruption, depth-2 is absolute max)
- dropout=0.015: regression (den_r2→-0.007)
- aux_mse_obs=0.05: near-zero effect

### KEEP9 base = KEEP8 + decoder.norm=group
- decoder.norm="group" was the unlock
- Current experiment: pre_transformer_bottleneck=True (small-init LatentBottleneck)

### Next experiments on KEEP9 base (in priority order):
1. `pre_transformer_bottleneck=True` ← GUARD_FAIL (-0.613): small-init (σ=0.01) does NOT prevent DCR disruption; 100% grad clipping; GENERALIZED RULE confirmed regardless of init scale; ABANDONED
2. `decoder.conv_kernel_size=7` ← CURRENTLY RUNNING (larger spatial kernel k=5→7; GroupNorm may synergize)
3. `consistency_weight` re-test: try 0.05, 0.15 on KEEP9 base (den_r2=+0.121 changes optimal balance)
4. `encoder.dropout` re-test: 0.025, 0.03 on KEEP9 base
5. `encoder.transformer_layer_drop` re-test: was 0.05 (KEEP7), re-test 0.04, 0.07
6. `decoder.expansion_factor` re-test: currently 2, try 3 (failed on KEEP8, might work on KEEP9)
7. Previously-failed things that might work on KEEP9:
   - aux_mse_imp_weight=0.05 (near-miss on KEEP8, might KEEP on KEEP9)
   - aux_mse_obs_weight=0.05 (near-zero on KEEP8)
   - fusion_depth=3 (guard_fail on KEEP8, DCR disruption — unlikely to help)
8. Creative: decoder.norm="layer" (back to LN) to see if group was the key factor
9. Creative: decoder.norm="batch" or other norms

### New infrastructure (added this session):
- `encoder.pre_transformer_bottleneck: bool = False` in config.py
- `LatentBottleneck` class in encoder.py (small-init, near-identity start, residual)
- `decoder.aux_mse_obs_weight`: auxiliary MSE on observed positions
- `encoder.transformer_*` variants (all tried, all toxic)

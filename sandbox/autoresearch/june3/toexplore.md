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
- aux_mse_imp=0.05: improved by ~0.019 (KEEP8: -0.474→-0.455) BUT didn't beat KEEP9; retry on KEEP9

### KEEP9 base = KEEP8 + decoder.norm=group
- decoder.norm="group" gave +0.026 (largest gain in this phase)
- den_r2: +0.030 → +0.121 (4× denoising improvement)
- GENERALIZED RULE confirmed: results from old bases do NOT transfer; decoder.norm=group failed on earlier bases

### All KEEP9 base experiments tried → NO GAIN (order run):
1. `pre_transformer_bottleneck=True` ← GUARD_FAIL (-0.613): small-init (σ=0.01) still disrupts DCR; grad_clip=100%, grad_norm=17.5; GENERALIZED RULE holds regardless of init scale
2. `decoder.conv_kernel_size=7` ← NO_GAIN (-0.633): grad_norm=73.7 explosion; k=7 adds Conv1d params → Jacobian amplification → gradient explosion; k=5 LOCKED
3. `consistency_weight=0.15` ← NEAR-MISS (-0.448, -0.000091 below KEEP9); 0.10 is at flat optimum
4. `encoder.dropout=0.025` ← NO_GAIN (-0.452): den_r2→+0.166 improved but count_imp_loss→1.830 worsened; more dropout → better denoising / worse imputation; 0.02 LOCKED
5. `decoder.expansion_factor=3` ← NO_GAIN (-0.598): catastrophic quality collapse; +100k params can't converge in 10 epochs; PERMANENTLY LOCKED
6. `encoder.fusion_norm="group"` ← NO_GAIN (-0.499): den_r2→-0.013 collapsed; GroupNorm breaks cross-feature LayerNorm essential for cross-assay attention; fusion_norm=layer LOCKED
7. `encoder.transformer_attn_dropout=0.1` ← NO_GAIN (-0.480): den_r2→+0.008 collapsed; any attention modification hurts denoising; TRANSFORMER FULLY LOCKED rule confirmed again
8. `decoder.spatial_smoothness_weight=0.01` ← NEAR-MISS (-0.448, -0.000488 below KEEP9); GroupNorm already provides spatial smoothing; near-zero effect
9. `encoder.ff_glu=True` ← NO_GAIN (-0.584): CATASTROPHIC; param+84k (+84k from GEGLU doubling FFN first linear); den_r2→-0.074; same over-parameterization pattern; ff_glu ABANDONED
10. `decoder.learnable_depth_quadratic=True` ← GUARD_FAIL (-0.475): DCR=2.998<3.0; den_r2→-0.174; beta*(d-c)^2 competes with alpha gradient even starting at 0; quadratic depth ABANDONED; only depth_center+slope are safe

### NEW RULE: Depth head is extremely sensitive
Any new parameter in the depth calibration pathway (log2_mu = alpha*(d-c) + eta) fails:
- pre_transformer_bottleneck: disrupts alpha gradient path (upstream of depth head)
- learnable_depth_quadratic: competing gradient for alpha (same pathway)
- Only safe extensions: learnable_depth_center and learnable_depth_slope (direct parameterization, no competing paths)

### All KEEP9 base experiments tried → NO GAIN (complete list):
(preceding 10 in the block above, continuing:)
11. `aux_mse_imp_weight=0.05` ← NEAR-MISS (-0.448248, gap=-0.0006): near-zero effect; model is capacity-limited not gradient-limited; imputation bottleneck is architecture not gradient signal
12. `aux_mse_obs_weight=0.05` ← NEAR-MISS (-0.447810, gap=-0.000165): near-zero effect; denoising already near-optimal with GroupNorm
13. `aux_mse_obs_weight=0.02` ← NO_GAIN (-0.447799): uniform ±noise; both aux loss weights tried; auxiliary MSE on observed/imputed positions is exhausted
14. `encoder.conv_norm="group"` ← CATASTROPHIC (-0.748539): den_r2→-1.007; ENCODER LayerNorm is essential for cross-assay attention; encoder.conv_norm=layer PERMANENTLY LOCKED
15. `encoder.dropout=0.01` ← NO_GAIN (-0.480156): regression on BOTH tasks (imp_r2=-0.097, count_imp_loss=1.793); even TRAINING loss worsens; dropout=0.02 acts as data augmentation for cross-assay robustness; DROPOUT ≥ 0.02 PERMANENTLY LOCKED
16. `nhead=6` ← NO_GAIN (-0.629xxx): den_r2 regression; d_head=12 causes training instability; nhead=8 (d_head=9) LOCKED as optimal
17. `decoder.meta_embed_dim=6` ← NO_GAIN (-0.602xxx): FiLM can't encode depth signal at dim=6; meta_embed_dim=8 LOCKED
18. `encoder.nhead=9` ← GUARD_FAIL (-0.649233): +73728 unexpected params (total=912314); DCR=2.974<3.0; xtransformers allocates additional params for nhead=9 (mechanism unclear); nhead=8 PERMANENTLY LOCKED

### Next experiment QUEUED (train.py already updated, session ended):
- `encoder.signal_tower_output_ln=True` ← NEXT: LN after signal conv tower, before fusion; +144 params; no DCR risk; might stabilize signal for cleaner cross-modal fusion; different from fusion_norm=layer (which is AFTER fusion)

### Remaining experiments to try on KEEP9 base (priority order, post-session):
1. `encoder.signal_tower_output_ln=True` ← NEXT (already in train.py, fb239e0e)
2. `decoder.meta_embed_layernorm=True` — LN on FiLM meta embedding; +16 params; was no_gain on KEEP0 but not tried on KEEP9
3. `decoder.norm="weight"` — WeightNorm via nn.utils.weight_norm; normalizes WEIGHT direction not activations; 5th untested decoder norm; DCR-safe hypothesis
4. `aux_mse_imp=0.02 + aux_mse_obs=0.02` — combined at low weight; synergistic effect hypothesis; individual effects neutral but combination might compound
5. `decoder.spatial_smoothness_weight=0.05` — 5× stronger TV-L1 spatial regularization (was 0.01 near-miss; try stronger)
6. `decoder.dcr_penalty_weight=2.0` — slightly stronger DCR penalty; might refine depth calibration
7. `consistency_weight=0.08` — try slightly lower than 0.10 (since 0.15 was near-miss, 0.10 might not be exactly optimal)

### NEW RULES learned this session (june3 session 2 continued):
1. DEPTH HEAD SENSITIVITY: Any new competing parameter in depth calibration path → DCR disruption. Only direct parameterization (center/slope) is safe.
2. FUSION NORM: GroupNorm after fusion collapses den_r2 (breaks cross-feature LayerNorm for transformer). fusion_norm=layer is the only safe option.
3. TRANSFORMER IMMUTABILITY: ANY internal transformer change (attn_dropout, ff_glu, norm type, nhead, etc.) hurts den_r2 or fails DCR. The transformer configuration is completely locked.
4. PARAMETER BUDGET: Any addition >~25k params causes convergence failure in 10-epoch budget.
5. ENCODER CONV_NORM=LAYER LOCKED: encoder GroupNorm destroys cross-assay LayerNorm needed for transformer; even "encoder.conv_norm=group" (different from fusion) is CATASTROPHIC (den_r2→-1.007).
6. DROPOUT AUGMENTATION: encoder.dropout ≥ 0.02 is a hard minimum; it acts as data augmentation for cross-assay robustness, not pure regularization.
7. NHEAD=8 LOCKED: xtransformers allocates unexpected additional params for nhead=9 (73k+); nhead=8 is the only safe value.
8. AUXILIARY LOSSES NEUTRAL: aux_mse_imp and aux_mse_obs at any weight are within training noise on KEEP9; model is capacity-limited not gradient-limited.
9. META_EMBED_DIM=8 LOCKED: dim=6 can't encode depth signal (FiLM too compressed); dim=8 is minimum viable.

### New infrastructure (added this session):
- `encoder.pre_transformer_bottleneck: bool = False` in config.py
- `LatentBottleneck` class in encoder.py (small-init, near-identity start, residual)
- `decoder.aux_mse_obs_weight`: auxiliary MSE on observed positions
- `decoder.spatial_smoothness_weight`: TV-L1 penalty on log1p(mu)
- `encoder.fusion_norm`: ["layer","none","group"] option + _FusionGroupNorm wrapper

---

## SESSION 3 (2026-06-11) — single-knob search EXHAUSTED, noise floor identified

**Effective current best config = KEEP9 (decoder.norm="group"), primary≈-0.4476.**
KEEP10 (spatial_smoothness=0.05) and "KEEP11" are NOT real gains — see noise-floor finding.

### CRITICAL: measured run-to-run noise floor ≈ 0.0007
A get_config dead-code bug ran decoder.norm=group twice under the "KEEP11 layer" label, so KEEP10 and "KEEP11" were the IDENTICAL config — yet scored -0.447587 vs -0.446886 (Δ=0.0007). That Δ between identical configs = the training noise floor. Therefore:
- Any "gain" < ~0.0007 is meaningless. KEEP10 (+0.000058) and KEEP11 (+0.000701) are both noise.
- The only validated real gain since KEEP6 is KEEP9 (group norm, +0.026 ≈ 37× noise).
- A real single-knob gain now needs >~0.001 AND a confirming repeat run.
- train.py get_config was CLEANED (each field assigned once) to remove the dead duplicates.

### Session-3 experiments (ALL no_gain/guard_fail/catastrophic):
1. signal_tower_output_ln=True → no_gain (-0.484): LN on raw signal strips intensity
2. meta_embed_layernorm=True → no_gain (-0.470): LN on FiLM meta weakens depth conditioning
3. decoder.norm=weight → CATASTROPHIC (-2.171, den_r2 -6.65): WeightNorm scalar wrecks alpha
4. aux_mse_imp=0.02+aux_mse_obs=0.02 → near-miss (-0.450): den_r2+0.168 but imp_loss+0.051 (Pareto)
5. spatial_smoothness=0.05 → noise-keep (-0.447587, +0.000058)
6. consistency_weight=0.08 → no_gain (-0.467); 0.10 is curve peak
7. dcr_penalty=2.0 → no_gain (inert, DCR already in [3,5])
8. decoder.norm=layer (GENUINE) → no_gain (-0.523, den_r2 -0.152): LayerNorm FAR worse than GroupNorm
9. encoder.signal_transform=arcsinh → no_gain (-0.461): compresses signal, imp_loss+0.049
10. encoder.missing_data_mode=mask_stem → no_gain (-0.484): loses mask-token signal
11. decoder.diagonal_eta=True → guard_fail (-1.128, den_r2 -2.643): NB-head change toxic even at -56 params
12. decoder.film_mode=per_deconv_layer → no_gain (-0.788, +1008 params): over-conditions decoder
13. encoder.film_mode=post_conv → guard_fail (-0.648): transformer-stage FiLM is needed

### Generalized LOCKED rules (reinforced):
- decoder.norm=group is UNIQUELY optimal (layer/rms/batch/instance/weight all worse-to-catastrophic)
- NB/depth head (mu, eta, dispersion, alpha) is STRUCTURALLY IMMUTABLE — any change (add OR remove params, grouped OR dense) collapses den_r2. PREDICTS grouped_dispersion fails (skipped).
- Transformer internals fully locked; encoder/decoder FiLM structure locked at defaults
- den_r2 ↔ count_imp_loss is a hard Pareto frontier; no config knob escapes it

### STRATEGIC RECOMMENDATION (for the user):
Single-knob AR is at the noise floor — continuing samples noise. Real breakthroughs require relaxing FROZEN constraints (all outside AR scope):
1. **Training budget 10→20-30 epochs** (THE binding constraint — every capacity experiment fails to converge, not because it's wrong). Highest leverage.
2. **Masking/data strategy** (caps imputation ceiling from data side)
3. **Decoder redesign** that decouples imputation vs denoising capacity (escapes the Pareto frontier)

### Remaining untested non-toxic knobs (low EV, in-progress):
- encoder.fusion_residual, encoder.output_rms_norm, decoder.grouped_deconv,
  decoder.depth_slope_constrained, encoder/decoder.film_mode other variants

---

## SESSION 3+4 FINAL CONSOLIDATION (2026-06-11) — KEEP12 locked, AR concluded

### FINAL BEST: KEEP12 = encoder.output_rms_norm=True on the KEEP9 stack
Validated primary ≈ **-0.4438 avg** (two runs: -0.442593 lucky / -0.445070 confirm; real +0.0038 over KEEP9).
Full KEEP12 stack (effective, in train.py get_config):
  encoder: n_transformer_layers=4, nhead=8, conv_norm="layer", signal_transform="log1p",
           dropout=0.02, dna_pool_order="early", fusion_deep=True, fusion_norm="layer",
           attn_qk_norm=True, transformer_layer_drop=0.05, **output_rms_norm=True** (KEEP12)
  decoder: trunk="separate", learnable_depth_center=True, learnable_depth_slope=True,
           conv_kernel_size=5, meta_embed_dim=8, norm="group", dcr_penalty_weight=1.5,
           consistency_weight=0.1, spatial_smoothness_weight=0.05

### KEEP progression (real gains only, > noise floor ~0.002):
KEEP6 -0.476 → KEEP9 -0.4476 (decoder.norm=group, +0.026) → KEEP12 -0.4438 (output_rms_norm, +0.0038).
(KEEP10 spatial_smoothness +0.000058 and "KEEP11" layer were within-noise / dead-code; not real.)

### Why AR was concluded (user decision: consolidate & stop):
1. NOISE FLOOR ≈ 0.002 primary (den_r2 run-to-run variance ~0.06). Gains below this are meaningless.
2. Single-knob space exhausted: every untested knob is locked/toxic (NB head immutable, transformer internals immutable, encoder/decoder FiLM removal catastrophic, all norms mapped — group decoder + rms encoder-output are the wins).
3. Budget relaxation (authorized, run_budget.py @40ep) proved capacity WAS under-converging but capacity scaling can't beat the score: imp<->den Pareto frontier is rooted in the SHARED transformer backbone.
   - decoder capacity → den_r2 +0.256 (best ever), imputation wrecked
   - transformer capacity → imp_r2 -0.040 (best ever), denoising wrecked
4. Decoder "separate" trunks are per-HEAD not per-task → decoder redesign alone can't decouple. Only a dual-backbone (~2x params) could, and it won't converge at batch_size=4/40ep.

### The fundamental limits (need a different regime to break):
- Frozen training loss is denoising-heavy (obs_weight=3.5 >> imp_weight=0.59) while the SCORE weights imputation 0.65 > denoising 0.35 — a structural objective mismatch (loss_weights frozen).
- Shared backbone forces the imp<->den Pareto frontier.
- batch_size=4 + ~20-40 epoch budget caps the param scale that can converge.
To go further: relax loss_weights toward the score, OR larger compute/data for a dual-backbone model. Both outside the current AR scope.

### New artifact: run_budget.py (in-scope; monkeypatches prepare.EPOCHS in-process; user-authorized budget tool).

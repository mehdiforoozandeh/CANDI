# E34 june3 — Agent search space and hard guardrails

This document describes what the agent is **allowed** to change, what is **forbidden**,
and gives concrete examples at each level of abstraction.

Read this alongside `AGENT_SYSTEM_PROMPT.md` (scope rules) and `program.md` (loop procedure).

---

## Scientific soundness principle

Every run trains for **identical epochs, data, optimizer, loss weights, and eval protocol**.
The only variable is the **model** (architecture + loss objective code).

This mirrors the Karpathy autoresearch design: freeze *how* experiments run
(compute budget, metric, training protocol), liberate *what* gets tested (the model).
Breaking this principle makes runs incomparable and the search scientifically unsound.

---

## What is permanently frozen (enforced by prepare.py at runtime)

These cannot be changed even via `get_config()`. `agent_step` will refuse to train
if any of these deviate from their `ar_fixed.yaml` values:

| Setting | Frozen value | Why |
|---|---|---|
| `training.optimizer.name` | `adamax` | Same optimizer across all runs |
| `training.optimizer.adamax.lr` | `1e-3` | Same learning rate |
| `training.grad.clip_norm` | `2.0` | Same gradient clipping |
| `training.schedule.warmup_frac` | `0.1` | Same LR schedule |
| `training.loss_weights.obs_weight` | `3.5` | Same objective weighting |
| `training.loss_weights.imp_weight` | `0.59` | Same objective weighting |
| `training.loss_weights.count_weight` | `2.0` | Same objective weighting |
| `training.dsf.sampling` | `"off"` | Same data augmentation |
| `training.masking.p_full_assay` | `1.0` | Same masking protocol |
| `training.masking.p_full_loci` | `0.0` | Same masking protocol |
| `training.masking.p_chunks` | `0.0` | Same masking protocol |
| `training.batch_size` | `4` | Same compute per run |
| `training.amp` | `False` | Same precision |
| Epochs | `20` | Overridden by prepare.py |
| Train data | May31 pin manifest | Frozen by pins.py |
| Eval data | chr21, 10% | Frozen by pins.py |

Also permanently frozen (separate from config):
- **Training loop** (`train_one_epoch`) — imported from production `sandbox/train.py`
- **Eval pass** (`run_eval_pass`) — imported via frozen `eval_bridge.py`
- **DCR probe** (`prompt_sensitivity_depth_count_ratio`) — imported from production

---

## What the agent can change

Anything in `june3/candi_v2/` (encoder, decoder, model, loss, config) and
`june3/train.py` (get_config, build_model), subject to the frozen fields above.

---

## Tier 1 — Config switches in `get_config()` only

No code written. Set existing fields on the config object returned by
`_load_baseline_config()`. These change discrete modes or scalar dimensions
in the existing architecture without adding any new code.

**Encoder shape and topology:**
```python
cfg.encoder.n_transformer_layers = 3        # more transformer depth (default 2)
cfg.encoder.nhead = 8                        # more attention heads (default 4)
cfg.encoder.dropout = 0.05                   # less regularization (default 0.1)
cfg.encoder.metadata_embed_dim = 64          # wider metadata embed (default 32)
cfg.encoder.expansion_factor = 3             # wider per-assay conv channels (default 2)
cfg.encoder.n_cnn_layers = 4                 # deeper conv tower (default 3)
```

**Encoder mode switches:**
```python
cfg.encoder.fusion_mode = "gated"            # gated DNA↔signal fusion (default "linear")
cfg.encoder.fusion_norm = "layer"            # add LayerNorm after fusion (default "none")
cfg.encoder.film_mode = "per_conv"           # FiLM only in conv, not transformer
cfg.encoder.conv_norm = "group"              # GroupNorm instead of LayerNorm
cfg.encoder.dna_pool_order = "early"         # DNA downsampling early (default "late")
cfg.encoder.missing_data_mode = "mask_stem"  # alternative masking (default "mask_token")
cfg.encoder.transformer_type = "dual"        # dual-attention instead of xtransformers
cfg.encoder.meta_embed_layernorm = False     # disable meta embed LN (default True)
```

**Decoder shape and modes:**
```python
cfg.decoder.film_mode = "per_deconv_layer"   # FiLM at every deconv step (default "single_pre_decoder")
cfg.decoder.meta_embed_layernorm = True      # normalize decoder metadata embed
cfg.decoder.expansion_factor = 3             # wider decoder trunk (default 2)
cfg.decoder.norm = "rms"                     # RMSNorm in deconv blocks (default "layer")
cfg.decoder.meta_embed_dim = 64              # wider decoder metadata embed (default 32)
cfg.decoder.n_cnn_layers = 2                 # shallower decoder (default 3)
cfg.decoder.grouped_deconv = True            # per-assay grouped deconv
```

> **Remember:** after editing `get_config()`, run `scope --staged` before committing.
> DO NOT change `training.*` fields — they are validated and will abort training.

---

## Tier 2 — New logic inside existing module `forward()` or `__init__()`

Edit existing classes in `encoder.py`, `decoder.py`, or `decoder.py` to change
the computation without adding a new top-level `nn.Module`.

**Examples in encoder.py:**
```python
# Soft blending in MaskTokenInjector instead of hard replacement
alpha = torch.sigmoid(self.blend_scale)
x = x_conv * (1 - alpha * replace.float().unsqueeze(-1)) + alpha * token * replace.float().unsqueeze(-1)

# Per-assay LayerNorm after signal conv tower output
self.assay_norm = nn.LayerNorm(self.d_model // self.num_tracks)
sig = self.assay_norm(sig.view(*sig.shape[:2], self.num_tracks, -1)).view(*sig.shape)

# Learnable temperature on the DNA gate in GatedDNAFusion
self.temp = nn.Parameter(torch.ones(1))
gate = torch.sigmoid(self.gate_proj(dna) / self.temp.clamp(min=0.1))

# Bottleneck projection between conv tower and transformer
self.bottleneck = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, d))
fused = self.bottleneck(fused)
```

**Examples in decoder.py:**
```python
# Natural-log depth offset instead of log2 (alternative parameterization)
log_mu = eta + (depth_log2 - self.depth_center) * self.log2_to_loge
mu = torch.exp(log_mu.clamp(max=20))

# Per-assay grouped linear for the dispersion head
self.linear_n = nn.Conv1d(signal_dim, signal_dim, 1, groups=signal_dim)
n = F.softplus(self.linear_n(x.permute(0,2,1)).permute(0,2,1)) + self.eps
```

---

## Tier 3 — New `nn.Module` classes wired into encoder/decoder/model

Write a new class, add it to `__init__`, and call it in `forward()` / `encode()`.

**Cross-assay attention in encoder (after per-assay conv, before DNA fusion):**
```python
class CrossAssayAttention(nn.Module):
    """Reshape [B, L2, A*d] → [B*L2, A, d] → self-attention → reshape back."""
    def __init__(self, d_per_assay: int, num_assays: int, nhead: int = 4): ...

# In V2Encoder.encode(), after mask injection, before dna_tower:
sig = self.cross_assay_attn(sig, num_assays=self.num_tracks)
```

**U-Net skip connection from encoder conv to decoder deconv:**
```python
# In V2Encoder: capture intermediate conv activation
self.skip_proj = nn.Linear(mid_channels, decoder_skip_dim)
# In V2Decoder: fuse into trunk before last deconv layer
x = x + skip.permute(0, 2, 1)
```

**Bottleneck latent layer before the transformer stack:**
```python
class LatentBottleneck(nn.Module):
    def __init__(self, d_in: int, d_bottleneck: int):
        self.down = nn.Linear(d_in, d_bottleneck)
        self.up   = nn.Linear(d_bottleneck, d_in)
    def forward(self, x):
        return x + self.up(F.gelu(self.down(x)))
```

**DNA cross-attention (queries from signal, keys/values from DNA):**
```python
class SignalToDNACrossAttention(nn.Module):
    """Signal tokens [B, L2, C_sig] attend to DNA tokens [B, L2, C_dna]."""
```

---

## Tier 4 — New loss terms in loss.py

Add a new objective on top of the existing NB NLL. Hook into
`SandboxCompositeLoss.forward_with_terms()` or `CANDI_LOSS.forward()`.

**Adding new config fields for the loss weight:**
First add to `candi_v2/config.py` (in `EncoderConfig` or a new section):
```python
# In CANDIv2Config or EncoderConfig
kl_weight: float = 0.0   # 0 = off by default; agent sets e.g. 0.01 in get_config()
```

Then in `SandboxCompositeLoss._compute_terms()` or `CANDI_LOSS.forward()`:

**KL regularization on encoder latent (CVAE-style):**
```python
# Requires encoder to output (z_mean, z_log_var) — pair with Tier 5 change
kl_loss = -0.5 * (1 + z_log_var - z_mean.pow(2) - z_log_var.exp()).sum(dim=-1).mean()
terms["kl"] = kl_loss * cfg.kl_weight
terms["total_weighted"] = terms["total_weighted"] + terms["kl"]
```

**Cross-assay consistency loss:**
```python
obs_mean = (output_p * observed_map).sum(dim=1) / observed_map.sum(dim=1).clamp(min=1)
imp_mean = (output_p * masked_map).sum(dim=1)  / masked_map.sum(dim=1).clamp(min=1)
consistency = F.mse_loss(imp_mean, obs_mean.detach())
terms["consistency"] = consistency * self.consistency_weight
terms["total_weighted"] = terms["total_weighted"] + terms["consistency"]
```

**Auxiliary count MSE on the masked positions (lightweight extra signal):**
```python
masked_pred = output_p[masked_map]   # predicted p at masked positions
masked_true = y_data[masked_map]     # true counts
aux_mse = F.mse_loss(masked_pred, masked_true.log1p() / 10)
```

> NOTE: To add a new loss WEIGHT, add a new config field in `candi_v2/config.py`
> (e.g. `kl_weight: float = 0.0`) and read it in `build_v2_loss()`.
> DO NOT modify the frozen `training.loss_weights.obs/imp/count_weight` fields.

---

## Tier 5 — Coordinated architecture changes across model.py + encoder + decoder

These require touching multiple files and changing `forward_tuple()` to return or
accept new tensors. Higher crash risk; test carefully.

**Variational encoder (partial CVAE) — pairs with KL loss in Tier 4:**
```python
# encoder.py: add projection heads
self.mean_proj    = nn.Linear(self.d_model, self.d_model)
self.log_var_proj = nn.Linear(self.d_model, self.d_model)

# encoder.encode(): after transformer, before return
z_mean    = self.mean_proj(fused)
z_log_var = self.log_var_proj(fused)
z = z_mean + torch.exp(0.5 * z_log_var) * torch.randn_like(z_mean) * (1 if self.training else 0)
return z, z_mean, z_log_var   # model.py forward_tuple passes these to the loss

# model.py: forward_tuple stores mean/log_var for the loss wrapper to consume
# loss.py: build_v2_loss wraps SandboxCompositeLoss to accept and use z_mean/z_log_var
```

**Separate imputation vs. denoising decoder paths:**
```python
# model.py: two decoder instances sharing one encoder
self.imp_decoder = V2Decoder(cfg.decoder, ...)   # for cloze-masked assays
self.den_decoder = V2Decoder(cfg.decoder, ...)   # for observed assays
```

---

## Tier 6 — Forbidden changes (will be caught and aborted)

These are NEVER allowed, regardless of how they're implemented:

| What | Why |
|---|---|
| Change `training.optimizer.*` (lr, type, schedule) | Makes runs incomparable: different optimizer = different training dynamics |
| Change `training.loss_weights.*` (obs, imp, count) | Makes runs incomparable: different weighting = different objective |
| Change `training.masking.*`, `training.dsf.*`, `training.batch_size` | Makes runs incomparable: different training conditions |
| Edit `prepare.py`, `pins.py`, `eval_bridge.py`, `agent_step.py` | Frozen harness — changes affect ALL runs, not just this one |
| Edit `sandbox/train.py`, `sandbox/eval.py`, `sandbox/batch.py` | Production code — outside scope |
| Edit `ar_fixed.yaml`, `pin_manifest.json` | Frozen experiment constants |
| Change depth_center, heads mode, count_head type | Frozen architecture invariants |

`agent_step` will **refuse to run** (exit code 2) if any frozen training
config field deviates from `ar_fixed.yaml`. This is enforced at runtime by
`prepare.py`, not just documented.

---

## Decision guide: which tier to try first?

```
Start at Tier 1 — fast, low crash risk, ~12-15 experiments/hour.
↓ If Tier 1 ideas are exhausted or show diminishing returns:
Tier 2 — modify existing forward() logic. ~10-12 experiments/hour.
↓ If a specific architectural hypothesis needs a new module:
Tier 3 — add a new nn.Module. ~8-10 experiments/hour.
↓ If the training signal itself needs a new term:
Tier 4 — add a new loss objective. ~10-12 experiments/hour.
↓ If the encoder/decoder interface needs to change:
Tier 5 — multi-file coordinated change. ~5-8 experiments/hour. High crash risk.
```

After every crash: one typo-fix retry, then `git reset --hard HEAD~1` and
try a different hypothesis. Never spend more than 2 consecutive runs debugging
the same idea.

---

## Primary score reference

`primary_score = 0.45·imp_count_r2_gw + 0.25·den_count_r2_gw − 0.20·count_imp_loss − 0.10·count_obs_loss`

**Guards** (run is discarded regardless of primary if any fail):
- `depth_count_ratio ∈ [3, 5]`
- `count_imp_loss` and `count_obs_loss` are finite
- `peak_vram_mb ≤ 9500`
- `param_count ≤ 5× baseline (300696)`

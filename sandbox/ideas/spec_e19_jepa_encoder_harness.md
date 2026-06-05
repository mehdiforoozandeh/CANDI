# Spec: CANDI Encoder-Only JEPA Harness (Stage 1 for E19/E20)

**Status:** design-complete, not-yet-implemented  
**Date:** 2026-05-06  
**Covers:** E19 Stage 1 (encoder-only JEPA pretraining), forms the foundation for E18 and E20.  
**References:** [LeJEPA arXiv:2511.08544](https://arxiv.org/abs/2511.08544), [LeWM arXiv:2603.19312](https://arxiv.org/abs/2603.19312), [galilai-group/lejepa](https://github.com/galilai-group/lejepa), [lucas-maes/le-wm](https://github.com/lucas-maes/le-wm)

---

## 1. Design Decisions (grilled + locked)

| # | Question | Decision |
|---|---|---|
| Q1 | Prediction objective | **LeWM formulation**: Encoder + lightweight MLP Predictor: `MSE(pred_proj(Pred(proj_ctx)), proj_tgt)`. No stop-gradient. **Note**: LeJEPA MINIMAL.md uses a *different* loss — no predictor, just view-invariance `(proj.mean(0)-proj).square().mean()` — because all V views are symmetric random crops. We cannot use that formulation: our context (masked) and target (full) are informationally asymmetric, so we need an explicit predictor to bridge the gap. LeWM faces the same asymmetry (context frame → predict future frame) and solves it with the predictor. We follow LeWM. |
| Q2 | Projector attachment point | Bypass `latent_projection` entirely. `JEPAProjector` attaches directly onto `CANDI_DNA_Encoder` raw output. |
| Q3 | SIGReg granularity | Per spatial-position, LeWM-style: `sigreg(cat([proj_ctx, proj_tgt]).transpose(0,1))` |
| Q4 | Predictor mask conditioning | Per-position 2-layer MLP. `pred_use_mask_cond` config flag (default `true`) to ablate. |
| Q5 | Training harness | New `sandbox/train_jepa.py`. Does not modify `sandbox/train.py`. |
| Q6 | SIGReg code | Faithful to `lucas-maes/le-wm module.py SIGReg`. Two fixes vs MINIMAL.md: `device="cuda"` → `proj.device`; `num_proj` is a constructor arg (default 1024, per lewm.yaml). |
| Q7 | Target view | Default `target_dsf = "dsf1"`: `y_data + control_data`. Mode `"same"` uses `x_data + control_data` (pre-masking). Target metadata = `x_meta` unmasked. Context metadata = `x_meta_m` (masked assays have `-2` cloze token). |
| Q8 | Evaluation | Training losses + latent geometry stats (effective rank, per-dim std). All logged under `lejepa/` in W&B. |
| Q9 | Projector dimension | `proj_dim = F2` (same as encoder output = `signal_dim * expansion_factor^n_cnn_layers`). CLI-configurable. |
| Q10 | Loss formulation | LeWM: `loss = pred_loss + lambda_sigreg * sigreg_loss`. Default `lambda_sigreg = 0.1`. |
| Q11 | Config system | New `sandbox/jepa_config.py` (`JEPAConfig` + `JEPAModelConfig`). Reuses `OptimizerConfig`, `ScheduleConfig`, `GradConfig`, `WandbConfig`, `DataConfig`, `MaskingConfig`. New `sandbox/configs/jepa_default.yaml`. |
| Q12 | Masking strategy | Assay-only masking (`p_full_assay=1.0, p_full_loci=0.0, p_chunks=0.0`). Existing random-subset logic with min-1-assay-unmasked preserved. |
| Q13 | Experiment scope | E19 Stage 1 only (encoder-only). E17 (SIGReg on CANDI) and E18 (joint) deferred. |

**Iron invariant:** No stop-gradient anywhere. No teacher-student. No EMA. SIGReg is the sole anti-collapse mechanism.

---

## 2. Architecture

### 2.1 Components

```
CANDI_DNA_Encoder (shared weights for both passes)
    in:  [B, L, F+1]  signal+control  +  [B, 4, G] DNA  +  [B, meta_dim, F+1] x_meta_embed
    out: [B, L2, F2]  where L2 = L / pool^n_layers,  F2 = (F+1)*expansion^n_layers

    NOTE: latent_projection is NOT called. We bypass it entirely.

JEPAProjector (encoder projector, shared for ctx+tgt passes)
    Linear(F2, proj_hidden) → BatchNorm1d(proj_hidden) → GELU → Linear(proj_hidden, proj_dim)
    in:  [B, L2, F2]   (reshaped to [B*L2, F2], then back)
    out: [B, L2, proj_dim]

    Faithful to lucas-maes/le-wm module.py MLP class with norm_fn=BatchNorm1d.
    Actual train.py uses proj_hidden=2048; we add `proj_hidden_dim` config field.
    NOTE: LeJEPA MINIMAL.md uses a DEEPER projector: torchvision MLP(512, [2048,2048,proj_dim], BN)
    = 3-layer (2 hidden layers with BN+GELU each). We follow LeWM's 2-layer for simplicity.
    Motivation: CANDI_DNA_Encoder ends in LayerNorm → BN needed for SIGReg.

JEPAPredictor
    per-position 2-layer MLP with AdaLN-zero conditioning (faithful to LeWM Sec. 3.1)
    Architecture: Linear(proj_dim, pred_hidden) → GELU → [AdaLN-zero(mask_indicator)] → Linear(pred_hidden, proj_dim)
    AdaLN: nn.Sequential(nn.SiLU(), nn.Linear(A, 2*pred_hidden, bias=True))
           zero-initialized on the Linear layer (weight and bias), per lucas-maes/le-wm ConditionalBlock
           gamma, beta = adaLN(mask_indicator).chunk(2, dim=-1)
           h = h * (1 + gamma) + beta   ← identity at init
    if pred_use_mask_cond=False: AdaLN skipped (pure MLP)
    Applied position-wise: reshape [B, L2, proj_dim] → [B*L2, proj_dim] → MLP → [B, L2, proj_dim]
    NOTE: LeWM's actual predictor is a full Transformer (ARPredictor, 6 layers, ConditionalBlock);
          we use a per-position MLP because we have no temporal dimension — all positions are
          predicted independently given shared mask conditioning.

JEPAPredictorProjector (SEPARATE module — same architecture as JEPAProjector, different weights)
    Linear(proj_dim, pred_proj_hidden) → BatchNorm1d(pred_proj_hidden) → GELU → Linear(pred_proj_hidden, proj_dim)
    Faithful to lucas-maes/le-wm jepa.py: `self.pred_proj = pred_proj` is separate from `self.projector`.
    pred_proj output feeds into pred_loss. NOT included in SIGReg.
    in:  [B*L2, proj_dim]  (predictor output)
    out: [B*L2, proj_dim]  → reshaped to [B, L2, proj_dim]

SIGReg  (verbatim from LeJEPA MINIMAL.md, minor device fix)
    forward(proj: [N, D] or [T, N, D]) → scalar
```

### 2.2 Shapes for default sandbox config

```
signal_dim (F) = 8 assays,  control channel = 1  →  F+1 = 9
expansion_factor = 2,  n_cnn_layers = 3,  pool_size = 2
context_length (L) = 768

L2 = 768 / 2^3 = 96
F2 = 9 * 2^3 = 72        ← CANDI_DNA_Encoder output channels
proj_dim = 72             ← default (same as F2)
proj_hidden_dim = 256     ← default (~3.5× proj_dim; LeWM uses 2048 for ViT-tiny 192-dim)
pred_hidden_dim = 72      ← default (= proj_dim)
pred_proj_hidden_dim = 256← same as proj_hidden_dim
mask_indicator_dim = 8    ← A (when pred_use_mask_cond=True)
```

---

## 3. Forward Pass (pseudocode)

```python
# ── batch prep ────────────────────────────────────────────────────────────────
x_ctx  = cat([x_data_m, control_data], dim=2)   # [B, L, F+1]  masked context
meta_ctx = x_meta_m_embed                        # [B, meta_dim, F+1]  masked metadata

if target_dsf == "dsf1":
    x_tgt = cat([y_data, control_data], dim=2)   # [B, L, F+1]  DSF=1 target
else:
    x_tgt = cat([x_data, control_data], dim=2)   # [B, L, F+1]  same-DSF target

meta_tgt = x_meta_embed                          # [B, meta_dim, F+1]  unmasked metadata
mask_indicator = (x_avail_m[:, :A] == 0).float() # [B, A]

# ── encoder (shared weights, two passes) ──────────────────────────────────────
z_ctx_raw = candi_dna_encoder(x_ctx, x_dna, meta_ctx)   # [B, L2, F2]
z_tgt_raw = candi_dna_encoder(x_tgt, x_dna, meta_tgt)   # [B, L2, F2]

# ── encoder projector (shared, same as LeWM self.projector) ───────────────────
proj_ctx = jepa_projector(z_ctx_raw)   # [B, L2, proj_dim]
proj_tgt = jepa_projector(z_tgt_raw)   # [B, L2, proj_dim]

# ── predictor + SEPARATE pred_projector (same as LeWM self.pred_proj) ─────────
z_pred_raw = jepa_predictor(proj_ctx, mask_indicator)   # [B, L2, proj_dim]  (AdaLN-zero inside)
z_pred     = jepa_pred_projector(z_pred_raw)            # [B, L2, proj_dim]  separate BN projector

# ── losses ─────────────────────────────────────────────────────────────────────
pred_loss = F.mse_loss(z_pred, proj_tgt)   # NO detach — no stop-gradient
                                            # z_pred from pred_proj, proj_tgt from projector

# SIGReg: on encoder projections ONLY (NOT predictor output), all views, per-position
# Faithful to lucas-maes/le-wm train.py: sigreg(emb.transpose(0,1)) where emb = all encoder projections
proj_all    = cat([proj_ctx, proj_tgt], dim=0)    # [2B, L2, proj_dim]
sigreg_loss = sigreg(proj_all.transpose(0, 1))    # [L2, 2B, proj_dim] → scalar

loss = pred_loss + lambda_sigreg * sigreg_loss
```

---

## 4. SIGReg Implementation

Two reference implementations exist. We compose from both:
- **galilai-group/lejepa MINIMAL.md**: `num_proj=256` hardcoded, `device="cuda"` hardcoded, no parameter for num_slices.
- **lucas-maes/le-wm module.py**: `num_proj=1024` as a constructor parameter, device from tensor. Also in `lewm.yaml`: `num_proj: 1024`.

Our version: faithful to `lucas-maes/le-wm module.py` (which is itself faithful to MINIMAL.md but cleaner).
Two fixes vs MINIMAL.md: `device="cuda"` → `proj.device`; `256` → configurable `num_proj` defaulting to 1024.

```python
class SIGReg(nn.Module):
    """
    Sketched Isotropic Gaussian Regularizer (Balestriero & LeCun, 2025).
    Faithful to lucas-maes/le-wm module.py SIGReg (num_proj=1024 default per lewm.yaml).
    Symmetric ECF quadrature on [0, t_max=3] with trapezoidal weights.
    Fixes vs galilai-group/lejepa MINIMAL.md: device-agnostic; num_proj is a parameter.
    """
    def __init__(self, knots: int = 17, num_proj: int = 1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        # proj: (T, N, D)  e.g. [L2, 2B, proj_dim] for CANDI
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()
```

**Faithfulness notes:**
- `mean(-3)` averages over N (the samples dim = -2 of input proj `[T, N, D]`), yielding per-T statistics — matches both repos
- `proj.size(-2)` = N = the sample count; multiplying scales the statistic to be N-proportional — matches both repos
- Symmetric quadrature on `[0, 3]` with doubled interior weights — matches both repos (improved vs paper's `[-3, 3]`)
- 3D input `[T, N, D]`: `statistic.mean()` averages over T and proj slices — matches `lucas-maes/le-wm module.py` exactly

---

## 5. JEPAProjector Implementation (faithful to lucas-maes/le-wm module.py MLP)

Actual LeWM code (`module.py`, used in `train.py` with `norm_fn=BatchNorm1d`, `hidden_dim=2048`):
```python
# MLP: Linear(in→hidden) → BN(hidden) → GELU → Linear(hidden→out)
self.net = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    BatchNorm1d(hidden_dim),
    nn.GELU(),
    nn.Linear(hidden_dim, output_dim or input_dim),
)
```

Our adaptation (applied position-wise via [B*L2, …] reshape):
```python
class JEPAProjector(nn.Module):
    """
    2-linear-layer MLP with BatchNorm1d, faithful to lucas-maes/le-wm module.py MLP.
    Linear(in→hidden) → BN → GELU → Linear(hidden→out).
    Applied position-wise: reshape [B, L2, in_dim] → [B*L2, in_dim] → MLP → [B, L2, proj_dim].

    BatchNorm1d is required because CANDI_DNA_Encoder ends in LayerNorm (same
    motivation as LeWM: ViT ends in LayerNorm → BN projector needed for SIGReg).
    Note: LeWM uses hidden_dim=2048 regardless of encoder size; LeJEPA MINIMAL.md uses
    [2048, 2048, proj_dim] (3-layer). We use LeWM's 2-layer with a configurable hidden dim.
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, L2, in_dim]
        B, L2, D = z.shape
        out = self.proj(z.reshape(B * L2, D))   # [B*L2, out_dim]
        return out.reshape(B, L2, -1)            # [B, L2, out_dim]
```

Two separate instances are instantiated: `jepa_projector` (encoder) and `jepa_pred_projector`
(predictor output). Same architecture, separate weights — faithful to `JEPA.__init__`.

---

## 6. JEPAPredictor Implementation

AdaLN-zero faithful to `module.py ConditionalBlock.adaLN_modulation`. The predictor
output is then fed into a separate `JEPAPredictorProjector` (same MLP+BN arch as
encoder projector, different weights) before the prediction loss.

LeWM uses a full Transformer predictor (`ARPredictor`) because it needs temporal
autoregression. For CANDI, there is no temporal dimension — we predict masked positions
from context positions per-location, so a per-position MLP is appropriate.

```python
class JEPAPredictor(nn.Module):
    """
    Per-position 2-layer MLP with AdaLN-zero mask conditioning.
    AdaLN-zero: nn.Sequential(nn.SiLU(), nn.Linear(A, 2*hidden, bias=True))
    with zero-init on the Linear (weight AND bias), faithful to
    lucas-maes/le-wm module.py ConditionalBlock.adaLN_modulation.
    Output is fed through a SEPARATE JEPAProjector (pred_proj) before pred_loss.
    """
    def __init__(self, proj_dim: int, hidden_dim: int, num_assays: int, use_mask_cond: bool):
        super().__init__()
        self.use_mask_cond = use_mask_cond
        self.fc1 = nn.Linear(proj_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, proj_dim)
        if use_mask_cond:
            # SiLU → Linear, exactly as in ConditionalBlock.adaLN_modulation
            lin = nn.Linear(num_assays, 2 * hidden_dim, bias=True)
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
            self.adaLN = nn.Sequential(nn.SiLU(), lin)
        else:
            self.adaLN = None

    def forward(self, proj_ctx: torch.Tensor, mask_indicator: torch.Tensor) -> torch.Tensor:
        # proj_ctx:       [B, L2, proj_dim]
        # mask_indicator: [B, A]  float (0=available, 1=masked)
        B, L2, _ = proj_ctx.shape
        x = proj_ctx.reshape(B * L2, -1)                        # [B*L2, proj_dim]
        h = F.gelu(self.fc1(x))                                 # [B*L2, hidden_dim]
        if self.adaLN is not None:
            mask_exp = mask_indicator.unsqueeze(1).expand(-1, L2, -1).reshape(B * L2, -1)
            gamma, beta = self.adaLN(mask_exp).chunk(2, dim=-1) # each [B*L2, hidden_dim]
            h = h * (1.0 + gamma) + beta                        # AdaLN-zero: identity at init
        out = self.fc2(h)                                        # [B*L2, proj_dim]
        return out.reshape(B, L2, -1)
        # Caller: z_pred = jepa_pred_projector(jepa_predictor(proj_ctx, mask_indicator))
```

---

## 7. Files to Create

```
sandbox/
├── jepa.py                      # SIGReg, JEPAProjector, JEPAPredictor, CANDIJepa
├── jepa_config.py               # JEPAConfig, JEPAModelConfig dataclasses
├── train_jepa.py                # Training entrypoint (mirrors train.py structure)
├── configs/
│   └── jepa_default.yaml        # Default JEPA config
└── ideas/
    └── spec_e19_jepa_encoder_harness.md   # This document
```

No changes to `model.py`, `train.py`, `config.py`, `config_types.py`, `losses.py`.

---

## 8. `JEPAModelConfig` Fields

```python
@dataclass
class JEPAModelConfig:
    proj_dim: int = 0              # 0 = use F2 (encoder output dim)
    proj_hidden_dim: int = 256     # hidden dim of JEPAProjector MLP; LeWM uses 2048 for ViT-tiny
    pred_hidden_dim: int = 0       # 0 = use proj_dim
    pred_proj_hidden_dim: int = 0  # 0 = use proj_hidden_dim (pred projector mirrors encoder projector)
    pred_use_mask_cond: bool = True
    lambda_sigreg: float = 0.1    # LeWM paper says 0.1; actual lewm.yaml uses 0.09
    sigreg_num_proj: int = 1024   # matches both lewm.yaml (num_proj:1024) and paper default M=1024
                                  # MINIMAL.md hardcodes 256; paper ablation shows negligible impact
    sigreg_knots: int = 17
    target_dsf: Literal["dsf1", "same"] = "dsf1"
```

`JEPAConfig` reuses: `DataConfig`, `DsfConfig`, `MaskingConfig`, `OptimizerConfig`, `ScheduleConfig`, `GradConfig`, `WandbConfig`, `HpoConfig`, plus `JEPAModelConfig` and a `TrainingConfig` variant.

---

## 9. Evaluation / Logging

All metrics logged under the `lejepa/` W&B prefix:

| Key | What |
|---|---|
| `lejepa/pred_loss` | MSE between `z_pred` and `proj_tgt` |
| `lejepa/sigreg_loss` | SIGReg on `cat([proj_ctx, proj_tgt])` |
| `lejepa/total_loss` | `pred_loss + lambda * sigreg_loss` |
| `lejepa/latent_eff_rank` | Effective rank of `proj_tgt`: `exp(entropy(singular_values / sum))` |
| `lejepa/latent_std_mean` | Mean per-dimension std of `proj_tgt` |
| `lejepa/latent_std_min` | Min per-dimension std (collapse indicator) |
| `lejepa/latent_mean_abs` | Mean absolute value of `proj_tgt` (should be ~0) |

Geometry stats computed on a subsample of `proj_tgt` once per eval interval (no backward pass needed).

---

## 10. Data Preparation (changes to batch prep)

`prepare_jepa_batch(batch, masker, device, *, target_dsf)` returns:

```python
{
    # Context (masked) encoder inputs
    "x_ctx":       cat([x_data_m, control_data], dim=2),   # [B, L, F+1]
    "x_dna":       x_dna,                                   # shared
    "x_meta_ctx":  x_meta_m,                                # [B, 4, F+1]  masked

    # Target (full) encoder inputs
    "x_tgt":       cat([y_data or x_data, control], dim=2), # [B, L, F+1]
    "x_meta_tgt":  x_meta,                                  # [B, 4, F+1]  unmasked

    # Mask indicator for predictor conditioning
    "mask_indicator": (x_avail_m[:, :A] == 0).float(),     # [B, A]
}
```

Masker config for JEPA: `p_full_assay=1.0, p_full_loci=0.0, p_chunks=0.0`.

---

## 11. Key Risks and Watch-outs

1. **BatchNorm with small batch size**: BN stats may be noisy if `B * L2` is small. For sandbox default (B=8, L2=96), effective BN batch = 768 — fine.
2. **Two encoder passes ≈ 2× encoder compute**: may require reducing batch size or steps. Profile before running full sweeps.
3. **SIGReg gradient magnitude vs pred_loss**: log gradient norms per loss term at each training step. If SIGReg dominates, reduce `lambda_sigreg`. (LeWM paper: λ=0.1; actual lewm.yaml uses λ=0.09 — try 0.09 if 0.1 is too strong.)
4. **No stop-gradient**: if the encoder collapses (all positions → same vector), check `lejepa/latent_eff_rank` — it should increase early in training. If it drops to 1, SIGReg is not fighting collapse effectively; try increasing `lambda_sigreg`.
5. **Gradient clipping**: LeWM `lewm.yaml` uses `gradient_clip_val: 1.0`. **Must** apply grad clipping with max norm 1.0 in `train_jepa.py` — the no-stop-gradient regime without clipping can produce spike gradients through the predictor early in training.
6. **target_dsf="dsf1" with dsf_sampling="off"**: when DSF sampling is off, `y_data == x_data` so both modes are identical. Ensure this is tested to avoid accidental no-op.
7. **x_meta_m faithfulness**: the masked assays must have `-2` cloze token in `x_meta_m`, not `-1`. Verify `DataMasker.apply_mask` sets this correctly (it does, per current sandbox batch.py).
8. **Control channel**: control is always included in both `x_ctx` and `x_tgt` (never masked). The control metadata is always unmasked in both views.
9. **bf16 training**: LeWM trains in bf16 (`precision: bf16`). Use bf16 where possible — the SIGReg trig operations are numerically stable in bf16, and BN benefits from reduced memory for the large effective batch.

---

## 12. Implementation Order with Validation

Each step has inline validation gates. Run all gates before moving to the next step.
Validations marked **[code]** are pure unit checks runnable offline; **[sci]** require
a brief forward pass or training loop; **[train]** require a short training run.

---

### Step 1 — `sandbox/jepa.py`: `SIGReg`, `JEPAProjector`, `JEPAPredictor`, `CANDIJepa`

#### 1a. SIGReg

**[code]** Shape contract:
```python
reg = SIGReg(knots=17, num_proj=32)
x = torch.randn(96, 16, 64)  # [T, N, D]
out = reg(x)
assert out.shape == ()        # scalar
assert out.item() > 0
```

**[code]** Device propagation — buffers must follow input:
```python
reg = SIGReg().cuda()
assert reg.t.device.type == "cuda"
out = reg(torch.randn(4, 8, 32, device="cuda"))
assert out.device.type == "cuda"
```

**[sci]** Known-distribution calibration — SIGReg on true Gaussian should be near-zero:
```python
# with enough samples, true N(0,I) → statistic ≈ 0
x = torch.randn(96, 512, 64)   # large N
out = SIGReg(num_proj=256)(x)
assert out.item() < 0.05, f"SIGReg on N(0,I) too large: {out.item()}"
```

**[sci]** Collapse detection — constant (collapsed) input should produce a much larger loss than Gaussian:
```python
x_collapsed = torch.zeros(96, 64, 64)
x_gaussian  = torch.randn(96, 64, 64)
reg = SIGReg(num_proj=256)
assert reg(x_collapsed) > reg(x_gaussian) * 5, "SIGReg must penalize collapse strongly"
```

**[sci]** Gradient flows (no stop-gradient):
```python
x = torch.randn(4, 8, 32, requires_grad=True)
SIGReg(num_proj=16)(x).backward()
assert x.grad is not None and x.grad.abs().sum() > 0
```

#### 1b. JEPAProjector

**[code]** Shape contract:
```python
proj = JEPAProjector(in_dim=72, hidden_dim=256, out_dim=72)
z = torch.randn(8, 96, 72)       # [B, L2, F2]
out = proj(z)
assert out.shape == (8, 96, 72)
```

**[code]** BN is in the right place — verify BN trains:
```python
proj.train()
out1 = proj(torch.randn(8, 96, 72))  # BN uses batch stats
proj.eval()
out2 = proj(torch.randn(1, 96, 72))  # BN uses running stats
# no error; shapes match
```

**[sci]** BN anti-collapse property — after one forward pass in train mode the output
per-feature mean across the batch dimension should be approximately 0 and std ≈ 1
(at the hidden layer, before the second Linear):
```python
# verify BN output at hidden is ~N(0,1) across B*L2 dimension
# (done by inspecting the BN running_mean/var after a few random passes)
proj = JEPAProjector(72, 256, 72)
proj.train()
for _ in range(20):
    proj(torch.randn(8, 96, 72))
bn = proj.proj[1]   # the BatchNorm1d
assert bn.running_mean.abs().max() < 1.0    # mean should be drifting toward data mean
assert (bn.running_var - 1).abs().max() < 2.0
```

**[code]** Gradient flows:
```python
z = torch.randn(4, 8, 32, requires_grad=True)
JEPAProjector(32, 64, 32)(z).sum().backward()
assert z.grad is not None
```

#### 1c. JEPAPredictor

**[code]** Shape contract:
```python
pred = JEPAPredictor(proj_dim=72, hidden_dim=72, num_assays=8, use_mask_cond=True)
proj_ctx     = torch.randn(4, 96, 72)   # [B, L2, proj_dim]
mask_ind     = torch.zeros(4, 8)        # all available
out = pred(proj_ctx, mask_ind)
assert out.shape == (4, 96, 72)
```

**[sci]** AdaLN-zero invariant — at initialization, with any mask_indicator, the output
must equal the pure MLP path (no conditioning), because gamma=0 and beta=0:
```python
pred = JEPAPredictor(proj_dim=32, hidden_dim=32, num_assays=8, use_mask_cond=True)
pred.eval()
x = torch.randn(4, 8, 32)
mask_a = torch.zeros(4, 8)
mask_b = torch.ones(4, 8)   # all masked

# init weights of fc1/fc2 to something deterministic
torch.nn.init.normal_(pred.fc1.weight); torch.nn.init.zeros_(pred.fc1.bias)
torch.nn.init.normal_(pred.fc2.weight); torch.nn.init.zeros_(pred.fc2.bias)

with torch.no_grad():
    out_a = pred(x, mask_a)
    out_b = pred(x, mask_b)   # different mask — must give SAME output at init
assert torch.allclose(out_a, out_b, atol=1e-6), \
    "AdaLN-zero init broken: mask conditioning active before any training"
```

**[sci]** AdaLN-zero changes after one gradient step — after a backward pass the
conditioning should diverge from the identity:
```python
opt = torch.optim.SGD(pred.parameters(), lr=0.1)
loss = pred(x, mask_b).sum()
loss.backward(); opt.step()
with torch.no_grad():
    out_a2 = pred(x, mask_a)
    out_b2 = pred(x, mask_b)
assert not torch.allclose(out_a2, out_b2, atol=1e-4), \
    "AdaLN head should diverge after a gradient step"
```

**[code]** `use_mask_cond=False` path — no adaLN attribute used:
```python
pred_no_cond = JEPAPredictor(32, 32, 8, use_mask_cond=False)
assert pred_no_cond.adaLN is None
out = pred_no_cond(torch.randn(2, 4, 32), torch.zeros(2, 8))
assert out.shape == (2, 4, 32)
```

#### 1d. CANDIJepa wrapper (full forward pass)

**[sci]** Full forward, shapes and gradient flow:
```python
model = build_sandbox_candi_jepa(cfg)   # returns CANDIJepa
batch = make_dummy_jepa_batch(cfg)      # synthetic batch

output = model(batch)
assert "proj_ctx" in output
assert "proj_tgt" in output
assert "z_pred"   in output
assert output["proj_ctx"].shape == output["proj_tgt"].shape == output["z_pred"].shape

loss = output["pred_loss"] + 0.1 * output["sigreg_loss"]
loss.backward()   # must not error
# verify gradient reaches encoder weights
enc_param = next(model.encoder.parameters())
assert enc_param.grad is not None and enc_param.grad.abs().sum() > 0
```

**[sci]** No stop-gradient — `proj_tgt.grad_fn` must not be None (gradient flows
through the target encoder):
```python
assert output["proj_tgt"].grad_fn is not None, \
    "Target projection must be part of the compute graph (no stop-gradient)"
```

**[sci]** Shared encoder weights — verify the same parameter object is used for both
context and target passes (they share a module, not copies):
```python
# Both passes should modify the same grad tensor
model.zero_grad()
loss.backward()
enc_param = next(model.encoder.parameters())
# if weights were shared correctly, one backward pass accumulates grads from BOTH passes
assert enc_param.grad is not None
```

---

### Step 2 — `sandbox/jepa_config.py`: `JEPAModelConfig`, `JEPAConfig`

**[code]** Default instantiation:
```python
cfg = JEPAModelConfig()
assert cfg.lambda_sigreg == 0.1
assert cfg.sigreg_num_proj == 1024
assert cfg.proj_dim == 0           # sentinel: resolve to F2 at model build time
```

**[code]** Zero-value resolution at model build — `proj_dim=0` resolves to F2,
`pred_hidden_dim=0` resolves to proj_dim, etc. This logic lives in `CANDIJepa.__init__`:
```python
model = build_sandbox_candi_jepa(JEPAConfig())
assert model.jepa_projector.proj[0].in_features == F2
assert model.jepa_projector.proj[-1].out_features == F2   # proj_dim = F2 by default
```

**[code]** Unknown key rejection — strict dataclass behavior inherited from sandbox config:
```python
try:
    JEPAModelConfig(nonexistent_field=1)
    assert False, "Should have raised"
except TypeError:
    pass
```

**[code]** Round-trip YAML serialization — config survives `asdict → yaml.dump → yaml.load → from_dict`:
```python
import yaml, dataclasses
cfg = JEPAConfig()
d   = dataclasses.asdict(cfg)
d2  = yaml.safe_load(yaml.dump(d))
cfg2 = JEPAConfig(**d2)
assert cfg == cfg2
```

---

### Step 3 — `sandbox/configs/jepa_default.yaml`

**[code]** YAML loads without unknown keys:
```python
from sandbox.jepa_config import JEPAConfig
cfg = JEPAConfig.from_yaml("sandbox/configs/jepa_default.yaml")
assert cfg is not None
```

**[code]** CLI dotted-override works (inherits sandbox config merge logic):
```bash
python -m sandbox.train_jepa --config sandbox/configs/jepa_default.yaml \
    jepa_model.lambda_sigreg=0.05 --dry-run
# should print resolved config with lambda_sigreg=0.05 and exit
```

---

### Step 4 — `sandbox/train_jepa.py`: training loop, eval, W&B logging

#### 4a. Smoke test (2 steps, no NaN/Inf)

**[train]** Run 2 steps on the sandbox H5, assert no NaN/Inf in any loss:
```bash
python -m sandbox.train_jepa \
    --config sandbox/configs/jepa_default.yaml \
    --h5 sandbox/data/sandbox.h5 \
    train.max_steps=2 train.log_every=1 wandb.mode=disabled
```
Check: `pred_loss`, `sigreg_loss`, `total_loss` all finite and positive.

#### 4b. Gradient clipping

**[code]** Verify clipping is applied — instrument the training step to log pre-clip
grad norm and assert it is sometimes > 1.0 (so clipping is doing work) and the
post-clip norm is always ≤ 1.0 + ε:
```python
# in train_jepa.py, after loss.backward():
total_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# log: lejepa/grad_norm_pre_clip  (should sometimes be > 1.0 early in training)
# log: lejepa/grad_norm_post_clip (always ≤ 1.0)
```

#### 4c. Loss decomposition sanity

**[train]** After ~50 steps the SIGReg loss should drop sharply (latent distribution
approaches Gaussian quickly — this is the pattern reported in LeWM Fig. 18):
```
lejepa/sigreg_loss  at step 1   >> lejepa/sigreg_loss  at step 50
```
If sigreg_loss is flat or increasing, BN is not functioning or the projector is detached.

#### 4d. Anti-collapse check

**[train]** `lejepa/latent_eff_rank` must be > 1.0 by step 50. If it collapses to 1,
the SIGReg is not working:
```python
# effective rank = exp( H(σ/sum(σ)) ) where σ = singular values of proj_tgt reshaped [B*L2, D]
U, S, Vh = torch.linalg.svd(proj_tgt.reshape(-1, proj_dim), full_matrices=False)
p = S / S.sum()
eff_rank = torch.exp(-(p * p.log()).sum())
```
Expected: `eff_rank > 5` (well above 1) by step 100 for the sandbox config.

#### 4e. No stop-gradient audit

**[code]** Static check — grep the entire `train_jepa.py` and `jepa.py` for forbidden
patterns:
```bash
# Must be zero hits for all of these:
grep -n "\.detach()\|stop_grad\|torch\.no_grad\|@torch\.no_grad" \
    sandbox/train_jepa.py sandbox/jepa.py
```
Exception: `torch.no_grad()` is allowed only inside `@torch.inference_mode()` eval
blocks, never inside the training forward pass.

#### 4f. BN train/eval mode

**[sci]** Verify `model.train()` and `model.eval()` produce different outputs for the
projector (BN uses batch stats in train, running stats in eval):
```python
model.train()
out_train = model.jepa_projector(z)
model.eval()
out_eval  = model.jepa_projector(z)
assert not torch.allclose(out_train, out_eval), \
    "JEPAProjector must behave differently in train vs eval (BN)"
```

#### 4g. target_dsf="dsf1" ≠ "same" when DSF > 1

**[sci]** With a batch where the context was downsampled (DSF > 1), verify
`x_ctx != x_tgt` when `target_dsf="dsf1"`:
```python
# use a batch with DSF=4 context and DSF=1 target
assert not torch.allclose(batch["x_ctx"], batch["x_tgt"]), \
    "DSF=1 target must differ from DSF=4 context"
```

#### 4h. W&B logging completeness

**[train]** With `wandb.mode=online`, verify these keys are all present after step 1:
- `lejepa/pred_loss`, `lejepa/sigreg_loss`, `lejepa/total_loss`
- `lejepa/grad_norm_pre_clip`
- `lejepa/latent_eff_rank`, `lejepa/latent_std_min`

---

### Step 5 — SLURM submit script for first E19 run

**[code]** Static audit of the submit script before launching:
```bash
# GPU spec must be exactly this:
grep "gres" sandbox/slurm/submit_e19_jepa.sh
# expected: --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1

# Must activate both conda env and venv:
grep "conda activate\|source.*activate" sandbox/slurm/submit_e19_jepa.sh
```

**[train]** Dry-run on interactive session before full submit:
```bash
python -m sandbox.train_jepa \
    --config sandbox/configs/jepa_default.yaml \
    --h5 sandbox/data/sandbox.h5 \
    train.max_steps=10 wandb.mode=disabled
# verify: runs to completion, losses printed, no CUDA OOM
```

**[train]** First real run acceptance criteria (check after ~200 steps / ~5 min):
- `lejepa/sigreg_loss` has dropped by ≥ 50% from step-1 value
- `lejepa/pred_loss` is decreasing (not oscillating wildly)
- `lejepa/latent_eff_rank` > 3
- No NaN/Inf in any logged metric
- `lejepa/grad_norm_pre_clip` is sometimes > 1.0 (confirms clipping is active)

---

### Step 6 — Post-run documentation

After a successful run:
- Update `sandbox/ideas/idea_e19_jepa_frozen_decoder.md` with SLURM job ID, W&B run URL, and step-200 metric snapshot.
- Update `sandbox/ideas/EXPERIMENTS.md` with the new run entry.
- If `latent_eff_rank` is healthy (> 5 by step 500), flag the run as "Stage 1 passed" and open a new idea for Stage 2 (frozen-encoder decoder probing, E19 proper).

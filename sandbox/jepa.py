"""JEPA components: SIGReg, JEPAProjector, JEPAPredictor, CANDIJepa.

Faithful to:
  - galilai-group/lejepa MINIMAL.md  (SIGReg math, symmetric ECF quadrature)
  - lucas-maes/le-wm module.py        (SIGReg with device-agnostic num_proj param;
                                        MLP projector: Linear→BN→GELU→Linear;
                                        ConditionalBlock AdaLN-zero;
                                        separate encoder/pred projectors)

Design deviations from LeWM (documented in spec_e19_jepa_encoder_harness.md):
  - No temporal predictor (ARPredictor) — CANDI has no time axis; we use a
    per-position MLP predictor instead.
  - 2-layer projector (vs 3-layer in LeJEPA MINIMAL.md) — follows LeWM.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import CANDI


# ──────────────────────────────────────────────────────────────────────────────
# SIGReg
# ──────────────────────────────────────────────────────────────────────────────

class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularizer.

    Faithful to lucas-maes/le-wm module.py SIGReg (num_proj=1024 default per
    lewm.yaml).  Two fixes vs galilai-group/lejepa MINIMAL.md:
      - device-agnostic (``proj.device`` instead of hardcoded ``"cuda"``)
      - ``num_proj`` is a constructor parameter (not hardcoded 256)

    Args:
        knots:    Number of quadrature nodes on [0, 3].  Default 17.
        num_proj: Number of random projection directions M.  Default 1024
                  (matches lewm.yaml and paper default; MINIMAL.md uses 256;
                  paper ablation shows negligible impact).
    """

    def __init__(self, knots: int = 17, num_proj: int = 1024) -> None:
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3.0 / (knots - 1)
        weights = torch.full((knots,), 2.0 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(
        self, proj: torch.Tensor, return_stats: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            proj: ``(T, N, D)`` — e.g. ``[L2, 2B, proj_dim]`` for CANDI.
        Returns:
            If ``return_stats=False``: scalar loss.
            If ``return_stats=True``: ``(loss, stats)`` where stats contains
            per-projection diagnostics.
        """
        t = self.t.to(proj.dtype)           # type: ignore[attr-defined]
        phi = self.phi.to(proj.dtype)       # type: ignore[attr-defined]
        w = self.weights.to(proj.dtype)     # type: ignore[attr-defined]
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device, dtype=proj.dtype)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * t
        err = (x_t.cos().mean(-3) - phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ w) * proj.size(-2)
        loss = statistic.mean()
        if not return_stats:
            return loss
        stats = {
            "sigreg_projection_std": float(statistic.detach().float().std().item()),
        }
        return loss, stats


# ──────────────────────────────────────────────────────────────────────────────
# JEPAProjector
# ──────────────────────────────────────────────────────────────────────────────

class JEPAProjector(nn.Module):
    """2-layer MLP with BatchNorm1d, faithful to lucas-maes/le-wm module.py MLP.

    Architecture: ``Linear(in→hidden) → BN(hidden) → GELU → Linear(hidden→out)``

    Applied position-wise: ``[B, L2, in_dim] → [B*L2, in_dim] → MLP → [B, L2, out_dim]``.

    BatchNorm1d is required because CANDI_DNA_Encoder ends in LayerNorm — the same
    motivation as LeWM where ViT ends in LayerNorm and BN is needed for SIGReg to
    have an isotropic Gaussian target to regularise towards.

    Note: LeWM uses ``hidden_dim=2048`` regardless of encoder size; LeJEPA
    MINIMAL.md uses a 3-layer ``[2048, 2048, proj_dim]`` MLP.  We follow LeWM's
    simpler 2-layer with a configurable hidden dim.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, L2, D = z.shape
        return self.proj(z.reshape(B * L2, D)).reshape(B, L2, -1)


# ──────────────────────────────────────────────────────────────────────────────
# JEPAPredictor
# ──────────────────────────────────────────────────────────────────────────────

class JEPAPredictor(nn.Module):
    """Per-position 2-layer MLP with optional AdaLN-zero mask conditioning.

    AdaLN-zero: ``nn.Sequential(nn.SiLU(), nn.Linear(A, 2*hidden, bias=True))``
    with zero-initialised Linear — faithful to lucas-maes/le-wm module.py
    ``ConditionalBlock.adaLN_modulation``.

    At initialisation gamma=0 and beta=0, so conditioning has zero effect
    (identity pass-through).  The predictor gradually learns to use the mask
    signal as training progresses.

    LeWM uses a full Transformer (ARPredictor) because it needs temporal
    autoregression.  CANDI has no temporal dimension so a per-position MLP is
    appropriate.  The output is fed through a *separate* JEPAProjector (pred_proj,
    not shared weights) before the prediction loss.
    """

    def __init__(
        self,
        proj_dim: int,
        hidden_dim: int,
        mask_cond_dim: int,
        use_mask_cond: bool,
    ) -> None:
        """
        Args:
            proj_dim:      Projector output dimension.
            hidden_dim:    Hidden size of the 2-layer MLP.
            mask_cond_dim: AdaLN input size.
                           - "assay" mode:       num_assays (F)
                           - "loci" mode:        1  (per-position scalar)
                           - "meta_concat" mode: 2 * 4 * (num_assays + 1)
            use_mask_cond: If False, AdaLN is disabled (predictor is unconditional).
        """
        super().__init__()
        self.use_mask_cond = use_mask_cond
        self.fc1 = nn.Linear(proj_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, proj_dim)
        # Side-channel scalars populated during forward for external logging.
        self._last_adaLN_gamma_norm: float = 0.0
        self._last_adaLN_beta_norm: float = 0.0
        if use_mask_cond:
            # SiLU → Linear, matching ConditionalBlock.adaLN_modulation exactly.
            lin = nn.Linear(mask_cond_dim, 2 * hidden_dim, bias=True)
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
            self.adaLN: nn.Module | None = nn.Sequential(nn.SiLU(), lin)
        else:
            self.adaLN = None

    def forward(
        self,
        proj_ctx: torch.Tensor,
        mask_cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            proj_ctx:  ``[B, L2, proj_dim]``
            mask_cond: ``[B, D]``     — same conditioning broadcast to all L2 positions
                       OR ``[B, L2, D]`` — per-position conditioning (loci mode)
        Returns:
            ``[B, L2, proj_dim]``  — caller applies pred_projector before loss.
        """
        B, L2, _ = proj_ctx.shape
        x = proj_ctx.reshape(B * L2, -1)
        h = F.gelu(self.fc1(x))
        if self.adaLN is not None:
            if mask_cond.ndim == 2:
                # [B, D] → broadcast same cond to all positions → [B*L2, D]
                mask_exp = mask_cond.unsqueeze(1).expand(-1, L2, -1).reshape(B * L2, -1)
            else:
                # [B, L2, D] → per-position cond → [B*L2, D]
                mask_exp = mask_cond.reshape(B * L2, -1)
            gamma, beta = self.adaLN(mask_exp).chunk(2, dim=-1)
            h = h * (1.0 + gamma) + beta
            # Record norms for external logging (no graph attachment).
            self._last_adaLN_gamma_norm = float(gamma.detach().float().norm(p=2).item())
            self._last_adaLN_beta_norm = float(beta.detach().float().norm(p=2).item())
        return self.fc2(h).reshape(B, L2, -1)


# ──────────────────────────────────────────────────────────────────────────────
# CANDIJepa
# ──────────────────────────────────────────────────────────────────────────────

class CANDIJepa(nn.Module):
    """CANDI encoder wrapped with JEPA projector / predictor / pred-projector.

    Component inventory (faithful to lucas-maes/le-wm jepa.py):

      ``candi.encoder``         — shared weights for ctx + tgt passes
      ``jepa_projector``        — encoder output → proj space  (ctx and tgt share weights)
      ``jepa_predictor``        — MLP: proj_ctx → pred  (AdaLN-zero mask cond)
      ``jepa_pred_projector``   — pred → proj space  (SEPARATE weights from jepa_projector)
      ``sigreg``                — SIGReg on encoder projections only

    Loss (no stop-gradient anywhere):
      ``pred_loss  = MSE(pred_proj(pred(proj_ctx)), proj_tgt)``
      ``sigreg_loss = SIGReg(cat([proj_ctx, proj_tgt]).transpose(0,1))``
      ``loss = pred_loss + lambda_sigreg * sigreg_loss``
    """

    def __init__(
        self,
        candi: CANDI,
        *,
        proj_dim: int = 0,
        proj_hidden_dim: int = 256,
        pred_hidden_dim: int = 0,
        pred_proj_hidden_dim: int = 0,
        num_assays: int,
        use_mask_cond: bool = True,
        pred_mask_cond_type: str = "assay",
        lambda_sigreg: float = 0.1,
        sigreg_num_proj: int = 1024,
        sigreg_knots: int = 17,
        target_dsf: str = "dsf1",
        predictor: Optional[nn.Module] = None,
        pred_metadata_embedding: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.candi = candi
        self.lambda_sigreg = float(lambda_sigreg)
        self.target_dsf = str(target_dsf)
        self.num_assays = int(num_assays)
        self.pred_mask_cond_type = str(pred_mask_cond_type)
        self.pred_metadata_embedding = pred_metadata_embedding

        # Infer encoder output dim F2 = (signal_dim+1) * expansion_factor^n_cnn_layers
        # latent_projection is Linear(F2_raw → F2_proj, ...) — we want F2_raw (input).
        encoder_out_dim: int = int(candi.latent_projection[0].in_features)
        self.encoder_out_dim = encoder_out_dim

        # Resolve sentinel zeros
        if proj_dim <= 0:
            proj_dim = encoder_out_dim
        if pred_hidden_dim <= 0:
            pred_hidden_dim = proj_dim
        if pred_proj_hidden_dim <= 0:
            pred_proj_hidden_dim = proj_hidden_dim

        self.proj_dim = proj_dim

        # Determine AdaLN conditioning dim and effective use_mask_cond.
        # "none" mode disables AdaLN entirely (equivalent to pred_use_mask_cond=False).
        effective_use_mask_cond = use_mask_cond and (pred_mask_cond_type != "none")
        if pred_mask_cond_type == "loci":
            mask_cond_dim = 1                            # per-position scalar
        elif pred_mask_cond_type == "meta_concat":
            mask_cond_dim = 2 * 4 * (num_assays + 1)    # [ctx_meta ++ tgt_meta] flattened
        elif pred_mask_cond_type == "meta_tgt":
            mask_cond_dim = 4 * (num_assays + 1)         # target metadata only (task spec)
        else:  # "assay" or "none"
            mask_cond_dim = num_assays

        # Two separate projector instances (same architecture, different weights).
        self.jepa_projector = JEPAProjector(encoder_out_dim, proj_hidden_dim, proj_dim)
        self.jepa_pred_projector = JEPAProjector(proj_dim, pred_proj_hidden_dim, proj_dim)
        # Allow injecting an external predictor (e.g. JEPATransformerPredictor for run 2).
        # Must expose _last_adaLN_gamma_norm and _last_adaLN_beta_norm attributes.
        if predictor is not None:
            self.jepa_predictor: nn.Module = predictor
        else:
            self.jepa_predictor = JEPAPredictor(
                proj_dim, pred_hidden_dim, mask_cond_dim, effective_use_mask_cond
            )
        self.sigreg = SIGReg(knots=sigreg_knots, num_proj=sigreg_num_proj)

    def forward(
        self,
        x_ctx: torch.Tensor,   # [B, L, F+1]  masked context signal+control
        x_tgt: torch.Tensor,   # [B, L, F+1]  target signal+control
        x_dna: torch.Tensor,   # [B, 4, G]
        meta_ctx: torch.Tensor,  # [B, 4, F+1]  masked metadata
        meta_tgt: torch.Tensor,  # [B, 4, F+1]  unmasked/target metadata
        mask_cond: torch.Tensor, # shape depends on pred_mask_cond_type:
                                 #   "assay":       [B, F]    float (0=avail, 1=masked)
                                 #   "loci":        [B, L]    float (1=masked position)
                                 #   "meta_concat": [B, 8*(F+1)] float (ctx+tgt meta)
                                 #   "none":        any shape (ignored)
    ) -> Dict[str, torch.Tensor]:

        # ── two encoder passes, shared weights ─────────────────────────────
        z_ctx_raw = self.candi.encode(x_ctx, x_dna, meta_ctx)  # [B, L2, F2]
        z_tgt_raw = self.candi.encode(x_tgt, x_dna, meta_tgt)  # [B, L2, F2]

        # ── encoder projector (shared for ctx + tgt) ───────────────────────
        proj_ctx = self.jepa_projector(z_ctx_raw)   # [B, L2, proj_dim]
        proj_tgt = self.jepa_projector(z_tgt_raw)   # [B, L2, proj_dim]

        # ── route mask_cond for predictor ──────────────────────────────────
        # "loci": pool full-L binary mask to [B, L2, 1] per-position scalar.
        # Other modes: pass mask_cond as-is (2D [B, D], broadcast to all positions).
        if self.pred_mask_cond_type == "loci" and mask_cond.ndim == 2:
            L2 = proj_ctx.shape[1]
            L = mask_cond.shape[1]
            pool = L // L2
            # max-pool: any masked bin in a pool window → 1
            mc = mask_cond.view(mask_cond.shape[0], L2, pool).max(dim=-1).values  # [B, L2]
            mask_cond_pred: torch.Tensor = mc.unsqueeze(-1)  # [B, L2, 1]
        else:
            mask_cond_pred = mask_cond  # [B, D]

        # ── optional embedded conditioning for predictor ──────────────────
        if self.pred_metadata_embedding is not None and self.pred_mask_cond_type == "meta_tgt":
            mask_cond_pred = self.pred_metadata_embedding(meta_tgt.float()).reshape(meta_tgt.shape[0], -1)

        # ── predictor + separate pred_projector ────────────────────────────
        z_pred_raw = self.jepa_predictor(proj_ctx, mask_cond_pred)  # [B, L2, proj_dim]
        z_pred = self.jepa_pred_projector(z_pred_raw)               # [B, L2, proj_dim]

        # ── losses (no stop-gradient) ──────────────────────────────────────
        pred_loss = F.mse_loss(z_pred, proj_tgt)

        # SIGReg on encoder projections only (NOT predictor output).
        # Faithful to lucas-maes/le-wm train.py: sigreg(emb.transpose(0,1)).
        proj_all = torch.cat([proj_ctx, proj_tgt], dim=0)   # [2B, L2, proj_dim]
        sigreg_out = self.sigreg(
            proj_all.transpose(0, 1), return_stats=True
        )  # [L2, 2B, proj_dim] → scalar + stats
        assert isinstance(sigreg_out, tuple)
        sigreg_loss, sigreg_stats = sigreg_out

        total_loss = pred_loss + self.lambda_sigreg * sigreg_loss
        # LeJEPA Eq.10 scaling law for cross-lambda comparability.
        combined_loss_scaled = total_loss / (max(self.lambda_sigreg, 1e-8) ** 0.4)
        embedding_mean_norm = z_tgt_raw.detach().float().mean(dim=(0, 1)).norm(p=2)

        # Cosine similarity between context and target projections (mean over B×L2).
        cos_sim = F.cosine_similarity(
            proj_ctx.detach().float(), proj_tgt.detach().float(), dim=-1
        ).mean()

        return {
            "loss": total_loss,
            "pred_loss": pred_loss,
            "sigreg_loss": sigreg_loss,
            "combined_loss_scaled": combined_loss_scaled,
            "embedding_mean_norm": embedding_mean_norm,
            "proj_ctx": proj_ctx,
            "proj_tgt": proj_tgt,
            "z_tgt_raw": z_tgt_raw,   # raw encoder output (before projector)
            "z_pred": z_pred,
            "cos_sim": cos_sim,
            "adaLN_gamma_norm": self.jepa_predictor._last_adaLN_gamma_norm,
            "adaLN_beta_norm": self.jepa_predictor._last_adaLN_beta_norm,
            "sigreg_projection_std": sigreg_stats["sigreg_projection_std"],
        }


# ──────────────────────────────────────────────────────────────────────────────
# Latent geometry diagnostics
# ──────────────────────────────────────────────────────────────────────────────

# Metadata sensitivity contrasts: (metric_suffix, metadata_row, val_a, val_b)
# Row layout in sandbox: 0=depth_log2, 1=assay_idx (skipped), 2=read_length, 3=run_type
_META_CONTRASTS: List[Tuple[str, int, float, float]] = [
    ("depth",   0, 23.0,  25.0),   # log2 depth: ~8M vs ~33M reads (4× fold change)
    ("depth_wide", 0, 21.0, 25.0), # log2 depth: ~2M vs ~33M reads (16× fold change)
    ("readlen", 2, 35.0, 100.0),   # read length: short vs long reads
    ("runtype", 3,  0.0,   1.0),   # single-ended vs paired-ended
]


@torch.no_grad()
def compute_metadata_sensitivity(
    encoder: "CANDI",
    x_tgt: torch.Tensor,    # [B, L, A+1]  full-signal target (signal + control)
    x_dna: torch.Tensor,    # [B, G, 4]
    meta_tgt: torch.Tensor, # [B, 4, A+1]
) -> Dict[str, float]:
    """Measure how much encoder embeddings change when one metadata field is varied.

    For each contrast in _META_CONTRASTS we create two metadata tensors that
    differ only in the target row for the signal assay columns (0..A-1); the
    control column (index A) and all other metadata rows are identical.  We
    then encode x_tgt twice, compute per-position cosine similarity, and
    return 1 - cos_sim as a 'sensitivity' score (0 = encoder ignores field,
    positive = encoder responds to field).

    Two values are logged per contrast:
      lejepa/meta_sens_{name}      — mean sensitivity across B and L2 positions
      lejepa/meta_sens_{name}_max  — max sensitivity (most-sensitive position,
                                     averaged over batch)
    """
    x_tgt = x_tgt.float()
    x_dna = x_dna.float()
    meta_tgt = meta_tgt.float()
    A = x_tgt.shape[2] - 1  # signal assays, excluding control channel

    out: Dict[str, float] = {}
    for name, row, val_a, val_b in _META_CONTRASTS:
        meta1 = meta_tgt.clone()
        meta2 = meta_tgt.clone()
        meta1[:, row, :A] = val_a
        meta2[:, row, :A] = val_b

        z1 = encoder.encode(x_tgt, x_dna, meta1).float()  # [B, L2, D]
        z2 = encoder.encode(x_tgt, x_dna, meta2).float()  # [B, L2, D]

        # cos_sim per (batch, position): [B, L2]
        cos_sim = F.cosine_similarity(z1, z2, dim=-1)
        # Average over batch → [L2]; report mean and max sensitivity across positions
        cos_sim_per_pos = cos_sim.mean(dim=0)  # [L2]
        out[f"lejepa/meta_sens_{name}"] = float(
            (1.0 - cos_sim_per_pos.mean()).item()
        )
        out[f"lejepa/meta_sens_{name}_max"] = float(
            (1.0 - cos_sim_per_pos.min()).item()  # min cos_sim = max sensitivity
        )
    return out


def compute_latent_geometry(proj_tgt: torch.Tensor) -> Dict[str, float]:
    """Effective rank + per-dim stats on the target encoder projection.

    Logged under ``lejepa/`` W&B prefix as collapse diagnostics.
    Requires ``proj_tgt.shape[0] * proj_tgt.shape[1] >= proj_tgt.shape[2]``
    (enough samples for a meaningful SVD); returns empty dict otherwise.
    """
    z = proj_tgt.detach().float().reshape(-1, proj_tgt.shape[-1])
    N, D = z.shape
    if N < D:
        return {}
    out: Dict[str, float] = {}
    try:
        _, S, _ = torch.linalg.svd(z, full_matrices=False)
        p = S / (S.sum() + 1e-12)
        # Effective rank: exp(H(p)), where H is entropy.
        eff_rank = float(torch.exp(-(p * (p + 1e-12).log()).sum()).item())
        out["lejepa/latent_eff_rank"] = eff_rank
        out["lejepa/cov_condition_number"] = float(
            (S[0] / (S[-1] + 1e-12)).item()
        )
    except Exception:
        pass
    out["lejepa/embedding_mean_norm"] = float(z.mean(dim=0).norm(p=2).item())
    var_per_dim = z.var(dim=0, unbiased=False)
    out["lejepa/per_dim_variance_cv"] = float(
        (var_per_dim.std() / (var_per_dim.mean() + 1e-12)).item()
    )
    std_per_dim = z.std(dim=0)
    std_mean = float(std_per_dim.mean().item())
    out["lejepa/latent_std_mean"] = std_mean
    out["lejepa/latent_std_min"] = float(std_per_dim.min().item())
    out["lejepa/latent_std_max"] = float(std_per_dim.max().item())
    out["lejepa/latent_mean_abs"] = float(z.abs().mean().item())
    # Dead dimensions: dims whose std is less than 10% of the mean std.
    dead_threshold = max(std_mean * 0.1, 1e-6)
    out["lejepa/latent_n_dead"] = int((std_per_dim < dead_threshold).sum().item())
    return out

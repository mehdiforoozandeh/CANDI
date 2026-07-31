"""Model layer for the dual-conditioning testbed v2 (crux q15 / q16).

Reuses the CANDIv2 encoder (import-and-swap, no candi_v2 edits) but forks the decoder locally so that
conditioning is **per-assay on both sides** — the primary v2 fix. v1 pooled y_meta across assays
(`meta.mean(dim=1)`), which forced uniform-per-batch conditions and is the prime suspect for the v1
output-steering null (q16/h34).

Architecture (v2):
- **Encoder** = `V2Encoder` with `film_mode="per_conv"` (per-assay conv FiLM only — NO across-assay
  transformer pooling) and its 4-row `MetadataEmbedding` swapped for `DualCondMetaEmbedder` reading the
  3-row (family, param, [depth]) schema. `signal_transform="arcsinh"`. Depth-aware/naive is an arm
  (h30 ablation): depth-aware feeds base log2_depth (row 2) to the encoder embedder; depth-naive drops it.
- **Decoder** = `DualCondDecoder`: a per-assay feature tensor (trunk emits A*C channels), a **per-assay
  FiLM** conditioned on each assay's own y_meta rows 0-1 (adaLN-zero init), a weight-shared per-assay
  head, then a **depth-offset log-link NB head** keyed to y_meta row 2 (with an offset-off ablation).
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from sandbox.batch import CLOZE, MISSING
from sandbox.candi_v2.config import CANDIv2Config, EncoderConfig
from sandbox.candi_v2.encoder import V2Encoder
from sandbox.candi_v2.decoder import DecoderTrunk
from sandbox.diagnostics.dual_conditioning import transforms as T


# ---------------------------------------------------------------------------
# 3-row metadata embedder (steering rows 0-1 + optional non-steerable depth row 2)
# ---------------------------------------------------------------------------

class DualCondMetaEmbedder(nn.Module):
    """(aug_family, aug_param, [log2_depth]) -> [B, F, embed_dim].

    norm in {'none','zscore','log'} (h33) normalizes the raw param INSIDE the module, so the stored
    param stays raw-positive and never collides with the -1 availability sentinel. `depth_aware` adds
    the base log2_depth (row 2) as an extra covariate (encoder size-factor arm); the decoder FiLM
    embedder is always depth-naive so all steering flows through rows 0-1.
    """

    def __init__(self, embed_dim: int, norm: str = "zscore", *, depth_aware: bool = False,
                 use_layernorm: bool = True, eps: float = 1e-6):
        super().__init__()
        assert norm in ("none", "zscore", "log")
        self.norm = norm
        self.depth_aware = bool(depth_aware)
        self.eps = float(eps)
        self.family_embedding = nn.Embedding(T.N_FAMILIES + 1, embed_dim)   # +1 = missing
        self.param_proj = nn.Linear(1, embed_dim)
        self.param_missing_emb = nn.Parameter(torch.randn(embed_dim) * 0.02)
        stats = torch.tensor(T.family_param_stats(), dtype=torch.float32)   # [N_FAM, 2]
        self.register_buffer("fam_mean", stats[:, 0])
        self.register_buffer("fam_std", stats[:, 1])
        n_fields = 2
        if self.depth_aware:
            self.depth_proj = nn.Linear(1, embed_dim)
            self.depth_missing_emb = nn.Parameter(torch.randn(embed_dim) * 0.02)
            n_fields = 3
        layers = [nn.Linear(n_fields * embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim)]
        if use_layernorm:
            layers.append(nn.LayerNorm(embed_dim))
        self.fusion = nn.Sequential(*layers)

    def forward(self, metadata: torch.Tensor) -> torch.Tensor:
        fam_raw = metadata[:, 0, :].long()                     # [B, F] in {-1..6}
        param = metadata[:, 1, :].float()                      # raw positive, or -1 for missing
        missing = fam_raw < 0
        fam_idx = torch.where(missing, torch.full_like(fam_raw, T.N_FAMILIES), fam_raw)
        fam_emb = self.family_embedding(fam_idx)               # [B, F, d]

        fam_c = fam_raw.clamp(0, T.N_FAMILIES - 1)
        if self.norm == "zscore":
            x = (param - self.fam_mean[fam_c]) / self.fam_std[fam_c]
        elif self.norm == "log":
            x = torch.log(param.clamp_min(self.eps))
        else:
            x = param
        x = torch.where(missing, torch.zeros_like(x), x)
        par_emb = self.param_proj(x.unsqueeze(-1))             # [B, F, d]
        par_emb = torch.where(missing.unsqueeze(-1), self.param_missing_emb.to(par_emb.dtype), par_emb)

        fields = [fam_emb, par_emb]
        if self.depth_aware:
            depth = metadata[:, 2, :].float()                  # [B, F]; base log2_depth or -1
            dmiss = (depth == MISSING) | (depth == CLOZE)
            d_in = torch.where(dmiss, torch.zeros_like(depth), depth)
            dep_emb = self.depth_proj(d_in.unsqueeze(-1))
            dep_emb = torch.where(dmiss.unsqueeze(-1), self.depth_missing_emb.to(dep_emb.dtype), dep_emb)
            fields.append(dep_emb)

        return self.fusion(torch.cat(fields, dim=-1))


# ---------------------------------------------------------------------------
# Per-assay decoder (no across-assay pooling) + depth-offset log-link NB head
# ---------------------------------------------------------------------------

class DualCondDecoder(nn.Module):
    """Per-assay conditioned decoder with a depth-offset, log-linked NB head.

    Flow:  z [B,L2,d] --trunk--> feat [B,L,A*C] --view--> [B,L,A,C]
           per-assay FiLM (gamma_a, beta_a from y_meta[:,0:2,a]; adaLN-zero) modulates each assay's C
           features; a weight-shared head maps C -> (eta_a, raw_n_a).
           Depth-offset log-link (y_meta row 2 = base log2_depth):
             log2_mu = (d - depth_center) + eta   [offset ON, valid depth]
             log2_mu = eta                        [offset OFF, or sentinel depth]
             mu = 2^clamp(log2_mu); n = softplus(raw_n)+eps; p = n/(n+mu).
    Returns p, n, eta, log2_mu, mu (eta is the offset-free mean statistic for M2).
    """

    def __init__(self, *, encoder_d_model: int, num_assays: int, feat_per_assay: int = 16,
                 meta_embed_dim: int = 32, norm: str = "zscore", use_offset: bool = True,
                 pool_meta: bool = False, depth_center: float = 25.1, mu_eps: float = 1e-6,
                 log2_mu_clamp: tuple = (-15.0, 30.0),
                 n_cnn_layers: int = 3, expansion_factor: int = 2, pool_size: int = 2,
                 conv_kernel_size: int = 3, trunk_norm: str = "layer"):
        super().__init__()
        self.A = int(num_assays)
        self.C = int(feat_per_assay)
        self.use_offset = bool(use_offset)
        # pool_meta=True reproduces the v1 ACROSS-ASSAY POOLING regime (a single global FiLM from the
        # mean of the assay-metadata embeddings), the h34 baseline. False = the v2 per-assay fix.
        self.pool_meta = bool(pool_meta)
        self.depth_center = float(depth_center)
        self.eps = float(mu_eps)
        self.clamp_lo, self.clamp_hi = float(log2_mu_clamp[0]), float(log2_mu_clamp[1])

        signal_dim = self.A * self.C
        dec_input_dim = signal_dim * (int(expansion_factor) ** int(n_cnn_layers))
        self.trunk = DecoderTrunk(
            proj_dim=int(encoder_d_model), signal_dim=signal_dim,
            decoder_input_dim=dec_input_dim, n_cnn_layers=int(n_cnn_layers),
            expansion_factor=int(expansion_factor), pool_size=int(pool_size),
            conv_kernel_size=int(conv_kernel_size), grouped=False, norm=str(trunk_norm),
        )
        self.meta_embedding = DualCondMetaEmbedder(meta_embed_dim, norm, depth_aware=False)
        # per-assay FiLM projection (adaLN-zero: starts at identity so recon is stable, steering grows)
        self.film_proj = nn.Linear(meta_embed_dim, 2 * self.C)
        nn.init.zeros_(self.film_proj.weight)
        nn.init.zeros_(self.film_proj.bias)
        # weight-shared per-assay heads (applied on the last dim of [B,L,A,C])
        self.head_eta = nn.Sequential(nn.Linear(self.C, self.C), nn.GELU(), nn.Linear(self.C, 1))
        self.head_n = nn.Sequential(nn.Linear(self.C, self.C), nn.GELU(), nn.Linear(self.C, 1))

    def forward(self, z: torch.Tensor, y_meta: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.trunk(z)                                   # [B, L, A*C]
        B, Lq, _ = feat.shape
        feat = feat.view(B, Lq, self.A, self.C)                # [B, L, A, C]

        memb = self.meta_embedding(y_meta.float())             # [B, A, E] (rows 0-1)
        if self.pool_meta:
            memb = memb.mean(dim=1, keepdim=True).expand(-1, self.A, -1)   # v1 across-assay pooling
        gamma, beta = self.film_proj(memb).chunk(2, dim=-1)    # [B, A, C] each
        feat = feat * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)   # per-assay FiLM, broadcast over L

        eta = self.head_eta(feat).squeeze(-1)                  # [B, L, A]
        raw_n = self.head_n(feat).squeeze(-1)                  # [B, L, A]

        depth = y_meta[:, 2, :]                                # [B, A]
        valid = (depth != MISSING) & (depth != CLOZE)          # [B, A]
        if self.use_offset:
            d_off = (depth - self.depth_center).unsqueeze(1)   # [B, 1, A]
            log2_mu = torch.where(valid.unsqueeze(1), d_off + eta, eta)
        else:
            log2_mu = eta
        log2_mu = log2_mu.clamp(self.clamp_lo, self.clamp_hi)
        mu = torch.pow(2.0, log2_mu).clamp_min(self.eps)
        n = F.softplus(raw_n) + self.eps
        p = (n / (n + mu)).clamp(self.eps, 1.0 - self.eps)
        return dict(p=p, n=n, eta=eta, log2_mu=log2_mu, mu=mu)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class DualCondModel(nn.Module):
    """V2Encoder (per-assay, arcsinh, meta-swapped) + DualCondDecoder (per-assay, depth-offset head)."""

    def __init__(self, *, norm: str = "zscore", encoder_depth_aware: bool = False, use_offset: bool = True,
                 pool_meta: bool = False, embed_dim: int = 32, dropout: float = 0.1,
                 n_transformer_layers: int = 2, feat_per_assay: int = 16, depth_center: float = 25.1):
        super().__init__()
        enc = EncoderConfig(
            num_assays=8, context_length=768, metadata_embed_dim=embed_dim,
            signal_transform="arcsinh", missing_data_mode="mask_token",
            dropout=dropout, n_transformer_layers=n_transformer_layers,
            film_mode="per_conv",   # per-assay conv FiLM only; NO across-assay transformer pooling
        )
        self.encoder = V2Encoder(enc)
        # swap the 4-row MetadataEmbedding for the 3-row dual-cond embedder (import-and-swap)
        self.encoder.metadata_embedding = DualCondMetaEmbedder(
            embed_dim, norm, depth_aware=encoder_depth_aware,
            use_layernorm=bool(enc.meta_embed_layernorm))
        self.decoder = DualCondDecoder(
            encoder_d_model=self.encoder.d_model, num_assays=8, feat_per_assay=feat_per_assay,
            meta_embed_dim=embed_dim, norm=norm, use_offset=use_offset, pool_meta=pool_meta,
            depth_center=depth_center)

    def forward(self, x_data, x_dna, x_meta, y_meta) -> Dict[str, torch.Tensor]:
        z = self.encoder.encode(x_data, x_dna, x_meta, return_meta=False)
        return self.decoder(z, y_meta)

    def encode(self, x_data, x_dna, x_meta) -> torch.Tensor:
        return self.encoder.encode(x_data, x_dna, x_meta, return_meta=False)


def build_model(norm: str = "zscore", *, encoder_depth_aware: bool = False, use_offset: bool = True,
                pool_meta: bool = False, embed_dim: int = 32, dropout: float = 0.1,
                n_transformer_layers: int = 2, feat_per_assay: int = 16,
                depth_center: float = 25.1) -> DualCondModel:
    return DualCondModel(norm=norm, encoder_depth_aware=encoder_depth_aware, use_offset=use_offset,
                         pool_meta=pool_meta, embed_dim=embed_dim, dropout=dropout,
                         n_transformer_layers=n_transformer_layers,
                         feat_per_assay=feat_per_assay, depth_center=depth_center)


# ---- NB helpers (log-link head: p = n/(n+mu), mean = mu) ----
def forward_counts(model: DualCondModel, batch: dict):
    """Run the model -> (p, n) NB params, [B, L, A]."""
    out = model(batch["x_data"], batch["x_dna"], batch["x_meta"], batch["y_meta"])
    return out["p"], out["n"]


def forward_full(model: DualCondModel, batch: dict) -> Dict[str, torch.Tensor]:
    """Full head output dict (p, n, eta, log2_mu, mu) — used by the distributional M2 readout."""
    return model(batch["x_data"], batch["x_dna"], batch["x_meta"], batch["y_meta"])


def nb_mean(p: torch.Tensor, n: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return n * (1.0 - p) / (p + eps)


def nb_nll(p: torch.Tensor, n: torch.Tensor, target: torch.Tensor, avail: torch.Tensor,
           eps: float = 1e-6) -> torch.Tensor:
    """Masked NB negative log-likelihood over available assays. target = integer counts [B,L,A]."""
    probs = (1.0 - p).clamp(eps, 1.0 - eps)
    total = n.clamp_min(eps)
    dist = torch.distributions.NegativeBinomial(total_count=total, probs=probs)
    ll = dist.log_prob(target.clamp_min(0.0))               # [B, L, A]
    m = avail.unsqueeze(1).expand_as(ll)                    # [B, L, A]
    denom = m.sum().clamp_min(1.0)
    return -(ll * m).sum() / denom


def encode_latent(model: DualCondModel, batch: dict) -> torch.Tensor:
    """Encoder latent z [B, L2, d_model] (for M3); depends only on x_data/x_dna/x_meta."""
    return model.encode(batch["x_data"], batch["x_dna"], batch["x_meta"])

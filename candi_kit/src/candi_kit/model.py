#############################################################################################
#  FROZEN CONSTRUCTION ORDER — DO NOT REORDER ANYTHING IN THIS FILE.
#
#  `RealDualCondModel.__init__` and `DualCondDecoder.__init__` both draw from the GLOBAL torch
#  RNG. Every historical q19 checkpoint was produced under the exact draw sequence below, and
#  `candi_kit.compat` proves the kit reproduces it bit-exactly. Any of the following silently
#  invalidates --compat-q19 and every recorded anchor:
#    * moving `V2Encoder(enc)` after the `self.encoder.metadata_embedding = ...` replacement;
#    * skipping the `_LegacyPlaceholderMetaEmbedder` the decoder builds and then discards
#      (decoder state hash b3fe340929d15aeb -> 39682aaea5552718);
#    * scaling that stub's embedding table with `num_assays` (it is FIXED at 8 x embed_dim);
#    * creating `film_proj` as a zero tensor instead of constructing-then-zeroing it;
#    * reordering trunk -> placeholder -> film_proj -> head_eta -> head_n;
#    * an x-transformers version other than the pinned 2.11.23 (its per-module init order is
#      part of the stream).
#############################################################################################
"""Model layer for the q19 dual-conditioning recipe: V2Encoder + per-assay depth-offset NB decoder.

Provenance (vendored copies, see .BUILD_PLAN.md FILE MANIFEST):
  * `DualCondDecoder.__init__`  <- sandbox/diagnostics/dual_conditioning/model.py:115-147
  * `DualCondDecoder.forward`   <- sandbox/diagnostics/dual_conditioning_real/model_real.py:110-137
  * `RealDualCondModel`         <- sandbox/diagnostics/dual_conditioning_real/model_real.py:144-178
  * `build_real_model`          <- sandbox/diagnostics/dual_conditioning_real/model_real.py:181-187
  * NB helpers                  <- sandbox/diagnostics/dual_conditioning/model.py:223-252

Architecture. The model CORE is the golden-reference testbed architecture; the only change is the
metadata assembly — the synthetic 2/3-row knob is swapped for the REAL 4-row covariate schema
`[depth_log2, assay_id, read_length, run_type]`.

1. `RealMetaEmbedder` is the production candi_v2 `MetadataEmbedding`: depth(row0)->Linear,
   assay_id(row1)->Embedding(num_assays+3), read_length(row2)->Linear,
   run_type(row3)->Embedding(num_runtypes+2), per-field {-1 missing, -2 cloze} sentinels,
   fuse->LayerNorm; `[B,4,F] -> [B,F,E]`. It is used on BOTH the encoder side (F = num_assays+1,
   incl. the control column) and — the load-bearing change — the DECODER side, which in the testbed
   was the depth-naive 3-row `DualCondMetaEmbedder`. q19/h41 requires depth to be a decoder-FiLM
   covariate so the offset-independent `eta` can carry a learned depth response.
2. The depth-offset log-link reads depth from `y_meta[:, depth_row, :]` with `depth_row=0` (the
   testbed used synthetic row 2).

Everything else — per-assay adaLN-zero FiLM, weight-shared per-assay NB head, offset arithmetic
(`log2_mu=(d-center)+eta` when offset on & depth valid else `eta`; `mu=2^clamp`; `n=softplus+eps`;
`p=n/(n+mu)`), encoder `film_mode="per_conv"`, `signal_transform="arcsinh"` — is inherited unchanged.
NB COUNTS ONLY (no Gaussian signal head, no Bernoulli peak head).
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from candi_kit._vendored import CLOZE, MISSING
from candi_kit.config import EncoderConfig
from candi_kit.encoder import MetadataEmbedding, V2Encoder
from candi_kit.decoder import DecoderTrunk

__all__ = [
    "RealMetaEmbedder",
    "DualCondDecoder",
    "RealDualCondModel",
    "build_real_model",
    # model-agnostic testbed primitives
    "forward_full",
    "forward_counts",
    "nb_mean",
    "nb_nll",
    "encode_latent",
]


# ---------------------------------------------------------------------------
# RNG-equivalent stub for the discarded testbed metadata embedder
# ---------------------------------------------------------------------------

class _LegacyPlaceholderMetaEmbedder(nn.Module):
    """FROZEN: RNG-equivalent stub for the testbed DualCondMetaEmbedder that DualCondDecoder built
    and then discarded. It is never used in forward — it exists only so the global torch RNG stream
    matches the historical construction. Size is FIXED at 8 x E; do NOT scale it with num_assays."""

    def __init__(self, embed_dim):
        super().__init__()
        self.family_embedding = nn.Embedding(8, embed_dim)
        self.param_proj = nn.Linear(1, embed_dim)
        self.param_missing_emb = nn.Parameter(torch.randn(embed_dim) * 0.02)
        self.fusion = nn.Sequential(nn.Linear(2 * embed_dim, embed_dim), nn.GELU(),
                                    nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim))


# ---------------------------------------------------------------------------
# Real 4-row metadata embedder (= the proven production embedder)
# ---------------------------------------------------------------------------
# Real 4-row per-assay metadata embedder: `[B, 4, F] -> [B, F, embed_dim]`.
# Rows = `[depth_log2, assay_id, read_length, run_type]`. This is candi_v2's `MetadataEmbedding`
# (the real embedder used in production) under a named q19 handle, reused on BOTH the encoder side
# (F = num_assays+1, incl. the control column) and the decoder side (F = num_assays). Sentinel
# handling (per field): depth/read_length continuous with learned MISSING(-1)/CLOZE(-2) embeddings;
# assay_id -> `Embedding(num_assays+3)` with MISSING->num_assays+1, CLOZE->num_assays+2;
# run_type -> `Embedding(num_runtypes+2)` with MISSING->num_runtypes, CLOZE->num_runtypes+1.
# Fuse 4xE -> E (+ LayerNorm).
RealMetaEmbedder = MetadataEmbedding


# ---------------------------------------------------------------------------
# Per-assay decoder (no across-assay pooling) + depth-offset log-link NB head
# ---------------------------------------------------------------------------

class DualCondDecoder(nn.Module):
    """Per-assay conditioned decoder with a depth-offset, log-linked NB head.

    Flow:  z [B,L2,d] --trunk--> feat [B,L,A*C] --view--> [B,L,A,C]
           per-assay FiLM (gamma_a, beta_a from y_meta[:,:,a]; adaLN-zero) modulates each assay's C
           features; a weight-shared head maps C -> (eta_a, raw_n_a).
           Depth-offset log-link (y_meta row `depth_row` = base log2_depth; real = row 0):
             log2_mu = (d - depth_center) + eta   [offset ON, valid depth]
             log2_mu = eta                        [offset OFF, or sentinel depth]
             mu = 2^clamp(log2_mu); n = softplus(raw_n)+eps; p = n/(n+mu).
    Returns p, n, eta, log2_mu, mu (eta is the offset-free mean statistic for M2).
    """

    def __init__(self, *, encoder_d_model: int, num_assays: int, feat_per_assay: int = 16,
                 meta_embed_dim: int = 32, use_offset: bool = True,
                 pool_meta: bool = False, depth_center: float = 25.1, mu_eps: float = 1e-6,
                 log2_mu_clamp: tuple = (-15.0, 30.0),
                 n_cnn_layers: int = 3, expansion_factor: int = 2, pool_size: int = 2,
                 conv_kernel_size: int = 3, trunk_norm: str = "layer",
                 use_layernorm: bool = True, depth_row: int = 0):
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
        self.meta_embedding = _LegacyPlaceholderMetaEmbedder(meta_embed_dim)
        # per-assay FiLM projection (adaLN-zero: starts at identity so recon is stable, steering grows)
        self.film_proj = nn.Linear(meta_embed_dim, 2 * self.C)
        nn.init.zeros_(self.film_proj.weight)
        nn.init.zeros_(self.film_proj.bias)
        # weight-shared per-assay heads (applied on the last dim of [B,L,A,C])
        self.head_eta = nn.Sequential(nn.Linear(self.C, self.C), nn.GELU(), nn.Linear(self.C, 1))
        self.head_n = nn.Sequential(nn.Linear(self.C, self.C), nn.GELU(), nn.Linear(self.C, 1))
        self.depth_row = int(depth_row)
        self.meta_embedding = RealMetaEmbedder(
            num_assays=int(num_assays), embed_dim=int(meta_embed_dim), use_layernorm=bool(use_layernorm))

    def forward(self, z: torch.Tensor, y_meta: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Mirror of DualCondDecoder.forward. ONLY change: depth is read from row self.depth_row (real=0)
        # instead of the hardcoded testbed row 2. (Golden file held fixed -> we override here.)
        feat = self.trunk(z)                                   # [B, L, A*C]
        B, Lq, _ = feat.shape
        feat = feat.view(B, Lq, self.A, self.C)                # [B, L, A, C]

        memb = self.meta_embedding(y_meta.float())             # [B, A, E]  (real 4-row embedder)
        if self.pool_meta:
            memb = memb.mean(dim=1, keepdim=True).expand(-1, self.A, -1)   # v1 across-assay pooling
        gamma, beta = self.film_proj(memb).chunk(2, dim=-1)    # [B, A, C] each
        feat = feat * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)   # per-assay FiLM, broadcast over L

        eta = self.head_eta(feat).squeeze(-1)                  # [B, L, A]
        raw_n = self.head_n(feat).squeeze(-1)                  # [B, L, A]

        depth = y_meta[:, self.depth_row, :]                   # [B, A]  (real depth = row 0)
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
# Full model — V2Encoder (per-assay, arcsinh, real-meta) + DualCondDecoder
# ---------------------------------------------------------------------------

class RealDualCondModel(nn.Module):
    """`V2Encoder` (per-assay `per_conv` FiLM, arcsinh, real 4-row meta) + `DualCondDecoder`.

    Honours the testbed model interface exactly: `forward(x_data, x_dna, x_meta, y_meta)` and
    `encode(x_data, x_dna, x_meta)`, so the reused `forward_full` / `encode_latent` / `nb_nll` work.
    `x_data`/`x_meta` include the appended control column (`F = num_assays+1`); `y_meta` is signal-only
    (`F = num_assays`). NB COUNTS ONLY.
    """

    def __init__(self, *, embed_dim: int = 32, dropout: float = 0.1, n_transformer_layers: int = 2,
                 feat_per_assay: int = 16, depth_center: float = 25.1, use_offset: bool = True,
                 num_assays: int = 8, context_length: int = 768,
                 d_model: int = 0, nhead: int = 4, pool_meta: bool = False):
        super().__init__()
        enc = EncoderConfig(
            num_assays=num_assays, context_length=context_length, metadata_embed_dim=embed_dim,
            signal_transform="arcsinh", missing_data_mode="mask_token",
            dropout=dropout, n_transformer_layers=n_transformer_layers,
            d_model=d_model, nhead=nhead,
            film_mode="per_conv",   # per-assay conv FiLM only; NO across-assay transformer pooling
        )
        self.encoder = V2Encoder(enc)
        # swap the encoder's metadata embedding for the (named) real 4-row embedder (identical class to
        # the default, made explicit so both towers share one embedder type — q19 PRD item 1).
        self.encoder.metadata_embedding = RealMetaEmbedder(
            num_assays=num_assays, embed_dim=embed_dim, use_layernorm=bool(enc.meta_embed_layernorm))
        self.decoder = DualCondDecoder(
            encoder_d_model=self.encoder.d_model, num_assays=num_assays, feat_per_assay=feat_per_assay,
            meta_embed_dim=embed_dim, use_offset=use_offset, pool_meta=pool_meta,
            depth_center=depth_center,
            use_layernorm=bool(enc.meta_embed_layernorm), depth_row=0)
        if enc.d_model == 0:
            print(f"[model] d_model auto-derived = (num_assays+1) * expansion^n_cnn = "
                  f"{self.encoder.d_model}; set --d-model to decouple transformer width from panel size")

    def forward(self, x_data, x_dna, x_meta, y_meta) -> Dict[str, torch.Tensor]:
        z = self.encoder.encode(x_data, x_dna, x_meta, return_meta=False)
        return self.decoder(z, y_meta)

    def encode(self, x_data, x_dna, x_meta) -> torch.Tensor:
        return self.encoder.encode(x_data, x_dna, x_meta, return_meta=False)


def build_real_model(*, embed_dim: int = 32, dropout: float = 0.1, n_transformer_layers: int = 2,
                     feat_per_assay: int = 16, depth_center: float = 25.1, use_offset: bool = True,
                     num_assays: int = 8, context_length: int = 768,
                     d_model: int = 0, nhead: int = 4, pool_meta: bool = False) -> RealDualCondModel:
    return RealDualCondModel(embed_dim=embed_dim, dropout=dropout,
                             n_transformer_layers=n_transformer_layers, feat_per_assay=feat_per_assay,
                             depth_center=depth_center, use_offset=use_offset, num_assays=num_assays,
                             context_length=context_length,
                             d_model=d_model, nhead=nhead, pool_meta=pool_meta)


# ---- NB helpers (log-link head: p = n/(n+mu), mean = mu) ----
def forward_counts(model: RealDualCondModel, batch: dict):
    """Run the model -> (p, n) NB params, [B, L, A]."""
    out = model(batch["x_data"], batch["x_dna"], batch["x_meta"], batch["y_meta"])
    return out["p"], out["n"]


def forward_full(model: RealDualCondModel, batch: dict) -> Dict[str, torch.Tensor]:
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


def encode_latent(model: RealDualCondModel, batch: dict) -> torch.Tensor:
    """Encoder latent z [B, L2, d_model] (for M3); depends only on x_data/x_dna/x_meta."""
    return model.encode(batch["x_data"], batch["x_dna"], batch["x_meta"])

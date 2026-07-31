"""Model layer for q19 — dual conditioning on REAL CANDI sandbox data (pre-production).

q19 is a HYBRID: the model CORE is the golden-reference testbed architecture
(`sandbox/diagnostics/dual_conditioning/model.py`, held FIXED — do NOT edit it), and the ONLY change
is the metadata assembly: the synthetic 2/3-row knob is swapped for the REAL 4-row covariate schema
`[depth_log2, assay_id, read_length, run_type]`.

Two concrete deltas vs the testbed (per q19 PRD §Implementation guide, new-code items 1-2):

1. **`RealMetaEmbedder`** — the real 4-row per-assay metadata embedder. This IS the production candi_v2
   `MetadataEmbedding` (encoder.py): depth(row0)->Linear, assay_id(row1)->Embedding(num_assays+3),
   read_length(row2)->Linear, run_type(row3)->Embedding(num_runtypes+2), per-field {-1 missing,-2 cloze}
   sentinels, fuse->LayerNorm; `[B,4,F] -> [B,F,E]`. It replaces BOTH the encoder's metadata embedding
   AND — the load-bearing change — the DECODER's, which in the testbed was the depth-NAIVE 3-row
   `DualCondMetaEmbedder`. q19/h41 requires depth to be a decoder-FiLM covariate so the offset-independent
   `eta` can carry a learned depth response; RealMetaEmbedder feeds depth into the fusion, and the offset
   reads depth separately (item 2).

2. **Offset re-keyed to row 0** — the testbed `DualCondDecoder` depth-offset log-link reads
   `y_meta[:, 2, :]` (synthetic depth row 2). Real depth is **row 0** -> `RealDualCondDecoder` reads
   `y_meta[:, self.depth_row, :]` with `depth_row=0`. `depth_center` stays 25.1 (real sandbox mean depth).

Everything else — per-assay adaLN-zero FiLM, weight-shared per-assay NB head, offset arithmetic
(`log2_mu=(d-center)+eta` when offset on & depth valid else `eta`; `mu=2^clamp`; `n=softplus+eps`;
`p=n/(n+mu)`), encoder `film_mode="per_conv"`, `signal_transform="arcsinh"` — is inherited unchanged.
NB COUNTS ONLY (no Gaussian signal head, no Bernoulli peak head).

The NB helpers (`forward_full`, `forward_counts`, `nb_mean`, `nb_nll`, `encode_latent`) are reused
verbatim from the testbed model — they are model-agnostic (`model(x_data,x_dna,x_meta,y_meta)` /
`model.encode(...)`), and `RealDualCondModel` honours that exact interface, so they are re-exported here.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from sandbox.batch import CLOZE, MISSING
from sandbox.candi_v2.config import EncoderConfig
from sandbox.candi_v2.encoder import MetadataEmbedding, V2Encoder
from sandbox.diagnostics.dual_conditioning.model import (
    DualCondDecoder,
    encode_latent,
    forward_counts,
    forward_full,
    nb_mean,
    nb_nll,
)

__all__ = [
    "RealMetaEmbedder",
    "RealDualCondDecoder",
    "RealDualCondModel",
    "build_real_model",
    # re-exported model-agnostic testbed primitives
    "forward_full",
    "forward_counts",
    "nb_mean",
    "nb_nll",
    "encode_latent",
]


# ---------------------------------------------------------------------------
# Real 4-row metadata embedder (= the proven production embedder)
# ---------------------------------------------------------------------------

class RealMetaEmbedder(MetadataEmbedding):
    """Real 4-row per-assay metadata embedder: `[B, 4, F] -> [B, F, embed_dim]`.

    Rows = `[depth_log2, assay_id, read_length, run_type]`. Behaviour is inherited verbatim from
    candi_v2's `MetadataEmbedding` (the real embedder used in production), so this is a named q19
    handle for it, reused on BOTH the encoder side (F = num_assays+1, incl. the control column) and the
    decoder side (F = num_assays). Sentinel handling (per field): depth/read_length continuous with
    learned MISSING(-1)/CLOZE(-2) embeddings; assay_id -> `Embedding(num_assays+3)` with
    MISSING->num_assays+1, CLOZE->num_assays+2; run_type -> `Embedding(num_runtypes+2)` with
    MISSING->num_runtypes, CLOZE->num_runtypes+1. Fuse 4xE -> E (+ LayerNorm).
    """


# ---------------------------------------------------------------------------
# Per-assay decoder with the depth-offset re-keyed to real row 0
# ---------------------------------------------------------------------------

class RealDualCondDecoder(DualCondDecoder):
    """Golden-reference `DualCondDecoder` with the two q19 metadata deltas.

    (1) the per-assay FiLM `meta_embedding` is the real 4-row `RealMetaEmbedder` (depth IS a FiLM
        covariate -> `eta` can respond to told-depth offset-independently, h41), replacing the testbed's
        depth-naive 3-row `DualCondMetaEmbedder`; (2) the depth-offset log-link reads depth from
        `y_meta[:, depth_row, :]` with `depth_row=0` (testbed used row 2).

    All the rest (adaLN-zero FiLM init, weight-shared per-assay `(eta, raw_n)` head, the offset
    arithmetic, clamps, `mu_eps`) is inherited from `DualCondDecoder`.
    """

    def __init__(self, *, encoder_d_model: int, num_assays: int, feat_per_assay: int = 16,
                 meta_embed_dim: int = 32, use_offset: bool = True, pool_meta: bool = False,
                 depth_center: float = 25.1, use_layernorm: bool = True, depth_row: int = 0, **kw):
        # super() builds the trunk/film_proj/heads + a placeholder DualCondMetaEmbedder we then replace.
        super().__init__(encoder_d_model=encoder_d_model, num_assays=num_assays,
                         feat_per_assay=feat_per_assay, meta_embed_dim=meta_embed_dim,
                         use_offset=use_offset, pool_meta=pool_meta, depth_center=depth_center, **kw)
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
# Full model — V2Encoder (per-assay, arcsinh, real-meta) + RealDualCondDecoder
# ---------------------------------------------------------------------------

class RealDualCondModel(nn.Module):
    """`V2Encoder` (per-assay `per_conv` FiLM, arcsinh, real 4-row meta) + `RealDualCondDecoder`.

    Honours the testbed model interface exactly: `forward(x_data, x_dna, x_meta, y_meta)` and
    `encode(x_data, x_dna, x_meta)`, so the reused `forward_full` / `encode_latent` / `nb_nll` work.
    `x_data`/`x_meta` include the appended control column (`F = num_assays+1`); `y_meta` is signal-only
    (`F = num_assays`). NB COUNTS ONLY.
    """

    def __init__(self, *, embed_dim: int = 32, dropout: float = 0.1, n_transformer_layers: int = 2,
                 feat_per_assay: int = 16, depth_center: float = 25.1, use_offset: bool = True,
                 num_assays: int = 8, context_length: int = 768):
        super().__init__()
        enc = EncoderConfig(
            num_assays=num_assays, context_length=context_length, metadata_embed_dim=embed_dim,
            signal_transform="arcsinh", missing_data_mode="mask_token",
            dropout=dropout, n_transformer_layers=n_transformer_layers,
            film_mode="per_conv",   # per-assay conv FiLM only; NO across-assay transformer pooling
        )
        self.encoder = V2Encoder(enc)
        # swap the encoder's metadata embedding for the (named) real 4-row embedder (identical class to
        # the default, made explicit so both towers share one embedder type — q19 PRD item 1).
        self.encoder.metadata_embedding = RealMetaEmbedder(
            num_assays=num_assays, embed_dim=embed_dim, use_layernorm=bool(enc.meta_embed_layernorm))
        self.decoder = RealDualCondDecoder(
            encoder_d_model=self.encoder.d_model, num_assays=num_assays, feat_per_assay=feat_per_assay,
            meta_embed_dim=embed_dim, use_offset=use_offset, depth_center=depth_center,
            use_layernorm=bool(enc.meta_embed_layernorm), depth_row=0)

    def forward(self, x_data, x_dna, x_meta, y_meta) -> Dict[str, torch.Tensor]:
        z = self.encoder.encode(x_data, x_dna, x_meta, return_meta=False)
        return self.decoder(z, y_meta)

    def encode(self, x_data, x_dna, x_meta) -> torch.Tensor:
        return self.encoder.encode(x_data, x_dna, x_meta, return_meta=False)


def build_real_model(*, embed_dim: int = 32, dropout: float = 0.1, n_transformer_layers: int = 2,
                     feat_per_assay: int = 16, depth_center: float = 25.1, use_offset: bool = True,
                     num_assays: int = 8, context_length: int = 768) -> RealDualCondModel:
    return RealDualCondModel(embed_dim=embed_dim, dropout=dropout,
                             n_transformer_layers=n_transformer_layers, feat_per_assay=feat_per_assay,
                             depth_center=depth_center, use_offset=use_offset, num_assays=num_assays,
                             context_length=context_length)

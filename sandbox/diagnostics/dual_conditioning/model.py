"""Model layer for the dual-conditioning testbed (crux q15).

CANDIv2, count-only NB head (`plain`, so it does NOT read y_meta row 0 as depth), encoder
signal_transform=arcsinh. The real 4-row MetadataEmbedding is swapped for `DualCondMetaEmbedder`,
which reads only (aug_family, aug_param) and applies the h33 normalization arm INSIDE the module —
so the stored param stays raw-positive and never collides with the -1/-2 availability sentinels.

Import-and-swap: v2 encoder/decoder use `self.metadata_embedding(meta)` / `self.decoder_meta_embedding(meta)`
and rely only on the [B, F', embed] output contract, so replacing those attributes is transparent.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from sandbox.candi_v2.config import CANDIv2Config, EncoderConfig, DecoderConfig
from sandbox.candi_v2.model import CANDIv2
from sandbox.diagnostics.dual_conditioning import transforms as T

MISSING = -1


class DualCondMetaEmbedder(nn.Module):
    """(aug_family, aug_param) -> [B, F', embed_dim].  norm in {'none','zscore','log'} (h33)."""

    def __init__(self, embed_dim: int, norm: str = "zscore", use_layernorm: bool = True, eps: float = 1e-6):
        super().__init__()
        assert norm in ("none", "zscore", "log")
        self.norm = norm
        self.eps = float(eps)
        self.family_embedding = nn.Embedding(T.N_FAMILIES + 1, embed_dim)   # +1 = missing
        self.param_proj = nn.Linear(1, embed_dim)
        self.param_missing_emb = nn.Parameter(torch.randn(embed_dim) * 0.02)
        stats = torch.tensor(T.family_param_stats(), dtype=torch.float32)   # [N_FAM, 2]
        self.register_buffer("fam_mean", stats[:, 0])
        self.register_buffer("fam_std", stats[:, 1])
        layers = [nn.Linear(2 * embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim)]
        if use_layernorm:
            layers.append(nn.LayerNorm(embed_dim))
        self.fusion = nn.Sequential(*layers)

    def forward(self, metadata: torch.Tensor) -> torch.Tensor:
        fam_raw = metadata[:, 0, :].long()                     # [B, F'] in {-1..6}
        param = metadata[:, 1, :].float()                      # raw positive, or -1 for missing
        missing = fam_raw < 0
        fam_idx = torch.where(missing, torch.full_like(fam_raw, T.N_FAMILIES), fam_raw)
        fam_emb = self.family_embedding(fam_idx)               # [B, F', d]

        fam_c = fam_raw.clamp(0, T.N_FAMILIES - 1)
        if self.norm == "zscore":
            x = (param - self.fam_mean[fam_c]) / self.fam_std[fam_c]
        elif self.norm == "log":
            x = torch.log(param.clamp_min(self.eps))
        else:
            x = param
        x = torch.where(missing, torch.zeros_like(x), x)
        par_emb = self.param_proj(x.unsqueeze(-1))             # [B, F', d]
        par_emb = torch.where(missing.unsqueeze(-1), self.param_missing_emb.to(par_emb.dtype), par_emb)

        return self.fusion(torch.cat([fam_emb, par_emb], dim=-1))


def build_model(norm: str = "zscore", *, embed_dim: int = 32, dropout: float = 0.1,
                n_transformer_layers: int = 2, dec_film: str = "per_deconv_layer") -> CANDIv2:
    """CANDIv2 (count-only, plain NB, arcsinh input) with the 2-row embedder swapped in on both sides.

    dec_film: decoder metadata conditioning. 'single_pre_decoder' (one global FiLM on the latent) could
    not learn output steering (M2~0 while M1/M3 rose); 'per_deconv_layer' conditions at every decoder
    layer for stronger, more direct output steering.
    """
    enc = EncoderConfig(
        num_assays=8, context_length=768, metadata_embed_dim=embed_dim,
        signal_transform="arcsinh", missing_data_mode="mask_token",
        dropout=dropout, n_transformer_layers=n_transformer_layers,
    )
    dec = DecoderConfig(
        heads="count_only", count_head="plain",
        film_mode=dec_film, meta_embed_dim=embed_dim,
    )
    cfg = CANDIv2Config(encoder=enc, decoder=dec)
    model = CANDIv2(cfg)

    # --- swap the metadata embedders (import-and-swap; no candi_v2 edits) ---
    model.encoder.metadata_embedding = DualCondMetaEmbedder(
        embed_dim, norm, use_layernorm=bool(enc.meta_embed_layernorm))
    if model.decoder.decoder_meta_embedding is not None:
        model.decoder.decoder_meta_embedding = DualCondMetaEmbedder(
            embed_dim, norm, use_layernorm=bool(dec.meta_embed_layernorm))

    # --- adaLN-zero: zero-init decoder FiLM projections so conditioning starts as identity ---
    # (default v2 init perturbs the output from step 0 and destabilizes reconstruction; starting at
    # identity keeps M1 stable while output steering M2 grows gradually.)
    from sandbox.candi_v2.decoder import PreDecoderFiLM, PerDeconvLayerFiLM
    for mod in model.decoder.modules():
        if isinstance(mod, (PreDecoderFiLM, PerDeconvLayerFiLM)):
            nn.init.zeros_(mod.proj.weight)
            nn.init.zeros_(mod.proj.bias)
    return model


# ---- NB helpers (plain head: p = n/(n+mu), mean = mu) ----
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


def forward_counts(model: CANDIv2, batch: dict):
    """Run the model -> (p, n) NB params, [B, L, A]."""
    out = model(batch["x_data"], batch["x_dna"], batch["x_meta"], batch["y_meta"])
    return out["p"], out["n"]


def encode_latent(model: CANDIv2, batch: dict) -> torch.Tensor:
    """Encoder latent z [B, L2, d_model] (for M3); depends only on x_data/x_dna/x_meta."""
    return model.encoder.encode(batch["x_data"], batch["x_dna"], batch["x_meta"], return_meta=False)

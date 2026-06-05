"""NB head with library-size offset for synthetic diagnostics (E29 prototype).

log2(mu) = (log2_depth - depth_center) + eta,  mu = 2^log2(mu)
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from sandbox.candi_v2.decoder import V2Decoder


class DepthOffsetNegativeBinomialLayer(nn.Module):
    """Predict log-enrichment eta; multiply by known depth scale for mu."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        eps: float = 1e-6,
        depth_center: float = 0.0,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.depth_center = depth_center
        self.linear_eta = nn.Linear(input_dim, output_dim)
        self.linear_n = nn.Linear(input_dim, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        depth_log2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        eta = self.linear_eta(x)
        d_centered = depth_log2.unsqueeze(1).to(x.dtype) - self.depth_center
        log2_mu = d_centered + eta
        mu = torch.pow(2.0, log2_mu)
        mu = mu.clamp(min=self.eps)
        n = F.softplus(self.linear_n(x)) + self.eps
        p = n / (n + mu)
        p = torch.clamp(p, min=self.eps, max=1.0 - self.eps)
        return p, n


class V2DecoderDepthOffset(V2Decoder):
    """V2Decoder that applies depth-offset NB head using y_meta depth row."""

    def __init__(self, *args, depth_center: float = 0.0, **kwargs) -> None:
        self._depth_center = depth_center
        super().__init__(*args, **kwargs)
        if "count" in self._active_heads:
            self.neg_binom_layer = DepthOffsetNegativeBinomialLayer(
                self.signal_dim, self.signal_dim, depth_center=depth_center,
            )

    def forward(
        self,
        z: torch.Tensor,
        y_meta: torch.Tensor,
    ) -> Dict[str, Optional[torch.Tensor]]:
        saved = self.neg_binom_layer
        self.neg_binom_layer = None
        out = super().forward(z, y_meta)
        self.neg_binom_layer = saved
        if saved is None:
            return out
        pooled_meta = None
        dec_meta_embed = None
        if self.decoder_meta_embedding is not None:
            dec_meta_embed = self.decoder_meta_embedding(y_meta.float())
            pooled_meta = dec_meta_embed.mean(dim=1)
        z_mod = z
        if self.pre_decoder_film is not None and dec_meta_embed is not None:
            z_mod = self.pre_decoder_film(z, dec_meta_embed)
        decoded = self.shared_trunk(
            z_mod,
            film_layers=self.per_layer_film_shared
            if self.film_mode == "per_deconv_layer" else None,
            pooled_meta=pooled_meta,
        )
        p, n = saved(decoded, y_meta[:, 0, :])
        out["p"] = p
        out["n"] = n
        return out


def apply_depth_offset_decoder(model, *, depth_center: float = 0.0) -> None:
    """Swap decoder for depth-offset variant (in-place on CANDIv2)."""
    old = model.decoder
    new_dec = V2DecoderDepthOffset(
        old.cfg,
        encoder_d_model=old.encoder_d_model,
        signal_dim=old.signal_dim,
        depth_center=depth_center,
    )
    new_dec.load_state_dict(old.state_dict(), strict=False)
    model.decoder = new_dec.to(next(model.parameters()).device)

#!/usr/bin/env python3
"""Agent-editable training config and count head (ONLY file the agent modifies).

Usage:
    python -m sandbox.diagnostics.autoresearch.train
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from sandbox.candi_v2.decoder import V2Decoder
from sandbox.candi_v2.model import CANDIv2


@dataclass
class TrainConfig:
    use_depth_offset: bool = True
    depth_center: float = 27.0
    # Optimizer (agent-tunable)
    optimizer: str = "adamax"  # adam | adamw | adamax | sgd
    lr: float = 1e-3
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    sgd_momentum: float = 0.0
    clip_norm: float = 0.5
    # Loss weights
    obs_weight: float = 0.5
    imp_weight: float = 8.0
    count_weight: float = 1.0


def get_config() -> TrainConfig:
    return TrainConfig()


def _compat_patch() -> None:
    """Ignore preserve_assay_id when repo config is older than diagnostics harness."""
    import sandbox.config_types as ct

    _orig_cfd = ct.config_from_dict

    def _config_from_dict(cls, raw, path=""):
        if isinstance(raw, dict) and cls is ct.MaskingConfig:
            raw = {k: v for k, v in raw.items() if k != "preserve_assay_id"}
        return _orig_cfd(cls, raw, path=path)

    ct.config_from_dict = _config_from_dict

    import sandbox.batch as batch_mod

    _orig_mm = batch_mod.make_masker

    def _make_masker(**kwargs):
        kwargs.pop("preserve_assay_id", None)
        return _orig_mm(**kwargs)

    batch_mod.make_masker = _make_masker

    import sandbox.data as data_mod

    _orig_init = data_mod.SandboxH5Dataset.__init__

    def _dataset_init(self, *args, **kwargs):
        kwargs.pop("preserve_assay_id", None)
        return _orig_init(self, *args, **kwargs)

    data_mod.SandboxH5Dataset.__init__ = _dataset_init


class DepthOffsetNegativeBinomialLayer(nn.Module):
    """Predict log-enrichment eta; mu = 2^(depth - center) * exp(eta)."""

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
        d = depth_log2.unsqueeze(1).to(x.dtype) - self.depth_center
        mu = torch.pow(2.0, d) * torch.exp(eta)
        mu = mu.clamp(min=self.eps)
        n = F.softplus(self.linear_n(x)) + self.eps
        p = n / (n + mu)
        p = torch.clamp(p, min=self.eps, max=1.0 - self.eps)
        return p, n


class V2DecoderDepthOffset(V2Decoder):
    """V2Decoder with depth-offset NB head using y_meta depth row."""

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


def patch_count_head(model: CANDIv2, cfg: TrainConfig) -> None:
    """Apply count-head patch from TrainConfig (agent edits this + TrainConfig)."""
    if not cfg.use_depth_offset:
        return
    old = model.decoder
    new_dec = V2DecoderDepthOffset(
        old.cfg,
        encoder_d_model=old.encoder_d_model,
        signal_dim=old.signal_dim,
        depth_center=cfg.depth_center,
    )
    new_dec.load_state_dict(old.state_dict(), strict=False)
    model.decoder = new_dec.to(next(model.parameters()).device)


def build_optimizer(model: CANDIv2, cfg: TrainConfig) -> torch.optim.Optimizer:
    """Construct optimizer from TrainConfig (agent may change name and hparams)."""
    params = model.parameters()
    name = cfg.optimizer.lower().strip()
    if name == "adam":
        return torch.optim.Adam(
            params, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
            eps=cfg.eps, weight_decay=cfg.weight_decay,
        )
    if name == "adamw":
        return torch.optim.AdamW(
            params, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
            eps=cfg.eps, weight_decay=cfg.weight_decay,
        )
    if name == "adamax":
        return torch.optim.Adamax(
            params, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
            eps=cfg.eps, weight_decay=cfg.weight_decay,
        )
    if name == "sgd":
        return torch.optim.SGD(
            params, lr=cfg.lr, momentum=cfg.sgd_momentum,
            weight_decay=cfg.weight_decay,
        )
    raise ValueError(f"unsupported optimizer {cfg.optimizer!r}; use adam|adamw|adamax|sgd")


def count_head_param_count(model: CANDIv2) -> int:
    layer = getattr(model.decoder, "neg_binom_layer", None)
    if layer is None:
        return 0
    return sum(p.numel() for p in layer.parameters())


def main() -> int:
    _compat_patch()
    from sandbox.diagnostics.autoresearch import prepare

    return prepare.run_from_train()


if __name__ == "__main__":
    raise SystemExit(main())

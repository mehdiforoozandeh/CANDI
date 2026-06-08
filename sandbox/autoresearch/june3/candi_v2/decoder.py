"""CANDI v2 decoder — deconv towers with optional FiLM and configurable heads.

Architecture:
  1. Optional PreDecoderFiLM: single-shot FiLM on encoder latent, conditioned
     on target metadata (tells the decoder what to reconstruct).
  2. DeconvTower(s): ConvTranspose1d + residual skip, upsampling from L2 → L.
     Either a shared trunk (split into heads at the end) or separate towers.
  3. Output heads: NegativeBinomialLayer (counts) + PeakLayer (peaks),
     optionally GaussianLayer (pval) when heads="all".

Shapes (default 8 assays, L2=96, L=768):
  Input:  z [B, L2, d_model]   y_meta [B, 4, A]
  Output: dict with 'p', 'n', 'peak', optionally 'mu', 'var'
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from sandbox.batch import CLOZE, MISSING
from sandbox.autoresearch.june3.candi_v2.config import DecoderConfig
from sandbox.autoresearch.june3.candi_v2.encoder import MetadataEmbedding


class RMSNorm(nn.Module):
    """RMS layer norm for Conv1d (B, C, L) (vendored from model.py)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x = x * rms * self.weight
        return x.permute(0, 2, 1)


class NegativeBinomialLayer(nn.Module):
    """Plain NB head: predicts mu via softplus, n via softplus (vendored from model.py)."""

    def __init__(self, input_dim: int, output_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.linear_mean = nn.Linear(input_dim, output_dim)
        self.linear_n = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor):
        mu = F.softplus(self.linear_mean(x)) + self.eps
        n = F.softplus(self.linear_n(x)) + self.eps
        p = n / (n + mu)
        p = torch.clamp(p, min=self.eps, max=1.0 - self.eps)
        return p, n


class GaussianLayer(nn.Module):
    """Gaussian pval head: predicts mu and var (vendored from model.py)."""

    def __init__(self, input_dim: int, output_dim: int, var_min: float = 0.1) -> None:
        super().__init__()
        self.var_min = float(var_min)
        self.linear_mu = nn.Linear(input_dim, output_dim)
        self.linear_var = nn.Linear(input_dim, output_dim)
        nn.init.normal_(self.linear_mu.weight, mean=0.0, std=1e-4)
        nn.init.constant_(self.linear_mu.bias, 0.0)
        nn.init.normal_(self.linear_var.weight, mean=0.0, std=1e-4)
        nn.init.constant_(self.linear_var.bias, 0.5)

    def forward(self, x: torch.Tensor):
        mu = F.softplus(self.linear_mu(x))
        var = F.softplus(self.linear_var(x)) + self.var_min
        return mu, var


class PeakLayer(nn.Module):
    """Peak probability head: sigmoid output (vendored from model.py)."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear_peak = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear_peak(x))


# ---------------------------------------------------------------------------
# Deconv building blocks (from jepa_decoder.py, standalone copy)
# ---------------------------------------------------------------------------

class DeconvBlock(nn.Module):
    """ConvTranspose1d → Norm."""

    def __init__(
        self,
        in_c: int,
        out_c: int,
        kernel_size: int,
        stride: int,
        *,
        norm: str = "layer",
        groups: int = 1,
    ) -> None:
        super().__init__()
        padding = (int(kernel_size) - 1) // 2
        output_padding = int(stride) - 1
        self.normtype = str(norm)
        self.deconv: nn.Module = nn.ConvTranspose1d(
            in_c, out_c,
            kernel_size=int(kernel_size), dilation=1, stride=int(stride),
            padding=padding, output_padding=output_padding, groups=int(groups),
        )
        if self.normtype == "weight":
            self.deconv = nn.utils.weight_norm(self.deconv)
        elif self.normtype == "batch":
            self.norm = nn.BatchNorm1d(out_c)
        elif self.normtype == "layer":
            self.norm = nn.LayerNorm(out_c)
        elif self.normtype == "group":
            self.norm = nn.GroupNorm(groups, out_c)
        elif self.normtype == "instance":
            self.norm = nn.InstanceNorm1d(out_c, affine=True, eps=1e-5)
        elif self.normtype == "rms":
            self.norm = RMSNorm(out_c)
        else:
            raise ValueError(f"unsupported norm={norm}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.deconv(x)
        if self.normtype == "layer":
            x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        elif self.normtype in {"batch", "group", "instance", "rms"}:
            x = self.norm(x)
        return x


class DeconvTower(nn.Module):
    """DeconvBlock + 1×1 residual skip → GELU."""

    def __init__(
        self,
        in_c: int,
        out_c: int,
        kernel_size: int,
        *,
        stride: int = 2,
        groups: int = 1,
        norm: str = "layer",
    ) -> None:
        super().__init__()
        self.deconv = DeconvBlock(
            in_c, out_c, kernel_size, stride, norm=norm, groups=groups,
        )
        self.rdeconv = nn.ConvTranspose1d(
            in_c, out_c, kernel_size=1,
            stride=int(stride), output_padding=int(stride) - 1,
            groups=int(groups),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.deconv(x) + self.rdeconv(x))


# ---------------------------------------------------------------------------
# Depth-offset NB count head (E29 / E30)
# ---------------------------------------------------------------------------

class DepthOffsetNegativeBinomialLayer(nn.Module):
    """NB count head with a centered library-size offset and sentinel-aware gating.

    Instead of predicting the NB mean ``mu`` directly, the network predicts a
    per-position log2-enrichment ``eta`` and a dispersion ``n``. When depth is
    valid, the mean is reconstructed in log2 space using known sequencing depth
    as a fixed offset (DESeq2 / edgeR / scVI size-factor convention):

        d        = log2(seq_depth)                     # target metadata row 0
        log2_mu  = (d - depth_center) + eta
        mu       = 2^log2_mu
        p        = n / (n + mu)

    ``eta`` is in log2 units (not natural log). Centering keeps ``(d - c)`` near
    zero on typical depths so ``eta`` learns enrichment rather than absolute
    library size.

    **Valid depth:** any ``depth_log2`` value that is not a metadata sentinel
    (``MISSING=-1``, ``CLOZE=-2`` from ``sandbox.batch``).

    **Invalid depth (fallback):** sentinel columns use ``log2_mu = eta`` (no depth
    offset) — same ``linear_eta`` head, no fabricated depth. Raw sentinels in
    ``2^(d-c)`` would yield ``d-c ≈ -25`` and collapse ``mu`` toward zero.
    Fallback is equivalent to ``log2(seq_depth) == depth_center`` without writing
    a fake depth into ``y_meta``.

    The gate is always on (no config knob).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        depth_center: float = 24.0,
        eps: float = 1e-6,
        learnable_depth_center: bool = False,
        learnable_depth_slope: bool = False,
    ) -> None:
        super().__init__()
        if learnable_depth_center:
            self.depth_center = nn.Parameter(torch.tensor(float(depth_center)))
        else:
            self.depth_center = float(depth_center)
        if learnable_depth_slope:
            self.log_depth_slope = nn.Parameter(torch.zeros(1))
        else:
            self.log_depth_slope = None
        self.eps = float(eps)
        self.linear_eta = nn.Linear(input_dim, output_dim)
        self.linear_n = nn.Linear(input_dim, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        depth_log2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:          [B, L, signal_dim] — decoded trunk features
            depth_log2: [B, signal_dim]    — log2(seq_depth) per assay (y_meta row 0)
        Returns:
            (p, n), each [B, L, signal_dim]
        """
        eta = self.linear_eta(x)                                   # [B, L, A]
        # Per-assay gate: sentinels must not enter 2^(d-c) (see class docstring).
        valid = (depth_log2 != MISSING) & (depth_log2 != CLOZE)    # [B, A]
        d_centered = depth_log2.unsqueeze(1).to(x.dtype) - self.depth_center  # [B, 1, A]
        if self.log_depth_slope is not None:
            d_centered = torch.exp(self.log_depth_slope) * d_centered
        log2_mu_offset = d_centered + eta                            # [B, L, A]
        log2_mu_fallback = eta                                       # [B, L, A]
        log2_mu = torch.where(valid.unsqueeze(1), log2_mu_offset, log2_mu_fallback)
        mu = torch.pow(2.0, log2_mu)
        mu = mu.clamp(min=self.eps)
        n = F.softplus(self.linear_n(x)) + self.eps
        p = n / (n + mu)
        p = torch.clamp(p, min=self.eps, max=1.0 - self.eps)
        return p, n


# ---------------------------------------------------------------------------
# Pre-decoder FiLM
# ---------------------------------------------------------------------------

class PreDecoderFiLM(nn.Module):
    """Single-shot FiLM on encoder latent, conditioned on target metadata.

    Projects the mean of target metadata embeddings to (scale, shift) vectors
    and modulates the encoder output before it enters the decoder towers.
    """

    def __init__(self, meta_embed_dim: int, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(meta_embed_dim, 2 * d_model)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.normal_(self.proj.bias, mean=0.0, std=0.1)

    def forward(
        self, z: torch.Tensor, meta_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z:          [B, L2, d_model]
            meta_embed: [B, A, embed_dim]  (target metadata embeddings)
        Returns:
            [B, L2, d_model] — FiLM-modulated latent
        """
        pooled = meta_embed.mean(dim=1)  # [B, embed_dim]
        scale, shift = self.proj(pooled).chunk(2, dim=-1)
        return z * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class PerDeconvLayerFiLM(nn.Module):
    """FiLM applied at each deconv layer inside the decoder tower."""

    def __init__(self, meta_embed_dim: int, channel_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(meta_embed_dim, 2 * channel_dim)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.normal_(self.proj.bias, mean=0.0, std=0.1)

    def forward(
        self, x: torch.Tensor, pooled_meta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:           [B, C, L] — channel-first conv activations
            pooled_meta: [B, embed_dim]
        Returns:
            [B, C, L]
        """
        scale, shift = self.proj(pooled_meta).chunk(2, dim=-1)
        return x * (1.0 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)


# ---------------------------------------------------------------------------
# Decoder trunk
# ---------------------------------------------------------------------------

def _build_deconv_channel_schedule(
    input_dim: int,
    signal_dim: int,
    n_layers: int,
    expansion_factor: int,
) -> List[int]:
    """Compute the channel schedule for the deconv tower (high → low)."""
    channels = [input_dim]
    for i in range(1, n_layers + 1):
        div = expansion_factor ** i
        out_c = input_dim // div
        if i == n_layers:
            out_c = signal_dim
        if out_c < signal_dim:
            out_c = signal_dim
        channels.append(int(out_c))
    return channels


class DecoderTrunk(nn.Module):
    """Deconv tower: [B, L2, proj_dim] → [B, L, signal_dim].

    Optionally applies per-layer FiLM conditioning when film_layers is set.
    """

    def __init__(
        self,
        *,
        proj_dim: int,
        signal_dim: int,
        decoder_input_dim: int,
        n_cnn_layers: int,
        expansion_factor: int,
        pool_size: int,
        conv_kernel_size: int = 3,
        grouped: bool = False,
        norm: str = "layer",
    ) -> None:
        super().__init__()
        self.proj_dim = int(proj_dim)
        self.signal_dim = int(signal_dim)
        self.decoder_input_dim = int(decoder_input_dim)
        if self.decoder_input_dim % self.signal_dim != 0:
            raise ValueError(
                f"decoder_input_dim ({self.decoder_input_dim}) must be "
                f"divisible by signal_dim ({self.signal_dim})"
            )
        self.input_proj = nn.Linear(self.proj_dim, self.decoder_input_dim)

        channels = _build_deconv_channel_schedule(
            self.decoder_input_dim, self.signal_dim,
            int(n_cnn_layers), int(expansion_factor),
        )
        groups = self.signal_dim if bool(grouped) else 1
        if bool(grouped):
            for c in channels:
                if c % groups != 0:
                    raise ValueError(
                        f"grouped decoder channel {c} not divisible by groups={groups}"
                    )

        self.deconv = nn.ModuleList([
            DeconvTower(
                channels[i], channels[i + 1], int(conv_kernel_size),
                stride=int(pool_size), groups=groups, norm=norm,
            )
            for i in range(int(n_cnn_layers))
        ])
        self.channel_schedule = channels
        self.drop = nn.Dropout(p=0.1)

    def forward(
        self,
        z: torch.Tensor,
        film_layers: Optional[nn.ModuleList] = None,
        pooled_meta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            z:           [B, L2, proj_dim]
            film_layers: optional per-layer FiLM modules
            pooled_meta: [B, embed_dim] for per-layer FiLM
        Returns:
            [B, L, signal_dim]
        """
        x = self.input_proj(z).permute(0, 2, 1)  # [B, C, L2]
        n_layers = len(self.deconv)
        for i, layer in enumerate(self.deconv):
            x = layer(x)
            if film_layers is not None and pooled_meta is not None:
                x = film_layers[i](x, pooled_meta)
            if i < n_layers - 1:
                x = self.drop(x)
        return x.permute(0, 2, 1)  # [B, L, signal_dim]


# ---------------------------------------------------------------------------
# Full decoder
# ---------------------------------------------------------------------------

class V2Decoder(nn.Module):
    """CANDI v2 decoder: optional FiLM → deconv trunk(s) → output heads.

    Supports two trunk modes:
      - "shared": single deconv trunk, split into NB/peak heads at the output
      - "separate": independent deconv tower per head

    Output heads:
      - NegativeBinomialLayer → (p, n) for count reconstruction
      - PeakLayer → peak probability
      - GaussianLayer → (mu, var) for pval (only when heads="all")
    """

    def __init__(
        self,
        cfg: DecoderConfig,
        *,
        encoder_d_model: int,
        signal_dim: int,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.signal_dim = int(signal_dim)
        self.encoder_d_model = int(encoder_d_model)
        self.heads_mode = str(cfg.heads)
        self.trunk_mode = str(cfg.trunk)
        self.film_mode = str(cfg.film_mode)

        default_dec_dim = self.signal_dim * (
            int(cfg.expansion_factor) ** int(cfg.n_cnn_layers)
        )

        # Determine which heads are active
        self._active_heads = self._resolve_active_heads(self.heads_mode)

        # -- Decoder metadata embedding (separate from encoder) --
        self.decoder_meta_embedding: Optional[MetadataEmbedding] = None
        if self.film_mode != "none":
            self.decoder_meta_embedding = MetadataEmbedding(
                num_assays=self.signal_dim,  # decoder only sees signal assays (no control)
                embed_dim=int(cfg.meta_embed_dim),
                use_layernorm=bool(cfg.meta_embed_layernorm),
            )

        # -- Pre-decoder FiLM --
        self.pre_decoder_film: Optional[PreDecoderFiLM] = None
        if self.film_mode in ("single_pre_decoder", "per_deconv_layer"):
            self.pre_decoder_film = PreDecoderFiLM(
                int(cfg.meta_embed_dim), self.encoder_d_model,
            )

        # -- Per-deconv-layer FiLM --
        self.per_layer_film: Optional[nn.ModuleDict] = None

        # -- Build trunk(s) --
        trunk_kwargs = dict(
            proj_dim=self.encoder_d_model,
            signal_dim=self.signal_dim,
            decoder_input_dim=default_dec_dim,
            n_cnn_layers=int(cfg.n_cnn_layers),
            expansion_factor=int(cfg.expansion_factor),
            pool_size=int(cfg.pool_size),
            conv_kernel_size=int(cfg.conv_kernel_size),
            grouped=bool(cfg.grouped_deconv),
            norm=str(cfg.norm),
        )

        if self.trunk_mode == "shared":
            self.shared_trunk = DecoderTrunk(**trunk_kwargs)
            # Build per-layer FiLM for shared trunk
            if self.film_mode == "per_deconv_layer":
                self.per_layer_film_shared = nn.ModuleList([
                    PerDeconvLayerFiLM(int(cfg.meta_embed_dim), ch)
                    for ch in self.shared_trunk.channel_schedule[1:]
                ])
            else:
                self.per_layer_film_shared = None
        elif self.trunk_mode == "separate":
            self.separate_trunks = nn.ModuleDict()
            if self.film_mode == "per_deconv_layer":
                self.per_layer_film = nn.ModuleDict()
            for head_name in self._active_heads:
                trunk = DecoderTrunk(**trunk_kwargs)
                self.separate_trunks[head_name] = trunk
                if self.film_mode == "per_deconv_layer":
                    self.per_layer_film[head_name] = nn.ModuleList([
                        PerDeconvLayerFiLM(int(cfg.meta_embed_dim), ch)
                        for ch in trunk.channel_schedule[1:]
                    ])
        else:
            raise ValueError(f"Unsupported trunk={self.trunk_mode}")

        # -- Output heads --
        self.neg_binom_layer: Optional[nn.Module] = None
        self.peak_layer: Optional[PeakLayer] = None
        self.gaussian_layer: Optional[GaussianLayer] = None

        # Count-head parameterization (E29 / E30). When "depth_offset", the NB
        # mean is reconstructed from a known library-size offset using the depth
        # row of the target metadata; otherwise the plain head predicts mu.
        self.count_head_mode = str(cfg.count_head)
        self._count_depth_offset = self.count_head_mode == "depth_offset"

        if "count" in self._active_heads:
            if self._count_depth_offset:
                self.neg_binom_layer = DepthOffsetNegativeBinomialLayer(
                    self.signal_dim, self.signal_dim,
                    depth_center=float(cfg.depth_center),
                    eps=float(cfg.mu_eps),
                    learnable_depth_center=bool(cfg.learnable_depth_center),
                    learnable_depth_slope=bool(cfg.learnable_depth_slope),
                )
            else:
                self.neg_binom_layer = NegativeBinomialLayer(
                    self.signal_dim, self.signal_dim,
                )
        if "peak" in self._active_heads:
            self.peak_layer = PeakLayer(self.signal_dim, self.signal_dim)
        if "pval" in self._active_heads:
            self.gaussian_layer = GaussianLayer(
                self.signal_dim, self.signal_dim,
                var_min=float(cfg.gaussian_var_min),
            )

    @staticmethod
    def _resolve_active_heads(heads_mode: str) -> Tuple[str, ...]:
        if heads_mode == "count_peak":
            return ("count", "peak")
        if heads_mode == "count_only":
            return ("count",)
        if heads_mode == "peak_only":
            return ("peak",)
        if heads_mode == "all":
            return ("count", "peak", "pval")
        raise ValueError(f"Unknown heads={heads_mode}")

    @property
    def active_heads(self) -> Tuple[str, ...]:
        return self._active_heads

    def forward(
        self,
        z: torch.Tensor,
        y_meta: torch.Tensor,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            z:      [B, L2, d_model] — encoder latent
            y_meta: [B, 4, A]        — target metadata (signal assays only, no control)
        Returns:
            dict with keys: 'p', 'n', 'peak', 'mu', 'var' (None for inactive heads)
        """
        # Compute decoder metadata embedding if needed
        pooled_meta: Optional[torch.Tensor] = None
        if self.decoder_meta_embedding is not None:
            dec_meta_embed = self.decoder_meta_embedding(y_meta.float())
            pooled_meta = dec_meta_embed.mean(dim=1)  # [B, embed_dim]

        # Pre-decoder FiLM
        if self.pre_decoder_film is not None and self.decoder_meta_embedding is not None:
            z = self.pre_decoder_film(z, dec_meta_embed)

        # Depth-offset count head reads log2(seq_depth) from target metadata row 0.
        # y_meta is [B, 4, signal_dim] (signal assays only; no control column).
        # Sentinel depths (MISSING/CLOZE) are gated inside DepthOffsetNegativeBinomialLayer.
        count_depth = y_meta[:, 0, :] if self._count_depth_offset else None

        # Decode through trunk(s) and apply heads
        out: Dict[str, Optional[torch.Tensor]] = {
            "p": None, "n": None, "peak": None, "mu": None, "var": None,
        }

        if self.trunk_mode == "shared":
            decoded = self.shared_trunk(
                z,
                film_layers=self.per_layer_film_shared if self.film_mode == "per_deconv_layer" else None,
                pooled_meta=pooled_meta,
            )
            if self.neg_binom_layer is not None:
                if self._count_depth_offset:
                    p, n = self.neg_binom_layer(decoded, count_depth)
                else:
                    p, n = self.neg_binom_layer(decoded)
                out["p"] = p
                out["n"] = n
            if self.peak_layer is not None:
                out["peak"] = self.peak_layer(decoded)
            if self.gaussian_layer is not None:
                mu, var = self.gaussian_layer(decoded)
                out["mu"] = mu
                out["var"] = var
        else:
            for head_name in self._active_heads:
                film_l = None
                if self.per_layer_film is not None and head_name in self.per_layer_film:
                    film_l = self.per_layer_film[head_name]
                decoded = self.separate_trunks[head_name](
                    z, film_layers=film_l, pooled_meta=pooled_meta,
                )
                if head_name == "count" and self.neg_binom_layer is not None:
                    if self._count_depth_offset:
                        p, n = self.neg_binom_layer(decoded, count_depth)
                    else:
                        p, n = self.neg_binom_layer(decoded)
                    out["p"] = p
                    out["n"] = n
                elif head_name == "peak" and self.peak_layer is not None:
                    out["peak"] = self.peak_layer(decoded)
                elif head_name == "pval" and self.gaussian_layer is not None:
                    mu, var = self.gaussian_layer(decoded)
                    out["mu"] = mu
                    out["var"] = var

        return out

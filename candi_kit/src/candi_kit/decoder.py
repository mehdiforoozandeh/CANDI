"""CANDI kit decoder — deconv trunk only.

VENDORED VERBATIM from EpiDenoise/sandbox/candi_v2/decoder.py:33-77 (DeconvBlock),
80-104 (DeconvTower), 240-256 (_build_deconv_channel_schedule), 259-329 (DecoderTrunk).

Reduced to the 4 symbols reachable from build_real_model. The pval/peak heads
(V2Decoder + GaussianLayer/PeakLayer) are NOT shipped — see EXTENSION_HOOKS.md.

FROZEN CONSTRUCTION ORDER: DecoderTrunk.__init__ draws from the global torch RNG
(input_proj, then each DeconvTower in schedule order). Do not reorder, insert, or
remove module construction here — historical checkpoints depend on the draw sequence.

NOTE: DecoderTrunk.forward's `film_layers` / `pooled_meta` arguments are always None
on the shipped path (no decoder FiLM is constructed by build_real_model).
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        if self.normtype == "layer":
            self.norm = nn.LayerNorm(out_c)
        else:
            raise ValueError(f"norm={norm!r} not shipped; use 'layer'")

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
        for i, layer in enumerate(self.deconv):
            x = layer(x)
            if film_layers is not None and pooled_meta is not None:
                x = film_layers[i](x, pooled_meta)
        return x.permute(0, 2, 1)  # [B, L, signal_dim]

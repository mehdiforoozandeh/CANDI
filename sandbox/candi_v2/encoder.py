"""CANDI v2 encoder — standalone, first-principles design.

Lineage: derived from sandbox/jepa_model.py JEPAEncoder (E21–E26 promoted
defaults), copied here so v2 can diverge independently from JEPA experiments.

Architecture (default config):
  1. MetadataEmbedding: per-assay (depth, assay_id, readlen, runtype) → embed_dim
  2. SignalConvTower: grouped Conv1d + residual + MaxPool, FiLM at each layer
  3. MaskTokenInjector: per-assay learned vectors replace masked conv channels
  4. DNAConvTower: ungrouped Conv1d on one-hot DNA
  5. LinearFusion / GatedFusion: merge signal + DNA features
  6. x-transformers encoder with RoPE + optional per-layer FiLM

Shapes (default 8 assays, context_length=768, n_cnn_layers=3, pool=2):
  Input:  x_signal [B, 768, 9]   x_dna [B, 4, 19200]   x_meta [B, 4, 9]
  Output: z        [B, 96, d_model]
"""
from __future__ import annotations

from typing import List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from _utils import exponential_linspace_int
from model import MaskStem
from sandbox.batch import CLOZE, MISSING
from sandbox.candi_v2.config import EncoderConfig

try:
    from x_transformers import Encoder as XEncoder
except ImportError as exc:
    raise ImportError(
        "x-transformers is required for sandbox.candi_v2. "
        "Install with: pip install x-transformers"
    ) from exc


# ---------------------------------------------------------------------------
# Metadata embedding
# ---------------------------------------------------------------------------

class MetadataEmbedding(nn.Module):
    """Per-assay metadata encoder with distinct missing/cloze tokens.

    Each of the 4 covariate fields (depth, assay_id, read_length, run_type)
    is independently projected to `embed_dim`, then fused via a 2-layer MLP.
    Continuous fields (depth, read_length) use learned embeddings for the
    MISSING (-1) and CLOZE (-2) sentinel values.  Categorical fields
    (assay_id, run_type) have dedicated embedding entries for sentinels.

    Output shape: [B, num_assays+1, embed_dim]  (one vector per assay+control)
    """

    def __init__(
        self,
        num_assays: int,
        embed_dim: int,
        num_runtypes: int = 2,
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()
        self.num_assays = int(num_assays)
        self.num_runtypes = int(num_runtypes)
        self.embed_dim = int(embed_dim)

        # Continuous covariates: linear projection + sentinel embeddings
        self.depth_proj = nn.Linear(1, embed_dim)
        self.read_length_proj = nn.Linear(1, embed_dim)
        self.depth_missing_emb = nn.Parameter(torch.randn(embed_dim) * 0.02)
        self.depth_cloze_emb = nn.Parameter(torch.randn(embed_dim) * 0.02)
        self.readlen_missing_emb = nn.Parameter(torch.randn(embed_dim) * 0.02)
        self.readlen_cloze_emb = nn.Parameter(torch.randn(embed_dim) * 0.02)

        # Categorical covariates: embedding tables with sentinel entries
        # assay_id: [0..num_assays-1] + MISSING + CLOZE
        self.assay_embedding = nn.Embedding(num_assays + 3, embed_dim)
        # run_type: [0..num_runtypes-1] + MISSING + CLOZE
        self.runtype_embedding = nn.Embedding(num_runtypes + 2, embed_dim)

        # Fuse 4 × embed_dim → embed_dim
        fusion_layers: List[nn.Module] = [
            nn.Linear(4 * embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        ]
        if bool(use_layernorm):
            fusion_layers.append(nn.LayerNorm(embed_dim))
        self.fusion = nn.Sequential(*fusion_layers)

    def _embed_continuous(
        self,
        values: torch.Tensor,
        proj: nn.Linear,
        missing_emb: nn.Parameter,
        cloze_emb: nn.Parameter,
    ) -> torch.Tensor:
        """Project continuous values; substitute sentinel embeddings for MISSING/CLOZE."""
        missing_mask = values == MISSING
        cloze_mask = values == CLOZE
        emb = proj(values.unsqueeze(-1).float())
        if missing_mask.any():
            emb[missing_mask] = missing_emb.to(emb.dtype)
        if cloze_mask.any():
            emb[cloze_mask] = cloze_emb.to(emb.dtype)
        return emb

    def forward(self, metadata: torch.Tensor) -> torch.Tensor:
        """
        Args:
            metadata: [B, 4, num_assays+1] — rows are (depth, assay_id, readlen, runtype)
        Returns:
            [B, num_assays+1, embed_dim]
        """
        depth = metadata[:, 0, :]
        assay_id = metadata[:, 1, :]
        read_length = metadata[:, 2, :]
        runtype = metadata[:, 3, :]

        depth_emb = self._embed_continuous(
            depth, self.depth_proj, self.depth_missing_emb, self.depth_cloze_emb
        )
        readlen_emb = self._embed_continuous(
            read_length, self.read_length_proj,
            self.readlen_missing_emb, self.readlen_cloze_emb,
        )

        # Map sentinel values to dedicated embedding indices
        assay_id_long = assay_id.long()
        assay_id_long = torch.where(
            assay_id_long == MISSING,
            torch.full_like(assay_id_long, self.num_assays + 1),
            assay_id_long,
        )
        assay_id_long = torch.where(
            assay_id_long == CLOZE,
            torch.full_like(assay_id_long, self.num_assays + 2),
            assay_id_long,
        )
        assay_emb = self.assay_embedding(assay_id_long)

        runtype_long = runtype.long()
        runtype_long = torch.where(
            runtype_long == MISSING,
            torch.full_like(runtype_long, self.num_runtypes),
            runtype_long,
        )
        runtype_long = torch.where(
            runtype_long == CLOZE,
            torch.full_like(runtype_long, self.num_runtypes + 1),
            runtype_long,
        )
        runtype_emb = self.runtype_embedding(runtype_long)

        concat = torch.cat([depth_emb, assay_emb, readlen_emb, runtype_emb], dim=-1)
        return self.fusion(concat)


# ---------------------------------------------------------------------------
# Conv building blocks
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Conv1d → Norm → optional GELU."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        norm: str,
        groups: int = 1,
        apply_act: bool = False,
    ) -> None:
        super().__init__()
        self.normtype = str(norm)
        self.apply_act = bool(apply_act)
        self.conv = nn.Conv1d(
            in_ch, out_ch,
            kernel_size=kernel_size, dilation=1, stride=1,
            groups=groups, padding="same",
        )
        if self.normtype == "batch":
            self.norm = nn.BatchNorm1d(out_ch)
        elif self.normtype == "layer":
            self.norm = nn.LayerNorm(out_ch)
        elif self.normtype == "group":
            self.norm = nn.GroupNorm(groups, out_ch)
        else:
            raise ValueError(f"Unsupported conv_norm={self.normtype}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.normtype == "layer":
            x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            x = self.norm(x)
        if self.apply_act:
            x = F.gelu(x)
        return x


class ConvTower(nn.Module):
    """Conv + residual skip (1×1 conv) + MaxPool."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        groups: int = 1,
        pool_size: int = 2,
        norm: str = "layer",
    ) -> None:
        super().__init__()
        self.conv1 = ConvBlock(in_ch, out_ch, kernel_size, norm=norm,
                               groups=groups, apply_act=False)
        self.rconv = nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=1, groups=groups)
        self.pool = nn.MaxPool1d(pool_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = F.gelu(y + self.rconv(x))
        return self.pool(y)


# ---------------------------------------------------------------------------
# FiLM conditioning layers
# ---------------------------------------------------------------------------

class FiLMLayer(nn.Module):
    """Per-assay FiLM in channel-first conv space.

    metadata_embed [B, A+1, embed_dim] → project to per-channel (scale, shift)
    and modulate conv activations: x ← x * (1 + scale) + shift.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.normal_(self.proj.bias, mean=0.0, std=0.1)

    def forward(self, x: torch.Tensor, metadata_embed: torch.Tensor) -> torch.Tensor:
        bsz, channels, _ = x.shape
        assays = metadata_embed.shape[1]
        if channels % assays != 0:
            raise ValueError(f"C % F != 0 for FiLMLayer. C={channels}, F={assays}")
        params = self.proj(metadata_embed)
        scale, shift = params.chunk(2, dim=-1)
        scale = scale.contiguous().view(bsz, channels).unsqueeze(-1)
        shift = shift.contiguous().view(bsz, channels).unsqueeze(-1)
        return x * (1.0 + scale) + shift


class PerAssayFiLM(nn.Module):
    """FiLM in sequence-last (B, L, C) space, applied per-assay."""

    def __init__(self, emb_dim: int, d_per_assay: int) -> None:
        super().__init__()
        self.d_per_assay = int(d_per_assay)
        self.proj = nn.Linear(int(emb_dim), 2 * int(d_per_assay))
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.normal_(self.proj.bias, mean=0.0, std=0.1)

    def forward(self, x: torch.Tensor, meta_embed: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, channels = x.shape
        assays = meta_embed.shape[1]
        d = self.d_per_assay
        if channels != assays * d:
            raise ValueError(
                f"PerAssayFiLM channel mismatch: C={channels}, expected {assays * d}"
            )
        x4 = x.view(bsz, seq_len, assays, d)
        scale, shift = self.proj(meta_embed).chunk(2, dim=-1)
        x4 = x4 * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return x4.view(bsz, seq_len, channels)


class TransformerFeatureFiLM(nn.Module):
    """FiLM on the fused d_model features, conditioned on pooled metadata."""

    def __init__(self, emb_dim: int, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(int(emb_dim), 2 * int(d_model))
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.normal_(self.proj.bias, mean=0.0, std=0.1)

    def forward(self, x: torch.Tensor, pooled_meta: torch.Tensor) -> torch.Tensor:
        scale, shift = self.proj(pooled_meta).chunk(2, dim=-1)
        return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ---------------------------------------------------------------------------
# Mask token injection
# ---------------------------------------------------------------------------

class MaskTokenInjector(nn.Module):
    """Replace conv features of masked assays with learned embeddings.

    After the signal conv tower, the output has shape [B, L2, num_tracks * d_per_assay]
    with grouped convolutions keeping each assay's channels independent.  For each
    assay flagged as CLOZE or MISSING in the availability vector, we replace that
    assay's d_per_assay-sized channel slice with its learned mask embedding.
    Available assays keep their real conv features.

    The full mask token is [num_tracks, d_per_assay] — effectively a d_model-sized
    vector partitioned by assay, where available assay slices are overwritten with
    real signal features.
    """

    def __init__(self, num_tracks: int, d_per_assay: int) -> None:
        super().__init__()
        self.num_tracks = int(num_tracks)
        self.d_per_assay = int(d_per_assay)
        self.mask_embedding = nn.Parameter(
            torch.randn(self.num_tracks, self.d_per_assay) * 0.02
        )

    def forward(self, x_conv: torch.Tensor, availability: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_conv:      [B, L2, num_tracks * d_per_assay] — grouped conv output
            availability: [B, num_tracks] — per-assay availability flags
        """
        bsz, seq_len, _ = x_conv.shape
        assays = availability.shape[1]
        if assays != self.num_tracks:
            raise ValueError(
                f"availability tracks ({assays}) != num_tracks ({self.num_tracks})"
            )
        x = x_conv.view(bsz, seq_len, assays, self.d_per_assay)
        replace = (availability == CLOZE) | (availability == MISSING)
        token = self.mask_embedding.view(1, 1, self.num_tracks, self.d_per_assay).to(x.dtype)
        x = torch.where(replace.unsqueeze(1).unsqueeze(-1), token, x)
        return x.view(bsz, seq_len, assays * self.d_per_assay)


# ---------------------------------------------------------------------------
# Signal and DNA conv towers
# ---------------------------------------------------------------------------

class SignalConvTower(nn.Module):
    """Grouped Conv1d tower for signal tracks with configurable FiLM.

    Uses grouped convolutions (groups=num_tracks) so each assay's channels
    are processed independently.  FiLM conditioning can be placed at various
    points depending on film_mode.
    """

    def __init__(
        self,
        num_tracks: int,
        n_layers: int,
        expansion_factor: int,
        kernel_size: int,
        pool_size: int,
        meta_embed_dim: int,
        conv_norm: str,
        film_mode: str,
    ) -> None:
        super().__init__()
        self.num_tracks = int(num_tracks)
        self.film_mode = str(film_mode)

        # Channel schedule: each layer expands by expansion_factor (grouped)
        conv_channels = [
            self.num_tracks * (int(expansion_factor) ** l)
            for l in range(int(n_layers))
        ]
        out_channels_list: List[int] = []
        blocks: List[nn.Module] = []
        for i in range(int(n_layers)):
            out_ch = (
                conv_channels[i + 1]
                if i + 1 < int(n_layers)
                else int(expansion_factor) * conv_channels[i]
            )
            out_channels_list.append(out_ch)
            blocks.append(ConvTower(
                in_ch=conv_channels[i], out_ch=out_ch,
                kernel_size=int(kernel_size), groups=self.num_tracks,
                pool_size=int(pool_size), norm=str(conv_norm),
            ))
        self.blocks = nn.ModuleList(blocks)
        self.out_channels = out_channels_list[-1]
        self.out_per_assay = self.out_channels // self.num_tracks

        # Build FiLM layers based on mode
        self.pre_film: Optional[FiLMLayer] = None
        self.post_film: Optional[PerAssayFiLM] = None
        self.per_conv_film_layers: Optional[nn.ModuleList] = None
        if self.film_mode == "pre_conv":
            self.pre_film = FiLMLayer(int(meta_embed_dim), 2)
        elif self.film_mode == "post_conv":
            self.post_film = PerAssayFiLM(int(meta_embed_dim), self.out_per_assay)
        elif self.film_mode in ("per_conv", "per_conv_and_transformer"):
            self.per_conv_film_layers = nn.ModuleList([
                FiLMLayer(int(meta_embed_dim), 2 * (ch // self.num_tracks))
                for ch in out_channels_list
            ])

    def forward(self, x_signal: torch.Tensor, meta_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_signal:   [B, L, num_tracks] — raw signal (already transformed)
            meta_embed: [B, num_tracks, embed_dim]
        Returns:
            [B, L2, out_channels] where L2 = L // pool_size^n_layers
        """
        x = x_signal.permute(0, 2, 1).contiguous()  # [B, C, L]
        if self.pre_film is not None:
            x = self.pre_film(x, meta_embed)
        for i, block in enumerate(self.blocks):
            x = block(x)
            if self.per_conv_film_layers is not None:
                x = self.per_conv_film_layers[i](x, meta_embed)
        x = x.permute(0, 2, 1).contiguous()  # [B, L2, C]
        if self.post_film is not None:
            x = self.post_film(x, meta_embed)
        return x


class DNAConvTower(nn.Module):
    """Standard (ungrouped) Conv1d tower for one-hot DNA sequence.

    Downsamples DNA length to match signal tower output length via alternating
    small (pool_size) and large (dna_pool_size) pooling steps.
    """

    def __init__(
        self,
        target_dim: int,
        n_cnn_layers: int,
        conv_kernel_size: int,
        pool_size: int,
        dna_pool_size: int,
        conv_norm: str,
        pool_order: str,
    ) -> None:
        super().__init__()
        channels = [4] + list(
            exponential_linspace_int(4, int(target_dim), int(n_cnn_layers) + 2)
        )
        blocks: List[nn.Module] = []
        total = int(n_cnn_layers) + 2
        for i in range(total):
            if str(pool_order) == "late":
                use_large_pool = i >= int(n_cnn_layers)
            else:
                use_large_pool = i < 2
            p = int(dna_pool_size) if use_large_pool else int(pool_size)
            blocks.append(ConvTower(
                in_ch=channels[i], out_ch=channels[i + 1],
                kernel_size=int(conv_kernel_size), groups=1,
                pool_size=p, norm=str(conv_norm),
            ))
        self.blocks = nn.ModuleList(blocks)
        self.out_channels = channels[-1]

    def forward(self, x_dna: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_dna: [B, 4, G] or [B, G, 4] — one-hot DNA
        Returns:
            [B, L2, out_channels]
        """
        if x_dna.ndim != 3:
            raise ValueError(f"x_dna must be rank-3, got shape={tuple(x_dna.shape)}")
        if x_dna.shape[1] == 4:
            x = x_dna.float()
        elif x_dna.shape[2] == 4:
            x = x_dna.permute(0, 2, 1).contiguous().float()
        else:
            raise ValueError(f"x_dna must have a dim=4, got {tuple(x_dna.shape)}")
        for block in self.blocks:
            x = block(x)
        return x.permute(0, 2, 1).contiguous()


# ---------------------------------------------------------------------------
# Fusion layers
# ---------------------------------------------------------------------------

class LinearFusion(nn.Module):
    """Concatenate signal + DNA features → linear project → GELU → optional norm."""

    def __init__(
        self,
        signal_dim: int,
        dna_dim: int,
        out_dim: int,
        dropout: float,
        fusion_norm: str = "layer",
    ) -> None:
        super().__init__()
        self.fusion_proj = nn.Linear(signal_dim + dna_dim, out_dim)
        self.gelu = nn.GELU()
        if fusion_norm == "layer":
            self.norm: nn.Module = nn.LayerNorm(out_dim)
        elif fusion_norm == "none":
            self.norm = nn.Identity()
        else:
            raise ValueError(f"Unsupported fusion_norm={fusion_norm}")
        self.dropout = nn.Dropout(dropout)

    def forward(self, signal: torch.Tensor, dna: torch.Tensor) -> torch.Tensor:
        if signal.shape[:2] != dna.shape[:2]:
            raise ValueError(
                f"Fusion sequence mismatch: signal={tuple(signal.shape)}, dna={tuple(dna.shape)}"
            )
        fused = self.fusion_proj(torch.cat([signal, dna], dim=-1))
        return self.dropout(self.norm(self.gelu(fused)))


class GatedDNAFusion(nn.Module):
    """Gated fusion: signal * sigmoid(gate(dna)) + proj(dna)."""

    def __init__(
        self,
        signal_dim: int,
        dna_dim: int,
        dropout: float,
        fusion_norm: str = "layer",
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dna_dim, signal_dim)
        self.dna_proj = nn.Linear(dna_dim, signal_dim)
        self.gelu = nn.GELU()
        if fusion_norm == "layer":
            self.norm: nn.Module = nn.LayerNorm(signal_dim)
        elif fusion_norm == "none":
            self.norm = nn.Identity()
        else:
            raise ValueError(f"Unsupported fusion_norm={fusion_norm}")
        self.dropout = nn.Dropout(dropout)

    def forward(self, signal: torch.Tensor, dna: torch.Tensor) -> torch.Tensor:
        if signal.shape[:2] != dna.shape[:2]:
            raise ValueError(
                f"Fusion sequence mismatch: signal={tuple(signal.shape)}, dna={tuple(dna.shape)}"
            )
        gate = torch.sigmoid(self.gate_proj(dna))
        dna_contribution = self.dna_proj(dna)
        fused = signal * gate + dna_contribution
        return self.dropout(self.norm(self.gelu(fused)))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_availability_from_meta(meta: torch.Tensor) -> torch.Tensor:
    """Derive per-assay availability [B, A+1] from metadata [B, 4, A+1]."""
    has_cloze = (meta == CLOZE).any(dim=1)
    has_missing = (meta == MISSING).any(dim=1)
    avail = torch.ones_like(meta[:, 0, :], dtype=meta.dtype)
    avail = torch.where(has_missing, torch.full_like(avail, MISSING), avail)
    avail = torch.where(has_cloze, torch.full_like(avail, CLOZE), avail)
    return avail


def _infer_availability_from_signal(x_signal: torch.Tensor) -> torch.Tensor:
    """Derive per-assay availability [B, A+1] from signal [B, L, A+1]."""
    has_cloze = (x_signal == CLOZE).any(dim=1)
    has_missing = (x_signal == MISSING).any(dim=1)
    avail = torch.ones_like(x_signal[:, 0, :], dtype=x_signal.dtype)
    avail = torch.where(has_missing, torch.full_like(avail, MISSING), avail)
    avail = torch.where(has_cloze, torch.full_like(avail, CLOZE), avail)
    return avail


def _apply_signal_transform(x: torch.Tensor, mode: str) -> torch.Tensor:
    """Apply input signal compression, preserving sentinel values."""
    mode_l = str(mode).lower()
    if mode_l == "none":
        return x
    mask = (x != MISSING) & (x != CLOZE)
    if mode_l == "log1p":
        return torch.where(mask, torch.log1p(x), x)
    if mode_l == "arcsinh":
        return torch.where(mask, torch.asinh(x), x)
    raise ValueError(f"Unsupported signal_transform={mode}")


def _get_divisible_heads(dim: int, preferred_heads: int) -> int:
    """Find largest head count <= preferred_heads that divides dim."""
    for h in range(preferred_heads, 0, -1):
        if dim % h == 0:
            return h
    return 1


# ---------------------------------------------------------------------------
# Dual attention (for transformer_type="dual")
# ---------------------------------------------------------------------------

class RelativePositionBias(nn.Module):
    def __init__(self, num_heads: int, max_distance: int) -> None:
        super().__init__()
        self.max_distance = int(max_distance)
        self.relative_bias = nn.Parameter(
            torch.zeros(2 * self.max_distance - 1, num_heads)
        )
        nn.init.trunc_normal_(self.relative_bias, std=0.02)

    def forward(self, seq_len: int) -> torch.Tensor:
        pos = torch.arange(seq_len, device=self.relative_bias.device)
        rel_pos = pos[None, :] - pos[:, None] + self.max_distance - 1
        return self.relative_bias[rel_pos].permute(2, 0, 1).contiguous()


class DualAttentionEncoderBlock(nn.Module):
    """Sequence attention + channel attention + FFN with residual connections."""

    def __init__(
        self, d_model: int, num_heads: int, seq_length: int, dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.dropout = float(dropout)
        self.num_heads = _get_divisible_heads(self.d_model, int(num_heads))
        self.num_heads_chan = _get_divisible_heads(int(seq_length), int(num_heads))
        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.relative_bias = RelativePositionBias(
            self.num_heads, max(2, int(seq_length))
        )
        self.mha_channel = nn.MultiheadAttention(
            embed_dim=int(seq_length), num_heads=self.num_heads_chan,
            dropout=self.dropout, batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(2 * self.d_model, 2 * self.d_model),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(2 * self.d_model, self.d_model),
            nn.Dropout(self.dropout),
        )
        self.norm_seq = nn.LayerNorm(self.d_model)
        self.norm_chan = nn.LayerNorm(self.d_model)
        self.norm_ffn = nn.LayerNorm(self.d_model)

    def _relative_multihead_attention(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        head_dim = self.d_model // self.num_heads
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(float(head_dim), device=x.device, dtype=x.dtype)
        )
        scores = scores + self.relative_bias(seq_len).unsqueeze(0)
        attn_weights = F.dropout(
            F.softmax(scores, dim=-1), p=self.dropout, training=self.training
        )
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.out_proj(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_attn = self._relative_multihead_attention(x)
        x_seq = self.norm_seq(x + seq_attn)
        x_trans = x.transpose(1, 2)
        chan_attn, _ = self.mha_channel(x_trans, x_trans, x_trans)
        x_chan = self.norm_chan(x + chan_attn.transpose(1, 2))
        ffn_out = self.ffn(torch.cat([x_seq, x_chan], dim=-1))
        return self.norm_ffn(x_seq + x_chan + ffn_out)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class V2Encoder(nn.Module):
    """CANDI v2 encoder: signal conv → mask inject → DNA conv → fuse → transformer.

    This is a standalone module that takes raw signal, DNA, and metadata and
    produces a latent representation of shape [B, L2, d_model].
    """

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_tracks = int(cfg.num_assays) + 1  # +1 for control channel
        self.l1 = int(cfg.context_length)
        pool_factor = int(cfg.pool_size) ** int(cfg.n_cnn_layers)
        if self.l1 % pool_factor != 0:
            raise ValueError(
                f"context_length={self.l1} not divisible by "
                f"pool_size^n_cnn_layers={pool_factor}"
            )
        self.l2 = self.l1 // pool_factor
        self.missing_data_mode = str(cfg.missing_data_mode)
        self.film_mode = str(cfg.film_mode)
        self.transformer_type = str(cfg.transformer_type)

        # -- Metadata embedding --
        self.metadata_embedding = MetadataEmbedding(
            num_assays=int(cfg.num_assays),
            embed_dim=int(cfg.metadata_embed_dim),
            use_layernorm=bool(cfg.meta_embed_layernorm),
        )

        # -- Signal conv tower --
        self.signal_tower = SignalConvTower(
            num_tracks=self.num_tracks,
            n_layers=int(cfg.n_cnn_layers),
            expansion_factor=int(cfg.expansion_factor),
            kernel_size=int(cfg.conv_kernel_size),
            pool_size=int(cfg.pool_size),
            meta_embed_dim=int(cfg.metadata_embed_dim),
            conv_norm=str(cfg.conv_norm),
            film_mode=self.film_mode,
        )

        # -- Missing data handling --
        self.mask_stem: Optional[MaskStem] = None
        self.mask_injector: Optional[MaskTokenInjector] = None
        if self.missing_data_mode == "mask_stem":
            self.mask_stem = MaskStem(
                n_channels=self.num_tracks, sentinels=(MISSING, CLOZE)
            )
        elif self.missing_data_mode == "mask_token":
            self.mask_injector = MaskTokenInjector(
                self.num_tracks, self.signal_tower.out_per_assay
            )
        else:
            raise ValueError(f"Unsupported missing_data_mode={self.missing_data_mode}")

        # -- DNA conv tower --
        signal_dim = self.signal_tower.out_channels
        self.d_model = int(cfg.d_model) if int(cfg.d_model) > 0 else signal_dim
        self.dna_tower = DNAConvTower(
            target_dim=signal_dim,
            n_cnn_layers=int(cfg.n_cnn_layers),
            conv_kernel_size=int(cfg.conv_kernel_size),
            pool_size=int(cfg.pool_size),
            dna_pool_size=int(cfg.dna_pool_size),
            conv_norm=str(cfg.conv_norm),
            pool_order=str(cfg.dna_pool_order),
        )

        # -- Fusion --
        fusion_mode = str(cfg.fusion_mode)
        fusion_norm = str(cfg.fusion_norm)
        if fusion_mode == "gated":
            if self.d_model != signal_dim:
                raise ValueError(
                    f"gated fusion requires d_model == signal_dim; "
                    f"got {self.d_model} vs {signal_dim}"
                )
            self.fusion: nn.Module = GatedDNAFusion(
                signal_dim=signal_dim, dna_dim=self.dna_tower.out_channels,
                dropout=float(cfg.dropout), fusion_norm=fusion_norm,
            )
        elif fusion_mode == "linear":
            self.fusion = LinearFusion(
                signal_dim=signal_dim, dna_dim=self.dna_tower.out_channels,
                out_dim=self.d_model, dropout=float(cfg.dropout),
                fusion_norm=fusion_norm,
            )
        else:
            raise ValueError(f"Unsupported fusion_mode={fusion_mode}")

        # -- Transformer stack --
        if self.transformer_type == "dual":
            self.transformer_blocks = nn.ModuleList([
                DualAttentionEncoderBlock(
                    d_model=self.d_model, num_heads=int(cfg.nhead),
                    seq_length=self.l2, dropout=float(cfg.dropout),
                )
                for _ in range(int(cfg.n_transformer_layers))
            ])
        elif self.transformer_type == "xtransformers":
            self.transformer_blocks = nn.ModuleList([
                XEncoder(
                    dim=self.d_model, depth=1, heads=int(cfg.nhead),
                    rotary_pos_emb=True,
                    attn_dropout=float(cfg.dropout),
                    ff_dropout=float(cfg.dropout),
                    ff_mult=4, pre_norm=True,
                )
                for _ in range(int(cfg.n_transformer_layers))
            ])
        elif self.transformer_type == "production_dual":
            from model import DualAttentionEncoderBlock as ProdDualBlock
            self.transformer_blocks = nn.ModuleList([
                ProdDualBlock(
                    d_model=self.d_model, num_heads=int(cfg.nhead),
                    seq_length=self.l2, dropout=float(cfg.dropout),
                )
                for _ in range(int(cfg.n_transformer_layers))
            ])
        else:
            raise ValueError(f"Unsupported transformer_type={self.transformer_type}")

        # -- Optional per-transformer-layer FiLM --
        self.transformer_film_layers: Optional[nn.ModuleList] = None
        if self.film_mode == "per_conv_and_transformer":
            self.transformer_film_layers = nn.ModuleList([
                TransformerFeatureFiLM(int(cfg.metadata_embed_dim), self.d_model)
                for _ in range(int(cfg.n_transformer_layers))
            ])

    def _prepare_signal(
        self, x_signal_t: torch.Tensor, x_meta: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Zero out masked signal channels; return (prepared_signal, availability)."""
        availability_meta = _infer_availability_from_meta(x_meta)
        if self.missing_data_mode != "mask_token":
            return x_signal_t, availability_meta

        availability_signal = _infer_availability_from_signal(x_signal_t)
        if not torch.equal(availability_meta, availability_signal):
            raise ValueError(
                "Signal vs metadata assay availability mismatch. "
                "CANDI v2 requires assay-only masking (p_full_loci=0, p_chunks=0) "
                "when missing_data_mode=mask_token."
            )
        observed = (availability_meta != CLOZE) & (availability_meta != MISSING)
        x_zeroed = x_signal_t * observed.unsqueeze(1).to(x_signal_t.dtype)
        return x_zeroed, availability_meta

    def encode(
        self,
        x_signal: torch.Tensor,
        x_dna: torch.Tensor,
        x_meta: torch.Tensor,
        return_meta: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_signal: [B, L, A+1] — signal + control channel
            x_dna:    [B, 4, G]   — one-hot DNA
            x_meta:   [B, 4, A+1] — metadata
            return_meta: if True, also return metadata embeddings

        Returns:
            z: [B, L2, d_model]
            (z, meta_embed) if return_meta=True
        """
        x_signal_t = _apply_signal_transform(x_signal.float(), self.cfg.signal_transform)
        x_signal_t, availability = self._prepare_signal(x_signal_t, x_meta)

        meta_embed = self.metadata_embedding(x_meta.float())

        # Signal conv tower
        sig_input = x_signal_t
        if self.mask_stem is not None:
            sig_input = self.mask_stem(
                sig_input.permute(0, 2, 1)
            ).permute(0, 2, 1).contiguous()
        sig = self.signal_tower(sig_input, meta_embed)

        # Mask token injection (after conv, before fusion)
        if self.mask_injector is not None:
            sig = self.mask_injector(sig, availability)

        # DNA conv tower
        dna = self.dna_tower(x_dna)

        # Fuse signal + DNA
        fused = self.fusion(sig, dna)

        # Transformer stack with optional per-layer FiLM
        pooled_meta = meta_embed.mean(dim=1)
        for i, block in enumerate(self.transformer_blocks):
            if self.transformer_film_layers is not None:
                fused = self.transformer_film_layers[i](fused, pooled_meta)
            fused = block(fused)

        if return_meta:
            return fused, meta_embed
        return fused

    def forward(
        self,
        x_signal: torch.Tensor,
        x_dna: torch.Tensor,
        x_meta: torch.Tensor,
    ) -> torch.Tensor:
        return self.encode(x_signal, x_dna, x_meta, return_meta=False)

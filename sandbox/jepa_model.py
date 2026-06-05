"""Fresh JEPA model for sandbox experiments (E21).

This module intentionally avoids importing the production CANDI model to keep
encoder-only JEPA experiments lightweight and iteration-friendly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from _utils import exponential_linspace_int
from model import MaskStem
from sandbox.batch import CLOZE, MISSING
from sandbox.jepa import JEPAPredictor, JEPAProjector, SIGReg

try:
    from x_transformers import Encoder as XEncoder
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "x-transformers is required for sandbox.jepa_model. "
        "Install with: pip install x-transformers"
    ) from exc


@dataclass
class JEPAModelConfig:
    """Architecture config for the fresh JEPA model."""

    num_assays: int = 8
    context_length: int = 768
    metadata_embed_dim: int = 32
    n_cnn_layers: int = 3
    expansion_factor: int = 2
    conv_kernel_size: int = 3
    pool_size: int = 2
    dna_pool_size: int = 5
    n_transformer_layers: int = 2
    nhead: int = 4
    dropout: float = 0.1
    d_model: int = 0  # 0 = auto (signal tower output dim)
    proj_dim: int = 0  # 0 = auto (encoder output dim)
    proj_hidden_dim: int = 256
    pred_hidden_dim: int = 0  # 0 = proj_dim
    pred_proj_hidden_dim: int = 0  # 0 = proj_hidden_dim
    predictor_layers: int = 1
    predictor_heads: int = 4
    predictor_dim_head: int = 64
    predictor_ff_mult: int = 4
    predictor_type: str = "transformer"  # compatibility with train_jepa fresh config
    cond_source: str = "meta_tgt_embed"  # promoted 2026-05-18 (separate no-LN embed for predictor)
    cond_embed_shared: str = "separate"  # promoted 2026-05-18 (encoder LN + predictor no-LN)
    lambda_sigreg: float = 0.1
    sigreg_num_proj: int = 1024
    sigreg_knots: int = 17
    signal_transform: str = "log1p"  # none|log1p|arcsinh
    meta_embed_layernorm: bool = True  # encoder MetadataEmbedding keeps LN
    pred_meta_embed_layernorm: bool = False  # predictor's separate MetadataEmbedding: no LN (promoted 2026-05-18)
    missing_data_mode: Literal["mask_stem", "mask_token"] = "mask_token"  # promoted 2026-05-21 (E24: +2.6% combined_loss, +37% runtype_sens vs mask_stem control)
    fusion_mode: Literal["linear", "gated"] = "linear"
    fusion_norm: Literal["layer", "none"] = "none"  # promoted 2026-05-21 (E26: +1.1% combined_loss vs layer control; transformer has its own pre-norm)
    film_mode: Literal["per_conv", "per_conv_and_transformer", "post_conv", "pre_conv"] = "per_conv_and_transformer"  # promoted 2026-05-18 (best metadata retention, E23 batch 1)
    conv_norm: Literal["layer", "group", "batch"] = "layer"
    dna_pool_order: Literal["late", "early"] = "late"
    transformer_type: Literal["dual", "xtransformers", "production_dual"] = "xtransformers"  # promoted 2026-05-16 (E23: −17% pred_loss, stacks with pre_conv)


class MetadataEmbedding(nn.Module):
    """Per-assay metadata encoder with distinct missing/cloze tokens."""

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
        self.depth_proj = nn.Linear(1, embed_dim)
        self.read_length_proj = nn.Linear(1, embed_dim)
        self.depth_missing_emb = nn.Parameter(torch.randn(embed_dim) * 0.02)
        self.depth_cloze_emb = nn.Parameter(torch.randn(embed_dim) * 0.02)
        self.readlen_missing_emb = nn.Parameter(torch.randn(embed_dim) * 0.02)
        self.readlen_cloze_emb = nn.Parameter(torch.randn(embed_dim) * 0.02)
        self.assay_embedding = nn.Embedding(num_assays + 3, embed_dim)
        self.runtype_embedding = nn.Embedding(num_runtypes + 2, embed_dim)
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
        missing_mask = values == MISSING
        cloze_mask = values == CLOZE
        emb = proj(values.unsqueeze(-1).float())
        if missing_mask.any():
            emb[missing_mask] = missing_emb.to(emb.dtype)
        if cloze_mask.any():
            emb[cloze_mask] = cloze_emb.to(emb.dtype)
        return emb

    def forward(self, metadata: torch.Tensor) -> torch.Tensor:
        depth = metadata[:, 0, :]
        assay_id = metadata[:, 1, :]
        read_length = metadata[:, 2, :]
        runtype = metadata[:, 3, :]

        depth_emb = self._embed_continuous(
            depth, self.depth_proj, self.depth_missing_emb, self.depth_cloze_emb
        )
        readlen_emb = self._embed_continuous(
            read_length, self.read_length_proj, self.readlen_missing_emb, self.readlen_cloze_emb
        )

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


class ConvBlock(nn.Module):
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
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            dilation=1,
            stride=1,
            groups=groups,
            padding="same",
        )
        if self.normtype == "batch":
            self.norm = nn.BatchNorm1d(out_ch)
        elif self.normtype == "layer":
            self.norm = nn.LayerNorm(out_ch)
        elif self.normtype == "group":
            self.norm = nn.GroupNorm(groups, out_ch)
        else:
            raise ValueError(f"Unsupported conv_norm={self.normtype}; expected layer|group|batch")

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
        self.conv1 = ConvBlock(in_ch, out_ch, kernel_size, norm=norm, groups=groups, apply_act=False)
        self.rconv = nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=1, groups=groups)
        self.pool = nn.MaxPool1d(pool_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = F.gelu(y + self.rconv(x))
        return self.pool(y)


def _get_divisible_heads(dim: int, preferred_heads: int) -> int:
    for h in range(preferred_heads, 0, -1):
        if dim % h == 0:
            return h
    return 1


class RelativePositionBias(nn.Module):
    def __init__(self, num_heads: int, max_distance: int) -> None:
        super().__init__()
        self.max_distance = int(max_distance)
        self.relative_bias = nn.Parameter(torch.zeros(2 * self.max_distance - 1, num_heads))
        nn.init.trunc_normal_(self.relative_bias, std=0.02)

    def forward(self, seq_len: int) -> torch.Tensor:
        pos = torch.arange(seq_len, device=self.relative_bias.device)
        rel_pos = pos[None, :] - pos[:, None]
        rel_pos = rel_pos + self.max_distance - 1
        return self.relative_bias[rel_pos].permute(2, 0, 1).contiguous()


class DualAttentionEncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, seq_length: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.dropout = float(dropout)
        self.num_heads = _get_divisible_heads(self.d_model, int(num_heads))
        self.num_heads_chan = _get_divisible_heads(int(seq_length), int(num_heads))
        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.relative_bias = RelativePositionBias(self.num_heads, max(2, int(seq_length)))
        self.mha_channel = nn.MultiheadAttention(
            embed_dim=int(seq_length),
            num_heads=self.num_heads_chan,
            dropout=self.dropout,
            batch_first=True,
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
        attn_weights = F.dropout(F.softmax(scores, dim=-1), p=self.dropout, training=self.training)
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


class FiLMLayer(nn.Module):
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
            raise ValueError(f"PerAssayFiLM channel mismatch: got C={channels}, expected {assays * d}")
        x4 = x.view(bsz, seq_len, assays, d)
        scale, shift = self.proj(meta_embed).chunk(2, dim=-1)
        x4 = x4 * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return x4.view(bsz, seq_len, channels)


class TransformerFeatureFiLM(nn.Module):
    def __init__(self, emb_dim: int, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(int(emb_dim), 2 * int(d_model))
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.normal_(self.proj.bias, mean=0.0, std=0.1)

    def forward(self, x: torch.Tensor, pooled_meta: torch.Tensor) -> torch.Tensor:
        scale, shift = self.proj(pooled_meta).chunk(2, dim=-1)
        return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class MaskTokenInjector(nn.Module):
    def __init__(self, num_tracks: int, d_per_assay: int) -> None:
        super().__init__()
        self.num_tracks = int(num_tracks)
        self.d_per_assay = int(d_per_assay)
        self.mask_embedding = nn.Parameter(
            torch.randn(self.num_tracks, self.d_per_assay) * 0.02
        )

    def forward(self, x_conv: torch.Tensor, availability: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x_conv.shape
        assays = availability.shape[1]
        if assays != self.num_tracks:
            raise ValueError(
                f"availability tracks ({assays}) != configured num_tracks ({self.num_tracks})"
            )
        x = x_conv.view(bsz, seq_len, assays, self.d_per_assay)
        replace = (availability == CLOZE) | (availability == MISSING)
        token = self.mask_embedding.view(1, 1, self.num_tracks, self.d_per_assay).to(x.dtype)
        x = torch.where(
            replace.unsqueeze(1).unsqueeze(-1),
            token,
            x,
        )
        return x.view(bsz, seq_len, assays * self.d_per_assay)


class SignalConvTower(nn.Module):
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
        conv_channels = [self.num_tracks * (int(expansion_factor) ** l) for l in range(int(n_layers))]
        out_channels_list: List[int] = []
        blocks: List[nn.Module] = []
        for i in range(int(n_layers)):
            out_channels = (
                conv_channels[i + 1] if i + 1 < int(n_layers) else int(expansion_factor) * conv_channels[i]
            )
            out_channels_list.append(out_channels)
            blocks.append(
                ConvTower(
                    in_ch=conv_channels[i],
                    out_ch=out_channels,
                    kernel_size=int(kernel_size),
                    groups=self.num_tracks,
                    pool_size=int(pool_size),
                    norm=str(conv_norm),
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.out_channels = out_channels_list[-1]
        self.out_per_assay = self.out_channels // self.num_tracks
        self.pre_film: Optional[FiLMLayer] = None
        self.post_film: Optional[PerAssayFiLM] = None
        self.per_conv_film_layers: Optional[nn.ModuleList] = None
        if self.film_mode == "pre_conv":
            self.pre_film = FiLMLayer(int(meta_embed_dim), 2)
        elif self.film_mode == "post_conv":
            self.post_film = PerAssayFiLM(int(meta_embed_dim), self.out_per_assay)
        elif self.film_mode in ("per_conv", "per_conv_and_transformer"):
            self.per_conv_film_layers = nn.ModuleList(
                [
                    FiLMLayer(int(meta_embed_dim), 2 * (ch // self.num_tracks))
                    for ch in out_channels_list
                ]
            )

    def forward(self, x_signal: torch.Tensor, meta_embed: torch.Tensor) -> torch.Tensor:
        x = x_signal.permute(0, 2, 1).contiguous()
        if self.pre_film is not None:
            x = self.pre_film(x, meta_embed)
        for i, block in enumerate(self.blocks):
            x = block(x)
            if self.per_conv_film_layers is not None:
                x = self.per_conv_film_layers[i](x, meta_embed)
        x = x.permute(0, 2, 1).contiguous()
        if self.post_film is not None:
            x = self.post_film(x, meta_embed)
        return x


class DNAConvTower(nn.Module):
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
        channels = [4] + list(exponential_linspace_int(4, int(target_dim), int(n_cnn_layers) + 2))
        blocks: List[nn.Module] = []
        total = int(n_cnn_layers) + 2
        for i in range(total):
            if str(pool_order) == "late":
                use_large_pool = i >= int(n_cnn_layers)
            else:
                use_large_pool = i < 2
            p = int(dna_pool_size) if use_large_pool else int(pool_size)
            blocks.append(
                ConvTower(
                    in_ch=channels[i],
                    out_ch=channels[i + 1],
                    kernel_size=int(conv_kernel_size),
                    groups=1,
                    pool_size=p,
                    norm=str(conv_norm),
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.out_channels = channels[-1]

    def forward(self, x_dna: torch.Tensor) -> torch.Tensor:
        if x_dna.ndim != 3:
            raise ValueError(f"x_dna must be rank-3, got shape={tuple(x_dna.shape)}")
        if x_dna.shape[1] == 4:
            x = x_dna.float()
        elif x_dna.shape[2] == 4:
            x = x_dna.permute(0, 2, 1).contiguous().float()
        else:
            raise ValueError(f"x_dna shape must include channel dim=4, got {tuple(x_dna.shape)}")
        for block in self.blocks:
            x = block(x)
        return x.permute(0, 2, 1).contiguous()


class LinearFusion(nn.Module):
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
            raise ValueError(f"Unsupported fusion_norm={fusion_norm}; expected layer|none")
        self.dropout = nn.Dropout(dropout)

    def forward(self, signal: torch.Tensor, dna: torch.Tensor) -> torch.Tensor:
        if signal.shape[:2] != dna.shape[:2]:
            raise ValueError(
                f"Fusion sequence mismatch: signal={tuple(signal.shape)}, dna={tuple(dna.shape)}"
            )
        fused = self.fusion_proj(torch.cat([signal, dna], dim=-1))
        return self.dropout(self.norm(self.gelu(fused)))


class GatedDNAFusion(nn.Module):
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
            raise ValueError(f"Unsupported fusion_norm={fusion_norm}; expected layer|none")
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


def _infer_availability_from_meta(meta: torch.Tensor) -> torch.Tensor:
    has_cloze = (meta == CLOZE).any(dim=1)
    has_missing = (meta == MISSING).any(dim=1)
    avail = torch.ones_like(meta[:, 0, :], dtype=meta.dtype)
    avail = torch.where(has_missing, torch.full_like(avail, MISSING), avail)
    avail = torch.where(has_cloze, torch.full_like(avail, CLOZE), avail)
    return avail


def _infer_availability_from_signal(x_signal: torch.Tensor) -> torch.Tensor:
    has_cloze = (x_signal == CLOZE).any(dim=1)
    has_missing = (x_signal == MISSING).any(dim=1)
    avail = torch.ones_like(x_signal[:, 0, :], dtype=x_signal.dtype)
    avail = torch.where(has_missing, torch.full_like(avail, MISSING), avail)
    avail = torch.where(has_cloze, torch.full_like(avail, CLOZE), avail)
    return avail


def _apply_signal_transform(x: torch.Tensor, mode: str) -> torch.Tensor:
    mode_l = str(mode).lower()
    if mode_l == "none":
        return x
    mask = (x != MISSING) & (x != CLOZE)
    if mode_l == "log1p":
        return torch.where(mask, torch.log1p(x), x)
    if mode_l == "arcsinh":
        return torch.where(mask, torch.asinh(x), x)
    raise ValueError(f"Unsupported signal_transform={mode}")


class JEPAEncoder(nn.Module):
    """E23 ablation-ready fresh encoder."""

    def __init__(self, cfg: JEPAModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_tracks = int(cfg.num_assays) + 1
        self.l1 = int(cfg.context_length)
        pool_factor = int(cfg.pool_size) ** int(cfg.n_cnn_layers)
        if self.l1 % pool_factor != 0:
            raise ValueError(
                f"context_length={self.l1} must be divisible by pool_size**n_cnn_layers={pool_factor}"
            )
        self.l2 = self.l1 // pool_factor
        self.missing_data_mode = str(cfg.missing_data_mode)
        self.film_mode = str(cfg.film_mode)
        self.transformer_type = str(cfg.transformer_type)

        self.metadata_embedding = MetadataEmbedding(
            num_assays=int(cfg.num_assays),
            embed_dim=int(cfg.metadata_embed_dim),
            use_layernorm=bool(cfg.meta_embed_layernorm),
        )
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

        self.mask_stem: Optional[MaskStem] = None
        self.mask_injector: Optional[MaskTokenInjector] = None
        if self.missing_data_mode == "mask_stem":
            self.mask_stem = MaskStem(n_channels=self.num_tracks, sentinels=(MISSING, CLOZE))
        elif self.missing_data_mode == "mask_token":
            self.mask_injector = MaskTokenInjector(self.num_tracks, self.signal_tower.out_per_assay)
        else:
            raise ValueError(
                f"Unsupported missing_data_mode={self.missing_data_mode}; expected mask_stem|mask_token"
            )

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
        fusion_mode = str(cfg.fusion_mode)
        fusion_norm = str(cfg.fusion_norm)
        if fusion_mode == "gated":
            if self.d_model != signal_dim:
                raise ValueError(
                    f"gated fusion requires d_model == signal_dim; got {self.d_model} vs {signal_dim}"
                )
            self.fusion = GatedDNAFusion(
                signal_dim=signal_dim,
                dna_dim=self.dna_tower.out_channels,
                dropout=float(cfg.dropout),
                fusion_norm=fusion_norm,
            )
        elif fusion_mode == "linear":
            self.fusion = LinearFusion(
                signal_dim=signal_dim,
                dna_dim=self.dna_tower.out_channels,
                out_dim=self.d_model,
                dropout=float(cfg.dropout),
                fusion_norm=fusion_norm,
            )
        else:
            raise ValueError(f"Unsupported fusion_mode={fusion_mode}; expected linear|gated")
        if self.transformer_type == "dual":
            self.transformer_blocks = nn.ModuleList(
                [
                    DualAttentionEncoderBlock(
                        d_model=self.d_model,
                        num_heads=int(cfg.nhead),
                        seq_length=self.l2,
                        dropout=float(cfg.dropout),
                    )
                    for _ in range(int(cfg.n_transformer_layers))
                ]
            )
        elif self.transformer_type == "xtransformers":
            self.transformer_blocks = nn.ModuleList(
                [
                    XEncoder(
                        dim=self.d_model,
                        depth=1,
                        heads=int(cfg.nhead),
                        rotary_pos_emb=True,
                        attn_dropout=float(cfg.dropout),
                        ff_dropout=float(cfg.dropout),
                        ff_mult=4,
                        pre_norm=True,
                    )
                    for _ in range(int(cfg.n_transformer_layers))
                ]
            )
        elif self.transformer_type == "production_dual":
            from model import DualAttentionEncoderBlock as ProductionDualBlock
            self.transformer_blocks = nn.ModuleList(
                [
                    ProductionDualBlock(
                        d_model=self.d_model,
                        num_heads=int(cfg.nhead),
                        seq_length=self.l2,
                        dropout=float(cfg.dropout),
                    )
                    for _ in range(int(cfg.n_transformer_layers))
                ]
            )
        else:
            raise ValueError(
                f"Unsupported transformer_type={self.transformer_type}; expected dual|xtransformers|production_dual"
            )

        self.transformer_film_layers: Optional[nn.ModuleList] = None
        if self.film_mode == "per_conv_and_transformer":
            self.transformer_film_layers = nn.ModuleList(
                [
                    TransformerFeatureFiLM(int(cfg.metadata_embed_dim), self.d_model)
                    for _ in range(int(cfg.n_transformer_layers))
                ]
            )

    def _prepare_signal(
        self, x_signal_t: torch.Tensor, x_meta: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        availability_meta = _infer_availability_from_meta(x_meta)
        if self.missing_data_mode != "mask_token":
            return x_signal_t, availability_meta
        availability_signal = _infer_availability_from_signal(x_signal_t)
        if not torch.equal(availability_meta.to(torch.int64), availability_signal.to(torch.int64)):
            mismatch = (availability_meta.to(torch.int64) != availability_signal.to(torch.int64)).sum().item()
            raise ValueError(
                "mask_token mode availability mismatch between metadata and signal; "
                f"num_mismatched_assay_flags={int(mismatch)}"
            )
        observed = (availability_meta != CLOZE) & (availability_meta != MISSING)
        x_zeroed = x_signal_t * observed.unsqueeze(1).to(x_signal_t.dtype)
        return x_zeroed, availability_meta

    def encode(
        self, x_signal: torch.Tensor, x_dna: torch.Tensor, x_meta: torch.Tensor, return_meta: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        x_signal_t = _apply_signal_transform(x_signal.float(), self.cfg.signal_transform)
        x_signal_t, availability = self._prepare_signal(x_signal_t, x_meta)
        meta_embed = self.metadata_embedding(x_meta.float())
        sig_input = x_signal_t
        if self.mask_stem is not None:
            sig_input = self.mask_stem(sig_input.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        sig = self.signal_tower(sig_input, meta_embed)
        if self.mask_injector is not None:
            sig = self.mask_injector(sig, availability)
        dna = self.dna_tower(x_dna)
        fused = self.fusion(sig, dna)
        pooled_meta = meta_embed.mean(dim=1)
        for i, block in enumerate(self.transformer_blocks):
            if self.transformer_film_layers is not None:
                fused = self.transformer_film_layers[i](fused, pooled_meta)
            fused = block(fused)
        if return_meta:
            return fused, meta_embed
        return fused

    def forward(self, x_signal: torch.Tensor, x_dna: torch.Tensor, x_meta: torch.Tensor) -> torch.Tensor:
        return self.encode(x_signal, x_dna, x_meta, return_meta=False)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale) + shift


class Attention(nn.Module):
    """LeWM-style attention block with non-causal SDPA."""

    def __init__(self, dim: int, heads: int = 4, dim_head: int = 64, dropout: float = 0.0) -> None:
        super().__init__()
        self.heads = int(heads)
        self.dim_head = int(dim_head)
        inner = self.heads * self.dim_head
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_k = nn.Linear(dim, inner, bias=False)
        self.to_v = nn.Linear(dim, inner, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        b, n, _ = x.shape
        q = self.to_q(x).view(b, n, self.heads, self.dim_head).transpose(1, 2)
        k = self.to_k(x).view(b, n, self.heads, self.dim_head).transpose(1, 2)
        v = self.to_v(x).view(b, n, self.heads, self.dim_head).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=causal)
        out = out.transpose(1, 2).contiguous().view(b, n, self.heads * self.dim_head)
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConditionalBlock(nn.Module):
    """LeWM-style AdaLN-zero conditional transformer block."""

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa.unsqueeze(1), scale_msa.unsqueeze(1)),
            causal=False,
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp.unsqueeze(1), scale_mlp.unsqueeze(1))
        )
        return x


class JEPATransformerPredictor(nn.Module):
    """Conditional transformer predictor conditioned on flattened meta embeddings."""

    def __init__(
        self,
        proj_dim: int,
        hidden_dim: int,
        cond_dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        ff_mult: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(proj_dim, hidden_dim) if proj_dim != hidden_dim else nn.Identity()
        self.output_proj = nn.Linear(hidden_dim, proj_dim) if hidden_dim != proj_dim else nn.Identity()
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                ConditionalBlock(
                    dim=hidden_dim,
                    heads=heads,
                    dim_head=dim_head,
                    mlp_dim=hidden_dim * ff_mult,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self._last_adaLN_gamma_norm: float = 0.0
        self._last_adaLN_beta_norm: float = 0.0

    def forward(self, proj_ctx: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(proj_ctx)
        c = self.cond_proj(cond)
        for block in self.blocks:
            x = block(x, c)
            if hasattr(block, "adaLN_modulation"):
                with torch.no_grad():
                    shift_msa, scale_msa, _, shift_mlp, scale_mlp, _ = (
                        block.adaLN_modulation(c).chunk(6, dim=-1)
                    )
                    gamma = torch.cat([scale_msa, scale_mlp], dim=-1)
                    beta = torch.cat([shift_msa, shift_mlp], dim=-1)
                    self._last_adaLN_gamma_norm = float(gamma.float().norm(p=2).item())
                    self._last_adaLN_beta_norm = float(beta.float().norm(p=2).item())
        x = self.norm(x)
        return self.output_proj(x)


class JEPAModel(nn.Module):
    """Fresh JEPA model with a standalone encoder and LeWM-style predictor."""

    def __init__(self, cfg: JEPAModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.cond_source = str(cfg.cond_source)
        self.cond_embed_shared = str(cfg.cond_embed_shared)
        self.encoder = JEPAEncoder(cfg)
        self.candi = self.encoder
        self.encoder_out_dim = self.encoder.d_model
        proj_dim = int(cfg.proj_dim) if int(cfg.proj_dim) > 0 else self.encoder_out_dim
        pred_hidden_dim = int(cfg.pred_hidden_dim) if int(cfg.pred_hidden_dim) > 0 else proj_dim
        pred_proj_hidden_dim = (
            int(cfg.pred_proj_hidden_dim) if int(cfg.pred_proj_hidden_dim) > 0 else int(cfg.proj_hidden_dim)
        )
        self.proj_dim = proj_dim
        self.lambda_sigreg = float(cfg.lambda_sigreg)
        self.jepa_projector = JEPAProjector(self.encoder_out_dim, int(cfg.proj_hidden_dim), proj_dim)
        self.jepa_pred_projector = JEPAProjector(proj_dim, pred_proj_hidden_dim, proj_dim)

        num_assays_plus_ctrl = int(cfg.num_assays) + 1
        if self.cond_source == "raw_meta_tgt":
            cond_dim = 4 * num_assays_plus_ctrl
        else:
            cond_dim = num_assays_plus_ctrl * int(cfg.metadata_embed_dim)

        self.pred_metadata_embedding: Optional[MetadataEmbedding] = None
        if self.cond_source != "raw_meta_tgt" and self.cond_embed_shared == "separate":
            self.pred_metadata_embedding = MetadataEmbedding(
                num_assays=int(cfg.num_assays),
                embed_dim=int(cfg.metadata_embed_dim),
                use_layernorm=bool(cfg.pred_meta_embed_layernorm),
            )

        predictor_type = str(cfg.predictor_type)
        if predictor_type == "legacy_mlp":
            self.jepa_predictor: nn.Module = JEPAPredictor(
                proj_dim=proj_dim,
                hidden_dim=pred_hidden_dim,
                mask_cond_dim=cond_dim,
                use_mask_cond=True,
            )
        else:
            self.jepa_predictor = JEPATransformerPredictor(
                proj_dim=proj_dim,
                hidden_dim=pred_hidden_dim,
                cond_dim=cond_dim,
                depth=int(cfg.predictor_layers),
                heads=int(cfg.predictor_heads),
                dim_head=int(cfg.predictor_dim_head),
                ff_mult=int(cfg.predictor_ff_mult),
                dropout=float(cfg.dropout),
            )
        self.sigreg = SIGReg(knots=int(cfg.sigreg_knots), num_proj=int(cfg.sigreg_num_proj))

    def _compute_predictor_cond(
        self, meta_tgt: torch.Tensor, meta_tgt_embed: torch.Tensor
    ) -> torch.Tensor:
        if self.cond_source == "raw_meta_tgt":
            return meta_tgt.reshape(meta_tgt.shape[0], -1)
        if self.pred_metadata_embedding is not None:
            return self.pred_metadata_embedding(meta_tgt.float()).reshape(meta_tgt.shape[0], -1)
        return meta_tgt_embed.reshape(meta_tgt_embed.shape[0], -1)

    def forward(
        self,
        x_ctx: torch.Tensor,
        x_tgt: torch.Tensor,
        x_dna: torch.Tensor,
        meta_ctx: torch.Tensor,
        meta_tgt: torch.Tensor,
        mask_cond: torch.Tensor,  # kept for interface compatibility
    ) -> Dict[str, torch.Tensor]:
        del mask_cond
        z_ctx_raw = self.encoder.encode(x_ctx, x_dna, meta_ctx)
        z_tgt_raw, meta_tgt_embed = self.encoder.encode(x_tgt, x_dna, meta_tgt, return_meta=True)
        proj_ctx = self.jepa_projector(z_ctx_raw)
        proj_tgt = self.jepa_projector(z_tgt_raw)
        cond = self._compute_predictor_cond(meta_tgt, meta_tgt_embed)
        z_pred_raw = self.jepa_predictor(proj_ctx, cond)
        z_pred = self.jepa_pred_projector(z_pred_raw)
        pred_loss = F.mse_loss(z_pred, proj_tgt)
        proj_all = torch.cat([proj_ctx, proj_tgt], dim=0)
        sigreg_out = self.sigreg(proj_all.transpose(0, 1), return_stats=True)
        assert isinstance(sigreg_out, tuple)
        sigreg_loss, sigreg_stats = sigreg_out
        total_loss = pred_loss + self.lambda_sigreg * sigreg_loss
        combined_loss_scaled = total_loss / (max(self.lambda_sigreg, 1e-8) ** 0.4)
        embedding_mean_norm = z_tgt_raw.detach().float().mean(dim=(0, 1)).norm(p=2)
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
            "z_tgt_raw": z_tgt_raw,
            "z_pred": z_pred,
            "cos_sim": cos_sim,
            "adaLN_gamma_norm": self.jepa_predictor._last_adaLN_gamma_norm,
            "adaLN_beta_norm": self.jepa_predictor._last_adaLN_beta_norm,
            "sigreg_projection_std": sigreg_stats["sigreg_projection_std"],
        }

"""JEPA Stage 2 decoder training modules.

This module loads a pretrained JEPA encoder+predictor and attaches three
independent latent-to-signal decoder towers.  The decoders intentionally have no
metadata FiLM: target metadata conditioning happens upstream in the predictor.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from sandbox import SANDBOX_ASSAYS
from sandbox.batch import CLOZE, MISSING
from sandbox.jepa import CANDIJepa
from sandbox.jepa_config import JEPADecoderConfig
from sandbox.jepa_model import (
    JEPAModel,
    JEPAModelConfig as FreshJEPAModelConfig,
    JEPATransformerPredictor,
    MetadataEmbedding as FreshMetadataEmbedding,
)
from sandbox.model import build_sandbox_candi
from model import GaussianLayer, NegativeBinomialLayer, PeakLayer


class DeconvBlock(nn.Module):
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
            in_c,
            out_c,
            kernel_size=int(kernel_size),
            dilation=1,
            stride=int(stride),
            padding=padding,
            output_padding=output_padding,
            groups=int(groups),
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
            from model import RMSNorm

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
            in_c,
            out_c,
            kernel_size,
            stride,
            norm=norm,
            groups=groups,
        )
        self.rdeconv = nn.ConvTranspose1d(
            in_c,
            out_c,
            kernel_size=1,
            stride=int(stride),
            output_padding=int(stride) - 1,
            groups=int(groups),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.deconv(x) + self.rdeconv(x))


class JEPADecoderTower(nn.Module):
    """No-FiLM latent decoder: ``[B,L2,proj_dim] -> [B,L,F]``."""

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
                "decoder_input_dim must be divisible by signal_dim; "
                f"got {self.decoder_input_dim} and {self.signal_dim}"
            )
        self.input_proj = nn.Linear(self.proj_dim, self.decoder_input_dim)

        channels: List[int] = [self.decoder_input_dim]
        for i in range(1, int(n_cnn_layers) + 1):
            div = int(expansion_factor) ** i
            out_c = self.decoder_input_dim // div
            if i == int(n_cnn_layers):
                out_c = self.signal_dim
            if out_c < self.signal_dim:
                out_c = self.signal_dim
            channels.append(int(out_c))

        groups = self.signal_dim if bool(grouped) else 1
        if bool(grouped):
            for c in channels:
                if c % groups != 0:
                    raise ValueError(f"grouped decoder channel {c} not divisible by groups={groups}")

        self.deconv = nn.ModuleList(
            [
                DeconvTower(
                    channels[i],
                    channels[i + 1],
                    int(conv_kernel_size),
                    stride=int(pool_size),
                    groups=groups,
                    norm=norm,
                )
                for i in range(int(n_cnn_layers))
            ]
        )

    def forward(self, z_pred: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(z_pred).permute(0, 2, 1)
        for layer in self.deconv:
            x = layer(x)
        return x.permute(0, 2, 1)


def _build_stage1_model(cfg: JEPADecoderConfig, device: torch.device) -> nn.Module:
    signal_dim = len(SANDBOX_ASSAYS)
    meta_dim = int(cfg.model.metadata_embedding_dim_mult) * signal_dim
    if str(cfg.model_type) == "fresh":
        fresh_cfg = FreshJEPAModelConfig(
            num_assays=signal_dim,
            context_length=int(cfg.data.context_length),
            metadata_embed_dim=int(cfg.fresh.metadata_embed_dim),
            n_cnn_layers=int(cfg.fresh.n_cnn_layers),
            expansion_factor=int(cfg.fresh.expansion_factor),
            conv_kernel_size=int(cfg.fresh.conv_kernel_size),
            pool_size=int(cfg.fresh.pool_size),
            dna_pool_size=int(cfg.fresh.dna_pool_size),
            n_transformer_layers=int(cfg.fresh.n_transformer_layers),
            nhead=int(cfg.fresh.nhead),
            dropout=float(cfg.fresh.dropout),
            d_model=int(cfg.fresh.d_model),
            proj_dim=int(cfg.fresh.proj_dim),
            proj_hidden_dim=int(cfg.fresh.proj_hidden_dim),
            pred_hidden_dim=int(cfg.fresh.pred_hidden_dim),
            pred_proj_hidden_dim=int(cfg.fresh.pred_proj_hidden_dim),
            predictor_layers=int(cfg.fresh.predictor_layers),
            predictor_heads=int(cfg.fresh.predictor_heads),
            predictor_dim_head=int(cfg.fresh.predictor_dim_head),
            predictor_ff_mult=int(cfg.fresh.predictor_ff_mult),
            predictor_type=str(cfg.fresh.predictor_type),
            cond_source=str(cfg.fresh.cond_source),
            cond_embed_shared=str(cfg.fresh.cond_embed_shared),
            lambda_sigreg=float(cfg.fresh.lambda_sigreg),
            sigreg_num_proj=int(cfg.fresh.sigreg_num_proj),
            sigreg_knots=int(cfg.fresh.sigreg_knots),
            signal_transform=str(cfg.model.encode_input_transform),
            meta_embed_layernorm=bool(cfg.fresh.meta_embed_layernorm),
            pred_meta_embed_layernorm=bool(cfg.fresh.pred_meta_embed_layernorm),
            missing_data_mode=str(cfg.fresh.missing_data_mode),
            fusion_mode=str(cfg.fresh.fusion_mode),
            fusion_norm=str(cfg.fresh.fusion_norm),
            film_mode=str(cfg.fresh.film_mode),
            conv_norm=str(cfg.fresh.conv_norm),
            dna_pool_order=str(cfg.fresh.dna_pool_order),
            transformer_type=str(cfg.fresh.transformer_type),
        )
        return JEPAModel(fresh_cfg).to(device)

    candi = build_sandbox_candi(
        context_bins=int(cfg.data.context_length),
        signal_dim=signal_dim,
        metadata_embedding_dim=meta_dim,
        n_cnn_layers=int(cfg.model.n_cnn_layers),
        expansion_factor=int(cfg.model.expansion_factor),
        nhead=int(cfg.model.nhead),
        n_sab_layers=int(cfg.model.n_transformer_layers),
        dropout=float(cfg.model.dropout),
        separate_decoders=bool(cfg.model.separate_decoders),
        mask_stem=bool(cfg.model.mask_stem),
        dist_type="gaussian",
        signal_transform=str(cfg.model.encode_input_transform),
        linear_film=bool(cfg.model.linear_film),
        single_shot_decoder_film=bool(cfg.model.single_shot_decoder_film),
        gaussian_var_min=float(cfg.model.gaussian_var_min),
    )
    encoder_out_dim = int(candi.latent_projection[0].in_features)
    proj_dim = int(cfg.jepa.proj_dim) if int(cfg.jepa.proj_dim) > 0 else encoder_out_dim
    pred_hidden_dim = int(cfg.jepa.pred_hidden_dim) if int(cfg.jepa.pred_hidden_dim) > 0 else proj_dim
    injected_predictor = None
    pred_meta_embedding = None
    if str(cfg.jepa.predictor_type) == "fresh_transformer":
        if str(cfg.jepa.pred_cond_source) == "meta_tgt_embed":
            pred_meta_embedding = FreshMetadataEmbedding(
                num_assays=signal_dim,
                embed_dim=int(cfg.jepa.pred_meta_embed_dim),
                use_layernorm=bool(cfg.jepa.pred_meta_embed_layernorm),
            )
            cond_dim = (signal_dim + 1) * int(cfg.jepa.pred_meta_embed_dim)
        else:
            cond_dim = 4 * (signal_dim + 1)
        injected_predictor = JEPATransformerPredictor(
            proj_dim=proj_dim,
            hidden_dim=pred_hidden_dim,
            cond_dim=cond_dim,
            depth=int(cfg.jepa.predictor_layers),
            heads=int(cfg.jepa.predictor_heads),
            dim_head=int(cfg.jepa.predictor_dim_head),
            ff_mult=int(cfg.jepa.predictor_ff_mult),
            dropout=float(cfg.model.dropout),
        )

    return CANDIJepa(
        candi,
        proj_dim=int(cfg.jepa.proj_dim),
        proj_hidden_dim=int(cfg.jepa.proj_hidden_dim),
        pred_hidden_dim=int(cfg.jepa.pred_hidden_dim),
        pred_proj_hidden_dim=int(cfg.jepa.pred_proj_hidden_dim),
        num_assays=signal_dim,
        use_mask_cond=bool(cfg.jepa.pred_use_mask_cond),
        pred_mask_cond_type=str(cfg.jepa.pred_mask_cond_type),
        lambda_sigreg=float(cfg.jepa.lambda_sigreg),
        sigreg_num_proj=int(cfg.jepa.sigreg_num_proj),
        sigreg_knots=int(cfg.jepa.sigreg_knots),
        target_dsf=str(cfg.jepa.target_dsf),
        predictor=injected_predictor,
        pred_metadata_embedding=pred_meta_embedding,
    ).to(device)


class JEPADecoderModel(nn.Module):
    """CANDI-compatible model wrapper for JEPA Stage 2 decoder training."""

    def __init__(self, jepa_model: nn.Module, cfg: JEPADecoderConfig) -> None:
        super().__init__()
        self.jepa_model = jepa_model
        self.cfg = cfg
        self.signal_dim = len(SANDBOX_ASSAYS)
        self.heads = str(cfg.decoder.heads)
        self.freeze_mode = str(cfg.decoder.freeze_mode)
        self.pred_mask_cond_type = str(getattr(jepa_model, "pred_mask_cond_type", "meta_tgt"))
        self.encoder_out_dim = int(getattr(jepa_model, "encoder_out_dim"))
        self.proj_dim = int(getattr(jepa_model, "proj_dim"))
        default_dec_dim = self.signal_dim * (int(cfg.decoder.expansion_factor) ** int(cfg.decoder.n_cnn_layers))
        decoder_input_dim = int(cfg.decoder.decoder_input_dim) if int(cfg.decoder.decoder_input_dim) > 0 else default_dec_dim

        tower_kwargs = dict(
            proj_dim=self.proj_dim,
            signal_dim=self.signal_dim,
            decoder_input_dim=decoder_input_dim,
            n_cnn_layers=int(cfg.decoder.n_cnn_layers),
            expansion_factor=int(cfg.decoder.expansion_factor),
            pool_size=int(cfg.decoder.pool_size),
            conv_kernel_size=int(cfg.decoder.conv_kernel_size),
            grouped=bool(cfg.decoder.grouped_deconv),
            norm=str(cfg.decoder.norm),
        )
        self.count_decoder = JEPADecoderTower(**tower_kwargs)
        self.pval_decoder = JEPADecoderTower(**tower_kwargs)
        self.peak_decoder = JEPADecoderTower(**tower_kwargs)
        self.neg_binom_layer = NegativeBinomialLayer(self.signal_dim, self.signal_dim)
        self.signal_layer = GaussianLayer(
            self.signal_dim,
            self.signal_dim,
            var_min=float(cfg.decoder.gaussian_var_min),
        )
        self.peak_layer = PeakLayer(self.signal_dim, self.signal_dim)
        self.apply_freeze(self.freeze_mode)

    @classmethod
    def from_checkpoint(
        cls,
        cfg: JEPADecoderConfig,
        device: torch.device,
        *,
        strict: bool = True,
    ) -> "JEPADecoderModel":
        if not cfg.decoder.checkpoint_path:
            raise ValueError("decoder.checkpoint_path must be set")
        ckpt_path = Path(cfg.decoder.checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(ckpt_path)
        jepa_model = _build_stage1_model(cfg, device)
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = jepa_model.load_state_dict(state, strict=strict)
        if strict and (missing or unexpected):
            raise RuntimeError(f"checkpoint load mismatch: missing={missing}, unexpected={unexpected}")
        return cls(jepa_model, cfg).to(device)

    def encoder_modules(self) -> List[nn.Module]:
        if hasattr(self.jepa_model, "encoder"):
            return [self.jepa_model.encoder]
        return [self.jepa_model.candi]

    def predictor_modules(self) -> List[nn.Module]:
        modules: List[nn.Module] = [
            self.jepa_model.jepa_projector,
            self.jepa_model.jepa_predictor,
            self.jepa_model.jepa_pred_projector,
        ]
        pred_meta = getattr(self.jepa_model, "pred_metadata_embedding", None)
        if pred_meta is not None:
            modules.append(pred_meta)
        return modules

    def decoder_modules(self) -> List[nn.Module]:
        return [
            self.count_decoder,
            self.pval_decoder,
            self.peak_decoder,
            self.neg_binom_layer,
            self.signal_layer,
            self.peak_layer,
        ]

    def apply_freeze(self, freeze_mode: str) -> None:
        freeze_mode = str(freeze_mode)
        if freeze_mode not in {"decoder_only", "predictor_decoder", "encoder_decoder", "all"}:
            raise ValueError(f"unknown freeze_mode={freeze_mode}")
        train_encoder = freeze_mode in {"encoder_decoder", "all"}
        train_predictor = freeze_mode in {"predictor_decoder", "all"}
        for module in self.encoder_modules():
            for p in module.parameters():
                p.requires_grad = train_encoder
        for module in self.predictor_modules():
            for p in module.parameters():
                p.requires_grad = train_predictor
        for module in self.decoder_modules():
            for p in module.parameters():
                p.requires_grad = True
        self._upstream_trainable = bool(train_encoder or train_predictor)

    def active_heads(self) -> Tuple[str, ...]:
        if self.heads == "joint":
            return ("count", "pval", "peak")
        return (self.heads.replace("_only", ""),)

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return (p for p in self.parameters() if p.requires_grad)

    def _meta_tgt_with_control(self, x_meta: torch.Tensor, y_meta: torch.Tensor) -> torch.Tensor:
        return torch.cat([y_meta, x_meta[:, :, -1:].clone()], dim=2)

    def _mask_cond(self, x_ctx: torch.Tensor, meta_ctx: torch.Tensor, meta_tgt: torch.Tensor) -> torch.Tensor:
        bsz = x_ctx.shape[0]
        signal_ctx = x_ctx[:, :, : self.signal_dim]
        mode = str(getattr(self.jepa_model, "pred_mask_cond_type", self.pred_mask_cond_type))
        if mode == "meta_tgt":
            pred_meta = getattr(self.jepa_model, "pred_metadata_embedding", None)
            if pred_meta is not None:
                return pred_meta(meta_tgt.float()).reshape(bsz, -1)
            return meta_tgt.reshape(bsz, -1)
        if mode == "meta_concat":
            return torch.cat([meta_ctx.reshape(bsz, -1), meta_tgt.reshape(bsz, -1)], dim=-1)
        if mode == "loci":
            return (signal_ctx == CLOZE).any(dim=2).float()
        if mode == "none":
            return torch.zeros(bsz, self.signal_dim, device=x_ctx.device, dtype=x_ctx.dtype)
        meta_mask = ((meta_ctx[:, :, : self.signal_dim] == CLOZE) | (meta_ctx[:, :, : self.signal_dim] == MISSING)).any(dim=1)
        signal_mask = ((signal_ctx == CLOZE) | (signal_ctx == MISSING)).any(dim=1)
        return (meta_mask | signal_mask).float()

    def _predict_z(
        self,
        x_ctx: torch.Tensor,
        x_dna: torch.Tensor,
        meta_ctx: torch.Tensor,
        meta_tgt: torch.Tensor,
    ) -> torch.Tensor:
        jm = self.jepa_model
        if isinstance(jm, JEPAModel):
            z_ctx_raw = jm.encoder.encode(x_ctx, x_dna, meta_ctx)
            proj_ctx = jm.jepa_projector(z_ctx_raw)
            if str(jm.cond_source) == "raw_meta_tgt":
                cond = meta_tgt.reshape(meta_tgt.shape[0], -1)
            elif jm.pred_metadata_embedding is not None:
                cond = jm.pred_metadata_embedding(meta_tgt.float()).reshape(meta_tgt.shape[0], -1)
            else:
                cond = jm.encoder.metadata_embedding(meta_tgt.float()).reshape(meta_tgt.shape[0], -1)
            z_pred_raw = jm.jepa_predictor(proj_ctx, cond)
            return jm.jepa_pred_projector(z_pred_raw)

        z_ctx_raw = jm.candi.encode(x_ctx, x_dna, meta_ctx)
        proj_ctx = jm.jepa_projector(z_ctx_raw)
        mask_cond = self._mask_cond(x_ctx, meta_ctx, meta_tgt)
        if jm.pred_mask_cond_type == "loci" and mask_cond.ndim == 2:
            l2 = proj_ctx.shape[1]
            full_l = mask_cond.shape[1]
            pool = max(1, full_l // l2)
            mask_cond_pred = mask_cond.view(mask_cond.shape[0], l2, pool).max(dim=-1).values.unsqueeze(-1)
        else:
            mask_cond_pred = mask_cond
        z_pred_raw = jm.jepa_predictor(proj_ctx, mask_cond_pred)
        return jm.jepa_pred_projector(z_pred_raw)

    def forward(
        self,
        x_data: torch.Tensor,
        x_dna: torch.Tensor,
        x_meta: torch.Tensor,
        y_meta: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        query_mask_signal: Optional[torch.Tensor] = None,
    ):
        del query_mask, query_mask_signal
        meta_tgt = self._meta_tgt_with_control(x_meta, y_meta)
        with torch.set_grad_enabled(self._upstream_trainable and torch.is_grad_enabled()):
            z_pred = self._predict_z(x_data, x_dna, x_meta, meta_tgt)
        if not self._upstream_trainable:
            z_pred = z_pred.detach()

        count_decoded = self.count_decoder(z_pred)
        pval_decoded = self.pval_decoder(z_pred)
        peak_decoded = self.peak_decoder(z_pred)
        p, n = self.neg_binom_layer(count_decoded)
        mu, var = self.signal_layer(pval_decoded)
        peak = self.peak_layer(peak_decoded)
        return p, n, mu, var, None, peak


__all__ = ["JEPADecoderModel", "JEPADecoderTower", "DeconvTower", "DeconvBlock"]

"""Encoder configuration.

Provenance: EncoderConfig is a VERBATIM copy of EpiDenoise/sandbox/candi_v2/config.py:30-76.
DecoderConfig / CANDIv2TrainingConfig / CANDIv2Config / validate_v2_config are not shipped.

Note: num_metadata_rows (4) and num_runtypes (2) are deliberately NOT fields here. They are
fixed by the ingestion contract and fenced by asserts in encoder.MetadataEmbedding.forward.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


# ---------------------------------------------------------------------------
# Encoder config
# ---------------------------------------------------------------------------

@dataclass
class EncoderConfig:
    """Fresh encoder architecture (derived from JEPAEncoder promoted defaults)."""

    num_assays: int = 8
    context_length: int = 768
    metadata_embed_dim: int = 32
    meta_embed_layernorm: bool = True

    # Conv tower
    n_cnn_layers: int = 3
    expansion_factor: int = 2
    conv_kernel_size: int = 3
    pool_size: int = 2
    conv_norm: Literal["layer", "group", "batch"] = "layer"

    # DNA tower
    dna_pool_size: int = 5
    dna_pool_order: Literal["late", "early"] = "late"

    # Metadata FiLM conditioning placement
    film_mode: Literal[
        "pre_conv", "post_conv", "per_conv", "per_conv_and_transformer"
    ] = "per_conv_and_transformer"

    # Missing-data handling
    missing_data_mode: Literal["mask_stem", "mask_token"] = "mask_token"

    # Fusion of signal + DNA towers
    fusion_mode: Literal["linear", "gated"] = "linear"
    fusion_norm: Literal["layer", "none"] = "none"
    fusion_deep: bool = False  # 2-layer LinearFusion (extra Linear+GELU) instead of 1-layer

    # Transformer
    d_model: int = 0  # 0 = auto (signal tower output dim)
    n_transformer_layers: int = 2
    nhead: int = 4
    transformer_type: Literal[
        "dual", "xtransformers", "production_dual"
    ] = "xtransformers"
    dropout: float = 0.1
    attn_qk_norm: bool = False  # normalize Q/K vectors before attention (xtransformers)
    transformer_layer_drop: float = 0.0  # stochastic depth: drop transformer layers with this prob (train only)
    output_rms_norm: bool = False  # RMSNorm on encoder output [B, L2, d_model] before decoder

    # Input signal transform (applied internally by encoder)
    signal_transform: Literal["none", "log1p", "arcsinh"] = "log1p"

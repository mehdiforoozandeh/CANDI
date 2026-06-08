"""CANDI v2 configuration schema.

Hierarchical dataclass config with strict key rejection (same infra as
sandbox.config_types).  Merge order:
  dataclass defaults → candi_v2_default.yaml → extra --config overlays → CLI flags
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional

from sandbox.config_types import (
    DataConfig,
    DsfConfig,
    EvalConfig,
    GradConfig,
    HpoConfig,
    LossWeightsConfig,
    MaskingConfig,
    OptimizerConfig,
    ScheduleConfig,
    WandbConfig,
)


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

    # Transformer
    d_model: int = 0  # 0 = auto (signal tower output dim)
    n_transformer_layers: int = 2
    nhead: int = 4
    transformer_type: Literal[
        "dual", "xtransformers", "production_dual"
    ] = "xtransformers"
    dropout: float = 0.1
    attn_qk_norm: bool = False  # normalize Q/K vectors before attention (prevents score blowup)
    output_rms_norm: bool = False  # RMSNorm on encoder output [B, L2, d_model] before decoder

    # Input signal transform (applied internally by encoder)
    signal_transform: Literal["none", "log1p", "arcsinh"] = "log1p"

    # Cross-assay MHA after signal CNN + mask inject, before DNA fusion
    cross_assay_attention: bool = False

    # Stochastic depth: drop transformer layers with this prob during training
    transformer_layer_drop: float = 0.0


# ---------------------------------------------------------------------------
# Decoder config
# ---------------------------------------------------------------------------

@dataclass
class DecoderConfig:
    """Reconstruction decoder architecture."""

    # Which output heads to build
    heads: Literal["count_peak", "count_only", "peak_only", "all"] = "count_peak"

    # Trunk topology: shared deconv → split heads, or separate towers
    trunk: Literal["shared", "separate"] = "shared"

    # Pre-decoder FiLM on encoder latent (conditioned on target metadata)
    film_mode: Literal[
        "none", "single_pre_decoder", "per_deconv_layer"
    ] = "single_pre_decoder"

    # Decoder's own metadata embedding (separate from encoder)
    meta_embed_dim: int = 32
    meta_embed_layernorm: bool = False

    # Deconv tower geometry
    n_cnn_layers: int = 3
    expansion_factor: int = 2
    pool_size: int = 2
    conv_kernel_size: int = 3
    norm: Literal["layer", "batch", "group", "instance", "weight", "rms"] = "layer"

    # Grouped deconv (per-assay groups in ConvTranspose1d)
    grouped_deconv: bool = False

    # NB output layer
    gaussian_var_min: float = 0.1  # only used when heads="all"

    # Count-head parameterization (E29 / E30).
    #   "plain"        : NegativeBinomialLayer predicts mu directly (default).
    #   "depth_offset" : centered library-size offset, log2(mu) = (d - depth_center) + eta,
    #                    mu = 2^log2(mu); d = log2(seq_depth) from target metadata row 0.
    count_head: Literal["plain", "depth_offset"] = "depth_offset"
    depth_center: float = 22.5  # only used when count_head="depth_offset"
    mu_eps: float = 1e-6        # only used when count_head="depth_offset"
    learnable_depth_center: bool = False  # make depth_center an nn.Parameter
    learnable_depth_slope: bool = False   # make depth scaling an nn.Parameter (log-parameterized, init slope=1)
    depth_slope_init: float = 0.0        # initial log_depth_slope value (0=slope=1; set log(log2(3))≈0.461 for DCR=3 init)
    depth_slope: float = 1.0            # fixed depth slope when learnable_depth_slope=False (DCR=2^(slope*2); 0.795→DCR=3.013)
    depth_slope_constrained: bool = False  # sigmoid-map alpha to [log2(3)/2, log2(5)/2] → DCR∈[3,5] guaranteed; DCR_init=3.87
    learnable_depth_quadratic: bool = False  # quadratic depth term: beta*(d-center)^2 in log2_mu (+1 param, extends slope)
    grouped_dispersion: bool = False      # per-assay independent dispersion: replace linear_n(8→8) with Conv1d groups=8 (-56 params)
    diagonal_eta: bool = False           # per-assay eta head: replace linear_eta(8→8) with Conv1d groups=8 (-56 params, DCR-safe)
    encoder_skip: bool = False           # U-Net skip: add encoder pre-transformer features to decoder trunk input (+4672 params)


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class CANDIv2TrainingConfig:
    """Training knobs for CANDI v2."""

    epochs: int = 10
    batch_size: int = 16
    grad_accum_steps: int = 1
    seed: int = 42
    strict_determinism: bool = False
    device: Optional[str] = None
    run_dir: str = "."
    amp: bool = False
    augment_reverse_complement_prob: float = 0.0
    max_train_batches: Optional[int] = None
    eval_each_epoch: bool = True
    eval_max_batches: int = 0
    steps_per_epoch: int = 0
    training_stats_jsonl_every_n_steps: int = 200
    save_checkpoint: bool = False
    save_best_checkpoint: bool = False
    best_checkpoint_cooldown_epochs: int = 20
    early_stop_enabled: bool = False
    early_stop_patience: int = 5
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    grad: GradConfig = field(default_factory=GradConfig)
    masking: MaskingConfig = field(
        default_factory=lambda: MaskingConfig(
            p_full_assay=1.0,
            p_full_loci=0.0,
            p_chunks=0.0,
        )
    )
    loss_weights: LossWeightsConfig = field(default_factory=LossWeightsConfig)
    dsf: DsfConfig = field(default_factory=DsfConfig)


@dataclass
class CANDIv2Config:
    """Top-level config for CANDI v2 end-to-end training."""

    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: CANDIv2TrainingConfig = field(default_factory=CANDIv2TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    hpo: HpoConfig = field(default_factory=HpoConfig)


def validate_v2_config(cfg: CANDIv2Config) -> None:
    """Reject configs that are known to break CANDI v2 masking/availability invariants."""
    m = cfg.training.masking
    if float(m.p_full_loci) > 0.0 or float(m.p_chunks) > 0.0:
        raise ValueError(
            "CANDI v2 requires assay-only masking (p_full_loci=0, p_chunks=0). "
            "Position-level cloze is incompatible with mask_token availability logic."
        )
    if int(cfg.training.grad_accum_steps) < 1:
        raise ValueError("training.grad_accum_steps must be >= 1")
    if cfg.encoder.signal_transform not in ("none", "log1p", "arcsinh"):
        raise ValueError(f"unsupported encoder.signal_transform={cfg.encoder.signal_transform!r}")

"""Config schema for the CANDI JEPA training harness (sandbox/train_jepa.py).

Reuses the same config infrastructure as ``sandbox/config_types.py``
(``config_from_dict``, ``deep_merge``, etc.) but adds ``JEPAModelConfig``
and a top-level ``JEPAConfig`` that replaces ``SandboxConfig``.

Merge order (same as sandbox train.py):
  dataclass defaults → jepa_default.yaml → extra ``--config`` overlays → CLI dotted flags
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from sandbox.jepa_model import JEPAModelConfig as FreshJEPAModelConfig
from sandbox.config_types import (
    DataConfig,
    DsfConfig,
    EvalConfig,
    GradConfig,
    HpoConfig,
    LossWeightsConfig,
    MaskingConfig,
    ModelConfig,
    OptimizerConfig,
    ScheduleConfig,
    WandbConfig,
)


@dataclass
class JEPAModelConfig:
    """JEPA-specific model parameters (on top of the encoder defined by ModelConfig)."""

    # Projector output dim. 0 = use F2 (encoder output dim = (signal_dim+1)*expansion^n_layers).
    proj_dim: int = 0

    # Hidden dim of JEPAProjector MLP.  LeWM uses 2048; we scale to our smaller F2=72.
    proj_hidden_dim: int = 256

    # Hidden dim of JEPAPredictor MLP. 0 = use proj_dim.
    pred_hidden_dim: int = 0

    # Hidden dim of the predictor's output JEPAProjector.  0 = use proj_hidden_dim.
    pred_proj_hidden_dim: int = 0

    # Whether to use AdaLN-zero mask conditioning in the predictor.
    # Set to false to ablate: predictor ignores which assays were masked.
    pred_use_mask_cond: bool = True

    # Type of mask conditioning passed to the predictor's AdaLN.
    # "assay"       (default) — per-assay binary mask [B, F]; AdaLN dim = num_assays.
    # "loci"        — per-position binary mask [B, L], pooled to [B, L2, 1] in model;
    #                  requires training.masking.p_full_loci=1.0, p_full_assay=0.
    # "meta_concat" — flattened [ctx_meta ++ tgt_meta] [B, 8*(F+1)]; requires DSF
    #                  corruption setup (dsf_list=[4], no assay/loci masking).
    # "meta_tgt"    — flattened target metadata only [B, 4*(F+1)]; tells predictor
    #                  what metadata profile to reconstruct (task specification).
    #                  Works with both assay masking and DSF corruption. Uses y_meta
    #                  (DSF=1 metadata) as the target, so depth_log2 differs between
    #                  meta_ctx (DSF4) and meta_tgt (DSF1) in DSF corruption scenarios.
    # "none"        — AdaLN disabled; equivalent to pred_use_mask_cond=false.
    pred_mask_cond_type: Literal["assay", "loci", "meta_concat", "meta_tgt", "none"] = "assay"

    # SIGReg regularisation weight λ.
    # LeWM paper: 0.1; actual lewm.yaml: 0.09.  We start at 0.1 (paper default).
    lambda_sigreg: float = 0.1

    # Number of random projection directions M in SIGReg.
    # lewm.yaml: 1024 (= paper default M=1024).  MINIMAL.md hardcodes 256.
    # Paper ablation shows negligible impact on downstream performance.
    sigreg_num_proj: int = 1024

    # Number of quadrature nodes on [0, 3].
    sigreg_knots: int = 17

    # Target view construction.
    # "dsf1"  — use y_data (DSF=1 full-depth signal) as target.
    # "same"  — use x_data (pre-masking, same DSF as context) as target.
    target_dsf: Literal["dsf1", "same"] = "dsf1"

    # Predictor type for the candi-encoder branch.
    # "legacy_mlp"       — default JEPAPredictor (per-position 2-layer MLP + AdaLN-zero).
    # "fresh_transformer"— swap in JEPATransformerPredictor from jepa_model.py (E21 ablation).
    predictor_type: str = "legacy_mlp"

    # Architecture for fresh_transformer predictor (only used when predictor_type=fresh_transformer).
    predictor_layers: int = 1
    predictor_heads: int = 4
    predictor_dim_head: int = 64
    predictor_ff_mult: int = 4

    # Predictor conditioning source (only used when predictor_type=fresh_transformer).
    # "raw_meta_tgt"    — raw flattened target metadata (legacy E21 default).
    # "meta_tgt_embed"  — separate MetadataEmbedding (no LN) for proper categorical handling.
    pred_cond_source: str = "meta_tgt_embed"  # promoted 2026-05-18
    pred_meta_embed_dim: int = 32
    pred_meta_embed_layernorm: bool = False  # predictor embed: no LN (promoted 2026-05-18)


@dataclass
class JEPATrainingConfig:
    """Training knobs for the JEPA harness (mirrors TrainingConfig)."""

    epochs: int = 10
    batch_size: int = 8
    grad_accum_steps: int = 1
    seed: int = 42
    device: Optional[str] = None
    run_dir: str = "."
    amp: bool = True          # bf16/fp16 AMP — LeWM trains in bf16

    max_train_batches: Optional[int] = None

    # Cadence (steps) at which geometry metrics are logged.  0 = every step.
    geometry_log_every_n_steps: int = 50

    # Cadence (steps) at which a snapshot is written to metrics.jsonl.
    training_stats_jsonl_every_n_steps: int = 200

    # Save encoder checkpoint at end of training.
    save_checkpoint: bool = False
    save_best_checkpoint: bool = False
    # Minimum epochs between best-checkpoint disk writes.  The best state_dict is
    # kept in CPU memory so the exact best is never lost; this only throttles I/O.
    best_checkpoint_cooldown_epochs: int = 20

    # steps_per_epoch=0 → auto-estimate from dataset.
    steps_per_epoch: int = 0

    masking: MaskingConfig = field(default_factory=lambda: MaskingConfig(
        p_full_assay=1.0,
        p_full_loci=0.0,
        p_chunks=0.0,
    ))
    dsf: DsfConfig = field(default_factory=DsfConfig)
    optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(name="adamw"))
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    # grad.clip_norm=1.0 required: LeWM lewm.yaml uses gradient_clip_val=1.0
    grad: GradConfig = field(default_factory=lambda: GradConfig(clip_norm=1.0))


@dataclass
class JEPADecoderTrainingConfig:
    """Training knobs for JEPA Stage 2 decoder training."""

    epochs: int = 200
    batch_size: int = 16
    grad_accum_steps: int = 1
    seed: int = 42
    strict_determinism: bool = False
    device: Optional[str] = None
    run_dir: str = "."
    amp: bool = True
    augment_reverse_complement_prob: float = 0.0
    max_train_batches: Optional[int] = None
    eval_each_epoch: bool = True
    eval_max_batches: int = 0
    geometry_log_every_n_steps: int = 0
    steps_per_epoch: int = 0
    training_stats_jsonl_every_n_steps: int = 200
    save_checkpoint: bool = True
    save_best_checkpoint: bool = True
    best_checkpoint_cooldown_epochs: int = 20
    early_stop_enabled: bool = False
    early_stop_patience: int = 5
    masking: MaskingConfig = field(default_factory=lambda: MaskingConfig(
        p_full_assay=1.0,
        p_full_loci=0.0,
        p_chunks=0.0,
        min_available_frac=0.3,
        preserve_assay_id=True,
    ))
    dsf: DsfConfig = field(default_factory=lambda: DsfConfig(dsf_list=[1, 2, 4], sampling="uniform"))
    optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(name="adamax"))
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    grad: GradConfig = field(default_factory=lambda: GradConfig(clip_norm=2.0))
    loss_weights: LossWeightsConfig = field(default_factory=LossWeightsConfig)


@dataclass
class DecoderConfig:
    """JEPA Stage 2 decoder architecture and ablation knobs."""

    checkpoint_path: str = ""
    freeze_mode: Literal["decoder_only", "predictor_decoder", "encoder_decoder", "all"] = "decoder_only"
    heads: Literal["joint", "count_only", "pval_only", "peak_only"] = "joint"
    grouped_deconv: bool = False
    loss_mode: Literal["obs_imp", "unified"] = "obs_imp"
    decoder_input_dim: int = 0
    n_cnn_layers: int = 3
    expansion_factor: int = 2
    pool_size: int = 2
    conv_kernel_size: int = 3
    norm: Literal["layer", "batch", "group", "instance", "weight", "rms"] = "layer"
    gaussian_var_min: float = 0.1
    signal_dist: Literal["gaussian", "laplace", "student_t", "gamma", "mse", "mae"] = "gaussian"


@dataclass
class JEPAConfig:
    """Top-level config for sandbox/train_jepa.py."""

    model_type: Literal["candi", "fresh"] = "candi"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)       # encoder architecture
    jepa: JEPAModelConfig = field(default_factory=JEPAModelConfig)
    # Includes E23 encoder-ablation knobs (missing_data_mode, film_mode,
    # conv_norm, dna_pool_order, transformer_type) from sandbox.jepa_model.
    fresh: FreshJEPAModelConfig = field(default_factory=FreshJEPAModelConfig)
    training: JEPATrainingConfig = field(default_factory=JEPATrainingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    hpo: HpoConfig = field(default_factory=HpoConfig)


@dataclass
class JEPADecoderConfig:
    """Top-level config for sandbox/train_jepa_decoder.py."""

    model_type: Literal["candi", "fresh"] = "fresh"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    jepa: JEPAModelConfig = field(default_factory=JEPAModelConfig)
    fresh: FreshJEPAModelConfig = field(default_factory=FreshJEPAModelConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    training: JEPADecoderTrainingConfig = field(default_factory=JEPADecoderTrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    hpo: HpoConfig = field(default_factory=HpoConfig)

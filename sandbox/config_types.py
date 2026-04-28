"""Nested dataclass schema for sandbox (plan §10.1)."""
from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Literal, Optional, Type, TypeVar, Union, get_args, get_origin, get_type_hints


T = TypeVar("T")


@dataclass
class DataConfig:
    h5_path: str = "sandbox/data/sandbox.h5"
    regime: Literal["type1_chr19", "type2_loci"] = "type2_loci"
    context_length: int = 768
    resolution: int = 25
    signal_transform: str = "arcsinh"
    num_workers: int = 0
    region_stratified_loss: bool = True
    # If True and file size <= ram_cache_max_bytes, mmap the whole H5 into RAM once per worker epoch.
    h5_cache_ram: bool = True
    ram_cache_max_bytes: int = 10 * 1024 * 1024 * 1024  # 10 GiB


@dataclass
class DsfConfig:
    dsf_list: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    sampling: Literal["off", "uniform", "x_eq_y", "upsample_only"] = "uniform"
    eval_dsf: Literal["1", "x_dsf"] = "1"


@dataclass
class MaskingConfig:
    p_full_assay: float = 0.8    # prob of masking a randomly chosen set of whole assay columns
    p_full_loci: float = 0.5     # prob of masking the same loci chunk across all assays
    p_chunks: float = 0.0        # prob of per-assay chunk masking
    # mask_fraction controls the fraction of LOCI to mask for p_full_loci and p_chunks strategies.
    # It is NOT used by p_full_assay (the number of masked assay columns is drawn randomly).
    mask_fraction: float = 0.2
    chunk_size: int = 40


@dataclass
class ModelConfig:
    n_cnn_layers: int = 3
    expansion_factor: int = 2
    n_transformer_layers: int = 2
    nhead: int = 4
    dropout: float = 0.02
    metadata_embedding_dim_mult: int = 4
    mask_stem: bool = True
    decoder_type: Literal["fixed"] = "fixed"
    separate_decoders: bool = True
    signal_dist: Literal["gaussian", "laplace", "student_t", "gamma", "mse", "mae"] = "gaussian"
    # log1p compresses the input dynamic range inside `CANDI.encode`. Empirically (B7 in the
    # 2026-04 sweep) the only chr19 baseline whose late-stage `eval_losses/total_loss` did not
    # diverge — promoted to default 2026-04-27. Override with `--model.encode_input_transform none`
    # to recover the prior arcsinh-elsewhere-only behaviour.
    encode_input_transform: Literal["none", "arcsinh", "log1p"] = "log1p"



@dataclass
class LossWeightsConfig:
    count_weight: float = 1.0
    pval_weight: float = 1.0
    peak_weight: float = 1.0
    obs_weight: float = 1.0
    imp_weight: float = 1.0


@dataclass
class AdamHParams:
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-3  # match prod train.py defaults for Adam-family stability
    weight_decay: float = 0.0
    amsgrad: bool = False


@dataclass
class AdamWHParams:
    lr: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-3  # match prod train.py AdamW default
    weight_decay: float = 0.01
    amsgrad: bool = False


@dataclass
class AdamaxHParams:
    lr: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-3
    weight_decay: float = 0.0


@dataclass
class SgdHParams:
    lr: float = 1e-2
    momentum: float = 0.9
    dampening: float = 0.0
    weight_decay: float = 0.0
    nesterov: bool = False


@dataclass
class OptimizerConfig:
    name: Literal["adam", "adamw", "adamax", "sgd"] = "adamax"
    adam: AdamHParams = field(default_factory=AdamHParams)
    adamw: AdamWHParams = field(default_factory=AdamWHParams)
    adamax: AdamaxHParams = field(default_factory=AdamaxHParams)
    sgd: SgdHParams = field(default_factory=SgdHParams)


@dataclass
class ScheduleConfig:
    name: Literal["cosine", "constant", "linear"] = "cosine"
    warmup_frac: float = 0.1  # fraction of epochs for LR warmup (cosine only); 0 disables
    min_lr_ratio: float = 0.1


@dataclass
class GradConfig:
    # Gradient clipping cap value. Interpretation depends on clip_mode:
    # - norm:  clip global grad norm (torch.nn.utils.clip_grad_norm_)
    # - value: clip each gradient element to [-cap, +cap] (torch.nn.utils.clip_grad_value_)
    # <= 0 disables clipping.
    clip_norm: float = 1.0
    clip_mode: Literal["norm", "value"] = "norm"
    clip_log_window: int = 100


@dataclass
class TrainingConfig:
    epochs: int = 10
    batch_size: int = 16
    grad_accum_steps: int = 1
    seed: int = 42
    strict_determinism: bool = False
    device: Optional[str] = None
    run_dir: str = "."
    amp: bool = False
    # Probability (per batch) of reverse-complement + signal flip (prod train.py); 0 disables.
    augment_reverse_complement_prob: float = 0.0
    max_train_batches: Optional[int] = None
    eval_each_epoch: bool = True
    eval_max_batches: int = 0  # 0 = evaluate all available chr21 batches; >0 caps the count
    steps_per_epoch: int = 0  # 0 = auto-estimate from dataset length each run
    # Cadence (in optimizer steps) at which a `training_stats`/`training_losses` snapshot is
    # appended to metrics.jsonl as a record with `kind="training_step"`. 0 disables. This
    # is the only on-disk source of grad-norm / clip-fraction history available offline (W&B
    # has the per-step trace, but it is not always reachable from autoresearch agents).
    training_stats_jsonl_every_n_steps: int = 200
    # Save model checkpoint(s). OFF by default: sandbox runs are diagnostic, not production.
    # When True, writes `<run_dir>/checkpoint_last.pt` at the end of training.
    save_checkpoint: bool = False
    # Optional early stopping on `eval_losses/total_loss`. OFF by default. When enabled,
    # training stops if the latest evaluated total_loss is strictly greater than the best
    # observed total_loss for `early_stop_patience` consecutive eval points (i.e. the loss
    # is *increasing* from its minimum, signalling divergence/overfitting).
    early_stop_enabled: bool = False
    early_stop_patience: int = 5
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    grad: GradConfig = field(default_factory=GradConfig)
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    loss_weights: LossWeightsConfig = field(default_factory=LossWeightsConfig)
    dsf: DsfConfig = field(default_factory=DsfConfig)


@dataclass
class EvalConfig:
    # Path to eic_metadata.csv; provides canonical per-assay metadata for truly missing
    # assay slots (y_avail=0) during eval. Set to "" to disable.
    eic_metadata_path: str = "data/eic_metadata.csv"
    # When True (default), inject canonical metadata for truly missing assay slots (y_avail=0)
    # and extend query_mask so the model generates predictions for them. When False, those
    # slots keep the -1 MISSING token in y_meta and are not queried.
    use_canonical_missing_meta: bool = True
    # Metadata sensitivity probe interval (optimizer steps). 0 = disabled.
    # Every N steps during training: runs two forward passes (normal y_meta vs all-(-1) y_meta)
    # and logs MSE per output head as probe/meta_sensitivity_*_mse.
    meta_sensitivity_probe_every_n_steps: int = 200
    probe_depth_lo: int = 22
    probe_depth_hi: int = 24
    probe_read_length_lo: int = 36
    probe_read_length_hi: int = 100
    probe_run_types: List[int] = field(default_factory=lambda: [0, 1])
    probe_num_windows: int = 16
    eval_every_n_epochs: int = 5
    # All prefixes are searched and EVERY matching imp biosample is used (union, not priority).
    # e.g. if T_AAA has both V_AAA and B_AAA, eval cycles through (T_AAA,V_AAA) and (T_AAA,B_AAA).
    eval_imp_prefixes: List[str] = field(default_factory=lambda: ["V_", "B_"])


@dataclass
class WandbConfig:
    project: str = "candi_sandbox"
    entity: Optional[str] = None
    mode: Literal["online", "offline", "disabled"] = "online"
    run_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class HpoConfig:
    """HPO-graph metadata for one run (see `sandbox/hpo/graph.py`).

    `parent` declares the run_id(s) this run was derived from — used to draw edges
    in `sandbox/hpo_graph.json`. `experiment_label` groups related runs (e.g.
    `E1_lrfloor_low`). `disable` skips the graph update entirely (smoke tests).
    """
    parent: List[str] = field(default_factory=list)
    experiment_label: str = ""
    notes: str = ""
    disable: bool = False
    graph_path: str = "sandbox/hpo_graph.json"


@dataclass
class SandboxConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    hpo: HpoConfig = field(default_factory=HpoConfig)


def _strip_optional(tp: Any) -> Any:
    origin = get_origin(tp)
    if origin is Union:
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return tp


def _coerce_scalar(tp: Any, val: Any) -> Any:
    if val is None and get_origin(tp) is Union and type(None) in get_args(tp):
        return None
    tp = _strip_optional(tp)
    origin = get_origin(tp)
    if origin is Literal or (origin is not None and getattr(origin, "__name__", "") == "Literal"):
        choices = list(get_args(tp))
        if val not in choices:
            raise ValueError(f"value {val!r} not in Literal{choices}")
        return val
    if tp is bool:
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower() in ("1", "true", "yes", "y")
        return bool(val)
    if tp is int:
        return int(val)
    if tp is float:
        return float(val)
    if tp is str:
        return str(val)
    raise TypeError(f"unsupported scalar type {tp}")


def _coerce_container(tp: Any, val: Any) -> Any:
    tp = _strip_optional(tp)
    origin = get_origin(tp)
    if origin is list:
        (elem_tp,) = get_args(tp)
        if isinstance(val, str):
            parts = [p.strip() for p in val.split(",") if p.strip()]
            return [_coerce_scalar(elem_tp, p) for p in parts]
        if not isinstance(val, list):
            raise TypeError(f"expected list, got {type(val)}")
        if elem_tp is str:
            return [str(x) for x in val]
        return [_coerce_scalar(elem_tp, x) for x in val]
    raise TypeError(f"unsupported container {tp}")


def config_from_dict(cls: Type[T], data: Dict[str, Any], *, path: str = "") -> T:
    if not is_dataclass(cls):
        raise TypeError(cls)
    if not isinstance(data, dict):
        raise TypeError(f"{path or 'root'}: expected dict, got {type(data)}")
    hints = get_type_hints(cls)
    extra = set(data.keys()) - {f.name for f in fields(cls)}
    if extra:
        raise ValueError(f"unknown config keys at {path or 'root'}: {sorted(extra)}")
    kwargs: Dict[str, Any] = {}
    for f in fields(cls):
        raw = data.get(f.name, MISSING)
        ft = hints[f.name]
        inner = _strip_optional(ft)
        if raw is MISSING:
            if f.default_factory is not MISSING:
                kwargs[f.name] = f.default_factory()
            elif f.default is not MISSING:
                kwargs[f.name] = f.default
            else:
                raise ValueError(f"missing required key {(path + '.' + f.name).strip('.')}")
            continue
        if is_dataclass(inner):
            kwargs[f.name] = config_from_dict(inner, raw, path=f"{path}.{f.name}".strip("."))
        elif get_origin(ft) is list or get_origin(_strip_optional(ft)) is list:
            kwargs[f.name] = _coerce_container(ft, raw)
        else:
            kwargs[f.name] = _coerce_scalar(ft, raw)
    return cls(**kwargs)  # type: ignore[arg-type]

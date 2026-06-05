"""Load fixed chr19 batches from sandbox HDF5 for real-data overfit diagnostics."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from sandbox import SANDBOX_ASSAYS
from sandbox.candi_v2.config import CANDIv2Config, DecoderConfig, EncoderConfig
from sandbox.config import deep_merge, load_yaml
from sandbox.config_types import config_from_dict
from sandbox.data import SandboxH5Dataset


def default_h5_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "sandbox.h5"


def collect_batches(
    h5_path: Path,
    *,
    n_batches: int = 1,
    batch_size: int = 4,
    seed: int = 42,
    regime: str = "type1_chr19",
    dsf_sampling: str = "off",
) -> List[Dict[str, torch.Tensor]]:
    """Grab the first ``n_batches`` training batches (deterministic, chr19)."""
    ds = SandboxH5Dataset(
        h5_path,
        regime,
        train=True,
        batch_size=batch_size,
        biosample_prefix="T_",
        dsf_list=(1, 2, 4, 8),
        dsf_sampling=dsf_sampling,
        seed=seed,
        shuffle=False,
        h5_cache_ram=True,
        preserve_assay_id=True,
    )
    batches: List[Dict[str, torch.Tensor]] = []
    for batch in ds:
        batches.append(batch)
        if len(batches) >= n_batches:
            break
    if not batches:
        raise RuntimeError(f"No chr19 batches from {h5_path} regime={regime}")
    return batches


def build_real_v2_config(
    *,
    context_length: int = 768,
    heads: str = "count_only",
    lr: float = 1e-3,
    clip_norm: float = 2.0,
    dropout: float = 0.0,
) -> CANDIv2Config:
    """CANDI v2 config aligned with sandbox chr19 (8 assays, L=768)."""
    yaml_path = Path(__file__).resolve().parents[1] / "configs" / "candi_v2_default.yaml"
    cfg_dict: Dict[str, Any] = asdict(CANDIv2Config())
    if yaml_path.exists():
        cfg_dict = deep_merge(cfg_dict, load_yaml(yaml_path))
    cfg_dict["encoder"]["num_assays"] = len(SANDBOX_ASSAYS)
    cfg_dict["encoder"]["context_length"] = context_length
    cfg_dict["encoder"]["dropout"] = dropout
    cfg_dict["decoder"]["heads"] = heads
    cfg_dict["training"]["optimizer"]["adamax"]["lr"] = lr
    cfg_dict["training"]["grad"]["clip_norm"] = clip_norm
    cfg_dict["training"]["loss_weights"]["count_weight"] = 1.0
    cfg_dict["training"]["loss_weights"]["peak_weight"] = 0.0 if heads == "count_only" else 1.0
    cfg_dict["training"]["loss_weights"]["pval_weight"] = 0.0
    cfg_dict["training"]["loss_weights"]["obs_weight"] = 1.0
    cfg_dict["training"]["loss_weights"]["imp_weight"] = 1.0
    return config_from_dict(CANDIv2Config, cfg_dict)

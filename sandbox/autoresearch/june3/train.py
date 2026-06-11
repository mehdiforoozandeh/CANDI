#!/usr/bin/env python3
"""Agent-editable architecture + config for E34 june3 autoresearch."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import torch
import torch.nn as nn

from sandbox import SANDBOX_ASSAYS
from sandbox.autoresearch.june3.candi_v2.config import CANDIv2Config, validate_v2_config
from sandbox.autoresearch.june3.candi_v2.model import CANDIv2
from sandbox.config import deep_merge, load_yaml
from sandbox.config_types import config_from_dict

ARTIFACT_DIR = Path(__file__).resolve().parent
AR_FIXED_YAML = ARTIFACT_DIR / "ar_fixed.yaml"


def _load_baseline_config() -> CANDIv2Config:
    merged: dict = dataclasses.asdict(CANDIv2Config())
    merged = deep_merge(merged, load_yaml(AR_FIXED_YAML))
    merged["encoder"]["num_assays"] = len(SANDBOX_ASSAYS)
    enc_st = merged.get("encoder", {}).get("signal_transform")
    if enc_st is not None:
        merged.setdefault("data", {})["signal_transform"] = enc_st
    return config_from_dict(CANDIv2Config, merged)


def get_config() -> CANDIv2Config:
    """Return config for this run. Agent may edit encoder/decoder fields here."""
    cfg = _load_baseline_config()
    # --- KEEP base (cleaned 2026-06-11: removed dead duplicate assignments) ---
    # encoder
    cfg.encoder.n_transformer_layers = 4
    cfg.encoder.nhead = 8
    cfg.encoder.conv_norm = "layer"
    cfg.encoder.signal_transform = "log1p"
    cfg.encoder.dropout = 0.02
    cfg.encoder.dna_pool_order = "early"
    cfg.encoder.fusion_deep = True
    cfg.encoder.fusion_norm = "layer"
    cfg.encoder.attn_qk_norm = True
    cfg.encoder.transformer_layer_drop = 0.05
    # decoder
    cfg.decoder.trunk = "separate"
    cfg.decoder.learnable_depth_center = True
    cfg.decoder.learnable_depth_slope = True
    cfg.decoder.conv_kernel_size = 5
    cfg.decoder.meta_embed_dim = 8
    cfg.decoder.dcr_penalty_weight = 1.5
    cfg.decoder.consistency_weight = 0.1
    cfg.decoder.spatial_smoothness_weight = 0.05
    cfg.decoder.norm = "group"
    cfg.encoder.output_rms_norm = True  # KEEP12 (real +0.0038 avg over KEEP9)
    # --- EXPERIMENT (single knob, last assignment wins) ---
    cfg.encoder.n_transformer_layers = 5
    return cfg


def build_model(cfg: CANDIv2Config, device: torch.device) -> CANDIv2:
    validate_v2_config(cfg)
    return CANDIv2(cfg).to(device)


class V2TupleWrapper(nn.Module):
    """6-tuple forward for sandbox.train train_one_epoch / run_eval_pass."""

    def __init__(self, model: CANDIv2) -> None:
        super().__init__()
        self.v2 = model

    def forward(self, x_data, x_dna, x_meta, y_meta, **kwargs):
        return self.v2.forward_tuple(x_data, x_dna, x_meta, y_meta, **kwargs)


def build_train_wrapper(model: CANDIv2) -> V2TupleWrapper:
    return V2TupleWrapper(model)


def main() -> int:
    from sandbox.autoresearch.june3.prepare import print_summary, run_experiment

    result = run_experiment()
    print_summary(result)
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

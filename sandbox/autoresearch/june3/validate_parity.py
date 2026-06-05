#!/usr/bin/env python3
"""Verify vendored june3/candi_v2 matches production sandbox/candi_v2 at init."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

from sandbox import SANDBOX_ASSAYS
from sandbox.autoresearch.june3 import train as j3_train
from sandbox.candi_v2.config import CANDIv2Config
from sandbox.candi_v2.model import CANDIv2 as ProdV2
from sandbox.config import deep_merge, load_yaml

AR_YAML = Path(__file__).resolve().parent / "ar_fixed.yaml"


def _prod_cfg() -> CANDIv2Config:
    from sandbox.config_types import config_from_dict

    merged = __import__("dataclasses").asdict(CANDIv2Config())
    merged = deep_merge(merged, load_yaml(AR_YAML))
    merged["encoder"]["num_assays"] = len(SANDBOX_ASSAYS)
    return config_from_dict(CANDIv2Config, merged)


def main() -> int:
    device = torch.device("cpu")
    # Fix the same seed before each init so deterministic random params (torch.randn)
    # are identical. Without this, MetadataEmbedding sentinel embeddings diverge.
    torch.manual_seed(42)
    prod = ProdV2(_prod_cfg()).to(device)
    torch.manual_seed(42)
    vend = j3_train.build_model(j3_train.get_config(), device)
    max_diff = 0.0
    for (n1, p1), (n2, p2) in zip(prod.state_dict().items(), vend.state_dict().items()):
        if n1 != n2:
            print(f"name mismatch: {n1} vs {n2}", file=sys.stderr)
            return 1
        d = (p1 - p2).abs().max().item()
        max_diff = max(max_diff, float(d))
    n_params = sum(p.numel() for p in prod.parameters())
    print(f"param_count: {n_params}")
    print(f"max_weight_diff: {max_diff:.6e}")
    if max_diff > 1e-5:
        print("PARITY FAIL", file=sys.stderr)
        return 1
    print("parity ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Agent-editable training config and train_step hook (ONLY file the agent modifies).

Usage:
    python -m sandbox.autoresearch.may31.train
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

from sandbox.candi_v2.model import CANDIv2
from sandbox.diagnostics.synthetic_overfit import nb_mean


@dataclass
class TrainConfig:
    # Optimizer
    optimizer: str = "adamax"
    lr: float = 1e-3
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    sgd_momentum: float = 0.0
    clip_norm: float = 0.5

    # Loss weights (CANDI_LOSS) — baseline both 1.0
    count_weight: float = 1.0
    obs_weight: float = 2.0
    imp_weight: float = 0.5

    # Count head (agent-tunable)
    depth_center: float = 23.0

    # Calibration extras (agent may add; applied in train_step())
    lambda_mse_imp: float = 0.0
    lambda_mse_obs: float = 0.1
    mse_on_log1p: bool = False

    # Data (C)
    dsf_sampling: str = "off"  # uniform | off — exp1: identity dsf=1 only

    # Encoder transform ablation (D1)
    signal_transform: str = "log1p"  # log1p | none | arcsinh

    # D3 depth dropout
    y_meta_depth_dropout_p: float = 0.0


def get_config() -> TrainConfig:
    return TrainConfig()


def build_optimizer(model: CANDIv2, cfg: TrainConfig) -> torch.optim.Optimizer:
    params = model.parameters()
    name = cfg.optimizer.lower().strip()
    if name == "adam":
        return torch.optim.Adam(
            params, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
            eps=cfg.eps, weight_decay=cfg.weight_decay,
        )
    if name == "adamw":
        return torch.optim.AdamW(
            params, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
            eps=cfg.eps, weight_decay=cfg.weight_decay,
        )
    if name == "adamax":
        return torch.optim.Adamax(
            params, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
            eps=cfg.eps, weight_decay=cfg.weight_decay,
        )
    if name == "sgd":
        return torch.optim.SGD(
            params, lr=cfg.lr, momentum=cfg.sgd_momentum,
            weight_decay=cfg.weight_decay,
        )
    raise ValueError(f"unsupported optimizer {cfg.optimizer!r}; use adam|adamw|adamax|sgd")


def _apply_depth_dropout(y_meta: torch.Tensor, p: float, rng: random.Random) -> torch.Tensor:
    if p <= 0.0:
        return y_meta
    out = y_meta.clone()
    B, _, F = out.shape
    for b in range(B):
        for f in range(F):
            if out[b, 0, f].item() == -1.0:
                continue
            if rng.random() < p:
                out[b, 0, f] = -1.0
    return out


def train_step(
    model: CANDIv2,
    batch: Dict[str, torch.Tensor],
    prep: Dict[str, torch.Tensor],
    base_loss_fn,
    cfg: TrainConfig,
    *,
    global_step: int = 0,
    rng: random.Random | None = None,
) -> Tuple[torch.Tensor, dict]:
    """Agent-editable forward + loss. prepare.py calls this each step."""
    y_meta = prep["y_meta"]
    if cfg.y_meta_depth_dropout_p > 0.0:
        y_meta = _apply_depth_dropout(y_meta, cfg.y_meta_depth_dropout_p, rng or random.Random(0))

    p, n, mu, var, df, peak = model.forward_tuple(
        prep["x_data"], prep["x_dna"], prep["x_meta"], y_meta,
    )
    loss, stats, _terms = base_loss_fn.forward_with_terms(
        p, n, mu, var, df, peak,
        prep["y_data"], prep["y_pval"], prep["y_peaks"],
        prep["observed_map"], prep["masked_map"],
        prep["signal_observed_map"], prep["signal_masked_map"],
        global_step=global_step,
        fallback_imp_to_observed_when_no_masked=False,
    )

    pred_mean = nb_mean(p, n)
    extra = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)

    if cfg.lambda_mse_obs > 0.0 and prep["observed_map"].any():
        m = prep["observed_map"]
        tgt = prep["y_data"][m]
        pred = pred_mean[m]
        if cfg.mse_on_log1p:
            err = F.mse_loss(torch.log1p(pred), torch.log1p(tgt.clamp(min=0.0)))
        else:
            err = F.mse_loss(pred, tgt)
        extra = extra + cfg.lambda_mse_obs * err

    if cfg.lambda_mse_imp > 0.0 and prep["masked_map"].any():
        m = prep["masked_map"]
        tgt = prep["y_data"][m]
        pred = pred_mean[m]
        if cfg.mse_on_log1p:
            err = F.mse_loss(torch.log1p(pred), torch.log1p(tgt.clamp(min=0.0)))
        else:
            err = F.mse_loss(pred, tgt)
        extra = extra + cfg.lambda_mse_imp * err

    total = loss + extra
    out_stats = {
        "count_obs_loss": float(stats.get("loss_branch_count_obs_raw", float("nan"))),
        "count_imp_loss": float(stats.get("loss_branch_count_imp_raw", float("nan"))),
    }
    return total, out_stats


def main() -> int:
    from sandbox.autoresearch.may31 import prepare

    return prepare.run_from_train()


if __name__ == "__main__":
    raise SystemExit(main())

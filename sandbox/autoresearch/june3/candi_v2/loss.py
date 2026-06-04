"""CANDI v2 loss — thin adapter over production CANDI_LOSS.

Reuses SandboxCompositeLoss with pval_weight=0 by default (no pval head).
When heads="all", pval_weight can be set to a positive value via config.
"""
from __future__ import annotations

from sandbox.autoresearch.june3.candi_v2.config import CANDIv2Config
from sandbox.losses import SandboxCompositeLoss
from candi_loss import CANDI_LOSS


def build_v2_loss(cfg: CANDIv2Config) -> SandboxCompositeLoss:
    """Build the loss module for CANDI v2, respecting head configuration.

    When heads != "all", pval_weight is forced to 0.0 regardless of config,
    since the model produces zero tensors for mu/var and we don't want
    the pval NLL branch to contribute gradients.
    """
    lw = cfg.training.loss_weights
    heads = str(cfg.decoder.heads)

    pval_weight = float(lw.pval_weight)
    count_weight = float(lw.count_weight)
    peak_weight = float(lw.peak_weight)

    # Zero out weights for inactive heads
    if heads != "all":
        pval_weight = 0.0
    if heads == "peak_only":
        count_weight = 0.0
    if heads == "count_only":
        peak_weight = 0.0

    cand = CANDI_LOSS(
        dist_type="gaussian",
        count_weight=count_weight,
        pval_weight=pval_weight,
        peak_weight=peak_weight,
        obs_weight=float(lw.obs_weight),
        imp_weight=float(lw.imp_weight),
    )
    return SandboxCompositeLoss(cand)

"""Tests for per-assay median eval and head-isolated eval loss aggregation."""
from __future__ import annotations

import math

import torch

from sandbox.candi_v2.config import CANDIv2Config, validate_v2_config
from sandbox.config_types import config_from_dict
from sandbox.eval import (
    finalize_eval_metrics_median_gw,
    make_assay_gw_pools,
    update_assay_gw_pools,
)
from sandbox.losses import build_eval_losses


def test_build_eval_losses_skips_zero_weight_branches():
    stats_den = {
        "loss_branch_count_obs_raw": 0.5,
        "loss_branch_pval_obs_raw": float("nan"),
        "loss_branch_peak_obs_raw": float("nan"),
    }
    stats_imp = {
        "loss_branch_count_imp_raw": 1.2,
        "loss_branch_pval_imp_raw": float("nan"),
        "loss_branch_peak_imp_raw": float("nan"),
    }
    weights = {
        "count_weight": 1.0,
        "pval_weight": 0.0,
        "peak_weight": 0.0,
        "obs_weight": 1.0,
        "imp_weight": 1.0,
    }
    out = build_eval_losses(stats_den, stats_imp, loss_weights=weights)
    assert out["count_obs_loss"] == 0.5
    assert out["count_imp_loss"] == 1.2
    assert abs(out["total_loss"] - 1.7) < 1e-6
    assert "pval_obs_loss" not in out
    assert "peak_imp_loss" not in out


def test_median_gw_perfect_correlation():
    B, L, F = 2, 8, 2
    mu = torch.randn(B, L, F)
    y = mu.clone()
    obs = torch.ones(B, L, F, dtype=torch.bool)
    p = torch.full((B, L, F), 0.3)
    n = torch.ones(B, L, F)
    peak = torch.sigmoid(mu)
    yd = torch.abs(y) * 10
    ypk = (yd > 5).float()
    pools = make_assay_gw_pools(F)
    update_assay_gw_pools(
        pools,
        output_p=p,
        output_n=n,
        output_mu=mu,
        output_peak=peak,
        y_data=yd,
        y_pval=y,
        y_peaks=ypk,
        observed_map=obs,
        imp_eval_map=None,
    )
    out = finalize_eval_metrics_median_gw(pools)
    assert out["den_pval_pearson"] > 0.99
    assert out["den_pval_pearson_valid_assays"] == 2.0


def test_median_pools_accumulate_across_batches():
    B, L, F = 1, 4, 1
    pools = make_assay_gw_pools(F)
    for shift in (0.0, 1.0):
        mu = torch.randn(B, L, F) + shift
        y = mu.clone()
        obs = torch.ones(B, L, F, dtype=torch.bool)
        p = torch.full((B, L, F), 0.3)
        n = torch.ones(B, L, F)
        peak = torch.sigmoid(mu)
        update_assay_gw_pools(
            pools,
            output_p=p,
            output_n=n,
            output_mu=mu,
            output_peak=peak,
            y_data=mu.abs(),
            y_pval=y,
            y_peaks=(mu.abs() > 0).float(),
            observed_map=obs,
            imp_eval_map=None,
        )
    out = finalize_eval_metrics_median_gw(pools)
    assert out["den_pval_pearson_valid_assays"] == 1.0
    assert out["den_pval_pearson_n_points_median"] == B * L * 2


def test_validate_v2_rejects_loci_masking():
    cfg = config_from_dict(
        CANDIv2Config,
        {
            "training": {
                "masking": {
                    "p_full_assay": 1.0,
                    "p_full_loci": 0.5,
                    "p_chunks": 0.0,
                }
            }
        },
    )
    try:
        validate_v2_config(cfg)
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "assay-only" in str(e)


def test_heads_all_builds_with_var_min():
    from dataclasses import asdict

    from sandbox.candi_v2.model import CANDIv2
    from sandbox.config import deep_merge, load_yaml
    from pathlib import Path

    merged = asdict(CANDIv2Config())
    default = Path(__file__).resolve().parents[1] / "configs" / "candi_v2_default.yaml"
    merged = deep_merge(merged, load_yaml(default))
    merged["decoder"]["heads"] = "all"
    cfg = config_from_dict(CANDIv2Config, merged)
    validate_v2_config(cfg)
    model = CANDIv2(cfg)
    assert model.decoder.gaussian_layer is not None

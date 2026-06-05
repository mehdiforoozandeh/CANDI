"""Core correctness tests for CANDI v2 (config, shapes, masking, loss, gradients)."""
from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from sandbox import SANDBOX_ASSAYS
from sandbox.batch import CLOZE, MISSING, make_masker, prepare_masked_batch
from sandbox.candi_v2.config import (
    CANDIv2Config,
    DecoderConfig,
    EncoderConfig,
    validate_v2_config,
)
from sandbox.candi_v2.decoder import DepthOffsetNegativeBinomialLayer
from sandbox.candi_v2.loss import build_v2_loss
from sandbox.candi_v2.model import CANDIv2
from sandbox.config import deep_merge, load_yaml
from sandbox.config_types import config_from_dict
from sandbox.diagnostics.real_data import default_h5_path
from sandbox.losses import SandboxCompositeLoss
from sandbox.train import _global_grad_norm_for_loss, run_eval_pass
from sandbox.train_candi_v2 import _V2TupleWrapper, load_v2_config

SANDBOX_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = SANDBOX_ROOT / "configs"
H5_PATH = default_h5_path()

HEAD_MATRIX = [
    ("count_only", "plain"),
    ("count_only", "depth_offset"),
    ("count_peak", "plain"),
    ("count_peak", "depth_offset"),
    ("all", "plain"),
    ("all", "depth_offset"),
]


def _tiny_v2_config(*, heads: str = "count_peak", count_head: str = "plain") -> CANDIv2Config:
    enc = EncoderConfig(
        num_assays=4,
        context_length=96,
        metadata_embed_dim=16,
        n_cnn_layers=2,
        expansion_factor=2,
        n_transformer_layers=1,
        nhead=2,
        dropout=0.0,
        signal_transform="log1p",
        film_mode="per_conv",
        missing_data_mode="mask_token",
        fusion_norm="none",
    )
    dec = DecoderConfig(
        heads=heads,  # type: ignore[arg-type]
        count_head=count_head,  # type: ignore[arg-type]
        film_mode="single_pre_decoder",
        meta_embed_dim=16,
        n_cnn_layers=2,
        depth_center=20.0,
    )
    cfg = CANDIv2Config(encoder=enc, decoder=dec)
    cfg.training.loss_weights.peak_weight = 1.0 if heads in ("count_peak", "all") else 0.0
    cfg.training.loss_weights.pval_weight = 1.0 if heads == "all" else 0.0
    cfg.training.loss_weights.count_weight = 1.0 if heads != "peak_only" else 0.0
    return cfg


def _synthetic_batch(
    *,
    B: int = 2,
    L: int = 96,
    F: int = 4,
    Lbp: int = 2400,
    device: torch.device | None = None,
) -> dict:
    dev = device or torch.device("cpu")
    x_data = torch.abs(torch.randn(B, L, F, device=dev)) + 1.0
    x_meta = torch.zeros(B, 4, F, device=dev)
    x_meta[:, 0, :] = 20.0 + torch.arange(F, device=dev).float()
    x_meta[:, 1, :] = torch.arange(F, device=dev).float()
    x_meta[:, 2, :] = 76.0
    x_meta[:, 3, :] = 1.0
    x_avail = torch.ones(B, F, device=dev)
    x_dna = torch.zeros(B, Lbp, 4, device=dev)
    y_data = torch.abs(torch.randn(B, L, F, device=dev)) * 5
    y_meta = x_meta.clone()
    y_avail = torch.ones(B, F, device=dev)
    y_pval = torch.randn(B, L, F, device=dev)
    y_peaks = (torch.rand(B, L, F, device=dev) > 0.7).float()
    control_data = torch.abs(torch.randn(B, L, 1, device=dev)) + 0.5
    control_meta = torch.zeros(B, 4, 1, device=dev)
    control_meta[:, 0, :] = 22.0
    control_avail = torch.ones(B, 1, device=dev)
    y_dsf = torch.ones(B, F, dtype=torch.int64, device=dev)
    return {
        "x_data": x_data,
        "x_meta": x_meta,
        "x_avail": x_avail,
        "x_dna": x_dna,
        "y_data": y_data,
        "y_meta": y_meta,
        "y_avail": y_avail,
        "y_pval": y_pval,
        "y_peaks": y_peaks,
        "control_data": control_data,
        "control_meta": control_meta,
        "control_avail": control_avail,
        "y_dsf": y_dsf,
    }


def _active_output_keys(heads: str) -> set[str]:
    if heads == "count_only":
        return {"p", "n", "z"}
    if heads == "peak_only":
        return {"peak", "z"}
    if heads == "count_peak":
        return {"p", "n", "peak", "z"}
    if heads == "all":
        return {"p", "n", "peak", "mu", "var", "z"}
    raise ValueError(heads)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_candi_v2_default_yaml_parses():
    merged = asdict(CANDIv2Config())
    merged = deep_merge(merged, load_yaml(CONFIGS / "candi_v2_default.yaml"))
    cfg = config_from_dict(CANDIv2Config, merged)
    validate_v2_config(cfg)


@pytest.mark.parametrize("yaml_name", ["e30_v2_common.yaml", "e30_v2_baseline.yaml", "e30_v2_nboffset.yaml"])
def test_e30_yaml_configs_parse(yaml_name: str):
    cfg = load_v2_config([CONFIGS / yaml_name], [])
    validate_v2_config(cfg)


def test_load_v2_config_set_override_count_head():
    cfg = load_v2_config([], ["decoder.count_head=depth_offset", "decoder.heads=count_only"])
    assert cfg.decoder.count_head == "depth_offset"
    assert cfg.decoder.heads == "count_only"
    assert cfg.encoder.num_assays == len(SANDBOX_ASSAYS)


def test_unknown_config_key_rejected():
    with pytest.raises(ValueError, match="unknown"):
        config_from_dict(CANDIv2Config, {"not_a_real_key": 1})


# ---------------------------------------------------------------------------
# Head matrix: shapes + loss weights
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("heads,count_head", HEAD_MATRIX)
def test_head_matrix_builds_and_forward_shapes(heads: str, count_head: str):
    cfg = _tiny_v2_config(heads=heads, count_head=count_head)
    validate_v2_config(cfg)
    model = CANDIv2(cfg)
    batch = _synthetic_batch(F=cfg.encoder.num_assays)
    prep = prepare_masked_batch(
        batch,
        make_masker(p_full_assay=1.0, p_full_loci=0.0, p_chunks=0.0),
        torch.device("cpu"),
    )
    assert prep is not None
    out = model(
        prep["x_data"], prep["x_dna"], prep["x_meta"], prep["y_meta"],
    )
    active = _active_output_keys(heads)
    B, L = prep["x_data"].shape[0], prep["x_data"].shape[1]
    A = cfg.encoder.num_assays
    for key, val in out.items():
        if key == "z":
            assert val is not None
            assert val.shape[0] == B
            continue
        if key in active:
            assert val is not None, key
            assert val.shape == (B, L, A), (key, val.shape)
        else:
            assert val is None, key


@pytest.mark.parametrize("heads,count_head", HEAD_MATRIX)
def test_forward_tuple_inactive_heads_are_zeros(heads: str, count_head: str):
    cfg = _tiny_v2_config(heads=heads, count_head=count_head)
    model = CANDIv2(cfg)
    batch = _synthetic_batch(F=cfg.encoder.num_assays)
    prep = prepare_masked_batch(
        batch,
        make_masker(p_full_assay=1.0, p_full_loci=0.0, p_chunks=0.0),
        torch.device("cpu"),
    )
    assert prep is not None
    p, n, mu, var, df, peak = model.forward_tuple(
        prep["x_data"], prep["x_dna"], prep["x_meta"], prep["y_meta"],
    )
    assert df is None
    for tensor, name in ((p, "p"), (n, "n"), (mu, "mu"), (var, "var"), (peak, "peak")):
        assert torch.isfinite(tensor).all(), name
    if heads == "count_only":
        assert (peak == 0).all()
        assert (mu == 0).all()
    if heads == "peak_only":
        assert (p == 0).all()
        assert (n == 0).all()


@pytest.mark.parametrize("heads,count_head", HEAD_MATRIX)
def test_build_v2_loss_respects_head_isolation(heads: str, count_head: str):
    cfg = _tiny_v2_config(heads=heads, count_head=count_head)
    loss_fn = build_v2_loss(cfg)
    B, L, A = 2, 96, cfg.encoder.num_assays
    p = torch.sigmoid(torch.randn(B, L, A))
    n = torch.nn.functional.softplus(torch.randn(B, L, A)) + 0.5
    mu = torch.randn(B, L, A)
    scale = torch.nn.functional.softplus(torch.randn(B, L, A)) + 1e-3
    peak = torch.sigmoid(torch.randn(B, L, A))
    y_data = torch.abs(torch.randn(B, L, A)) * 3
    y_pval = torch.randn(B, L, A)
    y_peaks = (torch.rand(B, L, A) > 0.7).float()
    observed = torch.ones(B, L, A, dtype=torch.bool)
    masked = torch.zeros(B, L, A, dtype=torch.bool)
    masked[:, : L // 2, 0] = True
    observed = observed & ~masked
    sig_obs = observed.clone()
    sig_msk = masked.clone()

    _, stats = loss_fn(
        p, n, mu, scale, None, peak,
        y_data, y_pval, y_peaks,
        observed, masked, sig_obs, sig_msk,
        global_step=0,
    )
    if heads != "all":
        assert stats["loss_branch_pval_obs_weighted"] == 0.0
        assert stats["loss_branch_pval_imp_weighted"] == 0.0
    if heads == "count_only":
        assert stats["loss_branch_peak_obs_weighted"] == 0.0
        assert stats["loss_branch_peak_imp_weighted"] == 0.0
    if heads == "peak_only":
        assert stats["loss_branch_count_obs_weighted"] == 0.0
        assert stats["loss_branch_count_imp_weighted"] == 0.0


# ---------------------------------------------------------------------------
# Masking invariants
# ---------------------------------------------------------------------------


def test_assay_only_masker_produces_masked_assays():
    batch = _synthetic_batch()
    masker = make_masker(p_full_assay=1.0, p_full_loci=0.0, p_chunks=0.0)
    prep = prepare_masked_batch(batch, masker, torch.device("cpu"))
    assert prep is not None
    assert prep["masked_map"].any()
    assert prep["observed_map"].any()
    # Control channel appended after masking — not in signal maps.
    assert prep["x_data"].shape[-1] == batch["x_data"].shape[-1] + 1


def test_control_channel_never_in_masked_map():
    batch = _synthetic_batch()
    masker = make_masker(p_full_assay=1.0, p_full_loci=0.0, p_chunks=0.0)
    prep = prepare_masked_batch(batch, masker, torch.device("cpu"))
    assert prep is not None
    F = batch["x_data"].shape[-1]
    assert not prep["masked_map"][:, :, :F].any() or prep["masked_map"].shape[-1] == F
    # x_data includes control; masked_map is signal-only width F
    assert prep["masked_map"].shape[-1] == F


def test_encoder_zeros_cloze_signal_channels():
    cfg = _tiny_v2_config(heads="count_only")
    model = CANDIv2(cfg)
    batch = _synthetic_batch(F=cfg.encoder.num_assays)
    masker = make_masker(p_full_assay=1.0, p_full_loci=0.0, p_chunks=0.0)
    prep = prepare_masked_batch(batch, masker, torch.device("cpu"))
    assert prep is not None
    x_meta = prep["x_meta"]
    x_signal = prep["x_data"][..., : cfg.encoder.num_assays + 1]
    assert prep["masked_map"].any()
    cloze_assays = (x_meta[:, :, : cfg.encoder.num_assays] == CLOZE).any(dim=1)
    if cloze_assays.any():
        for b in range(x_signal.shape[0]):
            if not cloze_assays[b].any():
                continue
            for a in range(cfg.encoder.num_assays):
                if (x_meta[b, :, a] == CLOZE).all():
                    assert (x_signal[b, :, a] == CLOZE).all()
    z = model.encoder.encode(x_signal, prep["x_dna"], x_meta)
    assert torch.isfinite(z).all()


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def test_v2_loss_no_mask_matches_prod_style():
    cfg = _tiny_v2_config(heads="count_peak")
    loss_fn = build_v2_loss(cfg)
    B, L, A = 1, 8, cfg.encoder.num_assays
    p = torch.sigmoid(torch.randn(B, L, A))
    n = torch.nn.functional.softplus(torch.randn(B, L, A)) + 0.5
    mu = torch.randn(B, L, A)
    scale = torch.nn.functional.softplus(torch.randn(B, L, A)) + 1e-3
    peak = torch.sigmoid(torch.randn(B, L, A))
    y_data = torch.abs(torch.randn(B, L, A)) * 3
    y_pval = torch.randn(B, L, A)
    y_peaks = (torch.rand(B, L, A) > 0.7).float()
    observed = torch.ones(B, L, A, dtype=torch.bool)
    masked = torch.zeros(B, L, A, dtype=torch.bool)
    sig_obs = observed.clone()
    sig_msk = masked.clone()
    loss, stats = loss_fn(
        p, n, mu, scale, None, peak,
        y_data, y_pval, y_peaks,
        observed, masked, sig_obs, sig_msk,
        global_step=0,
    )
    assert torch.isfinite(loss)
    assert "loss_branch_count_imp" in stats


def test_count_only_peak_branch_has_zero_grad():
    cfg = _tiny_v2_config(heads="count_only")
    model = CANDIv2(cfg)
    loss_fn = build_v2_loss(cfg)
    batch = _synthetic_batch(F=cfg.encoder.num_assays)
    prep = prepare_masked_batch(
        batch,
        make_masker(p_full_assay=1.0, p_full_loci=0.0, p_chunks=0.0),
        torch.device("cpu"),
    )
    assert prep is not None
    p, n, mu, var, df, peak = model.forward_tuple(
        prep["x_data"], prep["x_dna"], prep["x_meta"], prep["y_meta"],
    )
    _, _, terms = loss_fn.forward_with_terms(
        p, n, mu, var, df, peak,
        prep["y_data"], prep["y_pval"], prep["y_peaks"],
        prep["observed_map"], prep["masked_map"],
        prep["signal_observed_map"], prep["signal_masked_map"],
        global_step=0,
        fallback_imp_to_observed_when_no_masked=False,
    )
    assert float(terms["peak_obs_weighted"].detach()) == 0.0
    assert float(terms["peak_imp_weighted"].detach()) == 0.0
    gn = _global_grad_norm_for_loss(model, terms["peak_obs_weighted"])
    assert gn == 0.0 or math.isnan(gn)


# ---------------------------------------------------------------------------
# Depth-offset math
# ---------------------------------------------------------------------------


def test_depth_offset_nb_layer_math():
    layer = DepthOffsetNegativeBinomialLayer(8, 4, depth_center=20.0, eps=1e-6)
    B, L, A = 2, 4, 4
    x = torch.zeros(B, L, 8)
    depth = torch.full((B, A), 22.0)
    with torch.no_grad():
        layer.linear_eta.weight.zero_()
        layer.linear_eta.bias.fill_(0.5)
        layer.linear_n.weight.zero_()
        layer.linear_n.bias.fill_(1.0)
    p, n = layer(x, depth)
    d = 22.0 - 20.0
    expected_mu = (2.0 ** d) * math.exp(0.5)
    expected_n = math.log1p(math.exp(1.0)) + 1e-6
    expected_p = expected_n / (expected_n + expected_mu)
    assert abs(float(p[0, 0, 0]) - expected_p) < 1e-4
    assert abs(float(n[0, 0, 0]) - expected_n) < 1e-4


def test_depth_offset_scales_mu_with_depth():
    layer = DepthOffsetNegativeBinomialLayer(8, 4, depth_center=20.0, eps=1e-6)
    B, L, A = 1, 1, 4
    x = torch.zeros(B, L, 8)
    with torch.no_grad():
        layer.linear_eta.weight.zero_()
        layer.linear_eta.bias.zero_()
        layer.linear_n.weight.zero_()
        layer.linear_n.bias.fill_(2.0)
    depth_lo = torch.full((B, A), 20.0)
    depth_hi = torch.full((B, A), 22.0)
    p_lo, _ = layer(x, depth_lo)
    p_hi, _ = layer(x, depth_hi)
    mu_lo = (1.0 - p_lo) * (layer.linear_n.bias.exp() + 1e-6) / p_lo.clamp(min=1e-6)
    mu_hi = (1.0 - p_hi) * (layer.linear_n.bias.exp() + 1e-6) / p_hi.clamp(min=1e-6)
    ratio = float(mu_hi.mean() / mu_lo.mean())
    assert 3.5 < ratio < 4.5


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_gradient_flow_active_modules():
    cfg = _tiny_v2_config(heads="count_peak", count_head="plain")
    model = CANDIv2(cfg)
    loss_fn = build_v2_loss(cfg)
    batch = _synthetic_batch(F=cfg.encoder.num_assays)
    prep = prepare_masked_batch(
        batch,
        make_masker(p_full_assay=1.0, p_full_loci=0.0, p_chunks=0.0),
        torch.device("cpu"),
    )
    assert prep is not None
    p, n, mu, var, df, peak = model.forward_tuple(
        prep["x_data"], prep["x_dna"], prep["x_meta"], prep["y_meta"],
    )
    loss, _, _ = loss_fn.forward_with_terms(
        p, n, mu, var, df, peak,
        prep["y_data"], prep["y_pval"], prep["y_peaks"],
        prep["observed_map"], prep["masked_map"],
        prep["signal_observed_map"], prep["signal_masked_map"],
        global_step=0,
        fallback_imp_to_observed_when_no_masked=False,
    )
    model.zero_grad(set_to_none=True)
    loss.backward()
    prefixes = (
        "encoder.metadata_embedding",
        "encoder.signal_tower",
        "encoder.mask_injector",
        "decoder.pre_decoder_film",
        "decoder.neg_binom_layer",
    )
    for prefix in prefixes:
        grads = [
            p.grad.norm().item()
            for name, p in model.named_parameters()
            if name.startswith(prefix) and p.grad is not None
        ]
        assert grads, f"no grads for {prefix}"
        assert sum(grads) > 0, prefix


def test_per_branch_grads_finite():
    cfg = _tiny_v2_config(heads="count_peak")
    model = CANDIv2(cfg)
    loss_fn = build_v2_loss(cfg)
    batch = _synthetic_batch(F=cfg.encoder.num_assays)
    prep = prepare_masked_batch(
        batch,
        make_masker(p_full_assay=1.0, p_full_loci=0.0, p_chunks=0.0),
        torch.device("cpu"),
    )
    assert prep is not None
    p, n, mu, var, df, peak = model.forward_tuple(
        prep["x_data"], prep["x_dna"], prep["x_meta"], prep["y_meta"],
    )
    _, _, terms = loss_fn.forward_with_terms(
        p, n, mu, var, df, peak,
        prep["y_data"], prep["y_pval"], prep["y_peaks"],
        prep["observed_map"], prep["masked_map"],
        prep["signal_observed_map"], prep["signal_masked_map"],
        global_step=0,
        fallback_imp_to_observed_when_no_masked=False,
    )
    for key in ("count_obs_weighted", "count_imp_weighted"):
        term = terms.get(key)
        assert term is not None
        gn = _global_grad_norm_for_loss(model, term)
        assert math.isfinite(gn) and gn > 0, key


# ---------------------------------------------------------------------------
# Eval integration (requires sandbox.h5)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not H5_PATH.exists(), reason="sandbox/data/sandbox.h5 missing")
def test_run_eval_pass_v2_smoke():
    cfg = load_v2_config([], ["decoder.heads=count_peak", "training.batch_size=2"])
    validate_v2_config(cfg)
    model = CANDIv2(cfg)
    wrapper = _V2TupleWrapper(model)
    loss_fn = build_v2_loss(cfg)
    device = torch.device("cpu")
    wrapper.to(device)
    masker = make_masker(p_full_assay=1.0, p_full_loci=0.0, p_chunks=0.0)
    metrics = run_eval_pass(
        wrapper,
        loss_fn,
        H5_PATH,
        "type2_loci",
        device,
        masker,
        batch_size=2,
        seed=42,
        max_batches=2,
        include_median_metrics=True,
    )
    assert math.isfinite(metrics["eval_losses/total_loss"])
    assert "eval_metrics/den_count_pearson_gw" in metrics or any(
        k.startswith("eval_metrics/") for k in metrics
    )
    median_keys = [k for k in metrics if k.startswith("eval_metrics_median/")]
    if median_keys:
        assert any(math.isfinite(metrics[k]) for k in median_keys)

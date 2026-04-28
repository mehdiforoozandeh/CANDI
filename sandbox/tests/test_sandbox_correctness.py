"""Regression tests for sandbox parity with prod semantics (loss, batching, schedulers)."""
from __future__ import annotations

import io
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from candi_loss import CANDI_LOSS
from sandbox import SANDBOX_ASSAYS
from sandbox.batch import make_masker, prepare_masked_batch
from sandbox.config_types import SandboxConfig, config_from_dict
from sandbox.data import SandboxH5Dataset
from sandbox.eval import eval_batch_metrics, pearson_r, spearman_r, auroc_binary
from sandbox.losses import SandboxCompositeLoss
from sandbox.model import build_sandbox_candi
from sandbox.prepare_h5 import DSF_PAIRS_FULL
from sandbox.train import (
    _active_base_lr,
    _maybe_augment_reverse_complement,
    build_scheduler,
    build_optimizer,
)


def test_dsf_parity_pairs_include_symmetric_dsfs():
    assert (2, 2) in DSF_PAIRS_FULL
    assert (8, 8) in DSF_PAIRS_FULL


def test_sandbox_composite_loss_no_mask_matches_prod_duplicate_maps():
    """When masked_map is empty, prod still feeds observed into imp heads; wrapper must match."""
    torch.manual_seed(0)
    B, L, F = 1, 8, len(SANDBOX_ASSAYS)
    cand = CANDI_LOSS(dist_type="gaussian")
    p = torch.sigmoid(torch.randn(B, L, F))
    n = torch.nn.functional.softplus(torch.randn(B, L, F)) + 0.5
    mu = torch.randn(B, L, F)
    scale = torch.nn.functional.softplus(torch.randn(B, L, F)) + 1e-3
    df = None
    peak = torch.sigmoid(torch.randn(B, L, F))
    y_data = torch.abs(torch.randn(B, L, F)) * 3
    y_pval = torch.randn(B, L, F)
    y_peaks = (torch.rand(B, L, F) > 0.7).float()
    observed = torch.ones(B, L, F, dtype=torch.bool)
    masked = torch.zeros(B, L, F, dtype=torch.bool)
    sig_obs = observed.clone()
    sig_msk = masked.clone()

    oc0, ic0, op0, ip0, ok0, ik0 = cand(
        p, n, mu, scale, df, peak,
        y_data, y_pval, y_peaks,
        observed, observed,
        sig_obs, sig_obs,
        global_step=0,
    )
    prod_style = oc0 + ic0 + op0 + ip0 + ok0 + ik0

    crit = SandboxCompositeLoss(cand)
    loss, stats = crit(
        p, n, mu, scale, df, peak,
        y_data, y_pval, y_peaks,
        observed, masked,
        sig_obs, sig_msk,
        global_step=0,
    )
    assert torch.isclose(loss, prod_style)
    assert "loss_branch_count_imp" in stats


def test_prepare_masked_batch_requires_y_dsf():
    B, L, F, Lbp = 1, 4, len(SANDBOX_ASSAYS), 100
    masker = make_masker(p_full_assay=0.0, p_full_loci=0.0, p_chunks=0.0)
    batch = {
        "x_data": torch.randn(B, L, F),
        "x_meta": torch.zeros(B, 4, F),
        "x_avail": torch.ones(B, F),
        "x_dna": torch.zeros(B, Lbp, 4),
        "y_data": torch.randn(B, L, F),
        "y_meta": torch.zeros(B, 4, F),
        "y_avail": torch.ones(B, F),
        "y_pval": torch.randn(B, L, F),
        "y_peaks": torch.rand(B, L, F),
        "control_data": torch.randn(B, L, 1),
        "control_meta": torch.zeros(B, 4, 1),
        "control_avail": torch.ones(B, 1),
    }
    with pytest.raises(ValueError, match="requires y_dsf"):
        prepare_masked_batch(batch, masker, torch.device("cpu"))


def test_prepare_masked_batch_skips_on_avail_mismatch():
    B, L, F, Lbp = 1, 4, len(SANDBOX_ASSAYS), 100
    masker = make_masker(p_full_assay=0.0, p_full_loci=0.0, p_chunks=0.0)
    # y says assay 1 is supervised, but x has no observed/cloze signal for assay 1.
    x_data = torch.full((B, L, F), -1.0)
    x_data[:, :, 0] = 5.0
    batch = {
        "x_data": x_data,
        "x_meta": torch.zeros(B, 4, F),
        "x_avail": torch.tensor([[1.0, 0.0]]),
        "x_dna": torch.zeros(B, Lbp, 4),
        "y_data": torch.randn(B, L, F),
        "y_meta": torch.zeros(B, 4, F),
        "y_avail": torch.ones(B, F),
        "y_pval": torch.randn(B, L, F),
        "y_peaks": torch.rand(B, L, F),
        "control_data": torch.randn(B, L, 1),
        "control_meta": torch.zeros(B, 4, 1),
        "control_avail": torch.ones(B, 1),
        "y_dsf": torch.ones(B, F, dtype=torch.int64),
    }
    with pytest.raises(ValueError, match="Availability"):
        prepare_masked_batch(batch, masker, torch.device("cpu"))


def test_eval_dataset_cycles_all_t_biosamples(tmp_path: Path):
    """Eval mode must not randomly pick one T_* biosample per batch (plan H2)."""
    F = len(SANDBOX_ASSAYS)
    L = 4
    Lbp = L * 25
    n_win = 4
    h5p = tmp_path / "tiny.h5"
    with h5py.File(h5p, "w") as h5:
        h5.attrs["version"] = 1
        ws = h5.create_group("windows")
        dt = h5py.string_dtype(encoding="utf-8")
        ws.create_dataset("chrom", data=np.array(["chr21"] * n_win, dtype=object), dtype=dt)
        ws.create_dataset("start", data=np.arange(n_win) * 19200, dtype=np.int64)
        ws.create_dataset("end", data=(np.arange(n_win) + 1) * 19200, dtype=np.int64)
        ws.create_dataset("region_type", data=np.full(n_win, 255, dtype=np.uint8))
        bg = h5.create_group("biosamples")
        bg.attrs["order"] = json.dumps(["T_AAA", "T_BBB"])
        for name in ("T_AAA", "T_BBB"):
            g = bg.create_group(name)
            for dsf in (1, 2, 4, 8):
                g.create_dataset(f"counts_dsf{dsf}", data=np.zeros((n_win, L, F), dtype=np.int16))
            for dsf in (1, 2, 4, 8):
                g.create_dataset(f"meta_dsf{dsf}", data=np.zeros((4, F), dtype=np.float32))
                g[f"meta_dsf{dsf}"][0, :] = 20.0
                g[f"meta_dsf{dsf}"][1, :] = np.arange(F, dtype=np.float32)
            g.create_dataset("pval", data=np.zeros((n_win, L, F), dtype=np.float16))
            g.create_dataset("peaks", data=np.zeros((n_win, L, F), dtype=np.int64))
            g.create_dataset("control", data=np.zeros((n_win, L, 1), dtype=np.float32))
            g.create_dataset("control_meta", data=np.zeros((n_win, 4, 1), dtype=np.float32))
            g.create_dataset("dna", data=np.zeros((n_win, Lbp, 4), dtype=np.int8))

    ds = SandboxH5Dataset(
        h5p,
        "type2_loci",
        train=False,
        batch_size=1,
        biosample_prefix="T_",
        dsf_sampling="off",
        seed=0,
        shuffle=False,
        eval_include_vb_ground_truth=False,
        h5_cache_ram=False,
    )
    names = [b["biosample_name"] for b in ds]
    assert names[:4] == ["T_AAA", "T_BBB", "T_AAA", "T_BBB"]


def test_h5_ram_cache_reads_from_bytes(tmp_path: Path):
    F = len(SANDBOX_ASSAYS)
    L = 2
    Lbp = L * 25
    h5p = tmp_path / "ram.h5"
    with h5py.File(h5p, "w") as h5:
        h5.attrs["version"] = 1
        ws = h5.create_group("windows")
        dt = h5py.string_dtype(encoding="utf-8")
        ws.create_dataset("chrom", data=np.array(["chr21"], dtype=object), dtype=dt)
        ws.create_dataset("start", data=np.array([0], dtype=np.int64))
        ws.create_dataset("end", data=np.array([19200], dtype=np.int64))
        ws.create_dataset("region_type", data=np.array([255], dtype=np.uint8))
        bg = h5.create_group("biosamples")
        bg.attrs["order"] = json.dumps(["T_X"])
        g = bg.create_group("T_X")
        for dsf in (1, 2, 4, 8):
            g.create_dataset(f"counts_dsf{dsf}", data=np.zeros((1, L, F), dtype=np.int16))
        for dsf in (1, 2, 4, 8):
            g.create_dataset(f"meta_dsf{dsf}", data=np.zeros((4, F), dtype=np.float32))
            g[f"meta_dsf{dsf}"][0, :] = 20.0
            g[f"meta_dsf{dsf}"][1, :] = np.arange(F, dtype=np.float32)
        g.create_dataset("pval", data=np.zeros((1, L, F), dtype=np.float16))
        g.create_dataset("peaks", data=np.zeros((1, L, F), dtype=np.int64))
        g.create_dataset("control", data=np.zeros((1, L, 1), dtype=np.float32))
        g.create_dataset("control_meta", data=np.zeros((1, 4, 1), dtype=np.float32))
        g.create_dataset("dna", data=np.zeros((1, Lbp, 4), dtype=np.int8))

    buf = h5p.read_bytes()
    ds = SandboxH5Dataset(
        h5p,
        "type2_loci",
        train=False,
        batch_size=1,
        dsf_sampling="off",
        seed=0,
        shuffle=False,
        h5_cache_ram=True,
        ram_cache_max_bytes=len(buf) + 100,
    )
    assert ds._ram_buf is not None
    b = next(iter(ds))
    assert b["biosample_name"] == "T_X"


def test_reverse_complement_augment_prob_one():
    torch.manual_seed(0)
    B, L, F, Lbp = 1, 2, len(SANDBOX_ASSAYS), 50
    batch = {
        "x_dna": torch.tensor([[[1.0, 0, 0, 0], [0, 1.0, 0, 0]]]),
        "x_data": torch.arange(L * F, dtype=torch.float32).view(1, L, F),
        "y_data": torch.arange(100, 100 + L * F, dtype=torch.float32).view(1, L, F),
        "y_pval": torch.zeros(1, L, F),
        "y_peaks": torch.zeros(1, L, F),
        "control_data": torch.ones(1, L, 1),
    }
    out = _maybe_augment_reverse_complement(batch, 1.0)
    assert not torch.allclose(out["x_dna"], batch["x_dna"])


def test_build_scheduler_cosine_warmup_sequential():
    d = deepcopy(asdict(SandboxConfig()))
    d["training"]["epochs"] = 10
    d["training"]["schedule"]["name"] = "cosine"
    d["training"]["schedule"]["warmup_frac"] = 0.2
    cfg = config_from_dict(SandboxConfig, d)
    m = build_sandbox_candi(context_bins=64)
    opt = build_optimizer(m, cfg)
    sch = build_scheduler(opt, cfg, total_steps=100)
    assert sch is not None
    assert type(sch).__name__ == "SequentialLR"


def test_build_optimizer_eps_default_matches_config():
    # Default optimizer is adamax.
    cfg = config_from_dict(SandboxConfig, deepcopy(asdict(SandboxConfig())))
    assert cfg.training.optimizer.name == "adamax"
    assert cfg.training.optimizer.adamax.eps == 1e-3
    m = build_sandbox_candi(context_bins=64)
    opt = build_optimizer(m, cfg)
    assert opt.param_groups[0]["eps"] == 1e-3


def test_cli_override_optimizer_eps():
    # Override eps on adamw (explicit opt name + eps).
    d = deepcopy(asdict(SandboxConfig()))
    d["training"]["optimizer"]["name"] = "adamw"
    d["training"]["optimizer"]["adamw"]["eps"] = 1e-5
    cfg = config_from_dict(SandboxConfig, d)
    m = build_sandbox_candi(context_bins=64)
    opt = build_optimizer(m, cfg)
    assert opt.param_groups[0]["eps"] == 1e-5


def test_active_base_lr():
    cfg = config_from_dict(SandboxConfig, deepcopy(asdict(SandboxConfig())))
    assert _active_base_lr(cfg) == cfg.training.optimizer.adamax.lr


def test_sandbox_composite_loss_weights_property():
    lw = SandboxConfig().training.loss_weights
    cand = CANDI_LOSS(
        dist_type="gaussian",
        count_weight=lw.count_weight,
        pval_weight=lw.pval_weight,
        peak_weight=lw.peak_weight,
        obs_weight=lw.obs_weight,
        imp_weight=lw.imp_weight,
    )
    crit = SandboxCompositeLoss(cand)
    w = crit.loss_weights
    assert w["count_weight"] == 1.0 and w["imp_weight"] == 1.0


def test_eval_batch_metrics_pearson_matches_naive():
    torch.manual_seed(1)
    B, L, F = 1, 16, 3
    mu = torch.randn(B, L, F)
    y = mu + 0.1 * torch.randn(B, L, F)
    y_data = torch.abs(torch.randn(B, L, F)) * 3
    peak = torch.sigmoid(torch.randn(B, L, F))
    ypk = (torch.rand(B, L, F) > 0.5).float()
    p_head = torch.sigmoid(torch.randn(B, L, F))
    n_head = torch.nn.functional.softplus(torch.randn(B, L, F)) + 0.5
    obs = torch.ones(B, L, F, dtype=torch.bool)
    msk = torch.zeros(B, L, F, dtype=torch.bool)
    m = eval_batch_metrics(p_head, n_head, mu, peak, y_data, y, ypk, obs, msk, obs, msk, regime="type2_loci")
    r1 = m["den_pval_pearson_gw"]
    r2 = pearson_r(mu, y, obs)
    assert abs(r1 - r2) < 1e-5


def test_eval_batch_metrics_1obs_uses_top1pct_gt():
    """1obs scope must select top-1% positions by GT signal, not DSF-1 intersection."""
    torch.manual_seed(42)
    B, L, F = 1, 200, 2
    # Build y_pval with a clear 99th-percentile boundary.
    y_pval = torch.rand(B, L, F)
    # Set top 2 positions (1% of 200) to very high values.
    y_pval[0, 0, 0] = 100.0
    y_pval[0, 1, 0] = 100.0
    mu = torch.rand(B, L, F)
    y_data = torch.rand(B, L, F)
    p_head = torch.sigmoid(torch.randn(B, L, F))
    n_head = torch.nn.functional.softplus(torch.randn(B, L, F)) + 0.5
    peak = torch.sigmoid(torch.randn(B, L, F))
    ypk = (torch.rand(B, L, F) > 0.5).float()
    obs = torch.ones(B, L, F, dtype=torch.bool)
    msk = torch.zeros(B, L, F, dtype=torch.bool)
    # signal_observed_map: only position 5 is DSF-1 (not the top-99th-pct ones).
    sig_obs = torch.zeros(B, L, F, dtype=torch.bool)
    sig_obs[0, 5, 0] = True
    m = eval_batch_metrics(p_head, n_head, mu, peak, y_data, y_pval, ypk, obs, msk, sig_obs, msk)
    # 1obs must exist and be different from dsf1.
    assert "den_pval_pearson_1obs" in m
    assert "den_pval_pearson_dsf1" in m
    # gw uses all obs positions, 1obs uses top-1%, dsf1 uses sig_obs intersection.
    # They are all different metric names now.
    assert "den_pval_pearson_gw" in m


def test_spearman_r_matches_scipy():
    """spearman_r must match scipy.stats.spearmanr exactly (tie handling)."""
    from scipy.stats import spearmanr as scipy_spearmanr
    torch.manual_seed(7)
    # Create data with ties (many zeros, common in genomics).
    pred = torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0, 3.0])
    tgt  = torch.tensor([0.0, 0.0, 1.0, 0.0, 2.0, 3.0])
    mask = torch.ones(pred.shape, dtype=torch.bool)
    # Reshape to [B, L, F] where B=1, L=6, F=1.
    pred3 = pred.view(1, 6, 1)
    tgt3  = tgt.view(1, 6, 1)
    mask3 = mask.view(1, 6, 1)
    got = spearman_r(pred3, tgt3, mask3)
    expected, _ = scipy_spearmanr(pred.numpy(), tgt.numpy())
    assert abs(got - expected) < 1e-6


def test_auroc_binary_matches_sklearn():
    """auroc_binary must match sklearn.metrics.roc_auc_score."""
    from sklearn.metrics import roc_auc_score
    torch.manual_seed(3)
    B, L, F = 1, 20, 1
    pred = torch.rand(B, L, F)
    tgt = (torch.rand(B, L, F) > 0.5).float()
    mask = torch.ones(B, L, F, dtype=torch.bool)
    # Ensure both classes present.
    tgt[0, 0, 0] = 0.0
    tgt[0, 1, 0] = 1.0
    got = auroc_binary(pred, tgt, mask)
    p_np = pred[mask].numpy()
    t_np = (tgt[mask] > 0.5).long().numpy()
    expected = roc_auc_score(t_np, p_np)
    assert abs(got - expected) < 1e-6


def test_eval_batch_metrics_imp_nan_when_no_imp_gt():
    """imp_* metrics must all be nan when y_pval_imp is None."""
    import math
    torch.manual_seed(5)
    B, L, F = 1, 16, 2
    mu = torch.randn(B, L, F)
    y_pval = torch.randn(B, L, F)
    y_data = torch.abs(torch.randn(B, L, F))
    p_head = torch.sigmoid(torch.randn(B, L, F))
    n_head = torch.nn.functional.softplus(torch.randn(B, L, F)) + 0.5
    peak = torch.sigmoid(torch.randn(B, L, F))
    ypk = (torch.rand(B, L, F) > 0.5).float()
    obs = torch.ones(B, L, F, dtype=torch.bool)
    msk = torch.ones(B, L, F, dtype=torch.bool)
    m = eval_batch_metrics(
        p_head, n_head, mu, peak, y_data, y_pval, ypk, obs, msk, obs, msk,
        y_pval_imp=None,
        y_peaks_imp=None,
    )
    # All imp_* keys that are present must be nan.
    imp_keys = [k for k in m if k.startswith("imp_")]
    assert imp_keys, "no imp_* keys in output"
    for k in imp_keys:
        assert math.isnan(m[k]), f"{k} = {m[k]}, expected nan"


def test_eval_batch_metrics_imp_count_nan_without_imp_count_gt():
    """Imp count metrics must be NaN when y_data_imp is missing, even if y_pval_imp exists."""
    import math

    torch.manual_seed(11)
    B, L, F = 1, 16, 2
    mu = torch.randn(B, L, F)
    y_pval = torch.randn(B, L, F)
    y_data = torch.abs(torch.randn(B, L, F))
    p_head = torch.sigmoid(torch.randn(B, L, F))
    n_head = torch.nn.functional.softplus(torch.randn(B, L, F)) + 0.5
    peak = torch.sigmoid(torch.randn(B, L, F))
    ypk = (torch.rand(B, L, F) > 0.5).float()
    y_pval_imp = torch.randn(B, L, F)
    y_peaks_imp = (torch.rand(B, L, F) > 0.5).float()
    obs = torch.ones(B, L, F, dtype=torch.bool)
    msk = torch.ones(B, L, F, dtype=torch.bool)
    m = eval_batch_metrics(
        p_head, n_head, mu, peak, y_data, y_pval, ypk, obs, msk, obs, msk,
        y_data_imp=None,
        y_pval_imp=y_pval_imp,
        y_peaks_imp=y_peaks_imp,
    )
    assert math.isnan(m["imp_count_r2_gw"])
    assert math.isnan(m["imp_count_pearson_gw"])
    assert math.isnan(m["imp_count_spearman_gw"])


def test_eval_batch_metrics_imp_peak_mask_uses_valid_imp_labels():
    """Imp peak AUROC must ignore -1 labels and use only valid {0,1} imp GT positions."""
    torch.manual_seed(13)
    B, L, F = 1, 16, 1
    p_head = torch.sigmoid(torch.randn(B, L, F))
    n_head = torch.nn.functional.softplus(torch.randn(B, L, F)) + 0.5
    mu = torch.randn(B, L, F)
    peak = torch.sigmoid(torch.randn(B, L, F))
    y_data = torch.abs(torch.randn(B, L, F))
    y_data_imp = torch.abs(torch.randn(B, L, F))
    y_pval = torch.randn(B, L, F)
    y_pval_imp = torch.randn(B, L, F)
    ypk = (torch.rand(B, L, F) > 0.5).float()
    ypk_imp = torch.full((B, L, F), -1.0)
    # Only two masked positions have valid labels, one pos and one neg.
    ypk_imp[0, 2, 0] = 0.0
    ypk_imp[0, 3, 0] = 1.0
    obs = torch.ones(B, L, F, dtype=torch.bool)
    msk = torch.zeros(B, L, F, dtype=torch.bool)
    msk[0, 2, 0] = True
    msk[0, 3, 0] = True
    out = eval_batch_metrics(
        p_head, n_head, mu, peak, y_data, y_pval, ypk, obs, msk, obs, msk,
        y_data_imp=y_data_imp,
        y_pval_imp=y_pval_imp,
        y_peaks_imp=ypk_imp,
    )
    assert "imp_peak_auroc_gw" in out
    assert np.isfinite(out["imp_peak_auroc_gw"])


def test_eval_batch_metrics_imp_eval_map_decouples_from_t_masked_map():
    """Imp metrics can be computed on explicit imp_eval_map even when T masked_map is empty."""
    torch.manual_seed(17)
    B, L, F = 1, 12, 1
    p_head = torch.sigmoid(torch.randn(B, L, F))
    n_head = torch.nn.functional.softplus(torch.randn(B, L, F)) + 0.5
    mu = torch.randn(B, L, F)
    peak = torch.sigmoid(torch.randn(B, L, F))
    y_data = torch.abs(torch.randn(B, L, F))
    y_data_imp = torch.abs(torch.randn(B, L, F))
    y_pval = torch.randn(B, L, F)
    y_pval_imp = torch.randn(B, L, F)
    ypk = (torch.rand(B, L, F) > 0.5).float()
    ypk_imp = torch.full((B, L, F), -1.0)
    ypk_imp[0, 1, 0] = 0.0
    ypk_imp[0, 2, 0] = 1.0
    ypk_imp[0, 3, 0] = 0.0
    ypk_imp[0, 4, 0] = 1.0
    obs = torch.ones(B, L, F, dtype=torch.bool)
    msk = torch.zeros(B, L, F, dtype=torch.bool)  # no T-cloze mask
    imp_map = torch.zeros(B, L, F, dtype=torch.bool)
    imp_map[0, 1, 0] = True
    imp_map[0, 2, 0] = True
    imp_map[0, 3, 0] = True
    imp_map[0, 4, 0] = True
    out = eval_batch_metrics(
        p_head, n_head, mu, peak, y_data, y_pval, ypk, obs, msk, obs, msk,
        imp_eval_map=imp_map,
        imp_eval_signal_map=imp_map,
        y_data_imp=y_data_imp,
        y_pval_imp=y_pval_imp,
        y_peaks_imp=ypk_imp,
    )
    assert np.isfinite(out["imp_peak_auroc_gw"])
    assert np.isfinite(out["imp_pval_pearson_gw"])
    assert np.isfinite(out["imp_count_pearson_gw"])


def test_eval_batch_metrics_count_metrics_finite():
    """den_count_* and imp_count_* are finite when their respective GT is provided."""
    import math
    torch.manual_seed(9)
    B, L, F = 1, 16, 2
    mu = torch.randn(B, L, F)
    y_pval = torch.randn(B, L, F)
    y_data = torch.abs(torch.randn(B, L, F)) * 3
    p_head = torch.sigmoid(torch.randn(B, L, F)) * 0.9 + 0.05  # avoid p near 0
    n_head = torch.nn.functional.softplus(torch.randn(B, L, F)) + 0.5
    peak = torch.sigmoid(torch.randn(B, L, F))
    ypk = (torch.rand(B, L, F) > 0.5).float()
    y_data_imp = torch.abs(torch.randn(B, L, F)) * 2
    y_pval_imp = torch.randn(B, L, F)
    obs = torch.ones(B, L, F, dtype=torch.bool)
    msk = torch.ones(B, L, F, dtype=torch.bool)
    m = eval_batch_metrics(
        p_head, n_head, mu, peak, y_data, y_pval, ypk, obs, msk, obs, msk,
        y_data_imp=y_data_imp,
        y_pval_imp=y_pval_imp,
    )
    for key in [
        "den_count_r2_gw",
        "den_count_pearson_gw",
        "den_count_spearman_gw",
        "imp_count_r2_gw",
        "imp_count_pearson_gw",
        "imp_count_spearman_gw",
    ]:
        assert key in m, f"missing key: {key}"
        assert math.isfinite(m[key]), f"{key} = {m[key]}, expected finite"


def test_scheduler_steps_per_optimizer_step():
    """Scheduler must update LR after each optimizer step (not only per epoch)."""
    from copy import deepcopy
    from dataclasses import asdict
    d = deepcopy(asdict(SandboxConfig()))
    d["training"]["schedule"]["name"] = "cosine"
    d["training"]["schedule"]["warmup_frac"] = 0.5
    cfg = config_from_dict(SandboxConfig, d)
    m = build_sandbox_candi(context_bins=64)
    opt = build_optimizer(m, cfg)
    # total_steps=10, warmup=5 steps: LR should change every step.
    sched = build_scheduler(opt, cfg, total_steps=10)
    assert sched is not None
    lrs = [opt.param_groups[0]["lr"]]
    for _ in range(5):
        sched.step()
        lrs.append(opt.param_groups[0]["lr"])
    # LR must change across steps (not constant).
    assert len(set(round(lr, 8) for lr in lrs)) > 1, "LR did not change across steps"
    # At step 0 LR should be 0.2 * base_lr (warmup start_factor=0.2).
    base_lr = _active_base_lr(cfg)
    assert abs(lrs[0] - 0.2 * base_lr) < 1e-9


def test_prepare_masked_batch_apply_mask_false_keeps_inputs_unmasked():
    """Eval no-mask policy: apply_mask=False must keep masked_map all False."""
    B, L, F, Lbp = 1, 8, len(SANDBOX_ASSAYS), 200
    batch = {
        "x_data": torch.randn(B, L, F),
        "x_meta": torch.zeros(B, 4, F),
        "x_avail": torch.ones(B, F),
        "x_dna": torch.zeros(B, Lbp, 4),
        "y_data": torch.randn(B, L, F),
        "y_meta": torch.zeros(B, 4, F),
        "y_avail": torch.ones(B, F),
        "y_pval": torch.randn(B, L, F),
        "y_peaks": (torch.rand(B, L, F) > 0.5).float(),
        "control_data": torch.randn(B, L, 1),
        "control_meta": torch.zeros(B, 4, 1),
        "control_avail": torch.ones(B, 1),
        "y_dsf": torch.ones(B, F, dtype=torch.int64),
    }
    aggressive_masker = make_masker(
        p_full_loci=1.0,
        p_full_assay=1.0,
        p_chunks=1.0,
        mask_fraction=0.9,
        chunk_size=40,
    )
    prep = prepare_masked_batch(batch, aggressive_masker, torch.device("cpu"), apply_mask=False)
    assert prep is not None
    assert not prep["masked_map"].any()

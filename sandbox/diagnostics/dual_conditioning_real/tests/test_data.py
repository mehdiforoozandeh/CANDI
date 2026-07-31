"""Block test: real data processing via SandboxH5Dataset on the mini fixture (q19 §Validation).

Verifies batch keys/shapes, the 4-row metadata layout, per-assay independent DSF wiring, T_-only train
pool, V/B ground-truth on eval, and the 12-target inventory.
"""
from __future__ import annotations

import itertools

import h5py
import numpy as np
import pytest
import torch

from sandbox.data import SandboxH5Dataset
from sandbox.diagnostics.dual_conditioning_real.tests.fixture import (
    bake_fixture, FIX_H5, HELD_OUT_TARGETS, derive_held_out_targets,
)

A = 8
L = 768
G = 25 * L


@pytest.fixture(scope="module")
def fix():
    return bake_fixture()


def _first_batch(ds):
    return next(iter(ds))


def test_batch_keys_and_shapes(fix):
    ds = SandboxH5Dataset(fix, "type1_chr19", train=True, batch_size=2, biosample_prefix="T_",
                          dsf_sampling="uniform", seed=0)
    b = _first_batch(ds)
    B = b["x_data"].shape[0]
    assert b["x_data"].shape == (B, L, A)
    assert b["y_data"].shape == (B, L, A)
    assert b["x_meta"].shape == (B, 4, A) and b["y_meta"].shape == (B, 4, A)
    assert b["x_dna"].shape == (B, G, 4)
    assert b["control_data"].shape == (B, L, 1) and b["control_meta"].shape == (B, 4, 1)
    assert b["x_avail"].shape == (B, A) and b["y_avail"].shape == (B, A)
    assert b["x_dsf"].shape == (B, A) and b["y_dsf"].shape == (B, A)
    for k in ("x_data", "y_data", "x_meta", "y_meta", "x_dna", "control_data", "control_meta"):
        assert torch.isfinite(b[k]).all(), k


def test_metadata_row_layout(fix):
    ds = SandboxH5Dataset(fix, "type1_chr19", train=True, batch_size=2, biosample_prefix="T_",
                          dsf_sampling="uniform", seed=0)
    b = _first_batch(ds)
    x_avail = b["x_avail"].bool()
    for a in range(A):
        present = x_avail[:, a]
        if present.any():
            # assay_id row (row 1) == column index for present assays (identity ordering)
            assert torch.all(b["x_meta"][present, 1, a] == float(a))
            assert torch.isfinite(b["x_meta"][present, :, a]).all()
        absent = ~present
        if absent.any():
            # absent assays are the -1 sentinel across all 4 rows
            assert torch.all(b["x_meta"][absent, :, a] == -1.0)


def test_per_assay_independent_dsf_and_depth_wiring(fix):
    # deterministic eval batch: shuffle off, batch covers all chr21 windows, biosample = first eval pair.
    ds = SandboxH5Dataset(fix, "type1_chr19", train=False, batch_size=4, biosample_prefix="T_",
                          dsf_sampling="uniform", seed=0, shuffle=False,
                          eval_include_vb_ground_truth=True)
    b = _first_batch(ds)
    bio = b["biosample_name"]
    x_dsf = b["x_dsf"].numpy(); y_dsf = b["y_dsf"].numpy()
    # per-assay independence: at least one assay has x_dsf != y_dsf somewhere in the batch
    assert (x_dsf != y_dsf).any(), "uniform DSF should decorrelate x_dsf and y_dsf per assay"
    # depth wiring: x_meta depth == meta_dsf{x_dsf} row 0; y_meta depth == meta_dsf{y_dsf} row 0
    with h5py.File(fix, "r") as h:
        g = h["biosamples"][bio.replace("/", "_")]
        md = {k: np.array(g[f"meta_dsf{k}"]) for k in (1, 2, 4, 8)}
        counts = {k: np.array(g[f"counts_dsf{k}"]) for k in (1, 2, 4, 8)}
    wi = sorted(ds._eval_indices)[:4]
    x_avail = b["x_avail"].numpy()
    for j in range(b["x_data"].shape[0]):
        for a in range(A):
            if x_avail[j, a] <= 0:
                continue
            xd, yd = int(x_dsf[j, a]), int(y_dsf[j, a])
            assert abs(float(b["x_meta"][j, 0, a]) - float(md[xd][0, a])) < 1e-4
            assert abs(float(b["y_meta"][j, 0, a]) - float(md[yd][0, a])) < 1e-4
            # x_data drawn from counts_dsf{x_dsf}
            exp = counts[xd][wi[j], :, a].astype(np.float32)
            assert np.allclose(b["x_data"][j, :, a].numpy(), exp), (bio, j, a, xd)


def test_train_pool_is_T_only(fix):
    ds = SandboxH5Dataset(fix, "type1_chr19", train=True, batch_size=2, biosample_prefix="T_",
                          dsf_sampling="uniform", seed=0)
    assert all(bn.startswith("T_") for bn in ds._bios_candidates())
    for b in itertools.islice(iter(ds), 3):
        assert b["biosample_name"].startswith("T_")


def test_eval_yields_vb_ground_truth(fix):
    ds = SandboxH5Dataset(fix, "type1_chr19", train=False, batch_size=2, biosample_prefix="T_",
                          dsf_sampling="off", seed=0, shuffle=False, eval_include_vb_ground_truth=True)
    found = None
    for b in itertools.islice(iter(ds), 20):
        if "y_data_imp" in b:
            found = b
            break
    assert found is not None, "expected at least one eval batch with V/B ground truth"
    B = found["x_data"].shape[0]
    assert found["y_data_imp"].shape == (B, L, A)
    assert found["y_meta_imp"].shape == (B, 4, A)
    assert found["imp_biosample_name"].startswith(("V_", "B_"))


def test_held_out_target_inventory(fix):
    got = derive_held_out_targets(fix)
    assert got == HELD_OUT_TARGETS
    labels = [t[4] for t in got]
    assert labels.count("paired") == 9 and labels.count("single") == 3

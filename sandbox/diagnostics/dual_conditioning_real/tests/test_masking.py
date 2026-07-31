"""Block test: cloze masking + control invariants + prompt construction (q19 §Validation).

Reuses sandbox `make_masker`/`prepare_masked_batch` and train `_build_mixed_meta` unchanged; asserts the
obs/imp == unmasked/masked semantics, that control is never masked, and the V/B prompt swap.
"""
from __future__ import annotations

import itertools

import torch

from sandbox.batch import CLOZE, MISSING, make_masker, prepare_masked_batch
from sandbox.data import SandboxH5Dataset
from sandbox.train import _build_mixed_meta
from sandbox.diagnostics.dual_conditioning_real.tests.fixture import bake_fixture

A = 8


def _raw_batch():
    ds = SandboxH5Dataset(bake_fixture(), "type1_chr19", train=True, batch_size=2, biosample_prefix="T_",
                          dsf_sampling="uniform", seed=0)
    return next(iter(ds))


def test_cloze_masking_and_obs_imp_semantics():
    torch.manual_seed(0)
    masker = make_masker(p_full_assay=1.0)
    prep = prepare_masked_batch(_raw_batch(), masker, torch.device("cpu"), apply_mask=True)
    assert prep is not None
    xin = prep["x_data"]                       # [B, L, A+1] (control appended)
    masked_map, observed_map = prep["masked_map"], prep["observed_map"]   # [B, L, A]
    assert bool(masked_map.any()), "p_full_assay=1.0 should cloze at least one assay"
    # masked positions carry the CLOZE sentinel; obs/imp == unmasked/masked (disjoint)
    sig = xin[:, :, :A]
    assert torch.all(sig[masked_map] == CLOZE)
    assert torch.all((sig[observed_map] != CLOZE) & (sig[observed_map] != MISSING))
    assert not bool((masked_map & observed_map).any())
    # y_data is retained (not masked) at masked positions
    assert torch.isfinite(prep["y_data"]).all()


def test_control_never_masked():
    torch.manual_seed(0)
    masker = make_masker(p_full_assay=1.0)
    prep = prepare_masked_batch(_raw_batch(), masker, torch.device("cpu"), apply_mask=True)
    # control is the appended column at index A in x_data/x_meta; the masker only touches the A-assay block
    ctrl_data = prep["x_data"][:, :, A]
    ctrl_meta = prep["x_meta"][:, :, A]
    assert not bool((ctrl_data == CLOZE).any())
    assert not bool((ctrl_meta == CLOZE).any())


def test_build_mixed_meta_swaps_masked_slots():
    B, L, F = 2, 4, A
    t_meta = torch.full((B, 4, F), 10.0)
    vb_meta = torch.full((B, 4, F), 20.0)
    masked_map = torch.zeros(B, L, F, dtype=torch.bool)
    masked_map[:, :, 3] = True                 # assay 3 cloze-masked
    mixed = _build_mixed_meta(t_meta, vb_meta, masked_map)
    assert torch.all(mixed[:, :, 3] == 20.0)   # masked slot takes V/B natural meta
    for a in range(F):
        if a != 3:
            assert torch.all(mixed[:, :, a] == 10.0)   # unmasked slots keep T_ meta


def test_missing_assays_get_sentinel_meta():
    # truly-missing assays (y_avail==0) carry the -1 sentinel across all metadata rows in the raw batch.
    ds = SandboxH5Dataset(bake_fixture(), "type1_chr19", train=True, batch_size=2, biosample_prefix="T_",
                          dsf_sampling="uniform", seed=0)
    b = next(iter(ds))
    y_avail = b["y_avail"].bool()
    for a in range(A):
        missing = ~y_avail[:, a]
        if missing.any():
            assert torch.all(b["y_meta"][missing, :, a] == -1.0)

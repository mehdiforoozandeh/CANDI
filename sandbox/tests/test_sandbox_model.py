"""CPU smoke: forward + backward on synthetic batch."""
from __future__ import annotations

import pytest
import torch

from sandbox.batch import make_masker, prepare_masked_batch
from sandbox.losses import SandboxCompositeLoss
from sandbox.model import build_sandbox_candi
from candi_loss import CANDI_LOSS


def _synthetic_batch(*, L: int, F: int = 8, Lbp: int, B: int = 2) -> dict:
    torch.manual_seed(0)
    x_data = torch.abs(torch.randn(B, L, F)) * 10
    x_meta = torch.zeros(B, 4, F)
    x_meta[:, 0, :] = 22.0
    x_meta[:, 1, :] = torch.arange(F).float()
    x_meta[:, 2, :] = 50.0
    x_meta[:, 3, :] = 1.0
    # Exercise MetadataEncoder missing (-1) / cloze (-2) paths so all special embeddings get grad.
    x_meta[:, 0, 0] = -1.0
    x_meta[:, 0, 1] = -2.0
    x_meta[:, 2, 2] = -1.0
    x_meta[:, 2, 3] = -2.0
    x_meta[:, 3, 4] = -1.0
    x_meta[:, 3, 5] = -2.0
    x_meta[:, 1, 6] = -1.0
    x_meta[:, 1, 7] = -2.0
    x_avail = torch.ones(B, F)
    x_dna = torch.zeros(B, Lbp, 4)
    x_dna[..., 0] = 1.0
    y_data = x_data.clone()
    y_meta = x_meta.clone()
    y_meta[:, 0, -1] = -1.0
    y_meta[:, 1, -2] = -2.0
    y_avail = torch.ones(B, F)
    y_pval = torch.log1p(y_data.float())
    y_peaks = (y_data > y_data.mean()).long().float()
    control_data = torch.abs(torch.randn(B, L, 1)) * 0.1
    control_meta = torch.zeros(B, 4, 1)
    control_meta[:, 0, 0] = 22.0
    control_meta[:, 1, 0] = float(F)
    control_avail = torch.ones(B, 1)
    y_dsf = torch.ones(B, F, dtype=torch.int64)
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
        "x_dsf": torch.ones(B, F, dtype=torch.int64),
        "y_dsf": y_dsf,
        "control_x_dsf": torch.ones(B, dtype=torch.int64),
        "biosample_name": "T_test",
        "region_type": torch.zeros(B, dtype=torch.uint8),
    }


def _run_forward_backward(
    *,
    L: int,
    separate_decoders: bool,
    mask_stem: bool,
    dist_type: str,
    cand_dist: str,
) -> None:
    F = 8
    Lbp = L * 25
    batch = _synthetic_batch(L=L, F=F, Lbp=Lbp)
    device = torch.device("cpu")
    # Avoid p_full_assay=1.0: it overwrites all metadata with cloze (-2), so missing (-1) tokens never reach the encoder.
    masker = make_masker(p_full_assay=0.0, p_full_loci=0.0, p_chunks=0.0)
    prep = prepare_masked_batch(batch, masker, device)
    assert prep is not None

    model = build_sandbox_candi(
        context_bins=L,
        separate_decoders=separate_decoders,
        mask_stem=mask_stem,
        dist_type=dist_type,
    ).to(device)
    crit = SandboxCompositeLoss(CANDI_LOSS(dist_type=cand_dist)).to(device)

    out = model(
        prep["x_data"],
        prep["x_dna"],
        prep["x_meta"],
        prep["y_meta"],
        query_mask=prep["query_mask"],
        query_mask_signal=prep["query_mask_signal"],
    )
    p, n, mu, scale, df, peak = out
    for t in (p, n, mu, scale, peak):
        assert t is not None and torch.isfinite(t).all()
    if df is not None:
        assert torch.isfinite(df).all()
    loss, _ = crit(
        p,
        n,
        mu,
        scale,
        df,
        peak,
        prep["y_data"],
        prep["y_pval"],
        prep["y_peaks"],
        prep["observed_map"],
        prep["masked_map"],
        prep["signal_observed_map"],
        prep["signal_masked_map"],
        global_step=0,
    )
    loss.backward()
    assert torch.isfinite(loss).item()
    for name, par in model.named_parameters():
        if not par.requires_grad:
            continue
        assert par.grad is not None, name
        assert torch.isfinite(par.grad).all(), name
        assert par.grad.detach().abs().sum() > 0.0, name


@pytest.mark.parametrize("separate_decoders", (True, False))
def test_build_model_forward_backward(separate_decoders: bool):
    _run_forward_backward(
        L=64,
        separate_decoders=separate_decoders,
        mask_stem=True,
        dist_type="gaussian",
        cand_dist="gaussian",
    )


def test_gate_c_alt_mask_stem_and_student_t():
    """Gate C (v): smoke alternate mask_stem + heavy-tail head once each."""
    _run_forward_backward(
        L=64,
        separate_decoders=True,
        mask_stem=False,
        dist_type="student_t",
        cand_dist="studentst",
    )

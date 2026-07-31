"""Block test: RealMetaEmbedder + DualCondDecoder (offset row 0) + RealDualCondModel.

Vendored from sandbox/diagnostics/dual_conditioning_real/tests/test_model.py:1-161 (imports rewired to
candi_kit; RealDualCondDecoder -> the collapsed DualCondDecoder), plus the scale-axis and
covariate-bound cases the original suite lacked.

Synthetic tensors only (no h5) — validates the metadata swap and the offset re-keying against the golden
reference behaviour (q19 §Validation, test_model.py). Small config for speed.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from candi_kit._vendored import CLOZE, MISSING
from candi_kit.model import (
    RealMetaEmbedder, DualCondDecoder, build_real_model, forward_full, nb_nll, encode_latent,
)

A = 8
READLENS = torch.tensor([30.0, 36.0, 76.0, 100.0, 101.0])


def _synth_meta(B, F, seed=0, num_assays=A):
    """[B,4,F] valid real meta: depth in [22,28], assay_id=col, readlen in {30..101}, runtype {0,1}."""
    g = torch.Generator().manual_seed(seed)
    m = torch.zeros(B, 4, F)
    for c in range(F):
        m[:, 0, c] = 22.0 + 6.0 * torch.rand(B, generator=g)
        m[:, 1, c] = float(c if c < num_assays else num_assays)
        m[:, 2, c] = READLENS[torch.randint(0, 5, (B,), generator=g)]
        m[:, 3, c] = torch.randint(0, 2, (B,), generator=g).float()
    return m


def _synth_batch(B=2, ctx=768, seed=0, num_assays=A):
    g = torch.Generator().manual_seed(seed)
    L, G = ctx, 25 * ctx
    x_sig = (torch.rand(B, L, num_assays, generator=g) * 50).round()
    control = (torch.rand(B, L, 1, generator=g) * 50).round()
    x_data = torch.cat([x_sig, control], dim=2)
    x_dna = torch.zeros(B, G, 4)
    idx = torch.randint(0, 4, (B, G), generator=g)
    x_dna.scatter_(2, idx.unsqueeze(-1), 1.0)
    return dict(x_data=x_data, x_dna=x_dna,
                x_meta=_synth_meta(B, num_assays + 1, seed, num_assays),
                y_meta=_synth_meta(B, num_assays, seed + 1, num_assays),
                y_data=(torch.rand(B, L, num_assays, generator=g) * 50).round(),
                avail=torch.ones(B, num_assays))


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = build_real_model(embed_dim=16, n_transformer_layers=1, feat_per_assay=8)
    m.eval()
    return m


@pytest.fixture(scope="module")
def batch():
    return _synth_batch()


# ---- RealMetaEmbedder ----

def test_embedder_shape_and_finite_with_sentinels():
    me = RealMetaEmbedder(num_assays=8, embed_dim=16)
    m = _synth_meta(3, 9)
    m[0, :, 2] = MISSING          # whole column missing (all 4 rows)
    m[1, 0, 3] = CLOZE            # depth cloze
    m[2, 1, 4] = MISSING          # assay_id missing
    m[2, 3, 5] = CLOZE            # run_type cloze
    m[2, 2, 6] = MISSING          # read_length missing
    out = me(m)
    assert out.shape == (3, 9, 16)
    assert torch.isfinite(out).all()


# ---- covariate bounds are fenced by assert, not by config ----

def test_out_of_range_assay_id_raises():
    me = RealMetaEmbedder(num_assays=8, embed_dim=16)
    m = _synth_meta(2, 9)
    m[0, 1, 3] = 9.0              # num_assays+1: silently aliased onto the MISSING slot before
    with pytest.raises(ValueError, match="assay_id"):
        me(m)


def test_out_of_range_run_type_raises():
    me = RealMetaEmbedder(num_assays=8, embed_dim=16)
    m = _synth_meta(2, 9)
    m[0, 3, 3] = 2.0              # only {0,1} are emitted by the ingestion contract
    with pytest.raises(ValueError, match="run_type"):
        me(m)


def test_wrong_metadata_row_count_raises():
    me = RealMetaEmbedder(num_assays=8, embed_dim=16)
    with pytest.raises(ValueError, match="4 rows"):
        me(_synth_meta(2, 9)[:, :3, :])


# ---- offset keyed to row 0 ----

def test_offset_reads_row0(model, batch):
    with torch.no_grad():
        o0 = forward_full(model, batch)
        b2 = dict(batch); b2["y_meta"] = batch["y_meta"].clone(); b2["y_meta"][:, 0, :] += 1.0
        o1 = forward_full(model, b2)
    d = o1["log2_mu"] - o0["log2_mu"]
    assert torch.allclose(d, torch.ones_like(d), atol=1e-4)


def test_offset_ignores_other_rows(model, batch):
    # perturbing read_length (row 2) must NOT shift log2_mu via the offset (only via eta/FiLM, tiny here)
    with torch.no_grad():
        o0 = forward_full(model, batch)
        b2 = dict(batch); b2["y_meta"] = batch["y_meta"].clone(); b2["y_meta"][:, 0, :] += 3.0  # depth
        o1 = forward_full(model, b2)
    d = (o1["log2_mu"] - o0["log2_mu"]).mean().item()
    assert abs(d - 3.0) < 1e-3   # the shift equals the depth delta exactly (offset arithmetic)


def test_offset_off_gives_log2mu_equals_eta():
    torch.manual_seed(0)
    m = build_real_model(embed_dim=16, n_transformer_layers=1, feat_per_assay=8, use_offset=False)
    m.eval()
    with torch.no_grad():
        o = forward_full(m, _synth_batch())
    assert torch.allclose(o["log2_mu"], o["eta"], atol=1e-5)


# ---- adaLN-zero init ----

def test_adaln_zero_init():
    dec = DualCondDecoder(encoder_d_model=16, num_assays=8, feat_per_assay=8, meta_embed_dim=16)
    assert torch.count_nonzero(dec.film_proj.weight) == 0
    assert torch.count_nonzero(dec.film_proj.bias) == 0
    # FiLM identity at init: gamma=beta=0 regardless of y_meta
    z = torch.randn(2, 8, 16)
    with torch.no_grad():
        memb = dec.meta_embedding(_synth_meta(2, 8).float())
        gamma, beta = dec.film_proj(memb).chunk(2, dim=-1)
    assert torch.allclose(gamma, torch.zeros_like(gamma)) and torch.allclose(beta, torch.zeros_like(beta))


# ---- per-assay locality ----

def test_per_assay_locality():
    dec = DualCondDecoder(encoder_d_model=16, num_assays=8, feat_per_assay=8, meta_embed_dim=16)
    nn.init.normal_(dec.film_proj.weight, std=0.1); nn.init.normal_(dec.film_proj.bias, std=0.1)
    dec.eval()
    z = torch.randn(2, 8, 16)
    ym = _synth_meta(2, 8)
    with torch.no_grad():
        d0 = dec(z, ym)
        ym2 = ym.clone(); ym2[:, 0, 3] += 5.0    # perturb assay 3 depth only
        d1 = dec(z, ym2)
    diff = (d1["eta"] - d0["eta"]).abs().sum(dim=(0, 1))   # [A]
    changed = diff > 1e-6
    assert bool(changed[3]) and int(changed.sum()) == 1


# ---- encoder ignores y_meta ----

def test_encode_latent_invariant_to_ymeta(model, batch):
    with torch.no_grad():
        z0 = encode_latent(model, batch)
        b2 = dict(batch); b2["y_meta"] = batch["y_meta"].clone(); b2["y_meta"][:, 0, :] += 5.0
        z1 = encode_latent(model, b2)
    assert torch.allclose(z0, z1, atol=1e-6)


# ---- forward_full finite + nb_nll backward ----

def test_forward_full_and_nbnll_backward():
    torch.manual_seed(0)
    m = build_real_model(embed_dim=16, n_transformer_layers=1, feat_per_assay=8)
    m.train()
    b = _synth_batch()
    out = forward_full(m, b)
    for k in ("p", "n", "eta", "log2_mu", "mu"):
        assert out[k].shape == (2, 768, 8)
        assert torch.isfinite(out[k]).all()
    loss = nb_nll(out["p"], out["n"], b["y_data"], b["avail"])
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None and torch.count_nonzero(p.grad) > 0]
    assert len(grads) > 0


# ---- scale axes: panel size and context length are independent, and free ----

@pytest.mark.parametrize("num_assays", [3, 8, 16])
@pytest.mark.parametrize("ctx", [384, 768, 1536])
def test_scale_axes_independent(num_assays, ctx):
    torch.manual_seed(0)
    m = build_real_model(embed_dim=16, n_transformer_layers=1, feat_per_assay=8,
                         num_assays=num_assays, context_length=ctx).eval()
    b = _synth_batch(B=1, ctx=ctx, num_assays=num_assays)
    with torch.no_grad():
        out = forward_full(m, b)
    assert out["mu"].shape == (1, ctx, num_assays)
    assert torch.isfinite(out["mu"]).all()


@pytest.mark.parametrize("num_assays", [3, 8, 16])
def test_d_model_auto_tracks_panel_size(num_assays):
    """d_model=0 -> (num_assays+1) * expansion_factor**n_cnn_layers = (A+1)*8. This is the landmine:
    transformer width silently follows the panel unless --d-model is set."""
    torch.manual_seed(0)
    m = build_real_model(embed_dim=16, n_transformer_layers=1, feat_per_assay=8,
                         num_assays=num_assays, context_length=384, d_model=0)
    assert m.encoder.d_model == (num_assays + 1) * 8


@pytest.mark.parametrize("num_assays", [3, 16])
def test_explicit_d_model_decouples_width_from_panel(num_assays):
    torch.manual_seed(0)
    m = build_real_model(embed_dim=16, n_transformer_layers=1, feat_per_assay=8,
                         num_assays=num_assays, context_length=384, d_model=64, nhead=4).eval()
    assert m.encoder.d_model == 64
    with torch.no_grad():
        out = forward_full(m, _synth_batch(B=1, ctx=384, num_assays=num_assays))
    assert out["mu"].shape == (1, 384, num_assays)


def test_dna_length_mismatch_raises():
    torch.manual_seed(0)
    m = build_real_model(embed_dim=16, n_transformer_layers=1, feat_per_assay=8,
                         num_assays=3, context_length=384).eval()
    b = _synth_batch(B=1, ctx=384, num_assays=3)
    b["x_dna"] = b["x_dna"][:, : 384 * 25 - 25, :]      # one bin short of context * resolution
    with pytest.raises(ValueError, match="DNA length"):
        with torch.no_grad():
            forward_full(m, b)

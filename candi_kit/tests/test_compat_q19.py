"""GATE A as a pytest: the vendoring is provably a COPY, not a reimplementation.

The single most important test in the kit. Under `torch.manual_seed(0)` the q19 model must come out
of `candi_kit` bit-for-bit identical to the historical `sandbox/diagnostics/dual_conditioning_real`
build — same parameter count, same state_dict digest, same forward output, and the four recorded
checkpoints must load with `strict=True`.

All anchors below were measured against the ORIGINAL implementation
(sandbox/diagnostics/dual_conditioning_real/model_real.py, torch 2.6.0, x-transformers 2.11.23).
If any of them fails, do NOT relax the anchor — the construction order in candi_kit/model.py has
drifted. See .BUILD_PLAN RISKS: the placeholder embedder must still be constructed and stay 8 x E,
film_proj must be constructed-then-zeroed, and V2Encoder must be built before its metadata_embedding
is replaced.
"""
from __future__ import annotations

import hashlib
import inspect
import os
from pathlib import Path

import pytest
import torch

from candi_kit.compat import Q19, build_compat_q19, state_dict_sha1, synthetic_batch
from candi_kit.model import DualCondDecoder, RealDualCondModel, forward_full

# --- measured anchors (ORIGINAL implementation, seed 0, compat.synthetic_batch) -------------------
N_PARAMS = 3_103_194
SD_SHA1 = "fd0e9493ac92a15f"          # identical for both arms (use_offset holds no parameters)
FWD_SHA1 = {True: "8c9271f29299b2e0", False: "cd18d437a13c041f"}
MU_MEAN = {True: 2.604400634765625, False: 1.1116695404052734}
ETA_STD = 0.15498559176921844         # eta is offset-independent -> same for both arms
XT_VERSION = "2.11.23"

# The four historical checkpoints: name -> use_offset.
CKPTS = {
    "main_s0_perassay.ckpt": True,
    "offoff_s0_perassay.ckpt": False,
    "wd0_on_s0.ckpt": True,
    "wd0_off_s0.ckpt": False,
}
CKPT_DIR = Path(os.environ.get(
    "CANDI_KIT_CKPT_DIR",
    "/project/6014832/mforooz/EpiDenoise/sandbox/diagnostics/dual_conditioning_real/results"))


def _digest(named_tensors) -> str:
    h = hashlib.sha1()
    for k in sorted(named_tensors):
        h.update(k.encode())
        h.update(named_tensors[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


@pytest.fixture(scope="module")
def golden_outputs():
    # `compat.synthetic_batch` is THE frozen Gate-A batch — its draw order must not change, or
    # FWD_SHA1/MU_MEAN below become meaningless.
    # Single-threaded: CPU float reductions are thread-count dependent, so the 0-ULP digest is only
    # defined at one thread. (Verified: OMP_NUM_THREADS=1 vs 64 give different bits, same values.)
    torch.set_num_threads(1)
    batch = synthetic_batch()
    out = {}
    for use_offset in (True, False):
        m = build_compat_q19(use_offset=use_offset).eval()
        with torch.no_grad():
            out[use_offset] = (m, forward_full(m, batch))
    return out


# ---------------------------------------------------------------------------
# A1 — construction is bit-identical
# ---------------------------------------------------------------------------

def test_xtransformers_pinned():
    from importlib.metadata import version
    got = version("x-transformers")
    assert got == XT_VERSION, (
        f"x-transformers {got} != {XT_VERSION}. The package has no __version__ attribute and its "
        f"internals fix both the transformer state_dict key names and the per-module init order, so a "
        f"minor bump breaks checkpoint loading and the RNG stream.")


@pytest.mark.parametrize("use_offset", [True, False])
def test_param_count_and_state_dict_sha1(use_offset):
    m = build_compat_q19(use_offset=use_offset)
    n = sum(p.numel() for p in m.parameters())
    assert n == N_PARAMS, f"{n:,} params != {N_PARAMS:,} — the module graph changed"
    assert state_dict_sha1(m)[:16] == SD_SHA1, "state_dict digest drifted: constructor RNG order changed"
    assert _digest(m.state_dict())[:16] == SD_SHA1


# ---------------------------------------------------------------------------
# A2 — the four historical checkpoints load strictly
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,use_offset", sorted(CKPTS.items()))
def test_historical_checkpoint_loads_strict(name, use_offset):
    path = CKPT_DIR / name
    if not path.exists():
        pytest.skip(f"checkpoint dir absent ({path}); set CANDI_KIT_CKPT_DIR to run this gate")
    sd = torch.load(path, map_location="cpu", weights_only=False)
    m = build_compat_q19(use_offset=use_offset)
    inc = m.load_state_dict(sd, strict=False)
    assert not inc.missing_keys, f"missing keys: {inc.missing_keys[:8]}"
    assert not inc.unexpected_keys, f"unexpected keys: {inc.unexpected_keys[:8]}"
    m.load_state_dict(sd, strict=True)          # also shape-exact


# ---------------------------------------------------------------------------
# A3 — forward matches the golden output
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("use_offset", [True, False])
def test_forward_golden_digest(golden_outputs, use_offset):
    _m, out = golden_outputs[use_offset]
    got = _digest(out)
    assert got == FWD_SHA1[use_offset], (
        f"forward digest {got} != golden {FWD_SHA1[use_offset]}. If test_forward_golden_values passes, "
        f"this is CPU float-reduction drift (thread count / BLAS build), not a construction bug.")


@pytest.mark.parametrize("use_offset", [True, False])
def test_forward_golden_values(golden_outputs, use_offset):
    _m, out = golden_outputs[use_offset]
    assert out["mu"].shape == (2, Q19["context_length"], Q19["num_assays"])
    assert torch.isfinite(out["mu"]).all() and torch.isfinite(out["eta"]).all()
    assert abs(float(out["mu"].mean()) - MU_MEAN[use_offset]) < 1e-6
    assert abs(float(out["eta"].std()) - ETA_STD) < 1e-6


# ---------------------------------------------------------------------------
# A3b — source-level guard on the frozen construction order
# ---------------------------------------------------------------------------

def _order_ok(src: str, markers) -> bool:
    pos = -1
    for mk in markers:
        i = src.find(mk, pos + 1)
        if i <= pos:
            return False
        pos = i
    return True


def test_model_construction_order_frozen():
    src = inspect.getsource(RealDualCondModel.__init__)
    markers = ["V2Encoder(", "self.encoder.metadata_embedding", "DualCondDecoder("]
    assert _order_ok(src, markers), f"RealDualCondModel.__init__ must construct {markers} in order:\n{src}"


def test_decoder_construction_order_frozen():
    src = inspect.getsource(DualCondDecoder.__init__)
    markers = ["DecoderTrunk(", "_LegacyPlaceholderMetaEmbedder", "self.film_proj",
               "zeros_", "head_eta", "head_n", "RealMetaEmbedder"]
    assert _order_ok(src, markers), f"DualCondDecoder.__init__ must construct {markers} in order:\n{src}"


def test_placeholder_embedder_is_fixed_at_8():
    from candi_kit.model import _LegacyPlaceholderMetaEmbedder
    src = inspect.getsource(_LegacyPlaceholderMetaEmbedder.__init__)   # body only, not the docstring
    assert "nn.Embedding(8," in "".join(src.split()), src
    assert "num_assays" not in src, (
        "the discarded placeholder's embedding table must stay 8 x E; scaling it with num_assays "
        "diverges the global RNG stream and breaks every historical checkpoint")


def test_film_proj_is_zero_at_init():
    m = build_compat_q19(use_offset=True)
    assert torch.count_nonzero(m.decoder.film_proj.weight) == 0
    assert torch.count_nonzero(m.decoder.film_proj.bias) == 0

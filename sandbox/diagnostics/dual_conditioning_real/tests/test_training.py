"""Block test: the counts-only training loop (q19 §Validation, test_training).

Overfit sanity (loss decreases), counts-only masked NB loss (obs+imp, control excluded), warmup+cosine
LR shape, and a checkpoint round-trip.
"""
from __future__ import annotations

import numpy as np
import torch

from sandbox.batch import make_masker, prepare_masked_batch
from sandbox.data import SandboxH5Dataset
from sandbox.diagnostics.dual_conditioning_real.model_real import build_real_model, forward_full
from sandbox.diagnostics.dual_conditioning_real.run_real import train, nb_count_loss, cosine_warmup
from sandbox.diagnostics.dual_conditioning_real.tests.fixture import bake_fixture

CPU = torch.device("cpu")


def _tiny_model():
    torch.manual_seed(0)
    return build_real_model(embed_dim=16, n_transformer_layers=1, feat_per_assay=8)


def test_train_loss_decreases():
    m = _tiny_model()
    losses = train(m, bake_fixture(), CPU, epochs=1, steps_per_epoch=60, batch_size=2, lr=1e-3, seed=0)
    assert len(losses) >= 40
    assert np.mean(losses[-10:]) < np.mean(losses[:10]), (np.mean(losses[:10]), np.mean(losses[-10:]))


def test_counts_only_obs_imp_and_control_excluded():
    torch.manual_seed(0)
    ds = SandboxH5Dataset(bake_fixture(), "type1_chr19", train=True, batch_size=2, biosample_prefix="T_",
                          dsf_sampling="uniform", seed=0)
    prep = prepare_masked_batch(next(iter(ds)), make_masker(p_full_assay=1.0), CPU, apply_mask=True)
    m = _tiny_model()
    out = forward_full(m, prep)
    # counts-only head: output is [B, L, 8] assays — control column is NOT a prediction target
    assert out["mu"].shape[2] == 8
    loss, terms = nb_count_loss(out, prep)
    assert torch.isfinite(loss)
    assert terms["obs"] > 0
    if bool(prep["masked_map"].any()):
        assert terms["imp"] > 0                              # masked (cloze) positions contribute an imp term


def test_cosine_warmup_shape():
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=1.0)
    sched = cosine_warmup(opt, total_steps=100, warmup_frac=0.1, min_ratio=0.1)
    lrs = []
    for _ in range(100):
        lrs.append(opt.param_groups[0]["lr"])
        sched.step()
    assert lrs[0] < lrs[9]                                   # warmup ramps up
    assert max(lrs) == max(lrs[8:12])                        # peak near end of warmup (~step 10)
    assert lrs[-1] < lrs[10]                                 # cosine decays after warmup
    assert lrs[-1] >= 0.09                                   # floors near min_ratio


def test_checkpoint_roundtrip(tmp_path):
    m1 = _tiny_model()
    ckpt = tmp_path / "ck.pt"
    torch.save(m1.state_dict(), ckpt)
    torch.manual_seed(999)
    m2 = build_real_model(embed_dim=16, n_transformer_layers=1, feat_per_assay=8)
    m2.load_state_dict(torch.load(ckpt))
    for (k1, v1), (k2, v2) in zip(m1.state_dict().items(), m2.state_dict().items()):
        assert k1 == k2 and torch.equal(v1, v2)

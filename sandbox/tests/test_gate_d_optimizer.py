"""Gate D subset: optimizer names route to the correct torch class."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict

import torch

from sandbox.config_types import SandboxConfig, config_from_dict
from sandbox.model import build_sandbox_candi
from sandbox.train import build_optimizer


def test_optimizer_routing():
    m = build_sandbox_candi(context_bins=64)
    for nm, cls in (
        ("adam", torch.optim.Adam),
        ("adamw", torch.optim.AdamW),
        ("adamax", torch.optim.Adamax),
        ("sgd", torch.optim.SGD),
    ):
        d = deepcopy(asdict(SandboxConfig()))
        d["training"]["optimizer"]["name"] = nm
        cfg = config_from_dict(SandboxConfig, d)
        opt = build_optimizer(m, cfg)
        assert isinstance(opt, cls), nm

#!/usr/bin/env python3
"""Gradient and masking audit for CANDI v2 on real sandbox data.

Loads one batch with production assay-only masking, runs forward/backward per
loss branch, and logs per-module gradient norms.

Usage:
    python -m sandbox.diagnostics.v2_gradient_audit --device cuda
    python -m sandbox.diagnostics.v2_gradient_audit --heads count_peak --count-head depth_offset
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from sandbox.batch import make_masker, prepare_masked_batch
from sandbox.candi_v2.config import validate_v2_config
from sandbox.candi_v2.loss import build_v2_loss
from sandbox.candi_v2.model import CANDIv2
from sandbox.data import SandboxH5Dataset
from sandbox.diagnostics.real_data import default_h5_path
from sandbox.train import _global_grad_norm_for_loss
from sandbox.train_candi_v2 import load_v2_config


class GradientMonitor:
    """Backward hooks for coarse module-level grad norms."""

    MODULE_KEYS = (
        "encoder.metadata_embedding",
        "encoder.signal_tower",
        "encoder.mask_injector",
        "encoder.fusion",
        "encoder.transformer_blocks",
        "decoder.decoder_meta_embedding",
        "decoder.pre_decoder_film",
        "decoder.neg_binom_layer",
        "decoder.peak_layer",
    )

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.handles: List[Any] = []
        self.grad_norms: Dict[str, float] = {}

    def _hook(self, name: str):
        def fn(_mod, _gin, gout):
            if gout[0] is not None:
                self.grad_norms[name] = float(gout[0].detach().norm().item())
        return fn

    def register(self) -> None:
        self.clear()
        for key in self.MODULE_KEYS:
            mod = self.model
            for part in key.split("."):
                mod = getattr(mod, part, None)
                if mod is None:
                    break
            if mod is not None:
                self.handles.append(mod.register_full_backward_hook(self._hook(key)))

    def clear(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self.grad_norms = {}


def param_grad_norms(model: nn.Module, prefix: str) -> float:
    sq = 0.0
    found = False
    for name, p in model.named_parameters():
        if name.startswith(prefix) and p.grad is not None:
            sq += float(p.grad.detach().float().pow(2).sum().item())
            found = True
    return math.sqrt(sq) if found else float("nan")


def run_audit(
    *,
    device: torch.device,
    heads: str,
    count_head: str,
    h5_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    overrides = [
        f"decoder.heads={heads}",
        f"decoder.count_head={count_head}",
        "training.batch_size=4",
    ]
    cfg = load_v2_config([], overrides)
    validate_v2_config(cfg)
    model = CANDIv2(cfg).to(device)
    loss_fn = build_v2_loss(cfg).to(device)

    h5 = h5_path or default_h5_path()
    if not h5.exists():
        raise FileNotFoundError(f"HDF5 not found: {h5}")

    ds = SandboxH5Dataset(
        h5,
        "type1_chr19",
        train=True,
        batch_size=4,
        biosample_prefix="T_",
        dsf_list=(1, 2, 4, 8),
        dsf_sampling="uniform",
        seed=42,
        shuffle=False,
        h5_cache_ram=True,
    )
    batch = next(iter(ds))
    masker = make_masker(
        p_full_assay=float(cfg.training.masking.p_full_assay),
        p_full_loci=0.0,
        p_chunks=0.0,
    )
    prep = prepare_masked_batch(batch, masker, device)
    if prep is None:
        raise RuntimeError("prepare_masked_batch returned None")

    monitor = GradientMonitor(model)
    monitor.register()

    p, n, mu, var, df, peak = model.forward_tuple(
        prep["x_data"], prep["x_dna"], prep["x_meta"], prep["y_meta"],
    )
    _, stats, terms = loss_fn.forward_with_terms(
        p, n, mu, var, df, peak,
        prep["y_data"], prep["y_pval"], prep["y_peaks"],
        prep["observed_map"], prep["masked_map"],
        prep["signal_observed_map"], prep["signal_masked_map"],
        global_step=0,
        fallback_imp_to_observed_when_no_masked=False,
    )

    branch_grads: Dict[str, float] = {}
    for key, term in terms.items():
        if not key.endswith("_weighted"):
            continue
        model.zero_grad(set_to_none=True)
        if term.requires_grad:
            term.backward(retain_graph=True)
            branch_grads[key] = param_grad_norms(model, "encoder")

    model.zero_grad(set_to_none=True)
    total = sum(v for v in terms.values() if isinstance(v, torch.Tensor))
    total.backward()
    monitor_grads = dict(monitor.grad_norms)
    total_encoder_grad = param_grad_norms(model, "encoder")
    total_decoder_grad = param_grad_norms(model, "decoder")

    film_scales: Dict[str, float] = {}
    if hasattr(model.decoder, "pre_decoder_film") and model.decoder.pre_decoder_film is not None:
        with torch.no_grad():
            dec_meta = model.decoder.decoder_meta_embedding(prep["y_meta"].float())
            scale, shift = model.decoder.pre_decoder_film.proj(dec_meta.mean(dim=1)).chunk(2, dim=-1)
            film_scales["pre_decoder_scale_mean"] = float(scale.mean().item())
            film_scales["pre_decoder_scale_std"] = float(scale.std().item())

    report: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": str(device),
        "heads": heads,
        "count_head": count_head,
        "h5_path": str(h5),
        "has_masked_regions": bool(prep["masked_map"].any().item()),
        "n_masked_positions": int(prep["masked_map"].sum().item()),
        "n_observed_positions": int(prep["observed_map"].sum().item()),
        "loss_stats": {k: float(v) for k, v in stats.items() if isinstance(v, (int, float))},
        "branch_encoder_grad_norms": branch_grads,
        "module_backward_hook_norms": monitor_grads,
        "total_encoder_param_grad_norm": total_encoder_grad,
        "total_decoder_param_grad_norm": total_decoder_grad,
        "film_scales": film_scales,
        "params": sum(p.numel() for p in model.parameters()),
    }

    out_dir = output_dir or Path("sandbox/runs/validation_gradient_audit")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"audit_{heads}_{count_head}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), file=sys.stderr)
    print(f"[v2_gradient_audit] wrote {out_path}", file=sys.stderr)
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CANDI v2 gradient/masking audit")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--heads", type=str, default="count_peak")
    p.add_argument("--count-head", type=str, default="plain", choices=["plain", "depth_offset"])
    p.add_argument("--h5", type=str, default=None)
    p.add_argument("--output-dir", type=str, default="sandbox/runs/validation_gradient_audit")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(dev)
    h5 = Path(args.h5) if args.h5 else None
    run_audit(
        device=device,
        heads=args.heads,
        count_head=args.count_head,
        h5_path=h5,
        output_dir=Path(args.output_dir),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

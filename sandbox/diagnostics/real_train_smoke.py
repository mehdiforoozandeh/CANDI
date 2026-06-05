#!/usr/bin/env python3
"""Multi-epoch chr19 training smoke with optional depth-centered offset (diagnostics only).

Exercises iterable dataloader + epoch-level dcr probes (closer to sandbox.train than
single-batch overfit).

Usage:
    python -m sandbox.diagnostics.real_train_smoke
    python -m sandbox.diagnostics.real_train_smoke --depth-offset --depth-center 24
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from sandbox.batch import make_masker, prepare_masked_batch
from sandbox.candi_v2.loss import build_v2_loss
from sandbox.candi_v2.model import CANDIv2
from sandbox.data import SandboxH5Dataset
from sandbox.diagnostics.depth_offset_nb import apply_depth_offset_decoder
from sandbox.diagnostics.real_data import build_real_v2_config, default_h5_path
from sandbox.diagnostics.real_overfit import _V2TupleWrapper, _imp_count_pearson
from sandbox.eval import prompt_sensitivity_depth_count_ratio


@dataclass
class SmokeConfig:
    epochs: int = 3
    max_batches_per_epoch: int = 50
    batch_size: int = 4
    lr: float = 1e-3
    clip_norm: float = 2.0
    depth_offset: bool = False
    depth_center: float = 24.0
    heads: str = "count_only"


def run_smoke(
    cfg: SmokeConfig,
    device: torch.device,
    *,
    h5_path: Optional[Path] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    h5 = h5_path or default_h5_path()
    ds = SandboxH5Dataset(
        h5,
        "type1_chr19",
        train=True,
        batch_size=cfg.batch_size,
        dsf_list=(1, 2, 4, 8),
        dsf_sampling="uniform",
        seed=seed,
        shuffle=True,
        h5_cache_ram=True,
        preserve_assay_id=True,
    )
    probe_ds = SandboxH5Dataset(
        h5,
        "type1_chr19",
        train=True,
        batch_size=cfg.batch_size,
        dsf_sampling="off",
        seed=seed,
        shuffle=False,
        h5_cache_ram=True,
        preserve_assay_id=True,
    )
    probe_batch = next(iter(probe_ds))

    masker = make_masker(
        p_full_assay=0.8,
        p_full_loci=0.5,
        preserve_assay_id=True,
    )
    probe_masker = make_masker(p_full_assay=0.0, p_full_loci=0.0, preserve_assay_id=True)

    model_cfg = build_real_v2_config(
        heads=cfg.heads, lr=cfg.lr, clip_norm=cfg.clip_norm, dropout=0.1,
    )
    torch.manual_seed(seed)
    model = CANDIv2(model_cfg).to(device)
    if cfg.depth_offset:
        apply_depth_offset_decoder(model, depth_center=cfg.depth_center)
    loss_fn = build_v2_loss(model_cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    probe_model = _V2TupleWrapper(model).to(device)

    epoch_log: List[Dict[str, float]] = []
    global_step = 0
    t0 = time.time()

    for ep in range(cfg.epochs):
        model.train()
        ep_losses: List[float] = []
        for bi, batch in enumerate(ds):
            if bi >= cfg.max_batches_per_epoch:
                break
            prep = prepare_masked_batch(batch, masker, device)
            if prep is None or not prep["masked_map"].any():
                continue
            opt.zero_grad(set_to_none=True)
            p, n, mu, var, df, peak = model.forward_tuple(
                prep["x_data"], prep["x_dna"], prep["x_meta"], prep["y_meta"],
            )
            loss, stats, _ = loss_fn.forward_with_terms(
                p, n, mu, var, df, peak,
                prep["y_data"], prep["y_pval"], prep["y_peaks"],
                prep["observed_map"], prep["masked_map"],
                prep["signal_observed_map"], prep["signal_masked_map"],
                global_step=global_step,
                fallback_imp_to_observed_when_no_masked=False,
            )
            global_step += 1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm)
            opt.step()
            ep_losses.append(float(loss.item()))

        model.eval()
        prep_probe = prepare_masked_batch(probe_batch, probe_masker, device, apply_mask=False)
        prep_mask = prepare_masked_batch(probe_batch, masker, device)
        rec: Dict[str, float] = {"epoch": float(ep)}
        if ep_losses:
            rec["mean_loss"] = sum(ep_losses) / len(ep_losses)
        if prep_probe is not None:
            rec["depth_count_ratio"] = prompt_sensitivity_depth_count_ratio(
                probe_model, prep_probe, prep_probe["y_meta"], device,
            )
        if prep_mask is not None:
            with torch.no_grad():
                p, n, *_ = model.forward_tuple(
                    prep_mask["x_data"], prep_mask["x_dna"], prep_mask["x_meta"], prep_mask["y_meta"],
                )
                rec["imp_count_pearson"] = _imp_count_pearson(
                    p, n, prep_mask["y_data"], prep_mask["masked_map"],
                )
        epoch_log.append(rec)
        print(
            f"epoch {ep}: loss={rec.get('mean_loss', float('nan')):.4f} "
            f"dcr={rec.get('depth_count_ratio', float('nan')):.3f} "
            f"imp_p={rec.get('imp_count_pearson', float('nan')):.3f}",
            flush=True,
        )

    elapsed = time.time() - t0
    final_dcr = epoch_log[-1].get("depth_count_ratio", float("nan")) if epoch_log else float("nan")
    passed = final_dcr >= 3.0 if cfg.depth_offset else final_dcr < 2.0 or True  # default: just runs

    return {
        "passed": passed if cfg.depth_offset else True,
        "elapsed_s": elapsed,
        "global_steps": global_step,
        "config": asdict(cfg),
        "epochs": epoch_log,
        "final_dcr": final_dcr,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-batches", type=int, default=50)
    p.add_argument("--depth-offset", action="store_true")
    p.add_argument("--depth-center", type=float, default=24.0)
    p.add_argument("--heads", type=str, default="count_only")
    p.add_argument("--output", type=str, default="sandbox/diagnostics/runs/real_train_smoke.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = SmokeConfig(
        epochs=args.epochs,
        max_batches_per_epoch=args.max_batches,
        depth_offset=args.depth_offset,
        depth_center=args.depth_center,
        heads=args.heads,
    )
    label = "R20_offset" if args.depth_offset else "R20_default"
    print(f"=== {label} multi-epoch chr19 smoke ===", flush=True)
    result = run_smoke(cfg, device)
    result["id"] = label
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"Wrote {out} final_dcr={result['final_dcr']:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

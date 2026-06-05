#!/usr/bin/env python3
"""Imputation + metadata conditioning diagnostics on real chr19 data.

Probes:
  - y_meta (prompt): depth dcr, readlen/runtype count MSE, wrong-depth-on-masked
  - x_meta (input): latent delta under depth/readlen perturbations

Usage:
    python -m sandbox.diagnostics.run_meta_diagnostics
    python -m sandbox.diagnostics.run_meta_diagnostics --ids M01,M02
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.optim as optim

from sandbox.batch import make_masker, prepare_masked_batch
from sandbox.candi_v2.loss import build_v2_loss
from sandbox.candi_v2.model import CANDIv2
from sandbox.diagnostics.depth_offset_nb import apply_depth_offset_decoder
from sandbox.diagnostics.meta_probes import run_probe_battery
from sandbox.diagnostics.real_data import build_real_v2_config, collect_batches, default_h5_path
from sandbox.diagnostics.real_overfit import (
    RealExperimentSpec,
    _V2TupleWrapper,
    _disable_encoder_film,
    _imp_count_pearson,
)


@dataclass
class MetaDiagSpec:
    id: str
    description: str
    train: RealExperimentSpec


SPECS: List[MetaDiagSpec] = [
    MetaDiagSpec(
        "M01", "Imputation assay-only mask, default v2",
        RealExperimentSpec(
            "M01", "", max_steps=1500, depth_offset=False,
            p_full_assay=1.0, p_full_loci=0.0, p_chunks=0.0,
        ),
    ),
    MetaDiagSpec(
        "M02", "Imputation assay-only, depth_center=24 offset",
        RealExperimentSpec(
            "M02", "", max_steps=1500, depth_offset=True, depth_center=24.0,
            p_full_assay=1.0, p_full_loci=0.0,
        ),
    ),
    MetaDiagSpec(
        "M03", "Imputation sandbox masking, default",
        RealExperimentSpec(
            "M03", "", max_steps=1500, depth_offset=False,
            p_full_assay=0.8, p_full_loci=0.5,
        ),
    ),
    MetaDiagSpec(
        "M04", "Imputation sandbox masking, depth_center=24",
        RealExperimentSpec(
            "M04", "", max_steps=1500, depth_offset=True, depth_center=24.0,
            p_full_assay=0.8, p_full_loci=0.5,
        ),
    ),
    MetaDiagSpec(
        "M05", "Assay-only + dec FiLM off + offset center=24",
        RealExperimentSpec(
            "M05", "", max_steps=1500, depth_offset=True, depth_center=24.0,
            decoder_film="none", p_full_assay=1.0, p_full_loci=0.0,
        ),
    ),
    MetaDiagSpec(
        "M06", "Assay-only + enc/dec FiLM off + offset center=24",
        RealExperimentSpec(
            "M06", "", max_steps=1500, depth_offset=True, depth_center=24.0,
            decoder_film="none", encoder_film="none",
            p_full_assay=1.0, p_full_loci=0.0,
        ),
    ),
    MetaDiagSpec(
        "M07", "Heavy imp: loci+assay mask, offset center=24",
        RealExperimentSpec(
            "M07", "", max_steps=1500, depth_offset=True, depth_center=24.0,
            p_full_assay=1.0, p_full_loci=1.0, mask_fraction=0.2,
        ),
    ),
]


def _train_and_probe(
    ts: RealExperimentSpec,
    device: torch.device,
    h5_path: Path,
    seed: int,
) -> Dict[str, Any]:
    batches = collect_batches(
        h5_path, n_batches=max(1, ts.n_batches), batch_size=ts.batch_size, seed=seed,
    )
    batch0 = batches[0]
    masker = make_masker(
        p_full_assay=ts.p_full_assay, p_full_loci=ts.p_full_loci,
        p_chunks=ts.p_chunks, mask_fraction=ts.mask_fraction,
        preserve_assay_id=True,
    )

    cfg = build_real_v2_config(
        heads=ts.heads, lr=ts.lr, clip_norm=ts.clip_norm, dropout=0.0,
    )
    if ts.decoder_film == "none":
        cfg.decoder.film_mode = "none"

    torch.manual_seed(seed)
    model = CANDIv2(cfg).to(device)
    if ts.encoder_film == "none":
        _disable_encoder_film(model)
    if ts.depth_offset:
        apply_depth_offset_decoder(model, depth_center=ts.depth_center)

    loss_fn = build_v2_loss(cfg)
    opt = optim.Adam(model.parameters(), lr=ts.lr)
    probe_model = _V2TupleWrapper(model).to(device)

    prep0 = prepare_masked_batch(batch0, masker, device)
    probes_init: Dict[str, float] = {}
    if prep0 is not None:
        model.eval()
        probes_init = run_probe_battery(model, probe_model, prep0)

    loss_start = None
    loss_end = None
    for step in range(1, ts.max_steps + 1):
        prep = prepare_masked_batch(batches[(step - 1) % len(batches)], masker, device)
        if prep is None or not prep["masked_map"].any():
            continue
        model.train()
        opt.zero_grad(set_to_none=True)
        p, n, mu, var, df, peak = model.forward_tuple(
            prep["x_data"], prep["x_dna"], prep["x_meta"], prep["y_meta"],
        )
        loss, _, _ = loss_fn.forward_with_terms(
            p, n, mu, var, df, peak,
            prep["y_data"], prep["y_pval"], prep["y_peaks"],
            prep["observed_map"], prep["masked_map"],
            prep["signal_observed_map"], prep["signal_masked_map"],
            global_step=step,
            fallback_imp_to_observed_when_no_masked=False,
        )
        lv = float(loss.item())
        if loss_start is None:
            loss_start = lv
        loss_end = lv
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), ts.clip_norm)
        opt.step()

    model.eval()
    prep_f = prepare_masked_batch(batch0, masker, device)
    probes_final = run_probe_battery(model, probe_model, prep_f) if prep_f else {}
    imp_p = float("nan")
    if prep_f is not None:
        with torch.no_grad():
            p, n, *_ = model.forward_tuple(
                prep_f["x_data"], prep_f["x_dna"], prep_f["x_meta"], prep_f["y_meta"],
            )
            imp_p = _imp_count_pearson(p, n, prep_f["y_data"], prep_f["masked_map"])

    return {
        "loss_start": loss_start,
        "loss_end": loss_end,
        "imp_count_pearson": imp_p,
        "probes_init": probes_init,
        "probes_final": probes_final,
    }


def run_meta_diag(
    meta_spec: MetaDiagSpec,
    device: torch.device,
    *,
    h5_path: Optional[Path] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    h5 = h5_path or default_h5_path()
    ts = meta_spec.train
    ts.description = meta_spec.description
    body = _train_and_probe(ts, device, h5, seed)

    print(f"\n[{meta_spec.id}] imp_p={body['imp_count_pearson']:.3f} "
          f"y_dcr {body['probes_init'].get('y_depth_dcr_all', float('nan')):.3f}"
          f"->{body['probes_final'].get('y_depth_dcr_all', float('nan')):.3f} "
          f"x_lat_d {body['probes_init'].get('x_depth_latent_delta', float('nan')):.4f}"
          f"->{body['probes_final'].get('x_depth_latent_delta', float('nan')):.4f}",
          flush=True)

    return {
        "id": meta_spec.id,
        "description": meta_spec.description,
        "train_config": asdict(ts),
        **body,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ids", type=str, default=None)
    p.add_argument("--output", type=str, default="sandbox/diagnostics/runs/meta_diagnostics.json")
    p.add_argument("--h5", type=str, default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h5 = Path(args.h5) if args.h5 else None
    specs = SPECS
    if args.ids:
        wanted = set(args.ids.split(","))
        specs = [s for s in SPECS if s.id in wanted]

    results = []
    for ms in specs:
        print(f"\n{'='*60}\n{ms.id}: {ms.description}\n{'='*60}", flush=True)
        results.append(run_meta_diag(ms, device, h5_path=h5))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"device": str(device), "results": results}, indent=2))
    print(f"\nWrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

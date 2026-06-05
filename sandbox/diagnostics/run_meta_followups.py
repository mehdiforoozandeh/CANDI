#!/usr/bin/env python3
"""Follow-up metadata / imputation diagnostics M08–M10.

M08 — fixed dcr probes on masked bins only (re-run M02 config)
M09 — x_meta ablation: wipe observed-assay input metadata at eval
M10 — training trajectory: x_meta / y_meta sensitivity vs step

Usage:
    python -m sandbox.diagnostics.run_meta_followups
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.optim as optim

from sandbox.batch import CLOZE, make_masker, prepare_masked_batch
from sandbox.candi_v2.loss import build_v2_loss
from sandbox.candi_v2.model import CANDIv2
from sandbox.diagnostics.depth_offset_nb import apply_depth_offset_decoder
from sandbox.diagnostics.meta_probes import (
    ablate_x_meta_observed_columns,
    latent_delta_ratio,
    run_probe_battery,
)
from sandbox.diagnostics.real_data import build_real_v2_config, collect_batches, default_h5_path
from sandbox.diagnostics.real_overfit import (
    RealExperimentSpec,
    _V2TupleWrapper,
    _imp_count_pearson,
)
from sandbox.diagnostics.run_meta_diagnostics import _train_and_probe


def run_m08(device: torch.device, h5: Path, seed: int = 42) -> Dict[str, Any]:
    """Re-run assay-only offset with fixed masked-bin dcr probes."""
    ts = RealExperimentSpec(
        "M08", "Fixed masked-bin dcr probes (M02 config)",
        max_steps=1500, depth_offset=True, depth_center=24.0,
        p_full_assay=1.0, p_full_loci=0.0,
    )
    body = _train_and_probe(ts, device, h5, seed)
    pf = body["probes_final"]
    print(
        f"[M08] dcr_all={pf.get('y_depth_dcr_all', float('nan')):.3f} "
        f"dcr_masked_bins={pf.get('y_depth_dcr_on_masked_bins', float('nan')):.3f} "
        f"dcr_median_assay={pf.get('y_depth_dcr_median_per_assay_masked_bins', float('nan')):.3f}",
        flush=True,
    )
    return {"id": "M08", "description": ts.description, **body}


def run_m09(device: torch.device, h5: Path, seed: int = 42) -> Dict[str, Any]:
    """Train offset assay-only; compare eval with vs without observed x_meta."""
    batch = collect_batches(h5, n_batches=1, seed=seed)[0]
    masker = make_masker(p_full_assay=1.0, p_full_loci=0.0, preserve_assay_id=True)
    cfg = build_real_v2_config(dropout=0.0)
    torch.manual_seed(seed)
    model = CANDIv2(cfg).to(device)
    apply_depth_offset_decoder(model, depth_center=24.0)
    loss_fn = build_v2_loss(cfg)
    opt = optim.Adam(model.parameters(), lr=1e-3)

    for step in range(1, 1501):
        prep = prepare_masked_batch(batch, masker, device)
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
            fallback_imp_to_observed_when_no_masked=False,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()

    model.eval()
    prep = prepare_masked_batch(batch, masker, device)
    probe = _V2TupleWrapper(model).to(device)
    probes_normal = run_probe_battery(model, probe, prep)

    x_ab = ablate_x_meta_observed_columns(prep["x_meta"], prep["masked_map"])
    prep_ab = dict(prep)
    prep_ab["x_meta"] = x_ab
    probes_ablated = run_probe_battery(model, probe, prep_ab)

    with torch.no_grad():
        p0, n0, *_ = model.forward_tuple(
            prep["x_data"], prep["x_dna"], prep["x_meta"], prep["y_meta"],
        )
        p1, n1, *_ = model.forward_tuple(
            prep_ab["x_data"], prep_ab["x_dna"], prep_ab["x_meta"], prep_ab["y_meta"],
        )
        imp_normal = _imp_count_pearson(p0, n0, prep["y_data"], prep["masked_map"])
        imp_ablated = _imp_count_pearson(p1, n1, prep["y_data"], prep["masked_map"])

    z_delta = latent_delta_ratio(model, prep, x_ab)

    print(
        f"[M09] imp_p normal={imp_normal:.3f} ablated={imp_ablated:.3f} "
        f"dcr normal={probes_normal.get('y_depth_dcr_all', float('nan')):.3f} "
        f"x_ablate_z_delta={z_delta:.4f}",
        flush=True,
    )
    return {
        "id": "M09",
        "description": "x_meta ablate observed assay columns at eval",
        "imp_pearson_normal": imp_normal,
        "imp_pearson_ablated": imp_ablated,
        "x_meta_ablate_latent_delta": z_delta,
        "probes_normal": probes_normal,
        "probes_ablated": probes_ablated,
    }


def run_m10(device: torch.device, h5: Path, seed: int = 42) -> Dict[str, Any]:
    """Probe trajectory during multi-epoch chr19 training (offset center=24)."""
    ds_batches = collect_batches(h5, n_batches=20, batch_size=4, seed=seed)
    batch0 = ds_batches[0]
    masker = make_masker(p_full_assay=0.8, p_full_loci=0.5, preserve_assay_id=True)
    probe_masker = make_masker(p_full_assay=0.0, p_full_loci=0.0, preserve_assay_id=True)

    cfg = build_real_v2_config(dropout=0.1)
    torch.manual_seed(seed)
    model = CANDIv2(cfg).to(device)
    apply_depth_offset_decoder(model, depth_center=24.0)
    loss_fn = build_v2_loss(cfg)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    probe = _V2TupleWrapper(model).to(device)

    trajectory: List[Dict[str, float]] = []
    probe_every = 25
    max_steps = 400
    step = 0

    for step in range(1, max_steps + 1):
        batch = ds_batches[(step - 1) % len(ds_batches)]
        prep = prepare_masked_batch(batch, masker, device)
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
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()

        if step % probe_every == 0 or step == 1:
            prep_probe = prepare_masked_batch(batch0, probe_masker, device, apply_mask=False)
            prep_mask = prepare_masked_batch(batch0, masker, device)
            if prep_probe is None or prep_mask is None:
                continue
            model.eval()
            bat = run_probe_battery(model, probe, prep_mask)
            with torch.no_grad():
                p2, n2, *_ = model.forward_tuple(
                    prep_mask["x_data"], prep_mask["x_dna"], prep_mask["x_meta"], prep_mask["y_meta"],
                )
                imp_p = _imp_count_pearson(p2, n2, prep_mask["y_data"], prep_mask["masked_map"])
            rec = {
                "step": float(step),
                "loss": float(loss.item()),
                "imp_count_pearson": imp_p,
                **{k: bat.get(k, float("nan")) for k in (
                    "y_depth_dcr_all", "y_depth_dcr_on_masked_bins",
                    "x_depth_latent_delta", "x_masked_fill_y_depth_latent_delta",
                )},
            }
            trajectory.append(rec)
            print(
                f"[M10] step={step} imp_p={imp_p:.3f} dcr={rec['y_depth_dcr_all']:.3f} "
                f"x_lat={rec['x_depth_latent_delta']:.4f}",
                flush=True,
            )

    return {
        "id": "M10",
        "description": "Probe trajectory 400 steps, probe every 25",
        "trajectory": trajectory,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ids", type=str, default="M08,M09,M10")
    p.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).resolve().parent / "runs" / "meta_followups.json"),
    )
    p.add_argument("--h5", type=str, default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h5 = Path(args.h5) if args.h5 else default_h5_path()
    wanted = set(args.ids.split(","))

    results = []
    if "M08" in wanted:
        print("=" * 60 + "\nM08\n" + "=" * 60, flush=True)
        results.append(run_m08(device, h5))
    if "M09" in wanted:
        print("=" * 60 + "\nM09\n" + "=" * 60, flush=True)
        results.append(run_m09(device, h5))
    if "M10" in wanted:
        print("=" * 60 + "\nM10\n" + "=" * 60, flush=True)
        results.append(run_m10(device, h5))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"device": str(device), "results": results}, indent=2))
    print(f"Wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

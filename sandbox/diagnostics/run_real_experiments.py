#!/usr/bin/env python3
"""Real chr19 sandbox diagnostic experiment batches.

Batch 1: baseline Q5 repro + offset fix on pinned chr19 batches.
Batch 2: adaptive follow-ups (run after batch 1 completes).

Usage:
    python -m sandbox.diagnostics.run_real_experiments --batch 1
    python -m sandbox.diagnostics.run_real_experiments --batch 2
    python -m sandbox.diagnostics.run_real_experiments --batch 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch

from sandbox.diagnostics.real_overfit import RealExperimentSpec, run_real_experiment


BATCH1: List[RealExperimentSpec] = [
    RealExperimentSpec(
        "R01", "chr19 1-batch overfit, default v2 count_only, sandbox masking",
        max_steps=1500, depth_offset=False, n_batches=1,
    ),
    RealExperimentSpec(
        "R02", "chr19 1-batch overfit, depth-offset, sandbox masking",
        max_steps=1500, depth_offset=True, n_batches=1,
    ),
    RealExperimentSpec(
        "R03", "chr19 1-batch reconstruct (no masking), default",
        max_steps=1200, depth_offset=False, p_full_assay=0.0, p_full_loci=0.0,
        require_masked=False, min_dcr=0.0,
    ),
    RealExperimentSpec(
        "R04", "chr19 1-batch reconstruct (no masking), depth-offset",
        max_steps=1200, depth_offset=True, p_full_assay=0.0, p_full_loci=0.0,
        require_masked=False, min_dcr=0.0,
    ),
    RealExperimentSpec(
        "R05", "chr19 8-batch cycle, default, sandbox masking",
        max_steps=800, depth_offset=False, n_batches=8,
    ),
    RealExperimentSpec(
        "R06", "chr19 8-batch cycle, depth-offset, sandbox masking",
        max_steps=800, depth_offset=True, n_batches=8,
    ),
]


def _build_batch2(batch1_path: Path) -> List[RealExperimentSpec]:
    """Choose batch-2 experiments from batch-1 outcomes."""
    data = json.loads(batch1_path.read_text())
    by_id = {r["id"]: r for r in data.get("results", [])}
    r01 = by_id.get("R01", {})
    r02 = by_id.get("R02", {})
    r03 = by_id.get("R03", {})
    r01_dcr = r01.get("metrics", {}).get("depth_count_ratio", 1.0)
    r02_dcr = r02.get("metrics", {}).get("depth_count_ratio", 1.0)
    r01_loss_ok = r01.get("loss_drop_frac", 0) > 0.15
    r03_loss_ok = r03.get("loss_drop_frac", 0) > 0.15

    specs: List[RealExperimentSpec] = []

    # Always: assay-only masking stress (isolates imputation from loci masking)
    specs.append(RealExperimentSpec(
        "R07", "chr19 assay-only mask p=1.0, default",
        max_steps=1500, depth_offset=False, p_full_assay=1.0, p_full_loci=0.0,
    ))
    specs.append(RealExperimentSpec(
        "R08", "chr19 assay-only mask p=1.0, depth-offset",
        max_steps=1500, depth_offset=True, p_full_assay=1.0, p_full_loci=0.0,
    ))

    if r02_dcr >= 3.0:
        specs.append(RealExperimentSpec(
            "R09", "chr19 offset + dec FiLM off (real data FiLM ablation)",
            max_steps=1500, depth_offset=True, decoder_film="none",
        ))
        specs.append(RealExperimentSpec(
            "R10", "chr19 offset + enc FiLM off",
            max_steps=1500, depth_offset=True, encoder_film="none",
        ))

    # Offset failed on real data — test depth-centered offset (2^(d-24))
    if r02.get("metrics", {}).get("depth_count_ratio", 1.0) < 2.0:
        specs.append(RealExperimentSpec(
            "R15", "chr19 offset depth_center=24 (real scale fix)",
            max_steps=1500, depth_offset=True, depth_center=24.0,
        ))
        specs.append(RealExperimentSpec(
            "R16", "chr19 offset depth_center=24 + no FiLM",
            max_steps=1500, depth_offset=True, depth_center=24.0,
            decoder_film="none", encoder_film="none",
        ))
        specs.append(RealExperimentSpec(
            "R17", "chr19 offset depth_center=24, assay-only mask",
            max_steps=1500, depth_offset=True, depth_center=24.0,
            p_full_assay=1.0, p_full_loci=0.0,
        ))

    if r01_dcr < 2.0 and r02_dcr >= 3.0:
        specs.append(RealExperimentSpec(
            "R11", "chr19 count_peak multi-head default (does pval/peak interfere?)",
            max_steps=1200, depth_offset=False, heads="count_peak",
        ))
        specs.append(RealExperimentSpec(
            "R12", "chr19 count_peak multi-head + depth-offset",
            max_steps=1200, depth_offset=True, heads="count_peak",
        ))

    if not r01_loss_ok and not r03_loss_ok:
        specs.append(RealExperimentSpec(
            "R13", "chr19 default lr=3e-3 (capacity / convergence rescue)",
            max_steps=1500, depth_offset=False, lr=3e-3,
        ))

    if r02_dcr >= 3.0:
        specs.append(RealExperimentSpec(
            "R14", "chr19 offset lr=3e-3 faster fit check",
            max_steps=800, depth_offset=True, lr=3e-3, n_batches=8,
        ))

    return specs


BATCH3: List[RealExperimentSpec] = [
    RealExperimentSpec(
        "R18", "chr19 8-batch cycle, depth_center=24",
        max_steps=1500, depth_offset=True, depth_center=24.0, n_batches=8,
    ),
    RealExperimentSpec(
        "R19", "chr19 count_peak default (multi-head Q5 check)",
        max_steps=1500, depth_offset=False, heads="count_peak",
    ),
    RealExperimentSpec(
        "R19b", "chr19 count_peak + depth_center=24",
        max_steps=1500, depth_offset=True, depth_center=24.0, heads="count_peak",
    ),
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, choices=[1, 2, 3], default=1)
    p.add_argument("--ids", type=str, default=None)
    p.add_argument("--h5", type=str, default=None)
    p.add_argument("--batch1-results", type=str,
                   default="sandbox/diagnostics/runs/real_batch1.json")
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h5 = Path(args.h5) if args.h5 else None

    if args.batch == 1:
        specs = BATCH1
        out = Path(args.output or "sandbox/diagnostics/runs/real_batch1.json")
    elif args.batch == 2:
        b1 = Path(args.batch1_results)
        if not b1.exists():
            raise SystemExit(f"Batch 2 requires batch 1 results at {b1}")
        specs = _build_batch2(b1)
        out = Path(args.output or "sandbox/diagnostics/runs/real_batch2.json")
    else:
        specs = BATCH3
        out = Path(args.output or "sandbox/diagnostics/runs/real_batch3.json")

    if args.ids:
        wanted = set(args.ids.split(","))
        specs = [s for s in specs if s.id in wanted]

    print(f"Running {len(specs)} real-data experiments on {device}", flush=True)
    results = []
    for spec in specs:
        print(f"\n{'='*60}\n{spec.id}: {spec.description}\n{'='*60}", flush=True)
        results.append(run_real_experiment(spec, device, h5_path=h5))

    payload = {"device": str(device), "batch": args.batch, "results": results}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

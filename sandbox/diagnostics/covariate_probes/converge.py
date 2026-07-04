"""Convergence check: per-biosample test metric vs #epochs, to pick a plateau epoch count.
Run: python -m sandbox.diagnostics.covariate_probes.converge"""
from __future__ import annotations
import numpy as np, torch
from sandbox.diagnostics.covariate_probes import data as D
from sandbox.diagnostics.covariate_probes.run import train_eval

# (assay, target, norm, metric_key) — a run_type (small & large) and a read_length probe
CASES = [
    ("DNase-seq", "run_type", "thin", "auroc"),
    ("H3K27ac",   "run_type", "thin", "auroc"),
    ("H3K27ac",   "read_length", "thin", "r2"),
]
EPOCHS = [2, 4, 8, 12, 20, 32]

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    for assay, target, norm, mk in CASES:
        insts = D.census(assays=[assay])
        pool = D.build_pool(insts)
        print(f"\n## {assay} / {target}  (instances={len(insts)})")
        prev = None
        for E in EPOCHS:
            vals = []
            for seed in (0, 1):  # 2 splits to reduce noise
                r = train_eval(pool, insts, target, norm, False, seed=seed, epochs=E, device=device)
                vals.append(r["biosample"][mk])
            m = float(np.nanmean(vals))
            d = "" if prev is None else f"  Δ={m-prev:+.3f}"
            print(f"   epochs={E:3d}  {mk}(bios, mean2)={m:.3f}{d}", flush=True)
            prev = m

if __name__ == "__main__":
    main()

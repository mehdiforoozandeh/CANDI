"""Plateau probe for the dual-conditioning testbed v2 (crux q15 / q16).

Trains one arm and logs the v2 metrics (CRPS, M1 ceiling-gap, distributional M2, M3 ratio) every
`--eval-every` epochs so the array walltime can be sized to when steering (M2) and reconstruction
plateau. Not part of the A-K gate; a dev utility.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from sandbox.diagnostics.dual_conditioning import transforms as T
from sandbox.diagnostics.dual_conditioning import metrics as MET
from sandbox.diagnostics.dual_conditioning.data import DualCondData
from sandbox.diagnostics.dual_conditioning.model import build_model, forward_counts, nb_nll
from sandbox.diagnostics.dual_conditioning.run import _cosine_warmup, _families


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--norm", default="zscore")
    ap.add_argument("--mode", default="per_assay", choices=["per_assay", "uniform"])
    ap.add_argument("--families", default="2a", choices=["2a", "2c"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps-per-epoch", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=3)
    ap.add_argument("--eval-units", type=int, default=8)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fams = _families(a.families)
    data = DualCondData()
    conds = T.all_conditions(fams)
    torch.manual_seed(0)
    model = build_model(a.norm).to(device)
    opt = torch.optim.Adam(model.parameters(), 5e-4, weight_decay=1e-4)
    sched = _cosine_warmup(opt, a.epochs * a.steps_per_epoch)
    rng = np.random.default_rng(0)
    units = MET.make_eval_units(data, a.eval_units, a.batch_size, np.random.default_rng(1), chrom="chr21")

    print("epoch  nll     crps   M1gap  M2inv  M3ratio", flush=True)
    for ep in range(a.epochs):
        model.train()
        losses = []
        for b in data.iter_train(a.batch_size, a.steps_per_epoch, rng, conds, device, mode=a.mode):
            loss = nb_nll(*forward_counts(model, b), b["y_target"], b["avail"])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); losses.append(float(loss))
        if ep % a.eval_every == 0 or ep == a.epochs - 1:
            model.eval()
            crps = MET.eval_reconstruction(model, data, units, device, families=fams, budget=40_000)["crps"]
            m1 = MET.eval_M1(model, data, units, device, families=fams, budget=40_000)["median_gap"]
            m2 = MET.eval_M2(model, data, units, device, families=fams)["median_invertible"]
            m3 = MET.eval_M3(model, data, units, device, families=fams)["ratio"]
            print(f"{ep:5d}  {np.mean(losses):.3f}  {crps:.3f}  {m1:.3f}  {m2:.3f}  {m3:.3f}", flush=True)


if __name__ == "__main__":
    main()

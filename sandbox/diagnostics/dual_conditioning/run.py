"""Train + eval sweep for the dual-conditioning testbed (crux q15).

Runs (GPU array):
  --norm {none,zscore,log}          full matrix, one normalization arm -> M1/M2/M3   (h30 + h33)
  --holdout                         hold out matrix cells, best norm arm -> h31 (seen vs held + baseline)
  --validate                        CPU gates + GPU train-control gates (hard gate before the sweep)

h32 is read off the per-family M1/M2/M3 of the h30 runs (no separate run).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from sandbox.diagnostics.dual_conditioning import transforms as T
from sandbox.diagnostics.dual_conditioning import metrics as MET
from sandbox.diagnostics.dual_conditioning.data import DualCondData, A
from sandbox.diagnostics.dual_conditioning.model import build_model, forward_counts, nb_nll, nb_mean

OUTDIR = "sandbox/diagnostics/dual_conditioning/results"

# h31 holdout: cells at seen-row x seen-col intersections (each family still trains on both sides).
HELD_CELLS = [("mult", "thin"), ("add", "cap"), ("power", "clog"),
              ("thin", "mult"), ("cap", "add"), ("clog", "power")]


def _wandb_init(tag, config, mode):
    if mode == "disabled":
        return None
    try:
        import wandb
        return wandb.init(project="candi_sandbox", name=f"dualcond_{tag}", config=config,
                          mode=mode, tags=["dual_conditioning", "q15"], reinit=True)
    except Exception as e:
        print(f"[wandb] init failed ({e}); continuing without wandb", flush=True)
        return None


def _wandb_summary(run, res):
    """Flatten headline + per-family metrics to wandb scalars."""
    if run is None:
        return
    m1, m2, m3 = res["M1"], res["M2"], res["M3"]
    log = {"M1/ceiling": m1["ceiling"], "M1/median_offdiag_r2": m1["median_offdiag_r2"],
           "M1/median_gap": m1["median_gap"], "M1/median_r2": m1["median_r2"],
           "M2/median": m2["median"], "M2/median_invertible": m2["median_invertible"],
           "M3/ratio": m3["ratio"], "M3/within": m3["within"], "M3/between": m3["between"]}
    for f, v in m2["per_family"].items():
        log[f"M2_family/{T.FAM_NAMES[int(f)]}"] = v
    for f, v in m3["per_family_ratio"].items():
        log[f"M3_family/{T.FAM_NAMES[int(f)]}"] = v
    if "h31" in res:
        h = res["h31"]
        log.update({"h31/seen_median": h["seen_median"], "h31/held_median": h["held_median"],
                    "h31/baseline_median": h["baseline_median"]})
    run.log(log)
    run.summary.update(log)


def _held_ids():
    return {(T.FAM[a], T.FAM[b]) for a, b in HELD_CELLS}


def make_allowed_cell(held):
    if not held:
        return None
    hs = _held_ids()
    return lambda fx, fy: (int(fx), int(fy)) not in hs


def _cosine_warmup(opt, total_steps, warmup_frac=0.1, min_ratio=0.1):
    warm = max(1, int(warmup_frac * total_steps))
    def fn(s):
        if s < warm:
            return s / warm
        t = (s - warm) / max(1, total_steps - warm)
        return min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * t))
    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


def train_model(norm, data, conds, *, epochs, steps_per_epoch, batch_size, device,
                allowed_cell=None, lr=5e-4, seed=0, log_every=0, wandb_run=None, on_epoch=None):
    torch.manual_seed(seed)
    model = build_model(norm).to(device)
    opt = torch.optim.Adam(model.parameters(), lr, weight_decay=1e-4)
    sched = _cosine_warmup(opt, epochs * steps_per_epoch)
    rng = np.random.default_rng(seed)
    step = 0
    for ep in range(epochs):
        model.train()
        ep_losses = []
        for b in data.iter_train(batch_size, steps_per_epoch, rng, conds, device, allowed_cell):
            loss = nb_nll(*forward_counts(model, b), b["y_target"], b["avail"])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            step += 1
            ep_losses.append(float(loss))
            if log_every and step % log_every == 0:
                print(f"  ep{ep} step{step} nll={float(loss):.3f} lr={sched.get_last_lr()[0]:.2e}", flush=True)
                if wandb_run is not None:
                    wandb_run.log({"train/nll": float(loss), "train/epoch": ep,
                                   "train/lr": sched.get_last_lr()[0]}, step=step)
        if on_epoch is not None:
            on_epoch(model, ep, float(np.mean(ep_losses)))
    return model


def full_eval(model, data, units, device):
    return dict(M1=MET.eval_M1(model, data, units, device),
                M2=MET.eval_M2(model, data, units, device),
                M3=MET.eval_M3(model, data, units, device))


@torch.no_grad()
def h31_eval(model, data, units, device):
    """M1 on seen vs held cells + a memorization baseline (output-identity when true target is f_y)."""
    held = _held_ids()
    m1 = MET.eval_M1(model, data, units, device)["cell_r2"]
    seen = [v for c, v in m1.items() if c not in held and c[0] != c[1]]
    held_r2 = {c: m1[c] for c in held}
    # memorization baseline: for each held cell (fx,fy), predict with output=identity, score vs true f_y target
    idc = T.FAM["identity"]
    base = {}
    for (fx, fy) in held:
        px = float(np.random.default_rng(fx).choice(T.PARAMS[T.FAM_NAMES[fx]]))
        py = float(np.random.default_rng(fy).choice(T.PARAMS[T.FAM_NAMES[fy]]))
        pred_id, tgt_fy = MET._cell_preds(model, data, units, fx, px, idc, 1.0, device)  # output identity
        _, tgt_true = MET._cell_preds(model, data, units, fx, px, fy, py, device)          # true target
        base[(fx, fy)] = MET.r2(pred_id, tgt_true)
    return dict(seen_median=float(np.nanmedian(seen)),
                held_median=float(np.nanmedian(list(held_r2.values()))),
                held_r2={f"{T.FAM_NAMES[a]}->{T.FAM_NAMES[b]}": v for (a, b), v in held_r2.items()},
                baseline_median=float(np.nanmedian(list(base.values()))),
                baseline_r2={f"{T.FAM_NAMES[a]}->{T.FAM_NAMES[b]}": v for (a, b), v in base.items()})


def _jsonable(x):
    if isinstance(x, dict):
        return {(f"{k[0]}_{k[1]}" if isinstance(k, tuple) else str(k)): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--norm", default="zscore", choices=["none", "zscore", "log"])
    ap.add_argument("--holdout", action="store_true", help="h31: exclude HELD_CELLS from training")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--steps-per-epoch", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--eval-units", type=int, default=16)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", "offline"),
                    choices=["online", "offline", "disabled"])
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} norm={a.norm} holdout={a.holdout} wandb={a.wandb_mode}", flush=True)

    if a.validate:
        import sys
        from sandbox.diagnostics.dual_conditioning.validate import run_cpu_gates, run_train_gates
        ok = run_cpu_gates() and run_train_gates(device)
        sys.exit(0 if ok else 1)

    os.makedirs(OUTDIR, exist_ok=True)
    t0 = time.time()
    tag = a.tag or ("holdout" if a.holdout else f"norm_{a.norm}")
    wb = _wandb_init(tag, dict(norm=a.norm, holdout=a.holdout, epochs=a.epochs,
                               steps_per_epoch=a.steps_per_epoch, batch_size=a.batch_size), a.wandb_mode)
    data = DualCondData()
    conds = T.all_conditions()
    rng = np.random.default_rng(0)
    units = MET.make_eval_units(data, a.eval_units, a.batch_size, rng, chrom="chr21")
    model = train_model(a.norm, data, conds, epochs=a.epochs, steps_per_epoch=a.steps_per_epoch,
                        batch_size=a.batch_size, device=device,
                        allowed_cell=make_allowed_cell(a.holdout), log_every=200, wandb_run=wb)
    res = dict(norm=a.norm, holdout=a.holdout, epochs=a.epochs,
               steps_per_epoch=a.steps_per_epoch, wall_s=None)
    res.update(full_eval(model, data, units, device))
    if a.holdout:
        res["h31"] = h31_eval(model, data, units, device)
    res["wall_s"] = round(time.time() - t0, 1)
    with open(f"{OUTDIR}/{tag}.json", "w") as f:
        json.dump(_jsonable(res), f, indent=2)
    _wandb_summary(wb, res)
    if wb is not None:
        wb.finish()
    m1, m2, m3 = res["M1"], res["M2"], res["M3"]
    print(f"[{tag}] ceiling={m1['ceiling']:.3f} med_offdiag_R2={m1['median_offdiag_r2']:.3f} "
          f"med_gap={m1['median_gap']:.3f} | M2_inv={m2['median_invertible']:.3f} | "
          f"M3_ratio={m3['ratio']:.3f} | {res['wall_s']}s", flush=True)


if __name__ == "__main__":
    main()

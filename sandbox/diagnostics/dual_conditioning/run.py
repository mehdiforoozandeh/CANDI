"""Train + eval driver for the dual-conditioning testbed v2 (crux q15 / q16).

One invocation = one ARM. The Phase-2 matrix (see plan_v2 §Arms):
  2a core grid : param-norm {none,zscore,log} x encoder-depth {naive,aware}   (6 runs; h30 + h33)
  2b deltas    : --offset off | --mode uniform | --force-identity-x          (h36, h34, h35)
  2c expansion : --families 2c  (adds thin/cap/clog; ONLY if 2a shows steering)

Each run trains denoising-only NBNLL, then evaluates the full metric suite + M2 (distributional) + M3 +
the q16 diagnostics on BOTH chr19 (train) and chr21 (test), and writes results/{tag}.json for report.py.
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
from sandbox.diagnostics.dual_conditioning.data import DualCondData, A, stratified_holdout
from sandbox.diagnostics.dual_conditioning.model import build_model, forward_counts, nb_nll

OUTDIR = "sandbox/diagnostics/dual_conditioning/results"


def _cosine_warmup(opt, total_steps, warmup_frac=0.1, min_ratio=0.1):
    warm = max(1, int(warmup_frac * total_steps))
    def fn(s):
        if s < warm:
            return s / warm
        t = (s - warm) / max(1, total_steps - warm)
        return min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * t))
    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


def _families(name: str):
    return {"2a": T.FAMILIES_2A, "2c": T.FAMILIES_2C}[name]


def _eval_chrom(model, data, device, units, families, seed=0, heldout=None, matrix_units=6):
    # M2 = the identity-input decomposition (per_family/mean_stat/tail_stat) MERGED with the full
    # f_x x f_y steering matrix. COST TRIM: the matrix is the dominant eval cost -> its lossy-f_x rows are
    # computed on a SMALLER unit subset (matrix_units), while the f_x=identity row is copied EXACTLY from
    # the full-units per_family (identity_row=), so the reference row loses nothing.
    m2 = dict(MET.eval_M2(model, data, units, device, families=families, seed=seed))
    mat_units = units[:matrix_units]
    m2.update(MET.eval_M2_matrix(model, data, mat_units, device, families=families, seed=seed,
                                 identity_row=m2["per_family"]))
    out = dict(
        recon=MET.eval_reconstruction(model, data, units, device, families=families, seed=seed),
        M1=MET.eval_M1(model, data, units, device, families=families, seed=seed),
        M2=m2,
        M3=MET.eval_M3(model, data, units, device, families=families, seed=seed),
        fg=MET.eval_foreground_M2(model, data, units, device, families=families, seed=seed),
        shuffle=MET.eval_shuffle_reliance(model, data, units, device, families=families, seed=seed),
    )
    if heldout:   # h31 holdout runs only: does the model read h_y or fall back on a memorized pairing?
        out["memorization"] = MET.eval_memorization_baseline(
            model, data, units, device, heldout=heldout, families=families, seed=seed)
    return out


def train_and_eval(norm="zscore", *, encoder_depth_aware=False, use_offset=True, pool_meta=False,
                   mode="per_assay", force_x_identity=False, families=None, epochs=25, steps_per_epoch=200,
                   batch_size=16, eval_units=16, lr=5e-4, weight_decay=1e-4, device="cpu",
                   seed=0, wandb_run=None, log_every=0, holdout_rho=0.0, holdout_seed=0):
    """Train one arm and evaluate on chr19 + chr21. Returns the structured result dict.

    holdout_rho>0 (h31) withholds ~rho of the eligible family cells (stratified_holdout) from training and
    auto-scales steps by retained/all so per-retained-cell exposure MATCHES the full-matrix (rho=0) run --
    retained cells must not over-train and inflate the gen-gap. The held-out cells are still EVALUATED
    (they're in `families`), giving the per-cell gen-gap vs the rho=0 reference at report time.
    """
    families = families if families is not None else T.FAMILIES_2A
    # cuDNN-deterministic (safe variant) so cross-arm M2 differences are real, not GPU-nondeterminism.
    # NOT torch.use_deterministic_algorithms -- that crashes the candi_v2 encoder.
    if device == "cuda" and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    data = DualCondData()
    conds = T.all_conditions(families)

    # h31 holdout mask + per-retained-cell budget matching. Rejection sampling in sample_conditions is
    # uniform over (family,param) CONDITIONS, so a family cell (fx,fy) has draw-weight |PARAMS[fx]|*|PARAMS[fy]|
    # -- NOT 1. Held cells are all weight-16 off-diagonal non-identity, so the raw cell-COUNT ratio would
    # over-expose retained cells; scale by the condition-WEIGHT ratio W_retained/W_all (cell-independent, so
    # every retained cell's per-cell exposure matches the rho=0 run). [adversarial review 2026-07-09]
    heldout = stratified_holdout(families, holdout_rho, holdout_seed) if holdout_rho > 0 else set()
    allowed_cell = (lambda fx, fy: (int(fx), int(fy)) not in heldout) if heldout else None
    _w = {T.FAM[f]: len(T.PARAMS[f]) for f in families}
    W_all = sum(_w[fx] * _w[fy] for fx in _w for fy in _w)
    W_held = sum(_w[int(fx)] * _w[int(fy)] for fx, fy in heldout)
    budget_scale = (W_all - W_held) / W_all if heldout else 1.0
    eff_steps = max(1, round(steps_per_epoch * budget_scale)) if heldout else steps_per_epoch

    torch.manual_seed(seed)
    model = build_model(norm, encoder_depth_aware=encoder_depth_aware, use_offset=use_offset,
                        pool_meta=pool_meta).to(device)
    opt = torch.optim.Adam(model.parameters(), lr, weight_decay=weight_decay)
    sched = _cosine_warmup(opt, epochs * eff_steps)
    rng = np.random.default_rng(seed)

    train_losses = []
    cell_counts: dict = {}
    step = 0
    for ep in range(epochs):
        model.train()
        for b in data.iter_train(batch_size, eff_steps, rng, conds, device,
                                 mode=mode, force_x_identity=force_x_identity,
                                 allowed_cell=allowed_cell, cell_counts=cell_counts):
            loss = nb_nll(*forward_counts(model, b), b["y_target"], b["avail"])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            step += 1
            train_losses.append(float(loss))
            if log_every and step % log_every == 0:
                print(f"  ep{ep} step{step} nll={float(loss):.3f} lr={sched.get_last_lr()[0]:.2e}", flush=True)
                if wandb_run is not None:
                    wandb_run.log({"train/nll": float(loss), "train/lr": sched.get_last_lr()[0]}, step=step)

    model.eval()
    eval_rng = np.random.default_rng(12345)
    out = dict(config=dict(norm=norm, encoder_depth_aware=encoder_depth_aware, use_offset=use_offset,
                           pool_meta=pool_meta, mode=mode, force_x_identity=force_x_identity,
                           families=list(families), epochs=epochs, steps_per_epoch=steps_per_epoch,
                           eff_steps_per_epoch=eff_steps, budget_scale=round(float(budget_scale), 4),
                           batch_size=batch_size,
                           holdout_rho=float(holdout_rho), holdout_seed=int(holdout_seed),
                           heldout=sorted([list(c) for c in heldout])),
               train_losses=train_losses, cell_coverage=cell_counts)
    for chrom in ("chr19", "chr21"):
        units = MET.make_eval_units(data, eval_units, batch_size, np.random.default_rng(int(eval_rng.integers(1 << 30))), chrom=chrom)
        out[chrom] = _eval_chrom(model, data, device, units, families, seed=seed, heldout=heldout)
    return out


def _jsonable(x):
    if isinstance(x, dict):
        return {(f"{k[0]}_{k[1]}" if isinstance(k, tuple) else str(k)): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def _wandb_init(tag, config, mode):
    if mode == "disabled":
        return None
    try:
        import wandb
        return wandb.init(project="candi_sandbox", name=f"dualcond2_{tag}", config=config,
                          mode=mode, tags=["dual_conditioning", "q15", "q16", "v2"], reinit=True)
    except Exception as e:
        print(f"[wandb] init failed ({e}); continuing without wandb", flush=True)
        return None


def _wandb_summary(run, res):
    if run is None:
        return
    log = {}
    for chrom in ("chr19", "chr21"):
        r = res[chrom]
        log[f"{chrom}/recon_crps"] = r["recon"]["crps"]
        log[f"{chrom}/recon_spearman"] = r["recon"]["spearman"]
        log[f"{chrom}/recon_ece"] = r["recon"]["ece"]
        log[f"{chrom}/M1_median_gap"] = r["M1"]["median_gap"]
        log[f"{chrom}/M2_median_invertible"] = r["M2"]["median_invertible"]
        log[f"{chrom}/M3_ratio"] = r["M3"]["ratio"]
    run.log(log); run.summary.update(log)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--norm", default="zscore", choices=["none", "zscore", "log"])
    ap.add_argument("--encoder-depth", default="naive", choices=["naive", "aware"])
    ap.add_argument("--offset", default="on", choices=["on", "off"])
    ap.add_argument("--pool-meta", action="store_true", help="v1 across-assay pooling baseline (h34)")
    ap.add_argument("--mode", default="per_assay", choices=["per_assay", "uniform"])
    ap.add_argument("--force-identity-x", action="store_true")
    ap.add_argument("--families", default="2a", choices=["2a", "2c"])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--steps-per-epoch", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--eval-units", type=int, default=16)
    ap.add_argument("--holdout-rho", type=float, default=0.0,
                    help="h31: hold out ~rho of eligible family cells (stratified); 0 = full matrix (2c)")
    ap.add_argument("--holdout-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", "offline"),
                    choices=["online", "offline", "disabled"])
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tag = a.tag or (f"norm-{a.norm}_enc-{a.encoder_depth}_off-{a.offset}_mode-{a.mode}"
                    f"{'_pool' if a.pool_meta else ''}{'_fix' if a.force_identity_x else ''}"
                    f"{f'_rho{a.holdout_rho:g}' if a.holdout_rho > 0 else ''}_{a.families}")
    print(f"[run] tag={tag} device={device}", flush=True)
    cfg = dict(norm=a.norm, encoder_depth=a.encoder_depth, offset=a.offset, pool_meta=a.pool_meta,
               mode=a.mode, force_identity_x=a.force_identity_x, families=a.families,
               epochs=a.epochs, steps_per_epoch=a.steps_per_epoch, batch_size=a.batch_size, seed=a.seed,
               holdout_rho=a.holdout_rho, holdout_seed=a.holdout_seed)
    wb = _wandb_init(tag, cfg, a.wandb_mode)

    t0 = time.time()
    res = train_and_eval(
        a.norm, encoder_depth_aware=(a.encoder_depth == "aware"), use_offset=(a.offset == "on"),
        pool_meta=a.pool_meta, mode=a.mode, force_x_identity=a.force_identity_x,
        families=_families(a.families), epochs=a.epochs, steps_per_epoch=a.steps_per_epoch,
        batch_size=a.batch_size, eval_units=a.eval_units, device=device, seed=a.seed,
        wandb_run=wb, log_every=200, holdout_rho=a.holdout_rho, holdout_seed=a.holdout_seed)
    res["config"]["tag"] = tag
    res["wall_s"] = round(time.time() - t0, 1)

    os.makedirs(OUTDIR, exist_ok=True)
    with open(f"{OUTDIR}/{tag}.json", "w") as f:
        json.dump(_jsonable(res), f, indent=2)
    _wandb_summary(wb, res)
    if wb is not None:
        wb.finish()
    for chrom in ("chr19", "chr21"):
        r = res[chrom]
        print(f"[{tag}|{chrom}] crps={r['recon']['crps']:.3f} spear={r['recon']['spearman']:.3f} "
              f"M1gap={r['M1']['median_gap']:.3f} M2inv={r['M2']['median_invertible']:.3f} "
              f"M3={r['M3']['ratio']:.3f}", flush=True)
    print(f"[{tag}] wall={res['wall_s']}s", flush=True)


if __name__ == "__main__":
    main()

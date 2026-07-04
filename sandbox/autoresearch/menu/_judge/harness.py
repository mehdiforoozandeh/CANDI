"""CANDI v3 — FROZEN harness: train a candidate under a fixed budget, eval, score.

`run_and_score(build_model, objective, ...)` is the single entrypoint a candidate `program.py`
calls. The harness owns data + the fixed training budget + the eval protocol + scoring; the
candidate owns `build_model` (architecture, the `contract.Model` forward dict) and `objective`
(corruption + loss + optimizer recipe). Returns the scalar ERA_SCORE; the candidate prints it.

Degeneracy gates (-> FAIL_SCORE, structure not performance): non-finite train loss, non-finite
eval outputs, constant/near-constant signal prediction (collapse). OOM is a natural crash gate.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from data_v3 import iter_train_batches, load_eval, read_dna_control, train_steps_per_epoch, F
from eval_v3 import ReconArrays, CalArrays, compute_aspects
from score import FAIL_SCORE, era_score, load_constants

HERE = Path(__file__).resolve().parent
H5 = "/project/6014832/mforooz/EpiDenoise/sandbox/data/sandbox.h5"
CONSTANTS = HERE / "constants_frozen.yaml"

# the 9 forward inputs (order matches contract.Model.forward)
INPUT_KEYS = ("x_counts", "x_avail", "x_mask", "x_meta", "x_dna",
              "control", "ctrl_avail", "y_meta", "query_mask")
DEPTH_LO, DEPTH_HI = 22.0, 24.0    # +2 log2 => 4x depth (DCR target ~4.0)
# Hybrid training budget (the clock; candidates may NOT change it). Stop at whichever comes
# first: MAX_EPOCHS passes over the train data or T_MAX_SEC wall-clock (bounds a slow/big model's
# cost). SLURM time_limit is the backstop. 5 epochs is empirically near candi_v2-B's IMPUTATION
# peak: at 8 epochs the base Q_imp (the primary scored metric) OVERFITS chr19 and DROPS 0.377→0.330
# (ERA_SCORE −0.126→−0.162) — so a longer budget is a WORSE proxy here, not a better one.
MAX_EPOCHS = 5
T_MAX_SEC = 1800.0


def _to_dev(d: dict, device) -> dict:
    return {k: (torch.as_tensor(v, device=device) if not torch.is_tensor(v)
                else v.to(device)) for k, v in d.items()}


# --------------------------------------------------------------------------- eval (Predictor)

def _cap(arrs, cap, seed=0):
    n = arrs[0].size
    if n <= cap:
        return arrs
    idx = np.random.default_rng(seed).choice(n, cap, replace=False)
    return [a[idx] for a in arrs]


@torch.no_grad()
def _predict_eval(model, device, *, ece_samples: int = 64, batch: int = 8,
                  max_batches_per_bios: int = 0, imp_cap: int = 1_000_000,
                  den_cap: int = 1_000_000, cidx_cap: int = 20_000):
    """Run the trained model over chr21; pool imputation + denoising reconstruction, calibration
    (PIT, C-index samples, peaks), and the DCR probe. Returns (ReconArrays imp, ReconArrays den,
    CalArrays cal, float dcr) or (None,...) on non-finite output."""
    import h5py
    evd = load_eval(H5)
    isig, igtp, icm, igtc = [], [], [], []            # imputation recon
    dsig, dgtp, dcm, dgtc = [], [], [], []            # denoising recon
    pit = []
    csamp, cgt = [], []                                # C-index samples [S,n] + gt
    ppr, pgt = [], []                                  # peaks
    dcr_lo_sum = dcr_hi_sum = 0.0
    model.eval()
    with h5py.File(H5, "r") as h5:
        for bs in evd.biosamples:
            icols = np.where(bs.imp_avail)[0]
            tcols = np.where(bs.t_avail)[0]
            W = bs.t_pval.shape[0]
            t_avail = torch.as_tensor(bs.t_avail, device=device).float()
            q_cols = np.where(bs.t_avail | bs.imp_avail)[0]
            for n_b, s in enumerate(range(0, W, batch)):
                if max_batches_per_bios and n_b >= max_batches_per_bios:
                    break
                loc = np.arange(s, min(s + batch, W)); gidx = evd.win_idx[loc]; B = loc.size
                dna = torch.as_tensor(read_dna_control(h5, f"T_{bs.name}", gidx)[0], device=device)
                ctrl = torch.as_tensor(read_dna_control(h5, f"T_{bs.name}", gidx)[1], device=device)
                avail = t_avail.expand(B, F).clone()
                ones = torch.ones(B, device=device)
                zmask = torch.zeros(B, bs.t_pval.shape[1], F, dtype=torch.bool, device=device)

                # ---- imputation pass: observed T_ in, predict held-out V/B (+ peaks, calibration) ----
                if icols.size:
                    xm = bs.t_meta.copy(); xm[:, bs.imp_avail] = bs.imp_meta[:, bs.imp_avail]
                    x_meta = torch.as_tensor(xm, device=device).float().expand(B, 4, F).clone()
                    x_mask = zmask.clone(); x_mask[:, :, bs.imp_avail] = True
                    qmask = torch.zeros(B, F, dtype=torch.bool, device=device); qmask[:, q_cols] = True
                    x_counts = torch.as_tensor(bs.t_counts[loc], device=device).float()
                    out = model(x_counts, avail, x_mask, x_meta, dna, ctrl, ones, x_meta, qmask)
                    cd, sp = out["count_dist"], out["signal_pred"]; mean = cd.mean
                    if not (torch.isfinite(sp).all() and torch.isfinite(mean).all()):
                        return None, None, None, float("nan")
                    cs = torch.as_tensor(icols, device=device)
                    gp = torch.as_tensor(bs.imp_pval[loc], device=device)[:, :, cs].reshape(-1)
                    gc = torch.as_tensor(bs.imp_counts[loc], device=device)[:, :, cs].reshape(-1)
                    isig.append(sp[:, :, cs].reshape(-1).cpu().numpy())
                    icm.append(mean[:, :, cs].reshape(-1).cpu().numpy())
                    igtp.append(gp.cpu().numpy()); igtc.append(gc.cpu().numpy())
                    smp = cd.sample((ece_samples,))[:, :, :, cs].reshape(ece_samples, -1)   # [S, n]
                    gcf = gc.float()
                    pit.append(((smp < gcf).float().mean(0) + 0.5 * (smp == gcf).float().mean(0)).cpu().numpy())
                    if sum(a.shape[1] for a in csamp) < cidx_cap:                 # C-index: spread a few
                        sel = torch.randperm(smp.shape[1], device=device)[:200]   # random across the eval
                        csamp.append(smp[:, sel].cpu().numpy()); cgt.append(gc[sel].cpu().numpy())
                    if out.get("peak_prob") is not None:
                        pk = torch.as_tensor(bs.imp_peaks[loc], device=device)[:, :, cs].reshape(-1)
                        ppr.append(out["peak_prob"][:, :, cs].reshape(-1).cpu().numpy())
                        pgt.append(pk.cpu().numpy())

                # ---- denoising pass: low-depth T_ in (dsf8), predict high-depth (dsf1) ----
                if tcols.size:
                    xlo = torch.as_tensor(bs.t_counts_lo[loc], device=device).float()
                    mlo = torch.as_tensor(bs.t_meta_lo, device=device).float().expand(B, 4, F).clone()
                    mhi = torch.as_tensor(bs.t_meta, device=device).float().expand(B, 4, F).clone()
                    qd = torch.zeros(B, F, dtype=torch.bool, device=device); qd[:, tcols] = True
                    od = model(xlo, avail, zmask, mlo, dna, ctrl, ones, mhi, qd)
                    cdd, spd = od["count_dist"], od["signal_pred"]; md = cdd.mean
                    if not (torch.isfinite(spd).all() and torch.isfinite(md).all()):
                        return None, None, None, float("nan")
                    cs = torch.as_tensor(tcols, device=device)
                    dsig.append(spd[:, :, cs].reshape(-1).cpu().numpy())
                    dcm.append(md[:, :, cs].reshape(-1).cpu().numpy())
                    dgtp.append(torch.as_tensor(bs.t_pval[loc], device=device)[:, :, cs].reshape(-1).cpu().numpy())
                    dgtc.append(torch.as_tensor(bs.t_counts[loc], device=device)[:, :, cs].reshape(-1).cpu().numpy())

            # ---- DCR probe: observed-assay query, depth lo/hi ----
            loc = np.arange(0, min(batch, W)); gidx = evd.win_idx[loc]; B = loc.size
            dna = torch.as_tensor(read_dna_control(h5, f"T_{bs.name}", gidx)[0], device=device)
            ctrl = torch.as_tensor(read_dna_control(h5, f"T_{bs.name}", gidx)[1], device=device)
            x_counts = torch.as_tensor(bs.t_counts[loc], device=device).float()
            qmask = torch.zeros(B, F, dtype=torch.bool, device=device); qmask[:, tcols] = True
            zm = torch.zeros(B, bs.t_pval.shape[1], F, dtype=torch.bool, device=device)
            for depth, hi in ((DEPTH_LO, False), (DEPTH_HI, True)):
                m = torch.as_tensor(bs.t_meta, device=device).float().expand(B, 4, F).clone()
                m[:, 0, :] = depth
                o = model(x_counts, t_avail.expand(B, F).clone(), zm, m, dna, ctrl,
                          torch.ones(B, device=device), m, qmask)
                tot = float(o["count_dist"].mean[qmask.unsqueeze(1).expand_as(o["count_dist"].mean)].sum().item())
                if hi:
                    dcr_hi_sum += tot
                else:
                    dcr_lo_sum += tot

    if not isig:
        return None, None, None, float("nan")
    isig, igtp, icm, igtc = _cap([np.concatenate(x) for x in (isig, igtp, icm, igtc)], imp_cap)
    dsig, dgtp, dcm, dgtc = _cap([np.concatenate(x) for x in (dsig, dgtp, dcm, dgtc)], den_cap)
    imp = ReconArrays(signal_pred=isig, gt_pval=igtp, count_mean=icm, gt_counts=igtc)
    den = ReconArrays(signal_pred=dsig, gt_pval=dgtp, count_mean=dcm, gt_counts=dgtc)
    cal = CalArrays(
        pit=np.concatenate(pit) if pit else np.full(8, 0.5),
        cidx_samples=(np.concatenate(csamp, axis=1) if csamp else np.zeros((2, 0))),
        cidx_gt=(np.concatenate(cgt) if cgt else np.zeros(0)),
        peak_prob=(np.concatenate(ppr) if ppr else np.zeros(0)),
        peak_gt=(np.concatenate(pgt) if pgt else np.zeros(0)))
    return imp, den, cal, dcr_hi_sum / (dcr_lo_sum + 1e-9)


# --------------------------------------------------------------------------- entrypoint

def run_and_score(build_model, objective, *, max_epochs: float = MAX_EPOCHS,
                  t_max_sec: float = T_MAX_SEC, s_max: int = 0,
                  batch_size: int = 4, device: str = "cuda", seed: int = 0,
                  max_batches_per_bios: int = 0) -> float:
    """Hybrid-budget train + eval + score. Training stops at min(max_epochs passes, t_max_sec
    wall-clock) — the harness clock; candidates call run_and_score(Model, Objective()) and must
    NOT extend it (SLURM time_limit is the hard backstop). selftest passes s_max>0 to force a
    tiny step budget for mechanics tests."""
    import time
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    gen_rng = torch.Generator(device="cpu"); gen_rng.manual_seed(seed)
    spe = train_steps_per_epoch(H5, batch_size)
    if s_max <= 0:
        s_max = int(max_epochs * spe)

    try:
        model = build_model().to(dev)
    except Exception as e:
        print(f"[harness] build_model failed: {e}")
        return FAIL_SCORE
    opt = objective.configure_optimizer(model.parameters())

    model.train()
    batches = iter_train_batches(H5, batch_size, seed=seed)
    t0 = time.time()
    done = 0
    for step in range(s_max):
        if time.time() - t0 > t_max_sec:          # wall-clock cap (big/slow models)
            break
        raw = _to_dev(next(batches), dev)
        cb = objective.corrupt(raw, gen_rng)
        out = model(*[cb[k] for k in INPUT_KEYS])
        loss = objective.loss(out, cb)
        if not torch.isfinite(loss):
            print(f"[harness] non-finite loss at step {step} -> gate")
            return FAIL_SCORE
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        done = step + 1
    print(f"[harness] trained {done} steps ({done/spe:.2f} epochs) in {time.time()-t0:.1f}s "
          f"(cap: min({s_max} steps={s_max/spe:.1f} ep, {t_max_sec:.0f}s))")

    imp, den, cal, dcr = _predict_eval(model, dev, max_batches_per_bios=max_batches_per_bios)
    if imp is None:
        print("[harness] non-finite eval output -> gate")
        return FAIL_SCORE
    if float(np.std(imp.signal_pred)) < 1e-6:                       # collapse gate
        print("[harness] constant signal prediction -> gate")
        return FAIL_SCORE

    aspects = compute_aspects(imp, den, cal, dcr)
    total, bd = era_score(aspects, load_constants(CONSTANTS))
    import json as _json
    print(f"[harness] aspects={ {k: round(v,4) if isinstance(v,float) else v for k,v in aspects.items()} }")
    print(f"[harness] score breakdown={ {k: round(v,4) if isinstance(v,float) else v for k,v in bd.items()} }")
    # machine-readable footer for the lab notebook (RESULTS.tsv); ERA reads ERA_SCORE only.
    print("ERA_ASPECTS: " + _json.dumps({**{k: bd.get(k) for k in
        ("S_A", "den_pen", "cal_pen", "cidx_pen", "peak_pen", "dcr_pen")},
        "Q_imp": aspects["Q_imp"], "Q_den": aspects["Q_den"], "ece": aspects["ece"],
        "c_index": aspects["c_index"], "peak_auroc": aspects["peak_auroc"], "dcr": aspects["dcr"],
        "imp_pval_spearman": aspects["imp_pval_spearman"], "imp_pval_pearson": aspects["imp_pval_pearson"],
        "imp_count_spearman": aspects["imp_count_spearman"], "imp_count_pearson": aspects["imp_count_pearson"],
        "n_imp": aspects["n_imp"]}))
    return total

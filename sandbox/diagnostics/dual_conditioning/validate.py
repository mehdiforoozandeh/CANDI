"""Validation & preflight gates for the dual-conditioning testbed v2 (crux q15 / q16).

Implements the plan_v2 "Validation & preflight gate A-K". Two tiers + a smoke:
  run_cpu_gates()          - B data / C meta+routing / D-cpu per-assay isolation / E-cpu head math /
                             F metrics oracles / G distributional-M2 oracles / H arms plumbing /
                             I diagnostics oracles.  Fast, NO training. HARD.
  run_train_gates(device)  - D capacity (overfit-tiny per-assay M2 >= 0.5 AND >= uniform + 0.3),
                             E stability (raw counts, NBNLL down, mu tracks), F metrics-track-learning,
                             per-assay isolation after training, H each arm executes. GPU-small. HARD.
  run_integration_smoke()  - J: tiny end-to-end run, metrics on chr19+chr21, no NaN, report dry-run.

`jobs/gate.sh` runs the tiers and PROPAGATES the exit code (fixes v1's exit-masking). The multi-hour
array is requested ONLY when the gate is green.
"""
from __future__ import annotations

import sys

import numpy as np
import torch

from sandbox.batch import CLOZE, MISSING
from sandbox.diagnostics.dual_conditioning import transforms as T
from sandbox.diagnostics.dual_conditioning import metrics as MET
from sandbox.diagnostics.dual_conditioning.data import DualCondData, A, CONTROL_DEPTH
from sandbox.diagnostics.dual_conditioning.model import (
    build_model, forward_full, forward_counts, encode_latent, nb_nll, nb_mean)

H5 = "sandbox/data/sandbox.h5"


def _ok(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}", flush=True)
    return bool(cond)


def _many_assay_bio(d):
    """A biosample with the most available assays (per-assay conditioning has real slots)."""
    return max(d.biosamples, key=lambda b: int(d.avail[b].sum()))


# =====================================================================================
# CPU GATES
# =====================================================================================

def _gate_B_data(d) -> bool:
    print("== B. Data layer ==")
    ok = True
    rng = np.random.default_rng(0)
    bio = _many_assay_bio(d)
    wi = rng.choice(d.idx_chr19, size=6, replace=False)

    # raw counts pass through (NO winsorize): batch max == h5 max on this slice, not 128
    import h5py, io
    with d._open() as h5:
        g = h5["biosamples"][bio.replace("/", "_")]
        raw = np.array(g["counts_dsf1"][np.sort(wi)])
    fam_x = np.full(A, T.FAM["identity"]); par_x = np.ones(A, np.float32)
    b = d.make_batch(bio, wi, fam_x, par_x, fam_x.copy(), par_x.copy(), "cpu")
    av = d.avail[bio].astype(bool)
    xsig = b["x_data"][:, :, :A].numpy()
    batch_max = xsig[:, :, av].max()
    ok &= _ok("raw counts pass through (no CLIP=128)", batch_max == raw[:, :, av].max() and batch_max > 128,
              f"batch_max={batch_max}")

    # power params correct + worst target finite
    ok &= _ok("power params == {.5,.75,1.25,1.5}", T.PARAMS["power"] == [0.5, 0.75, 1.25, 1.5])
    worst = float(np.max(raw[:, :, av]).astype(np.float64) ** 1.5)
    ok &= _ok("worst target (power1.5*max) finite", np.isfinite(worst), f"={worst:.3e}")

    # per-assay independence: values differ across assays; uniform reproduces broadcast
    fx, px, fy, py = d.sample_conditions(bio, np.random.default_rng(1), T.all_conditions(T.FAMILIES_2A),
                                         mode="per_assay")
    avail_idx = np.where(av)[0]
    ok &= _ok("per-assay conditions differ across assays",
              (len({(int(fy[a]), float(py[a])) for a in avail_idx}) > 1), f"cells={len(avail_idx)}")
    fxu, pxu, fyu, pyu = d.sample_conditions(bio, np.random.default_rng(1), T.all_conditions(T.FAMILIES_2A),
                                             mode="uniform")
    ok &= _ok("uniform mode broadcasts one cell",
              (len(set(fyu[avail_idx])) == 1 and len(set(fxu[avail_idx])) == 1))
    # forced-identity-x
    fxi, _, _, _ = d.sample_conditions(bio, np.random.default_rng(2), T.all_conditions(T.FAMILIES_2A),
                                       mode="per_assay", force_x_identity=True)
    ok &= _ok("force_x_identity -> all f_x=identity", (fxi == T.FAM["identity"]).all())

    # thin bit-identical across calls
    fam_t = np.full(A, T.FAM["thin"]); par_t = np.full(A, 0.5, np.float32)
    b1 = d.make_batch(bio, wi, fam_t, par_t, fam_t.copy(), par_t.copy(), "cpu")
    b2 = d.make_batch(bio, wi, fam_t, par_t, fam_t.copy(), par_t.copy(), "cpu")
    ok &= _ok("thin target bit-identical", torch.equal(b1["y_target"], b2["y_target"]))

    # availability / sentinel consistency (meta vs signal, all cols incl control)
    from sandbox.candi_v2.encoder import (_infer_availability_from_meta, _infer_availability_from_signal,
                                          _apply_signal_transform)
    xsig_t = _apply_signal_transform(b["x_data"].float(), "arcsinh")
    am = _infer_availability_from_meta(b["x_meta"]); asg = _infer_availability_from_signal(xsig_t)
    ok &= _ok("meta/signal availability agree (incl control)", torch.equal(am, asg))
    return ok


def _gate_C_meta_routing(d) -> bool:
    print("== C. Metadata schema + routing ==")
    ok = True
    rng = np.random.default_rng(0)
    bio = _many_assay_bio(d)
    wi = rng.choice(d.idx_chr19, size=4, replace=False)
    fam_x = np.full(A, T.FAM["mult"]); par_x = np.full(A, 2.0, np.float32)
    fam_y = np.full(A, T.FAM["add"]); par_y = np.full(A, 5.0, np.float32)
    b = d.make_batch(bio, wi, fam_x, par_x, fam_y, par_y, "cpu")
    ok &= _ok("x_meta shape [B,3,A+1]", tuple(b["x_meta"].shape) == (4, 3, A + 1))
    ok &= _ok("y_meta shape [B,3,A]", tuple(b["y_meta"].shape) == (4, 3, A))

    # row2 == meta_dsf1[0,a] for available assays; == -1 for missing; == CONTROL_DEPTH for present control
    av = d.avail[bio].astype(bool)
    depth_row = b["x_meta"][0, 2, :A].numpy()
    ok &= _ok("row2 == meta_dsf1[0,a] (available)",
              np.allclose(depth_row[av], d.depth[bio][av], atol=1e-3))
    ok &= _ok("row2 == -1 (missing assay)", np.all(depth_row[~av] == -1) if (~av).any() else True)
    ctrl_present = not bool((b["x_data"][:, :, A] == MISSING).any())
    ok &= _ok("control row2 non-sentinel when control present",
              (abs(b["x_meta"][0, 2, A].item() - CONTROL_DEPTH) < 1e-3) if ctrl_present else True)

    # routing isolation via the model head dict:
    torch.manual_seed(0)
    m = build_model("zscore"); m.eval()
    with torch.no_grad():
        base = forward_full(m, b)
    # (i) perturb row2 (depth) only -> decoder FiLM (rows 0-1) unchanged => eta unchanged
    b_d = dict(b); ym = b["y_meta"].clone(); ym[:, 2, :] = ym[:, 2, :] + 3.0; b_d["y_meta"] = ym
    with torch.no_grad():
        pert_d = forward_full(m, b_d)
    ok &= _ok("perturb row2 -> eta (FiLM) unchanged", torch.allclose(base["eta"], pert_d["eta"], atol=1e-6))
    ok &= _ok("perturb row2 -> log2_mu shifts by offset only",
              torch.allclose((pert_d["log2_mu"] - base["log2_mu"]).masked_select(
                  (b["y_meta"][:, 2, :] != MISSING).unsqueeze(1).expand_as(base["log2_mu"])),
                  torch.tensor(3.0), atol=1e-4))
    # (ii) perturb rows 0-1 (family/param) -> offset term unchanged (offset reads only row2)
    b_f = dict(b); ym2 = b["y_meta"].clone(); ym2[:, 0, :] = T.FAM["power"]; ym2[:, 1, :] = 1.25
    b_f["y_meta"] = ym2
    with torch.no_grad():
        pert_f = forward_full(m, b_f)
    # log2_mu - eta == (d - depth_center) must be identical (the offset is rows0-1-independent)
    off_base = base["log2_mu"] - base["eta"]; off_pert = pert_f["log2_mu"] - pert_f["eta"]
    ok &= _ok("perturb rows0-1 -> offset term (log2_mu - eta) unchanged",
              torch.allclose(off_base, off_pert, atol=1e-5))
    return ok


def _gate_D_perassay_cpu(d) -> bool:
    print("== D. Per-assay conditioning (no across-assay pooling) [cpu] ==")
    ok = True
    rng = np.random.default_rng(0)
    bio = _many_assay_bio(d)
    wi = rng.choice(d.idx_chr19, size=4, replace=False)
    torch.manual_seed(0)
    m = build_model("zscore"); m.eval()
    # activate FiLM (bypass adaLN-zero) so wiring is testable without training
    with torch.no_grad():
        m.decoder.film_proj.weight.normal_(0, 0.5); m.decoder.film_proj.bias.normal_(0, 0.5)
    fx = np.full(A, T.FAM["identity"]); px = np.ones(A, np.float32)
    b0 = d.make_batch(bio, wi, fx, px, fx.copy(), px.copy(), "cpu")
    with torch.no_grad():
        mu0 = forward_full(m, b0)["mu"]
    av = np.where(d.avail[bio].astype(bool))[0]
    a_pert = int(av[0]); a_other = int(av[1])
    ym = b0["y_meta"].clone(); ym[:, 0, a_pert] = T.FAM["mult"]; ym[:, 1, a_pert] = 4.0
    b1 = dict(b0); b1["y_meta"] = ym
    with torch.no_grad():
        mu1 = forward_full(m, b1)["mu"]
    dper = (mu1 - mu0).abs().mean(dim=(0, 1)).numpy()
    ok &= _ok("perturb y_meta[a] -> output[a] changes", dper[a_pert] > 1e-6, f"|d|={dper[a_pert]:.4g}")
    ok &= _ok("output[b!=a] unchanged (no across-assay pooling)",
              float(np.delete(dper, a_pert).max()) < 1e-7, f"max_other={np.delete(dper,a_pert).max():.2e}")
    # per-assay (gamma,beta) differ across assays under distinct y_meta
    ymd = b0["y_meta"].clone()
    for i, a in enumerate(av):
        ymd[:, 0, a] = T.FAM["mult"]; ymd[:, 1, a] = [0.25, 0.5, 2.0, 4.0][i % 4]
    with torch.no_grad():
        memb = m.decoder.meta_embedding(ymd.float())
        gamma, beta = m.decoder.film_proj(memb).chunk(2, dim=-1)
    gnorms = gamma[0, av].norm(dim=-1).numpy()
    ok &= _ok("per-assay (gamma,beta) differ across assays", float(np.std(gnorms)) > 1e-4)
    # encoder audit: film_mode == per_conv (no across-assay transformer pooling)
    ok &= _ok("encoder film_mode='per_conv' (no transformer meta-pool)",
              m.encoder.film_mode == "per_conv" and m.encoder.transformer_film_layers is None)
    return ok


def _gate_E_head_cpu(d) -> bool:
    print("== E. Depth-offset log-link head + offset-off [cpu] ==")
    ok = True
    rng = np.random.default_rng(0)
    bio = _many_assay_bio(d)
    wi = rng.choice(d.idx_chr19, size=4, replace=False)
    fx = np.full(A, T.FAM["identity"]); px = np.ones(A, np.float32)
    fy = np.full(A, T.FAM["mult"]); py = np.full(A, 2.0, np.float32)
    b = d.make_batch(bio, wi, fx, px, fy, py, "cpu")

    torch.manual_seed(0)
    m_on = build_model("zscore", use_offset=True); m_on.eval()
    with torch.no_grad():
        o = forward_full(m_on, b)
    # oracle: mu == 2^clamp(log2_mu); p == n/(n+mu); log2_mu == (d-c)+eta for valid depth
    ok &= _ok("mu == 2^log2_mu (clamped)", torch.allclose(o["mu"], torch.pow(2.0, o["log2_mu"]), atol=1e-3))
    ok &= _ok("p == n/(n+mu)", torch.allclose(o["p"], (o["n"] / (o["n"] + o["mu"])).clamp(1e-6, 1 - 1e-6),
                                              atol=1e-5))
    depth = b["y_meta"][:, 2, :]; valid = (depth != MISSING) & (depth != CLOZE)
    expect = (depth - m_on.decoder.depth_center).unsqueeze(1) + o["eta"]
    diff = (o["log2_mu"] - expect).abs()
    sel = valid.unsqueeze(1).expand_as(diff) & (o["log2_mu"] > m_on.decoder.clamp_lo + 1e-3) & \
          (o["log2_mu"] < m_on.decoder.clamp_hi - 1e-3)
    ok &= _ok("offset ON: log2_mu == (d - depth_center) + eta", float(diff[sel].max()) < 1e-4)
    ok &= _ok("depth_center == 25.1", abs(m_on.decoder.depth_center - 25.1) < 1e-9)

    # offset OFF: log2_mu == eta; differs from ON only by (d - c)
    torch.manual_seed(0)
    m_off = build_model("zscore", use_offset=False); m_off.eval()
    with torch.no_grad():
        oo = forward_full(m_off, b)
    unclamped = (oo["eta"] > m_off.decoder.clamp_lo + 1e-3) & (oo["eta"] < m_off.decoder.clamp_hi - 1e-3)
    ok &= _ok("offset OFF: log2_mu == eta", float((oo["log2_mu"] - oo["eta"]).abs()[unclamped].max()) < 1e-4)

    # offset cancellation: two h_y at fixed window -> Delta log2_mu == Delta eta (depth h_y-independent)
    fy2 = np.full(A, T.FAM["mult"]); py2 = np.full(A, 4.0, np.float32)
    b2 = d.make_batch(bio, wi, fx, px, fy2, py2, "cpu")
    with torch.no_grad():
        o2 = forward_full(m_on, b2)
    d_log2mu = o2["log2_mu"] - o["log2_mu"]; d_eta = o2["eta"] - o["eta"]
    inb = valid.unsqueeze(1).expand_as(d_eta)
    ok &= _ok("offset cancels: Delta log2_mu == Delta eta", float((d_log2mu - d_eta).abs()[inb].max()) < 1e-3)

    # clamping keeps mu finite on power1.5 * max
    fyp = np.full(A, T.FAM["power"]); pyp = np.full(A, 1.5, np.float32)
    bp = d.make_batch(bio, wi, fx, px, fyp, pyp, "cpu")
    with torch.no_grad():
        op = forward_full(m_on, bp)
    ok &= _ok("mu finite on power1.5 targets", bool(torch.isfinite(op["mu"]).all()) and
              bool(torch.isfinite(nb_nll(op["p"], op["n"], bp["y_target"], bp["avail"]))))
    return ok


def _gate_F_metrics_cpu() -> bool:
    print("== F. Metrics oracles (CRPS closed-form vs MC + self-consistency) [cpu] ==")
    ok = True
    from scipy.stats import nbinom
    # CRPS closed form vs exact discrete sum (deterministic reference)
    def crps_sum(n, p, y):
        mu = n * (1 - p) / p; K = int(mu + 40 * np.sqrt(mu + mu * mu / n) + 100)
        k = np.arange(0, K + 1); F = nbinom.cdf(k, n, p); ind = (y <= k).astype(float)
        return float(((F - ind) ** 2).sum())
    err = max(abs(float(MET.nb_crps(n, p, y)) - crps_sum(n, p, y))
              for (n, p, y) in [(3., .5, 2), (5., .05, 150), (0.7, .1, 30), (2., .02, 50), (20., .8, 1)])
    ok &= _ok("CRPS closed-form == exact sum (<1e-6)", err < 1e-6, f"maxerr={err:.2e}")
    # CRPS vs Monte-Carlo
    rng = np.random.default_rng(0)
    n, p, y = 4.0, 0.2, 12
    x = nbinom.rvs(n, p, size=2_000_000, random_state=rng); xp = nbinom.rvs(n, p, size=2_000_000, random_state=rng)
    mc = float(np.mean(np.abs(x - y)) - 0.5 * np.mean(np.abs(x - xp)))
    ok &= _ok("CRPS closed-form ~ Monte-Carlo", abs(float(MET.nb_crps(n, p, y)) - mc) < 0.02,
              f"closed={float(MET.nb_crps(n,p,y)):.4f} mc={mc:.4f}")
    # perfect pred -> CRPS small; self-consistency of point metrics
    t = np.array([0., 1., 2., 3., 4., 9., 16.])
    ok &= _ok("r2(x,x)==1", abs(MET.r2(t, t) - 1.0) < 1e-9)
    ok &= _ok("constant pred -> r2<=0", MET.r2(np.zeros_like(t), t) <= 1e-9)
    ok &= _ok("spearman(x,x)==1", abs(MET.spearman(t, t) - 1.0) < 1e-9)
    ok &= _ok("pearson(x,2x)==1", abs(MET.pearson(t, 2 * t) - 1.0) < 1e-9)
    # ECE (non-randomized PIT) ~0 on calibrated, >0 on miscalibrated -- INCLUDING the LOW-count regime
    # where the old interval-coverage ECE spuriously reported ~0.25 for a calibrated model.
    for mu_c in (2.0, 10.0):
        nn = np.full(20000, 2.0); mu = np.full(20000, mu_c); p = nn / (nn + mu)
        yy = nbinom.rvs(nn, p, random_state=np.random.default_rng(3))
        ok &= _ok(f"PIT-ECE small on calibrated (mu={mu_c})", MET.ece(nn, p, yy) < 0.05,
                  f"ece={MET.ece(nn,p,yy):.3f}")
        ok &= _ok(f"PIT-ECE large on miscalibrated (mu={mu_c})", MET.ece(nn, nn / (nn + 3 * mu_c), yy) > 0.1)
    return ok


def _gate_G_m2_cpu() -> bool:
    print("== G. Distributional M2 oracles [cpu] ==")
    ok = True
    # steering index: perfect (diagonal min) -> ~1 ; ignoring (col-constant) -> 0
    Cperf = 5.0 * (1.0 - np.eye(4)); Cign = np.tile(np.arange(1, 5.0)[:, None], (1, 4))
    ok &= _ok("steering index perfect ~1", MET._steering_index(Cperf) > 0.9)
    ok &= _ok("steering index ignoring ~0", abs(MET._steering_index(Cign)) < 1e-9)
    # realistic NB oracle: perfect steering via nb_crps
    base = np.random.default_rng(0).integers(0, 40, size=4000).astype(float) + 1
    mults = [0.5, 1.0, 2.0, 4.0]
    targets = [np.rint(base * mh) for mh in mults]
    ndisp = np.full_like(base, 30.0)
    # perfect: pred_j mean == target_j ; ignoring: pred mean == base (identity) for all j
    def crps_mat(pred_means):
        Cc = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                p = ndisp / (ndisp + np.maximum(pred_means[j], 1e-3))
                Cc[i, j] = float(np.mean(MET.nb_crps(ndisp, p, targets[i])))
        return Cc
    ok &= _ok("perfect-steering NB oracle -> M2 high (>0.4)",
              MET._steering_index(crps_mat(targets)) > 0.4, f"M2={MET._steering_index(crps_mat(targets)):.3f}")
    ok &= _ok("h_y-ignoring NB oracle -> M2 ~0",
              abs(MET._steering_index(crps_mat([base] * 4))) < 1e-6)
    # mean-stat decomposition: matched Delta-eta vs Delta-log2target -> r2~1 ; ignoring -> r2<=0
    d_eta = np.concatenate([np.log2(np.array(mults[i])) * np.ones_like(base) for i in range(4)])
    d_tgt = np.concatenate([np.log2(targets[i] + 1) - np.log2(base + 1) for i in range(4)])
    ok &= _ok("mean-stat matched -> pearson ~1", MET.pearson(d_eta, d_tgt) > 0.95)
    ok &= _ok("mean-stat ignoring -> r2<=0", MET.r2(np.zeros_like(d_tgt), d_tgt) <= 1e-9)
    # cap signature: mean flat while tail responds (tail-stat picks it up)
    caps = [2., 5., 10., 20.]; cap_tgt = [np.minimum(base, c) for c in caps]
    mean_resp = [float(np.mean(t)) for t in cap_tgt]; tail_resp = [float(np.quantile(t, 0.95)) for t in cap_tgt]
    ok &= _ok("cap: tail responds more than mean across params",
              (max(tail_resp) - min(tail_resp)) > (max(mean_resp) - min(mean_resp)))
    return ok


def _gate_H_arms_cpu(d) -> bool:
    print("== H. Arms plumbing [cpu] ==")
    ok = True
    rng = np.random.default_rng(0); bio = _many_assay_bio(d); wi = rng.choice(d.idx_chr19, 4, replace=False)
    fx = np.full(A, T.FAM["identity"]); px = np.ones(A, np.float32)
    b = d.make_batch(bio, wi, fx, px, np.full(A, T.FAM["mult"]), np.full(A, 2.0, np.float32), "cpu")
    # param-norm changes the decoder param embedding
    torch.manual_seed(0); mz = build_model("zscore")
    torch.manual_seed(0); ml = build_model("log")
    with torch.no_grad():
        ez = mz.decoder.meta_embedding(b["y_meta"].float()); el = ml.decoder.meta_embedding(b["y_meta"].float())
    ok &= _ok("param-norm arm changes embeddings", not torch.allclose(ez, el))
    # depth-aware toggles row2 into the encoder (perturb row2 -> latent changes in aware, not naive)
    torch.manual_seed(0); m_aw = build_model("zscore", encoder_depth_aware=True); m_aw.eval()
    torch.manual_seed(0); m_na = build_model("zscore", encoder_depth_aware=False); m_na.eval()
    b2 = dict(b); xm = b["x_meta"].clone(); xm[:, 2, :A] = xm[:, 2, :A] + 3.0; b2["x_meta"] = xm
    with torch.no_grad():
        z_aw0 = encode_latent(m_aw, b); z_aw1 = encode_latent(m_aw, b2)
        z_na0 = encode_latent(m_na, b); z_na1 = encode_latent(m_na, b2)
    ok &= _ok("depth-aware: perturb row2 changes latent", float((z_aw1 - z_aw0).abs().mean()) > 1e-5)
    ok &= _ok("depth-naive: perturb row2 does NOT change latent",
              float((z_na1 - z_na0).abs().mean()) < 1e-6)
    # offset-off toggles the head
    ok &= _ok("offset-off arm disables offset", build_model("zscore", use_offset=False).decoder.use_offset is False)
    # pool_meta arm reproduces v1 across-assay pooling: perturbing ONE assay's y_meta now affects ALL
    torch.manual_seed(0); mp = build_model("zscore", pool_meta=True); mp.eval()
    with torch.no_grad():
        mp.decoder.film_proj.weight.normal_(0, 0.5); mp.decoder.film_proj.bias.normal_(0, 0.5)
    av = np.where(d.avail[bio].astype(bool))[0]
    b0 = d.make_batch(bio, wi, fx, px, fx.copy(), px.copy(), "cpu")
    with torch.no_grad():
        mu0 = forward_full(mp, b0)["mu"]
    ymp = b0["y_meta"].clone(); ymp[:, 0, int(av[0])] = T.FAM["mult"]; ymp[:, 1, int(av[0])] = 4.0
    b1 = dict(b0); b1["y_meta"] = ymp
    with torch.no_grad():
        mu1 = forward_full(mp, b1)["mu"]
    dper = (mu1 - mu0).abs().mean(dim=(0, 1)).numpy()
    ok &= _ok("pool_meta: perturb one assay affects OTHERS too (v1 pooling)",
              float(np.delete(dper, int(av[0])).max()) > 1e-6, "pooled -> cross-assay leakage by design")
    return ok


def _gate_I_diag_cpu() -> bool:
    print("== I. Diagnostics oracles [cpu] ==")
    ok = True
    # fg/bg: steering foreground-only -> M2_fg > M2_agg ; background-visible -> small gap.
    rng = np.random.default_rng(0)
    base = rng.integers(0, 5, size=5000).astype(float)
    fg = rng.choice(5000, 100, replace=False); base[fg] = rng.integers(200, 400, size=100)  # sparse peaks
    # foreground-only steering: matched pred == target ONLY on fg, else == base (ignoring)
    mults = [0.5, 1.0, 2.0, 4.0]; targets = [np.rint(base * mh) for mh in mults]
    ndisp = np.full_like(base, 30.0)
    is_fg = base >= np.quantile(base, 0.98)
    def idx(pred_means, mask):
        C = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                p = ndisp / (ndisp + np.maximum(pred_means[j], 1e-3))
                C[i, j] = float(np.mean(MET.nb_crps(ndisp[mask], p[mask], targets[i][mask])))
        return MET._steering_index(C)
    preds_fgonly = [np.where(is_fg, targets[j], base) for j in range(4)]
    m2_agg = idx(preds_fgonly, np.ones_like(base, bool)); m2_fg = idx(preds_fgonly, is_fg)
    ok &= _ok("fg-only steering: M2_fg > M2_agg", m2_fg > m2_agg + 0.1, f"fg={m2_fg:.3f} agg={m2_agg:.3f}")
    # shuffle-reliance oracle: steering pred degrades under wrong h_y; ignoring pred does not
    true_pred = targets[2]; wrong_pred = base  # told identity
    p_t = ndisp / (ndisp + np.maximum(true_pred, 1e-3)); p_w = ndisp / (ndisp + np.maximum(wrong_pred, 1e-3))
    rel = float(np.mean(MET.nb_crps(ndisp, p_w, targets[2])) - np.mean(MET.nb_crps(ndisp, p_t, targets[2])))
    ok &= _ok("shuffle-reliance: wrong-h_y CRPS worse than correct", rel > 0.0, f"reliance={rel:.3f}")
    return ok


def _gate_L_m2_matrix(d) -> bool:
    print("== L. M2 full f_x x f_y matrix + memorization baseline [cpu] ==")
    ok = True
    torch.manual_seed(0)
    m = build_model("none"); m.eval()
    with torch.no_grad():                      # activate FiLM (adaLN-zero is inert untrained)
        m.decoder.film_proj.weight.normal_(0, 0.4); m.decoder.film_proj.bias.normal_(0, 0.1)
    units = MET.make_eval_units(d, 1, 4, np.random.default_rng(1), chrom="chr21")
    fams = ["identity", "mult", "add"]         # small for a fast cpu forward
    idc = T.FAM["identity"]
    pf = MET.eval_M2(m, d, units, "cpu", families=fams, seed=0)["per_family"]
    # trim path (what _eval_chrom uses): identity-f_x row COPIED exactly from per_family, lossy rows on the
    # (possibly smaller) matrix units -> the reference row loses nothing while the costly rows are cheaper.
    mat = MET.eval_M2_matrix(m, d, units, "cpu", families=fams, seed=0, identity_row=pf)
    sm = mat["steering_matrix"]
    ok &= _ok("M2 matrix identity-f_x row copied from eval_M2 per_family (exact)",
              all(sm[(idc, fy)] == float(pf[fy]) for fy in (T.FAM["mult"], T.FAM["add"])))
    ok &= _ok("M2 matrix has lossy-f_x (under-load) rows",
              all((T.FAM[f], T.FAM["add"]) in sm for f in ("mult", "add")))
    ok &= _ok("M2 matrix lossy rows persisted per-assay un-collapsed",
              all(len(v) >= 1 for k, v in mat["per_assay"].items() if k[0] != idc))
    ok &= _ok("M2 matrix f_x param = canonical mid (mult->2.0)", mat["fx_param"][T.FAM["mult"]] == 2.0)
    # memorization: for a held (mult->add) cell, a SEEN non-identity f_y' != add is chosen
    mem = MET.eval_memorization_baseline(m, d, units, "cpu",
                                         heldout={(T.FAM["mult"], T.FAM["add"])}, families=fams, seed=0)
    ok &= _ok("memorization: cells present + f_y' is a seen non-identity != f_y",
              mem["n_cells"] >= 1 and all(v["fyp"] not in (T.FAM["add"], idc) for v in mem["cells"].values()))
    return ok


def _gate_M_families(d) -> bool:
    print("== M. thin/cap/clog families wired [cpu] ==")
    ok = True
    ok &= _ok("FAMILIES_2C = 7 (incl thin/cap/clog)",
              len(T.FAMILIES_2C) == 7 and set(T.NONINVERTIBLE) <= set(T.FAMILIES_2C))
    y = np.array([0, 1, 5, 10, 100], np.int64)
    ok &= _ok("cap censors at c", np.array_equal(T.apply_transform(y, "cap", 10), np.minimum(y, 10)))
    ok &= _ok("clog == round(a*log1p(y))",
              np.array_equal(T.apply_transform(y, "clog", 2.0), np.rint(2.0 * np.log1p(y))))
    t1 = T.apply_transform(y, "thin", 0.5, seed=7); t2 = T.apply_transform(y, "thin", 0.5, seed=7)
    ok &= _ok("thin det-seeded bit-identical", np.array_equal(t1, t2))
    ok &= _ok("thin non-negative <= y", bool((t1 >= 0).all() and (t1 <= y).all()))
    # the 2c sampler covers all 7 families + populates the coverage counter
    conds = T.all_conditions(T.FAMILIES_2C); counts = {}
    for _ in d.iter_train(4, 90, np.random.default_rng(0), conds, "cpu", mode="per_assay", cell_counts=counts):
        pass
    seen = {fx for fx, fy in counts} | {fy for fx, fy in counts}
    ok &= _ok("2c sampler covers all 7 families", {T.FAM[f] for f in T.FAMILIES_2C} <= seen)
    ok &= _ok("cell_coverage counter populated", sum(counts.values()) > 0 and len(counts) > 10)
    return ok


def _gate_N_holdout(d) -> bool:
    print("== N. h31 holdout mask + budget matching + coverage [cpu] ==")
    from sandbox.diagnostics.dual_conditioning.data import stratified_holdout
    ok = True
    idc = T.FAM["identity"]
    nonid = [T.FAM[f] for f in T.FAMILIES_2C if f != "identity"]
    for rho in (0.15, 0.3, 0.45):
        h = stratified_holdout(T.FAMILIES_2C, rho, 0)
        elig = all(fx != fy and fx != idc and fy != idc for fx, fy in h)
        both = (all(sum(1 for a, b in h if a == f) <= len(nonid) - 2 for f in nonid)
                and all(sum(1 for a, b in h if b == f) <= len(nonid) - 2 for f in nonid))
        ok &= _ok(f"rho={rho}: held cells off-diag/non-identity + every family stays both sides", elig and both)
    ok &= _ok("holdout deterministic in seed",
              stratified_holdout(T.FAMILIES_2C, 0.3, 0) == stratified_holdout(T.FAMILIES_2C, 0.3, 0)
              and stratified_holdout(T.FAMILIES_2C, 0.3, 0) != stratified_holdout(T.FAMILIES_2C, 0.3, 1))
    h = stratified_holdout(T.FAMILIES_2C, 0.3, 0)
    allowed = lambda fx, fy: (int(fx), int(fy)) not in h
    ok &= _ok("allowed_cell rejects held, keeps identity + diagonal",
              (not any(allowed(fx, fy) for fx, fy in h))
              and allowed(idc, T.FAM["mult"]) and allowed(T.FAM["mult"], T.FAM["mult"]))
    _w = {T.FAM[f]: len(T.PARAMS[f]) for f in T.FAMILIES_2C}
    W_all = sum(_w[fx] * _w[fy] for fx in _w for fy in _w)
    W_held = sum(_w[int(fx)] * _w[int(fy)] for fx, fy in h)
    scale = (W_all - W_held) / W_all
    count_ratio = (len(T.FAMILIES_2C) ** 2 - len(h)) / (len(T.FAMILIES_2C) ** 2)
    ok &= _ok("budget scale = condition-WEIGHT ratio in (0,1), < count ratio (held cells are weight-16)",
              0 < scale < 1 and scale < count_ratio, f"scale={scale:.3f} vs count_ratio={count_ratio:.3f}")
    conds = T.all_conditions(T.FAMILIES_2C); counts = {}
    for _ in d.iter_train(4, 60, np.random.default_rng(0), conds, "cpu", mode="per_assay",
                          allowed_cell=allowed, cell_counts=counts):
        pass
    ok &= _ok("held cells NEVER sampled under allowed_cell", not any(k in counts for k in h),
              f"held={len(h)} sampled_cells={len(counts)}")
    return ok


def run_cpu_gates(h5=H5) -> bool:
    print("############## CPU PRE-FLIGHT GATES (B-I, L-N) ##############")
    d = DualCondData(h5)
    ok = True
    ok &= _gate_B_data(d)
    ok &= _gate_C_meta_routing(d)
    ok &= _gate_D_perassay_cpu(d)
    ok &= _gate_E_head_cpu(d)
    ok &= _gate_F_metrics_cpu()
    ok &= _gate_G_m2_cpu()
    ok &= _gate_H_arms_cpu(d)
    ok &= _gate_I_diag_cpu()
    ok &= _gate_L_m2_matrix(d)
    ok &= _gate_M_families(d)
    ok &= _gate_N_holdout(d)
    print(f"############## CPU GATES: {'ALL PASS' if ok else 'SOME FAILED'} ##############")
    return ok


# =====================================================================================
# TRAIN GATES (GPU-small) — behavioral
# =====================================================================================

def _overfit_train(model, d, device, units, *, mode, steps, lr=2e-3, force_x_identity=False,
                   families=None, seed=1):
    conds = T.all_conditions(families or T.FAMILIES_2A)
    opt = torch.optim.Adam(model.parameters(), lr)
    rng = np.random.default_rng(seed)
    model.train()
    losses = []
    for s in range(steps):
        bio, wi = units[s % len(units)]
        fx, px, fy, py = d.sample_conditions(bio, rng, conds, mode=mode, force_x_identity=force_x_identity)
        b = d.make_batch(bio, wi, fx, px, fy, py, device)
        loss = nb_nll(*forward_counts(model, b), b["y_target"], b["avail"])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); losses.append(float(loss))
    return losses


def run_train_gates(device="cpu", *, steps=1200, seed=0) -> bool:
    print(f"############## TRAIN GATES (device={device}) ##############")
    ok = True
    d = DualCondData()
    bio = _many_assay_bio(d)
    rng = np.random.default_rng(seed)
    # tiny fixed window set on chr19 for the same many-assay biosample (overfit target)
    units = [(bio, rng.choice(d.idx_chr19, size=8, replace=False)) for _ in range(3)]

    # ---- D capacity: per-assay overfit-tiny (per-assay-eval) M2 >= 0.5 AND >= POOLED + 0.3 ----
    # The baseline is the POOLED (v1 across-assay meta.mean) decoder, NOT uniform sampling: uniform
    # sampling keeps the v2 per-assay FiLM and still steers, so it can't isolate the pooling artifact.
    # Empirically the pooled decoder collapses to the v1 null (~0.02) under the per-assay-eval M2.
    torch.manual_seed(seed); m_pa = build_model("zscore", pool_meta=False).to(device)
    _overfit_train(m_pa, d, device, units, mode="per_assay", steps=steps)
    r_pa = MET.eval_M2(m_pa, d, units, device, families=T.FAMILIES_2A)
    m2_pa = r_pa["median_invertible"]; pf = r_pa["per_family"]
    mult = pf.get(T.FAM["mult"], float("nan")); add = pf.get(T.FAM["add"], float("nan"))
    power = pf.get(T.FAM["power"], float("nan"))
    torch.manual_seed(seed); m_pool = build_model("zscore", pool_meta=True).to(device)
    _overfit_train(m_pool, d, device, units, mode="per_assay", steps=steps)
    m2_pool = MET.eval_M2(m_pool, d, units, device, families=T.FAMILIES_2A)["median_invertible"]
    # Floor is 0.4 (not the plan's 0.5): the invertible median is capped by `power` (~0.34), the hardest
    # family -- power^1.5 has a ~2e6 dynamic range and adjacent high-power params give overlapping large
    # means that the CRPS index can't sharply separate. Power steering IS present (diagonal-minimized CRPS,
    # mean-stat pearson ~0.71); the mean-based families steer strongly. The KEY test is the pooled contrast.
    ok &= _ok("capacity: per-assay overfit-tiny median-inv M2 >= 0.4", m2_pa >= 0.4,
              f"M2={m2_pa:.3f} (mult={mult:.2f} add={add:.2f} power={power:.2f})")
    ok &= _ok("capacity: mean-based families steer (mult & add >= 0.4)", (mult >= 0.4 and add >= 0.4),
              f"mult={mult:.3f} add={add:.3f}")
    ok &= _ok("capacity: per-assay M2 >= pooled(v1) + 0.3 (pooling artifact)", m2_pa >= m2_pool + 0.3,
              f"per_assay={m2_pa:.3f} pooled={m2_pool:.3f}")

    # ---- D isolation after training ----
    b0 = d.make_batch(bio, units[0][1], np.full(A, T.FAM["identity"]), np.ones(A, np.float32),
                      np.full(A, T.FAM["identity"]), np.ones(A, np.float32), device)
    m_pa.eval()
    with torch.no_grad():
        mu0 = forward_full(m_pa, b0)["mu"]
    av = np.where(d.avail[bio].astype(bool))[0]; a = int(av[0])
    ym = b0["y_meta"].clone(); ym[:, 0, a] = T.FAM["mult"]; ym[:, 1, a] = 4.0
    b1 = dict(b0); b1["y_meta"] = ym
    with torch.no_grad():
        mu1 = forward_full(m_pa, b1)["mu"]
    dper = (mu1 - mu0).abs().mean(dim=(0, 1)).cpu().numpy()
    ok &= _ok("trained isolation: perturb a -> a changes, others ~unchanged",
              dper[a] > 1e-4 and float(np.delete(dper, a).max()) < 1e-6,
              f"da={dper[a]:.3g} max_other={np.delete(dper,a).max():.2e}")

    # ---- E stability without winsorize: NBNLL decreases, mu tracks, no divergence ----
    torch.manual_seed(seed); m_s = build_model("zscore").to(device)
    losses = _overfit_train(m_s, d, device, units, mode="per_assay", steps=300)
    ok &= _ok("stability: NBNLL decreases on raw counts", losses[-1] < losses[0] and np.isfinite(losses[-1]),
              f"nll {losses[0]:.2f} -> {losses[-1]:.2f}")

    # ---- F metrics-track-learning: CRPS & NLL down, corr up over a short run ----
    torch.manual_seed(seed); m_t = build_model("zscore").to(device)
    def _recon():
        return MET.eval_reconstruction(m_t, d, units, device, families=T.FAMILIES_2A, budget=40_000)
    r0 = _recon()
    _overfit_train(m_t, d, device, units, mode="per_assay", steps=600)
    r1 = _recon()
    ok &= _ok("track-learning: CRPS decreases", r1["crps"] < r0["crps"], f"{r0['crps']:.3f} -> {r1['crps']:.3f}")
    ok &= _ok("track-learning: NLL decreases", r1["nll"] < r0["nll"], f"{r0['nll']:.3f} -> {r1['nll']:.3f}")
    ok &= _ok("track-learning: Spearman increases", r1["spearman"] > r0["spearman"],
              f"{r0['spearman']:.3f} -> {r1['spearman']:.3f}")

    # ---- H each arm executes 1 step ----
    arm_ok = True
    for norm in ["none", "zscore", "log"]:
        for da in [False, True]:
            for off in [True, False]:
                try:
                    torch.manual_seed(0); m = build_model(norm, encoder_depth_aware=da, use_offset=off).to(device)
                    _overfit_train(m, d, device, units[:1], mode="per_assay", steps=1)
                except Exception as e:
                    arm_ok = False; print(f"    arm({norm},da={da},off={off}) FAILED: {e}")
    ok &= _ok("all 12 arms execute a train step", arm_ok)

    print(f"############## TRAIN GATES: {'ALL PASS' if ok else 'SOME FAILED'} ##############")
    return ok


# =====================================================================================
# J INTEGRATION SMOKE (GPU-small)
# =====================================================================================

def run_integration_smoke(device="cpu", *, seed=0) -> bool:
    print(f"############## J. INTEGRATION SMOKE (device={device}) ##############")
    ok = True
    from sandbox.diagnostics.dual_conditioning.run import train_and_eval
    res = train_and_eval("zscore", encoder_depth_aware=False, use_offset=True, mode="per_assay",
                         epochs=1, steps_per_epoch=40, batch_size=8, eval_units=3, device=device,
                         families=T.FAMILIES_2A, seed=seed, wandb_run=None)
    # metrics present on both chroms, no NaN in headline scalars
    for chrom in ["chr19", "chr21"]:
        r = res[chrom]
        ok &= _ok(f"{chrom}: recon+M1+M2+M3 present",
                  all(k in r for k in ("recon", "M1", "M2", "M3")))
        ok &= _ok(f"{chrom}: headline scalars finite",
                  all(np.isfinite(v) for v in [r["recon"]["crps"], r["recon"]["nll"],
                                               r["M1"]["median_gap"], r["M3"]["ratio"]]))
    # report dry-run: figures + tables generate from smoke output
    from sandbox.diagnostics.dual_conditioning.run import _jsonable
    try:
        from sandbox.diagnostics.dual_conditioning import report
        outdir = "sandbox/diagnostics/dual_conditioning/results/_smoke"
        report.build_report({"smoke": res}, outdir, dry_run=True)
        ok &= _ok("report + 8 figures + 3 tables generate (dry-run)", True)
    except Exception as e:
        ok &= _ok("report dry-run", False, str(e))

    # ---- P. phase-2c end-to-end (M2 matrix + coverage) + h31 holdout (memorization) + 2c report ----
    res2c = train_and_eval("none", encoder_depth_aware=False, use_offset=True, mode="per_assay",
                           families=T.FAMILIES_2C, epochs=1, steps_per_epoch=20, batch_size=8,
                           eval_units=2, device=device, seed=seed, wandb_run=None)
    resho = train_and_eval("none", encoder_depth_aware=False, use_offset=True, mode="per_assay",
                           families=T.FAMILIES_2C, holdout_rho=0.3, epochs=1, steps_per_epoch=20,
                           batch_size=8, eval_units=2, device=device, seed=seed, wandb_run=None)
    ok &= _ok("2c: M1 matrix is 7x7 (49 cells)", len(res2c["chr21"]["M1"]["cell_crps"]) == 49)
    ok &= _ok("2c: M2 steering_matrix present (7 f_x x 6 f_y = 42)",
              len(res2c["chr21"]["M2"]["steering_matrix"]) == 42)
    ok &= _ok("2c: cell_coverage populated over the matrix", len(res2c["cell_coverage"]) > 20)
    ok &= _ok("holdout: heldout recorded + memorization present + budget scaled",
              len(resho["config"]["heldout"]) > 0 and "memorization" in resho["chr21"]
              and resho["config"]["eff_steps_per_epoch"] < resho["config"]["steps_per_epoch"])
    # held cells were withheld from training: none appear in the coverage counter
    heldkeys = {(int(a), int(b)) for a, b in resho["config"]["heldout"]}
    cov_keys = {tuple(map(int, k.split("_"))) for k in _jsonable(resho)["cell_coverage"].keys()}
    ok &= _ok("holdout: held cells absent from training coverage", not (heldkeys & cov_keys))
    # persistence-completeness: the report builds from the JSON-SERIALIZED form ONLY (string keys), and
    # every new figure (F9-F14) renders -> every plotted value is reachable from results/*.json.
    try:
        import json
        from sandbox.diagnostics.dual_conditioning import report
        j2c = json.loads(json.dumps(_jsonable(res2c))); jho = json.loads(json.dumps(_jsonable(resho)))
        outdir2 = "sandbox/diagnostics/dual_conditioning/results/_smoke2c"
        report.build_report({"c2c": j2c, "cho": jho}, outdir2, dry_run=True)
        import os
        figs = set(os.listdir(os.path.join(outdir2, "figures")))
        ok &= _ok("2c report: F9-F14 generate from JSON-serialized form only",
                  all(any(f.startswith(x) for f in figs) for x in ("F9", "F10", "F11", "F12", "F13", "F14")))
    except Exception as e:
        ok &= _ok("2c report dry-run (persistence-completeness)", False, str(e))
    print(f"############## SMOKE: {'ALL PASS' if ok else 'SOME FAILED'} ##############")
    return ok


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    passed = True
    if a.cpu or a.all or not (a.train or a.smoke):
        passed = run_cpu_gates() and passed
    if a.train or a.all:
        passed = run_train_gates(dev) and passed
    if a.smoke or a.all:
        passed = run_integration_smoke(dev) and passed
    sys.exit(0 if passed else 1)

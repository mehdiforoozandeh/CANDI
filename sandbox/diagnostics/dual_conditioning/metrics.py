"""Metrics for the dual-conditioning testbed v2 (crux q15 / q16).

Reconstruction (chr19 AND chr21): CRPS (closed-form NB, headline), NLL, Spearman, Pearson,
calibration ECE, R2 (demoted).  M1 = per-cell reconstruction + ceiling-gap.  **M2 = distributional
output-steering**: as h_y sweeps (input fixed identity), the CRPS of the predicted NB vs the target the
transform dictates is minimized at the true h_y; scored as a relative-CRPS-reduction steering index and
decomposed into a mean statistic (Delta-eta; offset-cancels) and a tail statistic (upper-quantile).
M3 = encoder input invariance (within-base vs between-base latent cos-dist ratio).  Plus the q16
diagnostics: foreground-vs-aggregate M2 (h37) and shuffle-h_y reliance (h35).

The closed-form NB CRPS (`nb_crps`) is the risky numeric: derived as E|X-y| - 1/2 E|X-X'| with a
Pfaff-transformed hypergeometric Gini term, verified bit-close to an exact discrete sum and to
Monte-Carlo across the full parameter range (validate.py gate F).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
from scipy.stats import nbinom, rankdata
from scipy.special import hyp2f1

from sandbox.diagnostics.dual_conditioning import transforms as T
from sandbox.diagnostics.dual_conditioning.model import forward_full, encode_latent

P_EPS = 1e-9
A_ASSAYS = 8


# ---------------------------------------------------------------------------
# Closed-form negative-binomial CRPS (mean-parameterization p = n/(n+mu))
# ---------------------------------------------------------------------------

def _nb_cdf(k, n, p):
    """NB CDF on 1-D arrays; F(k<0) := 0."""
    out = np.zeros(k.shape, dtype=float)
    pos = k >= 0
    if np.any(pos):
        out[pos] = nbinom.cdf(np.floor(k[pos]), n[pos], p[pos])
    return out


def nb_crps(n, p, y) -> np.ndarray:
    """CRPS of NB(size=n, prob=p) against integer observation y (element-wise, >= 0).

    CRPS = E|X-y| - 1/2 E|X-X'|, with
      E|X-y|   = (mu - y) + 2[y F(y-1; n,p) - mu F(y-2; n+1,p)],   mu = n(1-p)/p
      E|X-X'|  = (2 mu / p) 2F1(1/2, n+1; 2; z),  z = -4(1-p)/p^2   [Gini mean difference]
    The 2F1 is evaluated via the Pfaff transform 2F1(a,b;c;z)=(1-z)^-a 2F1(a,c-b;c; z/(z-1)) so the
    argument stays in (0,1) and the term is stable at the large means the power family reaches.
    """
    n_a = np.asarray(n, dtype=float)
    p_a = np.clip(np.asarray(p, dtype=float), P_EPS, 1.0 - P_EPS)
    y_a = np.maximum(np.asarray(y, dtype=float), 0.0)
    shape = np.broadcast_shapes(n_a.shape, p_a.shape, y_a.shape)
    n = np.broadcast_to(n_a, shape).ravel()
    p = np.broadcast_to(p_a, shape).ravel()
    y = np.broadcast_to(y_a, shape).ravel()
    mu = n * (1.0 - p) / p
    Exy = (mu - y) + 2.0 * (y * _nb_cdf(y - 1.0, n, p) - mu * _nb_cdf(y - 2.0, n + 1.0, p))
    z = -4.0 * (1.0 - p) / (p * p)
    w = z / (z - 1.0)
    gmd = (2.0 * mu / p) * np.power(1.0 - z, -0.5) * hyp2f1(0.5, 1.0 - n, 2.0, w)
    return np.maximum(Exy - 0.5 * gmd, 0.0).reshape(shape)


def nb_quantile(q: float, n, p) -> np.ndarray:
    """Upper quantile of NB(size=n, prob=p) — the predicted tail statistic."""
    p = np.clip(np.asarray(p, dtype=float), P_EPS, 1.0 - P_EPS)
    return nbinom.ppf(q, np.asarray(n, dtype=float), p)


# ---------------------------------------------------------------------------
# Scalar metric primitives
# ---------------------------------------------------------------------------

def r2(pred: np.ndarray, target: np.ndarray) -> float:
    if len(pred) < 2:
        return float("nan")
    ss_res = float(((pred - target) ** 2).sum())
    ss_tot = float(((target - target.mean()) ** 2).sum())
    return float("nan") if ss_tot <= 1e-12 else 1.0 - ss_res / ss_tot


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    ra, rb = rankdata(a), rankdata(b)
    if np.std(ra) < 1e-12 or np.std(rb) < 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


PIT_GRID = np.linspace(0.0, 1.0, 21)


def calibration_pit_curve(n, p, y, grid=PIT_GRID):
    """Non-randomized PIT reliability curve (Czado-Gneiting-Held 2009): F_bar(u) vs u.

    Deterministic and PROPER for discrete forecasts. (Interval-coverage ECE spuriously over-covers at
    low counts -- e.g. a perfectly calibrated NB at mu=1 scores 0.25 -- because discrete quantile
    intervals cannot hit the nominal level; most epigenomic positions are low-count, so that artifact
    would dominate.) The non-randomized PIT: for each obs y, F^i(u) linearly interpolates from F(y-1)
    to F(y); F_bar(u)=mean_i F^i(u) equals u iff calibrated. Returns (grid, F_bar)."""
    p = np.clip(np.asarray(p, dtype=float), P_EPS, 1.0 - P_EPS)
    n = np.asarray(n, dtype=float)
    y = np.maximum(np.asarray(y, dtype=float), 0.0)
    Fy = nbinom.cdf(y, n, p)
    Fym1 = np.where(y > 0, nbinom.cdf(y - 1.0, n, p), 0.0)
    denom = np.maximum(Fy - Fym1, 1e-12)
    fbar = [float(np.mean(np.clip((u - Fym1) / denom, 0.0, 1.0))) for u in grid]
    return list(np.asarray(grid, dtype=float)), fbar


def ece(n: np.ndarray, p: np.ndarray, y: np.ndarray, grid=PIT_GRID) -> float:
    """Calibration error = mean |F_bar(u) - u| over the PIT grid (proper discrete calibration)."""
    g, fbar = calibration_pit_curve(n, p, y, grid)
    return float(np.mean([abs(f - u) for f, u in zip(fbar, g)]))


def _cos_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    an = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    bn = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    return 1.0 - (an * bn).sum(-1)


# ---------------------------------------------------------------------------
# Eval harness — fixed (biosample, window-batch) units reused across metrics
# ---------------------------------------------------------------------------

def make_eval_units(data, n_units: int, batch_size: int, rng: np.random.Generator,
                    chrom: str = "chr21") -> List[Tuple[str, np.ndarray]]:
    """Fixed list of (biosample, window_indices), reused across all metrics for comparability."""
    pool = data.eval_windows(chrom)
    units = []
    for _ in range(n_units):
        bio = data.biosamples[rng.integers(len(data.biosamples))]
        wi = rng.choice(pool, size=min(batch_size, len(pool)), replace=False)
        units.append((bio, wi))
    return units


def _uniform_cond(fx: int, px: float, fy: int, py: float, A: int = 8):
    return (np.full(A, fx, np.int64), np.full(A, px, np.float32),
            np.full(A, fy, np.int64), np.full(A, py, np.float32))


@torch.no_grad()
def _gather_cell(model, data, units, fx, px, fy, py, device, *,
                 fam_ym=None, par_ym=None) -> Dict[str, np.ndarray]:
    """Pooled per-position head outputs + target over available positions for a fixed cell.

    Metadata covariate can be decoupled from the applied transform via fam_ym/par_ym (shuffle control).
    Returns dict of flat arrays: mu, n, p, eta, target (all 1-D, aligned).
    """
    A = 8
    fam_x, par_x, fam_y, par_y = _uniform_cond(fx, px, fy, py, A)
    kw = {}
    if fam_ym is not None:
        kw = dict(fam_ym=np.full(A, fam_ym), par_ym=np.full(A, par_ym, np.float32))
    mu, nn_, eta, tgt = [], [], [], []
    for bio, wi in units:
        b = data.make_batch(bio, wi, fam_x, par_x, fam_y, par_y, device, **kw)
        out = forward_full(model, b)
        m = b["avail"].unsqueeze(1).expand_as(out["mu"]).bool()
        mu.append(out["mu"][m].cpu().numpy()); nn_.append(out["n"][m].cpu().numpy())
        eta.append(out["eta"][m].cpu().numpy())
        tgt.append(b["y_target"][m].cpu().numpy())
    MU = np.concatenate(mu).astype(np.float64); N = np.concatenate(nn_).astype(np.float64)
    # Reconstruct p from the TRUE decoder mean (float64), NOT the model's out["p"] which is floored at
    # 1e-6 for NBNLL torch-stability -- that floor would truncate mu by up to ~10x on the power1.5 tail
    # (mu>n*1e6) and corrupt CRPS/ECE/quantiles asymmetrically across the h_y sweep. (review: model.py:168)
    P = np.clip(N / (N + MU), P_EPS, 1.0 - P_EPS)
    return dict(mu=MU, n=N, p=P, eta=np.concatenate(eta), target=np.concatenate(tgt))


def _subsample(idx_len: int, budget: int, rng) -> np.ndarray:
    if idx_len <= budget:
        return np.arange(idx_len)
    return rng.choice(idx_len, size=budget, replace=False)


# ---------------------------------------------------------------------------
# Reconstruction + M1
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_reconstruction(model, data, units, device, *, families=None, n_param=2,
                        budget=200_000, seed=0) -> Dict:
    """Pool predictions/targets across the trained family matrix; report the full metric suite."""
    rng = np.random.default_rng(seed)
    fams = families if families is not None else T.FAMILIES_2A
    fam_ids = [T.FAM[f] for f in fams]
    parts = []
    for fx in fam_ids:
        for fy in fam_ids:
            for px in _draw_params(fx, n_param, rng):
                for py in _draw_params(fy, n_param, rng):
                    parts.append(_gather_cell(model, data, units, fx, px, fy, py, device))
    mu = np.concatenate([g["mu"] for g in parts]); n = np.concatenate([g["n"] for g in parts])
    p = np.concatenate([g["p"] for g in parts]); tgt = np.concatenate([g["target"] for g in parts])
    sel = _subsample(len(mu), budget, rng)
    mu, n, p, tgt = mu[sel], n[sel], p[sel], tgt[sel]
    calib_grid, calib_fbar = calibration_pit_curve(n, p, tgt)
    return dict(
        crps=float(np.mean(nb_crps(n, p, tgt))),
        nll=_nb_nll_np(n, p, tgt),
        spearman=spearman(mu, tgt), pearson=pearson(np.log1p(mu), np.log1p(tgt)),
        ece=ece(n, p, tgt), r2=r2(mu, tgt), n_points=int(len(mu)),
        calib=dict(grid=calib_grid, fbar=calib_fbar))


def _nb_nll_np(n, p, y) -> float:
    p = np.clip(p, P_EPS, 1.0 - P_EPS)
    lp = nbinom.logpmf(np.maximum(y, 0), n, p)
    return float(-np.mean(lp))


def _draw_params(fid: int, k: int, rng) -> List[float]:
    grid = T.PARAMS[T.FAM_NAMES[fid]]
    if len(grid) <= k:
        return list(grid)
    return [float(grid[i]) for i in rng.choice(len(grid), size=k, replace=False)]


@torch.no_grad()
def eval_M1(model, data, units, device, *, families=None, n_param=2, budget=120_000, seed=0) -> Dict:
    """Per (f_x, f_y) cell reconstruction (CRPS + Spearman), + gap to the identity-cell ceiling."""
    rng = np.random.default_rng(seed)
    fams = families if families is not None else T.FAMILIES_2A
    fam_ids = [T.FAM[f] for f in fams]
    cell_crps: Dict[Tuple[int, int], float] = {}
    cell_spear: Dict[Tuple[int, int], float] = {}
    for fx in fam_ids:
        for fy in fam_ids:
            mus, ns, ps, tgts = [], [], [], []
            for px in _draw_params(fx, n_param, rng):
                for py in _draw_params(fy, n_param, rng):
                    g = _gather_cell(model, data, units, fx, px, fy, py, device)
                    mus.append(g["mu"]); ns.append(g["n"]); ps.append(g["p"]); tgts.append(g["target"])
            mu = np.concatenate(mus); n = np.concatenate(ns); p = np.concatenate(ps); tg = np.concatenate(tgts)
            sel = _subsample(len(mu), budget, rng)
            cell_crps[(fx, fy)] = float(np.mean(nb_crps(n[sel], p[sel], tg[sel])))
            cell_spear[(fx, fy)] = spearman(mu[sel], tg[sel])
    idc = T.FAM["identity"]
    ceil_crps = cell_crps[(idc, idc)]
    gap = {c: v - ceil_crps for c, v in cell_crps.items()}   # CRPS is a loss: gap = cell - ceiling (>=0)
    offdiag = [cell_crps[c] for c in cell_crps if c[0] != c[1]]
    return dict(cell_crps=cell_crps, cell_spearman=cell_spear, ceiling_crps=ceil_crps,
                gap=gap, median_gap=float(np.nanmedian(list(gap.values()))),
                median_offdiag_crps=float(np.nanmedian(offdiag)),
                median_offdiag_spearman=float(np.nanmedian(
                    [cell_spear[c] for c in cell_spear if c[0] != c[1]])))


# ---------------------------------------------------------------------------
# M2 — distributional output steering
# ---------------------------------------------------------------------------

def _avail_assays(data, units):
    return sorted({int(a) for bio, _ in units
                   for a in np.where(data.avail[bio].astype(bool))[0]})


@torch.no_grad()
def _gather_assay(model, data, units, a, device, *, fam_ym, par_ym, applied_fy, applied_py,
                  applied_fx=None, applied_px=1.0):
    """Pool TARGET-ASSAY a's per-position (mu, n, p, eta, target) over units, holding every OTHER
    available assay at identity. Only assay a carries a condition -- this is the per-assay steering
    readout that reveals the pooling deficit (a pooled decoder averages assay a's condition with the
    identity others -> cannot steer assay a; a per-assay decoder can).

    The metadata covariate for assay a = (fam_ym, par_ym); the APPLIED output transform = (applied_fy,
    applied_py), decoupled so predictions use identity-applied output while targets use the real transform.
    The INPUT to assay a is transformed by (applied_fx, applied_px) [default identity] -- non-identity
    values feed the full-matrix M2 (dual-conditioning-under-load: steer the output while the encoder must
    ALSO undo a lossy input). make_batch propagates fam_x[a] into x_meta, so the encoder is TOLD the input
    transform. The TARGET is fy(true base), independent of applied_fx (make_batch applies f_y to raw counts).
    p is reconstructed from the true mu (float64), NOT the model's loss-floored p (review: model.py:168)."""
    idc = T.FAM["identity"]
    afx = idc if applied_fx is None else int(applied_fx)
    fam_x = np.full(A_ASSAYS, idc); par_x = np.ones(A_ASSAYS, np.float32)
    fam_x[a] = afx; par_x[a] = float(applied_px)
    fam_y = np.full(A_ASSAYS, idc); par_y = np.ones(A_ASSAYS, np.float32)
    fam_y[a] = applied_fy; par_y[a] = applied_py
    fam_ym_v = np.full(A_ASSAYS, idc); par_ym_v = np.ones(A_ASSAYS, np.float32)
    fam_ym_v[a] = fam_ym; par_ym_v[a] = par_ym
    mu, nn_, eta, tg = [], [], [], []
    for bio, wi in units:
        b = data.make_batch(bio, wi, fam_x, par_x, fam_y, par_y, device, fam_ym=fam_ym_v, par_ym=par_ym_v)
        out = forward_full(model, b)
        av = b["avail"][:, a].bool()                       # [B] rows where assay a is available
        if not bool(av.any()):
            continue
        mu.append(out["mu"][:, :, a][av].reshape(-1).cpu().numpy())
        nn_.append(out["n"][:, :, a][av].reshape(-1).cpu().numpy())
        eta.append(out["eta"][:, :, a][av].reshape(-1).cpu().numpy())
        tg.append(b["y_target"][:, :, a][av].reshape(-1).cpu().numpy())
    if not mu:
        return None
    MU = np.concatenate(mu).astype(np.float64); N = np.concatenate(nn_).astype(np.float64)
    P = np.clip(N / (N + MU), P_EPS, 1.0 - P_EPS)
    return dict(mu=MU, n=N, p=P, eta=np.concatenate(eta), target=np.concatenate(tg))


@torch.no_grad()
def _assay_sweep(model, data, units, a, fy, device, *, budget=40_000, seed=0,
                 applied_fx=None, applied_px=1.0):
    """Per-assay steering sweep for target assay a, family fy, with the input transformed by
    (applied_fx, applied_px) [default identity]. Returns position-aligned:
      ref       = identity-output prediction+target for assay a (base = ref['target'])
      preds[j]  = prediction when assay a is TOLD (fy, param_j) [output applied identity]
      targets[i]= counts the transform (fy, param_i) dictates for assay a (from the true base)
    All gathered from the SAME (bio, window, assay a position) order (availability is independent of the
    applied transform), so C[i,j] pairs identical positions even when applied_fx != identity."""
    rng = np.random.default_rng(seed); idc = T.FAM["identity"]
    params = list(T.PARAMS[T.FAM_NAMES[fy]])
    ref = _gather_assay(model, data, units, a, device, fam_ym=idc, par_ym=1.0, applied_fy=idc, applied_py=1.0,
                        applied_fx=applied_fx, applied_px=applied_px)
    if ref is None:
        return None
    n0 = len(ref["target"])
    sel = np.arange(n0) if n0 <= budget else rng.choice(n0, budget, replace=False)
    ref = {k: v[sel] for k, v in ref.items()}
    preds = [{k: v[sel] for k, v in _gather_assay(model, data, units, a, device,
              fam_ym=fy, par_ym=pj, applied_fy=idc, applied_py=1.0,
              applied_fx=applied_fx, applied_px=applied_px).items()} for pj in params]
    targets = [_gather_assay(model, data, units, a, device,
               fam_ym=fy, par_ym=pi, applied_fy=fy, applied_py=pi,
               applied_fx=applied_fx, applied_px=applied_px)["target"][sel] for pi in params]
    return dict(params=params, ref=ref, preds=preds, targets=targets, base=ref["target"])


def _steering_index(crps_mat: np.ndarray) -> float:
    """Relative CRPS reduction at matched h_y: mean_i (rowmean_i - C[i,i]) / rowmean_i.

    0 for an h_y-ignoring model (prediction independent of the metadata column j), -> 1 for perfect
    steering (matched prediction dominates). crps_mat[i,j] = CRPS(pred_j, target_i).
    """
    P = crps_mat.shape[0]
    diag = np.diag(crps_mat)
    rowmean = crps_mat.mean(axis=1)
    good = rowmean > 1e-9
    if not np.any(good):
        return float("nan")
    return float(np.mean((rowmean[good] - diag[good]) / rowmean[good]))


@torch.no_grad()
def eval_M2(model, data, units, device, *, families=None, seed=0, tail_q=0.95) -> Dict:
    """PER-ASSAY distributional M2 per family: for each target assay a (others held identity), the
    CRPS-response steering index + mean/tail decomposition; per_family = median over assays. This is the
    primary readout because per-assay conditioning is the v2 fix -- a uniform (all-assays) sweep cannot
    separate the per-assay decoder from the v1 pooled decoder (both respond to a shared swept value),
    whereas the per-assay sweep does (per-assay ~0.5 vs pooled ~0.02 = the v1 null)."""
    fams = families if families is not None else T.FAMILIES_2A
    idc = T.FAM["identity"]
    assays = _avail_assays(data, units)
    per_family, mean_stat, tail_stat, crps_matrix = {}, {}, {}, {}
    for fname in fams:
        fy = T.FAM[fname]
        if fy == idc:
            continue
        params = T.PARAMS[fname]; P = len(params)
        scores, d_eta, d_tgt, dpq, dtq, Cmats = [], [], [], [], [], []
        for a in assays:
            sw = _assay_sweep(model, data, units, a, fy, device, seed=seed)
            if sw is None:
                continue
            preds, targets, ref = sw["preds"], sw["targets"], sw["ref"]
            C = np.zeros((P, P))
            for i in range(P):
                for j in range(P):
                    C[i, j] = float(np.mean(nb_crps(preds[j]["n"], preds[j]["p"], targets[i])))
            scores.append(_steering_index(C)); Cmats.append(C)
            ref_pq = float(np.mean(nb_quantile(tail_q, ref["n"], ref["p"])))
            ref_tq = float(np.quantile(ref["target"], tail_q))
            for i in range(P):
                d_eta.append(preds[i]["eta"] - ref["eta"])
                d_tgt.append(np.log2(targets[i] + 1.0) - np.log2(ref["target"] + 1.0))
                dpq.append(np.log2(float(np.mean(nb_quantile(tail_q, preds[i]["n"], preds[i]["p"]))) + 1.0) - np.log2(ref_pq + 1.0))
                dtq.append(np.log2(float(np.quantile(targets[i], tail_q)) + 1.0) - np.log2(ref_tq + 1.0))
        per_family[fy] = float(np.nanmedian(scores)) if scores else float("nan")
        if d_eta:
            de = np.concatenate(d_eta); dt = np.concatenate(d_tgt)
            mean_stat[fy] = dict(r2=r2(de, dt), pearson=pearson(de, dt), spearman=spearman(de, dt))
            tail_stat[fy] = dict(pred_dq=dpq, target_dq=dtq, pearson=pearson(np.array(dpq), np.array(dtq)))
        if Cmats:
            # median-over-assays CRPS-response matrix C[i(true h_y), j(told h_y)] for F2 (min at j==i)
            crps_matrix[fy] = dict(params=list(params), C=np.median(np.stack(Cmats), axis=0).tolist())
    inv_ids = T.invertible_family_ids()
    inv = [per_family[f] for f in per_family if f in inv_ids and np.isfinite(per_family[f])]
    return dict(per_family=per_family, mean_stat=mean_stat, tail_stat=tail_stat, crps_matrix=crps_matrix,
                median=float(np.nanmedian([v for v in per_family.values() if np.isfinite(v)])) if per_family else float("nan"),
                median_invertible=float(np.nanmedian(inv)) if inv else float("nan"))


def _fx_canonical_param(fname: str) -> float:
    """Representative input param per f_x family for the M2 matrix (mid of the grid; identity -> 1.0).
    One px per f_x keeps the matrix eval ~|families|x the identity-only M2 rather than an extra |params|x."""
    grid = T.PARAMS[fname]
    return float(grid[len(grid) // 2])


@torch.no_grad()
def eval_M2_matrix(model, data, units, device, *, families=None, seed=0, identity_row=None) -> Dict:
    """Full f_x x f_y output-steering matrix: the per-assay CRPS-response steering index for every
    (f_x, f_y) cell, with the INPUT transformed by f_x (canonical mid param) while h_y sweeps f_y.

    Off-identity rows are dual-conditioning-under-load (steer the output while the encoder must also undo a
    lossy input) -- the q15 cell that identity-input M2 never tested. Persisted per-cell (median over assays)
    AND per-assay un-collapsed (persistence convention).

    COST TRIM: pass `identity_row` = eval_M2's per_family ({fy_id: steering}); the f_x=identity row is then
    COPIED from it EXACTLY (it IS the classic M2) instead of recomputed, so `units` here can be a small
    subset for the costly lossy-f_x rows only, with no loss on the reference row. Without it, the identity
    row is computed on `units` (and equals per_family only if `units` is the full set)."""
    fams = families if families is not None else T.FAMILIES_2A
    idc = T.FAM["identity"]
    assays = _avail_assays(data, units)
    steering: Dict[Tuple[int, int], float] = {}
    per_assay: Dict[Tuple[int, int], list] = {}
    fx_param: Dict[int, float] = {}
    for fxname in fams:
        fx = T.FAM[fxname]; px = _fx_canonical_param(fxname); fx_param[fx] = px
        for fyname in fams:
            fy = T.FAM[fyname]
            if fy == idc:
                continue
            if fx == idc and identity_row is not None and fy in identity_row:
                steering[(fx, fy)] = float(identity_row[fy]); per_assay[(fx, fy)] = []
                continue
            P = len(T.PARAMS[fyname])
            scores = []
            for a in assays:
                sw = _assay_sweep(model, data, units, a, fy, device, seed=seed,
                                  applied_fx=fx, applied_px=px)
                if sw is None:
                    continue
                preds, targets = sw["preds"], sw["targets"]
                C = np.zeros((P, P))
                for i in range(P):
                    for j in range(P):
                        C[i, j] = float(np.mean(nb_crps(preds[j]["n"], preds[j]["p"], targets[i])))
                scores.append(_steering_index(C))
            steering[(fx, fy)] = float(np.nanmedian(scores)) if scores else float("nan")
            per_assay[(fx, fy)] = [float(s) for s in scores]
    return dict(steering_matrix=steering, per_assay=per_assay, fx_param=fx_param)


# ---------------------------------------------------------------------------
# M3 — encoder input invariance
# ---------------------------------------------------------------------------

@torch.no_grad()
def _latent(model, data, bio, wi, fx, px, device) -> np.ndarray:
    fam_x, par_x, fam_y, par_y = _uniform_cond(fx, px, T.FAM["identity"], 1.0)
    b = data.make_batch(bio, wi, fam_x, par_x, fam_y, par_y, device)
    return encode_latent(model, b).mean(dim=1).cpu().numpy()      # [B, d]


@torch.no_grad()
def eval_M3(model, data, units, device, *, families=None, seed=0) -> Dict:
    """within-base transform cos-dist vs between-base separation. ratio << 1 = invariant + discriminative."""
    rng = np.random.default_rng(seed)
    fams = families if families is not None else T.FAMILIES_2A
    within, per_family, z_id_all = [], {}, []
    for bio, wi in units:
        z_id = _latent(model, data, bio, wi, T.FAM["identity"], 1.0, device)
        z_id_all.append(z_id)
        for fname in fams:
            fx = T.FAM[fname]
            if fx == T.FAM["identity"]:
                continue
            px = float(rng.choice(T.PARAMS[fname]))
            z_aug = _latent(model, data, bio, wi, fx, px, device)
            wd = _cos_dist(z_aug, z_id)
            within.append(wd); per_family.setdefault(fx, []).append(wd)
    within = np.concatenate(within)
    Z = np.concatenate(z_id_all)
    N = len(Z)
    idx = rng.integers(0, N, size=(min(4000, N * 4), 2)); idx = idx[idx[:, 0] != idx[:, 1]]
    between = _cos_dist(Z[idx[:, 0]], Z[idx[:, 1]])
    wmean, bmean = float(within.mean()), float(between.mean())
    fam_ratio = {f: float(np.concatenate(v).mean()) / (bmean + 1e-8) for f, v in per_family.items()}
    return dict(within=wmean, between=bmean, ratio=wmean / (bmean + 1e-8), per_family_ratio=fam_ratio)


# ---------------------------------------------------------------------------
# h37 — foreground vs aggregate M2 (background-domination diagnostic)
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_foreground_M2(model, data, units, device, *, families=None, fg_frac=0.02, seed=0) -> Dict:
    """Per-assay M2 steering index restricted to FOREGROUND positions (top fg_frac by BASE count within
    the target assay) vs aggregate. A foreground-localized steering signal (cap/thin/power/mult) shows
    M2_fg >> M2_agg; `add` (background-visible) shows a small gap. per_family = median over assays."""
    fams = families if families is not None else T.FAMILIES_2A
    idc = T.FAM["identity"]
    assays = _avail_assays(data, units)
    out = {}
    for fname in fams:
        fy = T.FAM[fname]
        if fy == idc:
            continue
        params = T.PARAMS[fname]; P = len(params); aggs, fgs = [], []
        for a in assays:
            sw = _assay_sweep(model, data, units, a, fy, device, seed=seed)
            if sw is None:
                continue
            preds, targets, base = sw["preds"], sw["targets"], sw["base"]
            thr = np.quantile(base, 1.0 - fg_frac); fg = base >= thr

            def _index(mask):
                C = np.zeros((P, P))
                for i in range(P):
                    for j in range(P):
                        C[i, j] = float(np.mean(nb_crps(preds[j]["n"][mask], preds[j]["p"][mask], targets[i][mask])))
                return _steering_index(C)
            aggs.append(_index(np.ones_like(fg, bool)))
            if fg.sum() > 10:
                fgs.append(_index(fg))
        m2_agg = float(np.nanmedian(aggs)) if aggs else float("nan")
        m2_fg = float(np.nanmedian(fgs)) if fgs else float("nan")
        out[fy] = dict(agg=m2_agg, fg=m2_fg,
                       gap=(m2_fg - m2_agg) if (np.isfinite(m2_fg) and np.isfinite(m2_agg)) else float("nan"))
    return out


# ---------------------------------------------------------------------------
# h35 — shuffle-h_y reliance (shortcut test, normal regime)
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_shuffle_reliance(model, data, units, device, *, families=None, seed=0, budget=40_000) -> Dict:
    """Per-assay h_y-reliance (h35 shortcut test): CRPS degradation when target assay a is TOLD a WRONG
    h_y (identity) while the applied transform is the true one. High reliance = the model genuinely uses
    h_y, not the input denoising shortcut. Also reports input-target approximability (mean |base - target|)
    for the dose-response scatter: reliance should be low where the input already approximates the target."""
    rng = np.random.default_rng(seed)
    fams = families if families is not None else T.FAMILIES_2A
    idc = T.FAM["identity"]
    assays = _avail_assays(data, units)
    out = {}
    for fname in fams:
        fy = T.FAM[fname]
        if fy == idc:
            continue
        rel, approx = [], []
        for a in assays:
            base_g = _gather_assay(model, data, units, a, device, fam_ym=idc, par_ym=1.0,
                                   applied_fy=idc, applied_py=1.0)
            if base_g is None:
                continue
            for pj in T.PARAMS[fname]:
                true_g = _gather_assay(model, data, units, a, device, fam_ym=fy, par_ym=pj,
                                       applied_fy=fy, applied_py=pj)                 # correct meta
                n0 = len(true_g["target"]); sel = np.arange(n0) if n0 <= budget else rng.choice(n0, budget, replace=False)
                tgt = true_g["target"][sel]
                crps_true = float(np.mean(nb_crps(true_g["n"][sel], true_g["p"][sel], tgt)))
                wrong_g = _gather_assay(model, data, units, a, device, fam_ym=idc, par_ym=1.0,
                                        applied_fy=fy, applied_py=pj)                # WRONG (identity) meta
                crps_wrong = float(np.mean(nb_crps(wrong_g["n"][sel], wrong_g["p"][sel], tgt)))
                rel.append(crps_wrong - crps_true)
                approx.append(float(np.mean(np.abs(base_g["target"][sel] - tgt))))
        out[fy] = dict(reliance=float(np.mean(rel)) if rel else float("nan"),
                       approx_gap=float(np.mean(approx)) if approx else float("nan"))
    return out


# ---------------------------------------------------------------------------
# h31 — memorization baseline (does the model read h_y or fall back on a seen pairing?)
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_memorization_baseline(model, data, units, device, *, heldout, families=None, seed=0,
                               budget=40_000) -> Dict:
    """h31: at each HELD-OUT cell (f_x -> f_y), told to produce f_y, does the model predict the correct
    f_y(base) or fall back on a seen-but-wrong f_y' it HAD been paired with f_x? Compare, position-aligned,
    CRPS(pred | told f_y ; input f_x) against the CORRECT target f_y(base) vs a WRONG seen target f_y'(base).
    `beats` = correct is closer. f_y' = the first non-identity family trained with f_x (not in `heldout`,
    != f_y). Canonical mid params throughout; median over assays. Runs only in holdout runs."""
    fams = families if families is not None else T.FAMILIES_2A
    idc = T.FAM["identity"]
    fam_ids = [T.FAM[f] for f in fams]
    nonid = [f for f in fam_ids if f != idc]
    held = {(int(c[0]), int(c[1])) for c in heldout}
    assays = _avail_assays(data, units)
    rng = np.random.default_rng(seed)
    cells: Dict[Tuple[int, int], dict] = {}
    for (fx, fy) in sorted(held):
        seen_alts = [g for g in nonid if g != fy and (fx, g) not in held]   # a SEEN output family for f_x
        if not seen_alts:
            continue
        fyp = seen_alts[0]
        px = _fx_canonical_param(T.FAM_NAMES[fx])
        py = _fx_canonical_param(T.FAM_NAMES[fy]); pyp = _fx_canonical_param(T.FAM_NAMES[fyp])
        cc, cw = [], []
        for a in assays:
            pred = _gather_assay(model, data, units, a, device, fam_ym=fy, par_ym=py,
                                 applied_fy=idc, applied_py=1.0, applied_fx=fx, applied_px=px)
            if pred is None:
                continue
            tgt_c = _gather_assay(model, data, units, a, device, fam_ym=fy, par_ym=py,
                                  applied_fy=fy, applied_py=py, applied_fx=fx, applied_px=px)["target"]
            tgt_w = _gather_assay(model, data, units, a, device, fam_ym=fy, par_ym=py,
                                  applied_fy=fyp, applied_py=pyp, applied_fx=fx, applied_px=px)["target"]
            n0 = len(pred["target"]); sel = np.arange(n0) if n0 <= budget else rng.choice(n0, budget, replace=False)
            cc.append(float(np.mean(nb_crps(pred["n"][sel], pred["p"][sel], tgt_c[sel]))))
            cw.append(float(np.mean(nb_crps(pred["n"][sel], pred["p"][sel], tgt_w[sel]))))
        if not cc:
            continue
        crps_c, crps_w = float(np.median(cc)), float(np.median(cw))
        cells[(fx, fy)] = dict(fyp=int(fyp), crps_correct=crps_c, crps_wrong=crps_w,
                               beats=bool(crps_c < crps_w))
    beats = [v["beats"] for v in cells.values()]
    return dict(cells=cells, frac_beats=float(np.mean(beats)) if beats else float("nan"),
                n_cells=len(cells))

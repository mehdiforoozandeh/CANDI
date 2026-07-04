"""Metrics for the dual-conditioning testbed (crux q15).

M1 - end-to-end reconstruction, per (f_x, f_y) family cell, R2 of NB-mean vs target; headline is the
     gap to the identity-cell ceiling (decouples conditioning from the lossy bottleneck).
M2 - decoder/output steering: fix input, sweep h_y over a family; R2(delta_pred, delta_target).
     An h_y-ignoring model gives delta_pred~0 -> R2<=0.
M3 - encoder/input invariance: within-base latent cos-dist (across f_x) vs between-base; want << 1.

All eval reuses a fixed set of (biosample, window-batch) units so cells/arms are comparable.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
import torch

from sandbox.diagnostics.dual_conditioning import transforms as T
from sandbox.diagnostics.dual_conditioning.model import nb_mean, forward_counts, encode_latent


def r2(pred: np.ndarray, target: np.ndarray) -> float:
    if len(pred) < 2:
        return float("nan")
    ss_res = float(((pred - target) ** 2).sum())
    ss_tot = float(((target - target.mean()) ** 2).sum())
    return float("nan") if ss_tot <= 1e-12 else 1.0 - ss_res / ss_tot


def make_eval_units(data, n_units: int, batch_size: int, rng: np.random.Generator,
                    chrom: str = "chr21") -> List[Tuple[str, np.ndarray]]:
    """Fixed list of (biosample, window_indices), reused across all metrics for comparability."""
    pool = data.eval_windows(chrom)
    units = []
    for _ in range(n_units):
        bio = data.biosamples[rng.integers(len(data.biosamples))]
        wi = rng.choice(pool, size=batch_size, replace=False)
        units.append((bio, wi))
    return units


def _uniform_cond(fx: int, px: float, fy: int, py: float):
    a = 8
    return (np.full(a, fx, np.int64), np.full(a, px, np.float32),
            np.full(a, fy, np.int64), np.full(a, py, np.float32))


@torch.no_grad()
def _cell_preds(model, data, units, fx, px, fy, py, device) -> Tuple[np.ndarray, np.ndarray]:
    """Pooled (nb_mean, target) over available positions for a fixed (fx,px,fy,py) cell."""
    fam_x, par_x, fam_y, par_y = _uniform_cond(fx, px, fy, py)
    preds, tgts = [], []
    for bio, wi in units:
        b = data.make_batch(bio, wi, fam_x, par_x, fam_y, par_y, device)
        p, n = forward_counts(model, b)
        mean = nb_mean(p, n)
        m = b["avail"].unsqueeze(1).expand_as(mean).bool()
        preds.append(mean[m].cpu().numpy())
        tgts.append(b["y_target"][m].cpu().numpy())
    return np.concatenate(preds), np.concatenate(tgts)


def _params(fid: int, k: int, rng) -> List[float]:
    grid = T.PARAMS[T.FAM_NAMES[fid]]
    if len(grid) <= k:
        return list(grid)
    return [float(grid[i]) for i in rng.choice(len(grid), size=k, replace=False)]


@torch.no_grad()
def eval_M1(model, data, units, device, *, n_param=2, seed=0) -> Dict:
    """R2 per family cell (pooled over n_param x n_param param draws), + gap to identity ceiling."""
    rng = np.random.default_rng(seed)
    fam_ids = list(range(T.N_FAMILIES))
    cell_r2: Dict[Tuple[int, int], float] = {}
    for fx in fam_ids:
        for fy in fam_ids:
            preds, tgts = [], []
            for px in _params(fx, n_param, rng):
                for py in _params(fy, n_param, rng):
                    pr, tg = _cell_preds(model, data, units, fx, px, fy, py, device)
                    preds.append(pr); tgts.append(tg)
            cell_r2[(fx, fy)] = r2(np.concatenate(preds), np.concatenate(tgts))
    ceiling = cell_r2[(T.FAM["identity"], T.FAM["identity"])]
    gaps = {c: ceiling - v for c, v in cell_r2.items()}
    offdiag = [cell_r2[c] for c in cell_r2 if c[0] != c[1]]
    return dict(cell_r2=cell_r2, ceiling=ceiling, gap=gaps,
                median_r2=float(np.nanmedian(list(cell_r2.values()))),
                median_offdiag_r2=float(np.nanmedian(offdiag)),
                median_gap=float(np.nanmedian(list(gaps.values()))))


@torch.no_grad()
def eval_M2(model, data, units, device, *, in_family=None, seed=0) -> Dict:
    """Output-steering per family: fix input (identity unless given), sweep h_y; R2(dpred, dtarget)."""
    rng = np.random.default_rng(seed)
    idc = T.FAM["identity"]
    fx = idc if in_family is None else in_family
    px = 1.0
    per_family = {}
    for fy in range(T.N_FAMILIES):
        if fy == idc:
            continue
        ref_pred, ref_tgt = _cell_preds(model, data, units, fx, px, idc, 1.0, device)  # identity output
        dpred, dtarget = [], []
        for py in T.PARAMS[T.FAM_NAMES[fy]]:
            pr, tg = _cell_preds(model, data, units, fx, px, fy, py, device)
            dpred.append(pr - ref_pred)
            dtarget.append(tg - ref_tgt)
        per_family[fy] = r2(np.concatenate(dpred), np.concatenate(dtarget))
    inv = [per_family[f] for f in per_family if T.FAM_NAMES[f] in T.INVERTIBLE]
    return dict(per_family=per_family,
                median=float(np.nanmedian(list(per_family.values()))),
                median_invertible=float(np.nanmedian(inv)) if inv else float("nan"))


@torch.no_grad()
def _latent(model, data, bio, wi, fx, px, device) -> np.ndarray:
    """Pooled encoder latent [B, d] for input transform (fx,px), identity output metadata."""
    fam_x, par_x, fam_y, par_y = _uniform_cond(fx, px, T.FAM["identity"], 1.0)
    b = data.make_batch(bio, wi, fam_x, par_x, fam_y, par_y, device)
    z = encode_latent(model, b)              # [B, L2, d]
    return z.mean(dim=1).cpu().numpy()       # [B, d]


def _cos_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    an = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    bn = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    return 1.0 - (an * bn).sum(-1)


@torch.no_grad()
def eval_M3(model, data, units, device, *, seed=0) -> Dict:
    """within-base transform cos-dist vs between-base separation. ratio << 1 = invariant + discriminative."""
    rng = np.random.default_rng(seed)
    within, per_family = [], {}
    z_id_all = []
    for bio, wi in units:
        z_id = _latent(model, data, bio, wi, T.FAM["identity"], 1.0, device)   # [B, d]
        z_id_all.append(z_id)
        for fx in range(T.N_FAMILIES):
            if fx == T.FAM["identity"]:
                continue
            px = float(rng.choice(T.PARAMS[T.FAM_NAMES[fx]]))
            z_aug = _latent(model, data, bio, wi, fx, px, device)
            wd = _cos_dist(z_aug, z_id)                                        # [B]
            within.append(wd)
            per_family.setdefault(fx, []).append(wd)
    within = np.concatenate(within)
    Z = np.concatenate(z_id_all)                                              # [N, d]
    # between-base: cos-dist across distinct base windows (sample pairs)
    N = len(Z); idx = rng.integers(0, N, size=(min(4000, N * 4), 2))
    idx = idx[idx[:, 0] != idx[:, 1]]
    between = _cos_dist(Z[idx[:, 0]], Z[idx[:, 1]])
    wmean, bmean = float(within.mean()), float(between.mean())
    fam_ratio = {f: float(np.concatenate(v).mean()) / (bmean + 1e-8) for f, v in per_family.items()}
    return dict(within=wmean, between=bmean, ratio=wmean / (bmean + 1e-8),
                per_family_ratio=fam_ratio)

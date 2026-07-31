"""Numeric metric primitives for candi_kit (NB CRPS, quantiles, PIT calibration, correlations).

VENDORED VERBATIM of sandbox/diagnostics/dual_conditioning/metrics.py
  L27 (P_EPS), L29-131 (_nb_cdf, nb_crps, nb_quantile, r2, pearson, spearman, PIT_GRID,
  calibration_pit_curve, ece, _cos_dist), L341-353 (_steering_index).
The synthetic-transform-bound eval harness (make_eval_units onward) is NOT shipped; the real-data
measurement stack lives in candi_kit.eval.

The closed-form NB CRPS (`nb_crps`) is the risky numeric: derived as E|X-y| - 1/2 E|X-X'| with a
Pfaff-transformed hypergeometric Gini term, verified bit-close to an exact discrete sum and to
Monte-Carlo across the full parameter range (tests/test_metrics_primitives.py).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import nbinom, rankdata
from scipy.special import hyp2f1

P_EPS = 1e-9


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

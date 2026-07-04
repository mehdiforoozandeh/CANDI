"""CANDI v3 — FROZEN metric math for the comprehensive ERA_SCORE aspects.

Pure functions over arrays pooled on the chr21 eval set, decoupled from how a candidate
produces predictions (baseline vs neural). See METRIC.md for the score that combines these.

Reconstruction quality (imputation on V/B held-out assays; denoising on T_ assays low->high
depth) = uniform mean of 4 correlations each: {pval,count} x {Spearman,Pearson}, all in [-1,1]:
  Q_imp -> S_A = Q_imp - Q_imp_baseline  (PRIMARY)
  Q_den -> denoising>=imputation gate (raw)
Plus calibration (ECE of the count dist), C-index (distributional ranking), peak AUROC, DCR.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr


@dataclass
class ReconArrays:
    """1-D arrays pooled over reconstruction positions (imputation or denoising)."""
    signal_pred: np.ndarray   # model enrichment prediction (vs arcsinh-pval)
    gt_pval: np.ndarray       # GT arcsinh -log10 pval
    count_mean: np.ndarray    # predicted E[counts]
    gt_counts: np.ndarray     # GT raw counts


@dataclass
class CalArrays:
    """Calibration inputs (imputation positions only)."""
    pit: np.ndarray                    # PIT = F_count(gt) in [0,1], for ECE
    cidx_samples: np.ndarray           # [S, n] count samples on a subsample, for C-index
    cidx_gt: np.ndarray                # [n] true counts for that subsample
    peak_prob: np.ndarray              # predicted peak probability
    peak_gt: np.ndarray                # GT peaks in {0,1}


def _spear(a, b):
    if a.size < 4:
        return float("nan")
    r, _ = spearmanr(a, b)
    return float(r) if math.isfinite(r) else float("nan")


def _pear(a, b):
    if a.size < 4 or a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    am, bm = a - a.mean(), b - b.mean()
    return float((am * bm).sum() / (np.linalg.norm(am) * np.linalg.norm(bm) + 1e-12))


def recon_corrs(r: ReconArrays) -> dict:
    """The 4 correlations + their uniform mean Q."""
    ps, pp = _spear(r.signal_pred, r.gt_pval), _pear(r.signal_pred, r.gt_pval)
    cs, cp = _spear(r.count_mean, r.gt_counts), _pear(r.count_mean, r.gt_counts)
    vals = [v for v in (ps, pp, cs, cp) if math.isfinite(v)]
    Q = float(np.mean(vals)) if vals else float("nan")
    return {"pval_spearman": ps, "pval_pearson": pp,
            "count_spearman": cs, "count_pearson": cp, "Q": Q}


def ece_from_pit(pit: np.ndarray, n_levels: int = 19) -> float:
    """CI-coverage calibration error from PIT values (0 = perfectly calibrated)."""
    if pit.size < 4:
        return float("nan")
    centered = np.abs(2.0 * pit - 1.0)
    levels = np.linspace(0.05, 0.95, n_levels)
    cov = np.array([(centered <= c).mean() for c in levels])
    return float(np.abs(cov - levels).mean())


def c_index_from_samples(samples: np.ndarray, gt: np.ndarray, n_pairs: int = 20000,
                         seed: int = 0) -> float:
    """Distributional concordance: over random position pairs (i,j) with gt_i != gt_j,
    P(Y_i > Y_j) estimated from the predictive samples, scored against the true ordering.
    Uncertainty-aware (a confident-correct pair scores near 1; an unsure one near 0.5).
    0.5 = random, 1 = perfect. samples [S,n], gt [n]."""
    n = gt.shape[0]
    if n < 4 or samples.ndim != 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, n_pairs); j = rng.integers(0, n, n_pairs)
    keep = gt[i] != gt[j]
    i, j = i[keep], j[keep]
    if i.size < 4:
        return float("nan")
    si, sj = samples[:, i], samples[:, j]                      # [S, P]
    p_igtj = (si > sj).mean(axis=0) + 0.5 * (si == sj).mean(axis=0)   # P(Y_i>Y_j)
    conc = np.where(gt[i] > gt[j], p_igtj, 1.0 - p_igtj)
    return float(conc.mean())


def auroc(pred: np.ndarray, gt: np.ndarray) -> float:
    """AUROC for binary peaks; NaN if a class is absent."""
    from sklearn.metrics import roc_auc_score
    m = (gt == 0) | (gt == 1)
    p, t = pred[m], gt[m].astype(int)
    if t.size < 4 or t.min() == t.max():
        return float("nan")
    try:
        return float(roc_auc_score(t, p))
    except Exception:
        return float("nan")


def compute_aspects(imp: ReconArrays, den: ReconArrays, cal: CalArrays, dcr: float) -> dict:
    ci = recon_corrs(imp)
    cd = recon_corrs(den)
    out = {
        "Q_imp": ci["Q"], "Q_den": cd["Q"], "dcr": float(dcr),
        "ece": ece_from_pit(cal.pit),
        "c_index": c_index_from_samples(cal.cidx_samples, cal.cidx_gt),
        "peak_auroc": auroc(cal.peak_prob, cal.peak_gt),
        "n_imp": int(imp.signal_pred.size),
    }
    out.update({f"imp_{k}": v for k, v in ci.items() if k != "Q"})
    out.update({f"den_{k}": v for k, v in cd.items() if k != "Q"})
    return out

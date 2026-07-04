"""CANDI v3 — naive imputation baseline: per-position average-reference track.

Predict a held-out V_/B_ assay as the per-position mean of THAT assay over the training
(T_) biosamples that have it (classic ChromImpute/Avocado average-reference baseline). This
anchors the S_A zero-point and the metric scales/floors (PLAN §0 row 5, METRIC.md).

`average_reference` is biosample-independent (same track imputed everywhere), so its skill
comes purely from constitutive/position structure — an honest floor a real imputer must beat.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_v3 import EvalData, F, load_eval  # noqa: E402


def _average(evd: EvalData, attr: str) -> np.ndarray:
    """[W,L,F] mean of `attr` per (window,pos,assay) over T_ biosamples that have the assay."""
    W, L = evd.win_idx.shape[0], evd.biosamples[0].t_pval.shape[1]
    acc = np.zeros((W, L, F), dtype=np.float64)
    cnt = np.zeros(F, dtype=np.float64)
    for bs in evd.biosamples:
        track = getattr(bs, attr)
        for a in np.where(bs.t_avail)[0]:
            acc[:, :, a] += track[:, :, a]
            cnt[a] += 1
    cnt[cnt == 0] = 1.0
    return (acc / cnt[None, None, :]).astype(np.float32)


def average_reference(evd: EvalData) -> np.ndarray:
    """[W,L,F] mean arcsinh-pval (the imputation signal prediction)."""
    return _average(evd, "t_pval")


def _fit_nb_per_assay(evd: EvalData):
    """Method-of-moments NB (n,p) per assay over all T_ counts (for calibration PIT)."""
    n = np.zeros(F); p = np.full(F, 0.5)
    for a in range(F):
        vals = [bs.t_counts[:, :, a].ravel() for bs in evd.biosamples if bs.t_avail[a]]
        if not vals:
            continue
        x = np.concatenate(vals).astype(np.float64)
        m, v = float(x.mean()), float(x.var())
        v = max(v, m * 1.001 + 1e-6)            # enforce mild overdispersion
        p[a] = m / v
        n[a] = m * m / (v - m)
    return n, p


def baseline_arrays(evd: EvalData, max_points: int = 1_000_000, cidx_cap: int = 20_000,
                    n_samples: int = 64, seed: int = 0):
    """Average-reference baseline as (imp ReconArrays, CalArrays). Signal/count/peak predictions
    are the per-position mean over T_ biosamples; count distribution is the per-assay MoM NB."""
    from scipy.stats import nbinom
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from eval_v3 import ReconArrays, CalArrays  # noqa: E402

    avg_sig = _average(evd, "t_pval"); avg_cnt = _average(evd, "t_counts")
    avg_pk = _average(evd, "t_peaks")
    nb_n, nb_p = _fit_nb_per_assay(evd)

    sig, gtp, cmean, gtc, ppr, pgt, assay = [], [], [], [], [], [], []
    for bs in evd.biosamples:
        for a in np.where(bs.imp_avail)[0]:
            sig.append(avg_sig[:, :, a].ravel()); gtp.append(bs.imp_pval[:, :, a].ravel())
            cmean.append(avg_cnt[:, :, a].ravel()); gtc.append(bs.imp_counts[:, :, a].ravel())
            ppr.append(avg_pk[:, :, a].ravel()); pgt.append(bs.imp_peaks[:, :, a].ravel())
            assay.append(np.full(bs.imp_pval[:, :, a].size, a, dtype=np.int64))
    sig, gtp, cmean, gtc, ppr, pgt, assay = (np.concatenate(x) for x in
                                             (sig, gtp, cmean, gtc, ppr, pgt, assay))
    rng = np.random.default_rng(seed)
    if sig.size > max_points:
        idx = rng.choice(sig.size, max_points, replace=False)
        sig, gtp, cmean, gtc, ppr, pgt, assay = (a[idx] for a in (sig, gtp, cmean, gtc, ppr, pgt, assay))

    y = np.rint(np.clip(gtc, 0, None)).astype(np.int64)
    n_a, p_a = nb_n[assay], nb_p[assay]
    pit = np.clip(nbinom.cdf(y - 1, n_a, p_a) + 0.5 * nbinom.pmf(y, n_a, p_a), 0, 1)
    # C-index samples on a subsample
    cs = rng.choice(sig.size, min(cidx_cap, sig.size), replace=False)
    csamp = nbinom.rvs(n_a[cs][None, :].repeat(n_samples, 0), p_a[cs][None, :].repeat(n_samples, 0),
                       random_state=seed).astype(np.float32)            # [S, n]
    imp = ReconArrays(signal_pred=sig, gt_pval=gtp, count_mean=cmean, gt_counts=gtc)
    cal = CalArrays(pit=pit, cidx_samples=csamp, cidx_gt=gtc[cs], peak_prob=ppr, peak_gt=pgt)
    return imp, cal


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from eval_v3 import recon_corrs, ece_from_pit, c_index_from_samples, auroc  # noqa: E402
    evd = load_eval("/project/6014832/mforooz/EpiDenoise/sandbox/data/sandbox.h5")
    imp, cal = baseline_arrays(evd)
    ci = recon_corrs(imp)
    print("BASELINE imputation 4-corr:", {k: round(v, 4) for k, v in ci.items()})
    print("BASELINE ECE      :", round(ece_from_pit(cal.pit), 4))
    print("BASELINE C-index  :", round(c_index_from_samples(cal.cidx_samples, cal.cidx_gt), 4))
    print("BASELINE peak AUROC:", round(auroc(cal.peak_prob, cal.peak_gt), 4))

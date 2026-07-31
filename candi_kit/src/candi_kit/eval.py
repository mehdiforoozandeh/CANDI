"""candi_kit measurement stack — M1/M2/M3 + the S14 depth counterfactual on real batches.

VENDORED (COPY+EDIT) of sandbox/diagnostics/dual_conditioning_real/metrics_real.py:1-1001,
plus _build_vb_natural_missing_meta VERBATIM out of sandbox/train.py:118-142.

Readouts run on real `CandiKitH5Dataset` batches and use ONLY the primitives in `candi_kit.metrics`
(`nb_crps`, `nb_quantile`, `ece`/`calibration_pit_curve`, `spearman`, `pearson`, `r2`, `_cos_dist`)
plus the p-from-true-mu-in-float64 convention.

- **M1 (health)** — imp/den count Spearman+Pearson, NB CRPS, PIT-ECE, per-assay honest marginal +
  oracle-scale decomposition, per-position encoder eff-rank.
- **M2 (depth / run_type / read_length)** — the counterfactual-prompt FLIP test on the held-out
  imputation targets (target absent from the T_ input => the prompt is the only channel), the told-depth
  sweep with clamp telemetry, and the sentinel-free cross-target metadata ablation.
- **M3 (invariance)** — same region encoded at input DSF in {1,2,4,8} (each with its TRUE x_meta) ->
  within/between latent cos-dist ratio, guarded by encoder eff-rank>1.
- **S14** — each told depth scored against its OWN ground truth `counts_dsf{k}`.

THE ASSAY ORDER IS NEVER DECLARED HERE. `build_eval_units` returns it alongside the units, read from
the h5 attrs via the dataset, and the caller threads it into every labelled instrument.

The encoder latent `z` is invariant to `y_meta`, so it is cached per eval unit and prompt variants only
re-run the (cheap) decoder.
"""
from __future__ import annotations

import json
from math import comb
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch

from candi_kit.batch import CLOZE, MISSING, make_masker, prepare_masked_batch
from candi_kit.dataset import CandiKitH5Dataset
from candi_kit.metrics import (
    nb_crps, nb_quantile, ece, calibration_pit_curve, spearman, pearson, r2, _cos_dist, P_EPS,
)
from candi_kit.model import forward_full

RUN_TYPE_ROW, DEPTH_ROW, READLEN_ROW = 3, 0, 2
OBSERVED_READLENS = (30.0, 36.0, 76.0, 100.0, 101.0)

# Keys emitted ONLY under include_deprecated=True. Each one is a measurement that was made, audited and
# found not to support the claim it was reporting; they ship with their verdict so a reader who finds one
# in a results JSON cannot mistake it for evidence.
DEPRECATED_VERDICTS = {
    "read_length": "7/12 flips land outside per-assay training support -> measures OOD extrapolation, "
                   "not read-length steering.",
    "null": "shuffled-depth null is a mathematical NO-OP: y_meta_imp is one [4,F] tensor broadcast over "
            "the batch, so base_d[perm] == base_d bitwise. A real null must permute ACROSS targets.",
    "null_clustered": "same no-op as `null`, target-clustered.",
    "frac_min_at_true": "scores every told depth against the fixed dsf1 target, so any mu-decreasing "
                        "model satisfies it (0.7588 vs 0.7597 between arms). Superseded by S14.",
    "direction": "position-level bootstrap CI — ~24x too narrow; positions within a target are not "
                 "independent draws. Use the *_clustered keys.",
    "overall": "position-level bootstrap CI — ~24x too narrow. Use overall_clustered.",
    "single": "position-level bootstrap CI — ~24x too narrow. Use single_clustered.",
    "paired": "position-level bootstrap CI — ~24x too narrow. Use paired_clustered.",
    "median_eta_slope": "offset-free residual; decided by the sign of ~1e-17 float noise under "
                        "offset-ON. total_slope = beta + eta_slope, so eta_slope ~ 0 under a correct "
                        "offset is arithmetically right, not a failure.",
    "offset_independent": "boolean read of median_eta_slope > 0 — i.e. of float noise.",
    "frac_direction": "strict `>` on mean_delta, so exact ties report 0.0% correct rather than "
                      "'no signal'.",
}


# ---------------------------------------------------------------------------
# lifted VERBATIM out of sandbox/train.py:118-142 (the only reason metrics_real.py imported sandbox.train)
# ---------------------------------------------------------------------------

def _build_vb_natural_missing_meta(
    t_meta: torch.Tensor,
    vb_meta: torch.Tensor,
    y_avail: torch.Tensor,
    canonical_meta: Optional[torch.Tensor],
) -> torch.Tensor:
    """V/B natural metadata for assays missing in T (y_avail==0); canonical fallback.

    Used when ``use_canonical_missing_meta=False`` (E32 / CANDI v2 default): inject the
    paired V_*/B_* biosample's real covariates at imp-eval slots so depth_offset heads
    see the correct sequencing depth. Falls back to EIC canonical medians when V/B meta
    row 0 is invalid (-1).
    """
    device = t_meta.device
    mixed = t_meta.clone()
    missing = (y_avail.to(device) == 0).unsqueeze(1).expand_as(mixed)
    vb = vb_meta.to(device)
    valid_vb = (vb[:, 0:1, :] != -1.0).expand_as(mixed)
    use_vb = missing & valid_vb
    mixed[use_vb] = vb[use_vb]
    if canonical_meta is not None:
        can_exp = canonical_meta.to(device).unsqueeze(0).expand_as(mixed)
        still_missing = missing & (mixed[:, 0:1, :] == -1.0).expand_as(mixed)
        mixed[still_missing] = can_exp[still_missing]
    return mixed


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def effective_rank(Z: np.ndarray) -> float:
    """Roy-Vetterli effective rank = exp(entropy of the normalized singular-value spectrum)."""
    if Z.ndim != 2 or Z.shape[0] < 2:
        return float("nan")
    Zc = Z - Z.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Zc, compute_uv=False)
    s = s[s > 1e-12]
    if s.size == 0:
        return float("nan")
    pk = s / s.sum()
    return float(np.exp(-np.sum(pk * np.log(pk))))


def _foreground_mask(target: np.ndarray, fg_frac: float, min_n: int = 5,
                     seed: int = 0) -> Tuple[np.ndarray, bool]:
    """RANK-based foreground: the top `fg_frac` positions by count, not `target >= quantile`.

    The quantile form degenerates on sparse tracks: with 36-67% exact zeros the 98th-percentile of a
    held-out target is often 0, so `target >= 0` selects EVERY position. Measured on the q19 eval:
    2% requested, 23.9% realised, 21.6% of records at 100% -- and those records supplied 90% of the
    bootstrap mass, so the steering tests ran mostly on background.

    Rank selection makes the realised fraction exact by construction. `target >= 1` is then applied as
    a purity filter whenever it still leaves `min_n` positions, so background can never enter while real
    signal is available; an all-zero target keeps the top-k so the mask is never empty.

    Returns `(mask, purity_fallback_fired)`. The flag is True when the `target >= 1` purity filter was
    DROPPED -- it fires on ~18% of real held-out records and the record still reports a healthy-looking
    `n_fg`, so it is propagated into every per-target record and excluded from `_cluster_bootstrap_ci`
    by default.

    Ties at the selection threshold are broken by a SEEDED RANDOM draw. `argsort(kind="stable")` takes
    the LAST index among ties and 87.8% of records break a tie there, giving a mean normalized genomic
    position of 0.618 against an unbiased 0.50.
    """
    n = target.size
    if n == 0:
        return np.zeros(0, bool), False
    k = int(min(n, max(min_n, round(fg_frac * n))))
    m = np.zeros(n, bool)
    jitter = np.random.default_rng(seed).random(n)
    m[np.lexsort((jitter, target))[-k:]] = True
    pos = target >= 1.0
    if int(pos.sum()) >= min(min_n, n):
        return m & pos, False
    return m, True


def _fg_pos_mean(fg: np.ndarray) -> float:
    """Realized mean normalized genomic position of the selected foreground (unbiased = 0.50)."""
    if fg.size < 2 or not bool(fg.any()):
        return float("nan")
    return float(np.mean(np.flatnonzero(fg) / (fg.size - 1.0)))


def _p_from_true_mu(n: np.ndarray, mu: np.ndarray) -> np.ndarray:
    return np.clip(n / (n + mu), P_EPS, 1.0 - P_EPS)


def _bootstrap_ci(delta: np.ndarray, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """LEGACY / over-confident: paired bootstrap over per-POSITION `delta` (= crps_flip - crps_true).

    Kept only as a labelled comparison. Positions within a target are not independent draws -- the real
    unit of replication is the target (12 of them / 5 biosample pairs), so this CI is ~24x too narrow.
    Use `_cluster_bootstrap_ci` for any inferential claim.
    """
    if delta.size == 0:
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"), excludes_zero=False, n=0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, delta.size, size=(n_boot, delta.size))
    boot = delta[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [alpha / 2, 1.0 - alpha / 2])
    return dict(mean=float(delta.mean()), lo=float(lo), hi=float(hi),
                excludes_zero=bool(lo > 0.0 or hi < 0.0), n=int(delta.size))


def _sign_test_p(n_pos: int, n: int) -> float:
    """Exact two-sided binomial sign test at p=0.5 (small-p method). n<=12 here, so enumerate."""
    if n <= 0:
        return float("nan")
    pmf = [comb(n, k) * 0.5 ** n for k in range(n + 1)]
    return float(sum(p for p in pmf if p <= pmf[n_pos] + 1e-12))


def _cluster_bootstrap_ci(per_target: List[Dict], *, value_key: str = "mean_delta",
                          n_boot: int = 1000, seed: int = 0, alpha: float = 0.05,
                          exclude_flagged: bool = True) -> Dict:
    """PRIMARY inference: bootstrap over TARGETS, `n_fg`-weighted, with a target-level sign test.

    Effective replication is 12 `(biosample, imp_biosample, assay)` targets / 5 biosample pairs, not the
    ~893k positions `_bootstrap_ci` resamples. Records are first collapsed to one `n_fg`-weighted number
    per target, then the TARGETS are resampled. The unweighted variant is dominated by a few blown-up
    records, so the weighting is not cosmetic.

    Also reports `sign` and `supports_direction = lo > 0` separately, because `excludes_zero` is
    sign-blind -- a significant effect in the WRONG direction would satisfy it.

    Records whose foreground fell back off the `target >= 1` purity filter are dropped by default
    (`exclude_flagged`): they look healthy in `n_fg` but were scored on background.
    """
    n_flagged = 0
    if exclude_flagged:
        n_flagged = sum(1 for r in per_target if r.get("purity_fallback_fired", False))
        per_target = [r for r in per_target if not r.get("purity_fallback_fired", False)]
    if not per_target:
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"), excludes_zero=False,
                    supports_direction=False, sign=0, n_clusters=0, n_records=0, n_pos=0,
                    sign_test_p=float("nan"), n_flagged_excluded=n_flagged)
    agg: Dict = {}
    for r in per_target:
        k = tuple(r["target"])
        v, w = float(r[value_key]), max(float(r.get("n_fg", 1) or 1), 1.0)
        s = agg.setdefault(k, [0.0, 0.0])
        s[0] += w * v
        s[1] += w
    keys = sorted(agg)
    vals = np.array([agg[k][0] / agg[k][1] for k in keys])
    wts = np.array([agg[k][1] for k in keys])
    C = len(keys)
    idx = np.random.default_rng(seed).integers(0, C, size=(n_boot, C))
    boot = (vals[idx] * wts[idx]).sum(axis=1) / wts[idx].sum(axis=1)
    lo, hi = np.quantile(boot, [alpha / 2, 1.0 - alpha / 2])
    mean = float((vals * wts).sum() / wts.sum())
    # Sign test: DROP exact ties (standard practice), do not score them as negatives. Counting ties as
    # negatives made a bit-exactly unresponsive arm (all 12 deltas == 0) read n_pos=0 -> p=0.00049, i.e.
    # the most perfectly NULL arm printed the most significant-looking p-value in the scorecard.
    n_pos = int((vals > 0).sum())
    n_neg = int((vals < 0).sum())
    n_eff = n_pos + n_neg
    n_tied = C - n_eff
    return dict(mean=mean, lo=float(lo), hi=float(hi),
                excludes_zero=bool(lo > 0.0 or hi < 0.0),
                supports_direction=bool(lo > 0.0), sign=int(np.sign(mean)),
                n_clusters=C, n_records=len(per_target), n_flagged_excluded=n_flagged,
                n_pos=n_pos, n_neg=n_neg, n_tied=n_tied,
                frac_pos=n_pos / C, sign_test_p=_sign_test_p(n_pos, n_eff),
                cluster_values=[float(v) for v in vals])


# ---------------------------------------------------------------------------
# eval units — real eval-chromosome batches with the imputation prompt; cache z + GT
# ---------------------------------------------------------------------------

class EvalUnit:
    """One (T_, V_/B_) eval batch: cached latent + prompt + den/imp masks + GT (all on device)."""

    def __init__(self, z, y_meta_true, den_map, y_data, imp_map=None, y_data_imp=None,
                 y_meta_imp=None, y_avail=None, biosample="", imp_biosample=""):
        self.z = z
        self.y_meta_true = y_meta_true           # [B,4,A] mixed prompt (VB-natural at imp slots)
        self.den_map = den_map                   # [B,L,A] unmasked available T_ (denoise GT = y_data)
        self.y_data = y_data                     # [B,L,A]
        self.imp_map = imp_map                    # [B,L,A] or None
        self.y_data_imp = y_data_imp             # [B,L,A] or None (imp GT)
        self.y_meta_imp = y_meta_imp             # [B,4,A] V/B natural meta or None
        self.y_avail = y_avail                   # [B,A] or None
        self.biosample = biosample
        self.imp_biosample = imp_biosample

    def imp_target_assays(self) -> List[int]:
        if self.imp_map is None:
            return []
        return [a for a in range(self.imp_map.shape[-1]) if bool(self.imp_map[:, :, a].any())]


@torch.no_grad()
def build_eval_units(model, h5_path, device, *, regime: str = "type1", batch_size: int = 4,
                     max_batches: Optional[int] = None, imp_prefixes=("V_", "B_"),
                     seed: int = 0) -> Tuple[List[EvalUnit], List[str]]:
    """Returns (units, assays). `assays` is the h5's own column order — the only assay labelling the
    kit ever uses; every labelled instrument below takes it as an explicit argument."""
    ds = CandiKitH5Dataset(h5_path, regime, train=False, batch_size=batch_size, biosample_prefix="T_",
                           dsf_sampling="off", seed=seed, shuffle=False,
                           eval_include_vb_ground_truth=True, imp_prefixes=imp_prefixes)
    noop = make_masker(p_full_loci=0.0, p_full_assay=0.0, p_chunks=0.0, mask_fraction=0.0)
    model.eval()
    units: List[EvalUnit] = []
    for bi, batch in enumerate(ds):
        if max_batches is not None and bi >= max_batches:
            break
        prep = prepare_masked_batch(batch, noop, device, apply_mask=False)
        if prep is None:
            continue
        y_meta_fwd = prep["y_meta"].clone()
        ymi = batch.get("y_meta_imp")
        yav = batch.get("y_avail")
        ydi = batch.get("y_data_imp")
        has_imp = isinstance(ymi, torch.Tensor) and isinstance(ydi, torch.Tensor)
        imp_map = ydi_d = ymi_d = yav_d = None
        if has_imp:
            ymi_d = ymi.to(device); yav_d = yav.to(device); ydi_d = ydi.to(device)
            y_meta_fwd = _build_vb_natural_missing_meta(y_meta_fwd, ymi_d, yav_d, None)
            imp_map = ((yav_d <= 0).unsqueeze(1).expand_as(ydi_d)) & (ydi_d != -1)
        z = model.encode(prep["x_data"], prep["x_dna"], prep["x_meta"])
        units.append(EvalUnit(
            z=z, y_meta_true=y_meta_fwd, den_map=prep["observed_map"], y_data=prep["y_data"],
            imp_map=imp_map, y_data_imp=ydi_d, y_meta_imp=ymi_d, y_avail=yav_d,
            biosample=str(batch.get("biosample_name", "")),
            imp_biosample=str(batch.get("imp_biosample_name", ""))))
    return units, list(ds.assays)


@torch.no_grad()
def _decode(model, u: EvalUnit, y_meta: torch.Tensor) -> Dict[str, torch.Tensor]:
    return model.decoder(u.z, y_meta)


def _gather(out: Dict[str, torch.Tensor], mask: torch.Tensor, a: int,
            target: torch.Tensor) -> Optional[Dict[str, np.ndarray]]:
    """Pool assay a's per-position (mu,n,p,eta,log2_mu,target) over a [B,L,A] bool mask; p from true mu."""
    m = mask[:, :, a]
    if not bool(m.any()):
        return None
    mu = out["mu"][:, :, a][m].double().cpu().numpy()
    n = out["n"][:, :, a][m].double().cpu().numpy()
    eta = out["eta"][:, :, a][m].float().cpu().numpy()
    log2_mu = out["log2_mu"][:, :, a][m].double().cpu().numpy()
    tgt = target[:, :, a][m].double().cpu().numpy()
    return dict(mu=mu, n=n, p=_p_from_true_mu(n, mu), eta=eta, log2_mu=log2_mu, target=tgt)


# ---------------------------------------------------------------------------
# honest per-assay marginal + oracle scale decomposition
# ---------------------------------------------------------------------------

def _marginal_nb(target: np.ndarray) -> Dict:
    """The CRPS-OPTIMAL constant NB forecast for one assay (the honest bar).

    The old baseline used `mu = median(target) + 1e-6`. On 4/8 assays the median count is 0, so the
    "NB marginal" was a point mass at 0 and `marg_crps` reduced to `mean(y)` -- a forecast of
    "predict nothing" that any model beats, which is how "beats marginal 8/8" was manufactured.

    Since the forecast is constant, its CRPS depends on the target only through its HISTOGRAM, so the
    (mu, n) grid search is exact and cheap on the unique values.
    """
    if target.size < 2:
        return dict(marg_crps=float("nan"), marg_mu=float("nan"), marg_n=float("nan"),
                    marg_mu_legacy_median=float("nan"), marg_crps_legacy_median=float("nan"))
    vals, cnts = np.unique(target, return_counts=True)
    w = cnts / cnts.sum()
    mean = float(np.dot(w, vals))
    var = float(np.dot(w, (vals - mean) ** 2))
    mu0 = max(mean, 1e-3)
    n0 = max((mu0 * mu0) / max(var - mu0, 1e-3), 1e-3)                  # NB var = mu + mu^2/n

    def _crps_const(mu_c, n_c):
        nv, mv = np.full_like(vals, n_c), np.full_like(vals, mu_c)
        return float(np.dot(w, nb_crps(nv, _p_from_true_mu(nv, mv), vals)))

    # the legacy median-based baseline, kept so the degeneracy is visible rather than asserted -- and
    # entered as a CANDIDATE, so the honest bar provably dominates it (on an all-zero target a near
    # point mass genuinely IS the CRPS-optimal constant; the audit's objection was that the MEDIAN rule
    # produced one when it was not optimal, not that it can never be).
    med = float(np.median(target)) + 1e-6
    med_n = max((med * med) / max(var - med, 1e-6), 1e-6)
    crps_legacy = _crps_const(med, med_n)
    best = (crps_legacy, med, med_n)
    for c in np.arange(-8.0, 3.01, 0.25):
        mu_c = mu0 * 2.0 ** c
        for f in np.arange(-3.0, 3.01, 0.5):
            n_c = max(n0 * 2.0 ** f, 1e-6)
            v = _crps_const(mu_c, n_c)
            if v < best[0]:
                best = (v, mu_c, n_c)
    return dict(marg_crps=best[0], marg_mu=best[1], marg_n=best[2],
                marg_mu_legacy_median=med, marg_crps_legacy_median=crps_legacy)


def _oracle_scale(mu: np.ndarray, n: np.ndarray, target: np.ndarray, *,
                  fit_budget: int = 20_000, seed: int = 0) -> Dict:
    """Oracle per-assay multiplicative scale `c* = argmin_c CRPS(NB(n, mu*2^c), y)`.

    Macro CRPS is location-dominated (a 4x mu error costs +84%, a 4x dispersion error +9%), so an arm
    that wins on scale and an arm that wins on shape sum into an apparent Pareto. This splits it:
    `crps_oracle_scaled` is the CAPABILITY term (what the model gets right once its per-assay level is
    granted) and `scale_error = crps - crps_oracle_scaled` is the FIXABLE calibration term.
    `n_star_log2` is the analogous oracle dispersion rescale.

    The grid search runs on a subsample; the reported CRPS values are evaluated on the FULL pool at the
    selected oracles, so they are directly comparable to the un-decomposed `crps`. That makes
    `crps_oracle_scaled` an IN-SAMPLE oracle -- an upper bound on capability -- and lets `scale_error`
    go slightly negative (-0.0008 observed).
    """
    N = len(mu)
    sel = (np.arange(N) if N <= fit_budget
           else np.random.default_rng(seed).choice(N, fit_budget, replace=False))
    mf, nf, tf = mu[sel], n[sel], target[sel]

    def _fit(c, k=0.0):
        m, nv = mf * 2.0 ** c, nf * 2.0 ** k
        return float(np.mean(nb_crps(nv, _p_from_true_mu(nv, m), tf)))

    c = float(min(np.arange(-6.0, 6.001, 0.25), key=_fit))
    c = float(min(np.arange(c - 0.25, c + 0.2501, 0.01), key=_fit))
    k = float(min(np.arange(-4.0, 4.001, 0.25), key=lambda kk: _fit(c, kk)))
    ms = mu * 2.0 ** c
    nk = n * 2.0 ** k
    return dict(c_star=c, n_star_log2=k,
                crps_oracle_scaled=float(np.mean(nb_crps(n, _p_from_true_mu(n, ms), target))),
                crps_oracle_scaled_and_n=float(np.mean(nb_crps(nk, _p_from_true_mu(nk, ms), target))))


# ---------------------------------------------------------------------------
# M1 — counts-only reconstruction / imputation health
# ---------------------------------------------------------------------------

@torch.no_grad()
def _spearman_diag(mu, tgt, label: str) -> float:
    """spearman(), but says WHY it is nan instead of emitting a bare nan.

    Undefined rank correlation is a real outcome on a small panel -- a held-out target that is all-zero
    on the eval chromosome has no ranking to recover. Reported silently it reads as a broken metric, so
    name the cause at the point it happens.
    """
    v = spearman(mu, tgt)
    if not np.isfinite(v):
        import sys as _sys
        why = ("target is constant (%d points, all == %g)" % (len(tgt), tgt[0])
               if len(tgt) and float(np.std(tgt)) < 1e-12 else
               "prediction is constant" if len(mu) and float(np.std(mu)) < 1e-12 else
               "fewer than 2 points (n=%d)" % len(tgt))
        print("[eval] WARNING %s: Spearman undefined -- %s. This is a property of the eval split, "
              "not a model failure; widen the panel or the eval chromosome set." % (label, why),
              file=_sys.stderr)
    return v


def eval_M1(model, units: List[EvalUnit], device, assays: Sequence[str], *,
            budget: int = 200_000, seed: int = 0, include_deprecated: bool = False) -> Dict:
    # `include_deprecated` is accepted for a uniform evaluate() call; M1 emits no deprecated keys.
    den, imp, zs = [], [], []
    A = len(assays)
    den_a = {a: [] for a in range(A)}
    imp_a = {a: [] for a in range(A)}
    for u in units:
        out = _decode(model, u, u.y_meta_true)
        for a in range(A):
            g = _gather(out, u.den_map, a, u.y_data)
            if g is not None:
                den.append(g)
                den_a[a].append(g)
        if u.imp_map is not None:
            for a in u.imp_target_assays():
                g = _gather(out, u.imp_map, a, u.y_data_imp)
                if g is not None:
                    imp.append(g)
                    imp_a[a].append(g)
        zs.append(u.z.reshape(-1, u.z.shape[-1]).float().cpu().numpy())   # [B*L2, d] per-position latent

    def _pool(parts):
        if not parts:
            return None
        return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}

    def _suite(pool, tag):
        if pool is None:
            return dict(n_points=0)
        rng = np.random.default_rng(seed)
        n0 = len(pool["mu"])
        sel = np.arange(n0) if n0 <= budget else rng.choice(n0, budget, replace=False)
        mu, n, p, tgt = pool["mu"][sel], pool["n"][sel], pool["p"][sel], pool["target"][sel]
        grid, fbar = calibration_pit_curve(n, p, tgt)
        # spearman is on RAW counts, pearson on log1p -- two different spaces under one heading, so the
        # keys carry their space and can never be quoted as a pair.
        return dict(spearman_raw=_spearman_diag(mu, tgt, tag), pearson_log1p=pearson(np.log1p(mu), np.log1p(tgt)),
                    crps=float(np.mean(nb_crps(n, p, tgt))), ece=ece(n, p, tgt),
                    r2=r2(mu, tgt), n_points=int(len(mu)), calib_grid=grid, calib_fbar=fbar)

    den_pool, imp_pool = _pool(den), _pool(imp)
    m_den, m_imp = _suite(den_pool, "den"), _suite(imp_pool, "imp")

    # Per-assay decomposition. Pooled Spearman/Pearson mix assays of very different count magnitudes,
    # so they reward getting the CROSS-ASSAY SCALE right -- exactly the offset's free 2^(d-center)
    # arithmetic. Per-assay corr is scale-free (within-assay ranking) => isolates SHAPE (the biology);
    # median_mu vs median_target exposes the cross-assay scale bias directly.
    def _suite_light(pool):
        n0 = len(pool["mu"])
        sel = np.arange(n0) if n0 <= budget else np.random.default_rng(seed).choice(n0, budget, replace=False)
        mu, n, p, tgt = pool["mu"][sel], pool["n"][sel], pool["p"][sel], pool["target"][sel]
        crps = float(np.mean(nb_crps(n, p, tgt)))
        mar = _marginal_nb(tgt)                                         # CRPS-optimal, not median
        orc = _oracle_scale(mu, n, tgt, seed=seed)                      # scale / capability split
        mcrps = mar["marg_crps"]
        return dict(spearman_raw=spearman(mu, tgt), pearson_log1p=pearson(np.log1p(mu), np.log1p(tgt)),
                    crps=crps, r2=r2(mu, tgt),
                    mse_log=float(np.mean((np.log1p(mu) - np.log1p(tgt)) ** 2)),
                    marg_crps=mcrps, beats_marginal=bool(np.isfinite(mcrps) and crps < mcrps),
                    marg_mu=mar["marg_mu"], marg_n=mar["marg_n"],
                    marg_mu_legacy_median=mar["marg_mu_legacy_median"],
                    marg_crps_legacy_median=mar["marg_crps_legacy_median"],
                    **orc, scale_error=crps - orc["crps_oracle_scaled"],
                    beats_marginal_oracle_scaled=bool(np.isfinite(mcrps)
                                                      and orc["crps_oracle_scaled"] < mcrps),
                    median_mu=float(np.median(mu)), median_target=float(np.median(tgt)),
                    n_points=int(len(mu)))

    def _per_assay(by_assay):
        return {assays[a]: _suite_light(_pool(parts))
                for a, parts in by_assay.items() if parts}

    def _macro(pa, key):
        vals = [v[key] for v in pa.values() if np.isfinite(v.get(key, np.nan))]
        return float(np.mean(vals)) if vals else float("nan")

    imp_pa, den_pa = _per_assay(imp_a), _per_assay(den_a)
    Z = np.concatenate(zs) if zs else np.zeros((0, 1))
    if len(Z) > budget:
        Z = Z[np.random.default_rng(seed).choice(len(Z), budget, replace=False)]
    eff = effective_rank(Z)
    return dict(imp=m_imp, den=m_den, encoder_eff_rank_perpos=eff,
                imp_per_assay=imp_pa, den_per_assay=den_pa,
                imp_macro_spearman_raw=_macro(imp_pa, "spearman_raw"),
                imp_macro_pearson_log1p=_macro(imp_pa, "pearson_log1p"),
                den_macro_spearman_raw=_macro(den_pa, "spearman_raw"),
                den_macro_pearson_log1p=_macro(den_pa, "pearson_log1p"),
                # macro CRPS split into the capability term and the fixable per-assay scale term.
                imp_macro_crps=_macro(imp_pa, "crps"),
                imp_macro_crps_oracle_scaled=_macro(imp_pa, "crps_oracle_scaled"),
                imp_macro_scale_error=_macro(imp_pa, "scale_error"),
                imp_macro_marg_crps=_macro(imp_pa, "marg_crps"),
                imp_beats_marginal_n=int(sum(v.get("beats_marginal", False) for v in imp_pa.values())),
                imp_beats_marginal_oracle_scaled_n=int(
                    sum(v.get("beats_marginal_oracle_scaled", False) for v in imp_pa.values())),
                den_macro_crps=_macro(den_pa, "crps"))


# ---------------------------------------------------------------------------
# M2 — counterfactual-prompt FLIP (depth, run_type, read_length)
# ---------------------------------------------------------------------------

def _flip_run_type(y_meta: torch.Tensor, a: int) -> torch.Tensor:
    ym = y_meta.clone()
    rt = ym[:, RUN_TYPE_ROW, a]
    ym[:, RUN_TYPE_ROW, a] = torch.where((rt == 0) | (rt == 1), 1.0 - rt, rt)   # 0<->1; leave sentinels
    return ym


def _flip_read_length(y_meta: torch.Tensor, a: int) -> torch.Tensor:
    ym = y_meta.clone()
    obs = torch.tensor(OBSERVED_READLENS, device=y_meta.device)
    cur = ym[:, READLEN_ROW, a]
    for b in range(ym.shape[0]):
        v = float(cur[b])
        if v <= 0:
            continue
        far = obs[(obs - v).abs().argmax()]    # flip to the FARTHEST observed read length
        ym[b, READLEN_ROW, a] = far
    return ym


def _set_depth(y_meta: torch.Tensor, a: int, depth_vec: torch.Tensor) -> torch.Tensor:
    ym = y_meta.clone()
    ym[:, DEPTH_ROW, a] = depth_vec
    return ym


def _target_run_type_label(u: "EvalUnit", a: int) -> str:
    """TRUE run_type of held-out target assay a, read off the VB-natural prompt (row 3)."""
    if u.y_meta_imp is None:
        return "?"
    rt = float(u.y_meta_imp[0, RUN_TYPE_ROW, a])
    return "paired" if rt == 1.0 else ("single" if rt == 0.0 else "?")


@torch.no_grad()
def _flip_covariate(model, units, device, covariate: str, assays: Sequence[str], *, fg_frac: float,
                    n_boot: int, seed: int, include_deprecated: bool = False) -> Dict:
    """Direction + responsiveness for a categorical flip (run_type / read_length) over held-out targets."""
    per_target, pooled_delta, split = [], {"all": []}, {"single": [], "paired": []}
    for u in units:
        if u.imp_map is None:
            continue
        for a in u.imp_target_assays():
            ym_true = u.y_meta_true
            ym_flip = _flip_run_type(ym_true, a) if covariate == "run_type" else _flip_read_length(ym_true, a)
            g_true = _gather(_decode(model, u, ym_true), u.imp_map, a, u.y_data_imp)
            g_flip = _gather(_decode(model, u, ym_flip), u.imp_map, a, u.y_data_imp)
            if g_true is None or g_flip is None:
                continue
            fg, fired = _foreground_mask(g_true["target"], fg_frac, seed=seed)
            crps_true = nb_crps(g_true["n"][fg], g_true["p"][fg], g_true["target"][fg])
            crps_flip = nb_crps(g_flip["n"][fg], g_flip["p"][fg], g_flip["target"][fg])
            delta = crps_flip - crps_true                     # >0 means TRUE prompt is better (direction)
            resp = float(np.mean(np.abs(g_true["mu"][fg] - g_flip["mu"][fg])))
            key = (u.biosample, u.imp_biosample, assays[a])
            label = _target_run_type_label(u, a)
            rec = dict(target=key, run_type=label, n_fg=int(fg.sum()),
                       fg_frac_realized=float(fg.mean()),
                       purity_fallback_fired=bool(fired), fg_pos_mean=_fg_pos_mean(fg),
                       crps_true=float(np.mean(crps_true)), crps_flip=float(np.mean(crps_flip)),
                       mean_delta=float(np.mean(delta)), responsiveness=resp,
                       direction_true_better=bool(np.mean(delta) > 0))
            per_target.append(rec)
            pooled_delta["all"].append(delta)
            if label in ("single", "paired"):
                split[label].append(delta)

    def _agg(dlist):
        if not dlist:
            return dict(n=0)
        return _bootstrap_ci(np.concatenate(dlist), n_boot=n_boot, seed=seed)

    def _cl(recs):
        return _cluster_bootstrap_ci(recs, n_boot=n_boot, seed=seed)

    by_label = {lab: [r for r in per_target if r["run_type"] == lab] for lab in ("single", "paired")}
    out = dict(covariate=covariate, per_target=per_target,
               # PRIMARY: target-clustered, n_fg-weighted, sign-aware.
               overall_clustered=_cl(per_target), single_clustered=_cl(by_label["single"]),
               paired_clustered=_cl(by_label["paired"]),
               n_targets=len(per_target),
               mean_responsiveness=float(np.mean([r["responsiveness"] for r in per_target]))
               if per_target else float("nan"))
    # honest-null flag: a bit-exactly zero prompt response is a MODEL statistic, so it is named as one.
    out["model_unresponsive"] = bool(
        per_target and out["mean_responsiveness"] < 1e-6)
    if include_deprecated:
        out.update(overall=_agg(pooled_delta["all"]),
                   single=_agg(split["single"]), paired=_agg(split["paired"]),
                   frac_direction=float(np.mean([r["direction_true_better"] for r in per_target]))
                   if per_target else float("nan"),
                   deprecated_verdicts={k: DEPRECATED_VERDICTS[k]
                                        for k in ("overall", "single", "paired", "frac_direction")})
    return out


@torch.no_grad()
def _depth_sweep(model, units, device, assays: Sequence[str], *, fg_frac: float, n_boot: int, seed: int,
                 include_deprecated: bool = False) -> Dict:
    """Depth: sweep told-depth over the imp biosample's achievable meta_dsf levels; direction +
    offset-independent eta/tail regression + shuffled-prompt null."""
    per_target = []
    pooled_dir_delta, pooled_null_delta = [], []
    dec = getattr(model, "decoder", None)
    src = dec if hasattr(dec, "clamp_lo") else model     # real model keeps it on .decoder; stubs on self
    clamp_lo = float(getattr(src, "clamp_lo", -np.inf))
    clamp_hi = float(getattr(src, "clamp_hi", np.inf))
    for u in units:
        if u.imp_map is None or u.y_meta_imp is None:
            continue
        for a in u.imp_target_assays():
            base_d = u.y_meta_imp[:, DEPTH_ROW, a]              # [B] true (VB-natural) depth
            if not torch.isfinite(base_d).all() or bool((base_d < 0).any()):
                continue
            told = [base_d - float(np.log2(k)) for k in (1, 2, 4, 8)]   # achievable set (downsample)
            gs, eta_means, tail_means, crps_means, dvals = [], [], [], [], []
            log2mu_means, sat_fracs = [], []
            fired_any = False
            for dvec in told:
                g = _gather(_decode(model, u, _set_depth(u.y_meta_true, a, dvec)), u.imp_map, a, u.y_data_imp)
                if g is None:
                    gs = []
                    break
                fg, fired = _foreground_mask(g["target"], fg_frac, seed=seed)
                fired_any = fired_any or fired
                gs.append((g, fg))
                eta_means.append(float(np.mean(g["eta"][fg])))
                log2mu_means.append(float(np.mean(g["log2_mu"][fg])))
                lm = g["log2_mu"][fg]
                sat_fracs.append(float(np.mean((lm <= clamp_lo + 1e-6) | (lm >= clamp_hi - 1e-6))))
                tail_means.append(float(np.mean(nb_quantile(0.95, g["n"][fg], g["p"][fg]))))
                crps_means.append(float(np.mean(nb_crps(g["n"][fg], g["p"][fg], g["target"][fg]))))
                dvals.append(float(dvec.float().mean()))
            if not gs:
                continue
            # direction: true depth (k=1, index 0) should have the lowest CRPS vs GT
            g_true, fg_true = gs[0]
            crps_true = nb_crps(g_true["n"][fg_true], g_true["p"][fg_true], g_true["target"][fg_true])
            # wrong depth = the most-downsampled told (k=8, index -1)
            g_wrong, fg_w = gs[-1]
            crps_wrong = nb_crps(g_wrong["n"][fg_true] if fg_w.shape == fg_true.shape else g_wrong["n"][fg_w],
                                 g_wrong["p"][fg_true] if fg_w.shape == fg_true.shape else g_wrong["p"][fg_w],
                                 g_true["target"][fg_true] if fg_w.shape == fg_true.shape else g_wrong["target"][fg_w])
            dir_delta = crps_wrong - crps_true
            # PRIMARY: total told-depth response d<log2_mu>/d(told depth); a correct exposure model
            # gives 1.0. `eta_slope` below is the offset-FREE residual = 1 - (offset coefficient), i.e. it
            # measures WHERE the response is implemented, not WHETHER the model responds -- demoted to a
            # labelled attribution diagnostic (`total_slope = beta + eta_slope` for any hybrid).
            dv = np.array(dvals)
            total_slope = float(np.polyfit(dv, np.array(log2mu_means), 1)[0]) if np.ptp(dv) > 1e-6 else float("nan")
            eta_slope = float(np.polyfit(dv, np.array(eta_means), 1)[0]) if np.ptp(dv) > 1e-6 else float("nan")
            tail_slope = float(np.polyfit(dv, np.log2(np.array(tail_means) + 1.0), 1)[0]) if np.ptp(dv) > 1e-6 else float("nan")
            # null: told a SHUFFLED (permuted-across-samples) depth -> direction should collapse
            perm = torch.randperm(base_d.shape[0],
                                  generator=torch.Generator().manual_seed(seed)).to(base_d.device)
            g_shuf = _gather(_decode(model, u, _set_depth(u.y_meta_true, a, base_d[perm])),
                             u.imp_map, a, u.y_data_imp)
            null_delta = np.zeros(0)
            if g_shuf is not None:
                null_delta = (nb_crps(g_shuf["n"][fg_true], g_shuf["p"][fg_true], g_true["target"][fg_true])
                              - crps_true)
            per_target.append(dict(
                target=(u.biosample, u.imp_biosample, assays[a]),
                crps_curve=crps_means, told_depth=dvals, eta_means=eta_means, tail_means=tail_means,
                log2mu_means=log2mu_means, n_fg=int(fg_true.sum()), fg_frac_realized=float(fg_true.mean()),
                purity_fallback_fired=bool(fired_any), fg_pos_mean=_fg_pos_mean(fg_true),
                frac_log2mu_at_clamp=float(np.max(sat_fracs)),
                min_at_true=bool(int(np.argmin(crps_means)) == 0),
                total_slope=total_slope, eta_slope=eta_slope, tail_slope=tail_slope,
                dir_mean_delta=float(np.mean(dir_delta)),
                null_mean_delta=float(np.mean(null_delta)) if null_delta.size else float("nan")))
            pooled_dir_delta.append(dir_delta)
            if null_delta.size:
                pooled_null_delta.append(null_delta)
    tot_slopes = [t["total_slope"] for t in per_target if np.isfinite(t["total_slope"])]
    med_tot = float(np.median(tot_slopes)) if tot_slopes else float("nan")
    out = dict(covariate="depth", per_target=per_target,
               # PRIMARY: target-clustered.
               direction_clustered=_cluster_bootstrap_ci(per_target, value_key="dir_mean_delta",
                                                         n_boot=n_boot, seed=seed),
               n_targets=len(per_target),
               # PRIMARY: total told-depth response; |median_total_slope - 1| is the honest bar.
               median_total_slope=med_tot, total_slope_err=abs(med_tot - 1.0),
               # ...but a slope measured through the saturating `log2_mu` clamp is not an exposure
               # coefficient. Flag it, or a saturated arm reads as sub-unit response the way `eta_slope`
               # read as "no response". The MEDIAN is 0 on all four recorded arms at full coverage, but
               # the distribution is heavy-tailed -- `wd0_on_s0` has 205/1215 targets with a nonzero
               # clamp fraction (mean 0.096, p90 0.475, max 0.967) -- so read `total_slope` next to the
               # tail stats, not the median alone.
               median_frac_log2mu_at_clamp=float(np.median([t["frac_log2mu_at_clamp"] for t in per_target]))
               if per_target else float("nan"),
               frac_targets_any_clamp=float(np.mean([t["frac_log2mu_at_clamp"] > 0 for t in per_target]))
               if per_target else float("nan"),
               p90_frac_log2mu_at_clamp=float(np.quantile([t["frac_log2mu_at_clamp"] for t in per_target],
                                                          0.9)) if per_target else float("nan"),
               max_frac_log2mu_at_clamp=float(np.max([t["frac_log2mu_at_clamp"] for t in per_target]))
               if per_target else float("nan"),
               total_slope_clamp_saturated=bool(
                   per_target and np.median([t["frac_log2mu_at_clamp"] for t in per_target]) > 0.01))
    if include_deprecated:
        eta_slopes = [t["eta_slope"] for t in per_target if np.isfinite(t["eta_slope"])]
        out.update(
            null_clustered=_cluster_bootstrap_ci(
                [t for t in per_target if np.isfinite(t["null_mean_delta"])],
                value_key="null_mean_delta", n_boot=n_boot, seed=seed),
            direction=_bootstrap_ci(np.concatenate(pooled_dir_delta), n_boot=n_boot, seed=seed)
            if pooled_dir_delta else dict(n=0),
            null=_bootstrap_ci(np.concatenate(pooled_null_delta), n_boot=n_boot, seed=seed)
            if pooled_null_delta else dict(n=0),
            frac_min_at_true=float(np.mean([t["min_at_true"] for t in per_target]))
            if per_target else float("nan"),
            median_eta_slope=float(np.median(eta_slopes)) if eta_slopes else float("nan"),
            eta_slope_is_attribution_diagnostic=True,
            offset_independent=bool(eta_slopes and np.median(eta_slopes) > 0.0),
            deprecated_verdicts={k: DEPRECATED_VERDICTS[k] for k in
                                 ("null", "null_clustered", "direction", "frac_min_at_true",
                                  "median_eta_slope", "offset_independent")})
    return out


META_ROWS = {0: "depth", 1: "assay_id", 2: "read_length", 3: "run_type"}


@torch.no_grad()
def _metadata_ablation(model, units, row: int, assays: Sequence[str], mode: str = "cross_target", *,
                       fg_frac: float = 0.02, seed: int = 0) -> Dict:
    """Covariate-agnostic "does the decoder use `y_meta` row `row` AT ALL?" probe (real z).

    For every held-out imputation target, re-decode the CACHED encoder `z` with ONE prompt row replaced:
      - `cross_target` (primary): by the value of a DIFFERENT eval target. Required because `meta_dsf{k}`
        has no window axis, so a *within-batch* permutation is structurally the identity and would report
        "uses nothing" for every model.
      - `within_batch`: that degenerate control, kept so the claim is testable rather than asserted; it
        must read exactly 0.
    Reports dCRPS (>0 = the honest prompt is better), mean/max |d mu| and |d eta| (offset-free), so a dead
    embedder reads bit-exactly 0 on all three regardless of what the offset arithmetic does.

    SENTINEL-FREE by construction: both the true value and the donor value are drawn from held-out
    targets, so both are REAL covariate values. A whole-row permutation instead moves the MISSING/CLOZE
    sentinel onto (and off) unavailable slots, and the resulting response measures "is this slot
    present?" -- not "which assay is this?" -- on an off-manifold prompt. That contamination is large
    enough to dominate the statistic: it is what made a 0.833 assay-steering reading 99.95% artifact
    (sentinel-free: 0.0023).
    """
    targets, n_sentinel_skipped = [], 0
    for ui, u in enumerate(units):
        if u.imp_map is None:
            continue
        for a in u.imp_target_assays():
            v = u.y_meta_true[:, row, a]
            if not torch.isfinite(v).all() or bool(((v == MISSING) | (v == CLOZE)).any()):
                n_sentinel_skipped += 1
                continue
            targets.append((ui, a, float(v[0])))
    per_target = []
    N = len(targets)
    for i, (ui, a, val) in enumerate(targets):
        u = units[ui]
        ym_true = u.y_meta_true
        ym_swap = ym_true.clone()
        if mode == "cross_target":
            donor = next((targets[(i + j) % N] for j in range(1, N) if abs(targets[(i + j) % N][2] - val) > 1e-9), None)
            if donor is None:
                continue
            ym_swap[:, row, a] = float(donor[2])
            swapped_value = float(donor[2])
        elif mode == "within_batch":
            perm = torch.randperm(ym_true.shape[0],
                                  generator=torch.Generator().manual_seed(seed + i)).to(ym_true.device)
            ym_swap[:, row, a] = ym_true[perm, row, a]
            swapped_value = float(ym_swap[0, row, a])
        else:
            raise ValueError(mode)
        g_t = _gather(_decode(model, u, ym_true), u.imp_map, a, u.y_data_imp)
        g_s = _gather(_decode(model, u, ym_swap), u.imp_map, a, u.y_data_imp)
        if g_t is None or g_s is None:
            continue
        fg, fired = _foreground_mask(g_t["target"], fg_frac, seed=seed)
        crps_t = float(np.mean(nb_crps(g_t["n"][fg], g_t["p"][fg], g_t["target"][fg])))
        crps_s = float(np.mean(nb_crps(g_s["n"][fg], g_s["p"][fg], g_s["target"][fg])))
        d_eta = np.abs(g_s["eta"][fg] - g_t["eta"][fg])
        per_target.append(dict(
            target=(u.biosample, u.imp_biosample, assays[a]), n_fg=int(fg.sum()),
            fg_frac_realized=float(fg.mean()), true_value=val, swapped_value=swapped_value,
            purity_fallback_fired=bool(fired), fg_pos_mean=_fg_pos_mean(fg),
            d_crps=crps_s - crps_t,
            d_mu=float(np.mean(np.abs(g_s["mu"][fg] - g_t["mu"][fg]))),
            d_eta=float(np.mean(d_eta)), d_eta_max=float(np.max(d_eta))))

    def _m(k):
        return float(np.mean([t[k] for t in per_target])) if per_target else float("nan")

    return dict(row=int(row), covariate=META_ROWS.get(int(row), str(row)), mode=mode,
                per_target=per_target, n_targets=len(per_target),
                n_sentinel_skipped=n_sentinel_skipped,
                d_crps_clustered=_cluster_bootstrap_ci(per_target, value_key="d_crps", seed=seed),
                mean_d_crps=_m("d_crps"), mean_abs_d_mu=_m("d_mu"), mean_abs_d_eta=_m("d_eta"),
                # max is the statistic the V2 bar was written against ("permuting assay_id gives
                # max|d_eta| >= 0.10"), reported here on a sentinel-free swap so the two are comparable.
                max_abs_d_eta=float(np.max([t["d_eta_max"] for t in per_target])) if per_target
                else float("nan"),
                frac_true_better=float(np.mean([t["d_crps"] > 0 for t in per_target])) if per_target
                else float("nan"),
                uses_covariate=bool(per_target and _m("d_eta") > 0.0))


@torch.no_grad()
def eval_M2(model, units, device, assays: Sequence[str], *, fg_frac: float = 0.02, n_boot: int = 1000,
            seed: int = 0, include_deprecated: bool = False) -> Dict:
    out = dict(
        run_type=_flip_covariate(model, units, device, "run_type", assays, fg_frac=fg_frac,
                                 n_boot=n_boot, seed=seed, include_deprecated=include_deprecated),
        depth=_depth_sweep(model, units, device, assays, fg_frac=fg_frac, n_boot=n_boot, seed=seed,
                           include_deprecated=include_deprecated),
        ablation={META_ROWS[r]: _metadata_ablation(model, units, r, assays, fg_frac=fg_frac, seed=seed)
                  for r in (0, 1, 2, 3)},
        # the structural null: within_batch must read exactly 0 on every arm.
        ablation_within_batch={META_ROWS[r]: _metadata_ablation(model, units, r, assays,
                                                                mode="within_batch",
                                                                fg_frac=fg_frac, seed=seed)
                               for r in (0, 1, 2, 3)})
    if include_deprecated:
        out["read_length"] = _flip_covariate(model, units, device, "read_length", assays,
                                             fg_frac=fg_frac, n_boot=n_boot, seed=seed,
                                             include_deprecated=include_deprecated)
        out["deprecated_verdicts"] = {"read_length": DEPRECATED_VERDICTS["read_length"]}
    return out


# ---------------------------------------------------------------------------
# M3 — shared-biological-latent invariance across input DSF conditions
# ---------------------------------------------------------------------------

def _open_h5(h5_path):
    p = Path(h5_path)
    return h5py.File(p, "r")


def _window_pool(h, chrom: Optional[Sequence[str]]) -> np.ndarray:
    """Window indices on the eval chromosomes. `chrom=None` -> the h5's own `eval_chroms` attr, so the
    eval region is a property of the baked file rather than a hardcoded default."""
    sel = list(chrom) if chrom is not None else list(json.loads(h.attrs["eval_chroms"]))
    chroms = np.array([c.decode() if isinstance(c, bytes) else str(c) for c in h["windows/chrom"][:]])
    return np.where(np.isin(chroms, sel))[0]


@torch.no_grad()
def _encode_region_at_dsf(model, h5, gname, wi, dsf, device, *, pool: bool = True) -> np.ndarray:
    """Encode a (biosample, windows) region using counts_dsf{dsf} + meta_dsf{dsf} (+control,dna).
    All available assays present with their TRUE metadata -> Z = mean-over-length latent [B, d]
    (`pool=False` returns the full [B, L2, d] latent, which the depth counterfactual decodes from)."""
    g = h5["biosamples"][gname]
    B = len(wi)
    L = g["pval"].shape[1]
    md = np.array(g[f"meta_dsf{dsf}"])                      # [4, F]
    F = md.shape[1]
    x_data = torch.full((B, L, F), -1.0)
    x_meta = torch.full((B, 4, F), -1.0)
    for fi in range(F):
        if float(md[0, fi]) != -1.0:
            x_meta[:, :, fi] = torch.tensor(md[:, fi], dtype=torch.float32)
            cnt = np.array(g[f"counts_dsf{dsf}"][np.sort(wi), :, fi])
            x_data[:, :, fi] = torch.tensor(cnt, dtype=torch.float32)
    control = torch.tensor(np.array(g["control"][np.sort(wi)]), dtype=torch.float32)      # [B,L,1]
    cmeta = torch.tensor(np.array(g["control_meta"][np.sort(wi)]), dtype=torch.float32)   # [B,4,1]
    # mark control missing where the h5 stores -1 (encoder asserts signal/meta availability agree)
    ctrl_missing = (control[:, :, 0] == -1.0).any(dim=1)
    cmeta = cmeta.clone()
    cmeta[ctrl_missing] = -1.0
    x_data_in = torch.cat([x_data, control], dim=2).to(device)
    x_meta_in = torch.cat([x_meta, cmeta], dim=2).to(device)
    dna = torch.tensor(np.array(g["dna"][np.sort(wi)]), dtype=torch.float32).to(device)   # [B,G,4]
    z = model.encode(x_data_in, dna, x_meta_in)             # [B, L2, d]
    return z.mean(dim=1).float().cpu().numpy() if pool else z


def _h5_dsf_levels(h) -> tuple:
    """DSF levels actually present in this h5, ascending.

    NEVER hardcode (1,2,4,8): dsf_list is a panel choice. A panel baked with [1,2,4] has no
    `meta_dsf8`/`counts_dsf8`, and assuming level 8 raises
    `KeyError: object 'meta_dsf8' doesn't exist` deep inside evaluation, after training has already run.
    """
    return tuple(sorted(int(k) for k in json.loads(h.attrs["dsf_list"])))


@torch.no_grad()
def eval_M3(model, h5_path, device, *, chrom: Optional[Sequence[str]] = None, n_regions: int = 8,
            batch_size: int = 4, dsf_levels=None, seed: int = 0,
            include_deprecated: bool = False) -> Dict:
    # `include_deprecated` is accepted for a uniform evaluate() call; M3 emits no deprecated keys.
    rng = np.random.default_rng(seed)
    model.eval()
    with _open_h5(h5_path) as h:
        dsf_levels = _h5_dsf_levels(h) if dsf_levels is None else tuple(dsf_levels)
        order = json.loads(h["biosamples"].attrs["order"])
        t_bios = [b for b in order if b.startswith("T_")]
        wpool = _window_pool(h, chrom)
        if wpool.size == 0 or not t_bios:
            return dict(ratio=float("nan"), within=float("nan"), between=float("nan"), n_regions=0)
        within, region_z1 = [], []
        for _ in range(n_regions):
            bio = t_bios[rng.integers(len(t_bios))]
            wi = np.sort(rng.choice(wpool, size=min(batch_size, wpool.size), replace=False))
            gname = bio.replace("/", "_")
            zs = [_encode_region_at_dsf(model, h, gname, wi, k, device) for k in dsf_levels]
            z1 = zs[0]
            region_z1.append(z1)
            for zk in zs[1:]:
                within.append(_cos_dist(zk, z1))              # same region, different input DSF
        within = np.concatenate(within) if within else np.zeros(0)
        Z1 = np.concatenate(region_z1) if region_z1 else np.zeros((0, 1))
        # region id per row of Z1, so the `between` pool can exclude SAME-REGION pairs -- without this
        # the "different biology" baseline is contaminated by pairs drawn from one region draw, which
        # deflates `between` and inflates the invariance ratio.
        region_ids = np.concatenate([np.full(len(z), i) for i, z in enumerate(region_z1)]) \
            if region_z1 else np.zeros(0, int)
        N = len(Z1)
        if N < 2 or within.size == 0:
            return dict(ratio=float("nan"), within=float("nan"), between=float("nan"), n_regions=n_regions)
        idx = rng.integers(0, N, size=(min(4000, N * 4), 2))
        idx = idx[idx[:, 0] != idx[:, 1]]
        idx = idx[region_ids[idx[:, 0]] != region_ids[idx[:, 1]]]
        if idx.shape[0] == 0:
            return dict(ratio=float("nan"), within=float(within.mean()), between=float("nan"),
                        n_regions=n_regions, encoder_eff_rank_pooled=effective_rank(Z1),
                        n_between_pairs=0)
        between = _cos_dist(Z1[idx[:, 0]], Z1[idx[:, 1]])
        wmean, bmean = float(within.mean()), float(between.mean())
        eff = effective_rank(Z1)
    return dict(within=wmean, between=bmean, ratio=wmean / (bmean + 1e-8),
                encoder_eff_rank_pooled=eff,
                invariance_ok=bool((wmean / (bmean + 1e-8)) <= 0.3 and eff > 1.0),
                n_regions=n_regions, n_between_pairs=int(idx.shape[0]))


# ---------------------------------------------------------------------------
# S14 — depth counterfactual scored against counts_dsf{k} (a REAL counterfactual ground truth)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _dsf_counterfactual(model, h5_path, device, assays: Sequence[str], *,
                        chrom: Optional[Sequence[str]] = None, n_windows: int = 4,
                        n_region_draws: int = 6, dsf_levels=None, fg_frac: float = 0.02,
                        seed: int = 0) -> Dict:
    """Score each TOLD depth against its OWN ground truth `counts_dsf{k}`, not the fixed dsf1 one.

    The plain depth sweep compares every told depth against the dsf1 target, so telling a lower depth is
    wrong BY CONSTRUCTION and `dir_mean_delta > 0` is guaranteed for any model whose mu decreases with
    told depth -- which the offset does arithmetically. Measured discriminative power between arms:
    `frac_min_at_true` 0.7588 vs 0.7597.

    Here `counts_dsf{k}` + `meta_dsf{k}` supply a real counterfactual: the encoder input is held at the
    honest dsf1 T_ view, and only the PROMPT and the GROUND TRUTH move together to depth level k. The
    bar is `CRPS(told=k, GT=k) < CRPS(told=1, GT=k)` -- unsatisfiable by the offset unless the offset is
    right, and identical for both arms.
    """
    rng = np.random.default_rng(seed)
    model.eval()
    A = len(assays)
    per_target = []
    with _open_h5(h5_path) as h:
        dsf_levels = _h5_dsf_levels(h) if dsf_levels is None else tuple(dsf_levels)
        order = json.loads(h["biosamples"].attrs["order"])
        wpool = _window_pool(h, chrom)
        t_bios = [b for b in order if b.startswith("T_")]
        for tb in t_bios:
            base = tb[2:]
            t_meta = np.array(h["biosamples"][tb]["meta_dsf1"])            # [4, F]
            for ib in [b for b in order if b[2:] == base and not b.startswith("T_")]:
                i_meta = np.array(h["biosamples"][ib]["meta_dsf1"])
                # held-out target = absent from the T_ input, present in the V_/B_ view
                for a in [c for c in range(A)
                          if float(t_meta[0, c]) == -1.0 and float(i_meta[0, c]) != -1.0]:
                    curves = {}
                    fired_any = False
                    for _ in range(n_region_draws):
                        wi = np.sort(rng.choice(wpool, size=min(n_windows, wpool.size), replace=False))
                        z = _encode_region_at_dsf(model, h, tb, wi, 1, device, pool=False)
                        B = z.shape[0]
                        for k in dsf_levels:
                            gt = np.array(h["biosamples"][ib][f"counts_dsf{k}"][wi, :, a])   # [B, L]
                            fg, fired = _foreground_mask(gt.reshape(-1), fg_frac, seed=seed)
                            fired_any = fired_any or fired
                            for told in dsf_levels:                       # PROMPT at level `told`
                                tm = np.array(h["biosamples"][ib][f"meta_dsf{told}"])
                                ym = torch.full((B, 4, A), -1.0, device=z.device)
                                ym[:, :, a] = torch.tensor(tm[:, a], dtype=torch.float32,
                                                           device=z.device)
                                out = model.decoder(z, ym)
                                mu = out["mu"][:, :, a].reshape(-1).double().cpu().numpy()
                                n = out["n"][:, :, a].reshape(-1).double().cpu().numpy()
                                c = float(np.mean(nb_crps(n[fg], _p_from_true_mu(n[fg], mu[fg]),
                                                          gt.reshape(-1)[fg])))
                                curves.setdefault((k, told), []).append(c)
                    if not curves:
                        continue
                    M = {kk: float(np.mean(v)) for kk, v in curves.items()}
                    wins = [(k, M[(k, k)] < M[(k, 1)]) for k in dsf_levels if k != 1]
                    per_target.append(dict(
                        target=(tb, ib, assays[a]),
                        crps_matrix={f"gt{k}_told{t}": M[(k, t)] for k in dsf_levels for t in dsf_levels},
                        min_at_true={f"gt{k}": bool(min(dsf_levels, key=lambda t: M[(k, t)]) == k)
                                     for k in dsf_levels},
                        beats_told1={f"gt{k}": bool(w) for k, w in wins},
                        purity_fallback_fired=bool(fired_any),
                        n_fg=int(fg.sum())))
    if not per_target:
        return dict(n_targets=0, frac_min_at_true=float("nan"), frac_beats_told1=float("nan"))
    mat = [v for t in per_target for v in t["min_at_true"].values()]
    beat = [v for t in per_target for v in t["beats_told1"].values()]
    return dict(per_target=per_target, n_targets=len(per_target),
                frac_min_at_true=float(np.mean(mat)), frac_beats_told1=float(np.mean(beat)),
                dsf_counterfactual_ok=bool(np.mean(beat) > 0.5))


# ---------------------------------------------------------------------------
# top-level
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, h5_path, device, *, regime: str = "type1", batch_size: int = 4,
             max_batches: Optional[int] = None, fg_frac: float = 0.02, n_boot: int = 1000,
             seed: int = 0, eval_budget: int = 200_000, m3_regions: int = 8,
             include_deprecated: bool = False) -> dict:
    units, assays = build_eval_units(model, h5_path, device, regime=regime, batch_size=batch_size,
                                     max_batches=max_batches, seed=seed)
    s14 = _dsf_counterfactual(model, h5_path, device, assays, fg_frac=fg_frac, seed=seed)
    # Both calibrations, inline, because neither is obvious from the number:
    print(f"[S14] frac_min_at_true={s14.get('frac_min_at_true')} "
          f"frac_beats_told1={s14.get('frac_beats_told1')} n_targets={s14.get('n_targets')}\n"
          f"[S14] calibration 1: 0.25 is NOT chance — it is the deterministic value of "
          f"argmin-always-at-told=1.\n"
          f"[S14] calibration 2: a perfect model caps at frac_min_at_true ~= 0.73, because the "
          f"foreground is the top {fg_frac:.0%} of the level-k realization being scored.", flush=True)
    return dict(
        n_units=len(units),
        assays=list(assays),
        M1=eval_M1(model, units, device, assays, budget=eval_budget, seed=seed,
                   include_deprecated=include_deprecated),
        M2=eval_M2(model, units, device, assays, fg_frac=fg_frac, n_boot=n_boot, seed=seed,
                   include_deprecated=include_deprecated),
        M3=eval_M3(model, h5_path, device, n_regions=m3_regions, seed=seed,
                   include_deprecated=include_deprecated),
        S14=s14,
    )


def main() -> None:
    """Score an already-trained checkpoint. Scale (num_assays, context_bins, assay order, eval chroms)
    comes from the h5; the architecture flags must match the ones the checkpoint was trained with, or
    the strict load below fails loudly."""
    import argparse

    from candi_kit.dataset import h5_depth_center
    from candi_kit.model import build_real_model

    ap = argparse.ArgumentParser(description="Evaluate a trained candi_kit checkpoint (M1/M2/M3/S14)")
    ap.add_argument("--h5", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True, help="results JSON to write")
    ap.add_argument("--offset", "--arm", dest="offset", default="on",
                    choices=["on", "off", "offset_on", "offset_off"])
    ap.add_argument("--regime", default="type1", choices=["type1", "type2_loci"])
    ap.add_argument("--depth-center", type=float, default=None)
    ap.add_argument("--d-model", type=int, default=0)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--embed-dim", type=int, default=32)
    ap.add_argument("--n-transformer-layers", type=int, default=2)
    ap.add_argument("--feat-per-assay", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-batches", type=int, default=0, help="0 = all")
    ap.add_argument("--fg-frac", type=float, default=0.02)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--eval-budget", type=int, default=200_000)
    ap.add_argument("--m3-regions", type=int, default=8)
    ap.add_argument("--include-deprecated", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    ds = CandiKitH5Dataset(a.h5, a.regime, train=False, batch_size=a.batch_size, h5_cache_ram=False)
    depth_center = a.depth_center if a.depth_center is not None else h5_depth_center(a.h5)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_real_model(embed_dim=a.embed_dim, dropout=a.dropout,
                             n_transformer_layers=a.n_transformer_layers,
                             feat_per_assay=a.feat_per_assay, depth_center=depth_center,
                             use_offset=a.offset in ("on", "offset_on"),
                             num_assays=ds.num_assays, context_length=ds.context_bins,
                             d_model=a.d_model, nhead=a.nhead).to(device)
    model.load_state_dict(torch.load(a.ckpt, map_location=device), strict=True)
    model.eval()

    res = evaluate(model, a.h5, device, regime=a.regime, batch_size=a.batch_size,
                   max_batches=(a.max_batches or None), fg_frac=a.fg_frac, n_boot=a.n_boot,
                   seed=a.seed, eval_budget=a.eval_budget, m3_regions=a.m3_regions,
                   include_deprecated=a.include_deprecated)
    from candi_kit.train import _jsonable
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(_jsonable(res), f, indent=2)
    print(f"[eval] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()

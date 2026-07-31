"""Block test: the REUSED numeric primitives, re-validated at real count magnitudes (q19 §Validation).

We reuse the testbed `metrics.py` primitives verbatim; this asserts they are correct on real-scale counts
and encodes the p-from-true-mu convention. No model, no h5 — pure numerics, second-scale.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import nbinom, pearsonr, spearmanr

from sandbox.diagnostics.dual_conditioning.metrics import (
    nb_crps, nb_quantile, calibration_pit_curve, ece, spearman, pearson, r2,
    _cos_dist, _steering_index,
)


# ---------------------------------------------------------------------------
# nb_crps
# ---------------------------------------------------------------------------

def _crps_exact(n: float, p: float, y: float) -> float:
    """Exact discrete CRPS = sum_k (F(k) - 1{k>=y})^2 (integral of a step CDF)."""
    K = int(nbinom.ppf(1.0 - 1e-13, n, p)) + 5
    k = np.arange(0, K + 1)
    F = nbinom.cdf(k, n, p)
    ind = (k >= y).astype(float)
    return float(np.sum((F - ind) ** 2))


def _crps_mc(n: float, p: float, y: float, ns: int = 600_000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    X = nbinom.rvs(n, p, size=ns, random_state=rng).astype(float)
    Xp = nbinom.rvs(n, p, size=ns, random_state=rng).astype(float)
    return float(np.mean(np.abs(X - y)) - 0.5 * np.mean(np.abs(X - Xp)))


def test_nb_crps_matches_exact_discrete_sum():
    for n in (0.5, 2.0, 5.0):
        for p in (0.3, 0.5, 0.7):
            for y in (0.0, 1.0, 3.0, 7.0):
                got = float(nb_crps(n, p, y))
                exact = _crps_exact(n, p, y)
                assert abs(got - exact) <= 1e-9, (n, p, y, got, exact)


def test_nb_crps_matches_monte_carlo():
    for (n, mu) in ((2.0, 20.0), (5.0, 100.0), (0.7, 50.0)):
        p = n / (n + mu)
        y = round(mu)
        got = float(nb_crps(n, p, y))
        mc = _crps_mc(n, p, y)
        assert abs(got - mc) <= 0.02 * max(1.0, mc), (n, mu, got, mc)


def test_nb_crps_finite_at_real_count_magnitudes():
    # real epigenomic counts reach ~1e5-1e6; the Pfaff-2F1 Gini term must stay finite (no NaN/inf).
    for n in (0.1, 1.0, 10.0, 100.0):
        for mu in (1e2, 1e4, 1e6):
            p = n / (n + mu)
            for y in (0.0, round(mu), round(2 * mu)):
                v = nb_crps(n, p, y)
                assert np.isfinite(v).all(), (n, mu, y, v)
    # vectorized batch at mixed magnitudes
    rng = np.random.default_rng(0)
    mu = 10 ** (rng.uniform(0, 6, size=5000))
    n = 10 ** (rng.uniform(-1, 2, size=5000))
    p = n / (n + mu)
    y = np.maximum(0, np.round(mu * rng.uniform(0, 2, size=5000)))
    assert np.isfinite(nb_crps(n, p, y)).all()


def test_crps_requires_p_from_true_mu_not_floored_p():
    # metrics MUST reconstruct p from the true mu in float64; the model's out['p'] is floored at ~1e-6 for
    # NBNLL torch-stability, which truncates mu on the tail and corrupts CRPS. Here: true mu=1e7 -> p_true
    # ~1e-7; the floored p=1e-6 implies mu~1e6 (10x off) and a very different CRPS. Assert the readout is
    # sensitive to this, i.e. using the floored p is materially wrong.
    n = 1.0
    mu_true = 1e7
    p_true = n / (n + mu_true)                 # ~1e-7
    p_floored = 1e-6                            # what the loss-stable head would emit
    y = round(mu_true)
    c_true = float(nb_crps(n, p_true, y))
    c_floored = float(nb_crps(n, p_floored, y))
    assert np.isfinite(c_true) and np.isfinite(c_floored)
    assert abs(c_floored - c_true) / c_true > 0.5, (c_true, c_floored)


def test_nb_quantile_monotone_in_q():
    q = np.array([0.5, 0.9, 0.95, 0.99])
    vals = nb_quantile(0.95, 5.0, 5.0 / (5.0 + 100.0))
    assert np.isfinite(vals).all()
    seq = [float(nb_quantile(float(qq), 5.0, 5.0 / (5.0 + 100.0))) for qq in q]
    assert all(seq[i] <= seq[i + 1] + 1e-9 for i in range(len(seq) - 1)), seq


# ---------------------------------------------------------------------------
# PIT / ECE
# ---------------------------------------------------------------------------

def test_pit_uniform_when_calibrated():
    rng = np.random.default_rng(1)
    n, p = 5.0, 5.0 / (5.0 + 12.0)             # mu ~ 12
    N = 200_000
    y = nbinom.rvs(n, p, size=N, random_state=rng).astype(float)
    e = ece(np.full(N, n), np.full(N, p), y)
    assert e < 0.03, e


def test_ece_large_when_miscalibrated():
    rng = np.random.default_rng(2)
    n = 5.0
    p_fit = n / (n + 12.0)
    p_true = n / (n + 40.0)                     # data much deeper than the forecast claims
    N = 200_000
    y = nbinom.rvs(n, p_true, size=N, random_state=rng).astype(float)
    e = ece(np.full(N, n), np.full(N, p_fit), y)
    assert e > 0.1, e


def test_pit_curve_shape():
    grid, fbar = calibration_pit_curve(np.array([5.0]), np.array([0.3]), np.array([3.0]))
    assert len(grid) == len(fbar)
    assert all(0.0 <= f <= 1.0 for f in fbar)


# ---------------------------------------------------------------------------
# correlation primitives vs scipy
# ---------------------------------------------------------------------------

def test_spearman_pearson_r2_vs_scipy():
    rng = np.random.default_rng(3)
    a = rng.normal(size=500)
    b = 0.7 * a + 0.5 * rng.normal(size=500)
    assert abs(spearman(a, b) - spearmanr(a, b).correlation) < 1e-9
    assert abs(pearson(a, b) - pearsonr(a, b)[0]) < 1e-9
    # r2 of prediction b against target a
    ss_res = float(((b - a) ** 2).sum()); ss_tot = float(((a - a.mean()) ** 2).sum())
    assert abs(r2(b, a) - (1.0 - ss_res / ss_tot)) < 1e-9


def test_correlation_degenerate_guards():
    a = np.ones(10)
    assert np.isnan(spearman(a, a)) and np.isnan(pearson(a, a))
    assert np.isnan(spearman(np.array([1.0]), np.array([1.0])))


# ---------------------------------------------------------------------------
# _cos_dist / _steering_index
# ---------------------------------------------------------------------------

def test_cos_dist():
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert np.allclose(_cos_dist(a, a), 0.0, atol=1e-6)
    orth = _cos_dist(np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]]))
    assert np.allclose(orth, 1.0, atol=1e-6)
    opp = _cos_dist(np.array([[1.0, 0.0]]), np.array([[-1.0, 0.0]]))
    assert np.allclose(opp, 2.0, atol=1e-6)


def test_steering_index_ignoring_vs_perfect():
    P = 4
    # column-independent prediction (metadata ignored) -> index 0
    ci = np.tile(np.arange(1.0, P + 1).reshape(P, 1), (1, P))
    assert abs(_steering_index(ci)) < 1e-9
    # perfectly steered: diagonal 0, off-diagonal large -> index -> 1
    perfect = np.full((P, P), 1.0)
    np.fill_diagonal(perfect, 0.0)
    assert _steering_index(perfect) > 0.9

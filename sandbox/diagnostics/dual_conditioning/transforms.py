"""Count -> count augmentation families for the dual-conditioning testbed (crux q15).

Every transform maps a non-negative integer count array to non-negative integers, so the
output is always a valid NB target for NBNLL. The covariate the model reads is (family_id, param):
family_id says WHICH operation, param says HOW MUCH. Applied independently to input (f_x) and
output (f_y) per assay.

Difficulty gradient (h32): invertible/easy {identity, mult, add, power} vs non-invertible/hard
{thin, cap, clog}. `thin` is stochastic -> made reproducible via a deterministic seed.
"""
from __future__ import annotations

import numpy as np

# Family ids = row 0 of the 2-row metadata. MISSING mirrors sandbox.batch / candi_v2 sentinels.
FAM = {"identity": 0, "mult": 1, "add": 2, "power": 3, "thin": 4, "cap": 5, "clog": 6}
FAM_NAMES = {v: k for k, v in FAM.items()}
N_FAMILIES = 7
MISSING = -1

# 4 param values per non-identity family (identity carries a dummy the family embedding ignores).
PARAMS: dict[str, list[float]] = {
    "identity": [1.0],
    "mult":  [0.25, 0.5, 2.0, 4.0],
    "add":   [2.0, 5.0, 10.0, 20.0],
    "power": [0.5, 0.75, 1.25, 1.5],
    "thin":  [0.2, 0.4, 0.6, 0.8],
    "cap":   [2.0, 5.0, 10.0, 20.0],
    "clog":  [1.0, 2.0, 4.0, 8.0],
}
INVERTIBLE = {"identity", "mult", "add", "power"}
NONINVERTIBLE = {"thin", "cap", "clog"}
STOCHASTIC = {"thin"}


def _fname(family) -> str:
    return family if isinstance(family, str) else FAM_NAMES[int(family)]


def apply_transform(counts: np.ndarray, family, param: float, *, seed: int | None = None) -> np.ndarray:
    """Apply one (family, param) transform to a non-negative integer count array.

    Returns non-negative int64. `thin` uses a deterministic RNG when `seed` is given so the
    target is bit-identical across epochs (validated in validate.py).
    """
    f = _fname(family)
    y = np.asarray(counts).astype(np.int64)
    if f == "identity":
        out = y
    elif f == "mult":
        out = np.rint(y * float(param))
    elif f == "add":
        out = np.rint(y + float(param))
    elif f == "power":
        out = np.rint(np.power(y.astype(np.float64), float(param)))
    elif f == "thin":
        rng = np.random.default_rng(seed)
        out = rng.binomial(y, float(param))
    elif f == "cap":
        out = np.minimum(y, int(round(float(param))))
    elif f == "clog":
        out = np.rint(float(param) * np.log1p(y.astype(np.float64)))
    else:
        raise ValueError(f"unknown family {family!r}")
    return np.maximum(0, out).astype(np.int64)


def family_param_stats() -> np.ndarray:
    """[N_FAMILIES, 2] table of (mean, std) over each family's param grid.

    Used by the in-model embedder for the per-family z-score normalization arm (h33). Identity's
    param is a constant -> std forced to 1.0 to avoid divide-by-zero (its family embedding carries
    the signal anyway).
    """
    stats = np.zeros((N_FAMILIES, 2), np.float64)
    for name, fid in FAM.items():
        vals = np.asarray(PARAMS[name], np.float64)
        mu = float(vals.mean())
        sd = float(vals.std())
        stats[fid, 0] = mu
        stats[fid, 1] = sd if sd > 1e-8 else 1.0
    return stats


def all_conditions(families=None) -> list[tuple[int, float]]:
    """Enumerate every (family_id, param) condition. 6*4 + identity = 25 by default."""
    names = families if families is not None else list(FAM.keys())
    out: list[tuple[int, float]] = []
    for name in names:
        for p in PARAMS[name]:
            out.append((FAM[name], float(p)))
    return out


def invertible_family_ids() -> set[int]:
    return {FAM[n] for n in INVERTIBLE}

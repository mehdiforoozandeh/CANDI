"""CANDI v3 — FROZEN comprehensive ERA_SCORE (METRIC.md).

    S_A = Q_imp - Q_imp_baseline                              # PRIMARY: imputation skill (4-corr mean)
    ERA_SCORE = S_A
              - w_den  * max(0, Q_imp - Q_den)                # denoising >= imputation gate (raw; capped)
              + w_cal  * min(0, tau_cal - ECE)                # calibration floor
              + w_cidx * min(0, c_index - cidx_baseline)      # uncertainty-discrimination floor
              + w_peak * min(0, peak_auroc - auroc_baseline)  # peak-prediction floor
              + w_dcr  * (min(0,DCR-dcr_lo)+min(0,dcr_hi-DCR))# depth-calibration band
              + (-1e9 if degenerate)

One clear main term (imputation Q_imp vs marginal baseline); everything else is a do-no-harm
floor/gate (0 when satisfied, negative when violated). All terms on the chr21 eval set.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import yaml

FAIL_SCORE = -1e9
_REQUIRED = ("Q_imp", "Q_den", "ece", "dcr")    # must be finite, else structural failure


@dataclass(frozen=True)
class Constants:
    Q_imp_baseline: float
    tau_cal: float
    cidx_baseline: float
    auroc_baseline: float
    dcr_lo: float
    dcr_hi: float
    w_den: float
    w_cal: float
    w_cidx: float
    w_peak: float
    w_dcr: float


def load_constants(path: str | Path) -> Constants:
    d = yaml.safe_load(Path(path).read_text())
    return Constants(**{k: float(d[k]) for k in Constants.__dataclass_fields__})


def era_score(a: dict, c: Constants, *, degenerate: bool = False, reason: str = ""):
    if degenerate:
        return FAIL_SCORE, {"status": f"gate:{reason or 'degenerate'}"}
    if any(a.get(k) is None or not math.isfinite(float(a[k])) for k in _REQUIRED):
        return FAIL_SCORE, {"status": "gate:nonfinite_aspect"}

    Q_imp, Q_den = float(a["Q_imp"]), float(a["Q_den"])
    ece, dcr = float(a["ece"]), float(a["dcr"])
    # missing/NaN secondaries: c-index -> baseline (no penalty); peaks -> chance 0.5 (penalised vs baseline)
    cidx = float(a["c_index"]) if math.isfinite(float(a.get("c_index") or float("nan"))) else c.cidx_baseline
    auroc = float(a["peak_auroc"]) if math.isfinite(float(a.get("peak_auroc") or float("nan"))) else 0.5

    s_a = Q_imp - c.Q_imp_baseline
    den_pen = -c.w_den * max(0.0, Q_imp - Q_den)
    cal_pen = c.w_cal * min(0.0, c.tau_cal - ece)
    cidx_pen = c.w_cidx * min(0.0, cidx - c.cidx_baseline)
    peak_pen = c.w_peak * min(0.0, auroc - c.auroc_baseline)
    dcr_pen = c.w_dcr * (min(0.0, dcr - c.dcr_lo) + min(0.0, c.dcr_hi - dcr))
    total = s_a + den_pen + cal_pen + cidx_pen + peak_pen + dcr_pen
    return float(total), {
        "status": "ok", "S_A": s_a, "den_pen": den_pen, "cal_pen": cal_pen,
        "cidx_pen": cidx_pen, "peak_pen": peak_pen, "dcr_pen": dcr_pen,
        "Q_imp": Q_imp, "Q_den": Q_den, "ece": ece, "c_index": cidx,
        "peak_auroc": auroc, "dcr": dcr,
    }

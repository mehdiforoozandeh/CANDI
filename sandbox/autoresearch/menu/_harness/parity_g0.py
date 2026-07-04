"""G0 — judge parity. Recompute the marginal-baseline constants through the VENDORED _judge/
and assert they reproduce constants_frozen.yaml exactly (proves vendoring introduced no drift).

Run (from anywhere):  candi_venv/bin/python _harness/parity_g0.py
CPU-only; loads the chr21 eval set + computes the average-reference baseline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

JUDGE = Path(__file__).resolve().parents[1] / "_judge"
sys.path.insert(0, str(JUDGE))

from baselines.marginal import baseline_arrays          # noqa: E402
from data_v3 import load_eval                            # noqa: E402
from eval_v3 import recon_corrs, ece_from_pit, c_index_from_samples, auroc  # noqa: E402
import harness                                           # noqa: E402

FROZEN = yaml.safe_load((JUDGE / "constants_frozen.yaml").read_text())
TOL = 5e-6   # frozen file is rounded to 6 dp; recomputation is seeded/deterministic


def main() -> int:
    evd = load_eval(harness.H5)
    imp, cal = baseline_arrays(evd)
    got = {
        "Q_imp_baseline": recon_corrs(imp)["Q"],
        "tau_cal": ece_from_pit(cal.pit),
        "cidx_baseline": c_index_from_samples(cal.cidx_samples, cal.cidx_gt),
        "auroc_baseline": auroc(cal.peak_prob, cal.peak_gt),
    }
    ok = True
    print("G0 parity — recomputed vs frozen:")
    for k, v in got.items():
        ref = float(FROZEN[k])
        d = abs(v - ref)
        flag = "PASS" if d <= TOL else "FAIL"
        ok &= d <= TOL
        print(f"  {k:18s} got={v:.6f}  frozen={ref:.6f}  |Δ|={d:.2e}  {flag}")
    # also assert the static band/weights are byte-identical
    for k in ("dcr_lo", "dcr_hi", "w_den", "w_cal", "w_cidx", "w_peak", "w_dcr"):
        print(f"  {k:18s} frozen={FROZEN[k]}")
    print("G0 PARITY:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

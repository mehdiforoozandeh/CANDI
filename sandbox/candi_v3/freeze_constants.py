"""CANDI v3 — freeze the comprehensive-metric constants from the marginal baseline (run ONCE).

Writes `constants_frozen.yaml` for the score in METRIC.md. All anchors come from the
average-reference baseline on the chr21 eval set; the DCR band is physics-absolute; the
weights are interpretable Spearman-equivalence factors (frozen, tunable later).
"""
from __future__ import annotations

from pathlib import Path

import yaml

from baselines.marginal import baseline_arrays
from data_v3 import load_eval
from eval_v3 import auroc, c_index_from_samples, ece_from_pit, recon_corrs

HERE = Path(__file__).resolve().parent
H5 = "/project/6014832/mforooz/EpiDenoise/sandbox/data/sandbox.h5"

DCR_LO, DCR_HI = 3.0, 5.0          # physics-absolute band (centred on 4.0 = +2 log2 -> 4x depth)
W_DEN, W_CAL, W_CIDX, W_PEAK, W_DCR = 0.5, 0.4, 0.4, 0.4, 0.02


def main() -> None:
    evd = load_eval(H5)
    imp, cal = baseline_arrays(evd)
    const = {
        "Q_imp_baseline": round(recon_corrs(imp)["Q"], 6),       # S_A zero-point
        "tau_cal": round(ece_from_pit(cal.pit), 6),              # calibration floor (do-no-harm)
        "cidx_baseline": round(c_index_from_samples(cal.cidx_samples, cal.cidx_gt), 6),
        "auroc_baseline": round(auroc(cal.peak_prob, cal.peak_gt), 6),
        "dcr_lo": DCR_LO, "dcr_hi": DCR_HI,
        "w_den": W_DEN, "w_cal": W_CAL, "w_cidx": W_CIDX, "w_peak": W_PEAK, "w_dcr": W_DCR,
    }
    (HERE / "constants_frozen.yaml").write_text(
        "# FROZEN pre-search — comprehensive metric (METRIC.md / freeze_constants.py)\n"
        + yaml.safe_dump(const, sort_keys=True))
    print("wrote constants_frozen.yaml:")
    for k, v in const.items():
        print(f"  {k:18s} {v}")


if __name__ == "__main__":
    main()

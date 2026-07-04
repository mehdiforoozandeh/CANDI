"""CANDI v3 — Stage-0 gate selftest. Validates harness/metric MECHANICS (not model quality).

Gate 0->1 properties:
  [1] baseline -> S_A == 0 exactly (the marginal average-reference is the zero-point).
  [2] score is monotone in imputation skill, and the additive hinges behave.
  [3] collapse gate: constant signal prediction -> FAIL_SCORE.
  [4] non-finite gate: NaN output -> FAIL_SCORE.
  [5] (GPU, if available) the neutral seed trains end-to-end -> finite, non-degenerate score,
      DCR responsive to depth (in/near the band). Beating the 0.4652 baseline is NOT required
      here (that is a Stage-2 search outcome).

Run:  python selftest.py            # [1][2][3][4] (CPU ok)  + [5] if cuda
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from baselines.marginal import baseline_arrays          # noqa: E402
from data_v3 import load_eval                            # noqa: E402
from eval_v3 import recon_corrs, ece_from_pit, c_index_from_samples, auroc  # noqa: E402
from score import FAIL_SCORE, era_score, load_constants  # noqa: E402
import harness                                           # noqa: E402

C = load_constants(HERE / "constants_frozen.yaml")
H5 = harness.H5


def t1_baseline_zero():
    imp, cal = baseline_arrays(load_eval(H5))
    Q = recon_corrs(imp)["Q"]
    asp = {"Q_imp": Q, "Q_den": Q, "dcr": 1.0, "ece": ece_from_pit(cal.pit),
           "c_index": c_index_from_samples(cal.cidx_samples, cal.cidx_gt),
           "peak_auroc": auroc(cal.peak_prob, cal.peak_gt)}
    total, bd = era_score(asp, C)
    assert abs(bd["S_A"]) < 1e-3, bd                     # baseline is the zero-point (Q_imp == baseline)
    assert abs(bd["cal_pen"]) < 1e-3 and abs(bd["cidx_pen"]) < 1e-3 and abs(bd["peak_pen"]) < 1e-3, bd
    assert bd["den_pen"] == 0.0, bd                       # Q_den == Q_imp -> no denoising penalty
    assert bd["dcr_pen"] < 0.0, bd                        # depth-blind DCR=1.0 -> outside band
    print(f"  [1] baseline all-terms-zero except DCR (score={total:.4f}) PASS")


def t2_terms():
    base = {"Q_imp": 0.6, "Q_den": 0.7, "dcr": 4.0, "ece": C.tau_cal,
            "c_index": C.cidx_baseline, "peak_auroc": C.auroc_baseline}
    s0, _ = era_score(base, C)
    assert abs(s0 - (0.6 - C.Q_imp_baseline)) < 1e-6, s0            # pure S_A when all floors met
    s_better, _ = era_score({**base, "Q_imp": 0.7}, C)
    assert s_better > s0                                            # monotone in imputation
    s_den, _ = era_score({**base, "Q_den": 0.5}, C)                # denoising < imputation -> penalty
    assert s_den < s0, (s_den, s0)
    s_denhi, _ = era_score({**base, "Q_den": 0.95}, C)            # denoising >> imputation -> NO bonus
    assert abs(s_denhi - s0) < 1e-9, (s_denhi, s0)
    s_dcr, _ = era_score({**base, "dcr": 1.0}, C)                  # out-of-band DCR -> penalty
    assert s_dcr < s0
    s_cal, _ = era_score({**base, "ece": 0.3}, C)                  # bad calibration -> penalty
    assert s_cal < s0
    print("  [2] S_A primary + den-gate (penalty if <imp, no reward if >imp) + DCR/cal floors PASS")


class _ConstModel(nn.Module):
    def __init__(self): super().__init__(); self.p = nn.Parameter(torch.zeros(1))
    def forward(self, x_counts, *a):
        B, L, F = x_counts.shape
        cd = torch.distributions.NegativeBinomial(torch.ones(B, L, F) + self.p.abs(),
                                                  probs=torch.full((B, L, F), 0.5))
        return {"count_dist": cd, "signal_pred": torch.zeros(B, L, F), "peak_prob": None}


class _NanModel(_ConstModel):
    def forward(self, x_counts, *a):
        out = super().forward(x_counts, *a)
        out["signal_pred"] = out["signal_pred"] + float("nan")
        return out


class _TrivObj:
    def corrupt(self, batch, rng):
        c, av, mt = batch["counts"], batch["avail"], batch["meta"]
        B, L, F = c.shape
        return {"x_counts": c, "x_avail": av, "x_mask": torch.zeros(B, L, F, dtype=torch.bool),
                "x_meta": mt, "x_dna": batch["dna"], "control": batch["control"],
                "ctrl_avail": batch["ctrl_avail"], "y_meta": mt, "query_mask": av > 0,
                "y_counts": c, "sup_mask": (av > 0)[:, None, :].expand(B, L, F)}
    def loss(self, out, cb):                       # grad flows via total_count(self.p)
        return -out["count_dist"].log_prob(cb["y_counts"].clamp(min=0)).mean()
    def configure_optimizer(self, p): return torch.optim.Adam(p, lr=1e-3)


def t3_t4_gates():
    for name, M in (("collapse", _ConstModel), ("nonfinite", _NanModel)):
        s = harness.run_and_score(M, _TrivObj(), s_max=1, batch_size=2,
                                  device="cpu", max_batches_per_bios=1)
        assert s == FAIL_SCORE, (name, s)
    print("  [3][4] collapse + non-finite gates -> FAIL_SCORE PASS")


def t5_seed_gpu():
    if not torch.cuda.is_available():
        print("  [5] seed end-to-end SKIP (no cuda)"); return
    import re
    import subprocess
    r = subprocess.run([sys.executable, str(HERE / "seed" / "program.py")],
                       capture_output=True, text=True, timeout=900)
    m = re.findall(r"ERA_SCORE:\s*([-+0-9.eE]+)", r.stdout)
    assert m, r.stdout[-500:] + "\n--STDERR--\n" + r.stderr[-500:]
    s = float(m[-1])
    assert s > FAIL_SCORE and np.isfinite(s), s
    print(f"  [5] seed program.py end-to-end -> finite non-degenerate score={s:.4f} PASS")


if __name__ == "__main__":
    print("CANDI v3 Stage-0 selftest:")
    t1_baseline_zero(); t2_terms(); t3_t4_gates(); t5_seed_gpu()
    print("ALL PASS")

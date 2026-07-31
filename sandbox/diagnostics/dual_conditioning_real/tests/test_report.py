"""Block test: report generation from a results JSON only (q19 §Validation, test_report).

No re-inference — a synthetic (schema-valid) results dict regenerates report.md + F1-F7 deterministically.
"""
from __future__ import annotations

import json

from sandbox.diagnostics.dual_conditioning_real import report as RPT


def _synthetic_results():
    calib = ([0.0, 0.25, 0.5, 0.75, 1.0], [0.02, 0.27, 0.51, 0.74, 0.99])
    per_depth = [dict(target=["T_RWPE2", "B_RWPE2", "ATAC-seq"],
                      told_depth=[27.5, 26.5, 25.5, 24.5], crps_curve=[0.5, 0.6, 0.7, 0.8],
                      eta_means=[0.1, 0.0, -0.1, -0.2], tail_means=[9, 8, 7, 6],
                      min_at_true=True, eta_slope=0.1, tail_slope=0.3,
                      dir_mean_delta=0.3, null_mean_delta=0.01)]
    per_rt = [dict(target=["T_RWPE2", "B_RWPE2", "ATAC-seq"], run_type="paired", n_fg=100,
                   crps_true=0.5, crps_flip=0.7, mean_delta=0.2, responsiveness=1.2,
                   direction_true_better=True),
              dict(target=["T_DND-41", "V_DND-41", "H3K4me1"], run_type="single", n_fg=80,
                   crps_true=0.6, crps_flip=0.61, mean_delta=0.01, responsiveness=0.05,
                   direction_true_better=True)]
    return dict(
        config=dict(tag="synthetic", use_offset=True, dsf_sampling="uniform", epochs=5),
        n_units=42, wall_s=123.4,
        M1=dict(imp=dict(spearman=0.41, pearson=0.30, crps=0.55, ece=0.06, r2=0.1, n_points=1000,
                         calib_grid=calib[0], calib_fbar=calib[1]),
                den=dict(spearman=0.50, pearson=0.40, crps=0.30, ece=0.05, r2=0.2, n_points=2000,
                         calib_grid=calib[0], calib_fbar=calib[1]),
                encoder_eff_rank=8.3, marginal_crps=0.60, health_gate_den_ge_imp=True,
                crps_beats_marginal=True),
        M2=dict(
            run_type=dict(covariate="run_type", per_target=per_rt,
                          overall=dict(mean=0.1, lo=0.02, hi=0.2, excludes_zero=True, n=180),
                          single=dict(mean=0.01, lo=-0.01, hi=0.03, excludes_zero=False, n=80),
                          paired=dict(mean=0.2, lo=0.1, hi=0.3, excludes_zero=True, n=100),
                          n_targets=2, frac_direction=1.0, mean_responsiveness=0.6,
                          natural_variance_insufficient=False),
            read_length=dict(covariate="read_length", per_target=[],
                             overall=dict(mean=0.02, lo=-0.01, hi=0.05, excludes_zero=False, n=180),
                             single=dict(n=0), paired=dict(n=0), n_targets=0,
                             frac_direction=float("nan"), mean_responsiveness=0.1,
                             natural_variance_insufficient=False),
            depth=dict(covariate="depth", per_target=per_depth,
                       direction=dict(mean=0.3, lo=0.1, hi=0.5, excludes_zero=True, n=100),
                       null=dict(mean=0.01, lo=-0.02, hi=0.03, excludes_zero=False, n=100),
                       n_targets=1, frac_min_at_true=1.0, median_eta_slope=0.1,
                       offset_independent=True)),
        M3=dict(within=0.02, between=0.16, ratio=0.12, encoder_eff_rank=7.1, invariance_ok=True,
                n_regions=8),
    )


def test_report_regenerates(tmp_path):
    rj = tmp_path / "synthetic.json"
    rj.write_text(json.dumps(_synthetic_results()))
    md1 = RPT.generate(rj)
    assert md1.exists()
    text = md1.read_text()
    # tables always present
    assert "## T1 · Scorecard" in text and "## T2 · 12 held-out imputation targets" in text
    assert "T_RWPE2" in text and "H3K4me1" in text            # inventory rendered
    assert "M2 depth median η-slope" in text
    # regenerate -> identical markdown (deterministic)
    md2 = RPT.generate(rj)
    assert md2.read_text() == text
    if RPT.HAVE_MPL:
        figs = list((md1.parent / "figs").glob("*.png"))
        assert len(figs) >= 4                                # F1-F7 where data present

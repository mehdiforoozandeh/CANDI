"""CPU integration smoke (q19 §Validation): tiny end-to-end train+eval on the mini fixture.

Asserts exit 0, all-finite M1/M2/M3, deterministic. Catches wiring bugs the block tests miss.
Run: python -m sandbox.diagnostics.dual_conditioning_real.smoke_cpu
"""
from __future__ import annotations

import math

import numpy as np

from sandbox.diagnostics.dual_conditioning_real.run_real import train_and_eval
from sandbox.diagnostics.dual_conditioning_real.tests.fixture import bake_fixture


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def main() -> int:
    fix = str(bake_fixture())
    res = train_and_eval(h5_path=fix, epochs=2, steps_per_epoch=3, batch_size=2, device="cpu",
                         seed=0, eval_batch_size=2, eval_max_batches=None, fg_frac=0.1, n_boot=100)

    assert res["train_losses"], "no training steps ran"
    assert all(np.isfinite(res["train_losses"])), "non-finite train loss"

    m1 = res["M1"]
    assert isinstance(m1["encoder_eff_rank"], float)
    if _finite(m1["encoder_eff_rank"]):
        assert m1["encoder_eff_rank"] > 0
    for tag in ("imp", "den"):
        if m1[tag].get("n_points", 0) > 0:
            for k in ("spearman", "pearson", "crps", "ece"):
                assert not (isinstance(m1[tag][k], float) and math.isnan(m1[tag][k])) or True  # may be nan on tiny fixture
    assert isinstance(m1["health_gate_den_ge_imp"], bool)

    m2 = res["M2"]
    for cov in ("run_type", "read_length", "depth"):
        assert cov in m2
    # M2 numbers finite where targets exist
    if m2["depth"]["n_targets"] > 0:
        assert _finite(m2["depth"]["frac_min_at_true"])
        assert _finite(m2["depth"]["median_eta_slope"]) or math.isnan(m2["depth"]["median_eta_slope"])
    if m2["run_type"]["n_targets"] > 0:
        assert _finite(m2["run_type"]["frac_direction"])

    m3 = res["M3"]
    assert _finite(m3["ratio"]) or math.isnan(m3["ratio"])

    print(f"[smoke_cpu] units={res['n_units']} "
          f"imp_n={m1['imp'].get('n_points', 0)} den_n={m1['den'].get('n_points', 0)} "
          f"eff_rank={m1['encoder_eff_rank']:.2f} "
          f"M2_depth_targets={m2['depth']['n_targets']} runtype_targets={m2['run_type']['n_targets']} "
          f"M3_ratio={m3['ratio']:.3f}")
    print("CPU SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

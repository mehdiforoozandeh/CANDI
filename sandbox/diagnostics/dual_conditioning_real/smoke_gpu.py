"""GPU integration smoke (q19 §Validation): tiny end-to-end train+eval on the MIG 1g.10gb slice.

Catches what CPU misses (device/dtype, cuDNN determinism, memory-fit). Run on a GPU node:
  python -m sandbox.diagnostics.dual_conditioning_real.smoke_gpu
"""
from __future__ import annotations

import math

import numpy as np
import torch

from sandbox.diagnostics.dual_conditioning_real.run_real import train_and_eval
from sandbox.diagnostics.dual_conditioning_real.tests.fixture import bake_fixture


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def main() -> int:
    if not torch.cuda.is_available():
        print("[smoke_gpu] CUDA not available — FAIL")
        return 2
    print(f"[smoke_gpu] device={torch.cuda.get_device_name(0)}")
    fix = str(bake_fixture())
    res = train_and_eval(h5_path=fix, epochs=2, steps_per_epoch=5, batch_size=2, device="cuda",
                         seed=0, eval_batch_size=2, eval_max_batches=None, fg_frac=0.1, n_boot=100)
    assert res["train_losses"] and all(np.isfinite(res["train_losses"]))
    m1, m2, m3 = res["M1"], res["M2"], res["M3"]
    assert isinstance(m1["encoder_eff_rank"], float)
    for cov in ("run_type", "read_length", "depth"):
        assert cov in m2
    assert _finite(m3["ratio"]) or math.isnan(m3["ratio"])
    print(f"[smoke_gpu] units={res['n_units']} eff_rank={m1['encoder_eff_rank']:.2f} "
          f"M2_depth_targets={m2['depth']['n_targets']} M3_ratio={m3['ratio']:.3f}")
    print("GPU SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

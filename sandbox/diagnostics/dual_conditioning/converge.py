"""Single-model trajectory: train once, eval M1(ceiling/offdiag)/M2/M3 every few epochs + log NLL.
Confirms steering (M2) rises and picks the sweep epoch count. Run (GPU):
  python -m sandbox.diagnostics.dual_conditioning.converge"""
from __future__ import annotations

import numpy as np
import torch

from sandbox.diagnostics.dual_conditioning import transforms as T
from sandbox.diagnostics.dual_conditioning import metrics as MET
from sandbox.diagnostics.dual_conditioning.data import DualCondData
from sandbox.diagnostics.dual_conditioning.run import train_model

MAX_EPOCHS = 40
EVAL_EVERY = 5


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    data = DualCondData()
    conds = T.all_conditions()
    units = MET.make_eval_units(data, 8, 16, np.random.default_rng(0), chrom="chr21")

    def hook(model, ep, ep_nll):
        if (ep + 1) % EVAL_EVERY == 0:
            m1 = MET.eval_M1(model, data, units, device, n_param=1)
            m2 = MET.eval_M2(model, data, units, device)
            m3 = MET.eval_M3(model, data, units, device)
            print(f"  epochs={ep+1:3d} nll={ep_nll:.3f} | M1_ceiling={m1['ceiling']:.3f} "
                  f"M1_offdiag={m1['median_offdiag_r2']:.3f} gap={m1['median_gap']:.3f} | "
                  f"M2_inv={m2['median_invertible']:.3f} M2_all={m2['median']:.3f} | "
                  f"M3={m3['ratio']:.3f}", flush=True)

    train_model("zscore", data, conds, epochs=MAX_EPOCHS, steps_per_epoch=200,
                batch_size=16, device=device, seed=0, log_every=400, on_epoch=hook)
    print("[converge] DONE")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Pre-flight smoke test for ONE ERA candidate program (generate.py calls this in an
isolated subprocess). Runs the candidate's REAL train + distribution mechanics on a
single tiny CPU batch from the actual data loader, to catch structural CRASHES —
missing batch keys, shape/broadcast errors, a bad output dict, non-finite loss, a
count_dist that can't .mean/.sample — BEFORE a ~20-min GPU job is spent.

It deliberately does NOT judge performance (the collapse / score gates are the GPU
job's domain): a clean mechanical run prints SMOKE_OK. Note it runs on CPU, so it
cannot catch cuda-only device mismatches — those still gate to -1e9 on the GPU job.

Three outcomes (generate.py rejects ONLY the first):
  SMOKE_FAIL : the candidate itself crashed / produced a bad output (exit 1)
  SMOKE_OK   : clean mechanical run (exit 0)
  SMOKE_SKIP : the harness/data setup failed (environmental, not the candidate) (exit 2)

    python smoke.py <program.py>
"""
import importlib.util
import os
import sys
import traceback
from pathlib import Path

import torch

torch.set_num_threads(1)            # keep the login-node smoke single-threaded (pids/fork budget)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_judge"))  # vendored: frozen judge on path

OK, FAIL, SKIP = "SMOKE_OK", "SMOKE_FAIL", "SMOKE_SKIP"
# The smoke batch is fixed (seed=0) and tiny; cache it so only the FIRST smoke pays the
# ~60s cold read of the 1.6 GB HDF5 — the rest load a ~1 MB .pt in <1s. Delete it if you
# re-bake sandbox.h5.
_BATCH_CACHE = Path(__file__).resolve().parent / ".smoke_batch.pt"


def _load_batch(dev):
    if _BATCH_CACHE.exists():
        return {k: v.to(dev) for k, v in torch.load(_BATCH_CACHE, weights_only=True).items()}
    from data_v3 import iter_train_batches
    from harness import H5
    raw = {k: torch.as_tensor(v) for k, v in next(iter_train_batches(H5, 4, seed=0)).items()}
    tmp = _BATCH_CACHE.with_suffix(".tmp")
    torch.save(raw, tmp); os.replace(tmp, _BATCH_CACHE)     # atomic; concurrent writers are fine
    return {k: v.to(dev) for k, v in raw.items()}


def _load_candidate(path: str):
    spec = importlib.util.spec_from_file_location("candidate_smoke", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)    # import / syntax errors surface here
    return mod


def main(path: str) -> int:
    dev = torch.device("cpu")
    # ---- SETUP (harness + data): failures here are environmental, NOT the candidate's ----
    try:
        from harness import INPUT_KEYS
        raw = _load_batch(dev)
    except Exception:
        print(f"{SKIP}: harness/data setup failed")
        traceback.print_exc()
        return 2

    # ---- CANDIDATE: any failure here is the candidate's fault -> reject ----
    try:
        mod = _load_candidate(path)
        model = mod.Model().to(dev)
        obj = mod.Objective()
        gen = torch.Generator(device="cpu"); gen.manual_seed(0)
        ref = tuple(raw["counts"].shape)                  # [B,L,F] reference shape

        # training mechanics: corrupt -> forward -> loss -> backward -> step
        cb = obj.corrupt(raw, gen)
        out = model(*[cb[k] for k in INPUT_KEYS])
        for k in ("count_dist", "signal_pred"):
            if k not in out:
                print(f"{FAIL}: Model.forward output missing required key '{k}'"); return 1
        cd, sp = out["count_dist"], out["signal_pred"]
        if tuple(sp.shape) != ref:
            print(f"{FAIL}: signal_pred shape {tuple(sp.shape)} != counts {ref}"); return 1
        if tuple(cd.mean.shape) != ref:
            print(f"{FAIL}: count_dist.mean shape {tuple(cd.mean.shape)} != counts {ref}"); return 1
        cd.sample((2,))                                   # eval uses sampling for PIT / C-index
        loss = obj.loss(out, cb)
        if not torch.isfinite(loss):
            print(f"{FAIL}: non-finite training loss ({float(loss)})"); return 1
        opt = obj.configure_optimizer(model.parameters())
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

        # eval-style forward: all-observed query, no corruption mask (the eval path)
        ev = {k: cb[k] for k in INPUT_KEYS}
        ev["x_mask"] = torch.zeros_like(cb["x_mask"])
        ev["query_mask"] = torch.ones_like(cb["query_mask"])
        with torch.no_grad():
            out2 = model(*[ev[k] for k in INPUT_KEYS])
        if not (torch.isfinite(out2["count_dist"].mean).all()
                and torch.isfinite(out2["signal_pred"]).all()):
            print(f"{FAIL}: non-finite eval-pass output"); return 1
    except Exception:
        print(f"{FAIL}: exception during candidate run")
        traceback.print_exc()
        return 1

    print(OK)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1]))
    except Exception:                                     # truly unexpected -> inconclusive, don't reject
        print(f"{SKIP}: unexpected smoke-harness error")
        traceback.print_exc()
        sys.exit(2)

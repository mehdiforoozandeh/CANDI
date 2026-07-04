"""Budget probe: train node 70 (best ERA candidate) FAR past the 600s/2.3-epoch cap and
eval held-out imputation at checkpoints. Answers: is the ERA budget the binding constraint
(Q_imp still climbing at the 2.3-ep cutoff) or is it already converged?

Everything reuses the FROZEN harness internals so numbers are comparable to the search.
Writes one PROBE line per checkpoint (epoch, ERA_SCORE, Q_imp, the 4 imp corrs, ece, dcr).
"""
from __future__ import annotations

import importlib.util
import sys
import time

import numpy as np
import torch

HERE = "/project/6014832/mforooz/EpiDenoise/sandbox/candi_v3"
sys.path.insert(0, HERE)

from data_v3 import iter_train_batches, train_steps_per_epoch          # noqa: E402
from eval_v3 import compute_aspects                                    # noqa: E402
from harness import H5, INPUT_KEYS, _predict_eval, _to_dev            # noqa: E402
from score import era_score, load_constants                           # noqa: E402

# load node 70 program (its __main__ guard makes import side-effect free)
spec = importlib.util.spec_from_file_location("cand70", HERE + "/runs/cand0071/program.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
Model, Objective = mod.Model, mod.Objective

SEED = 0
BATCH = 4                       # matches harness default (what the search used)
WALL_LIMIT_S = 6600.0           # hard stop well inside the SLURM walltime
CKPT_EPOCHS = [0.5, 1.0, 2.3, 4.0, 6.0, 8.0, 11.0, 14.0, 18.0, 22.0]

torch.manual_seed(SEED)
np.random.seed(SEED)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gen_rng = torch.Generator(device="cpu")
gen_rng.manual_seed(SEED)

spe = train_steps_per_epoch(H5, BATCH)
ckpt_steps = sorted({int(round(e * spe)) for e in CKPT_EPOCHS})
max_steps = ckpt_steps[-1]
consts = load_constants(HERE + "/constants_frozen.yaml")

model = Model().to(dev)
obj = Objective()
opt = obj.configure_optimizer(model.parameters())
batches = iter_train_batches(H5, BATCH, seed=SEED)

print(f"[probe] steps/epoch={spe}  checkpoints(steps)={ckpt_steps}  max_steps={max_steps}",
      flush=True)


def evaluate():
    model.eval()
    imp, den, cal, dcr = _predict_eval(model, dev)        # FULL chr21 eval (comparable to search)
    asp = compute_aspects(imp, den, cal, dcr)
    total, _ = era_score(asp, consts)
    model.train()
    return total, asp


model.train()
t0 = time.time()
ci = 0
for step in range(max_steps + 1):
    if ci < len(ckpt_steps) and step == ckpt_steps[ci]:
        tot, asp = evaluate()
        print(f"PROBE step={step} epoch={step / spe:.2f} t={time.time() - t0:.0f}s "
              f"ERA_SCORE={tot:.4f} Q_imp={asp['Q_imp']:.4f} Q_den={asp['Q_den']:.4f} "
              f"ip_sp={asp['imp_pval_spearman']:.4f} ic_sp={asp['imp_count_spearman']:.4f} "
              f"ip_pe={asp['imp_pval_pearson']:.4f} ic_pe={asp['imp_count_pearson']:.4f} "
              f"ece={asp['ece']:.4f} dcr={asp['dcr']:.3f}", flush=True)
        ci += 1
    if step == max_steps:
        break
    if time.time() - t0 > WALL_LIMIT_S:
        print(f"[probe] wall limit hit at step {step} ({step / spe:.2f} ep); final eval", flush=True)
        tot, asp = evaluate()
        print(f"PROBE step={step} epoch={step / spe:.2f} t={time.time() - t0:.0f}s "
              f"ERA_SCORE={tot:.4f} Q_imp={asp['Q_imp']:.4f} Q_den={asp['Q_den']:.4f} "
              f"ip_sp={asp['imp_pval_spearman']:.4f} ic_sp={asp['imp_count_spearman']:.4f} "
              f"ip_pe={asp['imp_pval_pearson']:.4f} ic_pe={asp['imp_count_pearson']:.4f} "
              f"ece={asp['ece']:.4f} dcr={asp['dcr']:.3f}", flush=True)
        break
    raw = _to_dev(next(batches), dev)
    cb = obj.corrupt(raw, gen_rng)
    out = model(*[cb[k] for k in INPUT_KEYS])
    loss = obj.loss(out, cb)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

print(f"[probe] DONE Q_imp_baseline=0.4857 (S_A=0 line). total wall {time.time() - t0:.0f}s", flush=True)

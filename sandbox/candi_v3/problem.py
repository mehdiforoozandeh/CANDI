"""CANDI v3 ERA problem. `get_problem() -> (Problem, initial_solution, initial_score)`.

DESCRIPTION = DESIGN_MENU.md (problem + hard constraints + design menu + scoring contract)
+ the concrete candidate program contract (how to call the frozen harness). The candidate is a
whole `program.py` that defines `Model` + `Objective` and prints `ERA_SCORE:`.
"""
from __future__ import annotations

from pathlib import Path

from futs import Problem, Solution

HERE = Path(__file__).resolve().parent

_CONTRACT = '''
## How to write a candidate

Write ONE self-contained `program.py`. Define a `Model` (torch.nn.Module) and an `Objective`,
then call the FROZEN harness and print the score footer. Do NOT edit any file under
sandbox/candi_v3/ except your own program; import the harness.

```python
import sys; sys.path.insert(0, "/project/6014832/mforooz/EpiDenoise/sandbox/candi_v3")
import torch, torch.nn as nn
from harness import run_and_score

class Model(nn.Module):
    def forward(self, x_counts, x_avail, x_mask, x_meta, x_dna,
                control, ctrl_avail, y_meta, query_mask):
        # returns dict: {"count_dist": torch.distributions.Distribution over raw counts [B,L,F],
        #                "signal_pred": Tensor[B,L,F], "peak_prob": Tensor[B,L,F] or None}
        ...

class Objective:
    def corrupt(self, batch, rng): ...        # clean train batch -> harness inputs + targets/masks
    def loss(self, out, cb): ...              # scalar training loss
    def configure_optimizer(self, params): ... # torch optimizer

if __name__ == "__main__":
    print(f"ERA_SCORE: {run_and_score(Model, Objective())}")   # harness owns the fixed training budget
```

The harness trains under a FIXED budget = min(5 epochs over the train data, 1800 s wall-clock),
whichever comes first — you cannot extend it. 5 epochs is reachable in 1800 s for most models, so
the epoch cap is usually binding; a model too slow to complete a meaningful fraction of 5 epochs
in 1800 s will score poorly. The data is small and overfits quickly.

INPUT tensors (B batch, L=768 bins, F=8 assays): x_counts[B,L,F] raw counts (0 where not
observed); x_avail[B,F]; x_mask[B,L,F] bool (positions to predict — do not peek at truth);
x_meta[B,4,F] (depth_log2, assay_idx, read_len, run_type); x_dna[B,19200,4] one-hot;
control[B,L,1]; ctrl_avail[B] (may be 0 — CONTROL-OPTIONAL, must still run); y_meta[B,4,F]
TARGET-assay covariates; query_mask[B,F]. `Objective.corrupt` must return those 9 input keys
PLUS the targets/masks YOUR loss reads — at minimum y_counts[B,L,F] + sup_mask[B,L,F] (bool);
optionally y_signal[B,L,F] / y_peaks[B,L,F] only if you supervise those heads. You DERIVE
those targets — they are not handed to you (see next paragraph).

The CLEAN batch given to `Objective.corrupt` is a dict with EXACTLY these keys (index ONLY
these — e.g. `batch["signal"]` does NOT exist and will crash):
  counts, counts_dsf2, counts_dsf4, counts_dsf8   each [B,L,F]   (counts=dsf1 full depth)
  meta,   meta_dsf2,   meta_dsf4,   meta_dsf8      each [B,4,F]   (meta=dsf1; lower depth_log2 per level)
  avail [B,F]   dna [B,19200,4]   control [B,L,1]   ctrl_avail [B]
There is NO ground-truth signal/pval or peak key in the batch: if you want a y_signal target,
DERIVE it from counts (e.g. arcsinh(counts)); y_counts is just counts (or a dsf level).
The multiple depth levels let you build denoising (low-depth input -> high-depth target) pairs;
unified corruption = compose per-(assay,pos) dropout with depth downsampling (CRITIQUE §4).
Keep every tensor you create on the input tensors' OWN device (e.g. `dev = counts.device`).

OUTPUT: count_dist is THE one learned likelihood (any torch Distribution over counts; harness
uses .mean and .sample). signal_pred drives the headline imputation Spearman (may be a readout
of count_dist or a head). peak_prob optional.

## Scoring (frozen — the harness computes it; you only print its return)
ERA_SCORE = (imp_pval_spearman_gw - 0.4652)        # S_A: held-out V_/B_ imputation skill
          + 0.4*min(0, 0.0734 - ECE)               # calibration floor (count predictive coverage)
          + 0.02*(min(0,DCR-3)+min(0,5-DCR))       # DCR band ~[3,5] around 4.0 (+2 log2 => 4x)
Higher is better. Crash / OOM / non-finite / constant-output / forbidden import => ERA_SCORE: -1e9.
Optionally print one line `ERA_REFLECTION: <2-3 sentences on what you tried and why>` for the lab notebook.
'''


def get_problem():
    desc = (HERE / "DESIGN_MENU.md").read_text() + _CONTRACT
    seed = (HERE / "seed" / "program.py").read_text()
    return Problem(desc), Solution(seed), None   # None -> driver scores the seed root once

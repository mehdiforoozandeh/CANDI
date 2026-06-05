# E6 - Linear FiLM

Status: incomplete  
Parent: E7 / prompt-collapse investigation  
Run name: E6_linear_film  
Checklist entry: [EXPERIMENTS.md](EXPERIMENTS.md#experiment-sweep-e6-e16)

## Problem Statement

Decoder FiLM currently uses exponential scaling plus a hard clamp, which can zero gradients when the scale path saturates and may make metadata conditioning harder to learn.

## Idea / Hypothesis

Replacing `x * exp(scale) + shift` with linear FiLM `x * (1 + scale) + shift` should preserve gradients through the conditioning path while matching the original FiLM formulation.

## Planned Intervention

- Submit/config path: `sandbox/configs/e6_linear_film.yaml`
- Run name: `E6_linear_film`
- Parent run or idea: `baseline_anchor` (HPO graph parent)
- Config/code/data deltas: see Implementation Plan below

## Implementation Plan

### Step 1 — Add `linear` flag to `FiLMLayer` (`model.py`)

Location: class `FiLMLayer` (~line 2390).

Change `__init__` signature:
```python
def __init__(self, input_dim, output_dim, linear: bool = False):
    super().__init__()
    self.proj = nn.Linear(input_dim, output_dim)
    self.linear = linear
    nn.init.xavier_uniform_(self.proj.weight)
    nn.init.normal_(self.proj.bias, mean=0.0, std=0.1)
```

Change `forward` (replace the clamp + exp block at the end):
```python
        scale = scale.unsqueeze(-1)
        shift = shift.unsqueeze(-1)

        if self.linear:
            return x * (1.0 + scale) + shift
        scale = torch.clamp(scale, min=-4.0, max=4.0)
        return x * torch.exp(scale) + shift
```

Note: the clamp is intentionally skipped in linear mode — gradients are always alive.
Watch for sign-flips when `scale < -1` (feature negation); these should be visible in
grad-norm diagnostics if they cause instability.

### Step 2 — Thread `linear_film` through `CANDI_Decoder` (`model.py`)

Location: class `CANDI_Decoder.__init__` (~line 1231).

Change signature:
```python
def __init__(self, signal_dim, metadata_embedding_dim, conv_kernel_size,
             n_cnn_layers, context_length, pool_size=2, expansion_factor=3,
             norm="layer", linear_film: bool = False):
```

Pass `linear=linear_film` to every `FiLMLayer(...)` call in the `self.film_layers` loop
(both the standard path and the fallback path, lines ~1266–1272).

### Step 3 — Thread `linear_film` through the encoder `CANDI_SignalEncoder` (`model.py`)

Location: encoder class that owns `self.film_layers` (the X-side metadata FiLM loop,
~lines 1625–1632).

Change the encoder `__init__` to accept `linear_film: bool = False` and pass
`linear=linear_film` to each `FiLMLayer(...)` in the `self.film_layers` loop.

### Step 4 — Thread `linear_film` through `CANDI.__init__` (`model.py`)

Location: `CANDI.__init__` (~line 1707).

Add `linear_film: bool = False` to the `CANDI.__init__` signature.

Pass it when instantiating the encoder (if it owns the X-side film_layers directly,
the param goes there; if CANDI owns them directly at ~line 1631, pass it there).

Pass it when instantiating each decoder at lines ~1818–1822:
```python
self.count_decoder = CANDI_Decoder(
    signal_dim, metadata_embedding_dim, conv_kernel_size,
    n_cnn_layers, context_length, pool_size, expansion_factor, norm,
    linear_film=linear_film,
)
self.pval_decoder = CANDI_Decoder(...)   # same
self.peak_decoder = CANDI_Decoder(...)   # same
# also self.decoder = CANDI_Decoder(...)  for the shared-decoder path
```

### Step 5 — Add `linear_film` to `ModelConfig` (`sandbox/config_types.py`)

Location: `ModelConfig` dataclass (~line 44). Add:
```python
linear_film: bool = False
```

### Step 6 — Forward `linear_film` in `build_sandbox_candi` (`sandbox/model.py`)

Add `linear_film=cfg.model.linear_film` (or direct bool kwarg) to the `CANDI(...)`
call in `build_sandbox_candi`. Signature change:
```python
def build_sandbox_candi(
    ...
    linear_film: bool = False,
) -> CANDI:
    ...
    return CANDI(
        ...
        linear_film=linear_film,
    )
```

The caller in `sandbox/train.py` or wherever `build_sandbox_candi` is called should
pass `linear_film=cfg.model.linear_film`.

### Step 7 — Create experiment config (`sandbox/configs/e6_linear_film.yaml`)

```yaml
model:
  linear_film: true
```

No other overrides needed — inherits all defaults including `log1p` input transform
and type1_chr19 regime when passed as the regime config.

### Step 8 — Create SLURM submit script (`sandbox/slurm/submit_experiments_e6_e7.sh`)

Reuse `baseline_train.sh` template with `BASELINE_PREFIX=""` (experiment naming).
E6 entry:
```bash
E6=$(BASELINE_NAME="E6_linear_film" \
     BASELINE_EXTRA="--config sandbox/configs/e6_linear_film.yaml \
                     --hpo.parent baseline_anchor \
                     --hpo.experiment_label E6_linear_film" \
     sbatch --job-name=sbx_e6_linear_film \
            "${SBATCH_RES[@]}" \
            --parsable \
            "$SCRIPT")
echo "Submitted E6 (linear_film) → SLURM job $E6"
```

### Step 9 — Smoke test before submitting

From the repo root (after activating `candi_venv` and `conda activate candi`):
```bash
python -c "
from sandbox.model import build_sandbox_candi
m = build_sandbox_candi(linear_film=True)
import torch
# [B, L, A+1] signal, [B, 4, 25*L] DNA, [B, A+1, 4] metadata
x = torch.zeros(2, 768, 9)
s = torch.zeros(2, 4, 19200)
meta = torch.zeros(2, 9, 4, dtype=torch.long)
out = m(x, s, meta, meta)
print('E6 smoke test passed, output shapes:', [o.shape for o in out if o is not None])
"
```

Verify no NaN/Inf in outputs and all output shapes match the non-linear baseline.

### Step 10 — Submit

```bash
bash sandbox/slurm/submit_experiments_e6_e7.sh
```

Monitor: `squeue -u mforooz` and `sandbox/slurm_logs/`.

## Verifiables

- Validate if: training remains stable, no NaN/Inf appears, prompt-sensitivity metrics
  improve or stay measurable, and per-module metadata/FiLM grad norms remain live.
- Disvalidate if: losses degrade materially, prompt-sensitivity metrics remain flat,
  or linear scaling causes negative-gain instability.
- Required artifacts: `resolved_config.yaml`, `metrics.jsonl`, SLURM logs, and W&B
  metadata when available.

## Risks / Watch-outs

- Linear FiLM can flip feature signs when `scale < -1`; watch for gradient explosions
  in early training before the scale path has converged.
- This may remove the clamp problem without fixing fixed-channel decoder shortcutting.
- Interpret with E9/E10/E11 grad diagnostics if available.

## Run Links

- Run directory: `sandbox/runs/E6_linear_film/`
- SLURM job ID: 38762403
- Resolved config: `sandbox/runs/E6_linear_film/resolved_config.yaml` (after run starts)
- Metrics: `sandbox/runs/E6_linear_film/metrics.jsonl` (after run starts)
- SLURM logs: `sandbox/slurm_logs/baseline_sbx_e6_linear_film_38762403.{out,err}`
- HPO graph node: TBD
- W&B run: TBD

## Findings

### Run 1 (200 epochs, old defaults lr=1e-4, clip=1.0, SLURM job 38762403)

- Walltime-killed at epoch 124. No divergence. best total_loss=4.922 @ ep 119. count_imp_loss=2.033 (stagnated), pval_imp_loss=0.684, depth_count_ratio=1.003.
- Inconclusive due to early kill. Marked incomplete.

### Run 2 (400 epochs, new defaults lr=1e-3, clip=2.0, SLURM job 38823403) — **current**

- Walltime-killed at epoch 289 of 400 (~72% budget). Diverged (last=49.43, best=4.99 @ ep29).
- At best epoch (ep29): `imp_peak_auroc=0.509`, `imp_count_pearson=0.033`, `imp_pval_pearson=0.181`.
- Best `imp_peak_auroc` overall: 0.653 @ ep269.
- Divergence pattern: `pval_imp` explodes at ep34 (first among all FiLM variants, similar to baseline ep29). `count_imp` diverges later at ep169. `peak_imp` never diverges.
- Grad-norm: `pval_obs` median=8.8 (vs baseline 25.9) — linear FiLM significantly suppresses pval gradients, consistent with Run 1 finding. `peak` grad norms higher than baseline (0.85 vs 0.34 median).
- `depth_count_ratio` last=0.984 — metadata collapse persists.
- Ranker: ineligible (diverged, last > 1.5× best); all multi-head runs diverge.
- Interpretation: Linear FiLM weakens pval gradients, causing earlier and harder count divergence relative to E7. No meaningful improvement over baseline across any metric at best epoch. The linear FiLM change on its own does not help — it modestly hurts count while leaving peak and pval mostly unchanged.
- Decision: **Do not adopt linear_film as default.** E6 is strictly dominated by E7 on every metric. The linear scaling hypothesis is rejected for the current architecture. See [synthesis_e6_e7_film_ablation.md](synthesis_e6_e7_film_ablation.md).

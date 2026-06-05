# E7 - Single-shot decoder FiLM

Status: incomplete  
Parent: E6 / prompt-collapse investigation  
Run name: E7_single_shot_decoder_film  
Checklist entry: [EXPERIMENTS.md](EXPERIMENTS.md#experiment-sweep-e6-e16)

## Problem Statement

Applying FiLM at every decoder layer makes the decoder a repeated function of metadata,
which creates redundant conditioning sites and makes gradient balancing hard to interpret.

## Idea / Hypothesis

Conditioning the decoder latent once — applied to the compressed latent `[N, F2, L']`
immediately before the deconv tower — should make the downstream decoder a pure spatial
upsampler with no further metadata entanglement, improving gradient attribution and
making per-group clipping more meaningful.

## Planned Intervention

- Submit/config path: `sandbox/configs/e7_single_shot_decoder_film.yaml`
- Run name: `E7_single_shot_decoder_film`
- Parent run or idea: `baseline_anchor` (HPO graph); architecturally follows E6
- Config/code/data deltas: see Implementation Plan below

Note: E7 can be run independently (exp FiLM, single-shot) or combined with E6
(linear FiLM + single-shot) via a combined config. The submit script runs E7 standalone
first; add a separate E6+E7 combined entry to compare all four variants.

## Implementation Plan

### Step 1 — Add `single_shot_film` flag to `CANDI_Decoder.__init__` (`model.py`)

Location: class `CANDI_Decoder.__init__` (~line 1231).

Change signature:
```python
def __init__(self, signal_dim, metadata_embedding_dim, conv_kernel_size,
             n_cnn_layers, context_length, pool_size=2, expansion_factor=3,
             norm="layer", linear_film: bool = False,
             single_shot_film: bool = False):
```

Store the flag:
```python
self.single_shot_film = single_shot_film
```

Conditionally allocate film modules. Replace the existing `self.film_layers`
construction block (~lines 1259–1273) with:

```python
if self.single_shot_film:
    # One FiLM on the full latent before the deconv tower.
    # Latent has F2 = signal_dim * expansion_factor^n_cnn_layers channels,
    # grouped into signal_dim assays, so d_per_assay = expansion_factor^n_cnn_layers.
    d_per_latent = self.f2 // signal_dim       # e.g. 2^3 = 8 for sandbox defaults
    self.latent_film = FiLMLayer(
        input_dim=metadata_embedding_dim,
        output_dim=d_per_latent * 2,
        linear=linear_film,                    # honour the E6 flag if combined
    )
    # self.film_layers intentionally not allocated — no dead params
else:
    # Per-layer FiLM Adapter for Y-side metadata (existing code)
    self.film_layers = nn.ModuleList()
    for i in range(n_cnn_layers):
        layer_channels = (reverse_conv_channels[i + 1]
                          if i + 1 < n_cnn_layers
                          else int(reverse_conv_channels[i] / expansion_factor))
        if layer_channels % signal_dim == 0:
            d_per_assay = layer_channels // signal_dim
            self.film_layers.append(
                FiLMLayer(input_dim=metadata_embedding_dim,
                          output_dim=d_per_assay * 2,
                          linear=linear_film)
            )
        else:
            self.film_layers.append(
                FiLMLayer(input_dim=metadata_embedding_dim,
                          output_dim=layer_channels * 2 // signal_dim,
                          linear=linear_film)
            )
```

### Step 2 — Update `CANDI_Decoder.forward` (`model.py`)

Location: `CANDI_Decoder.forward` (~line 1275). Replace the loop with a branch:

```python
def forward(self, src, y_metadata_embed):
    src = src.permute(0, 2, 1)  # [N, F2, L']

    if self.single_shot_film:
        # Condition once on the compressed latent, then upsample clean.
        src = self.latent_film(src, y_metadata_embed)
        for dconv in self.deconv:
            src = dconv(src)
    else:
        for i, dconv in enumerate(self.deconv):
            src = dconv(src)
            src = self.film_layers[i](src, y_metadata_embed)

    src = src.permute(0, 2, 1)  # [N, L, F1]
    return src
```

Shape note: `self.latent_film` receives `[N, F2, L']` where `F2 = signal_dim *
expansion_factor^n_cnn_layers`. `FiLMLayer` checks `C % F == 0`; with sandbox defaults
`F2 = 8 * 2^3 = 64` and `signal_dim = 8`, `d_per_latent = 8`. Confirm this holds before
submitting (smoke test below will catch a mismatch).

### Step 3 — Thread `single_shot_decoder_film` through `CANDI.__init__` (`model.py`)

Location: `CANDI.__init__` (~line 1707).

Add `single_shot_decoder_film: bool = False` to the `CANDI.__init__` signature.
Pass to each `CANDI_Decoder` instantiation at lines ~1818–1822:

```python
self.count_decoder = CANDI_Decoder(
    signal_dim, metadata_embedding_dim, conv_kernel_size,
    n_cnn_layers, context_length, pool_size, expansion_factor, norm,
    linear_film=linear_film,
    single_shot_film=single_shot_decoder_film,
)
self.pval_decoder = CANDI_Decoder(...)   # same
self.peak_decoder = CANDI_Decoder(...)   # same
# also self.decoder = CANDI_Decoder(...) for the shared-decoder path
```

### Step 4 — Add `single_shot_decoder_film` to `ModelConfig` (`sandbox/config_types.py`)

Location: `ModelConfig` dataclass (~line 44). Add:
```python
single_shot_decoder_film: bool = False
```

(If E6 has not yet been implemented, also add `linear_film: bool = False` here — see
E6 Step 5.)

### Step 5 — Forward `single_shot_decoder_film` in `build_sandbox_candi` (`sandbox/model.py`)

Add the kwarg to `build_sandbox_candi` and forward to `CANDI(...)`:
```python
def build_sandbox_candi(
    ...
    linear_film: bool = False,
    single_shot_decoder_film: bool = False,
) -> CANDI:
    ...
    return CANDI(
        ...
        linear_film=linear_film,
        single_shot_decoder_film=single_shot_decoder_film,
    )
```

The caller (train loop or config resolution) passes `cfg.model.linear_film` and
`cfg.model.single_shot_decoder_film`.

### Step 6 — Create experiment configs

`sandbox/configs/e7_single_shot_decoder_film.yaml` (standalone, exp FiLM):
```yaml
model:
  single_shot_decoder_film: true
```

`sandbox/configs/e6_e7_combined.yaml` (linear FiLM + single-shot, optional variant):
```yaml
model:
  linear_film: true
  single_shot_decoder_film: true
```

### Step 7 — Add E7 entry to SLURM submit script (`sandbox/slurm/submit_experiments_e6_e7.sh`)

In the same script as E6 (see E6 Step 8), add after the E6 block:

```bash
E7=$(BASELINE_NAME="E7_single_shot_decoder_film" \
     BASELINE_EXTRA="--config sandbox/configs/e7_single_shot_decoder_film.yaml \
                     --hpo.parent baseline_anchor \
                     --hpo.experiment_label E7_single_shot_decoder_film" \
     sbatch --job-name=sbx_e7_ssfilm \
            "${SBATCH_RES[@]}" \
            --parsable \
            "$SCRIPT")
echo "Submitted E7 (single_shot_decoder_film) → SLURM job $E7"

# Optional combined run (E6 + E7)
E6E7=$(BASELINE_NAME="E6E7_linear_single_shot" \
      BASELINE_EXTRA="--config sandbox/configs/e6_e7_combined.yaml \
                      --hpo.parent baseline_anchor \
                      --hpo.experiment_label E6E7_linear_single_shot" \
      sbatch --job-name=sbx_e6e7 \
             "${SBATCH_RES[@]}" \
             --parsable \
             "$SCRIPT")
echo "Submitted E6+E7 combined        → SLURM job $E6E7"
```

### Step 8 — Smoke test before submitting

From the repo root (after `conda activate candi && source candi_venv/bin/activate`):
```bash
python -c "
from sandbox.model import build_sandbox_candi
import torch

# Test E7 standalone
m = build_sandbox_candi(single_shot_decoder_film=True)
x = torch.zeros(2, 768, 9)
s = torch.zeros(2, 4, 19200)
meta = torch.zeros(2, 9, 4, dtype=torch.long)
out = m(x, s, meta, meta)
print('E7 smoke test passed:', [o.shape for o in out if o is not None])

# Test E6+E7 combined
m2 = build_sandbox_candi(linear_film=True, single_shot_decoder_film=True)
out2 = m2(x, s, meta, meta)
print('E6+E7 combined smoke test passed:', [o.shape for o in out2 if o is not None])

# Confirm no film_layers allocated in single-shot decoders
assert not hasattr(m.count_decoder, 'film_layers'), 'film_layers should not exist in single_shot mode'
assert hasattr(m.count_decoder, 'latent_film'), 'latent_film should exist'
print('Parameter allocation check passed.')
"
```

### Step 9 — Submit

```bash
bash sandbox/slurm/submit_experiments_e6_e7.sh
```

Monitor: `squeue -u mforooz` and `sandbox/slurm_logs/`.

## Evaluation Matrix

Compare across four runs using `python -m sandbox.hpo.view_graph --leaderboard`:

| Run | `linear_film` | `single_shot_decoder_film` | purpose |
|-----|---|---|---|
| `baseline_anchor` | false | false | reference |
| `E6_linear_film` | true | false | E6 alone |
| `E7_single_shot_decoder_film` | false | true | E7 alone |
| `E6E7_linear_single_shot` | true | true | combined |

## Verifiables

- Validate if: prompt-sensitivity metrics remain measurable or improve, module grad
  norms become easier to attribute, and core branch losses do not regress beyond
  expected noise.
- Disvalidate if: prompt sensitivity worsens, decoder losses regress sharply, or the
  single conditioning point cannot carry enough metadata information.
- Required artifacts: `resolved_config.yaml`, `metrics.jsonl`, SLURM logs, and W&B
  metadata when available.

## Risks / Watch-outs

- Removing deeper FiLM sites reduces conditional capacity in the decoder; a single
  latent-level signal must now carry all metadata information through 3 deconv stages.
- Fixed-channel shortcutting in the encoder can still persist even with single-shot
  decoder conditioning — this experiment only cleanses the decoder path.
- Compare E7 alone vs E6+E7 to separate the contribution of single-shot vs linear
  scaling from the combined effect.

## Run Links

- Run directory: `sandbox/runs/E7_single_shot_decoder_film/`
- SLURM job ID: 38762404 (E7 standalone); 38762405 (E6+E7 combined → `sandbox/runs/E6E7_linear_single_shot/`)
- Resolved config: `sandbox/runs/E7_single_shot_decoder_film/resolved_config.yaml` (after run starts)
- Metrics: `sandbox/runs/E7_single_shot_decoder_film/metrics.jsonl` (after run starts)
- SLURM logs: `sandbox/slurm_logs/baseline_sbx_e7_ssfilm_38762404.{out,err}`
- HPO graph node: TBD
- W&B run: TBD

## Findings

### Run 1 (200 epochs, old defaults, SLURM jobs 38762404 / 38762405)

- E7 walltime-killed at ep114. No divergence. best total_loss=5.103 @ ep99. count_imp=1.645 (best FiLM), pval_imp plateaued at 0.97. E6E7 similar. Inconclusive due to early kill.

### Run 2 (400 epochs, new defaults lr=1e-3, clip=2.0, SLURM jobs 38823404 / 38823405) — **current**

**E7 standalone:**
- Walltime-killed at epoch 294 of 400. Diverged (last=26.55, best=2.885 @ ep84).
- At best epoch (ep84): `imp_peak_auroc=0.765`, `imp_count_pearson=0.339`, `imp_pval_pearson=0.277`. **All three substantially better than any other multi-head run.**
- Best `imp_peak_auroc` overall: 0.796 @ ep79.
- Divergence pattern: `pval_imp` diverges at ep94 — the **latest onset of any multi-head run** (baseline ep29, E6 ep34, E6E7 ep59). `count_imp` and `peak_imp` remain stable throughout.
- Grad-norm: global pre-clip median=7.45 (vs baseline 35.5) — E7 significantly tames overall gradient scale. clip_fraction=0.78 (vs baseline 0.88). `pval` median 3.76–4.77 (vs baseline 25.9), `count` median 0.93, `peak` median 0.63.
- `depth_count_ratio` last=1.000 — metadata collapse persists.

**E6E7 combined:**
- Walltime-killed at epoch 303. Diverged (last=50.06, best=3.060 @ ep54).
- At best epoch: `imp_peak_auroc=0.758`, `imp_count_pearson=0.210`, `imp_pval_pearson=0.288`.
- Best `imp_peak_auroc`: 0.795 @ ep129. Virtually tied with E7 standalone on peak.
- Pval divergence at ep59. Count and peak stable.

**Interpretation:**
- E7 is the **best multi-head architecture tested to date**. Single-shot latent FiLM dramatically improves all three metrics at best epoch vs baseline (peak +54%, count +914%, pval +66%) — not a marginal win. The delayed pval divergence (ep94 vs ep29) confirms that reducing per-layer FiLM noise in the decoder slows instability, likely by reducing high-frequency gradient noise that destabilizes the pval Gaussian head.
- Adding linear FiLM (E6E7) slightly worsens count (0.210 vs 0.339) and accelerates pval divergence (ep59 vs ep94), confirming E6 adds no value over E7 alone.
- The pval_imp divergence in E7 (ep94 onwards) is NOT fixed by the architecture change — it is a structural property of the Gaussian NLL head that becomes overconfident on observed assays and degenerates on masked ones (obs/imp split, see F7 in FINDINGS.md).
- **Decision: Promote E7 (single_shot_decoder_film=True) as the new default decoder architecture.** Update `default.yaml` with `model.single_shot_decoder_film: true`. The pval divergence must be addressed separately via logvar clamping (E13) or distributional hardening.

See [synthesis_e6_e7_film_ablation.md](synthesis_e6_e7_film_ablation.md).

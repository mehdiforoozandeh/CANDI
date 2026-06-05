# Autoresearch report — chr19 count head (May 28, 2026)

Status: synthesis (complete)  
Branch: `autoresearch/may28`  
Best commit: `35e54515` (config + A2 knobs at `b87156b7`)  
Artifacts: `sandbox/diagnostics/autoresearch/results.tsv`, `run.log`, `sweep.log`  
Harness: `sandbox/diagnostics/autoresearch/prepare.py` (fixed) + `train.py` (tuned)

---

## Executive summary

We ran an automated search over **count-head parameterization, optimizer settings, and loss weights** for CANDI v2 on real chr19 sandbox data (single pinned batch, assay-only masking, 1500 steps). Starting from a depth-offset baseline (`composite_score = 0.111`), the search reached **`composite_score = 0.009298`** — a **~92% reduction** — without changing model width, data, or training steps.

The winning recipe combines:

1. A **centered depth-offset NB head** (E29-style: μ = 2^(d−c)·exp(η))
2. **`depth_center = 27`** (not the diagnostic prior of 24)
3. **Adamax**, `lr = 1e-3`, **`clip_norm = 0.5`**
4. **`imp_weight = 8`**, **`obs_weight = 0.5`**

After the discovery phase, a structured sweep (A1/A2/A3, **131 total runs**, three queue cycles) **did not beat** the winner. The optimum appears sharp and reproducible on this harness.

---

## What was being optimized?

### Fixed harness (not searched)

| Setting | Value |
|---------|-------|
| Model | Sandbox CANDI v2 shell (~0.3M params, L=768, 8 assays, 2 transformer layers) |
| Data | `sandbox/data/sandbox.h5`, chr19, **batch_size=4**, **1 pinned training batch** |
| Steps | 1500 |
| Masking | Assay-only imputation (`p_full_assay=1`, `p_full_loci=0`, `p_chunks=0`) |
| Heads | Count only |
| VRAM cap | 9500 MB (all runs ~110 MB peak) |

### Composite score (lower is better)

Weighted sum of eval metrics after training:

| Term | Weight | Metric |
|------|--------|--------|
| Imputation Pearson | 0.35 | `imp_pearson` (masked assay reconstruction) |
| Depth ratio on masked bins | 0.25 | `dcr_masked_bins` (target ≈ 4.0 for Q5 pass) |
| Denoising | 0.20 | `1 − denoise_pearson` (DSF-corrupted input) |
| x_meta depth → latent | 0.10 | `x_depth_latent_delta` (want ≥ 0.02) |
| x_meta readlen → latent | 0.10 | `x_readlen_latent_delta` (want ≥ 0.02) |

Eval runs imputation probes (assay mask), denoising on unmasked DSF input, and metadata sensitivity battery from `meta_probes.py`.

### Search space (what *was* searched)

All changes were confined to `sandbox/diagnostics/autoresearch/train.py`:

- **Count head:** depth offset on/off, `depth_center`, pow2 vs linexp scaling, `n` parameterization (softplus vs exp), `mu_eps`
- **Optimizer:** adam / adamw / adamax / sgd; lr, weight_decay, beta1/beta2, clip_norm
- **Loss weights:** `obs_weight`, `imp_weight`, `count_weight`

Architecture trunk, data pipeline, step count, and eval protocol were frozen.

---

## Results

### Score trajectory (kept commits only)

| Commit | Key change | composite_score |
|--------|------------|-----------------|
| baseline | depth_center=24, Adam, lr=1e-3 | 0.111235 |
| f4793954 | compat patch (harness fix) | 0.105858 |
| 893cfc0b | depth_center=25 | 0.097360 |
| 0b3a260e | depth_center=26 | 0.092628 |
| f56e1e6f | depth_center=27 | 0.090981 |
| 242b4d41 | **adamax** | 0.076779 |
| b9b28a12 | clip_norm=0.5 | 0.072446 |
| 96b31635 | imp_weight=6 | 0.031988 |
| cdb59fec | imp_weight=8 | 0.009299 |
| **35e54515** | **obs_weight=0.5** | **0.009298** |

**Dominant lever:** raising `imp_weight` from 6→8 dropped score 0.032→0.009. Everything else was refinement.

### Structured sweep summary (post-discovery)

| Category | Experiments | Best discard | Verdict |
|----------|-------------|--------------|---------|
| **A1** hyperparams | clip_norm, imp/obs weight, lr, depth_center, WD, betas, count_weight | obs_weight=0.6 → 0.0107 | Optimum confirmed; lr=5e-4 → 0.166 |
| **A2** head variants | linexp, n_mode=exp, mu_eps, no offset | n_mode=exp → 0.0095 | pow2 offset required; exp-n near miss |
| **A3** optimizers | adamw, adam, lr×clip combos | imp7.5 obs0.55 → 0.011 | Adamax wins decisively |

Three full queue cycles produced **identical rankings** (deterministic seed + pinned batch). **Zero improvements** after commit `35e54515`.

### Closest challengers (never kept)

| Config | Score | Gap vs best |
|--------|-------|-------------|
| A2 n_mode=exp (cycle 1) | 0.009508 | +2.3% |
| A1 obs_weight=0.6 | 0.010746 | +15% |
| A3 imp7.5 obs0.55 | 0.010668 | +15% |
| A2 use_depth_offset=False | 0.499 | Q5 collapse (sanity) |

---

## Winning configuration

Restore with:

```bash
git checkout autoresearch/may28
git reset --hard b87156b7   # best config + A2 head knobs
```

### TrainConfig (winning values)

```python
use_depth_offset = True
depth_center = 27.0
depth_scale_mode = "pow2"
n_mode = "softplus"
mu_eps = 1e-6

optimizer = "adamax"
lr = 1e-3
weight_decay = 0.0
beta1 = 0.9
beta2 = 0.999
clip_norm = 0.5

obs_weight = 0.5
imp_weight = 8.0
count_weight = 1.0
```

---

## Implementation: depth-offset count head

This extends the E29 library-size offset idea ([`idea_e29_libsize_offset_nb.md`](idea_e29_libsize_offset_nb.md), [`libsize_offset_NB.md`](libsize_offset_NB.md)) for the CANDI v2 sandbox decoder. The default v2 `NegativeBinomialLayer` predicts μ directly from decoded features; depth enters only indirectly via FiLM. That fails Q5 on real chr19 (dcr ≈ 1, imp_p ≈ 0.12) unless an explicit depth scale is injected into the NB mean.

### Parameterization

For each sample `s`, assay `b`, position `p`:

- Network predicts **log-enrichment** η_{sbp} and dispersion-related **n**_{sbp} from decoded features.
- **Prompt depth** d_s = log₂(seq_depth) comes from **y_meta row 0** (output metadata — available even for masked/imputed assays).
- **Centered offset** c = `depth_center` (learned constant, not a network parameter in the winning config):

$$\mu_{sbp} = 2^{(d_s - c)} \cdot \exp(\eta_{sbp})$$

Then standard NB reparameterization:

$$p_{sbp} = \frac{n_{sbp}}{n_{sbp} + \mu_{sbp}}$$

with p clamped to `[eps, 1−eps]` and μ clamped below at `mu_eps`.

**Why center?** Raw `μ = 2^d · exp(η)` without centering fails on real EIC depths (~log₂ depth ≈ 24–27). Subtracting c ≈ 27 puts the scale factor near unity so η learns enrichment rather than absolute library size. Prior diagnostics used c ≈ 24; autoresearch moved the optimum to **27** on this pinned chr19 batch.

### Code structure (`train.py`)

The head is swapped in at runtime without modifying `sandbox/candi_v2/`:

```
CANDIv2 (frozen shell from prepare.py)
  └── decoder patched → V2DecoderDepthOffset
        └── neg_binom_layer → DepthOffsetNegativeBinomialLayer
              ├── linear_eta: decoded → η  [signal_dim × signal_dim]
              └── linear_n:   decoded → n  [signal_dim × signal_dim]
```

**`DepthOffsetNegativeBinomialLayer.forward(x, depth_log2)`**

1. η = linear_eta(x)
2. d = depth_log2.unsqueeze(1) − depth_center   # broadcast over positions × assays
3. μ = 2^d · exp(η)   (pow2 mode; linexp alternative tested and rejected)
4. n = softplus(linear_n(x)) + eps
5. p = n / (n + μ)

**`V2DecoderDepthOffset.forward(z, y_meta)`**

The v2 decoder normally computes count outputs inside `super().forward`. The patch:

1. Temporarily removes `neg_binom_layer` so the parent skip the default NB head.
2. Runs the standard decoder path (FiLM, shared deconv trunk) for signal/pval/peak if active.
3. Re-applies the custom NB head on the **shared trunk output**, conditioning on **y_meta[:, 0, :]** (depth row per assay).

This ensures imputed assays use **output depth** in the offset — the metadata path that carries depth for masked bins (see M08/M09 in [`META_CONDITIONING.md`](../diagnostics/META_CONDITIONING.md)).

**`patch_count_head(model, cfg)`**

At train start, replaces `model.decoder` with `V2DecoderDepthOffset`, copies compatible weights from the old decoder (`strict=False`), and moves to device. Head adds **144 parameters** (two linear layers on signal_dim); total model ~0.30M params.

### Alternatives tested in A2 (not kept)

| Mode | Formula | Result |
|------|---------|--------|
| **pow2** (winner) | μ = 2^(d−c) · exp(η) | 0.009298 |
| linexp | μ = exp(η + α·(d−c)), α ∈ {0.5, 0.693, 1.0} | 0.012–0.110 |
| n_mode=exp | n = exp(linear_n) | 0.0095 (near miss) |
| no offset | default v2 NB head | 0.499 (dcr collapse) |

---

## What was achieved?

On the **pinned chr19 single-batch overfit harness**:

| Metric | Baseline (depth_center=24) | Best |
|--------|---------------------------|------|
| composite_score | 0.111 | **0.0093** |
| imp_pearson | ~0.998 | ~0.998+ |
| dcr_masked_bins | ~4.14 | ~4.0+ (Q5 pass) |
| peak_vram_mb | ~110 | ~110 |

Qualitatively:

- **Q5 depth ratio** on masked bins is preserved (dcr ≈ 4) — the original motivation for E29 offset.
- **Imputation quality** (Pearson on masked assays) remains excellent.
- **Training knobs** (Adamax, high imp_weight, clip 0.5) strongly affect composite on this harness; defaults are far from optimal.

---

## What was *not* tested / known gaps

This search optimized a **single-batch overfit composite**. It did **not** validate:

| Gap | Why it matters |
|-----|----------------|
| Multi-batch / multi-locus generalization | Optimum may memorize one chr19 batch |
| Long training (3k–8k steps) | P2 late divergence documented elsewhere |
| NB calibration (coverage, C-index) | High imp_weight may sharpen point preds, miscalibrate uncertainty |
| x_meta → Z sensitivity | Composite penalizes low delta, but high imp_weight may collapse encoder metadata path |
| y_meta readlen / runtype probes | Not in composite; runtype unused in count head per M01–M10 |
| Production 42M CANDI | Sandbox shell is ~0.3M params |
| Architecture (FiLM placement, norm, gating) | Frozen; E23-scale ablations not in scope |

---

## Recommended next steps

1. **Promote head to v2 default** — centered depth offset with configurable `depth_center` (batch median or fixed 27); see E29 promotion path.
2. **Extend harness** — multi-batch chr19 cycling, train-loss curves, best-checkpoint restore, NB calibration metrics.
3. **Dual objective** — separate "fit" (current composite) from "science" (calibration + x_meta sensitivity + multi-batch imp_p).
4. **Targeted follow-up** — `n_mode=exp` at winning stack (only near-miss); joint (imp_weight, obs_weight, clip_norm) fine grid if another train.py-only session is cheap.
5. **Reset git HEAD** if on stale sweep commit: `git reset --hard b87156b7`.

---

## Reproduce

```bash
# Compute node
cd ~/projects/def-maxwl/mforooz/EpiDenoise
source candi_venv/bin/activate && module load samtools
export PYTHONPATH=$PWD

git checkout autoresearch/may28
git reset --hard b87156b7

python -m sandbox.diagnostics.autoresearch.train \
  > sandbox/diagnostics/autoresearch/run.log 2>&1

grep -E '^(composite_score|imp_pearson|dcr_masked|denoise_pearson|peak_vram_ok|status):' \
  sandbox/diagnostics/autoresearch/run.log
```

Expected: `composite_score ≈ 0.0093`, `status: ok`, `peak_vram_ok: True`.

---

## References

- Harness playbook: `sandbox/diagnostics/autoresearch/program.md`
- Prior chr19 diagnostics: `sandbox/diagnostics/FINDINGS.md`, `META_CONDITIONING.md`
- E29 theory: `sandbox/ideas/libsize_offset_NB.md`, `idea_e29_libsize_offset_nb.md`
- Full run log: `sandbox/diagnostics/autoresearch/results.tsv` (131 rows)

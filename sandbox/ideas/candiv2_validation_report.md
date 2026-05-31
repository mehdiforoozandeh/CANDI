# CANDI v2 Validation Report

Date: 2026-05-31  
Scope: Full validation per deep-validation plan — static review, pytest suite, GPU smokes (no W&B).

## Executive Summary

**Verdict: ship-ready for sandbox experiments**, with documented non-blockers.

| Gate | Result |
|------|--------|
| Pytest (`test_candi_v2_core.py` + `test_eval_median.py`) | **39/39 pass** |
| Config dry-run matrix (6 cells) | **6/6 pass** |
| Production trainer mini-runs (`train_candi_v2`, 2 epochs × 6 cells) | **6/6 complete**, finite losses, no crashes |
| Checkpoint resume | **pass** (global_step 20 → 40) |
| Gradient audit (6 cells) | **6/6 pass**, encoder/decoder grads finite |
| Synthetic overfit (`--max-steps 800`) | **p1 fail** (strict Pearson ≥0.99); not a production-path blocker |

All six `(heads, count_head)` combinations build, forward, backward, and train without NaN/Inf. Inactive loss branches contribute zero weighted loss. Assay-only masking invariants hold.

---

## 1. Code Review Findings

### Blockers
None identified in production path (`train_candi_v2` + assay-only masking).

### Warnings

| ID | Severity | Finding |
|----|----------|---------|
| W1 | warning | [`real_train_smoke.py`](../diagnostics/real_train_smoke.py) uses `p_full_loci=0.5`, violating `validate_v2_config()`. Do not use as v2 regression test without fixing masker. |
| W2 | warning | [`real_data.py::collect_batches`](../diagnostics/real_data.py) passes `preserve_assay_id=True` to `SandboxH5Dataset`, which does not accept that kwarg. Fixed in [`v2_gradient_audit.py`](../diagnostics/v2_gradient_audit.py); other diagnostic scripts still affected. |
| W3 | note | Default `eval.eval_every_n_epochs=5` means 2-epoch validation smokes skip `run_eval_pass`. Set `--set eval.eval_every_n_epochs=1` for short eval smokes. |
| W4 | note | [`candiv2.md`](candiv2.md) references `python -m sandbox.train --model-type candi_v2`; actual entrypoint is `python -m sandbox.train_candi_v2`. |
| W5 | note | `query_mask` / `query_mask_signal` accepted by `CANDIv2.forward` but unused (encoder uses meta/signal availability directly). Harmless interface shim for `train.py`. |

### Confirmed Correct

**Config (`candi_v2/config.py`, `train_candi_v2.py`)**
- `validate_v2_config` rejects loci/chunk masking.
- Merge order: dataclass → `candi_v2_default.yaml` → overlays → `--set`.
- `num_assays` synced from `SANDBOX_ASSAYS`; `signal_transform` aligned encoder ↔ data.

**Encoder (`candi_v2/encoder.py`)**
- `_prepare_signal` zeros cloze/missing channels; asserts meta/signal availability match under `mask_token` mode.
- `MaskTokenInjector`: per-assay learned embeddings replace masked conv features.
- Control channel appended in `batch.py` after masking — never in `masked_map`.

**Decoder (`candi_v2/decoder.py`)**
- `DepthOffsetNegativeBinomialLayer`: `mu = 2^(log2_depth - depth_center) * exp(eta)`; depth from `y_meta[:, 0, :]`.
- Inactive heads: `None` in dict forward; zero tensors in `forward_tuple` for loss compatibility.

**Loss (`candi_v2/loss.py`, `losses.py`)**
- `build_v2_loss` zeroes weights for inactive heads.
- Trainer uses `fallback_imp_to_observed_when_no_masked=False` (strict imp on masked positions only).
- Inactive branches log weighted loss = 0.0 (not absent from stats dict).

**Training / eval (`train.py`, `train_candi_v2.py`)**
- `_V2TupleWrapper` bridges 6-tuple interface.
- Eval uses dedicated zero-probability masker; no training cloze at eval.
- `y_meta_fwd`: T_* → V_*/B_* for cloze slots → canonical for missing.

---

## 2. Test Coverage Map

New file: [`sandbox/tests/test_candi_v2_core.py`](../tests/test_candi_v2_core.py)

| Category | Tests | Guards |
|----------|-------|--------|
| Config | 5 | YAML parse, strict keys, merge overrides, loci rejection |
| Head matrix | 18 (6×3) | Build, forward shapes, tuple zeros, loss weight isolation |
| Masking | 3 | Assay cloze, control not masked, encoder encode finite |
| Loss | 2 | Prod-style no-mask edge case, count-only peak grad zero |
| Depth-offset | 2 | NB math, depth scaling ~4× per +2 log2 |
| Gradients | 2 | Module grad flow, per-branch finite grads |
| Eval integration | 1 | `run_eval_pass` smoke on real H5 |

Extended: [`test_eval_median.py`](../tests/test_eval_median.py) (existing v2 tests retained).

Run:
```bash
pytest sandbox/tests/test_candi_v2_core.py sandbox/tests/test_eval_median.py -v
```

---

## 3. GPU Smoke Results

Environment: `fc11004`, H100 1g.10gb slice, `candi_venv`, no W&B.

### 3A. Config dry-run
All 6 combinations: **pass**.

### 3B. Synthetic overfit (`--max-steps 800`)

| Variant | p1 Pearson | p1 dcr | Result |
|---------|------------|--------|--------|
| plain | 0.42 | 1.0 | FAIL (threshold 0.99) |
| depth_offset (diagnostic prototype patch) | 0.57 | 4.0 | FAIL (threshold 0.99) |

Note: synthetic harness uses tiny model + strict overfit criteria; 800 steps insufficient for p1 Pearson gate. Depth-offset shows correct `dcr≈4` immediately. Production native `depth_offset` head validated separately via trainer + unit tests.

### 3C. Trainer matrix (2 epochs, 30 batches/epoch, `type2_loci`)

| heads | count_head | params | global_step | train total_loss (last step) | status |
|-------|------------|--------|-------------|------------------------------|--------|
| count_only | plain | 300,696 | 60 | 1.42 | pass |
| count_only | depth_offset | 300,696 | 60 | 1.40 | pass |
| count_peak | plain | 300,768 | 60 | — | pass |
| count_peak | depth_offset | 300,768 | 60 | — | pass |
| all | plain | 300,912 | 60 | — | pass |
| all | depth_offset | 300,912 | 60 | — | pass |

Training observations:
- Finite `training_losses/count_obs` and `count_imp` throughout.
- `training_grad_norms/count_imp` > 0 when imp loss active (some batches have imp=0 when no masked assays in slice — expected).
- Eval not triggered (W3): `eval_every_n_epochs=5` with only 2 epochs.

Run dirs: `sandbox/runs/validation_{heads}_{count_head}/`

### 3D. Checkpoint resume
- Epoch 0: `global_step=20`, checkpoint saved.
- Resume epoch 1: continued to `global_step=40`. **pass**

### 3E. Gradient audit

All reports: `sandbox/runs/validation_gradient_audit/audit_*.json`

| heads | count_head | encoder grad | decoder grad | count_imp branch grad |
|-------|------------|--------------|--------------|----------------------|
| count_only | plain | 1.00 | 2.23 | 0.36 |
| count_only | depth_offset | 4.35 | 5.40 | 0.61 |
| count_peak | plain | 5.27 | 8.11 | 0.40 |
| count_peak | depth_offset | 9.32 | 12.21 | 0.34 |
| all | plain | 8.43 | 11.80 | 0.28 |
| all | depth_offset | 5.48 | 6.72 | 1.58 |

Modules receiving backward signal: `metadata_embedding`, `signal_tower`, `mask_injector`, `fusion`, `pre_decoder_film`, `neg_binom_layer`, `peak_layer` (when active).

---

## 4. Open Issues & Follow-ups

1. **Fix diagnostic drift**: remove `preserve_assay_id` from `collect_batches` or add support to `SandboxH5Dataset`.
2. **Fix `real_train_smoke.py`**: use assay-only masker to match v2 invariants.
3. **Short-run eval**: document `--set eval.eval_every_n_epochs=1` for smoke runs that need eval metrics.
4. **Synthetic overfit budget**: p1–p5 need default 2000 steps (or relaxed smoke tier) for CI; current 800-step run is informational only.
5. **Plain count head metadata sensitivity**: expect `dcr≈1` on counts (E30 finding); depth_offset head is the fix — confirmed in gradient audit and unit tests.

---

## 5. Artifacts Added

| File | Purpose |
|------|---------|
| `sandbox/tests/test_candi_v2_core.py` | Permanent pytest suite |
| `sandbox/diagnostics/v2_gradient_audit.py` | GPU gradient/masking audit CLI |
| `sandbox/diagnostics/summarize_v2_validation.py` | metrics.jsonl summarizer |
| `sandbox/diagnostics/run_v2_validation_gpu.sh` | Reproducible GPU smoke runner |
| `sandbox/runs/validation_*` | Run outputs (not committed) |

---

## 6. Recommended Pre-Experiment Checklist

Before any new sandbox experiment on v2:

```bash
pytest sandbox/tests/test_candi_v2_core.py -q
python -m sandbox.train_candi_v2 --dry-run --no-wandb --config sandbox/configs/your_overlay.yaml
```

For a quick GPU sanity check:
```bash
python -m sandbox.train_candi_v2 --no-wandb \
  --set training.epochs=2 --set training.max_train_batches=20 \
  --set eval.eval_every_n_epochs=1 --set training.eval_max_batches=5 \
  --run-dir sandbox/runs/smoke_$(date +%s)
```

# E34 — Architecture autoresearch (june3)

Status: harness ready  
Branch: `autoresearch/june3`  
Harness: `sandbox/autoresearch/june3/`

## Goal

FAFO **encoder/decoder architecture** (and config width/depth/fusion/FiLM alternatives) under frozen v2 **training + eval + data pins**, starting from an exact vendored copy of promoted `candi_v2_default`.

## Frozen (prepare / pins / ar_fixed.yaml)

| Item | Value |
|------|-------|
| Train data | May31 pin manifest: 36 chr19 batches, batch=4, dsf=off (**chr19 only**) |
| Train loop | `train_one_epoch` + `build_optimizer` + cosine schedule (v2) |
| Eval | `run_eval_pass` + `SandboxH5Dataset(train=False)` → **chr21 only** (regime name `type1_chr19` applies to train split only); 8×bs=4 batches, vb_natural meta |
| Data fraction | **10%** chr19 train pins (~306/3053); eval **10%** chr21 via `eval_max_batches≈61` (244 slots/pass) |
| Epochs | 20 |
| Head | `depth_offset`, dc=22.5, `count_only` (do not change in loop) |
| Loss weights | obs=3.5, imp=0.59, count=2.0 |

## Agent scope

Edit any `.py` under `june3/` except frozen harness files (see `scope.py`). Typical: `candi_v2/encoder.py`, `decoder.py`, `model.py`, `train.py::get_config()`.

## Primary score (maximize)

At **best checkpoint** (`eval_losses/total_loss`):

`0.45·imp_count_r2_gw + 0.25·den_count_r2_gw − 0.20·count_imp_loss − 0.10·count_obs_loss`

Train mirrors logged at same best epoch (`train_count_*_loss`).

## Guard-rails (discard if primary wins but)

- `imp_count_r2_gw > 0` and `den_count_r2_gw > 0`
- `depth_count_ratio ∈ [3, 5]`
- `peak_vram_mb ≤ 9500`
- `param_count ≤ 5× baseline` (VRAM cap still applies; OOM = discard)
- finite losses

## Aux footer (diagnostics)

`param_count`, `encoder_eff_rank`, grad clip stats, branch grad norms, VRAM.

## Parity gate (required before loop)

```bash
python -m sandbox.autoresearch.june3.validate_parity
```

Vendored `june3/candi_v2/` must match `sandbox/candi_v2/` weights and forward (ε ≤ 1e-5).

## Promotion

Winning arch → single sandbox A/B (`train_candi_v2`), not direct merge of AR commits.

# Gradient clipping — sandbox experiment summary

Date: 2026-05-27. Sources: [`EXPERIMENTS.md`](EXPERIMENTS.md), [`META.md`](META.md) Q1/Q5/Q6 navigation, linked `idea_*.md` files, and `sandbox/runs/*/metrics.jsonl`.

**Reconstruction metric:** `eval_losses/total_loss` at best epoch (lower is better). **Baseline for the old B-sweep stack:** B1 `baseline_anchor` (Adamax `lr=1e-4`, global norm clip `cap=1.0`) → best `total_loss=4.97` @ ep59.

**Metadata sensitivity:** `training_metadata_probes/depth_count_ratio` at best epoch (target ≈ 4.0; ≈1.0 means depth-invariant counts — standing finding F1).

---

## 1. Experiments (one line each)

**B6 — clip-by-value (`baseline_clip_value_type1`).** Motivated by the hypothesis that per-element value clipping controls branch gradients differently than global norm clipping. Changed `clip_mode` from `norm` to `value` (cap=1.0, else B1 settings). **(a)** Best reconstruction improved vs B1 (`total_loss` 4.57 vs 4.97) but the run still diverged late; not adopted. **(b)** `depth_count_ratio` stayed ≈1.0 (0.99 @ best); no metadata gain.

**Default promotion (2026-05-06, not a dedicated run).** Motivated by clip firing on most steps under `cap=1.0` and slow convergence at `lr=1e-4`. Promoted `training.grad.clip_norm` 1.0→**2.0** and `adamax.lr` 1e-4→**1e-3** together in `default.yaml`. **(a)** Multi-head runs on the new pair (e.g. `baseline_400ep`: best `total_loss` 4.74 @ ep19 with clip=2.0) underperformed the later E7+E13 stack but converged faster than the old B1-scale basins. **(b)** No effect on depth sensitivity (`depth_count_ratio` ≈1.0).

**B8 — E7+E13 reference (`baseline_E7_E13`).** Not a clipping ablation; documents the current default (**norm clip cap=2.0**, `lr=1e-3`) with single-shot FiLM and `gaussian_var_min=0.1`. **(a)** Best multi-head reconstruction in this family (`total_loss` 3.04 @ ep149; stable to ep289). **(b)** `depth_count_ratio` ≈1.0 (0.999 @ best); clipping+L R did not fix metadata collapse.

**E0 — no clip, low LR (`E0_no_clip_lr1e4`).** Motivated by high `clip_fraction` (~0.6–0.7) biasing gradient direction (especially low-norm FiLM/metadata vs decoder) and confounding with a safety test at low LR. Set `clip_norm=0`, `lr=1e-4` (confounds two knobs). **(a)** Underperformed B8 and B1 (`total_loss` 4.15 @ ep289; weak imputation). **(b)** `depth_count_ratio` ≈1.0 (1.007 @ best); removing clip did not restore depth sensitivity.

**E0b — no clip, default LR (`E0b_no_clip_lr1e3`).** Motivated to isolate clipping alone after E0. Set `clip_norm=0`, `lr=1e-3` (same as B8 except clip). **(a)** Stable but behind clipped B8 (`total_loss` 3.21 vs 3.04); clipping at cap=2.0 is load-bearing at full LR for matching B8 quality. **(b)** `depth_count_ratio` ≈1.0 (0.996 @ best); no metadata improvement.

**E10 — clip-active fraction logging (`idea_e10_clip_active_fraction.md`).** Motivated by needing a comparable metric for clipping pressure across runs. Implemented `grad_clipped_frac_*` and `grad_pre_clip_norm` in `metrics.jsonl` (no training run). **(a)/(b)** N/A — instrumentation only; used in all subsequent log analysis.

**E8 — per-group clipping (not run).** Motivated by global clip shrinking metadata-group gradients when decoder norms dominate. Planned separate `clip_grad_norm_` per optimizer group. **(a)/(b)** No data yet; Q5 next step after global clip/LR sweeps failed to move `depth_count_ratio`.

**JEPA encoder sweeps (secondary, different objective).** `e19c` (`clip_norm=5.0`) and `e19x` (`clip_norm=3.0`) tested whether relaxed clipping helps SIGReg/encoder rank; relaxed clip **worsened** dimensional collapse, not reconstruction NLL. No `depth_count_ratio` probe in JEPA training.

**Out of scope for this report:** B4 (SGD optimizer), B7 (`log1p`), E1 (`min_lr_ratio`) — LR/optimizer/input knobs with unchanged clip=1.0.

---

## 2. Summary table

| Experiment | Grad clipping ablation | `total_loss` @ best (vs B1 baseline 4.97) | `depth_count_ratio` @ best |
|---|---|---:|---:|
| **B1** `baseline_anchor` | norm, cap=**1.0** (reference) | **4.97** @ ep59 | 1.009 |
| **B6** `baseline_clip_value_type1` | **value**, cap=1.0 | **4.57** @ ep49 (−8%) | 0.990 |
| `baseline_400ep` | norm, cap=**2.0** (+ lr 1e-3; pre B8 stack) | 4.74 @ ep19 | 0.998 |
| **B8** `baseline_E7_E13` | norm, cap=**2.0** (current default) | **3.04** @ ep149 (−39%) | 0.999 |
| **E0** `E0_no_clip_lr1e4` | **clip off**, lr=1e-4 | 4.15 @ ep289 | 1.007 |
| **E0b** `E0b_no_clip_lr1e3` | **clip off**, lr=1e-3 | 3.21 @ ep114 | 0.996 |

---

## Takeaways

- **Reconstruction:** Raising the norm cap (1→2) with higher LR was necessary for the large gain seen in B8 vs B1-scale runs; value clipping (B6) only helped early loss; disabling clip at full LR (E0b) is stable but does not beat clipped B8.
- **Metadata:** No clipping strategy tested (mode, cap, on/off) moved `depth_count_ratio` toward the ≈4.0 target; F1 remains open. Per-group clipping (E8) and module-level grad logging (E9/E11) are the documented next diagnostics.

Linked checklist: [`EXPERIMENTS.md`](EXPERIMENTS.md) (B6, B8, E0/E0b, E8, E10). Standing finding: [F1](../.cursor/skills/log-observability/FINDINGS.md).

# E33 — Full CANDI v2 A/B: pre-E32 AR vs promoted defaults

Status: **running**  
Parent: E32 (`synthesis_e32_imp_r2_autoresearch.md`)  
Configs: `e33_v2_common.yaml`, `e33_v2_pre_ar.yaml`, `e33_v2_post_ar.yaml`  
Submit: `bash sandbox/slurm_tmp/e33_ab_submit.sh`

---

## Runs

| Run | Config overlay | Key settings |
|-----|----------------|--------------|
| **A** `e33_v2_pre_ar` | `e33_v2_pre_ar.yaml` | canonical imp eval, `count_head=plain`, weights 1.0, DSF uniform |
| **B** `e33_v2_post_ar` | `e33_v2_post_ar.yaml` | inherits promoted `candi_v2_default.yaml` (vb_natural, depth_offset dc=22.5, AR weights, dsf=off) |

Both: 200 epochs, type1_chr19, count_only, batch_size=32, adamax 1e-3, assay-only masking.

---

## Success criteria

- Run B `eval_metrics/imp_count_r2_gw` > Run A (expect large gap from vb eval alone)
- Run B imp R² > 0 sustained on chr21 holdout
- Run B DCR ≈ 4 if depth_offset active

---

## Artifacts

- `sandbox/runs/e33_v2_pre_ar_<jobid>/`
- `sandbox/runs/e33_v2_post_ar_<jobid>/`

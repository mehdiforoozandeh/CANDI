# E33c — Post-AR stack + neutral loss weights

Status: **running**  
Parent: E33 post-AR (`e33_v2_post_ar_42988908`)  
Config: `e33_v2_common.yaml` + `e33_v2_post_ar_w1.yaml`  
Submit: `bash sandbox/slurm_tmp/e33_post_ar_w1_submit.sh`

---

## Single knob vs post-AR (42988908)

| field | post-AR | E33c |
|-------|---------|------|
| `training.loss_weights.count_weight` | 2.0 | **1.0** |
| `training.loss_weights.obs_weight` | 3.5 | **1.0** |
| `training.loss_weights.imp_weight` | 0.59 | **1.0** |

Everything else: promoted `candi_v2_default` (vb_natural, depth_offset dc=22.5, dsf=off, count_only).

---

## Hypothesis

AR-tuned weights helped early imp R² on the pin but post-AR **42988908** collapsed after ep44 (`imp_count_r2_gw` +0.16 → −2.0, `count_imp_loss` rising). Neutral 1/1/1 should reduce late magnitude drift while keeping depth_offset + vb eval.

---

## Success criteria

- Last-epoch `imp_count_r2_gw` ≥ post-AR ep199 (−2.01) and `count_imp_loss` ≤ post-AR ep199 (2.04)
- Best-epoch `imp_count_r2_gw` within ~0.05 of post-AR peak (0.162)
- DCR stays in [3.25, 4.75]

Compare with: `sandbox/runs/e33_v2_post_ar_42988908/`

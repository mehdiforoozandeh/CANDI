# Sandbox Default Config Promotions

Chronological log of promoted defaults in `sandbox/configs/default.yaml`, `sandbox/configs/jepa_default.yaml`, and related dataclasses. Linked from [`EXPERIMENTS.md`](EXPERIMENTS.md); do not duplicate tables in the checklist.

---

## E6 onwards — reconstruction stack (2026-05-06)

File: `sandbox/configs/default.yaml`

| Parameter | Old | New | Rationale |
|---|---|---|---|
| `training.optimizer.adamax.lr` | `1e-4` | `1e-3` | Previous runs under-converged in 400 epochs. |
| `training.grad.clip_norm` | `1.0` | `2.0` | Clip was active every step; relaxing gives room for useful steps. |
| `training.schedule.min_lr_ratio` | `0.1` | `0.1` (unchanged) | With lr=1e-3, final LR decays to 1e-4 — same absolute floor as old peak LR. |
| `model.single_shot_decoder_film` | `false` | `true` | E7 400ep: best multi-head architecture in sweep (F8). |
| `model.gaussian_var_min` | `1e-6` | `0.1` | E13: prevents variance collapse (F7 mitigated in pval-only isolation). |

E6/E7/E6E7 runs prior to 2026-05-06 used old defaults and are marked `incomplete` in the checklist.

Evidence: [`idea_e7_single_shot_decoder_film.md`](idea_e7_single_shot_decoder_film.md), [`synthesis_e13_var_floor.md`](synthesis_e13_var_floor.md), [`synthesis_e6_e7_film_ablation.md`](synthesis_e6_e7_film_ablation.md).

---

## E23 batch 2 — fresh encoder interim defaults (2026-05-16)

File: `sandbox/configs/jepa_default.yaml`, `JEPAModelConfig`

| Parameter | Old | New | Evidence |
|---|---|---|---|
| `fresh.film_mode` | `per_conv` | `pre_conv` | E23 batch 1: e23h = 0.7135 combined_loss, −3% vs per_conv, best geometry. |
| `fresh.transformer_type` | `dual` | `xtransformers` | E23 batch 1: e23d −17% pred_loss; stacks with pre_conv (e23i: 0.7269 vs e23a: 0.7353). |
| `fresh.cond_source` | `meta_tgt_embed` | `raw_meta_tgt` | E23 batch 2: e23i vs e23j = −2.5% combined_loss; adaLN_gamma 129.5 vs 0.4. |

Superseded in part by E23.5 promotions below. Evidence: [`synthesis_e23_encoder_ablation.md`](synthesis_e23_encoder_ablation.md), [`idea_e23_encoder_ablation.md`](idea_e23_encoder_ablation.md).

---

## E23.5 — JEPA encoder defaults (2026-05-18)

Files: `jepa_default.yaml`, `JEPAModelConfig`, `MaskingConfig`

| Parameter | Old | New | Evidence |
|---|---|---|---|
| `fresh.cond_source` | `raw_meta_tgt` | `meta_tgt_embed` | Proper categorical handling; raw treats assay_id as ordinal. |
| `fresh.cond_embed_shared` | `shared` | `separate` | E23 batch 3: LN helps encoder but kills predictor. |
| `fresh.meta_embed_layernorm` | `false` | `true` | Encoder FiLM benefits from LN (e23q vs e23t: 4% worse without). |
| `fresh.pred_meta_embed_layernorm` | (new) | `false` | Predictor embed: no LN (gamma 0.1→54 when removed). |
| `fresh.film_mode` | `pre_conv` | `per_conv_and_transformer` | e23f: runtype_last=0.168, 2.3× control. |
| `MaskingConfig.preserve_assay_id` | `false` | `true` | E23 batch 3: 0.3–5.3% consistent improvement. |
| `data.regime` | (unset) | `type2_loci` | E21: better UMAP structure than type1_chr19. |
| `jepa.pred_mask_cond_type` | `none` | `meta_tgt` | FJ7: meta_tgt is dominant metadata sensitivity lever. |
| `jepa.pred_cond_source` | (new) | `meta_tgt_embed` | Candi-path predictor: separate embed no-LN. |

Evidence: [`idea_e23_encoder_ablation.md`](idea_e23_encoder_ablation.md).

---

## Clean A/B — model_type default (2026-05-19)

File: `jepa_default.yaml`

| Parameter | Old | New | Evidence |
|---|---|---|---|
| `model_type` | `candi` | `fresh` | clean_ab: fresh wins combined_loss (0.7141 vs 0.7277) and geometry (cov=63.9 vs 81.2); CANDI retains runtype_best (0.751 vs 0.469), partially dropout-confounded. |

CANDI encoder remains available via `model_type=candi`. Evidence: [`synthesis_clean_ab_encoder.md`](synthesis_clean_ab_encoder.md).

---

## E24–E26 — fresh encoder structure (2026-05-21)

File: `jepa_default.yaml`, `JEPAModelConfig`

| Parameter | Old | New | Evidence |
|---|---|---|---|
| `fresh.missing_data_mode` | `mask_stem` | `mask_token` | E24: +2.6% combined_loss, +37% runtype_sens vs mask_stem. |
| `fresh.fusion_norm` | `layer` | `none` | E26: +1.1% combined_loss; transformer pre-norm sufficient. |

Rejected: `fresh.fusion_mode=gated` (E25) — +0.2% within noise, worse geometry and biology.

Evidence: [`idea_e24_dmodel_mask_token.md`](idea_e24_dmodel_mask_token.md), [`idea_e25_gated_dna_fusion.md`](idea_e25_gated_dna_fusion.md), [`idea_e26_remove_fusion_layernorm.md`](idea_e26_remove_fusion_layernorm.md).

---

## E32 — CANDI v2 count stack (2026-06-03)

File: `sandbox/configs/candi_v2_default.yaml`, `sandbox/candi_v2/config.py`, `sandbox/train.py::run_eval_pass`

| Parameter | Old | New | Rationale |
|---|---|---|---|
| `eval.use_canonical_missing_meta` | `true` | `false` | E32 A1: V/B natural metadata at imp-eval slots; canonical depressed imp R² ~0.28 vs vb. |
| `decoder.count_head` | `plain` | `depth_offset` | E30 validated; required for depth-aware calibration (E32 built on this). |
| `decoder.depth_center` | `24.0` | `22.5` | E32 AR best on pinned chr19/21 (E31 sweep pending full confirm). |
| `training.loss_weights.count_weight` | `1.0` | `2.0` | E32 AR winning recipe (`be0d38e2`). |
| `training.loss_weights.obs_weight` | `1.0` | `3.5` | E32 AR winning recipe. |
| `training.loss_weights.imp_weight` | `1.0` | `0.59` | E32 dominant lever; imp R² 0.063→0.122 after vb fix. |
| `training.dsf.sampling` | `uniform` | `off` | Align train with dsf=1 identity/den eval. |
| `decoder.heads` | `count_peak` | `count_only` | E32 AR count-axis focus; count_peak runs use config overlay. |
| `data.regime` | (unset in v2 default) | `type1_chr19` | Match E30/E31/E32 training regime. |

Not promoted: `lambda_mse_obs=0.2` (AR-only aux; rejected for default per user).

Evidence: [`synthesis_e32_imp_r2_autoresearch.md`](synthesis_e32_imp_r2_autoresearch.md), [`autoresearch_may31_r2vscorr_disparity.md`](autoresearch_may31_r2vscorr_disparity.md). Validation: E33 A/B (`e33_v2_pre_ar` vs `e33_v2_post_ar`).

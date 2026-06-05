# Meta — Open Research Questions

Headlines and navigation only. Experiments: [`EXPERIMENTS.md`](EXPERIMENTS.md). Detail: `idea_*.md`, `synthesis_*.md`, [`FINDINGS.md`](../../.cursor/skills/log-observability/FINDINGS.md).

---

<details>
<summary><strong>Q1 — Training stability: why do runs diverge late?</strong> · open</summary>

**Experiments:** B1 · B4 · B6 · B7 · E1 · E8 · E10 · B8 · E0 · E0b

**Detail:** B-sweep — `idea_b1_anchor.md` … `idea_b7_log1p_type1.md` · LR floor — [`synthesis_e1_e5_head_interference.md`](synthesis_e1_e5_head_interference.md) · clipping — [`grad_clipping_summary.md`](grad_clipping_summary.md)

</details>

<details>
<summary><strong>Q2 — Multi-head training: count / pval / peak cooperation vs interference</strong> · open</summary>

**Experiments:** E2 · E3 · E4 · E5

**Detail:** [`synthesis_e1_e5_head_interference.md`](synthesis_e1_e5_head_interference.md) · `idea_e2_head_count_only.md` · `idea_e3_head_pval_only.md` · `idea_e4_head_peak_only.md` · `idea_e5_head_count_peak.md`

</details>

<details>
<summary><strong>Q3 — Loss formulation: per-head likelihood and uncertainty weighting</strong> · open</summary>

**Experiments:** E13 · E14 · E15 · E16 · E29 · E30 · E31

**Detail:** variance floor — [`synthesis_e13_var_floor.md`](synthesis_e13_var_floor.md) · NB offset — [`idea_e29_libsize_offset_nb.md`](idea_e29_libsize_offset_nb.md) · v2 validation — [`idea_e30_v2_depth_offset_head.md`](idea_e30_v2_depth_offset_head.md) · center sweep — [`idea_e31_depth_center_sweep.md`](idea_e31_depth_center_sweep.md) · diagnostic — [`autoresearch_may28_count_head.md`](autoresearch_may28_count_head.md)

</details>

<details>
<summary><strong>Q4 — Conditioning architecture: FiLM + metadata routing</strong> · partially resolved (E7 promoted)</summary>

**Experiments:** E6 · E7

**Detail:** [`synthesis_e6_e7_film_ablation.md`](synthesis_e6_e7_film_ablation.md) · [`idea_e7_single_shot_decoder_film.md`](idea_e7_single_shot_decoder_film.md)

</details>

<details>
<summary><strong>Q5 — Metadata collapse: depth-of-coverage ignored</strong> · open (partial fix on v2)</summary>

**Experiments:** E6 · E7 · E8 · E9 · E11 · E29 · E30 · E31

**Detail:** F1 — [`FINDINGS.md`](../../.cursor/skills/log-observability/FINDINGS.md) · v2 offset — [`idea_e30_v2_depth_offset_head.md`](idea_e30_v2_depth_offset_head.md) · center sweep — [`idea_e31_depth_center_sweep.md`](idea_e31_depth_center_sweep.md) · grad diagnostics — `idea_e8_per_group_clip.md` · `idea_e9_grad_norm_breakdown.md` · `idea_e11_optimizer_group_grad_norms.md`

</details>

<details>
<summary><strong>Q6 — HPO: which knobs move the cornerstone metric?</strong> · open</summary>

**Experiments:** B1–B7 · E1 · E12

**Detail:** B-sweep — `idea_b*.md` · head leverage — [`synthesis_e1_e5_head_interference.md`](synthesis_e1_e5_head_interference.md)

</details>

<details>
<summary><strong>Q7 — Self-supervised pretraining: JEPA / SIGReg vs reconstruction-only</strong> · open</summary>

<details>
<summary>Stage 1 encoder · E17–E27 · e19 sweeps</summary>

**Detail:** [`synthesis_e19_jepa_stage1.md`](synthesis_e19_jepa_stage1.md) · [`synthesis_e19_jepa_lam_sweep.md`](synthesis_e19_jepa_lam_sweep.md) · [`synthesis_e19_cdef_sweep.md`](synthesis_e19_cdef_sweep.md) · [`synthesis_e19_ghjil_sweep.md`](synthesis_e19_ghjil_sweep.md) · [`synthesis_e19_kmnopar_sweep.md`](synthesis_e19_kmnopar_sweep.md) · [`synthesis_e19_sz_sweep.md`](synthesis_e19_sz_sweep.md) · [`idea_e19_jepa_frozen_decoder.md`](idea_e19_jepa_frozen_decoder.md) · [`idea_e27_lambda_sigreg_sweep.md`](idea_e27_lambda_sigreg_sweep.md)

</details>

<details>
<summary>Fresh encoder · E21–E26 · E23</summary>

**Detail:** [`idea_e21_jepa_model_first_principles.md`](idea_e21_jepa_model_first_principles.md) · [`synthesis_e21efg_diagnostic_sweep.md`](synthesis_e21efg_diagnostic_sweep.md) · [`synthesis_e21h_mnop_2x2.md`](synthesis_e21h_mnop_2x2.md) · [`synthesis_e23_encoder_ablation.md`](synthesis_e23_encoder_ablation.md) · [`synthesis_ab_encoder_dropout_comparison.md`](synthesis_ab_encoder_dropout_comparison.md) · E24–E26 — `idea_e24_dmodel_mask_token.md` · `idea_e25_gated_dna_fusion.md` · `idea_e26_remove_fusion_layernorm.md`

</details>

<details>
<summary>Stage 2 decoder · E28</summary>

**Detail:** [`idea_e28_jepa_decoder_training.md`](idea_e28_jepa_decoder_training.md) · [`synthesis_e28_jdec_vs_b8.md`](synthesis_e28_jdec_vs_b8.md)

</details>

</details>

<details>
<summary><strong>Q8 — Reference backbone: modular CANDI v2 vs production</strong> · open</summary>

**Experiments:** CANDI-v2 · E30

**Detail:** [`candiv2.md`](candiv2.md) · [`idea_e30_v2_depth_offset_head.md`](idea_e30_v2_depth_offset_head.md)

</details>

<details>
<summary><strong>Q9 — Depth-offset head: missing/cloze depth sentinels</strong> · partially resolved (idea_e30_v2_depth_offset_head.md, 2026-05-31)</summary>

Sentinel gate landed in `DepthOffsetNegativeBinomialLayer` (valid → offset; MISSING/CLOZE → `exp(η)` fallback). Eval prompt policy still open.

**Detail:** [`idea_e30_v2_depth_offset_head.md`](idea_e30_v2_depth_offset_head.md#spawned-design-questions)

</details>

<details>
<summary><strong>Q10 — Encoder vs decoder metadata geometry: control-channel pooling</strong> · resolved (idea_e30_v2_depth_offset_head.md, 2026-05-31)</summary>

**Detail:** [`idea_e30_v2_depth_offset_head.md`](idea_e30_v2_depth_offset_head.md#q10--encoder-vs-decoder-metadata-geometry-control-channel-pooling)

</details>

<details>
<summary><strong>Q12 — v2 architecture: what encoder/decoder changes move imp R² under fixed train/eval?</strong> · open</summary>

**Experiments:** E34

**Detail:** [`autoresearch_june3_arch.md`](autoresearch_june3_arch.md)

</details>

<details>
<summary><strong>Q11 — Imp count calibration: Pearson ~0.5 but R² ≈ 0 (rank vs magnitude)</strong> · partially resolved (synthesis_e32_imp_r2_autoresearch.md, 2026-06-02)</summary>

**Experiments:** E32

**Detail:** [`autoresearch_may31_r2vscorr_disparity.md`](autoresearch_may31_r2vscorr_disparity.md) · [`synthesis_e32_imp_r2_autoresearch.md`](synthesis_e32_imp_r2_autoresearch.md)

</details>

---

## Index

- Checklist: [`EXPERIMENTS.md`](EXPERIMENTS.md)
- Config promotions: [`config_promotions.md`](config_promotions.md)
- Cross-run syntheses: `synthesis_*.md`
- Standing findings F*: [`FINDINGS.md`](../../.cursor/skills/log-observability/FINDINGS.md)
- Hub conventions: [`sandbox-idea-hub`](../../.cursor/skills/sandbox-idea-hub/SKILL.md)

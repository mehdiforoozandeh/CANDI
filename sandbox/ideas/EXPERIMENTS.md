# Sandbox Experiments

Checklist index only. Each entry: status, linked title, one-line problem, one-line hypothesis, one-line finding (accepted / rejected / partial / not run). Detail, artifacts, and metrics live in linked `idea_*.md` or `synthesis_*.md`. Default config promotions: `[config_promotions.md](config_promotions.md)`.

## Status Legend

- `[ ] idea` — hypothesis only, not staged
- `[ ] staged` — ready to submit
- `[ ] running` — submitted or in progress
- `[x] done` — findings recorded
- `[x] incomplete` — useful artifacts but incomplete or missing ranking metadata
- `[x] superseded` — preserved; use successor

---

## Baseline Sweep: B1–B8

- `B1` - [Anchor: type1 chr19, raw input](idea_b1_anchor.md) (`done`)
  - Problem: Raw-input type1 chr19 reference before single-knob ablations.
  - Hypothesis: Baseline loop is stable enough to compare later ablations.
  - Findings: **Rejected as stable reference** — diverged after a good basin; weak depth response.
- `B2` - [DSF sampling off](idea_b2_dsf1_only.md) (`done`)
  - Problem: Does dynamic DSF sampling help or destabilize the type1 baseline?
  - Hypothesis: DSF-off isolates DSF contribution to instability.
  - Findings: **Rejected** — improved best quality_score but worse late divergence than B1.
- `B3` - [Assay masking only](idea_b3_assay_mask_only.md) (`done`)
  - Problem: Is full-locus masking too hard at sandbox scale?
  - Hypothesis: Assay-only masking improves assay-level learning vs mixed masking.
  - Findings: **Rejected** — modest peak gain but still diverged; not a stability fix.
- `B4` - [SGD lr 1e-4](idea_b4_sgd_lr1e4.md) (`done`)
  - Problem: Is Adamax driving late divergence?
  - Hypothesis: Conservative SGD should be slower but more stable.
  - Findings: **Accepted for stability** — no divergence; weak pval imputation; depth collapse persists.
- `B5` - [Type2 loci only](idea_b5_type2_loci_only.md) (`done`)
  - Problem: Does type2 loci regime behave differently from type1 chr19?
  - Hypothesis: Regime overlay exposes regime-specific failures or gains.
  - Findings: **Rejected** — weakest baseline run; poor imputation metrics.
- `B6` - [Clip by value](idea_b6_clip_value_type1.md) (`done`)
  - Problem: Does clip mode affect best-epoch quality and divergence?
  - Hypothesis: Value clipping controls branch gradients differently than norm clipping.
  - Findings: **Partial** — best B-sweep quality_score but still diverged; not adopted.
- `B7` - [Log1p type1 baseline](idea_b7_log1p_type1.md) (`done`)
  - Problem: Does log1p input scaling stabilize type1 vs raw input?
  - Hypothesis: log1p reduces count-scale pathologies without large quality loss.
  - Findings: **Accepted for stability** — no divergence; improved imputation vs B1; depth collapse persists.
- `B8` - [Baseline: E7 + E13 defaults](idea_b8_baseline_e7_e13.md) (`incomplete`)
  - Problem: No multi-head run has combined E7 FiLM + E13 variance floor.
  - Hypothesis: Combined defaults eliminate or delay pval_imp divergence and set new reference.
  - Findings: **Partial** — stable to ep289 without pval blow-up but below E7 400ep reference. → [idea_b8_baseline_e7_e13.md](idea_b8_baseline_e7_e13.md)

---

## Experiment Sweep: E0

- `E0` - [No gradient clipping, lr=1e-4](idea_e0_no_clip_lr1e4.md) (`incomplete`)
  - Problem: High clip_fraction may bias low-norm FiLM/metadata gradients.
  - Hypothesis: No clip + lower LR improves imputation if clipping was harmful.
  - Findings: **Rejected** — stable but strongly underperforms clipped B8; confounds clip and LR.
- `E0b` - [No gradient clipping, lr=1e-3](idea_e0b_no_clip_lr1e3.md) (`incomplete`)
  - Problem: E0 confounds clipping removal with 10× LR reduction.
  - Hypothesis: No clip at default LR isolates clipping effect alone.
  - Findings: **Rejected** — stable but trails clipped B8; clipping load-bearing at full LR. → [grad_clipping_summary.md](grad_clipping_summary.md)

---

## Masking Sweep: M1–M3

Same stack as B8; only `training.masking.p`_* changes. See linked idea files for `DataMasker` semantics.

- `M1` - [Full assay masking only](idea_m1_mask_assay_only.md) (`idea`)
  - Problem: Quantify cross-assay imputation under assay-only masking.
  - Hypothesis: Emphasizes assay completion vs B8's mixed masking.
  - Findings: Not run. Old-stack analogue: B3.
- `M2` - [Full loci masking only](idea_m2_mask_loci_only.md) (`idea`)
  - Problem: Quantify synchronous spatial holes with visible per-assay metadata.
  - Hypothesis: Stresses shared-locus structure vs default mixture.
  - Findings: Not run.
- `M3` - [Chunk masking only](idea_m3_mask_chunks_only.md) (`idea`)
  - Problem: Quantify independent per-assay spatial masking.
  - Hypothesis: Sharpens denoising or weakens cross-assay alignment.
  - Findings: Not run.

---

## Experiment Sweep: E1–E5

- `E1` - [Lower LR floor](idea_e1_lrfloor_low.md) (`incomplete`)
  - Problem: Is cosine LR floor too high and contributing to late divergence?
  - Hypothesis: Lower `min_lr_ratio` reduces late training pressure under log1p.
  - Findings: **Rejected under log1p** — tied B7 on all metrics; F3 mitigated. → [synthesis_e1_e5_head_interference.md](synthesis_e1_e5_head_interference.md)
- `E2` - [Count head only](idea_e2_head_count_only.md) (`done`)
  - Problem: Can count branch learn when pval and peak are muted?
  - Hypothesis: Multi-head competition hurts counts; isolation should lift count metrics.
  - Findings: **Partial** — count obs improves indefinitely; count imp plateaus ~1.92. → [idea_e2_head_count_only.md](idea_e2_head_count_only.md)
- `E3` - [Pval head only](idea_e3_head_pval_only.md) (`done`)
  - Problem: Isolate pval behavior without count/peak competition.
  - Hypothesis: Pval-only training should improve pval if gradients were diluted.
  - Findings: **Rejected** — variance collapse on obs, pval_imp explodes (F7 root cause). → [idea_e3_head_pval_only.md](idea_e3_head_pval_only.md)
- `E4` - [Peak head only](idea_e4_head_peak_only.md) (`done`)
  - Problem: Isolate peak behavior without count/pval competition.
  - Hypothesis: Peak-only clarifies whether AUROC is head-limited or objective-limited.
  - Findings: **Accepted** — healthiest head; strong isolation AUROC ceiling, no divergence. → [idea_e4_head_peak_only.md](idea_e4_head_peak_only.md)
- `E5` - [Count plus peak heads](idea_e5_head_count_peak.md) (`incomplete`)
  - Problem: Does muting pval improve count+peak objectives?
  - Hypothesis: Pval noise dominates; count+peak should beat full multi-head on those branches.
  - Findings: **Partial** — best count_imp but peak AUROC still needs pval gradients. → [synthesis_e1_e5_head_interference.md](synthesis_e1_e5_head_interference.md)

---

## Experiment Sweep: E6–E16

- `E6` - [Linear FiLM](idea_e6_linear_film.md) (`done`)
  - Problem: Exp FiLM scale can zero gradients when saturated.
  - Hypothesis: Linear FiLM preserves conditioning gradients.
  - Findings: **Rejected** — dominated by E7 on all multi-head metrics. → [synthesis_e6_e7_film_ablation.md](synthesis_e6_e7_film_ablation.md)
- `E7` - [Single-shot decoder FiLM](idea_e7_single_shot_decoder_film.md) (`done`)
  - Problem: Per-layer decoder FiLM is redundant and hard to attribute.
  - Hypothesis: One latent FiLM makes decoder a pure spatial upsampler.
  - Findings: **Accepted** — best multi-head run in sweep; promoted default (F8). → [idea_e7_single_shot_decoder_film.md](idea_e7_single_shot_decoder_film.md)
- `E8` - [Per-group gradient clipping](idea_e8_per_group_clip.md) (`idea`)
  - Problem: Global clip may starve metadata gradients via shared norm budget.
  - Hypothesis: Per-group clip preserves metadata updates while capping decoder spikes.
  - Findings: Not run.
- `E9` - [Per-module grad-norm logging](idea_e9_grad_norm_breakdown.md) (`idea`)
  - Problem: Global grad norm hides which modules dominate clipping.
  - Hypothesis: Module-level norms reveal metadata starvation.
  - Findings: Not run.
- `E10` - [Clip-active fraction logging](idea_e10_clip_active_fraction.md) (`done`)
  - Problem: Need durable metric for clipping pressure across runs.
  - Hypothesis: Clip-active fraction makes clipping comparable in log analysis.
  - Findings: **Accepted** — implemented in metrics.jsonl; no training run needed. → [idea_e10_clip_active_fraction.md](idea_e10_clip_active_fraction.md)
- `E11` - [Optimizer-group grad-norm logging](idea_e11_optimizer_group_grad_norms.md) (`idea`)
  - Problem: Per-group clipping needs pre/post-clip evidence per optimizer group.
  - Hypothesis: Group-level logging verifies whether E8 would help metadata.
  - Findings: Not run.
- `E12` - [Scheduler warmup comment cleanup](idea_e12_warmup_comment_cleanup.md) (`done`)
  - Problem: Cosine scheduler comments misread warmup vs start_factor.
  - Hypothesis: Comment fix removes ambiguity before schedule experiments.
  - Findings: **Accepted** — comments fixed; no behavior change. → [idea_e12_warmup_comment_cleanup.md](idea_e12_warmup_comment_cleanup.md)
- `E13` - [GaussianLayer variance floor](idea_e13_uncertainty_logvar_clamp.md) (`done`)
  - Problem: Near-zero variance floor allows pval obs/imp divergence (F7).
  - Hypothesis: `gaussian_var_min=0.1` prevents collapse without hurting quality.
  - Findings: **Accepted in pval-only isolation** — F7 mitigated; multi-head validation pending. → [synthesis_e13_var_floor.md](synthesis_e13_var_floor.md)
- `E14` - [Six uncertainty logvars](idea_e14_uncertainty_six_logvars.md) (`idea`)
  - Problem: One logvar per head ties obs/imp branches with different noise scales.
  - Hypothesis: Six logvars rebalance branches more cleanly.
  - Findings: Not run.
- `E15` - [Uncertainty optimizer-scheduler hardening](idea_e15_uncertainty_optimizer_scheduler_hardening.md) (`idea`)
  - Problem: Uncertainty params must attach to optimizer before scheduler captures base LRs.
  - Hypothesis: Ordering checks prevent silent scheduler drift.
  - Findings: Not run.
- `E16` - [Uncertainty variant documentation](idea_e16_uncertainty_variant_documentation.md) (`idea`)
  - Problem: One Kendall-Gal convention across regression and classification heads.
  - Hypothesis: Explicit documentation improves interpretability.
  - Findings: Not run.

---

## Experiment Sweep: E17–E20

- `E17` - [SIGReg latent regularizer](idea_e17_sigreg_latent_regularizer.md) (`idea`)
  - Problem: Reconstruction-only training may yield poorly conditioned latents.
  - Hypothesis: SIGReg auxiliary improves latent geometry without changing inference.
  - Findings: Not run.
- `E18` - [Joint JEPA assay-mask prediction](idea_e18_joint_jepa_assay_mask_prediction.md) (`idea`)
  - Problem: Raw likelihood may dominate before assay-completion structure emerges.
  - Hypothesis: Joint JEPA + assay masking improves imputation quality.
  - Findings: Not run.
- `E19` - [Encoder-only JEPA sweep](idea_e19_jepa_frozen_decoder.md) (`done`)
  - Problem: Can LeJEPA-style encoder pretraining yield usable latents for decoding?
  - Hypothesis: Sufficient λ, capacity, masking, and conditioning prevent collapse.
  - Findings: **Accepted for Stage 1** — λ=0.5 + pred_hidden=16 best; Stage 1 complete. → [synthesis_e19_sz_sweep.md](synthesis_e19_sz_sweep.md)
- `E20` - [JEPA encoder then low-LR fine-tuning](idea_e20_jepa_low_lr_finetune.md) (`idea`)
  - Problem: Frozen encoder may cap likelihood performance.
  - Hypothesis: Low-LR fine-tune improves reconstruction while preserving geometry.
  - Findings: Not run.

---

## Experiment Sweep: E21–E23

- `E21` - [Fresh JEPA model from first principles](idea_e21_jepa_model_first_principles.md) (`staged`)
  - Problem: Production CANDI coupling blocks encoder iteration (F1, import baggage).
  - Hypothesis: Purpose-built jepa_model.py matches e19q while enabling faster ablation.
  - Findings: **Partial** — fresh encoder is root cause of blob UMAPs (FJ15); see 2×2 matrix. → [synthesis_e21h_mnop_2x2.md](synthesis_e21h_mnop_2x2.md)
- `E22` - [Embedded predictor conditioning](idea_e22_embedded_predictor_conditioning.md) (`idea`)
  - Problem: Raw meta_tgt treats assay_id as ordinal at scale.
  - Hypothesis: MetadataEmbedding for predictor matches or beats e21o with proper categoricals.
  - Findings: Not run.
- `E23` - [Ablation-ready JEPA encoder redesign](idea_e23_encoder_ablation.md) (`done`)
  - Problem: Coupled architecture differences prevent root-cause attribution on fresh encoder.
  - Hypothesis: Single-knob toggles isolate causal knobs and recover geometry.
  - Findings: **Partial** — promoted several defaults; all 22 fresh runs fail v2 geometry gate. → [synthesis_e23_encoder_ablation.md](synthesis_e23_encoder_ablation.md)

---

## Experiment Sweep: E23.5

- `E23.5-H1` - [Best-of-all combo](idea_e23.5_h1_best_combo.md) (`done`)
  - Problem: CANDI enc + fresh xfm + pred_hidden=16 never tested together with embedded cond.
  - Hypothesis: Combo yields best JEPA checkpoint for Stage 2.
  - Findings: **Superseded** — covered by clean A/B pair. → [synthesis_clean_ab_encoder.md](synthesis_clean_ab_encoder.md)
- `E23.5-H2` - [DualAttention in fresh encoder](idea_e23.5_h2_dual_attention_fresh.md) (`done`)
  - Problem: All E23 fresh runs fail v2 gate; DualAttention untested as collapse fix.
  - Hypothesis: Production DualAttention block prevents dimensional collapse.
  - Findings: **Superseded** — deferred pending clean A/B encoder result. → [synthesis_clean_ab_encoder.md](synthesis_clean_ab_encoder.md)
- `E23.5-H4` - [Clean fresh baseline with all fixes](idea_e23.5_h4_clean_fresh_baseline.md) (`done`)
  - Problem: E23 runs used leaky MaskStem and stale defaults.
  - Hypothesis: Clean baseline establishes true fresh capability post-fix.
  - Findings: **Superseded** — covered by clean_ab_fresh_enc. → [synthesis_clean_ab_encoder.md](synthesis_clean_ab_encoder.md)

---

## JEPA A/B batches (unregistered)

- `ab_encoder_compare` — CANDI vs fresh, dropout=0.1 (`done`)
  - Problem: Encoder-type advantage under E23.5 defaults.
  - Hypothesis: CANDI retains biology; fresh wins geometry/combined_loss.
  - Findings: **Partial** — confounded by film_mode on fresh side. → [synthesis_ab_encoder_dropout_comparison.md](synthesis_ab_encoder_dropout_comparison.md)
- `ab_encoder_compare_dropout001` — same pair, dropout=0.01 (`done`)
  - Problem: Does lower dropout improve combined_loss acceptably?
  - Hypothesis: Less regularization tightens prediction quality.
  - Findings: **Rejected** — ~2.5% loss gain but 30–36% runtype drop; keep dropout=0.1. → [synthesis_ab_encoder_dropout_comparison.md](synthesis_ab_encoder_dropout_comparison.md)
- `clean_ab_candi_enc` — CANDI encoder, matched E23.5 defaults (`done`)
  - Problem: ab_encoder_compare confounded film_mode on fresh runs.
  - Hypothesis: CANDI reproduces runtype advantage under identical defaults (FJ15).
  - Findings: **Accepted for biology** — stronger runtype signal in clean pair. → [synthesis_clean_ab_encoder.md](synthesis_clean_ab_encoder.md)
- `clean_ab_fresh_enc` — fresh encoder, matched E23.5 defaults (`done`)
  - Problem: Same film_mode confound as ab_encoder_compare.
  - Hypothesis: Fresh closes gap or confirms encoder-type causality.
  - Findings: **Accepted for loss/geometry** — promoted model_type=fresh default. → [synthesis_clean_ab_encoder.md](synthesis_clean_ab_encoder.md)

---

## Experiment Sweep: E24–E30

- `E24` - [d_model-sized assay-specific mask token](idea_e24_dmodel_mask_token.md) (`done`)
  - Problem: Shared mask token collapses masked assay identities.
  - Hypothesis: Per-assay slices in one d_model token reduce aliasing.
  - Findings: **Accepted** — promoted default; geometry tradeoff noted. → [idea_e24_dmodel_mask_token.md](idea_e24_dmodel_mask_token.md)
- `E25` - [Gated DNA fusion](idea_e25_gated_dna_fusion.md) (`done`)
  - Problem: Static concat+project treats DNA equally at every position.
  - Hypothesis: Sigmoid gate improves position-specific fusion.
  - Findings: **Rejected** — no meaningful gain; worse biology and geometry. → [idea_e25_gated_dna_fusion.md](idea_e25_gated_dna_fusion.md)
- `E26` - [Remove fusion LayerNorm](idea_e26_remove_fusion_layernorm.md) (`done`)
  - Problem: Fusion LN may crush variance before transformer pre-norm.
  - Hypothesis: Removing fusion LN improves representation without instability.
  - Findings: **Accepted** — modest loss gain; promoted default. → [idea_e26_remove_fusion_layernorm.md](idea_e26_remove_fusion_layernorm.md)
- `E27` - [Lambda SIGReg sweep](idea_e27_lambda_sigreg_sweep.md) (`running`)
  - Problem: E24 improved loss but degraded geometry; λ may need retuning.
  - Hypothesis: Retuned λ recovers geometry under E24+E26 defaults.
  - Findings: Running. → [idea_e27_lambda_sigreg_sweep.md](idea_e27_lambda_sigreg_sweep.md)
- `E28` - [JEPA decoder training (Stage 2)](idea_e28_jepa_decoder_training.md) (`running`)
  - Problem: No signal-space decoder on top of JEPA latents yet.
  - Hypothesis: Frozen z_pred decoders match or beat end-to-end CANDI reconstruction.
  - Findings: **Rejected vs B8 so far** — trails B8; batch 2 tests predictor fine-tune. → [synthesis_e28_jdec_vs_b8.md](synthesis_e28_jdec_vs_b8.md)
- `E29` - [NB library-size offset](idea_e29_libsize_offset_nb.md) (`partially validated via E30`)
  - Problem: Count head ignores depth in mean; dcr ≈ 1 everywhere.
  - Hypothesis: Depth-offset parameterization restores depth sensitivity and count quality.
  - Findings: **Partial on v2** — mechanism validated via E30; production B8 stack still open. → [idea_e29_libsize_offset_nb.md](idea_e29_libsize_offset_nb.md)
- `E30` - [NB depth-offset head on CANDI v2](idea_e30_v2_depth_offset_head.md) (`done`)
  - Problem: E29 untested on real runs; autoresearch code not usable in sandbox.
  - Hypothesis: Native depth_offset head moves dcr above ~1 without regressing count imputation.
  - Findings: **Accepted** — offset wins A/B; promote as v2 default candidate. → [idea_e30_v2_depth_offset_head.md](idea_e30_v2_depth_offset_head.md)
- `E31` - [depth_center sweep on v2 depth-offset](idea_e31_depth_center_sweep.md) (`running`)
  - Problem: E30 used fixed c=24; optimal center vs EIC depth distribution unknown.
  - Hypothesis: A data-aligned depth_center matches or beats off-median values on count metrics and dcr.
  - Findings: **Not run** — 7 jobs submitted; blocks E29 closure until sweep completes. → [idea_e31_depth_center_sweep.md](idea_e31_depth_center_sweep.md)
- `E32` - [Autoresearch: imp count R² vs correlation disparity](autoresearch_may31_r2vscorr_disparity.md) (`done`)
  - Problem: Imp count Pearson/Spearman ~0.5 but R² ≈ 0; den R² also capped ~0.4 — rank–magnitude decoupling and possible eval/metadata bugs.
  - Hypothesis: Karpathy autoresearch with **gated primary** (den≥0.35 + DCR≈4 → maximize `imp_count_r2_gw`) will FAFO calibration losses, loss weights, DSF, architecture wrappers.
  - Findings: **Partial** — vb_natural eval + `imp_weight≈0.59` raised imp R² to **0.122** (below 0.15 validate gate); den peaked ~0.31. Promoted items 1–5 to `candi_v2_default.yaml`. → [synthesis_e32_imp_r2_autoresearch.md](synthesis_e32_imp_r2_autoresearch.md), [config_promotions.md](config_promotions.md)
- `E33` - [Full v2 A/B: pre-AR vs post-E32 defaults](idea_e33_v2_ar_ab.md) (`done`)
  - Problem: E32 gains were on a 5000-step pinned pin; need 200ep full-data confirmation.
  - Hypothesis: Promoted defaults reproduce positive imp R² on chr21 holdout vs pre-AR baseline.
  - Findings: **Partial** — post-AR peaks imp R² **0.162 @ ep44** then late collapse; pre-AR stable but canonical eval. → [idea_e33_v2_ar_ab.md](idea_e33_v2_ar_ab.md)
- `E33c` - [Post-AR + neutral weights (1/1/1)](idea_e33_v2_post_ar_w1.md) (`running`)
  - Problem: post-AR AR weights (2/3.5/0.59) may drive late imp R² and count_imp_loss divergence.
  - Hypothesis: obs=imp=count=1 on same post-E32 stack improves late-epoch stability without losing peak imp R².
  - Findings: Not run. → [idea_e33_v2_post_ar_w1.md](idea_e33_v2_post_ar_w1.md)
- `E34` - [Autoresearch: v2 architecture (june3)](autoresearch_june3_arch.md) (`planned`)
  - Problem: Loss/head AR (E32) saturated; architecture (fusion, FiLM, depth) unexplored under real v2 train+eval.
  - Hypothesis: Karpathy loop on vendored `candi_v2` with pinned train + production `run_eval_pass` finds imp R² gains without touching loss recipe.
  - Findings: Harness at `sandbox/autoresearch/june3/`; parity gate + 10ep baseline pending. → [autoresearch_june3_arch.md](autoresearch_june3_arch.md)

---

## CANDI v2

- `CANDI-v2` - [Modular reference implementation](candiv2.md) (`done`)
  - Problem: Production model.py blocks iteration (size, pval instability, coupling).
  - Hypothesis: Fresh encoder + configurable decoder provides clean modular backbone.
  - Findings: **Accepted** — validated; promoted reference for future sandbox work. → [candiv2.md](candiv2.md)


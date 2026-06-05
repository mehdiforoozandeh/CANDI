# E23 Encoder Ablation Sweep + e21o Reference

Status: synthesis (read-only)
Parents: [idea_e23_encoder_ablation.md](idea_e23_encoder_ablation.md), [idea_e21_jepa_model_first_principles.md](idea_e21_jepa_model_first_principles.md)
Linked from: EXPERIMENTS.md
Date: 2026-05-15

## Headline conclusions

1. **FiLM placement is the highest-impact encoder knob in E23.** Pre-conv FiLM (e23h) achieves the best combined_loss_scaled=0.7135 (−3.0% vs control), best SIGReg convergence (sigreg_loss=0.996), and lowest condition number (139) among all fresh runs. Per_conv_and_transformer (e23f) achieves the best metadata sensitivity retention (runtype_last=0.168, 2.3× control). Post-conv (e23c) is strictly worst (+7.8% combined_loss, −63% runtype_best). `Confidence: High` (clean single-axis deltas, consistent across metrics).

2. **mask_token is strictly harmful and should be rejected.** Switching from mask_stem to mask_token (e23b) raises pred_loss by 64%, lowers runtype_best by 74%, and kills predictor gamma (0.33 vs 2.17). The zero-fill + grouped conv approach dominates the post-conv learned embedding replacement on every metric. `Confidence: High` (clean single-axis, no competing explanation).

3. **xtransformers is marginally positive.** e23d matches control on combined_loss (−0.2%), achieves the best pred_loss_best=0.028 (−17%) and highest runtype_best=0.675 (+28%). Slight acceleration of enc_er collapse (17.4 vs 19.6). Net positive but not transformative. `Confidence: Medium` (small effect size on primary metric; pred_loss and runtype gains are clearer).

4. **DNA pooling order is neutral.** Early (e23g) vs late (e23a) differs by <2% on combined_loss. Marginal centering improvement (mean_norm 6.1 vs 6.4) and slightly better enc_er retention (20.3 vs 19.6) are the only visible signals. `Confidence: Medium` (small effects could vanish with different seeds).

5. **ALL E23 fresh encoder runs fail the v2 geometry gate due to universal dimensional collapse.** Every run ends with cov_condition_number > 100 (gate threshold: 50). The enc_er collapse (FJ5/FJ13) persists across all ablation variants — no single E23 knob prevents it. The CANDI encoder reference (e21o: cov_cond=52.9, enc_er=26.2 and rising) is qualitatively different. `Confidence: High` (9/9 gate failures, single reference PASS).

6. **The fresh encoder achieves comparable peak metadata sensitivity to CANDI, but cannot sustain it.** Peak runtype_best in E23: e23d=0.675, e23f=0.645 (vs e21o=0.751). But by training end, all E23 runs degrade to runtype_last < 0.17. The CANDI encoder retains 94% of its peak (0.708/0.751). Collapse is the bottleneck, not initial capacity. `Confidence: High` (pattern universal across all 8 E23 runs vs the stable e21o reference).

## Cross-run quantitative table

| Run | Ablation | combined ↓ | pred_loss ↓ | sigreg ↓ | enc_er ↑ (last) | cov_cond ↓ | runtype (best) ↑ | runtype (last) ↑ | gamma | v2 gate |
|-----|----------|-----------|-------------|----------|----------------|-----------|-----------------|-----------------|-------|---------|
| e23h | pre_conv FiLM | **0.7135** | 0.0340 | **0.996** | 20.1 | **139** | 0.482 | 0.036 | **5.87** | FAIL |
| e23f | per_conv+xfm FiLM | 0.7169 | 0.0326 | 1.055 | 14.5 ⚠ | 168 | **0.645** | **0.168** | 5.29 | FAIL |
| e23d | xtransformers | 0.7336 | **0.0279** | 1.047 | 17.4 | 179 | 0.675 | 0.094 | 0.54 | FAIL |
| e23a | CONTROL | 0.7353 | 0.0338 | 1.156 | 19.6 | 191 | 0.526 | 0.074 | 2.17 | FAIL |
| e23g | dna_early | 0.7481 | 0.0331 | 1.109 | **20.3** | 177 | 0.445 | 0.088 | 1.11 | FAIL |
| e23e | combo (b+c+d) | 0.7801 | 0.0311 | 1.195 | 16.2 | 149 | 0.110 | 0.019 | 0.31 | FAIL |
| e21o | CANDI ref | 0.7890 | 0.0276 | 1.945 | **26.2** ✓ | **52.9** | **0.751** | **0.708** ✓ | 123.2 | FAIL* |
| e23b | mask_token | 0.7912 | 0.0555 | 1.055 | 16.4 | 183 | 0.137 | 0.028 | 0.33 | FAIL |
| e23c | post_conv FiLM | 0.7925 | 0.0568 | 1.055 | 18.2 | 161 | 0.195 | 0.093 | 4.44 | FAIL |

Bold = best in column. ⚠ = below gate threshold (enc_er < 15). ✓ = healthy / near gate threshold.
*e21o FAIL is marginal: cov_cond=52.9 (gate=50), pred_slope=+0.43.

Ranking order is by `combined_loss_scaled` (primary), then `cov_condition_number` (tiebreaker 1), then `enc_er_last` (tiebreaker 2), per v2 ranking protocol.

## Per-experiment outcome vs hypothesis

| Run | Hypothesis | Outcome | Confidence |
|-----|-----------|---------|------------|
| e23b (mask_token) | BERT-style post-conv mask_embedding may give transformer a cleaner "unknown assay" signal | **Rejected** — all metrics worse; mask_stem's zero-fill+conv path is strictly superior | High |
| e23c (post_conv) | Post-tower single-shot FiLM may concentrate metadata signal effectively | **Rejected** — worst combined_loss, lowest runtype_best among FiLM variants; per-layer injection matters | High |
| e23d (xtransformers) | Pre-norm + RoPE may improve prediction quality | **Partial** — pred_loss improves 17%, runtype_best +28%, but enc_er collapse slightly worse; net marginally positive | Medium |
| e23e (combo) | mask_token + post_conv + xtransformers together may recover fresh-style quality | **Rejected** — negative effects of mask_token and post_conv dominate xtransformers' gains | High |
| e23f (per_conv+xfm) | Adding transformer FiLM increases conditioning capacity | **Confirmed** — best runtype retention (0.168 last), highest gamma (5.29), but accelerates enc_er collapse to 14.5 | Medium |
| e23g (dna_early) | Early DNA pooling may improve sequence-signal fusion | **Inconclusive** — <2% combined_loss difference; marginally better enc_er retention but marginal overall | Medium |
| e23h (pre_conv) | Pre-conv FiLM conditions input signal directly for better optimization | **Confirmed** — best combined_loss (0.7135), best SIGReg (0.996), best condition number (139), highest enc_er peak (40.3) | High |
| e21o (CANDI ref) | CANDI encoder + fresh predictor as quality ceiling for E23 | **Confirmed** — qualitatively different: no collapse, runtype_last=0.708 (4-38× E23), enc_er stable at 26.2 | High |

## Implications for next batch

Priority-ordered. Each is a single-axis or well-motivated combination.

1. **e23i: pre_conv + xtransformers** — Combine the two individually positive ablations (orthogonal mechanisms: FiLM placement vs transformer architecture). Predicted: combined_loss ≤ 0.72, pred_loss_best < 0.030. Cost: 1 run, ~3h.

2. **e23j: pre_conv + per_conv_and_transformer FiLM (FiLM-everywhere)** — Combine best optimization (pre_conv) with best metadata retention (transformer FiLM). Tests whether maximum FiLM capacity sustains runtype sensitivity without the enc_er collapse that e23f shows alone. Predicted: runtype_last > 0.15 if FiLM layers are additive. Cost: 1 run, ~3h.

3. **e23k: E23 control + fresh_transformer predictor + meta_tgt** — Isolate predictor contribution. The E23 runs all use legacy_mlp with no metadata conditioning; FJ7 shows meta_tgt is the dominant runtype lever. Adding it to the fresh encoder tests whether predictor-side metadata can compensate for encoder collapse. Predicted: runtype_last jumps to >0.3 if predictor sustains the signal encoder-collapse destroys. Cost: 1 run, ~3h.

4. **e23l: pre_conv + lambda_sigreg=0.75** — Fight the universal collapse directly. All E23 runs use λ=0.5; the cov_cond explosion (>100 in all runs) suggests SIGReg needs more weight. Predicted: cov_cond_last < 100 if λ is sufficient; may trade off pred_loss. Cost: 1 run, ~3h.

5. ~~**e23m: pre_conv + post-transformer LayerNorm**~~ — RETRACTED 2026-05-15. FJ12 candidate #1 was wrong: `CANDIJepa` calls `candi.encode()` which returns raw encoder output *before* `latent_projection`. Neither CANDI nor fresh encoder has LayerNorm before the JEPA projector. This was never a real difference. Slot freed for other experiments.

6. **e23n: pre_conv + xtransformers + fresh_transformer predictor + meta_tgt** — Best E23 encoder (from 1) + best predictor setup (from e21o). Full "best of everything" fresh configuration. Depends on results from (1) and (3). Cost: 1 run, ~3h.

## Standing findings (carried forward)

| Finding | Status | What this synthesis adds |
|---------|--------|------------------------|
| FJ5 (enc_er monotonic collapse) | **open** — universal in E23 | All 8 E23 runs reproduce the collapse. No single E23 knob prevents it. The collapse is intrinsic to the fresh encoder architecture, not a consequence of FiLM placement, transformer type, or missing-data mode. |
| FJ12 (fresh encoder 11 structural divergences) | **open** — partially resolved | E23 eliminates 4 candidates: mask_token (harmful, not helpful), post_conv FiLM (harmful), xtransformers (marginal), dna_pool_order (neutral). Remaining suspects: post-transformer LayerNorm, DualAttention vs XEncoder, cross-assay conv mixing. |
| FJ13 (fresh encoder burst-then-collapse) | **open** — confirmed in E23 | Pattern reproduced in all 8 runs: enc_er_best ranges 26–40, enc_er_last ranges 14–20. e23h has the most dramatic burst (40.3 peak) but still collapses. |
| FJ14 (AdaLN activation is encoder-dependent) | **open** — not directly tested | E23 uses legacy_mlp predictor, not the fresh transformer with AdaLN. The gamma_norm values in E23 (0.3–5.9) track MLP predictor activity, not AdaLN gating. |
| FJ15 (encoder confirmed as root cause) | **open** — reinforced | E23 demonstrates that 5 encoder-knob ablations cannot close the quality gap to CANDI. The gap is systemic, not attributable to any single E23-switchable component. |
| FJ7 (meta_tgt is dominant runtype lever) | **open** — not directly tested in E23 | All E23 runs use pred_mask_cond_type=none. The runtype signal comes solely from encoder-side FiLM, which achieves respectable peaks (0.48–0.68) but collapses. Predictor-side meta_tgt (as in e21o) may sustain the signal — proposed as e23k. |
| FJ9 (optimization pressure accelerates collapse) | **open** — consistent with E23 | e23f has the most FiLM capacity (highest gamma) and the fastest enc_er collapse (14.5). More conditioning = faster prediction optimization = faster collapse. |

## Caveats and limits

- **Single seed.** All results are from one random seed. The small effect sizes (xtransformers: −0.2% combined; dna_early: +1.7%) could vanish with different seeds.
- **Predictor confound vs e21o.** The CANDI reference (e21o) differs from E23 in model_type, predictor_type, AND pred_mask_cond_type. Direct metric comparisons (especially runtype_last) confound encoder architecture with predictor conditioning. Within-E23 comparisons are clean.
- **No trajectory plots.** matplotlib not available on the compute node; analysis uses terminal values and first/best/last summaries from extract_jepa_metrics.py. Convergence shape (early peaking, plateau position) could not be visually verified.
- **Short runs (90 training steps ≈ 142 epochs).** All E23 runs completed 90 logged training steps. The collapse may look different at 200+ epochs. e21o ran 107 steps (17 spike steps filtered).
- **Legacy MLP predictor in E23.** All E23 runs use the legacy_mlp predictor with no metadata conditioning (pred_mask_cond_type=none). The fresh transformer predictor + meta_tgt may interact differently with these encoder ablations.

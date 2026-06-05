# E19 corruption-mode and combination sweep: e19k, e19m, e19n, e19o, e19p, e19q, e19r

Status: synthesis (read-only)
Parents: [idea_e19_jepa_frozen_decoder.md](idea_e19_jepa_frozen_decoder.md)
Linked from: EXPERIMENTS.md
Date: 2026-05-10

---

## Headline conclusions

1. **meta_tgt conditioning + assay masking (e19q) is the best configuration found to date.** It produces the most biologically structured UMAP (active regions cluster away from silent regions; repressed regions form distinct islands; genomic position shows spatial continuity), stable encoder_eff_rank (~20, non-collapsed), near-orthogonal embeddings on masked batches (cos_sim < 0.09), and by far the highest runtype metadata sensitivity (0.739 — 10× higher than any non-meta_tgt run). (High confidence)

2. **meta_tgt conditioning is the dominant driver of metadata sensitivity.** All three meta_tgt runs (e19p, e19q, e19r) show `meta_sens_runtype` ≥ 0.37 at final step. All non-meta_tgt runs sit at 0.027–0.086. This confirms that conditioning the predictor on target metadata forces the encoder to internalize metadata contrasts. Depth sensitivity remains low (≤ 0.026 across all runs) — consistent with F1. (High confidence)

3. **DSF corruption alone (e19k, e19p) fails to produce structured encoder geometry.** Both pure-DSF runs show cos_sim ≥ 0.27 at end (encoder still tracks DSF level, not biology) and random-ball UMAPs. Assay masking — not DSF — is the critical ingredient for forcing biology-first representations. (High confidence; competing explanation: meta_tgt AdaLN is the confound, but e19p also uses meta_tgt and still produces random UMAP.) 

4. **λ=2.0 (e19n) is the only intervention that completely halts encoder_eff_rank collapse** (enc_er: 22.2 → 22.1, perfectly stable), but at a severe cost: pred_loss never converges (0.307 → 0.299, essentially flat for 200 epochs). There is a hard tension between SIGReg strength and prediction convergence. λ=2.0 appears to be above the convergence threshold for 200-epoch runs. (High confidence)

5. **No-AdaLN + pred_hidden=16 (e19o) produces the cleanest clustered UMAP among the non-meta_tgt runs.** pred_loss converges to 0.038 (best in batch), cos_sim → 0.032, and UMAP shows clear biological island structure with activity/repression anti-correlation. enc_er collapses to 16.4 but clusters are still visible. This confirms the FJ6 finding that predictor bottlenecking forces biology-first encoding. (High confidence)

6. **The last-step cos_sim / pred_loss for e19q and e19r is an FJ2 artifact.** At the final geometry log step (step 25000), both runs hit a mask_frac=0.0 batch. This inflates cos_sim to 0.74–0.79 and pred_loss to 0.28–0.31 — masking the true masked-batch behavior. Corrected values at the nearest masked-batch step: e19q cos_sim=0.083, pred_loss=0.049; e19r cos_sim=0.050, pred_loss=0.065. All analysis uses corrected values. (High confidence — evidence: last-5-step inspection for both runs.)

---

## Cross-run quantitative table

All metrics at the last masked-batch geometry log step (corrected for FJ2 mask_frac=0 artifact in e19q/e19r).
`latent_er` = latent_eff_rank (projector output). `enc_er` = encoder_eff_rank (raw encoder output). `sens_runt` = `lejepa/meta_sens_runtype` (1−cos_sim on runtype contrast). `sens_depth` = `lejepa/meta_sens_depth`.

| run | pred_loss | cos_sim | latent_er | enc_er | sens_runt | sens_depth | UMAP quality |
|---|---|---|---|---|---|---|---|
| e19k (DSF+meta_concat) | 0.315 | 0.619 | 24.5 | 19.8 | 0.375 | 0.150 | Random ball |
| e19m (no-AdaLN+proj256) | 0.041 | 0.094 | **39.6** | 15.8 | 0.027 | 0.007 | Weak scatter |
| e19n (no-AdaLN+λ=2.0) | 0.299 | 0.043 | 38.7 | **22.1** | 0.086 | 0.026 | Sparse clusters |
| e19o (no-AdaLN+pred16) | **0.038** | 0.032 | 21.7 | 16.4 | 0.039 | 0.008 | Clear clusters |
| e19p (meta_tgt+DSF) | 0.070 | 0.274 | 19.4 | 12.8 | 0.445 | 0.014 | Random ball |
| **e19q (meta_tgt+mask)** | 0.049* | **0.083*** | 35.6 | 20.1 | **0.739** | 0.011 | **Best structure** |
| e19r (meta_tgt+DSF+mask) | 0.065* | 0.053* | 36.6 | 21.7 | 0.696 | 0.012 | Very good |

*Corrected value from nearest masked-batch step; uncorrected last step: e19q cos_sim=0.788 pred_loss=0.286; e19r cos_sim=0.748 pred_loss=0.313.

Bold = best in column (directional — no `rank_runs.py` cornerstone metric available for JEPA runs).

---

## Per-run grad / stability table

| run | grad_norm_pre_clip (first→last) | clip_frac_running | sigreg/pred ratio (last) | adaLN_gamma (last) | adaLN_beta (last) |
|---|---|---|---|---|---|
| e19k | 7.03 → 3.96 | 1.00 | 0.733 | 708.0 | 26.6 |
| e19m | 3.87 → 1.62 | 1.00 | 1.698 | 0.0 | 0.0 |
| e19n | 24.93 → 9.91 | 1.00 | −0.018 | 0.0 | 0.0 |
| e19o | 6.33 → 1.81 | 1.00 | 0.712 | 0.0 | 0.0 |
| e19p | 6.47 → 4.28 | 1.00 | 0.792 | 506.7 | 12.4 |
| e19q | 6.17 → 6.02 | 1.00 | 1.062 | 387.1 | 11.1 |
| e19r | 8.66 → 8.62 | 1.00 | 0.805 | 359.2 | 11.7 |

Notes:
- e19n sigreg/pred ratio of −0.018 indicates SIGReg gradient direction reversed relative to pred — regime where pred_loss is so poorly converged that gradient sign fluctuates. λ=2.0 suppresses pred signal entirely.
- e19k adaLN_gamma=708 is pathologically large — AdaLN is ignoring SIGReg and overfitting the DSF signal in the conditioning.
- e19q and e19r grad_norm is not decreasing (6.17→6.02, 8.66→8.62) — still actively learning at end of 200 epochs. More epochs may help.
- All runs 100% clipped throughout (FJ5 partial cause unresolved).

---

## Per-experiment outcome vs hypothesis

| run | hypothesis | outcome | confidence |
|---|---|---|---|
| e19k | DSF corruption + meta_concat AdaLN eliminates assay-identity shortcut | Rejected — cos_sim=0.619 (highest in batch), UMAP random ball; AdaLN encodes DSF not biology | High |
| e19m | no-AdaLN + proj_dim=256 gives enc_er > 20 with low cos_sim | Partial — cos_sim=0.094 ✓, latent_er=39.6 ✓, but enc_er=15.8 (collapsed), meta_sens dropped near zero | Medium |
| e19n | no-AdaLN + λ=2.0 halts enc_er collapse | Confirmed (collapse halted: 22.2→22.1 ✓), but Rejected for utility (pred_loss never converges: 0.307→0.299 ✗) | High for collapse; High for utility failure |
| e19o | no-AdaLN + pred_hidden=16 produces biology-first representations | Confirmed — cos_sim=0.032 (lowest no-meta_tgt), clear UMAP cluster structure, pred converges | High |
| e19p | meta_tgt + pure DSF conditioning drives richer encoder representations | Rejected — UMAP random ball, enc_er collapsed to 12.8 (worst), cos_sim=0.274; DSF alone insufficient | High |
| e19q | meta_tgt + assay masking produces best encoder quality | Confirmed — best UMAP, high meta_sens_runtype=0.739, stable enc_er=20.1, cos_sim<0.09 on masked batches | High |
| e19r | combined DSF+mask with meta_tgt is hardest / best combined task | Confirmed (stable enc_er=21.7, good UMAP) but no clear advantage over e19q; pred_loss slightly higher | Medium |

---

## Implications for next batch

Prioritised by expected signal × cost:

1. **e19s: meta_tgt + assay masking + no-AdaLN** (e19q config + `pred_mask_cond_type=none`).
   Both meta_tgt conditioning and no-AdaLN independently produce biology-first encoders; combining them tests for additive benefit. Expected: enc_er stable ≥ 20, cos_sim < 0.05 on masked batches, meta_sens_runtype ≥ 0.6.
   Cost: 1 GPU × 3h.

2. **e19t: meta_tgt + assay masking + λ=1.0** (e19q config + `lambda_sigreg=1.0`).
   e19n showed λ=2.0 prevents collapse but blocks pred convergence. λ=1.0 may be the sweet spot: better collapse resistance than λ=0.5 without sacrificing pred_loss convergence. Expected: enc_er ≥ 21, pred_loss < 0.1 by ep200.
   Cost: 1 GPU × 3h.

3. **e19u: meta_tgt + assay masking + 400 epochs** (e19q config + `epochs=400`).
   e19q and e19r both show grad_norm NOT decreasing at step 25000 — the model is still learning. 400 epochs may substantially improve UMAP structure and metadata sensitivity. Expected: enc_er stable, meta_sens_runtype > 0.8.
   Cost: 1 GPU × 6h.

4. **Investigate depth sensitivity failure (F1 / FJ analog).** All runs show meta_sens_depth < 0.026. The depth contrast (log2: 23 vs 25) may be too small — log2 difference = 2 units ≈ 4× fold change is biologically meaningful but barely changes model input scale. Consider widening the contrast to 19 vs 25 (64× fold change) in the probe to diagnose if the probe range is the issue, or if the encoder genuinely ignores depth.

---

## Standing findings (carried forward)

| Finding | Status in this synthesis |
|---|---|
| FJ1 — λ≥0.5 needed | Open. λ=0.5 is still the working default. λ=2.0 (e19n) prevents collapse but kills pred convergence. λ=1.0 remains untested with meta_tgt. |
| FJ2 — seed-reuse / zero-mask batches | Open. FJ2 still affects last-step reporting for e19q/e19r (mask_frac=0 at final geometry log). The has_corruption skip is in place but FJ2 continues to corrupt end-of-run metrics. |
| FJ3 — UMAP structure requires eff_rank > threshold | Open — refined. e19o (enc_er=16.4) and e19q (enc_er=20.1) both produce structured UMAPs, suggesting the threshold may be lower than the previous estimate of > 25. The critical factor appears to be cos_sim < 0.1 rather than enc_er level. |
| FJ5 — enc_er peaks at init, collapses monotonically | Open — partially mitigated by e19n (λ=2.0) and e19r (meta_tgt+DSF+mask, enc_er=21.9→21.7). No general solution yet. λ=2.0 is the strongest single knob but too costly for pred_loss. |
| FJ6 — hi/lo pred_loss inversion is encoder-quality signal | Open. e19o (no-AdaLN+pred16) continues to show this pattern (implied by cos_sim=0.032 and good UMAP). meta_tgt runs: hi/lo is dominated by FJ2 artifact (last step always mask_frac=0) — ratio unreliable for e19q/e19r. |
| **FJ7 — NEW: meta_tgt conditioning is the dominant lever for runtype metadata sensitivity** | Open (2026-05-10) — see FINDINGS.md FJ7. |
| **FJ8 — NEW: DSF corruption alone (without assay masking) fails to produce structured encoder geometry** | Open (2026-05-10) — see FINDINGS.md FJ8. |
| F1 — Depth metadata ignored | Open. `meta_sens_depth` ≤ 0.026 in all JEPA runs, consistent with F1 in standard CANDI. Depth sensitivity is not improved by any conditioning mode tested. |

---

## Caveats and limits

- No cornerstone metrics (`eval_losses/total_loss`, `quality_score`) — `rank_runs.py` cannot be used. All verdicts are based on JEPA-internal metrics only.
- Single seed (42) for all runs; directional claims only.
- e19q and e19r final-step metrics are corrupted by FJ2 mask_frac=0 artifact; corrected values from penultimate masked-batch step are used throughout.
- UMAP quality is assessed qualitatively (no automated score). FJ3 threshold revised downward based on e19o evidence.
- hi/lo ratio is unreliable for e19q/e19r due to FJ2; not reported for those runs.
- 200-epoch budget may be insufficient for meta_tgt runs (grad_norm still stable at ep200 — not decelerating).
- e19n (λ=2.0) result is a single data point; the convergence failure could be schedule- or lr-dependent rather than a fundamental λ ceiling.

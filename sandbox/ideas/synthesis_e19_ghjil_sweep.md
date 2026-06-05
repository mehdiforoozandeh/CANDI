# E19 biology-focus sweep: e19g–e19l (5 completed runs, e19k failed)

Status: synthesis (read-only)
Parents: [idea_e19_jepa_frozen_decoder.md](idea_e19_jepa_frozen_decoder.md)
Linked from: EXPERIMENTS.md
Date: 2026-05-09

---

## Headline conclusions

1. **No-AdaLN (e19l) and small predictor (e19h) are the strongest encoders** — lowest cos_sim_ctx_tgt
   (0.014 and −0.011), meaning context and target embeddings became nearly orthogonal; encoder cannot
   shortcut via input identity. (High confidence — corroborated by hi/lo mask inversion below.)

2. **A new inversion metric reveals encoder-quality tier.** When the predictor is constrained or blind,
   pred_loss at *high* masking becomes **lower** than at *low* masking (hi/lo ratios: e19l=0.702,
   e19h=0.746, e19g=1.641). Ratios < 1 mean the encoder encodes heavy-masking contexts *more*
   informatively than light-masking contexts — the hallmark of mask-invariant biology-first representations.
   e19g's ratio of 1.641 shows it still uses identity shortcuts. (High confidence)

3. **proj_dim=256 (e19g) improves latent_eff_rank 2× in projection space (49.5 vs ~21–23) but barely
   moves encoder_eff_rank (19.6 vs ~18–22).** SIGReg with larger proj space successfully regularises the
   *projection*, but the encoder itself collapses equally fast. SIGReg must act on the encoder directly —
   a larger proj_dim alone is insufficient. (High confidence; competing explanation: proj collapse
   stabilises independently given enough SIGReg, but 200 epochs is not enough.)

4. **Loci masking (e19j) produces the best total_loss (0.589) but the worst encoder_eff_rank (12.7).**
   Loci masking does not remove assays, so the encoder still exploits assay-identity as a shortcut
   (cos_sim=0.256, worst in sweep). The low total_loss is driven by strong sigreg_loss (1.02 — lowest),
   not by good prediction. Loci masking is geometrically harmful in its current form. (High confidence)

5. **e19k failed at startup due to CLI config-parse bug** (`dsf_list=[4]` passed as a string literal;
   `_coerce_scalar` tries `int("[4]")` → `ValueError`). Fixed to `dsf_list=4`. Must be re-submitted.
   (Confirmed, exit_code=1 at 155 s.)

6. **FJ5 is universal across all 5 completed runs.** eff_rank peaks at init (40–126 depending on proj_dim)
   and collapses monotonically. No new run halted or reversed this collapse. (High confidence)

---

## Cross-run quantitative table

All metrics are averages over the last 10% of logged training_step rows.

| run | total_loss | pred_loss | sigreg_loss | cos_sim | latent_er | enc_er | AdaLN β | AdaLN γ | hi/lo ratio |
|---|---|---|---|---|---|---|---|---|---|
| e19b (baseline) | 0.762 | 0.102 | 1.319 | 0.242 | ~30 (W&B) | — | — | — | — |
| e19g proj_dim=256 | 0.761 | 0.059 | 1.405 | **0.091** | **49.5** | 19.6 | 2.83 | 4.13 | 1.641 |
| e19h pred_hid=16 | 0.614 | 0.040 | 1.147 | **−0.011** | 22.4 | 17.8 | 4.83 | 5.00 | **0.746** |
| e19i min30pct | **0.596** | **0.036** | 1.120 | 0.054 | 21.1 | 15.9 | 4.08 | 7.00 | N/A (mask_frac<0.5 always) |
| e19j loci | 0.589* | 0.079 | **1.019** | 0.256 | 20.0 | 12.7 | 1.17 | 4.27 | N/A (mask_frac=0) |
| e19l no-AdaLN | 0.625 | 0.048 | 1.155 | 0.014 | 23.3 | **18.4** | 0.00 | 0.00 | **0.702** |

*e19j total_loss is best numerically but driven by low sigreg_loss, not good prediction — see caveat below.

Bold = best in column (excluding e19b where metrics were not logged in jsonl).

---

## Per-run grad / stability table

| run | grad_norm_pre_clip (first→last) | clip_frac_running | sigreg/pred ratio (last) |
|---|---|---|---|
| e19g | 9.7 → 3.6 | 1.00 | 1.54 |
| e19h | 7.6 → 5.1 | 1.00 | 1.14 |
| e19i | 7.1 → 4.0 | 1.00 | 1.13 |
| e19j | 5.7 → 5.2 | 1.00 | 1.04 |
| e19l | 8.3 → 3.9 | 1.00 | 1.19 |

All runs remain 100% clipped throughout. Gradient clipping saturation is a persistent standing concern
(FJ5 partial cause). No run resolved this.

---

## Per-experiment outcome vs hypothesis

| run | hypothesis | outcome | confidence |
|---|---|---|---|
| e19g | proj_dim=256 fixes collapse without higher λ | Rejected — enc_er 19.6 (same as others); latent_er improves in proj space only | High |
| e19h | pred_hidden=16 forces encoder mask-invariance | Confirmed — cos_sim −0.011 (lowest), hi/lo=0.746 (<1.0), second-best enc_er | High |
| e19i | min30pct reduces identity shortcut | Partial — lowest pred_loss (0.036) but enc_er collapses most (15.9), AdaLN gamma=7.0 (overfitted conditioner) | Medium |
| e19j | loci masking eliminates assay-identity shortcut | Rejected — cos_sim 0.256 (worst), enc_er 12.7 (worst); assay identity still exploited | High |
| e19l | no AdaLN forces mask-invariant representations | Confirmed — cos_sim 0.014 (second best), hi/lo=0.702 (<1.0), second-best enc_er (18.4) | High |
| e19k | DSF corruption with meta_concat AdaLN | Failed (startup crash, CLI bug) | — |

---

## Implications for next batch

Prioritised by expected signal × cost:

1. **e19m: no-AdaLN + proj_dim=256** (2 changes: e19l ∩ e19g) — one axis from the best-cos_sim run,
   one axis from the best-latent_er run. Predicts: enc_er > 20, latent_er > 40, hi/lo < 1.0.
   Cost: 1 GPU × 3h.

2. **e19n: no-AdaLN + lambda=2.0** — test if stronger SIGReg combined with maximum encoder pressure
   halts enc_er collapse. e19l shows enc_er 18.4 with λ=0.5; doubling λ may push it above 25.
   Predicts: enc_er > 22, sigreg_loss < 0.95, cos_sim < 0.01.
   Cost: 1 GPU × 3h.

3. **Re-submit e19k** with fixed `dsf_list=4` (CLI bug fixed). DSF corruption remains untested and is a
   conceptually distinct corruption mode.
   Cost: 1 GPU × 3h.

4. **e19o: pred_hidden=16 + lambda=2.0** — the two highest-signal single knobs combined. Predicts: enc_er
   stable > 20, cos_sim < −0.02, hi/lo < 0.8.
   Cost: 1 GPU × 3h.

---

## Standing findings (carried forward)

| Finding | Status in this synthesis |
|---|---|
| FJ1 — λ≥0.5 needed | Open — still holding; λ=0.5 is minimum baseline but not sufficient alone. |
| FJ2 — seed-reuse spikes | Mitigated — training-level skip of has_corruption=False active in all e19g-l runs. e19j shows mask_frac=0.0 throughout (expected: loci masking doesn't set assay mask_frac). |
| FJ3 — UMAP structure requires eff_rank>threshold | Open — user reports "interesting results" on UMAPs/PCAs; e19h and e19l are most likely to show biological structure given cos_sim data. Awaiting visual confirmation. |
| FJ5 — eff_rank peaks at init, collapses monotonically | Open — confirmed in ALL 5 new runs. e19g starts at 126 (proj_dim=256) and collapses to 49.5. No intervention reverses the trend. |
| **FJ6 — NEW** | **Open (2026-05-09)** — hi/lo pred_loss inversion is a reliable encoder-quality signal: ratios < 1.0 indicate mask-invariant biology-focused representations; ratios > 1.0 indicate identity shortcutting. e19h and e19l both < 1.0. See FJ6 in FINDINGS.md. |

---

## Caveats and limits

- All metrics from `metrics.jsonl` `training_step` rows; only ~90 geometry log rows per run (logged
  every ~278 steps). First/last comparisons use 10% quantile windows (9 rows).
- Standard cornerstone metrics (`eval_losses/total_loss`, quality_score) are not emitted by
  `sandbox.train_jepa` — cannot run `rank_runs.py`. JEPA-internal metrics (`lejepa/*`) are the only
  quantitative basis for this synthesis.
- Single seed (42) for all runs; findings may not generalise.
- e19k data is entirely absent — DSF corruption hypothesis untested.
- hi/lo mask ratio has only 5–6 data points per run (steps where mask_frac > 0.5). Low statistics;
  treat as directional.
- UMAP/PCA visual quality not quantitatively scored here (no automated metric). FJ3 threshold
  (eff_rank > 25 dual criterion) based on e19b qualitative observation.

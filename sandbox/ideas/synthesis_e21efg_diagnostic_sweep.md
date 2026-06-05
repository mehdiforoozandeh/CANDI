# E21 Diagnostic Sweep (e21e / e21f / e21g) — Predictor Conditioning Failure Analysis

Status: synthesis (read-only) — **IMPORTANT: see Addendum (2026-05-14) below; gamma_norm values in this doc are on the OLD (incorrect) scale**
Parents: [idea_e21_jepa_model_first_principles.md](idea_e21_jepa_model_first_principles.md)
Linked from: EXPERIMENTS.md
Date: 2026-05-14

---

## Headline conclusions

1. **The predictor conditioning (AdaLN gamma) is the primary diagnostic lever separating candi from all fresh runs.** e21a (candi) ends with `gamma_norm_last=730`; all fresh runs end ≤ 155. e21b and e21g end at ≈ 0.1 — the AdaLN weights never meaningfully escape their zero initialization. Without active conditioning, the predictor cannot force the encoder to encode biology, explaining blob UMAPs. **Confidence: High** (gamma is logged across all runs; the pattern is consistent and monotone).

2. **Raw `meta_tgt` conditioning (e21f) partially rescues predictor activity.** Switching from an embedded `meta_tgt` to the raw covariate values raises `gamma_norm_last` from 0.1 (e21b/e21g) to **155.8** (e21f) and from 35.2 (e21e MLP+embed) to 155.8. e21f also has the highest `runtype_sens_best` (0.7016) and `runtype_sens_last` (0.5802) in the fresh group, and produces the best UMAP structure of the three diagnostic runs. This matches FJ7 — raw covariate conditioning provides a shorter, more direct gradient path to the AdaLN parameters. **Confidence: High**.

3. **Transformer predictor + AdaLN-zero initialization is the architectural dead-end.** e21g (transformer + separate embed) has `gamma_norm_last=0.158`, identical to e21b (transformer + shared embed). Decoupling the embedding path makes no difference when the transformer's AdaLN-zero gating is the dominant bottleneck. The zero initialization means the initial gradient to `cond_mod` is pure noise from the frozen-scale transformer blocks. **Confidence: High** (e21b vs e21g is a clean controlled pair; both gamma_norms are effectively zero).

4. **All fresh encoders pass the two-signal geometry gate but still produce blob UMAPs.** enc_er_last for fresh runs is 29.9–31.6 (above the ≥18 threshold); cos_sim_best_filtered is −0.04 to +0.03 (below 0.15). The candi encoder passes the same gate at enc_er=20.1, cos_sim=0.035 — but with `gamma_norm=730`, it produces biological islands. This is the first evidence that the two-signal gate is **necessary but not sufficient** for structured geometry: a third criterion, `gamma_norm_last ≥ 100`, appears required. New Standing Finding FJ11. **Confidence: High** (5 runs confirm the pattern).

5. **The untested combination — MLP predictor + raw `meta_tgt` (R5) — is now the highest-priority next run.** e21e (MLP + embed) reaches gamma=35; e21f (transformer + raw) reaches gamma=155. The natural product of both interventions — MLP architecture with its single direct AdaLN layer receiving raw covariate signals — should approach or exceed candi's gamma=730 and produce structured UMAPs. **Confidence: Medium** (prediction, not yet confirmed empirically).

---

## Cross-run quantitative table

| Run | Predictor | Cond source | gamma last | gamma best | cos_sim best | enc_er last | runtype_sens best | runtype_sens last | pred_loss best | UMAP |
|-----|-----------|-------------|-----------|-----------|-------------|------------|------------------|------------------|---------------|------|
| e21a (candi) | MLP (candi) | raw meta_tgt | **730** | 1032 | 0.035 | 20.1 | **0.914** | **0.802** | 0.035 | biological islands |
| e21b (fresh baseline) | Transformer | meta_tgt_embed (shared) | 0.11 | 0.11 | 0.003 | 31.6 | 0.601 | 0.446 | 0.038 | diffuse blob |
| e21e (R2: MLP+embed) | MLP | meta_tgt_embed (shared) | 35.2 | 35.2 | −0.043 | 29.9 | 0.740 | 0.615 | **0.032** | diffuse blob |
| e21f (R3: raw meta) | Transformer | raw meta_tgt | 155.8 | 233.6 | **−0.059** | 30.7 | 0.702 | 0.580 | 0.030 | faint structure |
| e21g (R4: sep embed) | Transformer | meta_tgt_embed (separate) | 0.16 | 0.16 | −0.008 | **31.3** | 0.726 | 0.465 | 0.038 | diffuse blob |

All runs: `type2_loci`, 200 epochs, `lambda_sigreg=0.5`, `pred_hidden_dim=0` (→ full proj_dim, no bottleneck).
gamma thresholds per JEPA-skill: healthy = 100–500; fail = > 800 (overcompensating). Bold = best in column.

---

## Per-run grad / stability table

| Run | pred_gnorm last | sig_gnorm last | grad_norm last | clip_frac |
|-----|----------------|----------------|----------------|-----------|
| e21a | 6.38 | 18.64 | 7.01 | 1.0 |
| e21b | 34.19 | 42.86 | 18.39 | 1.0 |
| e21e | 25.42 | 43.54 | 13.44 | 1.0 |
| e21f | 31.91 | 42.41 | 17.61 | 1.0 |
| e21g | 27.79 | 35.45 | 17.67 | 1.0 |

The fresh predictor gradient norms (pred_gnorm 25–34) are 4–5× larger than candi's (6.4), but the conditioning is not being used (gamma≈0). This means fresh predictors have large gradients flowing through the transformer layers but almost none reaching `cond_mod`. The AdaLN-zero gating dissipates the gradient signal across the gating parameters before it reaches the conditioning input.

---

## Per-experiment outcome vs hypothesis

| Run | Hypothesis | Outcome | Confidence |
|-----|-----------|---------|------------|
| e21e (MLP+embed) | Simpler MLP architecture would activate AdaLN conditioning better than transformer | Partial — gamma rises from 0.11 to 35.2 (300× gain), but still 20× below candi. UMAP remains blob. MLP improvement is real but insufficient alone. | Medium |
| e21f (transformer+raw) | Raw meta_tgt bypasses embedding entanglement, gives stronger gradient to cond_mod | Confirmed — gamma rises from 0.11 to 155.8 (1500× gain), best runtype_sens and weakest cos_sim in fresh group, faint UMAP structure visible. Raw conditioning is clearly better than embedded for the transformer. | High |
| e21g (transformer+sep embed) | Separate embedding module decouples predictor from encoder, reduces gradient interference | Rejected — gamma=0.16, identical to e21b. Decoupling the embedding path does not help when the zero-gated transformer is the bottleneck. | High |

---

## Implications for next batch

Priority order: address the one unconfirmed combination first, then add the FJ10 bottleneck.

1. **e21h — MLP predictor + raw `meta_tgt`** (`predictor_type=mlp`, `cond_source=raw_meta_tgt`, `cond_embed_shared=shared`). One-axis change from e21f (swap transformer→MLP) and one-axis change from e21e (swap embed→raw). Predicted gamma_norm_last: 400–800 (MLP direct AdaLN + raw metadata → shortest possible gradient path). Predicted UMAP: biological islands. Cost: 1 GPU × 3h.

2. **e21i — e21h + pred_hidden_dim=16** (add the FJ10 bottleneck on top of the predicted-best predictor). Predicted: enc_er ≥ 23 (analogous to e19u), gamma in 200–600 range, best UMAP in the E21 family. Cost: 1 GPU × 3h.

3. **Only if e21h fails** (gamma < 100): then the problem is encoder-side, not predictor-side. In that case, investigate whether the fresh encoder's MetadataEmbedding output has insufficient gradient magnitude to drive any conditioning module (test: replace MetadataEmbedding with a fixed learned projection, bypassing the embedding network).

Run all on `type2_loci`; add `type1_chr19` comparison only after structured UMAP is confirmed.

---

## Standing findings (carried forward)

| Finding | Status in this synthesis |
|---------|--------------------------|
| FJ1 — λ=0.1 insufficient | Open. All E21 runs use λ=0.5; no new evidence. |
| FJ2 — Periodic zero-mask spikes | Mitigated. All E21 runs show 200/200 spike epochs consistent with the known every-8th-epoch pattern; extract script filters correctly. No new behavior. |
| FJ3 — Two-signal geometry gate (cos_sim + enc_er) | Open, now refined. **E21e/f/g all PASS the two-signal gate yet produce blob UMAPs.** A third criterion `gamma_norm_last ≥ 100` is now required. New finding FJ11 promoted from this synthesis. |
| FJ5 — enc_er peaks at random init then collapses | Open. Fresh enc_er starts at 34–35, declines to 29–31 in 200 epochs — same monotone collapse pattern. Fresh starts from a higher init (likely due to deeper/wider encoder), the slope is similar. |
| FJ7 — meta_tgt conditioning is the dominant metadata-sensitivity lever | Confirmed by e21f (raw meta_tgt gives the strongest conditioning signal) and e21g (decoupling embedding doesn't help). Now extended to also cover conditioning *gradient path* quality. |
| FJ9 — Optimization pressure accelerates collapse | Open. No new evidence in this batch (all 200ep, same λ). |
| FJ10 — pred_hidden=16 is best single encoder knob | Open. None of E21e/f/g tested pred_hidden=16 (all used pred_hidden_dim=0 → full proj_dim). Recommended in e21i. |
| FJ11 — Predictor AdaLN activity (gamma_norm ≥ 100) is a necessary condition for structured UMAP | **New.** Promoted from e21a/b/e/f/g 5-run evidence. Added to FINDINGS.md. |

---

## Caveats and limits

- **Single seed** — all E21 runs use `seed=42`. The gamma_norm gap between fresh and candi is so large (730 vs ≤ 155) that seed noise is unlikely to explain it, but the exact threshold for structured UMAPs may shift with a second seed.
- **pred_hidden_dim=0 across all fresh runs** — the FJ10 bottleneck was not tested in this batch. e21h should be run with and without the bottleneck to disentangle the predictor-arch effect from the capacity effect.
- **No epoch-level hi/lo pred_loss ratio** — all runs report `N/A` for this metric (logging v1 rows only). Cannot apply FJ6 shortcut criterion in this analysis.
- **UMAP structure is subjective** — the "faint structure" in e21f is a qualitative judgment. The quantitative proxy (gamma_norm + runtype_sens_last) supports it but cross-seed confirmation is warranted.
- **rank_runs.py ineligible** — all E21 runs are JEPA-only (no `eval_losses/total_loss`); cornerstone ranking does not apply. All comparisons in this document use JEPA-specific metrics only.

---

## Addendum — gamma_norm scale correction (2026-05-14)

**A logging bug was found and fixed in `jepa_model.py` on 2026-05-14.** The old `JEPAPredictor` (in `jepa.py`) logged the L2 norm of the expanded `[B*L2, hidden_dim]` gamma tensor. The fresh predictors (`JEPAMLPPredictor`, `JEPATransformerPredictor`) were logging the norm of the pre-expansion `[B, hidden_dim]` tensor. With B=16, L2=96, the scale factor is sqrt(L2) ≈ 9.8×.

**All gamma_norm values in the tables above are on the OLD scale (fresh predictors under-reported by ~9.8×).** Corrected per-element RMS gamma (comparable across all runs):

| Run | Old gamma_norm_last | Corrected per-element RMS gamma |
|-----|--------------------|---------------------------------|
| e21a (candi) | 730 | ≈ 2.2 |
| e21b/e21g (transformer+embed) | 0.1 | ≈ 0.003 (genuinely dead) |
| e21e (MLP+embed) | 35.2 | ≈ 1.0 |
| e21f (transformer+raw) | 155.8 | ≈ 3.2 (**HIGHER than candi**) |

**Revised interpretation:** e21b/e21g are confirmed dead (per-element gamma < 0.01). However, e21f has a HIGHER per-element gamma than candi and still produces blob UMAPs. The conclusion from headline #1 (gamma is the primary discriminator) is **partially wrong**: gamma activity is necessary but not sufficient. The primary suspect for blob UMAPs shifts to the **encoder architecture** (single-shot vs per-layer FiLM; XEncoder vs DualAttention). See FJ11 revised and FJ12 in `log-observability/FINDINGS.md`.

The 2×2 ablation matrix (e21m/n/o/p, submitted 2026-05-14) will decisively test this by holding one component fixed while varying the other.

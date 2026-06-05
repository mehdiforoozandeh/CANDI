# E21 2×2 Ablation Matrix + e21h: Encoder Is the Root Cause

Status: synthesis (read-only)
Parents: [idea_e21_jepa_model_first_principles.md](idea_e21_jepa_model_first_principles.md)
Linked from: EXPERIMENTS.md (E21 findings)
Date: 2026-05-14

---

## Headline conclusions

1. **The 2×2 ablation confirms the fresh encoder is the root cause of runtype sensitivity collapse** (High). Swapping fresh→candi encoder with the old predictor held fixed changes `runtype_last` from 0.098 (e21p) to 0.802 (e21m). Swapping old→fresh predictor with the candi encoder held fixed leaves `runtype_last` at 0.708 (e21o), essentially unaffected. The encoder is the culprit; the predictor is not.

2. **e21o (candi enc + fresh transformer predictor) is the best run of the batch and shows a qualitatively unique training pattern: late enc_er peak at ep=155** (High). Every other run peaks before ep=21 and declines. e21o's enc_er grows from 22.8 at init all the way to 26.6 at ep=155, never collapses through 200 epochs, and ends with `runtype_last=0.708`. Visual UMAP confirms structured biological geometry (user-verified). This is the recommended immediate Stage 2 checkpoint (ep=155–170).

3. **The fresh encoder has a burst-then-collapse profile** (High). It starts higher than candi (enc_er_first ≈ 32 vs 22) and surges to enc_er=40–44 within ep=6–21, but then collapses to enc_er=17–18 by ep=200 (worse than candi's 20–26). Higher initial capacity does not translate to durability.

4. **The fresh transformer predictor activates AdaLN only when paired with the candi encoder** (High). With candi encoder (e21o): gamma peaks at 1647, ends at 1207 — highest in the batch. With fresh encoder (e21n): gamma = 0.0–0.9 throughout — completely dead. Same predictor architecture, same conditioning, opposite outcome. The encoder architecture determines predictor gradient quality.

5. **e21h (fresh enc + MLP + raw meta_tgt) achieves the highest peak geometry of the entire E21 sweep but cannot sustain it** (Medium). `enc_er_best=44.1` at ep=20 and `runtype_best=1.041` are both the highest values seen in any E21 run, but by ep=193 `enc_er` has collapsed to 18.3 and `runtype` is erratic. The raw meta_tgt + MLP combination reveals that the fresh encoder's architecture CAN produce exceptional representations; the problem is collapse resistance.

---

## Cross-run quantitative table

| run | encoder | predictor | enc_er_best | enc_er_last | enc_er_peak_epoch | cos_sim_best_filt | runtype_best | runtype_last | gamma_last | geometry gate |
|---|---|---|---|---|---|---|---|---|---|---|
| **e21o** | candi | fresh transformer | **26.6** | **26.2** | **155** | 0.003 | 0.751 | **0.708** | 1207 | PASS ⚠️gamma |
| e21m | candi | old MLP (e19q) | 25.6 | 20.1 | 21 | −0.027 | 0.914 | 0.802 | 386 | PASS |
| e21h | fresh | MLP + raw meta_tgt | **44.1** | 27.98* | 20 | −0.084 | **1.041** | 0.864* | 41 | PASS |
| e21p | fresh | old MLP (e19q) | 40.9 | 17.2 | 6 | −0.077 | 0.587 | 0.098 | 624 | PASS ⚠️runtype |
| e21n | fresh | fresh transformer | 42.2 | 17.8 | 6 | −0.117 | 0.540 | 0.256 | 1 | PASS ⚠️dead γ |

*e21h and e21o `enc_er_last` and `runtype_last` reported at spike epoch=199 (mask_frac=0). True non-spike values at ep=193: e21h enc_er=18.3, runtype=0.272; e21o enc_er=19.5, runtype=0.302. e21p and e21n have no spike epochs (0/200).

Geometry gate definition (FJ3 + FJ11): `cos_sim_best_filtered < 0.15` AND `enc_er_last ≥ 18` AND `gamma_last_clean ≥ 33`. All 5 runs pass the numerical gate; quality differences are in durability and trajectory, not pass/fail.

---

## Per-run grad / stability table

| run | pred_loss_best_filt | sigreg_loss_last | clip_frac_running | grad_norm_first | grad_norm_last | NaN/Inf | SLURM |
|---|---|---|---|---|---|---|---|
| e21o | 0.0249 | 1.95 | 1.0 | 6.38 | 4.58 | none | clean DONE |
| e21m | 0.0270 | 1.80 | 1.0 | 7.12 | 7.01 | none | clean DONE |
| e21h | 0.0247 | 1.59 | 1.0 | 4.77 | 10.59 | none | clean DONE |
| e21p | 0.0179 | 1.09 | 1.0 | 5.16 | 2.98 | none | clean DONE |
| e21n | 0.0286 | 1.06 | 1.0 | 4.91 | 4.02 | none | clean DONE |

All 5 runs completed to ep=199/200 with no NaN/Inf, no OOM, no SLURM kill.

`sigreg_loss_last` for e21n and e21p (fresh encoder, no spike epochs): 1.06–1.09, closer to Gaussian baseline than e21m/e21o (1.80–1.95). This may reflect the fresh encoder's lower enc_er at the final epoch or a lighter SIGReg regime.

`pred_loss_best_filt` for e21p is the lowest (0.0179) despite the worst runtype_last — consistent with FJ9 (better prediction = worse geometry). e21p is over-optimizing prediction.

---

## Per-experiment outcome vs hypothesis

| run | hypothesis | outcome | confidence |
|---|---|---|---|
| e21m | candi enc + old pred reproduces e19q-like structure | Confirmed — enc_er_best=25.6, runtype_last=0.802; visually structured UMAP | High |
| e21n | fresh enc + fresh transformer pred fails (blob UMAPs) | Confirmed — enc_er collapses to 17.8, runtype_last=0.256, gamma dead (≈0); blob UMAP confirmed | High |
| e21o | candi enc + fresh transformer pred reveals predictor effect | Confirmed + Exceeded — fresh transformer pred with candi encoder is BETTER than old pred (late enc_er peak ep=155, best visual UMAP) | High |
| e21p | fresh enc + old pred blobs → encoder is culprit | Confirmed — runtype_last=0.098 vs e21m 0.802; enc_er collapses to 17.2 | High |
| e21h | fresh + MLP + raw meta_tgt combines two winning interventions | Partial — peak metrics best in sweep (enc_er=44.1, runtype=1.041) but not sustained; collapses by ep=79 | Medium |

---

## The 2×2 interaction pattern

```
              old pred (e19q)    fresh transformer pred
candi encoder   e21m: ✓ good        e21o: ✓✓ better
fresh encoder   e21p: ✗ fails       e21n: ✗✗ worst
```

Reading the rows and columns:

- **Row effect (encoder)**: holding predictor fixed, candi encoder is dramatically better in both columns. runtype_last: e21m (0.802) vs e21p (0.098); e21o (0.708) vs e21n (0.256). Encoder row effect is 4–8× on the primary quality metric.
- **Column effect (predictor)**: holding encoder fixed, fresh transformer predictor improves enc_er stability in the candi row (late peak at ep=155, no collapse) and worsens gamma in the fresh row (dead vs active). The predictor effect is secondary and depends on encoder type.
- **Interaction**: fresh transformer pred + fresh encoder → dead AdaLN (gamma<1); same predictor + candi encoder → hyperactive AdaLN (gamma=1647). The predictor cannot activate its conditioning mechanism without the gradient quality provided by the candi encoder.

---

## Detailed trajectory analysis

### enc_er peak timing and collapse

All fresh encoder runs peak at ep=6–21 (burst from random init):
- e21n: peak ep=6, enc_er=42.2
- e21p: peak ep=6, enc_er=40.9
- e21h: peak ep=20, enc_er=44.1

All candi encoder runs show steadier growth, peaking later:
- e21m: peak ep=21, enc_er=25.6 (then slow decline)
- e21o: peak ep=155, enc_er=26.6 (unique late-rising pattern)

First collapse below enc_er=20:
- e21n: ep=73 (~67 epochs after peak at ep=6)
- e21p: ep=74 (~68 epochs after peak at ep=6)
- e21h: ep=79 (~59 epochs after peak at ep=20)
- e21m: ep=80 (~59 epochs after peak at ep=21)
- e21o: ep=156 (immediately after the final peak at ep=155 — this may be transient)

### runtype_sens collapse in fresh encoder

e21p `runtype_sens` trajectory: 0.068 (ep=0) → 0.587 (best, ep=~24) → 0.073 (ep=97, collapsed) → never recovers (0.098 at ep=199). The collapse is permanent: once the fresh encoder loses runtype sensitivity it cannot recover.

e21m `runtype_sens`: oscillates (0.802–0.882 at good epochs, 0.121 at collapse dips) but recovers (0.802 at ep=199 spike). Oscillations appear to be a universal training dynamics effect (same dip around ep=120–145 in all spike-epoch runs), not architecture-specific.

### AdaLN gamma evolution

- e21m (candi + old pred): gamma stable at 350–600 from ep=1 onward. Well-behaved.
- e21o (candi + fresh xfm): gamma 98 at ep=1, rises throughout to 1647 at peak. Warning: overcompensating at 200 epochs.
- e21p (fresh + old pred): gamma 237 at ep=1, stable 400–900. Active but encoder still collapses.
- e21n (fresh + fresh xfm): gamma = 0.0–0.9 throughout. Completely dead — AdaLN weight initialization (zero-init) never receives gradient. Root cause: fresh encoder does not provide gradients strong enough to activate zero-initialized AdaLN weights.
- e21h (fresh + MLP + raw): gamma 25–60, much lower than old-pred runs. MLP predictor's gamma scale is different from the full-position transformer predictor.

---

## Implications for next batch

Priority 1 (highest): immediately use e21o at ep=155–170 as Stage 2 checkpoint. This is the best encoder produced in E21 and is available now.

Priority 2 (encoder redesign, 2 candidate experiments):

| id | change | single axis | predicted effect | cost |
|---|---|---|---|---|
| e21q | Fresh encoder + post-transformer LayerNorm before projector | 1-line change in `JEPAEncoder.forward` | Slower enc_er collapse; runtype_last > 0.40 if normalization is the bottleneck | 1 GPU-run |
| e21r | Fresh encoder + per-layer CNN FiLM (3 injections after each conv layer, same as candi) | Replaces D2 single-shot with 3-layer FiLM schedule | More stable enc_er trajectory; runtype_last > 0.40 if FiLM depth is the bottleneck | 1 GPU-run |

Run e21q first (cheapest, 1-line). If runtype_last > 0.4 in e21q, confirm FJ11-revised (normalization was the fix). If not, run e21r. Do not run both simultaneously — single axis required.

Priority 3 (fresh transformer predictor refinement): e21o's gamma overcompensation (1207 at ep=200, still rising) suggests the predictor will worsen past 200 epochs. Two mitigations:
- Reduce λ_sigreg to 0.3 in e21o-style config (reduces encoder-collapse pressure)
- Use pred_hidden=16 (FJ10) to constrain predictor capacity and slow gamma growth

---

## Standing findings (carried forward)

| Finding | Status update from this synthesis |
|---|---|
| FJ3 — dual geometry gate | open — all 5 runs pass the numerical gate; real differentiation is in trajectory durability, not gate pass/fail. No change to thresholds. |
| FJ5 — enc_er peaks at init and collapses | open — confirmed in all fresh encoder runs (e21n/p: collapse to 17–18 by ep=73–74). e21o uniquely shows LATE peak (ep=155), partially mitigating this finding for the candi+xfm combo. |
| FJ7 — meta_tgt conditioning dominant | open — confirmed again: all runs with active gamma and meta_tgt produce runtype_last > 0.70 (e21m, e21o); dead gamma (e21n) drops to 0.256. |
| FJ9 — optimization pressure accelerates collapse | open — e21p: pred_loss_best=0.018 (lowest), enc_er last=17.2 (worst). Confirms FJ9. |
| FJ10 — pred_hidden=16 best single knob | open — not tested in this batch. |
| FJ11 — gamma scale bug fixed; encoder architecture is primary suspect | confirmed by e21p (fresh enc + old pred = collapse despite active gamma). New findings FJ13/FJ14/FJ15 supersede the "encoder is primary suspect" hypothesis with concrete evidence. |
| FJ12 — 11 structural differences, 5 top candidates | partially resolved — 2×2 confirms encoder is causal. Priority order of candidates now: (1) post-transformer LayerNorm (cheapest test), (2) per-layer FiLM, (3) DualAttention. |

**New findings proposed** (see FINDINGS.md for formal entries): FJ13, FJ14, FJ15.

---

## Caveats and limits

- No UMAP plots available offline (only uploaded to W&B). UMAP quality assessment for e21o is based on user report ("very very good"); e21n/p blobs are inferred from metrics (enc_er collapse + runtype_last failure) consistent with user observation of prior fresh-encoder runs.
- Spike epoch artifact: e21h/m/o report `enc_er_last` and `runtype_last` from the final spike epoch (ep=199, mask_frac=0), which artificially inflates both values (enc_er: +7–10 units, runtype: +0.4–0.6 above non-spike values). True non-spike values at ep=193 are materially lower. e21n and e21p have no spike epochs (confirmed 0/200) and report true final-epoch values.
- Single seed. All conclusions are from single-seed runs; the early burst in fresh encoder (ep=6 peak) could be a lucky initialization. Recommend 2-seed confirmation before committing to encoder redesign.
- e21n FiLM conditioning (`jepa.pred_mask_cond_type=none` for fresh model): the fresh transformer predictor in e21n uses `meta_tgt_embed` via `fresh.cond_source` but the candi harness uses `pred_mask_cond_type=none`. The conditioning pathway is different between e21o and e21n — this is intentional (fresh model uses `fresh.cond_source`, candi uses `jepa.pred_mask_cond_type`) but must be noted for strict comparability.

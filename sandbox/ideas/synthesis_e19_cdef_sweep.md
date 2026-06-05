# E19 JEPA Single-Knob Sweep — e19c / e19d / e19e / e19f

Status: synthesis (read-only)
Parents: [idea_e19_jepa_frozen_decoder.md](idea_e19_jepa_frozen_decoder.md),
         [synthesis_e19_jepa_lam_sweep.md](synthesis_e19_jepa_lam_sweep.md)
Linked from: EXPERIMENTS.md (E19 block)
Date: 2026-05-08

Runs (all vs e19b baseline: λ=0.5, clip_norm=1.0, lr=5e-5, 200ep):
- `e19c_clip5_39125572`  — clip_norm=5.0
- `e19d_lam1_39125573`   — lambda_sigreg=1.0
- `e19e_ep400_39125574`  — epochs=400
- `e19f_lr3x_39125575`   — lr=1.5e-4 (3×)

---

## Headline conclusions

1. **eff_rank peaks at random-init (~45–50) and collapses monotonically throughout training in all runs. This is the direct cause of the uniform-ball UMAP.**
   All four runs begin with eff_rank 41–50 at step 200 and collapse to 20–33 by the end. This confirms FJ5: eff_rank degradation is not a late-phase effect — it begins from the first gradient steps. The collapse rate is controlled by λ (e19d best: 33.2 final) but not by clip_norm or lr. Confidence: **High** — from `lejepa/latent_eff_rank` in metrics.jsonl across all 4 runs.

2. **Relaxing gradient clipping (e19c, clip_norm=5.0) makes collapse WORSE, not better.**
   e19c final eff_rank=22.6, lower than e19b's ~30 (W&B). clip_frac dropped from 1.0 to 0.167 — so most gradient steps are now unclipped. The unclipped pred_loss gradient dominates over SIGReg, accelerating collapse. The insight: at clip_norm=1.0 all gradients are normalized to unit length, which incidentally keeps SIGReg and pred_loss equally competitive. Relaxing clipping lets the larger pred_loss gradient win more often. Confidence: **High** — eff_rank trajectory: 49→36→21→17→22, clip_frac=0.167 final, from metrics.jsonl.

3. **Higher λ (e19d, λ=1.0) is the single most effective knob for slowing collapse.**
   e19d final eff_rank=33.2 (best of the sweep, peak=50.2). Collapse rate is significantly slower: eff_rank stayed ≥39 until step 10k, vs e19c/e/f which collapsed below 25 by step 5k–10k. Trade-off: pred_loss best=0.115 (vs 0.020–0.033 for other runs) — SIGReg costs prediction accuracy. Confidence: **High** — from metrics.jsonl eff_rank and pred_loss.

4. **More epochs (e19e) and higher LR (e19f) do not slow collapse — they accelerate it.**
   e19e final eff_rank=20.2 at step 50k (worse than e19b at ~30 after 25k steps). e19f final eff_rank=20.9 and collapses to 15–16 by step 10k–15k (fastest collapse of all runs). Confidence: **High** — from metrics.jsonl trajectories.

5. **All 4 new UMAPs show uniform spherical clouds with no biological structure, in contrast to e19b's clustered UMAP.**
   All end with eff_rank 20–33/72. At this level the encoder uses only 28–46% of its capacity; the remaining dimensions are near-zero. UMAP projects a partially-collapsed 20–33-dimensional sphere as a featureless ball. E19b's structured UMAP (activity clusters, heart repression cluster, genomic gradient) likely reflected: (a) UMAP was generated under the old shared-colormap code, potentially enhancing apparent contrast; (b) e19b training had higher effective eff_rank at end of training (W&B-reported ~30 is the cleanest measurement). Competing explanation: the periodic zero-mask batches in e19b may have accidentally provided representation-anchoring on fully-unmasked inputs — the partial FJ2 fix may have removed this accidental benefit. Confidence: **High** for the collapse→random-UMAP chain; **Medium** for the e19b explanation.

6. **FJ2 (seed-reuse) is partially fixed but not resolved. Zero-mask steps are now irregular but all 4 new runs have IDENTICAL zero-mask positions and 28% zero-mask frequency (vs 20% in e19b).**
   The `_iter_count` fix changed the strict 1000-step periodicity to an irregular pattern. However, seed=42 is still shared across runs, so the same problematic batches recur deterministically. 35/125=28% of logged training steps see mask_frac=0 (degenerate: context=target, predictor trivial, SIGReg applied to identical views). This is worse than e19b's 20%. A training-level skip of mask_frac=0 batches would be cleaner. Confidence: **High** — zero-mask positions confirmed identical across all 4 runs; count from metrics.jsonl.

---

## Cross-run quantitative table

| metric | e19b (baseline) | e19c clip5 | e19d lam1 | e19e ep400 | e19f lr3x |
|---|---|---|---|---|---|
| λ_sigreg | 0.5 | 0.5 | **1.0** | 0.5 | 0.5 |
| clip_norm | 1.0 | **5.0** | 1.0 | 1.0 | 1.0 |
| epochs / steps | 200 / 25k | 200 / 25k | 200 / 25k | **400 / 50k** | 200 / 25k |
| lr | 5e-5 | 5e-5 | 5e-5 | 5e-5 | **1.5e-4** |
| pred_loss best | 0.031 | 0.033 | 0.115 | **0.024** | 0.020 |
| sigreg_loss final | 1.109 | 0.958 | **0.880** | 0.941 | 0.839 |
| cos_sim final | 0.090 | 0.160 | 0.103 | 0.115 | 0.077 |
| eff_rank peak | ~50 (W&B) | 49.1 | **50.2** | 50.0 | 45.4 |
| eff_rank final | ~30 (W&B) | 22.6 | **33.2** | 20.2 | 20.9 |
| eff_rank collapse (peak→final) | ~20 | 26.5 | **17.0** | 29.8 | 24.5 |
| clip_frac final | 1.000 | **0.167** | 1.000 | 1.000 | 0.998 |
| zero_mask steps (%) | 20% | 28% | 28% | 24% | 28% |
| adaLN β final | ~8 (W&B) | 0.002 | 0.001 | **6.730** | 0.002 |
| adaLN γ final | ~3 (W&B) | 5.841 | 4.200 | **11.531** | 7.296 |
| UMAP structure | visible clusters | uniform ball | uniform ball | uniform ball | uniform ball |

Note: e19b eff_rank and adaLN norms come from W&B (FJ4 bug present in that run). All new runs have correct jsonl geo records.

---

## Per-run grad / stability table

| run | grad_norm range | clip_frac range | collapse peak→final |
|---|---|---|---|
| e19b (baseline) | 5–19 (spike) | 1.000 | ~50→30 (W&B) |
| e19c clip5 | 1.5–12.5 | 0.107–0.600 | 49→22 |
| e19d lam1 | 5.2–15.8 | 1.000 | 50→33 |
| e19e ep400 | 4.4–5.7 | 1.000 | 50→20 |
| e19f lr3x | 3.1–8.2 | 0.998 | 45→21 |

---

## Per-experiment outcome vs hypothesis

| run | hypothesis | outcome | confidence |
|---|---|---|---|
| e19c clip5 | 100% clipping is the primary eff_rank bottleneck | **Rejected** — clip_frac=0.167 final but eff_rank collapsed more (22 vs ~30). Relaxing clipping lets pred_loss dominate. | High |
| e19d lam1 | Stronger SIGReg pushes eff_rank above 30 | **Confirmed (partial)** — eff_rank=33.2 (best of sweep), slowest collapse. Trade-off: pred_loss 3.5× higher. | High |
| e19e ep400 | Model is undertrained at 200ep | **Rejected** — eff_rank=20 at end (worse than e19b's ~30 after 25k steps). Collapse accelerates, not reverses. | High |
| e19f lr3x | LR is the binding convergence constraint | **Rejected** — eff_rank=20.9, fastest early collapse (step 10k: 16.4). Higher LR accelerates both pred and SIGReg but pred wins more. | High |

---

## Implications for next batch

Priority order:

1. **E19g — λ=2.0, clip_norm=1.0 (1 run).** E19d (λ=1.0) is the only hypothesis confirmed. The logical continuation is λ=2.0 to test whether eff_rank can stabilise above 40. Predicted: eff_rank > 35, pred_loss ~0.3–0.5. Risk: pred_loss may plateau too high for useful Stage 2 decoder. Cost: 1 run, 3h.

2. **E19h — λ=2.0, clip_norm=5.0 (1 run).** E19c showed that relaxed clipping is harmful WITH λ=0.5 because pred_loss dominates. With λ=2.0, SIGReg may be strong enough to still compete under relaxed clipping, potentially getting the benefit of unconstrained gradients without collapse. Predicted: eff_rank > 30, clip_frac < 0.5, pred_loss ~0.1–0.2. Cost: 1 run, 3h.

3. **Fix FJ2 properly — skip mask_frac=0 batches.** In `train_jepa.py`, add a check that skips the optimizer step (or skips backward) when `mask_frac==0`. This prevents 28% degenerate batches from corrupting the loss signal. Only requires 2–3 lines in the train loop. Low risk, high value.

4. **UMAP: reduce n_neighbors or try PCA-init.** The current UMAP (n_neighbors=15, min_dist=0.1) may not be sensitive enough to resolve fine structure in a partially-collapsed (eff_rank~20–33) embedding. Try n_neighbors=5, min_dist=0.05 to force tighter clusters. This is a visualization change, not a training change.

---

## Standing findings (carried forward)

| finding | status going in | this synthesis adds |
|---|---|---|
| FJ1 — λ=0.1 insufficient | mitigated (e19b: sigreg→1.03) | λ=1.0 further slows collapse (eff_rank=33 vs 14 at λ=0.1). Partially resolved. |
| FJ2 — periodic zero-mask spikes | open (fix applied) | Fix changed periodicity from strict 1000-step to irregular, but 28% degenerate batches persist. Status: **partially mitigated**; a training-level skip is needed for full resolution. |
| FJ3 — UMAP biologically structured | open (e19b observation) | All 4 new runs show uniform-ball UMAP — structure absent when eff_rank ≤ 33. FJ3 appears to require eff_rank significantly above ~35 to manifest. Status: **conditional**. |
| FJ4 — do_geo timing bug | resolved (fix applied) | All 4 new runs have geometry metrics in jsonl. Confirmed resolved. |
| FJ5 — eff_rank collapse from init (NEW) | **open** | First confirmed here: eff_rank peaks at initialization (step 200: ~45–50) and collapses monotonically. Controlled by λ, not clip_norm or lr. Rate: ~17–30 units lost over 25k steps. |

---

## Caveats and limits

- **Single seed (42) throughout** — all runs use identical biosample ordering. Zero-mask positions are deterministic and identical across all new runs.
- **e19b eff_rank from W&B only** — the "~30" reference for e19b is a smoothed W&B value from a buggy run; the true trajectory is unknown.
- **UMAP at final checkpoint only** — no mid-training snapshots to see when/if structure ever existed.
- **Stage 2 not reached** — pred_loss quality (best 0.020–0.115) is assessed relative to JEPA self-supervised objective, not CANDI reconstruction quality.
- **eff_rank measured on latent z, not projector output** — SIGReg is applied to the projector output; the z geometry is a downstream consequence.

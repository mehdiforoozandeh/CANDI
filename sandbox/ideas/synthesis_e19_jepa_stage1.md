# E19 JEPA Stage 1 — Encoder-only JEPA/SIGReg first run

Status: synthesis (read-only) — superseded for cross-run comparison by [synthesis_e19_jepa_lam_sweep.md](synthesis_e19_jepa_lam_sweep.md) (2026-05-08)
Parents: [idea_e19_jepa_frozen_decoder.md](idea_e19_jepa_frozen_decoder.md)
Linked from: EXPERIMENTS.md (Q7 block in META.md)
Date: 2026-05-07
Run: `e19_jepa_stage1_39046758`  |  job 39046758  |  100 epochs  |  12,500 steps  |  0.73h

---

## Headline conclusions

1. **The JEPA objective is learnable on this data: pred_loss dropped 94% (0.204 → 0.012) in 100 epochs with no divergence (last/best ratio = 1.00).** The encoder can successfully learn to predict the full-assay target latent from masked context. Confidence: High — direct from `lejepa/pred_loss` trajectory.

2. **SIGReg is partially working but representations are not yet Gaussian: sigreg_loss went 7.97 → 1.83, but Gaussian baseline is ~1.05.** At end of training, representations are still 74% above what a perfectly Gaussian distribution would score. Dimensional collapse (eff_rank 49.7 → 14.3) is occurring faster than SIGReg is correcting it. Confidence: High — SIGReg calibration value (1.05) confirmed in unit tests.

3. **Dimensional collapse is the dominant geometry concern: eff_rank fell from 49.7 to 14.3 (proj_dim=72), meaning only ~20% of the projection space is actively used at epoch 100.** Some dims are growing (std_mean 0.341 → 0.469) while others are shrinking (std_min 0.222 → 0.147). This is partial anisotropic collapse, not total collapse. Confidence: High — from geometry snapshots in metrics.jsonl, 62 records.

4. **The run is still improving at epoch 100: the last logged step is also the best.** `total_loss=0.195` at step 12,400 = best ever. This means stopping at 100 epochs was premature; more training is likely to continue improving pred_loss and possibly eff_rank. Confidence: High.

5. **Gradient clipping is very frequent throughout: clip_frac went from 1.00 to 0.72.** At the final step, grad_norm_pre_clip = 1.30 against a cap of 1.0 — still being clipped most of the time. Two notable loss spikes (ep 54: total 0.28, ep 87: total 0.53) suggest brief gradient instability despite clipping. Confidence: High — from `lejepa/grad_norm_pre_clip` and `lejepa/grad_clipped_frac_running`.

6. **Walltime was massively over-requested: run completed in 0.73h against a 6h allocation.** At 26.3 sec/epoch, 200 epochs ≈ 1.5h; 400 epochs ≈ 2.9h. Future submissions should use 2–3h for ≤200 epochs. Confidence: High.

---

## Cross-run quantitative table

| metric | e19_jepa_stage1 |
|---|---|
| epochs completed | 100 / 100 |
| steps | 12,500 |
| total elapsed | 0.73h |
| sec/epoch (mean) | 26.3 |
| pred_loss first→last→best | 0.20357 → 0.01196 → **0.01087** |
| sigreg_loss first→last→best | 7.96875 → 1.82812 → **1.82031** |
| total_loss first→last→best | 1.00044 → **0.19457** → 0.19457 |
| last/best total_loss ratio | **1.00** (not diverged) |
| eff_rank first→last | 49.7 → 14.3 |
| latent_std_mean first→last | 0.341 → 0.469 |
| latent_std_min first→last | 0.222 → 0.147 |
| latent_mean_abs first→last | 0.284 → 0.379 |
| grad_norm_pre_clip first→last | 2.42 → 1.31 |
| clip_frac (running) first→last | 1.000 → 0.720 |
| walltime used | 12% of 6h |

Note: No eval_losses or cornerstone metrics apply — this is an encoder-only JEPA run with no decoders. Rank_runs.py is not applicable here; all comparisons use JEPA-specific diagnostics.

---

## Per-run gradient / stability table

| window | grad_norm_pre_clip | clip_frac_running | notable event |
|---|---|---|---|
| step 200 (ep 1) | 2.42 | 1.000 | — |
| step 1400 (ep 11) | — | — | loss 0.356, stabilizing |
| step 5200 (ep 41) | **0.343** (best) | — | lowest grad norm in run |
| step 6800 (ep 54) | — | — | loss spike: total 0.281, sigreg 2.234, pred 0.057 |
| step 11000 (ep 87) | — | — | **major spike**: total 0.531, sigreg 4.531, pred 0.077 |
| step 12400 (ep 99) | 1.307 | 0.720 | best total_loss; still clipping |

The ep 87 spike is the largest instability: total_loss hits 0.531 (2.7× the preceding plateau of ~0.20) before recovering to the new best at ep 99. sigreg_loss at that spike = 4.531, vs steady-state ~1.9–2.2 — suggests a sudden loss of Gaussian structure, possibly a bad batch or momentary collapse of several projection dims.

---

## Per-experiment outcome vs hypothesis

| run | hypothesis | outcome | confidence |
|---|---|---|---|
| e19_jepa_stage1 | Encoder-only JEPA is learnable from masked assay context; SIGReg prevents collapse | **Partial** — JEPA objective is learnable (pred_loss −94%); SIGReg slows but does not prevent dimensional collapse (eff_rank −71%) | Medium |

---

## Implications for next batch

Priority order:

1. **E19b — increase `lambda_sigreg` from 0.1 → 0.5 (one-axis)**. The SIGReg loss at end of training (1.83) is still 74% above the Gaussian baseline (1.05). The collapse pressure from pred_loss is dominating. Predicted move: eff_rank stabilises higher (>20), std_min stops decreasing; pred_loss may be slightly higher. Cost: 1 run, 2h walltime.

2. **E19c — train 300 epochs (same config)**. Last step is best step; more training is cheap at 26 sec/epoch. A 300-epoch run costs ~2.2h. Predicted move: pred_loss → below 0.01, sigreg_loss → below 1.5 if not collapsed. Cost: 1 run, 2.5h.

3. **E19d — increase `sigreg_num_proj` from 1024 to 2048 with higher `lambda_sigreg=0.5`**. If eff_rank is still collapsing under E19b, stronger SIGReg with better sketching precision may help. Cost: 1 run.

4. **E19e — Stage 2: freeze encoder from E19b/E19c and train decoder head**. Once we have a stable JEPA encoder with eff_rank > 25, this tests whether the pretrained latent is usable for CANDI reconstruction. Outcome will be the first cornerstone comparison for Q7.

**Fix for all future JEPA submits**: use `--time=02:00:00` (2h) for ≤200 epochs, `--time=03:00:00` for ≤400 epochs.

---

## Standing findings (carried forward)

| finding | status going into this synthesis | this synthesis adds |
|---|---|---|
| F1 — depth metadata ignored | open | Not applicable to JEPA encoder-only; no depth_count_ratio probe in train_jepa.py. Does not change status. |
| F7 — pval Gaussian NLL variance collapse | open in multi-head | Not applicable to JEPA (no decoder heads). Does not change status. |
| F8 — E7 best multi-head architecture | open | Not directly applicable; JEPA doesn't use decoder. Does not change status. |

**New finding from this synthesis: FJ1 — SIGReg with lambda=0.1 is insufficient to prevent dimensional collapse in CANDI JEPA training.** Over 100 epochs, eff_rank fell from 49.7 to 14.3 even as sigreg_loss decreased 77%. SIGReg is active and working (collapse > Gaussian >> no-SIGReg), but its weight at 0.1 is too small relative to the pred_loss gradient. Recommended minimum: lambda ≥ 0.5.

---

## Caveats and limits

- **Single seed (42)**: all conclusions are single-seed; the ep 87 spike may be a bad-luck batch rather than a structural instability.
- **No downstream eval**: pred_loss and eff_rank are proxy metrics. Whether this latent actually improves CANDI imputation (Q7 core question) requires Stage 2 (frozen decoder training).
- **Sandbox scale**: 8 assays, type2_loci regime. Results may not generalize to the 35-assay full CANDI model.
- **SIGReg calibration at this scale**: the Gaussian baseline of ~1.05 was measured with 96×64×64 random tensors in unit tests. At the actual batch size (16) and L2=96, the effective N per SIGReg call is 2B×L2 = 2×16×96 = 3072, which may give slightly different calibration.
- **eff_rank computed post-projector**: the SVD is on the projected representation (72-dim), not the raw encoder output (also 72-dim in this run since proj_dim=0=auto). If the projector itself introduces collapse, the encoder may be healthier than eff_rank suggests.

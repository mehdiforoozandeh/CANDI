# E19 - Encoder-only JEPA sweep → frozen decoder

Status: done (Stage 1 complete through e19s–e19z; Stage 2 pending)
Parent: E17 or matched full-head baseline, TBD
Run naming: e19a–e19b (λ sweep), e19c–e19f (single-knob ablations), e19g–e19l (biology-focus sweep), e19m–e19r (meta_tgt + combo follow-ups), e19s–e19z (conclusive batch)
Checklist entry: [EXPERIMENTS.md](EXPERIMENTS.md#experiment-sweep-e17-e20)

## Metrics note (2026-05-15)

- FJ3 geometry gate from earlier notes is superseded by the paper-grounded v2 gate (`combined_loss_scaled`, `sigreg_converged`, `encoder_eff_rank`, `cov_condition_number`).
- FJ5 interpretation is revised: encoder effective-rank decline is treated as SIGReg weakness/failure, not normal JEPA behavior.
- Keep historical metrics (`cos_sim_ctx_tgt`, loss ratios, hi/lo ratio) for continuity, but treat them as secondary diagnostics only.

## Problem Statement

We do not know whether a clean LeJEPA/LeWM-style encoder-only objective can produce representations sufficient for CANDI decoding, or which hyperparameters are needed to avoid dimensional collapse during JEPA pretraining.

## Idea / Hypothesis

Pretrain the CANDI encoder with only JEPA latent prediction plus SIGReg, then freeze that encoder and train a separate CANDI decoder on top. The hypothesis is that a faithful encoder-only objective will learn reusable cross-assay biological latents that remain competitive when decoded into raw CANDI outputs.

## Planned Intervention

- Submit/config path: `sandbox/jobs/submit_e19_jepa_*.sh`
- Parent run or idea: No parent — this is the first encoder-only JEPA run for CANDI.
- Config/code/data deltas: see [spec_e19_jepa_encoder_harness.md](spec_e19_jepa_encoder_harness.md) for full Stage 1 design. Stage 1 trains `CANDI_DNA_Encoder` + `JEPAProjector` + `JEPAPredictor` with `L = L_pred + lambda_sigreg * L_sigreg`; Stage 2 discards projector and predictor, freezes the encoder, trains `latent_projection` + decoder/output heads with standard `L_candi`.
- Faithfulness constraint: NO stop-gradient anywhere in Stage 1 (faithful LeJEPA/LeWM). SIGReg (Epps-Pulley, per-position, both views) is the sole anti-collapse mechanism. Two corruption modes: assay-masking only (default), with option for DSF=1 target + downsampled context.

## Stage 1 run history

All runs use `sandbox/train_jepa.py`, 8-assay EIC sandbox h5, assay-masking only (`p_full_assay=1.0`), `batch_size=16`, `bf16 AMP`, AdaLN-zero in predictor. The FJ2 (seed-reuse) and FJ4 (do_geo timing) bugs were both fixed after e19b and are active in e19c+.

### e19a — lambda_sigreg=0.1, 100 epochs
- Run: `e19_jepa_stage1_39046758`
- Key result: JEPA objective learnable (pred_loss −94%: 0.204→0.012); no divergence. SIGReg insufficient at λ=0.1: eff_rank collapsed 49.7→14.3 over 100 epochs (74% above Gaussian). Still improving at ep100 → warranted e19b.
- Synthesis: [synthesis_e19_jepa_stage1.md](synthesis_e19_jepa_stage1.md)

### e19b — lambda_sigreg=0.5, 200 epochs
- Run: `e19_jepa_lam05_39109506`
- Single change from e19a: `lambda_sigreg 0.1 → 0.5`, epochs 100 → 200
- Key result: sigreg_loss improved to 1.03 (near Gaussian baseline ~1.05); pred_loss converged to 0.031 (−89% from init); eff_rank ~30/72 from W&B (improved from 14 but still partial collapse). cos_sim_ctx_tgt decreased 0.22→0.09 (encoder learned to encode input identity). AdaLN beta/gamma norms grew 0→8/0→3 (predictor learned strong mask conditioning). Gradient clipping saturated 100% throughout (grad_norm_pre_clip 8→12.6). UMAP shows biologically structured latent space. FJ2+FJ4 bugs still active in this run.
- Synthesis: [synthesis_e19_jepa_lam_sweep.md](synthesis_e19_jepa_lam_sweep.md)

## Completed Stage 1 single-knob ablations (e19c–f vs e19b baseline)

Each run changed exactly one hyperparameter vs e19b (`lambda_sigreg=0.5, clip_norm=1.0, epochs=200, lr=5e-5`).

### e19c — clip_norm=5.0
- Hypothesis: 100% gradient clipping throughout e19b (grad_norm_pre_clip 8→12.6, clip_frac=1.0) means the optimizer's effective lr ≈ lr × clip_norm/grad_norm ≈ 6e-6. Raising clip_norm to 5.0 allows the full SIGReg gradient to reach the encoder and should improve eff_rank recovery.
- Change: `clip_norm 1.0 → 5.0`

### e19d — lambda_sigreg=1.0
- Hypothesis: At λ=0.5, sigreg_loss reached 1.03 (near Gaussian) but eff_rank was ~30/72. Doubling λ shifts the gradient balance further toward isotropy; tests whether eff_rank > 50 is achievable with stronger regularization pressure.
- Change: `lambda_sigreg 0.5 → 1.0`

### e19e — epochs=400
- Hypothesis: pred_loss and eff_rank were both still improving at step 25k — the model was not saturated. Simply doubling training time (with FJ2+FJ4 bugs now fixed) may be sufficient to close the eff_rank gap.
- Change: `epochs 200 → 400` (SLURM walltime: 6h)

### e19f — lr=1.5e-4 (3× current)
- Hypothesis: LeJEPA uses lr~1e-3 and our effective lr is further suppressed by 100% clipping. Testing 3× lr in isolation measures whether the LR is the binding convergence constraint, independent of the clipping issue.
- Change: `lr 5e-5 → 1.5e-4`

## Biology-focus sweep (e19g–e19l, staged 2026-05-08)

All runs inherit e19b base: `lambda_sigreg=0.5, clip_norm=1.0, epochs=200, lr=5e-5`.
Code changes applied before this sweep: 4 new metrics (encoder_eff_rank, pred_loss_hi/lo_mask, latent_std_max, sigreg_to_pred_ratio), min_available_frac support, pred_mask_cond_type routing (loci/meta_concat/none).

### e19g — proj_dim=256 (from 72=F2)
- Hypothesis: proj_dim=F2=72 is 3.5× below LeWM/LeJEPA standard. Larger projection space gives SIGReg more isotropy room; may fix collapse without higher λ.
- Change: `jepa.proj_dim=256`
- Script: `sandbox/jobs/submit_e19g_proj256.sh`

### e19h — pred_hidden_dim=16 (from 72=proj_dim)
- Hypothesis: e19b's structured UMAP may have resulted from FJ2 periodically constraining AdaLN beta (accidental predictor throttling). Deliberately constraining pred capacity forces encoder mask-invariance.
- Change: `jepa.pred_hidden_dim=16`
- Script: `sandbox/jobs/submit_e19h_pred16.sh`

### e19i — min_available_frac=0.3 (at least 30% assays unmasked)
- Hypothesis: current masking can leave only 1/8 assays visible (12.5%). Requiring ≥3/8 means context always retains biological signal, reducing encoder's incentive to encode input-identity.
- Change: `training.masking.min_available_frac=0.3`
- Script: `sandbox/jobs/submit_e19i_min30pct.sh`

### e19j — loci masking (pred_mask_cond_type=loci)
- Hypothesis: assay masking lets encoder use "which assays are present" as a shortcut. Loci masking keeps all assays but masks genomic positions — eliminates the shortcut, forces biological variation encoding.
- Change: `jepa.pred_mask_cond_type=loci`, `training.masking.p_full_assay=0.0`, `training.masking.p_full_loci=1.0`
- AdaLN conditioning: per-position scalar (1 = this position is masked at L2 resolution)
- Script: `sandbox/jobs/submit_e19j_loci.sh`

### e19k — DSF corruption (pred_mask_cond_type=meta_concat, dsf_list=4)
- Hypothesis: using DSF=4 context / DSF=1 target preserves assay identity in both views. Encoder cannot use assay-presence shortcut; must encode biological magnitude. AdaLN conditioned on concat(meta_ctx, meta_tgt) carrying the DSF difference.
- Change: `jepa.pred_mask_cond_type=meta_concat`, `training.masking.p_full_assay=0.0`, `training.masking.p_full_loci=0.0`, `training.dsf.dsf_list=4`
- Script: `sandbox/jobs/submit_e19k_dsf.sh`
- **FAILED at startup** (job 39246312, exit code 1, 155s). Root cause: CLI override `dsf_list=[4]` was
  parsed as the string literal `"[4]"`, which `_coerce_scalar(int, "[4]")` rejects. Fixed to `dsf_list=4`
  (comma-separated scalar). Script updated; re-submit pending.

### e19l — no AdaLN (pred_mask_cond_type=none)
- Hypothesis: without any predictor mask conditioning, the encoder must produce mask-invariant biology-focused representations. Tests whether implicit FiLM conditioning in z is sufficient.
- Change: `jepa.pred_mask_cond_type=none`
- Script: `sandbox/jobs/submit_e19l_no_adaln.sh`

## Verifiables

- Validate if: (Stage 1) eff_rank stable > 40 in a bug-fixed run; cos_sim_ctx_tgt stabilises > 0.15; pred_loss continues declining; clip_frac < 0.5 in at least one run.
- Disvalidate if: eff_rank remains < 20 despite increased λ and relaxed clipping; pred_loss plateaus above 0.05.
- Specific checks per run: `lejepa/latent_eff_rank` trajectory in metrics.jsonl (now fixed), clip_frac_running, sigreg_loss, pred_loss, cos_sim_ctx_tgt.
- Required artifacts: `resolved_config.yaml`, `metrics.jsonl`, SLURM logs, W&B run, UMAP figure.

## Risks / Watch-outs

- Frozen decoding (Stage 2) may underperform even if pretraining is good, because the encoder never saw CANDI's exact raw likelihood objective.
- Higher λ (e19d) may push pred_loss up, hurting downstream Stage 2 fine-tuning.
- Relaxed clipping (e19c) may increase gradient noise — monitor grad_norm_pre_clip for instability.
- Single seed (42) throughout — findings may not generalise.

## Run Links

| run | directory | metrics | SLURM log | W&B |
|---|---|---|---|---|
| e19a | `sandbox/runs/e19_jepa_stage1_39046758` | `metrics.jsonl` | `slurm_logs/e19_jepa_s1_39046758.out` | candi_sandbox |
| e19b | `sandbox/runs/e19_jepa_lam05_39109506` | `metrics.jsonl` | `slurm_logs/e19_jepa_lam05_39109506.out` | candi_sandbox |
| e19c | `sandbox/runs/e19c_clip5_39125572` | `metrics.jsonl` | `slurm_logs/` | candi_sandbox |
| e19d | `sandbox/runs/e19d_lam1_39125573` | `metrics.jsonl` | `slurm_logs/` | candi_sandbox |
| e19e | `sandbox/runs/e19e_ep400_39125574` | `metrics.jsonl` | `slurm_logs/` | candi_sandbox |
| e19f | `sandbox/runs/e19f_lr3x_39125575` | `metrics.jsonl` | `slurm_logs/` | candi_sandbox |
| e19g | `sandbox/runs/e19g_proj256_39246308` | `metrics.jsonl` | `slurm_logs/` | candi_sandbox |
| e19h | `sandbox/runs/e19h_pred16_39246309` | `metrics.jsonl` | `slurm_logs/` | candi_sandbox |
| e19i | `sandbox/runs/e19i_min30pct_39246310` | `metrics.jsonl` | `slurm_logs/` | candi_sandbox |
| e19j | `sandbox/runs/e19j_loci_39246311` | `metrics.jsonl` | `slurm_logs/` | candi_sandbox |
| e19k | FAILED (job 39246312, 155s, CLI bug) | — | `slurm_logs/e19k_dsf_39246312.err` | not logged |
| e19l | `sandbox/runs/e19l_no_adaln_39246313` | `metrics.jsonl` | `slurm_logs/` | candi_sandbox |

## Findings (cumulative)

- e19a: JEPA objective learnable; lambda=0.1 insufficient for collapse prevention (eff_rank 49.7->14.3). See synthesis_e19_jepa_stage1.md.
- e19b: lambda=0.5 drives sigreg_loss to near-Gaussian (1.03); eff_rank ~30/72; encoder learns input-identity-sensitive representations (cos_sim 0.22->0.09); predictor learns strong mask modulation (AdaLN beta->8, gamma->3); clipping fully saturated. UMAP biologically structured. See synthesis_e19_jepa_lam_sweep.md.
- e19c-f: FJ5 confirmed (eff_rank peaks at init ~45-50, collapses monotonically). lambda=1.0 (e19d) slowest collapse; clip_norm=5.0 worsens collapse; UMAPs uniform-ball. AdaLN beta smallest in e19b (FJ2 accidental throttling hypothesis). See synthesis_e19_cdef_sweep.md.
- e19g-l (2026-05-09): e19h (pred_hidden=16) and e19l (no AdaLN) confirmed as best runs — cos_sim → −0.011 and 0.014 (most invariant encoders). Both show hi/lo pred_loss ratio < 1.0 (new FJ6: inversion = biology-focused encoder). e19g (proj_dim=256): improves latent_eff_rank (49.5) but enc_er still 19.6 — SIGReg only regularises proj space, not encoder. e19j (loci masking): worst enc_er (12.7), still high cos_sim (0.256); loci masking does not eliminate assay-identity shortcut. e19k failed (CLI bug `dsf_list=[4]` → fixed to `dsf_list=4`). See synthesis_e19_ghjil_sweep.md.

## Next staged batch (e19s–e19z)

- e19s: meta_tgt+masking, `lambda_sigreg=1.0` (single-axis from e19q).
- e19t: meta_tgt+masking, `epochs=400` (single-axis from e19q).
- e19u: meta_tgt+masking, `pred_hidden_dim=16` (single-axis from e19q).
- e19v: meta_tgt+masking, `proj_dim=256` (single-axis from e19q).
- e19w: meta_tgt+masking + DSF secondary corruption (`dsf_list=2`, `sampling=context_down`).
- e19x: meta_tgt+masking + `lr=5e-4` + `clip_norm=3.0` (requested combo).
- e19y: meta_tgt+masking + `lr=5e-4` + `epochs=400` (requested combo).
- e19z: meta_tgt+masking + `lambda_sigreg=0.1` (paper-default sanity check).

Code-level instrumentation update for this batch:
- Added widened depth metadata sensitivity contrast (21 vs 25) with new logged metrics:
  `lejepa/meta_sens_depth_wide` and `lejepa/meta_sens_depth_wide_max`.

## Decision

e19s–e19z was the conclusive Stage 1 batch. See [synthesis_e19_sz_sweep.md](synthesis_e19_sz_sweep.md) for full analysis.

## Findings — e19s–e19z (2026-05-10)

Run links: e19s–e19z in `sandbox/runs/e19{s..z}_metatgt_*_3951684{4..1}/`.
Full analysis: [synthesis_e19_sz_sweep.md](synthesis_e19_sz_sweep.md).

- **λ sweep confirmed** (e19s lam=1.0, e19z lam=0.1, e19q lam=0.5 baseline): λ=0.5 is the
  optimal value. λ=0.1 → trivial genomic-position solution (cos_sim_last=0.968, pred_loss_best=0.007,
  PCA parabola ordered by chr21 position). λ=1.0 → SIGReg-dominant: diffuse structureless UMAP,
  pred_loss_best=0.094. λ=0.5 → balanced: structured biological clusters, pred_loss_best=0.035.

- **400 epochs causes collapse** (e19t): enc_er 22→13.1, UMAP becomes a uniform cloud at epoch 400.
  Runtype sensitivity falls to 0.082 at epoch 399 (best was 0.922 at epoch ~40, but highly noisy).
  adaLN_gamma_norm explodes to 1336 (predictor overcompensates for encoder collapse). Early stopping
  at epoch ~150–175 is required. New finding FJ9 confirmed: optimization pressure accelerates collapse.

- **pred_hidden=16 (e19u) is the best JEPA encoder** of the entire e19 sweep: enc_er=23.5 (highest
  in 200-epoch runs), tightest UMAP biological clusters (better than e19q), runtype_sens_best=0.842,
  pred_loss_best=0.040. Predictor bottleneck forces encoder mask-invariance. New finding FJ10.
  **Recommended Stage 2 checkpoint: e19u at epoch ~150–175.**

- **proj_dim=256 (e19v)**: high lat_er (66.0) but enc_er drops to 15.65 — projection regularisation
  doesn't fully transfer to encoder when proj_dim >> encoder_dim. UMAP quality similar to e19q.

- **DSF secondary corruption neutral** (e19w): indistinguishable from e19q. FJ8 extended/confirmed.

- **LR override bug** (e19x, e19y): `training.optimizer.adamax.lr=5e-4` silently ignored;
  both runs used effective lr=1e-4. e19y is an exact duplicate of e19t (0 differences across 221 steps).
  e19x is a clip_norm=3.0-only ablation. clip_norm=3.0 alone causes collapse (cloud UMAP). Bug must
  be fixed before any LR-sensitive JEPA follow-ups.

- **FJ3 updated**: cos_sim < 0.10 AND enc_er > 15 are both required (e19t shows low cos_sim=0.046
  with collapsed enc_er=13.1 → cloud UMAP; neither criterion alone is sufficient).

# E19 Conclusive Batch: e19s–e19z Sweep Synthesis

Status: synthesis (read-only)
Parents: [idea_e19_jepa_frozen_decoder.md](idea_e19_jepa_frozen_decoder.md)
Linked from: EXPERIMENTS.md
Date: 2026-05-10

All runs inherit the e19q baseline: `meta_tgt conditioning + assay masking, lambda_sigreg=0.5,
lr=1e-4, clip_norm=1.0, epochs=200`. Controlled against `e19q_metatgt_mask_39293820` (the best
run from the prior batch).

---

## Headline conclusions

1. **λ=0.5 is the sweet spot; both λ=0.1 and λ=1.0 fail in opposite ways.** λ=0.1 (e19z) yields
   trivial position-encoding (cos_sim_last=0.968, parabolic PCA arc ordered by genomic position,
   pred_loss_best=0.007); λ=1.0 (e19s) suppresses prediction so strongly that UMAP geometry
   becomes a diffuse ball (pred_loss_best=0.094, lat_er_last=46 but structureless).
   Confidence: **High**.

2. **400 epochs causes encoder collapse.** e19t (400ep, lr=1e-4) shows a uniform-cloud UMAP at
   step 50000 and enc_er_last=13.1 (vs 20.1 at 200 epochs). Runtype sensitivity falls to 0.082
   by epoch 399 despite a peak of 0.922 earlier in training. The PCA arc shape survives, indicating
   macro linear structure is preserved, but UMAP local neighborhood structure is gone.
   Confidence: **High**.

3. **pred_hidden_dim=16 (e19u) produces the best encoder geometry of the batch** and is the
   recommended JEPA checkpoint for Stage 2. UMAP shows the tightest biological island clusters of
   any run; enc_er_last=23.5 (highest among 200-epoch runs); cos_sim_best=0.031; runtype
   sens_best=0.842. The predictor bottleneck forces encoder mask-invariance without hurting
   prediction quality meaningfully (pred_loss_best=0.040 vs 0.035 for e19q).
   Confidence: **High**.

4. **The lr=5e-4 config override silently failed in both e19x and e19y** (confirmed: max logged LR
   for e19x=4.9996e-5, identical to e19q's 4.9996e-5, despite resolved_config showing lr=0.0005).
   e19y is therefore an exact duplicate of e19t (0 metric differences across 221 steps). e19x is
   a clip_norm=3.0 ablation only, not the intended lr+clip combo. Confidence: **High**.

5. **clip_norm=3.0 (e19x) accelerates encoder collapse** despite better gradient flow. UMAP is a
   uniform cloud. enc_er_last=17.1 (vs 20.1 for e19q). pred_loss_best=0.028 improves but at the
   cost of latent geometry. The larger clip budget lets pred_loss gradients dominate faster.
   Confidence: **High**.

6. **DSF secondary corruption (e19w) adds no value on top of assay masking.** Performance is
   indistinguishable from e19q across all metrics (cos_sim_best=0.020 vs 0.033, enc_er_last=20.6
   vs 20.1). Confirms FJ8: once assay masking is active, DSF corruption is redundant.
   Confidence: **High**.

---

## Cross-run quantitative table

`—` = metric not emitted or not relevant.

| run | change vs e19q | pred_best | pred_last | cos_best | cos_last | enc_er_last | lat_er_last | run_sens_best | run_sens_last | depth_sens_best | UMAP quality |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **e19q** (baseline) | — | 0.0347 | 0.2861 | 0.0328 | 0.7884 | 20.08 | 35.56 | 0.8664 | 0.7391 | 0.0229 | ★★★★ clusters |
| e19s | λ=1.0 | 0.0941 | 0.5622 | **−0.0185** | 0.5247 | **21.46** | **46.00** | 0.8823 | 0.7948 | 0.0235 | ★★ diffuse ball |
| e19t | epochs=400 | **0.0226** | **0.0418** | −0.0138 | **0.0460** | 13.12 | 16.66 | 0.9222 | 0.0825 | 0.0236 | ★ cloud |
| **e19u** | pred16 | 0.0398 | 0.3719 | 0.0305 | 0.6876 | **23.52** | 39.16 | 0.8421 | 0.6559 | 0.0236 | **★★★★★** tight |
| e19v | proj256 | 0.0363 | 0.3537 | 0.0824 | 0.7308 | 15.65 | 66.02 | 0.8217 | 0.6879 | 0.0235 | ★★★★ good |
| e19w | dsf2+ctx_dn | 0.0365 | 0.2925 | 0.0199 | 0.7862 | 20.62 | 35.70 | 0.8110 | 0.6101 | 0.0229 | ★★★ good |
| e19x | clip=3.0 only* | **0.0276** | 0.3149 | 0.0162 | 0.7446 | 17.12 | 26.00 | 0.8792 | 0.6952 | 0.0228 | ★ cloud |
| e19y | dup of e19t** | 0.0226 | 0.0418 | −0.0138 | 0.0460 | 13.12 | 16.66 | 0.9222 | 0.0825 | 0.0236 | ★ cloud |
| e19z | λ=0.1 | **0.0073** | 0.0527 | −0.0144 | **0.9681** | 18.74 | 17.72 | 0.7000 | 0.5954 | **0.0254** | ★★★ trivial |

*e19x: lr override silently failed; only clip_norm=3.0 took effect.
**e19y: exact duplicate of e19t (0 metric differences across 221 steps); lr=5e-4 override failed.

Bold = best value in column (ignoring trivial solutions).

---

## UMAP / PCA analysis

**e19q (baseline):** Well-separated biological clusters per cell type. Activity gradient along
cluster axes. Repression (H3K27me3/H3K9me6) forms a distinct dark mass. Genomic position is NOT
the primary axis. PCA shows a clean semicircular arc with smooth activity gradient and anti-correlated
repression. Reference geometry.

**e19s (λ=1.0):** Structureless UMAP — all points scattered in a roughly uniform cloud. No
biological islands. Activity and repression colors are mixed. Strong SIGReg has pushed the encoder
toward maximum isotropy, destroying discriminative structure. PCA shows a wider, flatter arc with
reduced gradient clarity.

**e19t (epochs=400):** UMAP is a uniform diffuse cloud at step 50000. No clusters despite low
final cos_sim=0.046. PCA still shows a semicircular arc with some activity gradient — macro linear
structure survives but local neighborhoods collapse. The encoder has over-converged into a
lower-dimensional but geometrically smoother space. This is distinct from SIGReg-forced isotropy
(e19s): the encoder is not isotropic (PCA has structure) but lacks the local discriminative
distances needed for biological clustering.

**e19u (pred_hidden=16):** Best UMAP in the batch — tighter, more separated islands than e19q.
Activity gradients are steeper and more consistent across all five cell types examined. Repression
structure is clear. PCA arc is clean and well-populated. The predictor bottleneck forced the encoder
to compress biological content into the latent rather than relying on identity shortcuts.

**e19v (proj_dim=256):** UMAP quality similar to e19q. Distinct clusters, good activity gradient,
clear repression structure. Slightly more fragmented islands than e19u. PCA arc is wider (consistent
with higher lat_er=66), but enc_er drops to 15.65 by end, suggesting the larger projection space
regularizes latent but doesn't transfer fully to the encoder.

**e19w (dsf2+ctx_dn):** UMAP similar to e19q. Mild extra fragmentation. No meaningful gain from
DSF corruption. PCA arc shape identical to e19q. Confirms DSF is neutral when masking is active.

**e19x (clip=3.0, lr=1e-4 effective):** Uniform cloud — similar collapse to e19t. The clip_norm=3.0
allows pred_loss gradients to dominate before SIGReg can maintain structure. PCA shows reduced
arc quality with scattered points.

**e19z (λ=0.1):** Visually striking but biologically trivial: UMAP shows ribbon-like elongated
structures per cell type, with genomic position (col 2) coloring perfectly along the ribbon axis.
PCA shows a nearly perfect parabolic arc with points ordered by genomic position. The encoder has
learned to map chr21 position → 1D parabola. This perfectly predicts locus context (pred_loss
best=0.007) while encoding zero cross-cell-type biology. Canonical example of the trivial
position-encoding solution SIGReg is designed to prevent.

---

## Per-run grad / stability table

| run | grad_norm_first | grad_norm_last | grad_norm_best | clip_frac_last | adaLN_gamma_last |
|---|---|---|---|---|---|
| e19q | 6.17 | 6.02 | 1.34 | 1.00 | 387 |
| e19s | 14.09 | 4.30 | 2.24 | 1.00 | 385 |
| e19t | 7.28 | 7.99 | 1.61 | 1.00 | 1336 |
| e19u | 6.80 | 6.69 | 1.57 | 1.00 | 289 |
| e19v | 4.25 | 12.86 | 1.48 | 1.00 | 720 |
| e19w | 7.47 | 6.47 | 1.24 | 1.00 | 372 |
| e19x | 6.82 | 6.86 | 1.69 | **0.77** | 362 |
| e19z | 1.70 | 7.24 | 0.63 | **0.82** | 474 |

Notable: e19t adaLN_gamma_last=1336 (3.5× any other run) — the predictor learns explosively
strong metadata conditioning as the encoder collapses; the predictor overcompensates. e19x and e19z
are the only runs where gradient clipping is not fully saturated (0.77 and 0.82 respectively),
consistent with the clip_norm=3.0 (e19x) and smaller grad_norm (e19z due to low SIGReg gradients).

---

## Per-experiment outcome vs hypothesis

| run | hypothesis | outcome | confidence |
|---|---|---|---|
| e19s (λ=1.0) | λ=1.0 hits collapse-vs-convergence sweet spot under meta_tgt+masking | **Rejected** — stronger SIGReg prevents collapse (lat_er=46) but pred convergence degrades (0.094 best), UMAP structureless | High |
| e19t (epochs=400) | 200→400 epochs improves geometry and metadata sensitivity | **Rejected** — 400 epochs causes enc_er collapse to 13.1, runtype sensitivity falls to 0.08 by epoch 399, UMAP uniformly cloud-like | High |
| e19u (pred16) | Predictor bottleneck forces encoder mask-invariance | **Confirmed** — best UMAP in batch, enc_er=23.5 (highest 200ep), cos_sim_best=0.031, runtype sens=0.842 | High |
| e19v (proj256) | Larger projection space fixes eff_rank collapse via more SIGReg room | **Partial** — lat_er=66 (highest), but enc_er drops to 15.65; projection regularization doesn't fully transfer to encoder | Medium |
| e19w (dsf2+masking) | DSF secondary corruption adds richer multi-scale task | **Rejected** — indistinguishable from e19q on all metrics; FJ8 confirmed | High |
| e19x (lr+clip combo) | Higher LR + larger clip unlocks faster convergence | **Inconclusive (LR bug)** — lr override failed; e19x is clip=3.0 only. clip=3.0 alone accelerates collapse (cloud UMAP) | Medium |
| e19y (lr+epochs) | Higher LR + 400 epochs → faster convergence with geometry | **Invalid (duplicate)** — exact duplicate of e19t (0 metric diffs); lr override failed | N/A |
| e19z (λ=0.1) | Paper-default λ sanity check under best regimen | **Confirmed — trivial solution.** pred_loss_best=0.007, cos_sim_last=0.968, position-encoding UMAP confirmed | High |

---

## Bug report: lr config override silently ignored

**Symptom:** `training.optimizer.adamax.lr=5e-4` was written to `resolved_config.yaml` for e19x
and e19y, but the actual max logged LR (`lejepa/lr`) for both is ~5e-5, identical to e19q (which
uses the default lr=1e-4). e19y and e19t share 0 metric differences across all 221 training steps.

**Evidence:**
- e19q max_lr: 4.9996e-5 (configured lr=1e-4)
- e19x max_lr: 4.9996e-5 (configured lr=5e-4) — should be 5× higher
- e19y max_lr: 4.9999e-5 (configured lr=5e-4) — 0 diffs from e19t

**Scope:** Only the `adamax.lr` dotted-override is affected; `clip_norm`, `epochs`, `lambda_sigreg`,
`pred_hidden_dim`, `proj_dim`, and `dsf` overrides all worked correctly in this batch.

**Root cause (suspected):** `train_jepa.py` constructs the optimizer with a hardcoded or pre-resolved
LR before the config override reaches the optimizer constructor, or the override is applied to a
dead field. Needs investigation before any future LR-sensitive JEPA experiments.

**Impact:**
- e19x is a valid clip_norm=3.0 ablation, NOT the intended lr+clip experiment.
- e19y is an exact duplicate of e19t and carries zero new information.

---

## Implications for next batch

Priority order for Stage 2 (frozen decoder) and remaining Stage 1 questions:

1. **Use e19u (pred_hidden=16, epoch ~150–175) as Stage 2 encoder.** Best UMAP geometry of entire
   e19 sweep. Run Stage 2 frozen-decoder training to test JEPA representations on raw CANDI
   likelihood. This is the primary motivation for the entire E19 series.

2. **Fix the lr override bug before any LR-sensitive follow-up.** The `training.optimizer.adamax.lr`
   dotted override does not apply to the optimizer. Verify root cause in `train_jepa.py`. Once
   fixed, retry the intended lr=5e-4 experiment (pure single-axis, no epoch change).

3. **Implement early stopping for JEPA.** 200 epochs is near-optimal for the current setup; enc_er
   collapses beyond ep200 even with λ=0.5. Monitor enc_er and cos_sim_ctx_tgt trajectories; stop
   when enc_er drops below 18 or cos_sim exceeds 0.15 on a non-mask_frac=0 batch.

4. **Test e19u + proj_dim=256 combination.** Both improve geometry independently. Together, the
   predictor bottleneck + larger projection space may give better encoder regularization transfer.

5. **Investigate adaLN_gamma explosion in long runs.** e19t shows gamma_norm=1336 at epoch 399
   — a 3.5× increase over 200-epoch runs. This is a symptom of encoder collapse being compensated
   by a stronger predictor. Monitoring gamma_norm growth as a collapse early-warning signal.

6. **Fresh JEPA model (E21).** The e19 sweep has revealed the ceiling of the current
   production-CANDI-wrapped JEPA. E21's purpose-built architecture (BERT mask tokens, single-shot
   metadata injection, pre-norm transformer predictor) is the right next step before declaring
   JEPA Stage 1 done.

---

## Standing findings (carried forward)

| Finding | Status in this synthesis |
|---|---|
| FJ1 (λ=0.1 insufficient) | Reconfirmed — e19z λ=0.1 produces trivial position-encoding solution (cos_sim_last=0.968) |
| FJ3 (cos_sim < 0.10 primary criterion) | **Updated** — e19t shows cos_sim_last=0.046 (appears healthy) but UMAP is a cloud; enc_er_last=13.1 reveals collapse. Both cos_sim < 0.10 AND enc_er > 15 are now required simultaneously |
| FJ5 (eff_rank collapses monotonically) | Extended — confirmed in e19t (enc_er 22→13 over 400 epochs) and e19x (enc_er 22→17 over 200 epochs with larger clip) |
| FJ7 (meta_tgt dominates runtype sensitivity) | Confirmed in all 7 runs — all show runtype_sens_best in range 0.70–0.92; baseline e19q retains FJ7 as reference |
| FJ8 (DSF alone fails) | Confirmed by e19w (DSF+masking = e19q; DSF doesn't add value on top of masking) |
| **FJ9 (new)** | Optimization pressure accelerates encoder collapse — see FINDINGS.md |
| **FJ10 (new)** | pred_hidden=16 is the best 200-epoch encoder bottleneck configuration — see FINDINGS.md |

---

## Caveats and limits

- All runs use a single seed (42). Effects (especially at epoch 400) may not generalise.
- `rank_runs.py` returns INELIGIBLE for all JEPA runs (no `eval_losses/total_loss` in metrics.jsonl).
  Rankings above are based on per-branch comparisons (cos_sim, enc_er, UMAP quality, runtype sens).
- The lr override bug means e19x and e19y do not test what they were designed to test. Any LR
  conclusion from this batch is invalid.
- UMAP is computed on chr21 encoder embeddings only — chr19/genome-wide representations may differ.
- e19t runtype sensitivity spikes to 0.922 transiently (epoch ~40) but is 0.082 at final epoch —
  metric is highly noisy and epoch-indexed best is misleading without trajectory inspection.
- e19z's "trivial solution" label is confirmed by PCA (parabolic position curve) and high cos_sim,
  but it is possible that some genuine biological signal is co-encoded with position at low λ.

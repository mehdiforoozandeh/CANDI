# Standing Findings

Append-only log of recurring observations across sandbox runs. Every analysis report should
carry every **open** finding into its conclusions until the user marks it resolved. Never delete
an entry — mark `status: resolved (run <id>, <date>)` so historical reports remain interpretable.

Schema for each entry:
- `Fn — short title.`
- **Status:** `open` | `resolved (run <id>, <date>)`
- **Observed:** key/value evidence and the runs it was first seen in.
- **Interpretation:** likely cause.
- **Action:** what the autoresearch loop should do about it on every iteration.

---

## F1 — Depth metadata ignored

- **Status:** open (first seen 2026-04-26 sweep B1–B7).
- **Observed:** `training_metadata_probes/depth_count_ratio` ≈ 1.0 across all 7 runs (target ≈ 4.0;
  see REFERENCE.md interpretation table).
- **Interpretation:** the probe code is correct; the model's count head appears insensitive to the
  log2(depth) input on row 0 of `y_meta`. This is a model-level issue (lives in production
  `CANDI`), not in the sandbox.
- **Action:** flag in every report; do not propose sandbox-side fixes for this until the model
  head is investigated.

## F2 — Walltime under-utilization

- **Status:** mitigated 2026-04-26 (defaults changed); keep watching.
- **Observed:** with `eval_every_n_epochs=1` (old default) and `meta_sensitivity_probe_every_n_steps=100`,
  the 2026-04-26 baselines used only ~43% of their 3h SLURM walltime for training.
- **Interpretation:** eval + probe overhead dominated the budget.
- **Action:** new defaults are 5 and 200 respectively (since 2026-04-26 evening). Continue to
  compute `Σ epoch_seconds / SLURM walltime` per run; warn if < 70%, fail if < 40%.

## F3 — Late-stage divergence under low LR floor

- **Status:** mitigated by log1p input transform 2026-04-27 (E1_lrfloor_low vs B7).
  Re-flag if a non-log1p or longer-horizon run shows late divergence again.
- **Observed:** `min_lr_ratio=0.1` with cosine schedule produced high-quality models that then
  diverged after their best epoch (B2 epoch ~50, B5 ~55). E1 (`min_lr_ratio=0.01` with log1p)
  ties B7 (`min_lr_ratio=0.1` with log1p) on `quality_score` (8.5099 vs 8.5016), per-branch
  losses, and the grad-norm trajectory (median 32.8 vs 33.4, p95 220.8 vs 220.9). Since B7
  did not diverge in the first place, the low-floor change had no work to do.
- **Interpretation:** the LR floor was a candidate cause of late-stage divergence, but the
  log1p input transform (now the default) eliminates the divergence pattern at chr19 scale
  on its own — the floor is inert under that condition.
- **Action:** stop recommending `min_lr_ratio=0.01` as a stability fix. If a future run
  diverges late, first check whether log1p is active; only consider re-opening F3 if
  divergence reappears under log1p.

## F4 — 1obs R2 is uninformative

- **Status:** policy-resolved 2026-04-26; do not re-promote.
- **Observed:** `eval_metrics/*_r2_1obs` fluctuates by orders of magnitude across epochs because
  the 1obs target has very low denominator variance.
- **Interpretation:** R2 on a near-constant target is unstable by construction.
- **Action:** excluded from success criteria (see REFERENCE.md "Forbidden / excluded"). Use
  Spearman / Pearson on 1obs slices instead. Genome-wide `*_r2_gw` remains valid.

## F5 — Pval head interferes with count + peak training

- **Status:** open (first seen 2026-04-27, runs E2, E5).
- **Observed:** Muting the pval head (E5: `pval_weight=0`, count+peak active) and muting both
  pval and peak (E2: count-only) both improve count metrics substantially over multi-head B7.
  E5 `count_imp_loss=1.7177` and E2 `1.7367` vs B7 `1.7972`. E2 `imp_count_pearson_gw=0.1154`
  vs B7 `0.0592` (~2× higher). Pre-clip grad-norm collapses ~7× in E2 (median 4.89 vs B7 33.4).
  Pval head alone (E3) is the most fragile head: max grad spike 1408, late-stage divergence,
  degenerate end-state.
- **Interpretation:** under the current architecture and loss weights, the pval head's
  gradients dominate the shared encoder representation and prevent the count head from
  converging to a useful imputation signal. The peak head benefits from co-training with
  count even when pval is muted.
- **Action:** when count or peak imputation metrics regress, sweep `pval_weight ∈ {0.0, 0.1,
  0.3, 1.0}` with count+peak active to find the smallest pval contribution that does not
  hurt count. Also test `model.signal_dist=laplace` / `student_t` to see whether the pval
  head's fragility is distributional rather than architectural.

## F7 — Pval Gaussian NLL variance collapse: obs/imp split

- **Status:** mitigated in pval-only isolation (E13_var_floor, 2026-05-06); `gaussian_var_min=0.1` promoted to default same day. **Still open in multi-head training** — multi-head confirmation is a future data point, not a blocker.
- **Observed:** In E3 (pval-only, 400ep), `pval_obs_loss` reaches **−0.201** while `pval_imp_loss` explodes to **20.84**. In multi-head runs, `pval_imp` diverges at ep29–94. E13_ctrl (var_min=1e-6) re-confirmed the pattern: `pval_obs_loss` → −0.111 at ep64; `pval_imp_loss` → 45.5 at ep334. E13_var_floor (var_min=0.1): `pval_obs_loss` best=0.315 (never negative); `pval_imp_loss` last/best ratio=1.40 — **NOT diverged** through 356 epochs. See [`synthesis_e13_var_floor.md`](../../../sandbox/ideas/synthesis_e13_var_floor.md).
- **Interpretation:** The Gaussian NLL is unbounded below when σ² → 0 on observed assays. `gaussian_var_min=0.1` prevents the variance from reaching the degenerate regime. The fix also reduces the `(y-μ)²/var` gradient spike magnitude (~2.4× lower median pre-clip grad norm in treatment vs control).
- **Action:** Run E7 (single-shot FiLM) + `gaussian_var_min=0.1` as a multi-head experiment. If pval_imp divergence is eliminated there too, promote `gaussian_var_min=0.1` as the new default. Also consider testing `var_min=0.01` (less aggressive floor, more expressivity).

## F8 — E7 (single-shot decoder FiLM) is the best multi-head architecture to date

- **Status:** open (established 2026-05-06, runs E7 400-epoch, E6E7 400-epoch vs baseline_400ep).
- **Observed:** E7 at its best epoch (ep84): `imp_peak_auroc=0.765`, `imp_count_pearson=0.339`, `imp_pval_pearson=0.277` — all dramatically better than baseline_400ep at best epoch (ep19): `0.498`, `0.033`, `0.167`. E7 delays pval_imp divergence onset from ep29 (baseline) to ep94. Global pre-clip grad median=7.45 vs baseline 35.5; clip_fraction=0.78 vs 0.88.
- **Interpretation:** Removing per-layer decoder FiLM and replacing with a single latent-level injection reduces gradient noise in the decoder, allowing all three heads to converge further before pval instability dominates. The ~3× delay in pval divergence is consistent with reduced high-frequency gradient interference from per-layer FiLM sites.
- **Action:** Promote `model.single_shot_decoder_film: true` as the new default in `sandbox/configs/default.yaml`. All future ablations should be benchmarked against the E7-configured baseline, not the original baseline.

## FJ1 — SIGReg λ=0.1 insufficient to prevent dimensional collapse

- **Status:** mitigated (λ=0.5, run e19_jepa_lam05_39109506, 2026-05-08). See `synthesis_e19_jepa_lam_sweep.md`.
- **Observed:** eff_rank collapsed 49.7→14.3 in e19_jepa_stage1 (λ=0.1, 100ep). At λ=0.5 (200ep),
  sigreg_loss reached 1.031 (near Gaussian baseline ~1.05) and cos_sim_ctx_tgt stayed 0.02–0.11.
  eff_rank at λ=0.5 not confirmed offline (FJ4 bug); W&B data needed for full confirmation.
- **Interpretation:** pred_loss gradients dominate encoder update at λ=0.1; λ=0.5 rebalances the
  SIGReg contribution sufficiently to prevent visible cos_sim collapse and drive sigreg_loss near baseline.
- **Action:** Use λ≥0.5 as default for future JEPA runs. Sweep λ ∈ {0.5, 1.0, 2.0} once do_geo and
  seed-reuse bugs are fixed (FJ2, FJ4).

## FJ2 — Periodic zero-mask spikes: deterministic seed-reuse in SandboxH5Dataset

- **Status:** partially mitigated (2026-05-08, fix in data.py; confirmed in e19c–f 2026-05-08). Full resolution requires training-level skip of mask_frac=0 batches. See `synthesis_e19_cdef_sweep.md`.
- **Observed:** All 25 jsonl rows with mask_frac=0.000 occur at steps that are exact multiples of 1000
  (spacing=[1000]×24). Each triggers pred_loss 0.27–0.28 and grad_norm 12–20 (10× normal). Loss recovers
  immediately after. Steps per epoch = 125; 1000 / 125 = 8 epochs — the problematic batch is the 125th
  of every 8th epoch.
- **Interpretation:** `SandboxH5Dataset.__iter__` initialises `rng = random.Random(self.seed)` fresh on
  every epoch → same shuffle order and biosample selection every epoch → same batch at position 125 always
  hits a biosample with all assays unavailable → masker adds zero CLOZE entries → mask_frac=0 → context = target → atypical loss. Fix: use `random.Random(self.seed + self._iter_count)` with `_iter_count`
  incremented per epoch.
- **Action:** Always check for periodic spikes (mask_frac=0 pattern) in JEPA run analysis. Filter spike
  steps before computing last/best divergence ratio. Apply seed-reuse fix to all sandbox runs.

## FJ3 — UMAP biological structure requires BOTH low cos_sim AND non-collapsed enc_er

- **Status:** open — refined again 2026-05-10 based on e19s–e19z. Earlier revision (cos_sim < 0.10 primary) was correct but incomplete.
- **Observed (new evidence):** e19t (enc_er=13.1, cos_sim_last=0.046): UMAP is a uniform diffuse cloud despite cos_sim appearing healthy. e19u (enc_er=23.5, cos_sim_best=0.031): best UMAP in batch (tight biological islands). e19z (enc_er=18.7, cos_sim_last=0.968): trivial position-encoding parabola. Prior evidence: e19q (enc_er=20.1, cos_sim<0.09): reference good geometry. e19k (enc_er=19.8, cos_sim=0.619): random ball.
- **Interpretation:** Neither criterion alone is sufficient. Low cos_sim ensures context/target are not trivially identical but does not prevent the encoder from mapping everything into a low-dimensional subspace (e19t: collapsed enc_er despite good cos_sim). Non-collapsed enc_er ensures spread but doesn't prevent trivial shortcuts (e19k: high enc_er, high cos_sim, random UMAP). **Both must hold simultaneously**: `cos_sim_ctx_tgt < 0.10` (on masked batches) AND `enc_er > 15`.
- **Action:** Use both criteria jointly as the JEPA encoder quality gate. Additionally monitor adaLN_gamma_norm trajectory — explosive growth (>500) correlates with encoder collapse compensated by predictor (seen in e19t: gamma=1336 at ep400). See `synthesis_e19_sz_sweep.md`.

## FJ5 — eff_rank peaks at random-init and collapses monotonically throughout JEPA training

- **Status:** open (first confirmed 2026-05-08, runs e19c–e19f; extended 2026-05-10, runs e19s–e19z).
- **Observed:** All four single-knob ablation runs (e19c–f) show eff_rank peaking at step 200 and collapsing monotonically. Extended to e19s–z: e19t enc_er 22.4→13.1 over 400 epochs; e19x enc_er 21.9→17.1 over 200 epochs (accelerated by clip_norm=3.0); e19z enc_er 22.9→18.7 (slower collapse with λ=0.1 pred_loss dominance). e19u (pred16) is the one run where enc_er is stable at epoch 200 (23.5 — highest in 200-ep batch), suggesting predictor bottleneck provides structural resistance to collapse.
- **Interpretation:** pred_loss gradient is larger and/or more consistent than the SIGReg gradient. At 400 epochs, the collapse continues beyond what 200 epochs reveals. Higher optimization pressure (larger clip_norm, more epochs, lower λ) accelerates collapse. The only structural knobs that resist collapse are: higher λ (e19d, e19s partial) and smaller predictor capacity (e19u confirmed).
- **Action:** Use 200 epochs with early stopping (monitor enc_er). Use pred_hidden=16 as default (e19u). e19u at epoch ~150–175 is the recommended JEPA checkpoint for Stage 2. See `synthesis_e19_sz_sweep.md`.

## FJ4 — do_geo timing bug: geometry metrics absent from metrics.jsonl

- **Status:** resolved (fix committed 2026-05-08, confirmed working in e19c–f 2026-05-08).
- **Observed:** `lejepa/latent_eff_rank`, `latent_n_dead`, `grad_pred`, `grad_sig`, `adaLN_*_norm`,
  `enc_gnorm`, `proj_gnorm`, `pred_gnorm` are all absent from metrics.jsonl in both E19 runs. W&B receives
  them every 50 steps.
- **Interpretation:** `do_geo` is computed with pre-increment `global_step` (step N); jsonl snapshot fires
  on post-increment (step N+1). Since `(N+1) % 200 == 0` implies N is odd, and `N % 50 == 0` implies N
  is even, the two conditions never align.
- **Action:** Fix: compute `do_geo = (global_step + 1) % geometry_log_every == 0` (using N+1 before
  backward, consistent with the future post-increment value). All future runs will have geometry in jsonl.

## FJ6 — hi/lo pred_loss inversion is a reliable encoder-quality signal

- **Status:** open (first identified 2026-05-09, runs e19g–e19l).
- **Observed:** When the predictor is constrained (pred_hidden=16, e19h) or blind (no AdaLN, e19l),
  the ratio `pred_loss_hi_mask / pred_loss_lo_mask` drops **below 1.0** (e19h=0.746, e19l=0.702),
  meaning the predictor finds heavy-masking batches *easier* to predict than light-masking ones.
  When the predictor is unconstrained (e19g, proj_dim=256), the ratio is 1.641 — heavier masking
  is harder, indicating the predictor relies on identity shortcuts from the context embedding.
- **Interpretation:** Ratio < 1.0 is the computational signature of an encoder that produces
  *more informative* representations under heavy masking — the encoder compensates for missing
  assays by encoding richer biological signal. Ratio > 1.0 indicates the encoder is encoding
  which assays are present (input-identity shortcut). Combined with cos_sim_ctx_tgt < 0.05,
  this metric forms a reliable two-signal "biology-focus" criterion.
- **Action:** Report hi/lo ratio in every JEPA analysis alongside cos_sim_ctx_tgt. Target hi/lo < 1.0
  as a necessary (but not sufficient) condition for biology-focused representations. Complement with
  encoder_eff_rank for the completeness check.

## FJ7 — meta_tgt conditioning is the dominant lever for runtype metadata sensitivity

- **Status:** open (first established 2026-05-10, runs e19k/m/n/o/p/q/r).
- **Observed:** All three meta_tgt runs (e19p, e19q, e19r) end with `meta_sens_runtype` ≥ 0.375 (e19p=0.445, e19q=0.739, e19r=0.696). All seven non-meta_tgt runs in earlier batches end ≤ 0.086. The effect is absent for depth (`meta_sens_depth` ≤ 0.026 in all runs, consistent with F1). See `synthesis_e19_kmnopar_sweep.md`.
- **Interpretation:** Conditioning the predictor on the flattened target metadata (`y_meta`) forces the encoder to produce embeddings that differ meaningfully when run-type varies, because the predictor must use that signal to complete its task. Depth conditioning is still ineffective — likely because the depth contrast in the probe (log2: 23 vs 25) is too subtle relative to other sources of variation, or because the depth signal in x_meta is too sparse to encode reliably.
- **Action:** Use `pred_mask_cond_type=meta_tgt` as default for runs that prioritise metadata sensitivity. Widen depth probe contrast (e.g. 19 vs 25 = 64× fold change) to diagnose whether sensitivity failure is probe-side or model-side.

## FJ8 — DSF corruption alone fails to produce structured encoder geometry

- **Status:** open (first established 2026-05-10, runs e19k and e19p); extended 2026-05-10 by e19w.
- **Observed:** e19k (DSF=4 context, uniform sampling, meta_concat AdaLN): cos_sim_ctx_tgt=0.619 at end (encoder tracks DSF, not biology); UMAP = random ball. e19p (DSF=4 context, context_down sampling, meta_tgt AdaLN): cos_sim=0.274 at end; UMAP = random ball; encoder_eff_rank collapsed to 12.8 (worst in batch). Both contrast with e19q (assay masking + meta_tgt) which produces structured UMAP and cos_sim<0.09 on masked batches. e19w (DSF=2 context_down + assay masking): indistinguishable from e19q — DSF adds nothing on top of masking.
- **Interpretation:** When context and target differ only in DSF level (no assay masking), the encoder can produce a trivial solution: encode DSF level in the representation so the predictor always knows what "signal boost" to apply. Assay masking provides a fundamentally harder task — the predictor must fill in missing assay profiles — which forces biology-first representations. DSF as secondary corruption (e19w) is neutral, not harmful but also not beneficial.
- **Action:** Do not use pure DSF corruption as the sole JEPA objective. Assay masking must be active. DSF secondary corruption is an option but adds no measurable value. See `synthesis_e19_sz_sweep.md`.

## FJ9 — Optimization pressure accelerates encoder collapse

- **Status:** open (established 2026-05-10, runs e19t, e19x, e19z).
- **Observed:** Three separate interventions that reduce pred_loss faster all cause faster/deeper encoder collapse: (1) e19t (400 epochs): enc_er 22→13, UMAP cloud, pred_loss_last=0.042. (2) e19x (clip_norm=3.0): enc_er 22→17 in 200ep, UMAP cloud, pred_loss_best=0.028. (3) e19z (λ=0.1): cos_sim_last=0.968 (trivial solution), pred_loss_best=0.007. In all three, better prediction metrics accompany geometric degradation.
- **Interpretation:** The JEPA encoder geometry and prediction quality are in tension for the current setup. Any intervention that allows the predictor to "win faster" (more epochs to over-optimize, looser gradient clipping, weaker SIGReg) causes the encoder to collapse. The optimal training window is moderate: enough to learn cross-assay structure but not enough to over-optimize prediction shortcuts.
- **Action:** Monitor pred_loss_best and enc_er jointly. If pred_loss_best < 0.025 and enc_er < 15 simultaneously, the run has over-optimized. Use early stopping or strengthen SIGReg. See `synthesis_e19_sz_sweep.md`.

## FJ11 — Predictor AdaLN inactivity (transformer+embed) confirmed; gamma_norm scale bug found; encoder architecture is primary suspect for blob UMAPs

- **Status:** revised 2026-05-14 (code fix applied; original finding partially superseded).
- **Original claim (wrong scale):** gamma_norm_last ≥ 100 threshold derived by comparing e21a (candi=730) vs fresh runs (e21f=155.8, e21e=35.2, e21b/e21g≈0.1). Concluded candi was ≈5× more active than e21f.
- **Revision:** The old predictor (`jepa.py JEPAPredictor`) logs norm of `[B*L2, hidden_dim]` (gamma expanded to all L2 positions), while fresh predictors were logging norm of `[B, hidden_dim]` (pre-expand). With B=16, L2=96: factor = sqrt(B*L2*H) / sqrt(B*H) = sqrt(L2) = 9.8×. The logging bug is now fixed in `jepa_model.py` — both `JEPAMLPPredictor` and `JEPATransformerPredictor` now expand to `[B*L2]` before taking the norm. **Code fix: 2026-05-14.**
- **Corrected per-element RMS gamma** (all numbers on same scale after fix):
  - e21a (candi): ~2.2 per element
  - e21f (transformer+raw): ~3.2 per element (**HIGHER than candi**)
  - e21e (MLP+embed): ~1.0 per element (half of candi)
  - e21b/e21g (transformer+embed): ~0.003 per element (truly inactive)
- **Revised interpretation:** Transformer+embedded conditioning is genuinely inactive (true finding). But raw conditioning (e21f) produces HIGHER gamma than candi and still generates blob UMAPs. The blob UMAPs are therefore **not explained by predictor inactivity alone** — the encoder architecture is the primary suspect. Key difference: CANDI encoder applies metadata FiLM at every CNN layer (3 layers), while fresh applies a single FiLM after the entire tower. Per-layer metadata conditioning may be necessary for biologically organized representations.
- **Action:** Monitor gamma_norm (now consistently logged on expanded tensor). Threshold for truly-dead predictor: gamma_per_element < 0.1 (corresponds to raw gamma_norm < 0.1 * sqrt(B*L2*H) ≈ 33 with B=16, L2=96, H=72). Blob UMAPs despite active predictor → suspect encoder architecture, not predictor. Next experiment: per-layer FiLM in fresh encoder.

## FJ12 — Fresh encoder has 11 structural divergences from CANDI encoder; encoder confirmed as root cause

- **Status:** partially resolved (2×2 ablation e21m/n/o/p, 2026-05-14). See `synthesis_e21h_mnop_2x2.md`.
- **Observed:** Pre-results: 11 structural differences catalogued. Post-results: e21p (fresh enc + old pred) collapses (`runtype_last=0.098`, `enc_er_last=17.2`) while e21m (candi enc + old pred) stays healthy (`runtype_last=0.802`, `enc_er_last=20.1`). Encoder is the confirmed causal factor.
- **Top 5 structural candidates (priority order for next fixes):**
  1. ~~**Post-encoder projection LayerNorm:** CANDI has `latent_projection` (Linear → GELU → LayerNorm); fresh passes raw transformer output to projector.~~ **RETRACTED 2026-05-15**: `CANDIJepa` calls `candi.encode()` which returns raw encoder output *before* `latent_projection`. Neither CANDI nor fresh encoder has LayerNorm before the JEPA projector. This was never a real difference in the JEPA pipeline.
  2. **FiLM depth:** CANDI applies FiLM after each of 3 CNN layers; fresh applies once post-tower. Most likely cause of runtype sensitivity collapse given FJ11 revised. E23 confirmed: per-conv FiLM is the control, post-conv is strictly worse (+7.8% combined_loss), pre-conv is best (−3.0%).
  3. **Post-fusion LayerNorm:** present in CANDI (`LinearFusion` in production has LayerNorm), also present in fresh (`LinearFusion` in `jepa_model.py` also has LayerNorm). **Not a difference** — both have it.
  4. **Transformer architecture:** CANDI uses `DualAttentionEncoderBlock` (sequence + channel); fresh uses standard `XEncoder`. E23 tested: xtransformers marginally positive (−0.2% combined, −17% pred_loss).
  5. **MaskStem vs post-conv mask_embedding:** E23 tested: mask_token strictly harmful, mask_stem confirmed as the better path.
- **Action:** With candidates 1, 3, and 5 eliminated or retracted, the remaining top candidates are FiLM depth (#2, partially tested in E23) and transformer architecture (#4, tested in E23). The fresh encoder collapse is likely driven by a systemic interaction between these differences rather than a single missing component. Next: combine E23 winners (pre_conv + xtransformers), add predictor meta_tgt conditioning, and test higher lambda_sigreg to directly target the collapse.

## FJ10 — pred_hidden=16 is the best single encoder knob in the 200-epoch JEPA regime

- **Status:** open (established 2026-05-10, confirmed across e19h and e19u).
- **Observed:** e19h (pred_hidden=16 vs e19b baseline): cos_sim_ctx_tgt→−0.011 (best in e19g–l batch). e19u (pred_hidden=16 vs e19q baseline): enc_er_last=23.5 (highest in e19s–z batch), UMAP quality ranked best of all 8 runs, runtype_sens_best=0.842. In both contexts, pred_hidden=16 produces the best encoder geometry without sacrificing pred_loss meaningfully (e19u pred_loss_best=0.040 vs e19q 0.035).
- **Interpretation:** Constraining the predictor to a 16-dim bottleneck prevents the predictor from memorising identity shortcuts through the context embedding. The encoder must compress more biological information into z to support the limited predictor capacity. This is structurally similar to the "inversion" signal in FJ6 (hi/lo pred_loss < 1.0).
- **Action:** Use pred_hidden_dim=16 as default for future JEPA runs. Combine with meta_tgt conditioning (FJ7). Candidate for E21 baseline. See `synthesis_e19_sz_sweep.md`.
  Before promoting this knob to the fresh E21 architecture, compare staged pairs
  e21a/e21b (type2_loci) and e21c/e21d (type1_chr19) to confirm transferability.

## F6 — 200-epoch / 3h budget too short for FiLM variant runs

- **Status:** open (first seen 2026-05-05, runs E6, E7, E6E7).
- **Observed:** E6 (124 epochs), E7 (114 epochs), E6E7 (99 epochs) all walltime-killed. In every case `best epoch = last eval epoch`. Peak_obs_loss still declining linearly at kill in all three. E6's pval improving; E6E7's pval, count, peak all improving. At ~50 sec/epoch, 200 epochs ≈ 2.8h training, leaving < 0.2h for eval/overhead in a 3h slot.
- **Interpretation:** The FiLM variants may converge more slowly than baseline (smaller FiLM grad norms reduce the speed of metadata-path adaptation), requiring more epochs to reach the same quality plateau that baseline_anchor reached around ep 50–60. The 3h / 200-epoch budget was calibrated for the non-FiLM baseline and is insufficient here.
- **Action:** Use `BASELINE_EPOCHS=400 BASELINE_TIME=06:00:00` for all future FiLM variant runs. Apply retroactively to the E6/E7/E6E7 re-runs. Also consider making 400 epochs the **new default** in `baseline_train.sh` / submit scripts to avoid a repeat of this pattern.

## FJ13 — Fresh encoder burst-then-collapse: high initial enc_er does not sustain

- **Status:** open (established 2026-05-14, runs e21h/n/p vs e21m/o).
- **Observed:** Fresh encoder runs (e21h, e21n, e21p) start at `enc_er_first ≈ 32–34` and surge to `enc_er_best = 40.9–44.1` within ep=6–21 — **~73% higher peak than candi** (enc_er_best=25.1–25.6). But all three collapse to enc_er=17–18 by ep=73–79 and end at 17.2–27.98 (spike-inflated for e21h). True non-spike final enc_er for e21n and e21p: 17.8 and 17.2, both below the WARN threshold of 18. Candi encoder in contrast peaks at 25–26 and stays above 20 throughout (e21m=20.1, e21o=26.2 both at spike ep; true non-spike: e21m at ep=193 was ~15.5, e21o at ep=193 was ~19.5).
- **Interpretation:** The fresh encoder's initial high enc_er reflects its pure-depthwise architecture producing more isotropic activations at random init. But single-shot FiLM and/or missing post-transformer LayerNorm fails to maintain that structure under sustained JEPA gradient pressure. The JEPA predictor loss finds low-dimensional shortcuts the encoder cannot resist.
- **Action:** Do not use enc_er_best alone to select Stage 2 checkpoints from fresh encoder runs — use enc_er at the earliest epoch where enc_er stops growing, before collapse begins (ep=6–20 for fresh, ep=155 for e21o). For encoder redesign, target the normalization path first (e21q).

## FJ14 — Fresh transformer predictor AdaLN activation is encoder-type-dependent

- **Status:** open (established 2026-05-14, runs e21n vs e21o).
- **Observed:** Fresh transformer predictor (same architecture, same AdaLN-zero init) behaves completely differently depending on encoder: with fresh encoder (e21n) `gamma_last = 0.9` (dead throughout, 0.0–0.9 across all 200 epochs); with candi encoder (e21o) `gamma_last = 1207` (peaks at 1647, hyperactive from ep=1 onward). The old MLP predictor with fresh encoder (e21p) has `gamma_last = 624` — active. So the freshness of the predictor is not the issue; the encoder architecture determines whether the transformer predictor's AdaLN zero-initialization receives meaningful gradient updates.
- **Interpretation:** AdaLN zero-initialization starts as an identity block. Breaking out of identity requires a gradient signal through `gate_msa * attn(...)` — the gate itself must differ from zero. The candi encoder's richer gradient landscape (from DualAttention and/or per-layer FiLM) provides the initial non-zero gradient needed to activate AdaLN. The fresh encoder's simpler gradient landscape does not.
- **Action:** When fresh transformer predictor shows dead gamma, suspect encoder gradient quality before adjusting predictor hyperparameters. Monitor gamma trajectory from ep=1 — if gamma < 10 at ep=10, the predictor will remain dead without encoder-side changes.

## FJ15 — 2×2 ablation (e21m/n/o/p) definitively isolates encoder as root cause of runtype collapse

- **Status:** open (established 2026-05-14, runs e21m/n/o/p).
- **Observed:** 2×2 encoder × predictor matrix with type2_loci regime, λ_sigreg=0.5, assay masking, 200 epochs:
  - e21m (candi + old): `runtype_last=0.802`, `enc_er_last=20.1` — healthy
  - e21o (candi + fresh xfm): `runtype_last=0.708`, `enc_er_last=26.2` — healthy + better enc_er stability
  - e21p (fresh + old): `runtype_last=0.098`, `enc_er_last=17.2` — fails
  - e21n (fresh + fresh xfm): `runtype_last=0.256`, `enc_er_last=17.8` — fails
  Row effect (encoder): 4–8× on `runtype_last` (candi vs fresh). Column effect (predictor, with candi encoder): e21o `runtype_last=0.708` vs e21m `0.802` — marginal difference, both healthy. Predictor column effect with fresh encoder: e21p vs e21n — both fail, neither recovers runtype sensitivity regardless of predictor.
- **Interpretation:** The fresh encoder's inability to maintain runtype sensitivity is not caused by the predictor architecture or conditioning pathway — it is intrinsic to the fresh encoder itself. The candi encoder is robust to predictor choice (both old and fresh predictor succeed with it).
- **Action:** Encoder redesign is the only productive direction for improving fresh model geometry. Proceed with e21q (post-transformer LayerNorm) and e21r (per-layer CNN FiLM) as the two ranked interventions. See `synthesis_e21h_mnop_2x2.md`.

---

## F-E32-1 — Imp count Pearson ~0.4 but imp R² ≤ 0 on v2 depth-offset (E31); mitigated on E32 AR pin

- **Status:** mitigated (pin only; full v2 pending) — 2026-06-02
- **Observed:** E31 v2 runs: `imp_count_pearson_gw` ~0.45–0.51 while `imp_count_r2_gw` ≤ 0. E32 autoresearch (`sandbox/autoresearch/may31/`, ~60+ commits, 5000-step chr19 pin): vb_natural eval fix + `imp_weight≈0.59`, `depth_center=22.5`, `dsf=off` → best keep `be0d38e2` with **imp_r2=0.122**, den_r2=0.279, canonical imp_r2=−0.161 (A1 gap).
- **Interpretation:** Rank–magnitude decoupling is partly eval-prompt scale (canonical vs V/B depth) and partly loss reweighting; not fixed by small MSE aux on imp. Validate gate (imp>0.15, den≥0.35) not met on pin.
- **Action:** Promote `be0d38e2` recipe to controlled `train_candi_v2` run; if plateau, try per-assay calibration (D4). See `synthesis_e32_imp_r2_autoresearch.md`.

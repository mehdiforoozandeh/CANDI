# q19 → new question: proposed crux restructure (h47-aware)

## THE NEW QUESTION (what h45 becomes)
**"How should CANDI's architecture and training condition on experimental metadata so that imputation
magnitude improves AND metadata is genuinely used, not ignored?"**
Framing: h47 is the pivot — the q19 "offset-ON/OFF Pareto" was mostly a weight-decay artifact; wd=0
already beats the old anchor (macro CRPS 1.495→1.341, every M1 axis) and the gap-fill's real-z probe
shows wd=0 restored *functional* assay steering (assay-permute Δη 0.833, not the synthetic-z 2.6e-4).
The residual problem is no longer "the pathway is dead" but "even alive, conditioning is
under-expressed / redundant with encoder-z / data-unidentifiable for run_type (B1)." Arbiter:
MAGNITUDE (macro CRPS, beat wd0_on 1.341, judged on the oracle-scale-decomposed capability term),
with SHAPE (macro Sp ≥ 0.56), CALIBRATION (ECE ≤ 0.053), and STEERING (real-z metadata-ablation) as
constraints. wd=0 is a baked DEFAULT for every child arm.

## H0 — INSTRUMENT FIX (blocking, 0 GPU, do FIRST) — merges HI1–HI4 + oracle-scale + labels
Re-score the FOUR existing checkpoints (main_s0, offoff_s0, wd0_on, wd0_off) on CPU with corrected
instruments, so h47's 1.341 and its steering claims are confirmed/overturned before any GPU:
- **S1**: replace `eta_slope` with TOTAL told-depth slope + a covariate-agnostic real-z
  metadata-ablation ΔCRPS/ΔSpearman.
- **S5/S18**: oracle per-assay-scale decomposition (c*=argmin CRPS(µ·2^c)); report CRPS_oracle_scaled
  (capability) vs CRPS−CRPS_oracle_scaled (fixable scale); replace the predict-0 marginal.
- **S3**: real foreground (rank top-k / target≥max(thr,1)). **S4**: cluster CIs over the 12 targets.
- **S6**: fix the permuted per-assay labels (H5_ASSAY_ORDER). **S14**: DSF-counterfactual depth
  scoring (score told-k against counts_dsf{k}, not fixed dsf1).
- **S23/HI4**: z-controlled condition-recoverability probe (catches a dead embedder in epoch 1).
Deliverable: honest re-scored baselines + a verdict on h47's V2. Predicted: wd0_on assay-ablation
ΔCRPS>0 (functional) while main_s0≈0; run_type≈0 for all (B1, data not architecture); the ON/OFF
"magnitude Pareto" compresses sharply under oracle-scale.

## RANKED CHILD HYPOTHESES (deduped from 24 candidates)

| # | hypothesis | core change | family | arms | port | why ranked here |
|---|---|---|---|---|---|---|
| **1** | **read_length in the exposure offset** (H1) | `log2_mu += log2(rl/25+1)−c`, fixed coeff | MAGNITUDE | 1 | ~5 lines | audit's #1 magnitude lever (read_length is the missing exposure physics); arithmetic ⇒ extrapolates to the 7/12 OOD targets a learned path can't; frees FiLM for identity. B5-safe (coeff fixed, no attribution). |
| **2** | **explicit per-assay output factor** (merge H2+H4-bₐ+Avocado16) | per-assay eta location+scale + per-assay log-n offset (~24 params, structural, metadata-independent) | MAGNITUDE+CALIB | 1 | ~free (prod head already has per-assay bias) | directly absorbs the oracle per-assay scale error (the thing that made ON/OFF look like a Pareto, h46); closes a fork-vs-prod gap. |
| **3** | **no-decay param group** (AdamW, trunk-only decay) | 2 param groups: decay trunk, wd=0 on embeddings/FiLM/norms/biases | MAGNITUDE(generalization)+PORT | 1+2 seeds | best | the production-safe h47 the PI asked for; keeps trunk regularized. HONEST: cannot out-steer h47 (identical protection) — value is generalization CRPS + portability, not novelty. |
| **4** | **live decoder-FiLM init** (xavier+N(0,0.1)) replacing adaLN-zero | re-init film_proj | STEERING-capacity | 1+1 seed | ~free (prod already live) | removes the *other* half of the annihilation mechanism h47 left in place; production house-style; best single steering-capacity lever. |
| **5** | **metadata-STEERED dispersion** (H3, read_length+imputation-context, NOT depth) | `log n += g(memb_exposure)` | CALIB+STEERING | 1–2 | moderate | the clean S11 PI-thesis test — a channel with no arithmetic shortcut and no encoder-z redundancy. (The told-depth version was KILLED, B6; this restricted form survives.) |
| **6** | **conditioning dropout + CFG dial** (candidate 0) | randomly set y_meta rows→MISSING in training; CFG blend at eval | STEERING+CALIB | 1–2 | trivial | manufactures the missing "use the metadata" gradient (S15, untouched by h47); revives the dead sentinels; free inference guidance knob. Reframe verifiables off S1-errors. |
| **7** | **grouped decoder trunk (groups=A) ± per-deconv per-assay FiLM** (17+18/8) | trunk grouped so [B,L,A,C] is a real per-assay slot; optional per-layer per-assay FiLM (A2-carveout) | SHAPE | 1–2 (hybrid) | moderate (de-pool + groups plumbing) | gives the revived FiLM genuine per-assay channels; the SHAPE lever. RISK: loses cross-assay strength-borrowing → pre-committed falsifier + hybrid fallback. |
| **8** | **add sequencing_platform + lab (+OOV)** = h44, now unblocked (HD3) | 2 new covariates (0.443/0.212 bits, identifiable) | MAGNITUDE+SENS | 4–5 grid | prod deliverable | the only NEW *identifiable* covariates (unlike run_type); the honest home for "metadata sensitivity" positives. Data-side track. |
| **9** | **re-panel for run_type identifiability** (HD1) | re-bake 5–8 biosamples so H(run_type\|assay,rl)>0 | SENSITIVITY | 1–2 + rebake | interpretive | the ONLY lever that lifts B1; the only place "run_type steering as success" is winnable. Data-side track. |
| **10** | **upward-DSF augmentation + thinning consistency** (HD2) | downsample-input mode + 2^d consistency term | MAGNITUDE(OOD)+CALIB | 1–2 | trivial | trains the upward-depth regime (7/12 targets above ceiling, B6); pairs with HD1 (alone can't exceed natural depth). Data-side track. |

## BUDGET FIT (8–12 arms, wd=0 default)
- **Do H0 first (0 GPU).** It may itself shrink the target: if oracle-scale shows most of wd0_on's
  edge is scale, arms #1/#2 (which supply exactly that scale) become the whole game.
- **Round 1 screen (1 seed, ~6 arms):** #1, #2, #3, #4, #6, #7. (These are the cheap, bound-clean,
  production-relevant architecture/training arms.)
- **Round 2 (≥3 seeds on the ~2 finalists that beat 1.341 on the capability term): +6 arms → ~12.**
- **Deferred to a later data-side round:** #5 (dispersion-steer), #8/#9/#10 (need re-bake / grids).

## DROP LIST (do NOT pursue)
- **Aux condition-decodability loss** (cand 3) — KILLED: decodability is necessary-not-sufficient and
  spuriously satisfiable; targets OOD/collinear read_length.
- **Told-depth-driven dispersion route** (cand 11) — KILLED: measured n *increases* with depth
  (+14.5%), so "widen under OOD-upward depth" is contradicted by data (B6). Survives only as #5's
  read_length/context form.
- **Un-sharing film_proj** (cand 10's headline) — UNSOUND: film_proj is a full-rank Linear; realized
  (γ,β) rank is ceilinged by the memb manifold (~3.75), not the projection. Keep only its per-assay
  head-affine, folded into #2.
- **SPADE-lite** (cand 9) — park: causal premise (FiLM is the shape bottleneck) unsupported; the
  2.8M-param ungrouped trunk owns shape; highest port cost. Revisit only if #7 shows shape is the wall.
- **run_type steering as an architecture success bar** — unwinnable (B1); only under #9.

## RELATION TO EXISTING NODES
- **h45** (offset-off/hybrid): SUPERSEDED — becomes this question. Its "learned-scale-head" idea lives
  on as #1 (fixed exposure) + an optional learned-β variant; its anneal/trade-curve arms are dropped
  (the audit + h47 show the Pareto framing was wrong).
- **h44** (platform+lab): BECOMES #8, now unblocked by h47.
- **h46** (offset-off cost is scale not biology): its insight is formalized as H0's oracle-scale
  decomposition and motivates #1/#2.
- **h47**: the pivot; wd=0 is the default for all children; H0 re-scores its headline honestly.

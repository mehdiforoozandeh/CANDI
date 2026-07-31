# Dual metadata-conditioning — v2 implementation plan

Tracked in crux as **q15** (h30 learnable · h31 composition · h32 difficulty · h33 param-encoding).
This supersedes `plan.md` (v1) for the **next iteration**; v1's finding and the confounds it exposed are
the motivation below. **Nothing launches until the PI greenlights.** As of **2026-07-08 the plan is COMPLETE** — the
**Validation** (preflight gate A–K) and **Deliverables** sections are locked; ready for implementation.

---

## Why v2 — what v1 established, and what we are fixing

**v1 result (robust across 3 decoder-FiLM configs):** encoder input-understanding WORKS (M3 within/between
latent ratio ~0.04–0.10, M1 ceiling ~0.20), but **decoder output-steering did NOT emerge** (M2 ~0.01–0.03,
flat with training).

**That result is confounded.** v1 had to make choices that each independently suppress or mis-measure
steering:
1. **Uniform-per-batch metadata** — forced because the v2 decoder **pools `y_meta` across assays**
   (`meta.mean(dim=1)`). One condition per step instead of `A` starves the conditioning signal; the model
   can sit in a lossy `h_y`-agnostic optimum with little per-step pressure to escape.
2. **Mean/R²-only M2** — blind to the families whose signature is NOT in the mean (`cap` saturates the
   mean by construction; `clog`/`power` reshape dispersion/tail). A correctly-steered `cap` model scores 0.
3. **Direct-`p` NB head** (`μ = n(1−p)/p`) — ill-conditioned on heavy-tailed counts (`dμ/dp ~ n/p²`
   explodes as `p→0`), patched with a `winsorize@128` band-aid that destroys real peaks.
4. **No depth offset** — the head wastes capacity relearning per-assay library size (~64× spread here).
5. **Eval on chr21 only** — no generalization-gap read (can't separate underfit from overfit).

**v2 removes each confound so a steering null *or* positive becomes trustworthy.** Changes, by priority:

1. **Per-assay conditioning on both `x_meta` and `y_meta`** — no pooling across assays (may pool across
   positions). *This is the primary fix and the most likely cause of the v1 null.*
2. **Distributional M2** — CRPS response of predicted-vs-target NB as `h_y` sweeps, decomposed into a mean
   statistic and a tail/dispersion statistic (covers reshaping families).
3. **Depth-offset log-link NB head** — DESeq2/scVI size-factor convention; drops winsorize; fixes range.
4. **Full metric suite (CRPS/NLL/Spearman/Pearson/calibration/R²) on chr19 AND chr21.**

**Honest claim v2 can support** (pinned from the dialectic, 2026-07-08): *"the predicted output distribution
is steerable by `h_y` in the direction and shape each family dictates,"* with the locus of steering (mean vs
tail) identified per family, and — from the offset-off arm — whether that steering is unconditional or
requires preconditioning. v2 explicitly does **not** claim offset ≡ plain.

---

## Scientific question (recap)
Can CANDI (i) **normalize** a covariate-transformed count INPUT (encoder, `x_meta`) and (ii) **steer** the
count OUTPUT under an independent covariate (decoder, `y_meta`), when the transform "type" is given as
metadata? Controlled synthetic testbed — we own every transform, so we have exact ground truth. Motivated by
`covariate_probes` (precondition: covariate fingerprint recoverable) and q9/h19 (real `y_meta` pathway
collapses depth, DCR~1).

## Data & base signal
- `sandbox/data/sandbox.h5`, **`counts_dsf1`** `[W,768,8]` as the base signal `x=y`. 8 assays.
- **chr19 → train · chr21 → test.** DSF **off** (fixed depth, DSF=1). Both chroms evaluated (§Metrics).
- Availability per `(biosample, assay)` from `meta_dsf1[0] != -1`; pool all 10 biosamples (~38 tracks).
- **No winsorize** (v1's `CLIP=128` is removed). The heavy tail is handled by the log-link head + `power`
  bound + arcsinh on the encoder input, not by clipping real peaks.
- Unavailable assays excluded from loss via the explicit availability mask (never the −1 sentinel).

## Transforms (count → non-negative integer)
Applied to `f_x` (input) and `f_y` (output), drawn **independently per assay**. Identity is the M1/M3 ref.
`family_id`: 0=identity, 1=mult, 2=add, 3=power, 4=thin, 5=cap, 6=clog.

| family | def | param(s) | class | phase |
|--------|-----|----------|-------|-------|
| identity | `y` | — (ref) | ref | 2a |
| mult `×h` | `round(y·h)` | h∈{0.25,0.5,2,4} | invertible (depth-rescale) | 2a |
| add `+h` | `max(0, round(y+h))` | h∈{2,5,10,20} | invertible (bg shift) | 2a |
| power | `round(y**h)` | h∈{0.5,0.75,1.25,1.5} | invertible (dyn-range) | 2a |
| thin | `Binomial(y,p)` | p∈{0.2,0.4,0.6,0.8} | non-invertible, det-seeded | 2c |
| cap | `min(y, c)` | c∈{2,5,10,20} | non-invertible (censor) | 2c |
| clog | `round(a·log1p(y))` | a∈{1,2,4,8} | non-invertible (compress) | 2c |

- **Staged (Q4 decision): invertible-first.** Phase **2a** trains only {identity, mult, add, power} to test
  whether steering emerges on the easier, mean-based half; **2c** adds {thin, cap, clog} *only if 2a shows
  steering*. De-risks another ambiguous null.
- **`power` bounded to ≤1.5** (drop 2.0) — `2.0` blew targets to ~1e8. 1.5 on a ~10k base is ~1e6, which the
  log-link head represents (`eta ≈ 20` in log₂).
- **Deterministic seeding** for `thin` (RNG keyed by `(biosample, assay, window_start, family, param, side)`).
- 4 param values per non-identity family → M2 gets its steering points; family-level matrix reported.

## Metadata schema — **3 rows** (steering knob + non-steerable size factor)
`x_meta`, `y_meta` are now **`[B, 3, F]`**:
- row 0 `aug_family` (categorical) — steering
- row 1 `aug_param` (continuous) — steering
- row 2 `log2_depth` (continuous) — **observed BASE per-assay library size**, `= meta_dsf1[0, a]`
  (the real ENCODE `log2(seq_depth)` row, re-admitted). **Non-steerable**: `h_y`-independent, transform-
  independent. Values in `sandbox.h5`: 22.2–28.1, mean 25.1.

**Routing rule (the isolation that makes the offset safe):**
- The `DualCondMetaEmbedder` (FiLM) reads **rows 0–1 only** → all steering flows through the conditioning
  pathway.
- The count-head **offset reads row 2 only** → base library size never enters FiLM, and the transform's own
  scale (the ×4 in `mult×4`, the downscale in `thin`) is **not** in the offset, so it must be learned via
  `h_y`. This keeps `mult`/`thin` steering honest.

## Model (CANDIv2, denoising-only, NB + NBNLL)

### (a) Per-assay conditioning — the primary architectural change
- **No metadata pooling across assays**, encoder and decoder. Replace the decoder FiLM's
  `meta.mean(dim=1)` with **per-assay FiLM**: each assay's decoder head is conditioned by its own
  `y_meta[:,:,a]` (per-assay `(γ_a, β_a)`). Pooling across **positions** is still allowed.
- **Audit the encoder** for any across-assay metadata collapse; fix the same way if present.
- This changes `candi_v2` decoder behavior → decide import-and-swap vs minimal fork at build (do not edit
  `candi_v2/` in place unless a tiny additive hook suffices).

### (b) Depth-offset, log-linked NB head (default) + offset-off ablation
- **Default = `DepthOffsetNegativeBinomialLayer`** (candi_v2 E29/E30), keyed to `log2_depth` (meta row 2):
  `eta = linear_eta(x)`; `log2_μ = (log2_depth − depth_center) + eta`; `μ = 2^log2_μ`;
  `n = softplus(linear_n) + eps`; `p = n/(n+μ)`. **This is the log-link** (mean = `2^linear predictor`),
  which also decouples mean (`eta`) from dispersion (`n`).
- **`depth_center` recalibrated 24.0 → 25.1** (the `sandbox.h5` mean `log2_depth`) so `(d−c)≈0` typically.
- **Clamping (point-2 decision):** clamp `log2_μ` (equivalently `eta`) to a finite range and `μ ≥ eps` to
  prevent `2^eta` overflow on the `power` tail. Tune the upper clamp so `power=1.5` targets are representable.
- **Offset-off ablation (Q1 "plain-head attribution" arm, made clean):** the **same log-link head with the
  offset term switched off** (`log2_μ = eta`, i.e. `μ = 2^eta` — the layer's own sentinel-fallback path).
  So offset-on vs offset-off differ in **exactly one variable** (the offset), *not* the link function. We do
  **not** use the legacy direct-`p` head as the ablation (it would confound offset with parameterization).

### (c) Input & channels
- Encoder input = `arcsinh(f_x(counts))`; `signal_transform` handled by the encoder; −1 sentinel preserved.
- **Encoder depth handling is an arm (Q3):** *depth-aware* — encoder also receives base `log2_depth`
  (`x_meta` row 2) as a size-factor input; *depth-naive* — encoder sees rows 0–1 only. Same offset decoder in
  both, isolating what the encoder-side depth signal buys. (Exact wiring — input normalization vs embedded
  covariate — is an implementation note for the build.)
- Control channel & DNA kept for fidelity, inert here.

## Task & batch construction
- **Denoising-only**: no cloze masking; all available assays visible in input, all supervised.
- **Per-assay conditions (default):** each available assay draws `(f_x,h_x)`, `(f_y,h_y)` **independently**
  — now possible because conditioning is per-assay. Off-diagonal cells (`f_x ≠ f_y`) are the real test.
- **Uniform-per-batch (baseline arm):** the v1 regime (one matrix cell/batch), kept to **quantify the lift**
  from per-assay conditioning.
- `make_batch` retains the applied-transform vs metadata-covariate decoupling (`fam_ym`/`par_ym`) for the
  shuffle leakage controls.
- Loss = NBNLL over available positions only. **CRPS-as-loss is deferred (not this phase).**

## M2 — redefined as a distributional steering readout
v1's mean-only `R²(Δpred,Δtarget)` is retired. New M2:

- **Universal readout:** as `h_y` sweeps (input fixed = identity), measure the **CRPS of the predicted NB
  `(μ̂(h_y), n̂(h_y))` against the target distribution `f_y(base)` dictates at each `h_y`**. Steering ⇒ the
  CRPS response curve has a **minimum at the true `h_y`** and rises away from it. Proper, covers
  mean+spread+tail in one number, identical across rescaling and reshaping families.
- **Decomposition (report alongside — says *where* steering lives):**
  - **mean statistic** `Δlog μ̂` — offset-CANCELS (the base-depth offset is `h_y`-independent, so it drops
    out of any `h_y`-difference; this is what makes the offset provably unable to fabricate steering in the
    readout).
  - **dispersion/tail statistic** `Δlog n̂` or a predicted upper quantile (e.g. 95th pct) — **offset-
    independent** by construction (the offset never enters `n̂`), so it's clean regardless of head.
- **Per-family pre-registered signature** (pass = correct sign + monotonicity + shape on the *appropriate*
  statistic):

  | family | expected steering signature |
  |--------|-----------------------------|
  | identity | flat (slope 0) |
  | mult / thin | `log μ̂` ∝ `log h` (slope +1 in log-log — the library-size line), in the **mean** |
  | add | `μ̂` rises monotonically with `h`; `log μ̂` concave/saturating, in the **mean** |
  | power | `log μ̂ ≈ h·log b` — linear in `h`, per-example slope `log b`, in the **mean** |
  | cap | **mean saturates/flattens**; signature moves to the **upper-tail** collapse as the cap tightens |
  | clog | monotone but **concave/sub-unit** slope of `Δlog μ̂` vs `Δlog(base)` — distinct from the unit line |

- **Comparison hygiene:** when contrasting arms (e.g. offset-on vs offset-off), judge steering by
  **direction/monotonicity + Spearman/Pearson of Δ**, not raw CRPS magnitude.

## Metrics (reconstruction + steering + invariance), on **chr19 AND chr21**
- **Reconstruction:** **CRPS** (headline probabilistic, count units), **NLL** (objective; tail-heavy),
  **Spearman** (primary point, rank-robust), **Pearson**, **calibration** error (predictive-interval
  coverage), **R²** (demoted, kept for continuity). Computed on both chroms → generalization gap.
- **M1 (end-to-end):** per `(f_x,f_y)` cell, gap to the identity-cell ceiling, on the above metrics.
- **M2 (output steering):** distributional CRPS response + decomposition, per family (above).
- **M3 (encoder invariance):** within-base vs between-base latent cos-dist ratio (≪1), guarded by M1>0 and a
  between-base floor. Interpretation adjusted per encoder-depth arm.

## Arms & run matrix (proposed — exact grid finalized with Deliverables)
Core architecture (always): per-assay conditioning · depth-offset log-link head · distributional M2 · full
metric suite · invertible families.

- **Phase 2a — core grid:** param-norm {none, z-score, log} × encoder-depth {naive, aware} = **6 runs**
  → reads **h30** (learnable, per-assay) + **h33** (param encoding) + the encoder-depth question.
- **Phase 2b — attribution/control deltas** (off the best 2a config): **3 runs** —
  (i) **offset-off** ablation (offset-moves / offset-off-flat ⇒ "learnable but needs preconditioning",
  now a *clean* conclusion because the offset cancels in the readout);
  (ii) **uniform-per-batch** baseline (lift from per-assay conditioning);
  (iii) **forced identity-input** positive control (`f_x=identity` in training ⇒ `h_y` is the ONLY route;
  isolates whether the pathway CAN steer at all).
- **Phase 2c — staged expansion** (best config, *only if 2a shows steering*): add {thin, cap, clog}
  → reads **h32** (invertible vs non-invertible difficulty + input/output asymmetry).
- **Deferred:** **h31** compositional holdout (unseen `f_x×f_y` cells) — informative only once steering
  works; scheduled for a later phase.

## Splits
- chr19 train / chr21 test, **both evaluated** (all metrics reported per chrom).
- h30: all 2a cells trainable, per-cell eval. h31 (deferred): intersection holdout + memorization baseline.

## Validation & preflight gate (`validate.py` + `jobs/gate.sh`) — locked 2026-07-08

**Rule.** Every v2 change clears BOTH (1) **correctness** (adversarial review + a deterministic oracle) and
(2) **behavioral** (run it; confirm the intended behavior emerged). **Gate-as-you-build** — a component is not
"done" until both pass; the agent keeps fixing until they do. `jobs/gate.sh` runs A–K and **propagates its
exit code** (fixes v1's exit-masking bug); the **multi-hour full array is requested ONLY on green**. Small
CPU/GPU/smoke allocations are used freely en route.

**A. Code review (per component).** Adversarial **multi-agent** review (workflow) on the high-risk components
— per-assay conditioning, depth-offset log-link head, distributional M2 / **closed-form NB CRPS**;
**single-pass** review on the mechanical ones (arms plumbing, data flags). Reviewed against this plan.

**B. Data layer** `[cpu]` — raw counts pass through (batch max == h5 max, not 128); `power ∈ {.5,.75,1.25,1.5}`;
worst target (power1.5·max) finite. `sample_conditions` per-assay-independent (A-vector, values differ across
assays); uniform-per-batch mode reproduces broadcast; thin bit-identical; availability/sentinel consistent.

**C. Metadata schema + routing** `[cpu]` — shapes `[B,3,A(+1)]`; row2 `log2_depth == meta_dsf1[0,a]`,
`== -1` where missing. Routing isolation: perturb row2 → FiLM output unchanged; perturb rows0–1 → offset term
unchanged. Row2 is constant across the h_y sweep (the offset-cancellation precondition).

**D. Per-assay conditioning (primary fix)** — `[cpu]` no across-assay pooling: perturb `y_meta[a]` → output[a]
changes, output[b] unchanged (≤ε); per-assay (γ,β) differ across assays (not identical, unlike v1); same audit
encoder-side. `[gpu-small]` **capacity gate: overfit-tiny per-assay distributional M2 ≥ 0.5 AND ≥ uniform-tiny
+ 0.3** — the mechanism must overfit steering on a handful of windows AND beat the pooled regime, else the full
run is pointless.

**E. Depth-offset log-link head + offset-off** `[cpu]` — oracle `log2_mu=(d−c)+eta`, `mu=2^…`, `p=n/(n+mu)`
match hand-computed; mean is log-linked (no direct-`p` path). **Offset cancels:** `Δlog2_mu == Δeta` across two
h_y at fixed window. Offset-off ⇒ `log2_mu = eta`; on-vs-off differ only by `(d−c)` (single-variable).
`depth_center=25.1`; clamping keeps `mu` finite on power1.5·max (no inf/NaN). `[gpu-small]` **stability without
winsorize:** a few hundred steps on raw counts → NBNLL decreases, `mu` tracks counts, no divergence.

**F. Metrics (CRPS/NLL/Spearman/Pearson/ECE/R²)** `[cpu]` — self-consistency oracles: perfect pred → CRPS=0,
R²=1, Spearman=1, ECE≈0; constant pred → R²≤0. **Closed-form NB CRPS verified vs a high-N Monte-Carlo
reference (within tolerance) + closed-form spot checks.** ECE≈0 on synthetic-calibrated, >0 on miscalibrated.
chr19/chr21 use the correct window pools (no leakage). `[gpu-small]` **your key requirement — metrics track
learning:** over a short run CRPS & NLL **decrease**, Spearman/Pearson/R² **increase**; if CRPS stays flat
while NLL drops → CRPS mis-implemented, keep fixing.

**G. Distributional M2** `[cpu]` — perfect-steering oracle → CRPS-response minimized at true h_y (M2 high);
h_y-ignoring oracle → flat (M2≈0). Per-family signature oracles pass (mult → +1 log-log slope in the mean;
**cap → tail responds while mean flat**; etc.). Mean-stat inherits E's offset-cancellation.

**H. Arms plumbing** `[cpu]` — each flag yields the intended config (param-norm changes embeddings; depth-aware
toggles row2 into the encoder — perturb row2 changes the latent in *aware*, not *naive*; offset-off toggles the
head; uniform toggles the sampler; forced-identity forces f_x=identity); arms orthogonal; distinct
result/wandb tags (no clobber). `[gpu-small]` a 1-step run of each arm executes without error.

**I. Diagnostics** `[cpu]` — **fg/bg (h37):** foreground = **top-2% by base count per assay**; oracle where
steering is foreground-only → M2_fg > M2_agg; `add` (background-visible) oracle → small gap. **shuffle-reliance
(h35):** independently-drawn wrong h_y with the true target; h_y-ignoring model → reliance≈0, steering model →
reliance high.

**J. Integration smoke** `[gpu-small]` — one tiny end-to-end run: all metrics logged (chr19+chr21); **all 8
figures + 3 tables generate from smoke output** (dry-run the report); no NaN; checkpoints save. Re-baseline
v1's 21 CPU gates for the v2 schema/head/metric.

**K. THE GATE** — `jobs/gate.sh` runs A–J and exits nonzero on any failure. **Green ⇒ cleared to request the
full multi-hour array. Not before.**

## Deliverables (phase 2) — locked 2026-07-08

**Story spine — the questions the report answers, in order** (each → its hypothesis + evidence):
1. Did output-steering emerge? *(h30 M2)* — **headline**
2. Was v1's null a pooling artifact? *(h34: per-assay vs uniform-per-batch)*
3. Unconditional or preconditioning-dependent? *(h36: offset-on vs offset-off)*
4. Does the input shortcut suppress it? *(h35: isolated floor + shuffle dose-response)*
5. Foreground/background artifact? *(h37: diagnostic + specificity)*
6. Best param-encoding? *(h33)* · 7. Encoder depth-awareness? *(h30 ablation)* ·
8. Reconstruction + generalization + calibration · 9. Encoder input invariance *(M3)*

**Report form.** One phase-2 **synthesis doc, sectioned by hypothesis** (h30, h34, h35, h36, h37, h33).
Each section = a **pre-registered-verifiable scorecard** (met / unmet / n-a + headline metric) → **evidence
figure** → **one-line verdict** (validated / rejected / partial / inconclusive). Lead the abstract + top
figure with **h30+h34** ("did steering emerge, and was v1's null a pooling artifact"). A **top-level
scorecard table** mirrors the crux verifiables so closing the crux nodes is mechanical.

**Figures.**
- **F1 (headline)** — grouped bar of **distributional M2 (median-invertible)** across arms {per-assay
  (=best 2a cell), uniform-per-batch, offset-off, forced-identity}, with a **v1-null reference line**.
  *(→ Q1/Q2/Q3; h30, h34, h36, h35-floor)*
- **F2** — **small-multiples of the CRPS-response curves** (predicted-vs-target CRPS vs swept `h_y`, min at
  true `h_y`), one panel per family. *(→ per-family steering; h30)*
- **F3** — **3×2 M2 heatmap** (param-norm {none/z/log} × encoder-depth {naive/aware}), **M1 ceiling-gap
  annotated**; the winning cell is the per-assay bar in F1. *(→ h33 + encoder-depth ablation)*
- **F4** — **f_x × f_y M1 matrix heatmap** (4×4 invertible, 2a) on **chr21**, plus a **paired chr19-vs-chr21**
  version (generalization guard). *(→ h30 recon + overfit check)*
- **F5** — **shortcut dose-response scatter**: per-cell `h_y`-reliance (output degradation under shuffled
  `h_y`) vs input-target approximability, expecting a negative trend. *(→ h35 shortcut)*
- **F6** — **grouped bar per family**, M2_foreground vs M2_aggregate, with **`add` highlighted** as the
  background-affecting control. *(→ h37 diagnostic + specificity)*
- **F7** — **calibration reliability diagram** (empirical coverage vs nominal CI) overlaying the arms.
  *(→ Q8 calibration)*
- **F8** — **M3 within/between cos-dist ratio bar per arm** (highlights depth-aware vs naive). *(→ Q9; h30)*

**Tables.**
- **T1 — Reconstruction:** rows = runs, cols = {CRPS, NLL, Spearman, Pearson, ECE, R²} × {chr19, chr21}.
- **T2 — Steering + invariance:** rows = runs, cols = {M2 mean-stat, M2 tail-stat, M3 within/between ratio}.
- **T3 — M3 per-family ratio** (supporting F8).

**Metric definitions** (canonical, referenced from §Metrics / §M2): CRPS · NLL · Spearman · Pearson ·
calibration/ECE · R² (reconstruction); **distributional M2** = median-over-invertible-families steering
score with mean & tail decomposition (steering); within/between cos-dist ratio (M3). **Foreground** = top-k%
by base count (or peak-called); **background** = the rest.

**Deferred to a later phase** (not produced this round): per-family difficulty ranking + mean-vs-tail
**locus** heatmap (h32, needs 2c families); **h31** composition figures; **h37 interventional** (loss-reweight
+ type2-balanced-data arms).

## Mapping to crux (verifiables to update — nodes edited in a later step)
- **h30** ← M1+M2(distributional)+M3 on the per-assay seen matrix (invertible families, 2a).
- **h31** ← held-out cells + memorization baseline — **deferred** to a later phase.
- **h32** ← per-family difficulty ranking + M3(input) vs M2(output) asymmetry, from the 2c expansion.
- **h33** ← the 3-arm param-normalization comparison (none / z-score / log), from the 2a grid.
- **New verifiables to add:** distributional-M2 definition (replaces mean-only R²); offset-on vs offset-off
  **attribution** (unconditional vs preconditioning-dependent steering); encoder depth-aware vs depth-naive.

## Module layout (changes from v1)
```
sandbox/diagnostics/dual_conditioning/
  plan.md          v1 (kept — finding + confounds = v2 motivation)
  plan_v2.md       this
  transforms.py    invertible-first staging; power ≤1.5; (thin/cap/clog gated to 2c)
  data.py          3-row meta (+log2_depth); NO winsorize; per-assay conditions + uniform-per-batch mode
  model.py         per-assay FiLM (no across-assay pooling); depth-offset log-link head + offset-off flag;
                   encoder depth-aware/naive arm
  metrics.py       CRPS/NLL/Spearman/Pearson/calibration/R²; distributional M2 + decomposition; chr19+chr21
  validate.py      PLACEHOLDER (re-baseline + v2 gates)
  converge.py      plateau check on the new metrics
  run.py           arms: param-norm × encoder-depth grid + offset-off/uniform/forced-identity deltas
  jobs/  results/
```

## Decision log (pingpong 2026-07-08 — locked)
- **Per-assay conditioning both sides** = primary fix; **uniform-per-batch kept as baseline arm**.
- **Depth-offset log-link head default**, keyed to observed base `log2_depth` (meta row 2, non-steerable),
  `depth_center=25.1`, with μ/`eta` clamping. **Offset-off ablation = same log-link, offset disabled**
  (clean single-variable). **Drop winsorize; bound `power`≤1.5.**
- **Distributional M2** (CRPS response + mean/tail decomposition, per-family signatures); retire mean-only R².
- **Metrics:** CRPS, NLL, Spearman, Pearson, calibration, R² — **chr19 + chr21**.
- **Param-norm {none, zscore, log}** and **encoder {depth-naive, depth-aware}** = arms.
- **Forced identity-input** positive control = arm.
- **Invertible families first**, non-invertible staged (2c). **h31 holdout deferred.**
- **CRPS-as-loss deferred** to a later phase.
- **Validation + Deliverables = placeholders**, to be designed with the PI before implementation.

## Implementation refinements (2026-07-08, gate GREEN — evidence-driven, PI-approved)
Discovered while building + GPU-gating; each backed by a measurement (see the memory note / gate logs).
- **M2 is PER-ASSAY eval** (sweep one target assay's `h_y`, others held identity), not a uniform
  all-assays sweep. Rationale: a uniform sweep cannot separate the per-assay decoder from the v1 pooled
  decoder — both respond to a single shared swept value (0.48 vs 0.33). The per-assay sweep does:
  **per-assay 0.53 vs pooled 0.02** (= the v1 null, reproduced). Primary steering readout everywhere;
  the CRPS-response steering index, mean-stat (Δeta), tail-stat are all computed per target assay then
  medianed over assays.
- **h34 baseline = POOLED decoder** (`pool_meta=True` reproduces v1 `meta.mean(dim=1)`), the actual
  pooling artifact — not "uniform-per-batch" sampling. Uniform sampling keeps the v2 per-assay FiLM and
  still steers (M2≈0.42), so it is a *sampling* control, kept as a separate arm. The run matrix is now
  **10 tasks**: the 6-cell 2a grid + {offset-off (h36), pooled (h34 pooling artifact), uniform-sampling
  (h34 sampling control), forced-identity (h35)}.
- **Capacity gate = per-assay median-inv M2 ≥ 0.4 (not 0.5) AND mult&add ≥ 0.4 AND per-assay ≥ pooled +
  0.3.** The invertible *median* is capped by **`power` (~0.34)**, the hardest family: power^1.5 spans
  ~2e6 and adjacent high-power params give overlapping large means the CRPS index can't sharply
  separate. Power steering IS present (CRPS diagonal-minimized in every row; mean-stat pearson ~0.71) —
  a difficulty result for **h32**, not a mechanism failure. mult/add steer strongly (0.53 / 0.64).
- **Calibration = non-randomized PIT** (Czado–Gneiting–Held), not interval-coverage ECE — the latter
  spuriously reports ~0.25 for a *calibrated* NB at low counts (which dominate the epigenome). PIT-ECE:
  0.001 calibrated / 0.25 miscalibrated. F7 = PIT reliability diagram.
- **Adversarial multi-agent review** (workflow, high-risk comps) confirmed 3 fixes, all applied:
  (1) metrics reconstruct `p` from the **true μ**, not the model's 1e-6-floored `p` (else CRPS/ECE/
  quantiles are off up to ~10× on the power tail); (2) PIT calibration (above); (3) `thin` seed via
  `zlib.crc32` (was salted `hash()` → not reproducible across processes). Refuted the rest.
- **`jobs/gate.sh` GREEN end-to-end** on a MIG `1g.10gb` slice (fail-fast, propagates exit). Cleared to
  request the array on PI go.

---

## Phase 2c (h32) + h31 holdout — locked with PI 2026-07-09

Phase-2a/2b are DONE and recorded (q16 resolved; h30/h33 closed). This section supersedes the "Deferred to a
later phase" notes above and closes the two remaining q15 hypotheses. **Two SEPARATE sweeps, in order** —
2c (h32) first, then h31 holdout — because the h31 holdout mask shrinks coverage and would confound h32's
per-family difficulty read if folded in; and because the 2c full-train run *is* h31's per-cell reference.

### The full matrices (instrumentation change, applies to BOTH sweeps)
Deliverables are the raw `f_x×f_y` matrices; references/normalizations are chosen at **interpretation**, never
baked into the metric (so a "hard" input-lossy family shows its real, larger gap rather than being normalized
to look easy — the within-family diagonal `cap_x→cap_y` is just another cell we can compare against post-hoc).
- **M1** — already a full per-cell matrix (`cell_crps`/`cell_spearman[(fx,fy)]`). Extends 4×4 → **7×7**.
- **M2** — **currently identity-input only** (`_assay_sweep` pins `f_x=identity`). **Extend to the full 7×7**:
  sweep `h_y` at **each `f_x`**, yielding a steering index per `(f_x, f_y)` cell. The `f_x=identity` column
  reproduces today's M2; the off-identity columns are **dual-conditioning-under-load** (steer the output while
  the encoder *also* undoes a nontrivial input) — core to q15 and untested until now. Persist **per-cell AND
  per-assay un-collapsed** (not just the median-over-assays).
- **M3** — input-side by construction → a **vector over `f_x`** (not 2D). Fine as-is.

### Phase 2c — h32 (invertibility difficulty ladder)
- **One run, not a grid.** Best 2a config: **norm=none** (h33 winner, 0.515) · **per-assay** · **offset-on**
  (kept for calibration; h36 says steering is offset-unconditional either way) · **enc-naive** (M3 was
  invariant to depth-awareness). No delta arms — h32 reads entirely off this single run's matrices.
- **Family menu 4 → 7** — add {thin, cap, clog} on both sides (params per §Transforms table). Matrix
  16 → **49 cells**.
- **Budget bump ~3×** (25ep → **~60–75ep**, equivalently ~3× steps) so per-cell coverage matches 2a — the
  difficulty read must NOT be confounded with undertraining (the ERA ranking artifact). Add a **per-cell
  sample-count log** (assert every cell sampled ≥ a floor) + a **loss-plateau check** (converge.py) so we can
  tell "hard family" from "undertrained".
- **Eval:** full M1/M2 (7×7) + M3 (vector), chr19 + chr21.
- **Reads h32 verifiables** (crux, updated 2026-07-09): per-family difficulty ranking; input-side M3 cost;
  invert-harder-than-apply asymmetry (M2 identity-col vs lossy-`f_x`-col drop + lossy-input vs lossy-output
  M1); steering-locus (mean-stat vs tail-stat per family).

### h31 — compositional generalization (how much does pairing-sparsity hurt)
- **Holdout sweep, after 2c**, same 7×7 menu, best config. Holdout fractions **ρ ∈ {0, 0.15, 0.3, 0.45}**,
  one run each; **ρ=0 is the phase-2c full-train reference** (reuse it, don't re-run).
- **Stratified holdout mask** — held-out and seen sets have **matched family composition**; every transform
  still appears on both sides (only specific *pairings* are unseen); the **identity row/col + the diagonal are
  always trained**.
- **Budget matched per-RETAINED-cell** to phase-2c — steps scaled by the condition-**weight** ratio
  `W_retained/W_all` (a family cell's draw-weight is `|PARAMS[fx]|·|PARAMS[fy]|`, not 1; held cells are all
  weight-16, so the raw cell-count ratio would over-expose retained cells) — same per-cell exposure,
  so retained cells don't over-train and inflate the apparent gap.
- **Core metric — per-cell gen-gap**, difficulty-controlled by the same-cell cross-run reference:
  `Δrecon(c) = CRPS_holdout(c) − CRPS_fulltrain(c)` and `Δsteer(c) = M2_fulltrain(c) − M2_holdout(c)` for
  held-out `c`. A within-run seen-vs-held-out contrast is the budget-controlled sanity check.
- **Memorization baseline** (constructed at eval, no extra training): at held-out cell `(f_x → f_y)`, compare
  the model's steering to the correct `f_y` against its steering to the seen-but-wrong `f_y'` it *had* been
  paired with `f_x`. Beating it proves the model reads `h_y` rather than falling back on a memorized pairing.
- **Reads h31 verifiables** (crux, updated 2026-07-09): per-cell generalization within Δ≤0.10 at ρ=0.3; beats
  memorization baseline; sparsity dose-response + per-family compose map.

### Persistence convention (locked) — "every number a figure draws lives in the run JSON"
Reduction-JSON, **but matrices stored un-collapsed** (per-cell, and per-assay for M2) — restyling *and* in-matrix
re-cuts stay possible without re-running. No raw per-position arrays dumped (aesthetic redraws don't need them;
a fundamentally new statistic would require a re-run — accepted). This already holds for the v2 runs (both
report.py and report_html.py read `results/*.json` only, no inference); it now extends to the un-collapsed
7×7 matrices + the new h31 arrays (per-ρ gen-gaps, per-family compose grid, memorization values).

### New deliverables (added to the existing F1–F8 / T1–T3)
- **F9** — M2 **`f_x×f_y` steering heatmap** (the new full matrix; identity-`f_x` column = old M2). *(h32/h30)*
- **F10** — h32 **mean-stat vs tail-stat locus** panel per family (reshaping families steer in the tail). *(h32)*
- **F11** — h31 **held-out-vs-seen matrix heatmap** (cells hatched by holdout, colored by gen-gap). *(h31)*
- **F12** — h31 **sparsity dose-response** (ρ vs median M1 & M2 gen-gap) — the headline "how much sparsity
  hurts". *(h31)*
- **F13** — h31 **per-family 7×7 compose grid** (mean gap-when-held-out; expect non-invertible-input worst). *(h31)*
- **F14** — h31 **memorization-baseline bar** (correct-`f_y` vs seen-wrong-`f_y'` steering at held-out cells). *(h31)*
- **T4** — h32 per-family difficulty ranking (M1 gap · M3 ratio · M2 locus). **T5** — h31 per-ρ gen-gap summary
  + fraction of held-out cells passing Δ≤0.10. F4 extends 4×4 → 7×7.

### Validation gate extension (new checks appended to `jobs/gate.sh` A–K)
- **L. M2 full-matrix** `[cpu]` — perfect-steering oracle minimized on the diagonal at **every `f_x`**;
  `f_x=identity` column reproduces the current scalar M2 bit-for-bit (no regression); per-cell + per-assay
  keys present in the JSON.
- **M. thin/cap/clog wired** `[cpu]` — the three families produce their spec'd targets (det-seeded thin
  bit-identical; cap censors at `c`; clog compresses); the 7×7 sampler covers all cells; per-cell count log
  hits the floor.
- **N. Budget/coverage** `[cpu/gpu-small]` — 2c per-cell sample count ≥ floor & plateau check fires;
  h31 budget-per-retained-cell matches 2c; holdout mask is stratified (matched family composition, diagonal
  + identity always in).
- **O. h31 gen-gap + memorization** `[cpu]` — cross-run gen-gap uses the identical cell (same `(fx,fy)`,
  same eval units/seed); memorization baseline selects a genuinely *seen* `f_y'` for the held `f_x`.
- **P. Deliverables + persistence** `[gpu-small]` — F9–F14 + T4/T5 generate from a tiny run's JSON;
  **persistence completeness assert**: every value each new figure plots is reloadable from `results/*.json`
  (no figure reaches into memory-only state).

### Run scripts (MUST use the MIG spec `--gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1`)
- `jobs/sweep_2c.sh` — 1 task (the single best-config 7×7 run at the bumped budget).
- `jobs/sweep_h31.sh` — 3 tasks (ρ ∈ {0.15, 0.3, 0.45}; ρ=0 reuses the 2c run).

### Sequence
1 crux pre-registration (h32/h31 verifiables — DONE 2026-07-09) → 2 this plan → 3 implement (M2 matrix +
persistence · thin/cap/clog · budget/coverage · h31 holdout/gen-gap/memorization · F9–F14/T4/T5) →
4 **deep validation** (gate A–P green + adversarial logic review) → 5 **PI green-light** → launch 2c, then h31.

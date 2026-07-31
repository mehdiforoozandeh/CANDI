# TRADEOFF.md — the offset head, and how to pick an arm

This is the honest results document for the recipe in this kit. Read it before you decide which arm to
train. **No checkpoints ship with this kit** — every number below comes from the internal research runs
that produced the recipe, and you will reproduce them from scratch (`## Reproduction on this kit`, below).

Provenance of every number here: the internal report `H48_REPORT.md` (2026-07-24, post-adversarial-
verification) and its scorecard `research/H48_SCORECARD.md`, plus the defect/bounds register
`METADATA_AUDIT.md`. Where an older internal document disagrees with `H48_REPORT.md`, `H48_REPORT.md` wins.

**All three ship with this kit**, copied into [`research/`](research/) so you can audit any number here
rather than take it on trust: [`research/H48_REPORT.md`](research/H48_REPORT.md),
[`research/H48_SCORECARD.md`](research/H48_SCORECARD.md),
[`research/METADATA_AUDIT.md`](research/METADATA_AUDIT.md). See
[`research/README.md`](research/README.md) for what each one is and how to read it.

---

## 1. The one-paragraph answer

**There is no configuration that is best at both imputation and covariate steering.** Turning the depth
offset head ON gives the best imputation in every M1 column, but its covariate steering is functionally
null — the depth response you get is a closed-form arithmetic identity, not something the network
learned, and its response to a change in the `assay_id` prompt is 43x below the pre-registered bar for
"responds at all". Turning the offset OFF gives genuinely learned steering, at +42% macro CRPS. The
hypothesis that a hybrid recovers both (internal node **h45**: offset warm-up→anneal, attenuated offset,
learned scale head) is recorded **refuted, 0 of 4 verifiables met** — but on its premises only, with
**no hybrid arm ever trained** (§4). The good news is that ~84% of the apparent
imputation gap between the arms is *per-assay output scale*, not modelling capability — which makes an
explicit per-assay output factor (~24 parameters) the obvious, and still unrun, next change.

---

## 2. The four arms

Each arm is one training run of the same code, differing in exactly two flags. The names are the internal
run tags; you will see them in the tables below.

| tag | `--offset` | `--weight-decay` | one-line character |
|---|---|---|---|
| `wd0_on_s0` | `on` | `0.0` | best imputation; steering functionally dead |
| `main_s0_perassay` | `on` | `1e-4` | as above, plus the decoder assay-embedding table is annihilated by decay |
| `offoff_s0_perassay` | `off` | `1e-4` | learned steering; worst raw calibration |
| `wd0_off_s0` | `off` | `0.0` | strongest learned steering; worst raw macro CRPS |

The kit defaults to `--weight-decay 0.0`, i.e. the `wd0_*` arms. Exact commands:

```bash
# best-imputation arm (wd0_on)
python -m candi_kit.train \
  --h5 /scratch/$USER/candi_kit/q19.h5 --out-dir /scratch/$USER/candi_kit/runs \
  --offset on --seed 0 --tag kit_on_s0 --weight-decay 0.0 \
  --dsf-sampling uniform --epochs 25 --batch-size 8 --full-coverage \
  --eval-batch-size 4 --eval-max-batches 0 --eval-budget 50000000 --m3-regions 40 \
  --fg-frac 0.02 --n-boot 1000

# learned-steering arm (wd0_off) — identical except --offset
python -m candi_kit.train \
  --h5 /scratch/$USER/candi_kit/q19.h5 --out-dir /scratch/$USER/candi_kit/runs \
  --offset off --seed 0 --tag kit_off_s0 --weight-decay 0.0 \
  --dsf-sampling uniform --epochs 25 --batch-size 8 --full-coverage \
  --eval-batch-size 4 --eval-max-batches 0 --eval-budget 50000000 --m3-regions 40 \
  --fg-frac 0.02 --n-boot 1000
```

Both arms as a 3-seed SLURM array: `sbatch candi_kit/slurm/train.sh` (edit `KIT`/`VENV`/`H5`/`OUT` at the
top). ~85 min wall per arm on one `nvidia_h100_80gb_hbm3_1g.10gb` MIG slice.

What the flag does, in code: `src/candi_kit/model.py:177-183`.

```
offset ON,  valid told depth:  log2_mu = (depth - depth_center) + eta
offset ON,  MISSING/CLOZE depth, or offset OFF:  log2_mu = eta
then:  mu = 2^clamp(log2_mu, -15, 30);  n = softplus(raw_n) + eps;  p = n / (n + mu)
```

`depth_center` is **25.1** in the historical q19 runs (and in `--compat-q19`), but at train time the kit
**derives** it from your h5 — the median of finite `meta_dsf1[0]` over `T_` biosamples — and prints it;
`--depth-center` overrides. Do not copy 25.1 onto a different panel (`RECIPE.md` § 4.4).

`eta` is the network's offset-free mean statistic; `depth` is `log2(sequencing depth)` read out of the
metadata prompt. That is the entire difference between the arms.

---

## 2b. What the metric names mean

Your run prints `M1`, `M2`, `M3`, `S14` and writes them into the results JSON. They are named after the
questions they answer, not after what they compute. This is the whole vocabulary:

| name | the question it answers | key fields in the JSON |
|---|---|---|
| **M1** | *Is the model any good at imputation and denoising?* Accuracy, calibration, and how it compares to a "predict the per-assay average" baseline. | `M1.imp_macro_crps`, `M1.imp_macro_crps_oracle_scaled`, `M1.imp_macro_spearman_raw`, `M1.imp.ece`, `M1.imp_beats_marginal_n` |
| **M2** | *Does the model actually respond when you change the covariates you ask for?* This is the steering measurement — the reason this kit exists. | `M2.depth.median_total_slope`, `M2.ablation.assay_id.mean_abs_d_eta`, `M2.run_type.overall_clustered` |
| **M3** | *Has the encoder learned biology rather than measurement artifacts?* Compares the latent for the same genomic region seen at different sequencing depths against different regions. | `M3.ratio`, `M3.invariance_ok` |
| **S14** | *Depth counterfactual:* if you tell the model a different depth, does its prediction get closer to what that depth's data actually looks like? | `S14.frac_min_at_true`, `S14.frac_beats_told1` |

### How to read the numbers you get

| field | what it is | good | notes |
|---|---|---|---|
| `imp_macro_crps` | average CRPS over held-out assays; **lower is better**. CRPS scores a whole predicted distribution, not just a point — it is the distributional analogue of mean absolute error. | ~1.35 on the 8-assay reference panel | Scale depends entirely on your panel's count magnitudes. Only compare runs on the *same* bake. |
| `imp_macro_crps_oracle_scaled` | the same CRPS after granting each assay its best possible single multiplicative rescale. The gap to `imp_macro_crps` is `scale_error` — the part of your error that is just a wrong per-assay scale factor, and is therefore cheaply fixable. | ~1.31 | This split is the most useful diagnostic in the kit. A large `scale_error` means "calibration", not "the model does not understand the biology". |
| `imp_macro_spearman_raw` | rank correlation with truth, averaged per assay; **higher is better**, max 1.0 | ~0.56 | Per-assay (not pooled) so it cannot be inflated by getting the cross-assay scale right. |
| `imp.ece` | calibration error: how far the predicted uncertainty is from the truth's actual spread. **Lower is better**, 0 is perfect. | ≤ ~0.05 | This is what "confidence-aware" buys you; a good CRPS with a bad ECE means overconfident predictions. |
| `imp_beats_marginal_n` | of your held-out assays, how many the model beats a per-assay marginal baseline on | as high as possible; 7/8 on the reference panel | **The honest sanity check.** If this is near zero the model is losing to "always predict this assay's average". |
| `M2.depth.median_total_slope` | you told it depth `d`, did its predicted mean scale like `2^d`? | ≈1.0 | **1.0 is not automatically impressive.** With the offset head ON this is a closed-form arithmetic identity, not something learned. See §4. |
| `M2.ablation.assay_id.mean_abs_d_eta` | how much the prediction moves when you change *only* the assay you ask for | large = real steering | ~0.005 means functionally none. |
| `M3.ratio` | within-region latent distance ÷ between-region distance; **lower is better** | **< 0.30** = `invariance_ok` | Below the bar means the encoder groups by genomic region rather than by sequencing depth — i.e. it learned biology. ~0.21 on the reference panel. |
| `S14.frac_min_at_true` | fraction of targets where the true told-depth scored best | **0.25 is not chance** — it is the value you get from always preferring depth 1. A perfect model caps near **0.73**, not 1.0. | The eval prints both calibrations at runtime so you cannot misread it. |

`den_*` fields are the *denoising* counterparts of the `imp_*` ones (predicting a track the model was
shown, rather than one held out). `obs`/`imp` in this codebase mean **unmasked / masked**, not
biological observed/imputed.

**Before comparing two runs**, read §5: the noise floor on `imp_macro_crps` is about **±0.09**, and a
seed change alone moves it by ~0.12. Differences smaller than that are not differences.

## 3. The 4-arm results table

Seed 0, full chr21 eval coverage: 608 units, 1215 target-records over **12 held-out targets**. Source:
`research/H48_SCORECARD.md` §M1.

| arm | macro CRPS | oracle-scaled (capability) | scale_error | macro Spearman | pooled imp Spearman | ECE | M3 ratio | beats-marginal |
|---|---|---|---|---|---|---|---|---|
| `wd0_on_s0` | **1.3413** | **1.3077** | 0.0336 | **0.5653** | **0.6372** | **0.0533** | 0.2154 | **7/8** |
| `main_s0_perassay` | 1.4950 | 1.4210 | 0.0740 | 0.5051 | 0.5327 | 0.0615 | 0.2443 | 5/8 |
| `offoff_s0_perassay` | 1.9023 | 1.3871 | 0.5152 | 0.4647 | 0.4007 | 0.0968 | 0.1974 | 2/8 |
| `wd0_off_s0` | 2.0561 | 1.4026 | 0.6535 | 0.4641 | 0.3800 | 0.0782 | 0.2185 | 1/8 |

**Do not read the 4-decimal ordering as a ranking.** See §5.

### Column glossary

| column | what it is |
|---|---|
| **macro CRPS** | Continuous Ranked Probability Score of the predicted Negative Binomial against the true integer counts, averaged per assay then across the 8 assays. CRPS is the integral of `(F_pred(k) - 1[k >= y])^2` over counts — a *proper* score that rewards a sharp distribution centred on the truth. Lower is better; units are counts, so a broad high-count assay dominates unless you macro-average, which is why we do. `src/candi_kit/metrics.py:35`. |
| **oracle-scaled (capability)** | macro CRPS recomputed after granting each assay its single best multiplicative rescale `c* = argmin_c CRPS(NB(n, mu·2^c), y)`. This strips out "the model got the overall level of this assay wrong" and leaves "the model got the shape and the relative structure wrong". It is an **in-sample upper bound** — `c*` is fitted on the same 12 targets it scores. |
| **scale_error** | `macro CRPS − oracle-scaled`. The part of the error that one constant per assay would fix. Can be very slightly negative (−0.0008 observed) because `c*` is fitted on a 20k subsample and evaluated on the full pool. |
| **macro Spearman** | within-assay rank correlation between predicted mean and true counts on held-out (masked) positions, averaged across assays. Scale-free, so it measures *shape/biology* independent of the level error above. Higher is better. |
| **pooled imp Spearman** | the same rank correlation pooled over all held-out positions of all assays at once. Dominated by whichever assay contributes the most positions; reported because it is the historically quoted number, not because it is the better one. |
| **ECE** | calibration error from the **non-randomized PIT** reliability curve (Czado–Gneiting–Held 2009): mean `|F̄(u) − u|` over a grid of `u`. `F̄(u) = u` exactly iff the forecast distribution is calibrated. 0 is perfect. Interval-coverage ECE is *not* used — it spuriously over-covers at low counts, and most epigenomic bins are low-count. `src/candi_kit/metrics.py:95-117`. |
| **M3 ratio** | latent invariance: encode the same genomic region at input downsampling factors {1,2,4,8} (each with its true metadata) and take `mean within-region cosine distance / mean between-region cosine distance`. Low = the encoder maps the same biology to the same latent regardless of sequencing depth, i.e. it used the depth metadata to normalize the nuisance away. ≤0.3 was the internal pass bar. **Caveat: these four M3 values were carried forward from a pre-fix run** — they use the old (permuted) assay labels and a `between` pool that admits same-region pairs, which deflates `between` and inflates the ratio. The kit's `src/candi_kit/eval.py:917-932` fixes the pool, so your M3 numbers will not be bit-comparable to this column. |
| **beats-marginal** | of 8 assays, how many the model beats on CRPS against an **honest** per-assay baseline: the CRPS-optimal *constant* NB forecast for that assay. The previously reported "8/8" used a legacy baseline that degenerated to a point mass at 0 — i.e. a forecast of "predict nothing", which anything beats. Both are emitted; only the honest one is quoted here. `src/candi_kit/eval.py:331-371`. |

Where the scale error actually lives: DNase-seq, badly. Per-assay CRPS 3.539 (`wd0_on`, `c*` = +0.12)
versus 7.438 (`wd0_off`, `c*` = −2.08) — a factor 2^2.08 ≈ 4x level error on one assay
(`research/H48_SCORECARD.md` §M1 per-assay). Note that all per-assay labels in the internal record were
permuted before finding F5; the labels above are the corrected ones.

---

## 4. What each arm actually buys you

### offset ON — the best imputer, with an arithmetic depth knob and no learned steering

- **Imputation.** Best in every M1 column, and it beats the honest per-assay marginal on 7 of 8 assays.
  If your deliverable is imputed or denoised counts, this is the arm.
- **"Depth control" — say this precisely.** Total told-depth slope is **1.0000**: tell the model a depth
  one log2 unit lower and its predicted `log2_mu` drops by exactly one. That is *not* learned. It is the
  identity `log2_mu = (depth − depth_center) + eta` evaluated at a different `depth`, and the Negative
  Binomial is closed under thinning with `n` preserved, so this closed-form response is the *arithmetically
  correct* answer for downsampling. The learned residual is `eta_slope = -0.0000` / `0.0000`: the network
  contributes nothing. Calling this "the model learned to condition on depth" is a category error. It is
  a useful, exact, free depth knob — describe it as arithmetic.
- **Covariate steering: functionally null.** Sentinel-free real→real `assay_id` ablation (swap a real
  assay id for another real assay id at one slot, measure `max|Δη|`) gives **0.0023** for `wd0_on` and
  **0.0000** for `main` — against a pre-registered functional bar of ≥0.10, i.e. 43x short, and against
  4.1772 / 9.7144 on the two offset-OFF arms measured with the identical probe (`H48_REPORT.md` §F2).
  The earlier headline "0.833" was a **MISSING-sentinel artifact**: that protocol permuted the whole
  `assay_id` row, sliding the MISSING(−1) sentinel on and off *unavailable* slots whose prompt columns are
  all `(−1,−1,−1,−1)`. Run_type is likewise dead: target-clustered CI **[−0.00066, +0.000087]**,
  sign-test p = 1.000.
- **Mechanism, if you want to fix it.** The assay signal survives the metadata fusion MLP intact and is
  destroyed at the **fusion LayerNorm**: `wd0_on`'s pre-LN activation norm is 396 versus 8.9 on the
  offset-OFF arm, so LayerNorm divides the assay perturbation by ~70 instead of ~1.6; `film_proj` then
  attenuates a further ~18x. End-to-end deficit ~4 orders of magnitude (`H48_REPORT.md` §F2).
- **Known confound on any assay-steering claim.** In this panel `assay_id ≡ slot index`, and the decoder
  trunk emits a dedicated channel block per slot, so assay identity is carried *structurally*; the prompt
  row is informationally redundant. Any measurement here bounds the **prompt pathway**, not the model's
  assay-awareness.
- **Do not add weight decay to this arm.** `main_s0_perassay` is `wd0_on` plus `weight_decay=1e-4`, and
  the decay annihilates the decoder's assay-embedding table (~1e-40, bit-exactly blind from the fusion's
  first `Linear` onward) while also costing 0.15 macro CRPS. Setting `wd=0` prevents the annihilation
  — but note the resulting table sits at *random-init statistics* (element std 0.94 vs 0.97 for a fresh
  N(0,1) table), so it was never destroyed **and** never trained. "Revived" is the wrong word.

### offset OFF — genuine learned steering, at a real imputation cost

- **Steering is learned and real.** Sentinel-free assay ablation `max|Δη|` = **4.1772** (`offoff`) /
  **9.7144** (`wd0_off`), 1816x–4224x the offset-ON arms on the identical probe. `offoff`'s run_type
  response is the only one that survives target-clustered inference: CI **[+0.1179, +2.1804]**,
  sign-test **p = 0.039** (`research/H48_SCORECARD.md` §M2). `wd0_off` moves more but not directionally:
  CI **[−0.2326, +9.4084]**.
- **The cost.** macro CRPS 1.9023 (`offoff`) / 2.0561 (`wd0_off`) versus 1.3413 — **+42% / +53%** — and
  beats-marginal collapses to 2/8 and 1/8. Pooled imp Spearman falls from 0.6372 to ~0.40.
- **The cost is almost entirely level, not biology.** `scale_error` is 0.5152 / 0.6535 on these arms
  versus 0.0336 on `wd0_on`; on the capability term they are 1.3871 / 1.4026 versus 1.3077. See §6.
- **Depth response.** `offoff` total slope 0.8869, `wd0_off` 1.0325 — learned, and *worse* than the
  offset-ON arms' exact 1.0000. Removing the offset does not improve depth handling; it replaces exact
  arithmetic with an approximation that happens to be attributable to the network.

### The hybrid was never built — and is not experimentally excluded either

Internal node **h45** pre-registered exactly the hybrid you are about to think of — offset warm-up then
anneal to zero, a fixed attenuation β ∈ {0.25, 0.5, 0.75}, and a learned metadata-driven scale head
replacing `2^(d − center)`. Its recorded verdict is **refuted, 0 of 4 verifiables met**. Read that
precisely, because it is easy to over-read:

- **No hybrid arm was ever trained.** The node's "Run Links" section is empty; all four verifiables were
  closed on *premises* (the offset-OFF η-slope 0.88 was a measurement artifact; run_type does not survive
  clustering and is unidentifiable on this panel; h47 appeared to get magnitude *and* steering without a
  hybrid, so the fourth was marked "obviated").
- **One leg of that refutation has since been retracted.** The "h47 gets both" leg rested on the assay
  Δη = 0.833 that `H48_REPORT.md` § F2 showed to be a MISSING-sentinel artifact (sentinel-free 0.0023).
  The node carries a **2026-07-28 flag that its refutation basis is under review**, with verdict/status
  deliberately left unchanged pending a PI call.

So: the hybrid is **neither demonstrated nor ruled out by experiment**. Do not present it as either.
The one idea that survived from h45 and is independently motivated is the explicit per-assay scale
factor in §6.

---

## 5. The statistics — read this before quoting any number above

**Effective replication is 12 held-out targets / 5 biosample pairs / 4 cell types**, with the single pair
`(T_RWPE2, B_RWPE2)` supplying **7 of the 12**. The `single` run_type subgroup is n=3 clusters
(`METADATA_AUDIT.md` bound B4).

- **Noise floor.** Target-clustered bootstrap noise floor on macro `crps_oracle_scaled` is **~0.09**;
  per-comparison uncertainty is **±0.13**. The full arm spread on that column is 0.113.
- **Only one pairwise comparison survives inference.** Paired bootstrap clustered on the 12 targets:

  | comparison | Δ | 95% CI | verdict |
  |---|---|---|---|
  | `offoff` − `wd0_on` | +0.093 | [+0.004, +0.217] | excludes 0 |
  | `main` − `offoff` | +0.023 | [−0.117, +0.153] | covers 0 |
  | `main` − `wd0_off` | +0.013 | [−0.091, +0.120] | covers 0 |
  | `wd0_off` − `offoff` | +0.010 | [−0.047, +0.055] | covers 0 |

  Defensible statement: **`wd0_on` is best on capability.** `main`, `offoff` and `wd0_off` are
  statistically indistinguishable from each other.
- **The reordering is NOT established.** The quoted point-estimate ordering is the modal bootstrap
  ordering at only **45% of replicates**. `P(main worst of four) = 0.54` — a coin flip. Target-level sign
  test `main` vs `offoff` is 7+/5−, **p = 0.77**. On an independent 1/7 genomic spread `main − offoff`
  **flips sign to −0.036**, and `main` is 3rd rather than last in 4 of 12 leave-one-target-out replicates.
- **Seed deltas are comparable to between-arm gaps.** A single seed change moves pooled imp CRPS by
  **0.1195**, pooled imp Spearman by **0.0562**, ECE by **0.0354**, M3 ratio by **0.0479**. Any single-seed
  difference below ~0.12 CRPS is uninterpretable. Run ≥3 seeds and compare means.
- **Sign-test resolution at n=12 is quantized:** 10/12 → p=0.039, 11/12 → 0.0063, 12/12 → 0.00049.
  "p < 0.05" here means "at least 10 of 12 targets agree". There is no finer resolution available.
- **⚠ Position-level CIs in any stored JSON are a fiction.** Older results bootstrapped ~893,528 *pooled
  positions*; positions within a target are not independent draws and the resulting intervals are ~24x too
  narrow, with median CI widening 18x once you cluster correctly. Use the `*_clustered` keys only. The kit
  gates every such key behind `--include-deprecated` and ships each one with its verdict string:
  `src/candi_kit/eval.py:49-68` (`DEPRECATED_VERDICTS`).
- **GPU-vs-CPU float:** macro CRPS reproduces to 4 decimals on identical weights; per-assay values move
  2e-6 … 3.3e-4. Anything at 1e-7 or below in `eta`/`mu` is float noise — an earlier boolean
  (`offset_independent`) was once decided by the sign of a 1e-17 quantity.
- **One withdrawn instrument.** The S23 "condition recoverability" probe (leave-one-target-out nearest
  centroid on latent features) is **not validated** and its ordering inverts against every other
  measurement. Do not cite it in either direction.

---

## 6. The 84% result — most of the gap is per-assay scale, not capability

Under the oracle per-assay rescale, the four-arm macro-CRPS spread compresses

> **0.7148 → 0.1133 (84% compression)**

— independently reproduced at 0.7099 → 0.1138 (84.0%) and at 83.5% on a separate genomic spread
(`H48_REPORT.md` §F1). This is the single strongest, most reproducible result in the whole node.

**What it means.** The offset-OFF arms are not much worse at modelling epigenomic structure. They are
worse at getting one number per assay right — the overall output level. Their entire visible deficit is
0.52–0.65 macro CRPS of *fixable per-assay scale*. The same fact shows up independently in the depth
counterfactual: at raw scale no arm passes, but correcting **one per-target constant** fitted at
(ground-truth depth = told depth = 1), leaving the told-depth response exactly as produced, flips all four
arms to passing (`frac_beats_told1` → 1.000 / 0.972 / 1.000 / 0.778). That is a **level** failure, not a
steering failure (`H48_REPORT.md` §F3).

**What to do about it.** The head in this kit is a weight-shared `Linear(16,16) → GELU → Linear(16,1)`
(`src/candi_kit/model.py:172-173`) with a single scalar output bias, so the only per-assay knob is a
rank-~2 FiLM — where 8 assays with very different dynamic range and dispersion need ~7 degrees of freedom.
The proposed fix, internal node **h50**, is an explicit metadata-*independent* per-assay output factor:
a per-assay scale and bias on `eta` plus a per-assay offset on `log n`, indexed by **slot** (not by the
metadata prompt), excluded from weight decay — **3 parameters × 8 assays = ~24 parameters**.

**h50 has not been run.** It is a hypothesis with a pre-registered bar, not a result. Its bar: macro
`crps_oracle_scaled` below 1.3077 *by more than the ~0.09 noise floor*, with the gain attributable to the
`scale_error` term shrinking rather than to shape, subject to macro Spearman ≥ 0.5653 and ECE ≤ 0.0533.
Rest the case for it on the **compression** result, which is rock solid — not on the arm reordering,
which is not established.

---

## 7. Decision guide

| if you need … | do this |
|---|---|
| **the best imputation / denoising** | `--offset on --weight-decay 0.0`. Best CRPS, best Spearman, best ECE, beats the honest marginal 7/8. Report the depth knob as exact arithmetic (slope 1.0000), never as learned conditioning. Expect no assay or run_type steering at all. |
| **to demonstrate learned covariate steering** | `--offset off --weight-decay 1e-4` (the `offoff` arm — it is the only one whose run_type direction survives clustering). Budget +42% macro CRPS and beats-marginal dropping to 2/8, and report both. |
| **both** | **This does not exist yet, and it has not been tested.** h45 pre-registered the hybrid and is recorded refuted 0/4, but no hybrid arm was ever trained (§4). Shortest path: implement **h50** (§6) on top of the `--offset off` arm — ~24 parameters, no decay, slot-indexed — and check whether `scale_error` collapses while the steering numbers hold. Second lever, from the §4 mechanism: the fusion-LayerNorm scale on the offset-ON arm, not embedder revival. Both are unrun. |
| **a depth-conditioned generative model of counts** | `--offset on`. The thinning identity is exactly what you want and it is free. But see the DSF bound below — the *upward* direction is untrained. |
| **a run_type steering demo on the shipped 8-assay panel** | **Impossible. Re-select the panel first.** See below. |

### Known bounds that cap what any arm can demonstrate

Source: `METADATA_AUDIT.md` §B.

- **B1 — run_type is analytically unidentifiable on this panel.** `H(run_type | assay_id, read_length) =
  0.000 bits` (n=26): run_type is a deterministic function of the other two covariates, so any function of
  it is reproducible at zero loss cost. Only one assay has within-assay variance, and its single paired
  record is simultaneously the unique read_length=100 record and the deepest — the entire learnable signal
  rests on **n=1**. The full ENCODE Imputation Challenge panel retains **0.551 bits** after conditioning
  on assay, so this is a property of the 5-biosample slice, not of the field. **A run_type steering demo
  needs a re-selected biosample panel** that breaks `run_type ⊥ (assay, read_length)` — a small
  combinatorial search over the source experiment table, no architecture change. Highest-leverage
  data-side fix available.
- **B5 — depth, read_length and run_type are mutually collinear** at n=38 (assay-centered
  `corr(d, log2 rl) = 0.763`, `corr(d, rt) = 0.590`, `corr(log2 rl, rt) = 0.697`). You can fit them
  jointly; you cannot cleanly credit them. Any claim of the form "covariate X carries Y% of the exposure
  signal" is unsupportable on data with this structure.
- **B6 — DSF only *down*-samples, so upward depth extrapolation is never trained.** Training support is
  `[natural_min − 3, natural_max]` in log2 depth, and **7 of 12 eval targets sit above their per-assay
  training ceiling** (worst by +1.43 log2). Depth steering measured on those targets is extrapolation, and
  the error direction is systematically negative by construction.
- **B7 — 9 of 12 eval targets are out-of-distribution in (assay × read_length)**, and the read_length flip
  lands outside per-assay training support in 7 of 12. This is why read_length responsiveness is shipped
  as a deprecated key.
- **B9 — eval `x_data == y_data`**, so nothing in the current suite measures *denoising* as distinct from
  autoencoding. That is an eval-configuration property (`dsf_sampling='off'`, `apply_mask=False`), not an
  architecture limit — but do not quote these numbers as denoising performance.

### What this model does not do

It is **counts-only Negative Binomial**. There is no Gaussian p-value head and no Bernoulli peak head, so
there are no peak precision/recall/AUROC numbers and no p-value track anywhere in this kit. `y_pval` and
`y_peaks` are carried through the h5 and the batch pipeline but are never supervised; re-adding those
heads is a documented extension, see `EXTENSION_HOOKS.md`.

---

## Reproduction on this kit

**Run, and it reproduces.** 2026-07-29, Fir, 6 SLURM arms (offset ON/OFF × seeds 0,1,2), each 25 epochs
on a `1g.10gb` MIG slice, ~68–76 min wall. Every number below is an output of *this* code trained from
scratch on the real EIC directory — not a re-score of a research-repo checkpoint.

Panel: `configs/panel.q19.json`, baked by `slurm/bake.sh` (18 min) — 7,485 windows = 5,485 type1
(chr19 3,053 train / chr21 2,432 eval) + 2,000 type2 (1,000 cCRE + 1,000 non-cCRE). Identical window
composition to the h5 the anchors were measured on.

### Offset ON — the imputation arm

| metric | kit, mean ± sd (n=3) | anchor (`wd0_on_s0`) | Δ | acceptance | verdict |
|---|---|---|---|---|---|
| macro CRPS | **1.3504** ± 0.0122 | 1.3413 | +0.009 | ±0.09 | PASS |
| capability (`crps_oracle_scaled`) | **1.3116** ± 0.0179 | 1.3077 | +0.004 | ±0.09 | PASS |
| macro Spearman | **0.5562** ± 0.0073 | 0.5653 | −0.009 | ±0.06 | PASS |
| pooled imp Spearman | **0.6116** ± 0.0119 | 0.6372 | −0.026 | ±0.06 | PASS |
| PIT-ECE | **0.0246** ± 0.0019 | 0.0533 | −0.029 | ±0.035 | PASS (better) |
| beats honest marginal | **7–8 / 8** | 7 / 8 | — | ≥ 7 | PASS |
| M3 latent-invariance ratio | **0.205–0.279** | 0.2154 | — | < 0.30 | PASS |
| total told-depth slope | **1.0002** ± 0.0003 | 1.0000 | +0.0002 | ≈1.000 | PASS |

### Offset OFF — the steering arm

| metric | kit, mean ± sd (n=3) | anchor (`offoff_s0_perassay`) | Δ | verdict |
|---|---|---|---|---|
| macro CRPS | **1.8307** ± 0.1803 | 1.9023 | −0.072 | PASS (within ±0.09) |
| capability | **1.4574** ± 0.0891 | 1.3871 | +0.070 | PASS |
| macro Spearman | **0.4726** ± 0.0212 | 0.4647 | +0.008 | PASS |
| macro scale_error | **0.2283–0.6535** | 0.5152 | — | PASS (≥ 0.2 every seed) |
| total told-depth slope | **0.8565** ± 0.1382 | 0.8869 | −0.030 | PASS |
| beats honest marginal | **1–3 / 8** | 2 / 8 | — | PASS |

### The structural sign pattern — the primary criterion

This is the sharper test, and it reproduces cleanly:

| | offset ON | offset OFF |
|---|---|---|
| depth slope | **1.0002** — the exact closed-form arithmetic identity | **0.8565** — learned, and seed-unstable |
| assay-prompt response (`mean_abs_d_eta`) | **0.0048** — functionally null, ~20× under the 0.10 bar | large |
| run_type direction supported? | **no** in 2/3 seeds; the one "yes" has CI `[+0.0002, +0.030]`, i.e. null in magnitude | **yes** in 2/3 seeds, CIs `[+0.32, +5.33]` and `[+0.54, +6.63]` |
| imputation | **best** | +36% macro CRPS |

So the Pareto is real, it reproduces from scratch, and it is not an artifact of one seed.

### Two things this run adds that the single-seed anchors could not show

1. **The offset-OFF arm is far less stable across seeds.** Its macro-CRPS sd is **0.1803** against the
   ON arm's **0.0122** — roughly 15×. Its depth slope ranges 0.700–1.036 across three seeds. Any
   single-seed offset-OFF number, including the recorded anchor, should be read as one draw from a wide
   distribution. The recorded arms were single-seed, so this could not have been seen before.
2. **run_type steering is seed-dependent even with the offset off** (supported in 2 of 3 seeds). Combined
   with bound **B1** — run_type is analytically unidentifiable on this panel,
   `H(run_type | assay_id, read_length) = 0.000 bits` — do not read a positive run_type result on this
   panel as evidence of a learned run_type response. Re-panel first (§6).

### One honest discrepancy

The kit stores counts as **`int32`**; the research pipeline used `int16` and **silently clipped** every
bin above 32,767. Real data exceeds that: `B_DND-41`/DNase-seq reaches **52,051** in a cCRE-sampled
type2 window. So the kit trained on *corrected* data where the anchors trained on *clipped* data. The 12
held-out eval targets are chr21 type1 windows, which top out near 5,961 and were never affected, so the
comparison above is sound — but it is a reproduction on repaired inputs, not a bit-identical rerun. The
clip touched type2 *training* windows only. `DATA.md` §6 has the detail.

Acceptance is deliberately set **at the noise floor** (±0.09 macro CRPS, ±0.06 Spearman), so these gates
confirm "the kit reproduces the recipe" but cannot rule out a small systematic regression — a limit of a
12-target / 5-biosample-pair / 4-cell-type panel, not of the thresholds. Full spec: `.BUILD_PLAN.md`
VALIDATION_PLAN Gate B.

### Reproducing it yourself

```bash
sbatch candi_kit/slurm/bake.sh                                    # ~18 min -> /scratch/$USER/candi_kit/q19.h5
sbatch --dependency=afterok:<BAKE_ID> candi_kit/slurm/train.sh    # 6 arms, ~76 min, ~6 GPU-hours
```

Per-arm JSON lands in `$OUT/kit_{on,off}_s{0,1,2}.json`. The fields above are `M1.imp_macro_crps`,
`M1.imp_macro_crps_oracle_scaled`, `M1.imp_macro_spearman_raw`, `M1.imp.ece`,
`M1.imp_beats_marginal_n`, `M2.depth.median_total_slope`, `M2.ablation.assay_id.mean_abs_d_eta`,
`M2.run_type.overall_clustered` and `M3.ratio`.

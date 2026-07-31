<!-- candi_kit-reader-preamble -->

> [!NOTE]
> **Reader preamble — added for this kit. Everything below the horizontal rule is the
> original document, unedited.**

**How to read this from outside the project.** This is a frozen, verbatim copy of the results table
produced by the re-score described in `H48_REPORT.md`. Its numbers are the source for the four-arm table
in `TRADEOFF.md`.

**Status has since changed — the body below was not edited.** It calls itself "PROPOSED" and says the
node "is not closed"; the verdict was accepted and recorded on 2026-07-28. Read that as historical.

**One thing to get right before quoting anything here:** the coverage note in its own header. The `M1`
and `M2` sections use full evaluation coverage, but the `S14` and `S23` sections do not — they use a much
smaller sample. The header's coverage figures do **not** apply to the last table.

### The `h<N>` / `q<N>` ids

Those are entries in the project's research tracker (`h` = a testable hypothesis, `q` = an open
question). The tracker is not shipped. **[`README.md` in this folder](README.md) has a decoder ring
translating every id used anywhere in this kit into a plain-language claim and its verdict.** Look an id
up there rather than inferring it from context.

---

# h48 — proposed re-scored scorecard (corrected instruments)

CPU re-score of the four existing q19/h47 checkpoints with the six h48 instrument fixes (S1, S3, S4, S5/S18, S6, S14/S23). No retraining, no GPU. M1 and M2 use full chr21 eval coverage (608 units, 1215 target-records over 12 held-out targets); the S14 columns use 4 region-draws x 4 windows and the S23 columns a 120-record even stride — **the header coverage does NOT apply to the last table.**

**This is a PROPOSED scorecard. The verdict on h48 is the PI's call — the node is not closed.**
Post-verification (6 adversarial skeptics + a critic); read `../H48_REPORT.md` §2-§3 for the claims these numbers do and do not support.

## M1 — magnitude, decomposed (S5/S18)

`crps_oracle_scaled` is the CAPABILITY term (CRPS after granting each assay its oracle multiplicative scale `c*`); `scale_error` is the FIXABLE per-assay calibration term. Macro CRPS mixes the two, which is what made the ON/OFF difference look like a Pareto.

> **The first two columns are a REGRESSION CHECK — they are expected to be identical.** Both are the same computation on the same points; differences of ~1e-4 are CPU-vs-GPU float and confirm the re-score recomputes from the checkpoint rather than reading a cache.
> **The capability column has a target-clustered noise floor of ~0.09.** Only `wd0_on`'s lead is significant (offoff - wd0_on = +0.093 [+0.004, +0.217]); the other three are statistically indistinguishable (pairwise CIs all cover 0). Do NOT read the 4-dp ordering as a ranking.
> `crps_oracle_scaled` is an **in-sample upper bound**: `c*` is fitted on the same 12 targets it scores, and `scale_error` can be slightly negative (subsample fit, full-pool evaluation).

| checkpoint | macro CRPS (old anchor) | macro CRPS (new) | **oracle-scaled (capability)** | scale_error | macro Sp | imp Sp (pooled) | ECE | beats-marg (honest) | beats-marg (legacy) |
|---|---|---|---|---|---|---|---|---|---|
| `main_s0_perassay` | 1.4950 | 1.4950 | **1.4210** | 0.0740 | 0.5051 | 0.5327 | 0.0615 | 5/8 | 8/8 |
| `offoff_s0_perassay` | 1.9020 | 1.9023 | **1.3871** | 0.5152 | 0.4647 | 0.4007 | 0.0968 | 2/8 | 3/8 |
| `wd0_on_s0` | 1.3410 | 1.3413 | **1.3077** | 0.0336 | 0.5653 | 0.6372 | 0.0533 | 7/8 | 8/8 |
| `wd0_off_s0` | 2.0560 | 2.0561 | **1.4026** | 0.6535 | 0.4641 | 0.3800 | 0.0782 | 1/8 | 5/8 |

Spread across the four arms: raw macro CRPS **0.7148** → oracle-scaled **0.1133** (84% compression). This is the audit §6.5 question: most of the apparent ON/OFF magnitude frontier is a per-assay scale-calibration difference, not a capability difference.

## M1 — per-assay, with CORRECTED labels (S6)

h5 column order is the CANDI handler's `experiment_aliases` order, not `SANDBOX_ASSAYS`; every previously reported per-assay label was permuted (38/38 vs 5/38 on a metadata join). Training saw only integer assay ids, so the numbers stand — only the biology reading changes.

| checkpoint | DNase-seq | H3K4me3 | H3K36me3 | H3K27ac | H3K9me3 | H3K27me3 | H3K4me1 | ATAC-seq |
|---|---|---|---|---|---|---|---|---|
| `main_s0_perassay` CRPS | 3.727 | 1.917 | 1.579 | 0.445 | 0.797 | 1.459 | 1.041 | 0.994 |
| `main_s0_perassay` c* | +0.45 | -1.14 | +0.53 | +0.39 | -0.33 | +0.76 | +0.78 | -0.36 |
| `offoff_s0_perassay` CRPS | 5.555 | 1.642 | 2.029 | 0.609 | 1.290 | 1.886 | 1.197 | 1.010 |
| `offoff_s0_perassay` c* | -1.59 | +1.53 | +2.33 | +2.28 | -1.44 | +3.29 | +2.48 | +1.42 |
| `wd0_on_s0` CRPS | 3.539 | 1.426 | 1.456 | 0.454 | 0.738 | 1.280 | 0.895 | 0.942 |
| `wd0_on_s0` c* | +0.12 | +0.08 | +0.38 | +0.19 | -0.39 | +0.79 | +0.69 | +0.16 |
| `wd0_off_s0` CRPS | 7.438 | 1.655 | 1.963 | 0.616 | 1.149 | 1.664 | 1.017 | 0.947 |
| `wd0_off_s0` c* | -2.08 | +1.48 | +1.82 | +2.33 | -0.49 | +2.00 | +1.36 | +0.47 |

## M2 — steering, target-clustered (S4) and sentinel-free (S1)

CIs are now bootstrapped over the **12 targets** (n_fg-weighted), not the ~893k positions; `supports_direction` is sign-aware. `assay Δη` is the cross-target, sentinel-free ablation.

| checkpoint | total depth slope | eta_slope (diagnostic) | assay mean\|Δη\| | assay max\|Δη\| | run_type clustered CI | supports dir? | sign-test p |
|---|---|---|---|---|---|---|---|
| `main_s0_perassay` | 1.0000 | -0.0000 | 0.000e+00 | 0.000e+00 | [+0.0000, +0.0000] (n_cl=12) | no | n/a |
| `offoff_s0_perassay` | 0.8869 | 0.8869 | 2.926e-01 | 3.021e+00 | [+0.1179, +2.1804] (n_cl=12) | YES | 0.039 |
| `wd0_on_s0` | 1.0000 | 0.0000 | 4.603e-05 | 8.202e-04 | [-0.0007, +0.0001] (n_cl=12) | no | 1.000 |
| `wd0_off_s0` | 1.0325 | 1.0587 | 1.078e+00 | 9.002e+00 | [-0.2326, +9.4084] (n_cl=12) | no | 0.039 |

`sign-test p` drops exact ties (a bit-exactly unresponsive arm has no signs to test and reads `n/a`, not a significant p-value). `total depth slope` should be read next to the clamp tail below — the median clamp fraction is 0 on all four arms, but a minority of targets do read the slope through the saturating `log2_mu` floor:

| checkpoint | targets with any clamp | p90 clamp frac | max clamp frac |
|---|---|---|---|
| `main_s0_perassay` | 0.0626 | 0.000 | 0.279 |
| `offoff_s0_perassay` | 0.0016 | 0.000 | 0.115 |
| `wd0_on_s0` | 0.1687 | 0.475 | 0.967 |
| `wd0_off_s0` | 0.1514 | 0.475 | 0.607 |

## S14 — real depth counterfactual

Each told depth is scored against its OWN `counts_dsf{k}` ground truth (the old sweep scored every told depth against the fixed dsf1 target, which any mu-decreasing model passes).

> **0.25 is NOT a chance level** for `frac_min_at_true` -- it is the deterministic value of "argmin always at told=1". And because the foreground is the top 2% of the level-k realization being scored, an exactly-correct model reads as 2.2x under-predicting at dsf8, capping `frac_min_at_true` at **~0.73** (verified positive control: a perfectly depth-scaled oracle scores 0.7292 / 0.9167).
> **This is a LEVEL failure, not a depth-steering failure.** Correcting ONE per-target constant fitted at (GT=1, told=1) -- leaving the told-depth response untouched -- flips all four arms to passing (`frac_beats_told1` main 1.000, wd0_on 0.972, wd0_off 1.000, offoff 0.778). It is the same per-assay scale error the M1 table quantifies.

| checkpoint | frac_min_at_true (ceiling ~0.73) | frac beats told=1 |
|---|---|---|
| `main_s0_perassay` | 0.2708 | 0.1111 |
| `offoff_s0_perassay` | 0.2292 | 0.0278 |
| `wd0_on_s0` | 0.2500 | 0.0556 |
| `wd0_off_s0` | 0.2708 | 0.1667 |

## S23 -- condition recoverability :warning: WITHDRAWN, DO NOT CITE

> **Not a validated instrument: its ordering is INVERTED against every other measurement.** `offoff_s0_perassay` -- one of the two arms with real assay steering, carrying ~5900x more feature energy than `wd0_on_s0` -- scores BELOW the 0.125 chance level, while `wd0_on_s0` scores ~2.5x higher on essentially no signal. Cause: leave-one-target-out nearest centroid on within-target-centred features penalises a target-ADAPTIVE response whose direction flips sign across targets, and can reward a deterministic near-zero one. De-prefixing the probe did not fix it. Reliable ONLY as a bit-exactly-dead detector (`main_s0_perassay` correctly reads exact chance at ~0 energy). Retained for the record.

| checkpoint | assay acc | assay energy | run_type acc | rt energy | depth acc | depth energy |
|---|---|---|---|---|---|---|
| `main_s0_perassay` | 0.1250 | 2.31e-17 | 0.5000 | 0.00e+00 | 0.3750 | 3.67e-08 |
| `offoff_s0_perassay` | 0.0907 | 1.62e-01 | 0.8333 | 5.97e-01 | 0.7899 | 9.87e-01 |
| `wd0_on_s0` | 0.3142 | 2.73e-05 | 0.9688 | 1.97e-04 | 0.5625 | 9.42e-05 |
| `wd0_off_s0` | 0.4197 | 4.13e-01 | 0.9167 | 1.13e+00 | 0.7674 | 1.25e+00 |

Chance: assay 0.125, run_type 0.500, depth 0.250. **Accuracy without feature energy is meaningless here.**

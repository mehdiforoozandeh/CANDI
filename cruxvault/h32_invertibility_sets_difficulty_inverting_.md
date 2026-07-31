---
id: h32
type: idea
title: Invertibility sets difficulty; inverting f_x (input) is harder than applying f_y (output)
parent: q15
status: done
verdict: partial
metric: invert-input harder than apply-output STRONG (M1 lossy-in 0.211 vs lossy-out -0.015; M2 0.442->~0.35); but invertibility!=difficulty (add is encoder-hard, not thin/cap/clog); tail-locus universal
created: "2026-07-03T01:59:54"
updated: "2026-07-10T13:04:48"
---

# h32 — Invertibility sets difficulty; inverting f_x (input) is harder than applying f_y (output)

Parent:: [[q15_can_candi_learn_dual_metadata_conditioni]]

## Problem Statement

Is learnability graded by transform family - invertible (x-h, +h, power) easy vs non-invertible/information-losing (binomial thinning, saturating cap, count-log) hard - and is the INPUT side (encoder must invert/undo f_x) harder than the OUTPUT side (decoder applies f_y forward)?

## Idea / Hypothesis

Invertibility sets difficulty; inverting f_x (input) is harder than applying f_y (output)

## Verifiables

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
- [x] per-family difficulty ranking, read from the FULL f_x×f_y M1 matrix (identity-cell ceiling reference, no per-family normalization -- references chosen at interpretation): invertible families (mult/add/power) reach an off-diagonal M1 CRPS-gap within tolerance of the identity ceiling (~2a bar); non-invertible families (thin/cap/clog) show a strictly larger gap; families are ranked and any input-side transform that never learns is flagged.   (found: MARGINAL -- non-inv mean off-diag gap 0.211 > inv 0.190, but the SPLIT IS NOT CLEAN: the single hardest input family is `mult` (0.279, INVERTIBLE); add=0.147, power=0.145 are the easiest. Difficulty is not graded by invertibility. No family is unlearnable.)
- [ ] input-side invertibility cost (M3, per-f_x vector): the encoder within/between cos-dist ratio is higher for non-invertible INPUT transforms (thin/cap/clog) than for invertible ones -- the encoder cannot fully undo information loss (thinning/censoring).   (found: UNMET, opposite of prediction -- the encoder normalizes the non-invertible inputs FINE (M3 thin 0.051 / cap 0.058 / clog 0.069); the encoder-HARD family is `add` (0.618, an invertible background shift). Info-loss on the input is not what the encoder struggles with; additive background shift is.)
- [x] invert-harder-than-apply asymmetry, from the full matrices: (a) the M2 f_x×f_y steering matrix drops from the f_x=identity column to lossy-f_x columns (output-steering degrades when the encoder must ALSO undo a lossy input first); (b) lossy-INPUT off-diagonal M1 cells degrade more than lossy-OUTPUT off-diagonal cells; the effect concentrates in the non-invertible families.   (found: STRONGLY MET, the headline result -- (a) M2 steering drops from identity-input 0.442 to EVERY lossy input row (thin 0.34 / cap 0.36 / clog 0.38 / mult 0.37 / add 0.28 / power 0.34); (b) lossy-INPUT off-diag M1 gap 0.211 vs lossy-OUTPUT off-diag -0.015 -- applying a lossy transform on the OUTPUT is essentially FREE, undoing one on the INPUT is not. The asymmetry is general, not specific to non-invertible families.)
- [x] steering LOCUS (M2 mean-stat vs tail-stat decomposition): reshaping families (cap/clog/power) carry steering in the tail/dispersion statistic (pearson >= 0.5) while the mean statistic stays flat (<= 0.15); mean-moving families (mult/add) carry it in the mean statistic -- a mean-only M2 would misread the reshaping families.   (found: TAIL confirmed as the dominant/universal locus (tail r 0.92-0.99 for ALL families) -- but the "mean stays flat" half FAILS: reshaping mean-stat is 0.54 (cap) / 0.52 (clog) / 0.60 (power), not <=0.15. So the tail is not the EXCLUSIVE locus; still, the tail readout is load-bearing for `thin` (mean r only 0.23, tail 0.92) -- a mean-only M2 WOULD under-read thin. Distributional-M2 justified, strict locus prediction not.)

## Planned Intervention

Phase-2c: a single best-config run (norm=none / per-assay / offset-on / enc-naive, from the 2a winners) with the family menu expanded from 4 to 7 -- adds thin/cap/clog on BOTH sides, so the matrix grows 4x4=16 -> 7x7=49 cells. Budget bumped ~3x vs the 2a 25ep so per-cell coverage matches 2a (the difficulty read must not be confounded with undertraining -- the ERA artifact); a per-cell sample-count log + loss-plateau check separate "hard" from "undertrained". Full raw M1 (7x7 cell CRPS + Spearman) and the EXTENDED M2 (7x7 steering index, swept f_y at each f_x, persisted per-cell AND per-assay un-collapsed) matrices; M3 as a per-f_x vector. Eval on chr19 + chr21. References for the M1 difficulty ranking are chosen at interpretation from the full matrix (e.g. within-family diagonal vs identity ceiling) -- NOT normalized into the metric.

## Run Links

- sandbox/diagnostics/dual_conditioning phase-2c sweep (jobs/sweep_2c.sh; job 47900426, EP=70, chr19+chr21). Results in `results/norm-none_enc-naive_off-on_mode-per_assay_2c.json`; report §"Phase 2c" (F4 7x7 M1, F9 M2 matrix, F10 locus, T4).

## Findings

**Partial. The ONE robust result is the input/output asymmetry: inverting a transform on the INPUT is genuinely harder than applying one on the OUTPUT.** A lossy transform on the output side is essentially free (off-diagonal M1 gap -0.015, i.e. no cost), while the encoder undoing a transform on the input side costs a real 0.211 M1 gap; and output-steering (M2) falls from 0.442 with a clean input to ~0.28-0.38 under EVERY non-identity input transform. That asymmetry -- the crux of h32 -- holds cleanly and generally.

**But "invertibility grades difficulty" -- the hypothesis's framing -- does NOT hold.** (1) On reconstruction, the hardest single input family is `mult` (gap 0.279, invertible), while non-invertible thin/cap/clog (0.21) sit between the invertible extremes; the non-inv-mean > inv-mean gap (0.211 vs 0.190) is marginal and driven by mult, not the invertible/non-invertible split. (2) On the encoder side (M3) the prediction inverts: the encoder normalizes the information-losing inputs FINE (thin/cap/clog ratios 0.05-0.07) and instead struggles with `add` (0.618) -- an *invertible* additive background shift. So the axis of difficulty is **additive-shift-vs-rescale and general input-inversion load, not invertibility**. This corroborates q17's later finding that additive/background structure is where the model's trouble concentrates.

**Steering locus:** the tail/dispersion statistic is the dominant steering signal for every family (tail pearson 0.92-0.99), and is load-bearing for `thin` (mean-stat r only 0.23 vs tail 0.92) -- so the distributional M2 was necessary and a mean-only readout would have under-scored thin. However the pre-registered "reshaping-families' mean stays flat (<=0.15)" is false (cap/clog/power mean-stat ~0.5-0.6): steering shows in BOTH stats, the tail merely leads. Verdict **partial** -- asymmetry + tail-locus confirmed; invertibility-as-difficulty and the encoder info-loss cost refuted. See [[q15_can_candi_learn_dual_metadata_conditioni]].

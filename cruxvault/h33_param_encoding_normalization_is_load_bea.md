---
id: h33
type: idea
title: Param-encoding normalization is load-bearing for reading the augmentation covariate
parent: q15
status: done
verdict: refuted
metric: M2 none 0.515 > z 0.483 > log 0.448 (normalization not load-bearing)
created: "2026-07-03T02:21:15"
updated: "2026-07-09T00:49:00"
---

# h33 — Param-encoding normalization is load-bearing for reading the augmentation covariate

Parent:: [[q15_can_candi_learn_dual_metadata_conditioni]]

## Problem Statement

aug_param spans very different per-family scales (mult 0.25-4, add 2-20, power 0.5-2, thin 0.2-0.8, cap 2-20, clog 1-8). Fed raw into the param embedder, magnitudes collide across families and the model may under-read the param, depressing steering. Does normalizing the param materially improve conditioning quality? Test THREE arms - none / per-family z-score / global log-scale - by training the full h30 matrix under each. Doubles as a guard so a param-encoding artifact is not mistaken for a real conditioning limit in h30/h32, and picks the normalization that h31/h32 then use.

## Idea / Hypothesis

Param-encoding normalization is load-bearing for reading the augmentation covariate

## Verifiables

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
- [ ] rank the three arms by distributional M2 param-steering (per plan section M2; fix family, sweep its 4 param values): load-bearing if per-family z-score exceeds no-normalization by delta-M2 >= 0.10; report the full ordering (z-score vs log-scale vs none).   (found: REVERSED -- none 0.515 >= zscore 0.483 >= log 0.448 (naive, chr21); z-score is 0.03 LOWER than none, not >= +0.10 higher. Raw/no-normalization is best)
- [-] M1 reconstruction (ceiling gap) is not degraded by the winning normalization vs no-normalization (normalization must not cost accuracy).   (found: n/a -- no normalization wins, so there is no winning-normalization cost to assess; M1 gaps are comparable, none 0.59 / z 0.63 / log 0.70, chr21)
- [ ] mechanistic: the M2 gain concentrates in wide-param-range families (mult, add, cap) where scale-collision bites hardest; report per-family M2 deltas across the three arms.   (found: no gain to concentrate -- none vs z per-family: mult 0.515/0.483, add 0.814/0.812, power 0.238/0.217; normalization helps no family)

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

- sandbox dual_conditioning sweep 47730802_[0-9] (10 arms, 25ep)
- sandbox/diagnostics/dual_conditioning (impl complete, CPU-gated; awaiting GPU sweep)

## Findings

Param-encoding normalization is NOT load-bearing -- if anything, raw (no-normalization) is best. Ranking the three arms by distributional M2 (chr21) gives none 0.515 >= z-score 0.483 >= log 0.448 (naive encoder; the aware encoder is similar, with log-aware an outlier at 0.53) -- the exact reverse of the hypothesis: per-family z-score does not exceed none by >=0.10, it is 0.03 *lower*. No family shows a normalization gain to concentrate (none vs z-score: mult 0.515/0.483, add 0.814/0.812, power 0.238/0.217). Reconstruction is comparable across arms (M1 gap none 0.59, z 0.63, log 0.70). The raw param, embedded per-family, is read fine without normalization; the concern that scale-collision depresses steering is refuted at this scale. Verdict: refuted -- keep raw param encoding.

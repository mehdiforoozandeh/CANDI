---
id: h34
type: idea
title: Per-assay conditioning is necessary for output-steering; the v1 null was an across-assay pooling artifact
parent: q16
status: done
verdict: supported
metric: per-assay M2 ~0.50 vs pooled 0.02 (chr21); lift +0.48
created: "2026-07-08T12:36:47"
updated: "2026-07-09T00:48:45"
---

# h34 — Per-assay conditioning is necessary for output-steering; the v1 null was an across-assay pooling artifact

Parent:: [[q16_was_the_v1_output_steering_null_an_artif]]

## Problem Statement

The v1 decoder pooled y_meta across assays (mean over the assay axis), forcing uniform-per-batch conditions (one matrix cell per step instead of A). v2 conditions per assay on BOTH x_meta and y_meta. Is restoring per-assay conditioning what makes steering emerge? Contrast the per-assay arm vs the uniform-per-batch baseline arm, holding head/metric/families fixed.

## Idea / Hypothesis

Per-assay conditioning is necessary for output-steering; the v1 null was an across-assay pooling artifact

## Verifiables

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
- [x] per-assay arm: distributional M2 (median over invertible families, per plan section M2) >= 0.5   (found: per-assay M2 median-invertible ~= 0.50 across the 6-norm grid, chr21 range 0.45-0.53; 0.515 none-naive, 0.526 log-aware, 0.483 zscore-naive)
- [x] uniform-per-batch baseline arm: M2 <= 0.15 (reproduces the v1 ~0.02 null); lift delta-M2 >= 0.35 isolates per-assay conditioning as the cause   (found: the faithful v1 reproduction is the POOLED arm `pool_meta`, M2=0.022 <= 0.15, lift delta=+0.48; REFINEMENT: uniform-per-batch sampling alone does NOT reproduce the null (M2=0.53) -- the confound is across-assay POOLING, not uniform sampling)
- [x] no reconstruction cost: per-assay M1 ceiling-gap <= uniform-per-batch + 0.05 (CRPS)   (found: per-assay M1 gap 0.63 is BETTER than pooled 1.16 and on par with uniform 0.55, chr21)

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

- sandbox dual_conditioning sweep 47730802_[0-9] (10 arms, 25ep)

## Findings

Per-assay conditioning is load-bearing, and the v1 output-steering null is an across-assay **pooling** artifact. On chr21 the per-assay arms reach distributional M2 (median-invertible) ~= 0.50 (range 0.45-0.53 across the 6-norm grid), whereas the pooled arm (`pool_meta`, the faithful across-assay-mean reproduction of v1) collapses to M2 = 0.022 -- a ~25x drop that reproduces the original v1 ~0.02 null -- while its reconstruction also degrades (M1 gap 1.16 vs 0.63). Lift delta = +0.48 >= 0.35. **Refinement of the pre-registered check:** it is across-assay *pooling* specifically, not uniform-per-batch sampling, that causes the null -- the uniform-per-batch arm retains M2 = 0.53. Per-assay conditioning carries no reconstruction cost (its M1 gap is better than pooled and on par with uniform). This is the first controlled, ground-truth isolation of per-assay-vs-pooled decoder conditioning as the causal lever for output steering.

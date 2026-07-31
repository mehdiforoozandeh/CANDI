---
id: h37
type: idea
title: "Whole-chromosome background domination suppresses steering: the metadata signal lives in the sparse foreground the per-position loss under-weights"
parent: q16
status: done
verdict: partial
metric: fg-agg gap power +0.12 / add -0.35 (sign-specific, magnitudes <0.2)
created: "2026-07-08T12:36:49"
updated: "2026-07-09T00:49:00"
---

# h37 — Whole-chromosome background domination suppresses steering: the metadata signal lives in the sparse foreground the per-position loss under-weights

Parent:: [[q16_was_the_v1_output_steering_null_an_artif]]

## Problem Statement

Per-position NBNLL over whole chromosomes is background-dominated (most positions are low-count), which can make the model conservative and under-fit the foreground. For most families (cap/thin/power/mult) the h_y-driven output change is foreground-concentrated, so a background-tuned model would show low M2 for reasons unrelated to the conditioning pathway. add is the control (shifts uniformly, so its contrast is background-visible). THIS PHASE: diagnostic + specificity only (free slices on the 2a runs). NEXT PHASE (deferred): an interventional loss-reweighting arm AND a type2 (foreground/background-balanced) training-data arm from sandbox.h5. See also [[q12]]/[[h23]] (count R2 rank-vs-magnitude, loss reweighting).

## Idea / Hypothesis

Whole-chromosome background domination suppresses steering: the metadata signal lives in the sparse foreground the per-position loss under-weights

## Verifiables

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
- [ ] diagnostic (free, on 2a runs): M2 on foreground positions (top-k% by base count or peak-called) exceeds aggregate M2 by delta-M2 >= 0.2 -- steering is present but foreground-localized and masked in the aggregate   (found: NOT met -- fg-minus-agg gaps (chr21) are mult -0.02, add -0.35, power +0.12; only power is positive and it is below 0.2; steering is generally not hidden in the foreground)
- [x] specificity: the foreground-vs-aggregate M2 gap concentrates in foreground-signature families (cap, thin, power, mult) and is minimal for add (background-affecting) -- isolates the imbalance mechanism from a generic capacity effect   (found: sign pattern holds -- power (foreground-reshaping) is the only positive gap (+0.12) while add (background-affecting) has the largest deficit (-0.35); magnitudes < 0.2. cap/thin are 2c families, not run)

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

- sandbox dual_conditioning sweep 47730802_[0-9] (10 arms, 25ep)

## Findings

Background domination is a partial, family-specific effect, not a general mask on steering. The core diagnostic fails: foreground M2 does NOT broadly exceed aggregate M2 by >=0.2 -- the per-family fg-minus-agg gaps (chr21) are mult -0.02, add -0.35, power +0.12, so the steering signal is generally not hidden in the foreground (for add the aggregate is actually inflated by background). But the specificity pattern holds in sign: the one foreground-reshaping family tested, power, is the only one with a positive foreground gap (+0.12), while the background-affecting additive family (add) shows the largest foreground deficit (-0.35) -- matching the mechanism's direction even though magnitudes stay below the 0.2 bar. The non-invertible foreground-signature families (cap, thin) that would test this most sharply are phase-2c and were not run. Verdict: partial -- the imbalance shapes *where* per-family steering reads out, but does not broadly suppress it.

---
id: h9
type: idea
title: A depth-centered size-factor reparam restores depth sensitivity to DCR ~ 4
parent: q4
status: done
verdict: partial
metric: 
created: "2026-07-02T20:37:11"
updated: "2026-07-02T20:37:11"
---

# h9 — A depth-centered size-factor reparam restores depth sensitivity to DCR ~ 4

Parent:: [[q4_can_candi_s_counts_be_made_depth_control]]

## Problem Statement

_(why this is worth testing)_

## Idea / Hypothesis

A depth-centered size-factor reparam restores depth sensitivity to DCR ~ 4

## Verifiables

- [x] the depth-centered size factor mu = 2^[d-24]·exp[eta] gives DCR ~ 4.0 across overfit, assay-only mask, count+peak and 3-epoch training   (found: R15-R20, DCR 3.99-4.02 from epoch 0)
- [ ] the raw 2^d offset works   (found: raw offset FAILS at DCR ~ 1.0)
- [-] it reproduces at 35-assay MERGED production scale   (found: validated only on the 8-assay chr19 diagnostic — needs confirm)

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Findings

depth_center ~ batch-median log2 depth (~24 on EIC) is the fix; raw 2^d fails. Enables controllable denoising to a canonical depth — the strongest candidate new results subsection; production-scale confirmation still open.

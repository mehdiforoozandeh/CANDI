---
id: h39
type: idea
title: Type2 foreground/background-balanced training data improves held-out reconstruction and steering
parent: q17
status: idea
verdict:
metric:
created: 2026-07-09T19:58:15
updated: 2026-07-09T19:58:15
---

# h39 — Type2 foreground/background-balanced training data improves held-out reconstruction and steering

Parent:: [[q17_is_candi_training_biased_by_foreground_b]]

## Problem Statement

Replace natural (background-dominated) training sampling with type2 sampling from sandbox.h5 that balances foreground vs background windows, changing the training DISTRIBUTION rather than the per-position weight. Same best-config as q15 2a; baseline = the natural-distribution model. Critically, eval on the SAME fixed natural eval set as the baseline so the comparison is a generalization test, not a distribution-shift artifact. Complements h_a (loss reweight): if both lift foreground-recon+steering the imbalance-bias claim is robust to the mechanism.

## Idea / Hypothesis

Type2 foreground/background-balanced training data improves held-out reconstruction and steering

## Verifiables

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
- [ ] Foreground reconstruction improves: trained on type2-balanced data, foreground CRPS/Spearman (top-k% by base count) improves by a meaningful margin vs the natural-distribution baseline, both evaluated on the SAME fixed natural eval set
- [ ] Steering improves: per-assay M2 rises vs the natural-distribution baseline
- [ ] Aggregate guardrail: aggregate reconstruction on the natural eval set does not collapse relative to baseline (foreground gain is not bought by whole-genome degradation beyond tolerance)

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Findings

_(written by the PI/agent when the case is closed)_

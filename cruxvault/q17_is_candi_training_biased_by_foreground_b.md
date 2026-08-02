---
id: q17
type: question
title: Is CANDI training biased by foreground/background imbalance — does rebalancing effective foreground exposure improve held-out reconstruction and steering?
parent: root
status: open
stale: false
created: "2026-07-09T19:57:50"
updated: "2026-08-01T18:18:11"
---

# q17 — Is CANDI training biased by foreground/background imbalance — does rebalancing effective foreground exposure improve held-out reconstruction and steering?

Parent:: [[candi]]
Literature:: [[wiki/imbalance-aware-objectives]], [[wiki/peak-calling-and-signal-tracks]], [[wiki/imputation-evaluation-measures]]

## Question

Per-position NBNLL over whole chromosomes is background-dominated (most positions are low-count), so the model may under-fit the sparse foreground where the biologically and conditioning-relevant signal lives. This question asks whether that imbalance measurably biases CANDI's reconstruction (CRPS/NLL/Spearman/Pearson/R2/calibration) and steering (per-assay M2). Two independent interventions rebalance effective foreground exposure: loss reweighting (h_a) and type2 foreground/background-balanced training data (h_b). Both eval on the SAME fixed natural eval set. Headline readout = FOREGROUND reconstruction (top-k% by base count) + steering M2; AGGREGATE reconstruction is a guardrail that must not collapse (a flat-loss model is trained to minimize aggregate NBNLL over a background-dominated distribution, so aggregate metrics structurally favor it). We monitor foreground-only and background-only separately and expect a FG<->BG Pareto frontier; a visible frontier is itself evidence the imbalance is real, and the escape move is focal/residual weighting (upweight foreground only where the model underfits it) to push the frontier outward rather than slide along it. CANDI-general claim, only TESTED in the dual-conditioning testbed; see-also [[q15_can_candi_learn_dual_metadata_conditioni]] (testbed + h37 steering-only precursor), [[q12_in_the_v2_backbone_why_is_imputed_count_]] / [[h23_autoresearch_e32_lifts_imputed_count_r2_]] (count R2 rank-vs-magnitude + loss reweighting), [[q4_can_candi_s_counts_be_made_depth_control]] (depth-controllable supertrack). PARKED: created now, runs deferred until after q15's h31/h32 close.

## Answer so far

_(interpretation — written by the PI/agent; auto-flagged stale when new evidence lands)_

<!-- crux:ledger:start -->
**2 children** · ideas 0/2 done (supported 0, partial 0, refuted 0, inconclusive 0)

- `h38` [[h38_foreground_background_loss_reweighting_i|Foreground/background loss reweighting improves held-out reconstruction and steering]] — *idea*
- `h39` [[h39_type2_foreground_background_balanced_tra|Type2 foreground/background-balanced training data improves held-out reconstruction and steering]] — *idea*
<!-- crux:ledger:end -->

---
id: h50
type: idea
title: An explicit per-assay output factor (location+scale on eta, dispersion offset on n) absorbs the oracle per-assay scale error
parent: q20
status: idea
verdict:
metric:
created: 2026-07-24T11:10:46
updated: 2026-07-24T11:10:46
---

# h50 — An explicit per-assay output factor (location+scale on eta, dispersion offset on n) absorbs the oracle per-assay scale error

Parent:: [[q20_how_should_candi_architecture_and_traini]]

## Problem Statement

Merge of per-assay dispersion (H2), per-assay eta location bias (H4), and the Avocado structural factor. The q19 head is weight-shared Linear(16,16)+GELU+Linear(16,1) with a SINGLE scalar output bias, so the only per-assay knob is a rank-~2 FiLM where 8 assays with very different dynamic range/dispersion need ~7 dof. Macro CRPS is location/scale-dominated (S18) and the audit's oracle per-assay scale (S5) is literally one scale per assay; h46 showed the offset-off 'magnitude cost' is per-assay scale, not biology. Add metadata-INDEPENDENT per-assay eta scale+bias and a per-assay log-n offset (~24 params, no decay), indexed by slot not y_meta. Closes a fork-vs-production gap (production Linear(8,8) already carries per-assay bias).

## Idea / Hypothesis

An explicit per-assay output factor (location+scale on eta, dispersion offset on n) absorbs the oracle per-assay scale error

## Verifiables

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
- [ ] macro CRPS (oracle-scale decomposed) < wd0_on 1.341 with the gain attributable to the SCALE-error term (CRPS - CRPS_oracle_scaled shrinks), not shape; macro Sp >= 0.56
- [ ] per-assay ECE / dispersion-error term drops for the broad low-SNR marks (correctly relabeled per S6: H3K9me3/H3K27me3); ECE <= 0.053 overall
- [ ] fitted per-assay eta bias b_a ~ the oracle c*_a from H0 (the learned factor recovers what the oracle supplies), and dispersion offset recovers sharp>broad n ordering
- [ ] real-z metadata-ablation degradation >= wd0_on (added structural identity does not weaken metadata use); total depth slope ~1

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Findings

_(written by the PI/agent when the case is closed)_

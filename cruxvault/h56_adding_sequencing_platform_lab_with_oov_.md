---
id: h56
type: idea
title: Adding sequencing_platform + lab (with OOV/MISSING tokens) supplies the only NEW covariates that are identifiable on this slice (= h44, unblocked by h47)
parent: q20
status: idea
verdict:
metric:
created: 2026-07-24T11:12:17
updated: 2026-07-24T11:12:17
---

# h56 — Adding sequencing_platform + lab (with OOV/MISSING tokens) supplies the only NEW covariates that are identifiable on this slice (= h44, unblocked by h47)

Parent:: [[q20_how_should_candi_architecture_and_traini]]

## Problem Statement

On the T_ slice run_type is dead (H(run_type|assay,read_length)=0.000 bits, B1) but H(platform|assay,read_length)=0.443 bits and H(lab|assay,read_length)=0.212 bits -- platform and lab are LIVE, non-degenerate batch fields already present in eic_metadata.csv/merged_metadata.csv. No architecture can extract signal absent from the covariate vector; adding these two fields is the only way to give identifiable batch information beyond assay/depth/read_length, plausibly carrying protocol/coverage biases. A reserved OOV embedding index is a hard production requirement for meeting an unseen platform/lab. This IS h44, deferred until dual conditioning was shown to work -- which h47 established.

## Idea / Hypothesis

Adding sequencing_platform + lab (with OOV/MISSING tokens) supplies the only NEW covariates that are identifiable on this slice (= h44, unblocked by h47)

## Verifiables

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
- [ ] real-z metadata-ablation for platform (then lab): randomize the row on real z, require CRPS/Spearman degradation > 0 with clustered CI over 12 targets -- the IDENTIFIABLE analogue of the run_type test B1 forbids
- [ ] leave-one-covariate-out identifies the minimal sufficient metadata set; imp CRPS <= wd0_on 1.341 with any gain attributed via the oracle-scale decomposition (honest marginal floor)
- [ ] OOV round-trip: train without one platform, evaluate its data via the OOV token, require finite metrics and no significant degradation vs a seen platform (production robustness certificate)

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Findings

_(written by the PI/agent when the case is closed)_

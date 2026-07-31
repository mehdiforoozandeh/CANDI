---
id: h53
type: idea
title: Metadata-steered dispersion (read_length + imputation-context, NOT told-depth) is the clean PI-thesis test on a channel with no arithmetic shortcut
parent: q20
status: idea
verdict:
metric:
created: 2026-07-24T11:11:30
updated: 2026-07-24T11:11:30
---

# h53 — Metadata-steered dispersion (read_length + imputation-context, NOT told-depth) is the clean PI-thesis test on a channel with no arithmetic shortcut

Parent:: [[q20_how_should_candi_architecture_and_traini]]

## Problem Statement

The PI thesis needs a channel where using metadata cannot be shortcut by a hardwired-correct path. The MEAN has the depth offset (exact on DSF, B2) as a free competitor, which is why h47's revived mean-pathway steering stayed weak; and assay identity is redundant with encoder z (gap-fill). The DISPERSION n has NO such competitor. Add log n = raw_n + g(memb) with g a small MLP driven by read_length + a told-vs-observed / imputation-context flag. CRITICAL (from the B6 kill of the told-depth variant): EXCLUDE told-depth from the route -- thinning pins n flat on the DSF axis, and the measured n-vs-depth response is +14.5% (n RISES with depth = tighter), so feeding told-depth would re-learn a wrong sign. Pair with conditioning dropout ([[h54_conditioning_dropout_manufactures_the_mi|h54]]) to supply a training dose.

## Idea / Hypothesis

Metadata-steered dispersion (read_length + imputation-context, NOT told-depth) is the clean PI-thesis test on a channel with no arithmetic shortcut

## Verifiables

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
- [ ] REAL-z metadata-ablation on n (the instrument h47 V2 lacked): n RESPONDS to read_length/imputation-context (|d log n| above a min-effect gate) with correct sign (wider under imputation/OOD)
- [ ] FALSIFIABLE B2 constraint: n stays FLAT vs told-depth on the DSF sweep (|d log n / d told-depth| ~ 0) -- respecting thinning by design
- [ ] per-assay ECE down (< 0.053); macro CRPS not regressed (mean term unchanged, total depth slope ~1); sign-aware + min-effect gate, clustered CI over 12 targets

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Findings

_(written by the PI/agent when the case is closed)_

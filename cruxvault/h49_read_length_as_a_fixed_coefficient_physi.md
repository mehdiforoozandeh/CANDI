---
id: h49
type: idea
title: read_length as a fixed-coefficient physical exposure term completes the size-factor offset
parent: q20
status: idea
verdict:
metric:
created: 2026-07-24T11:10:46
updated: 2026-07-24T11:10:46
---

# h49 — read_length as a fixed-coefficient physical exposure term completes the size-factor offset

Parent:: [[q20_how_should_candi_architecture_and_traini]]

## Problem Statement

The NB head is a size-factor GLM (log2_mu=(depth-c)+eta) but its exposure term is INCOMPLETE: it counts reads, not read footprint. A length-R read at 25bp covers ~R/25+1 bins, a SECOND exposure factor log2(R/25+1) spanning 1.2 log2 over 30-101bp. Audit [RV]: read_length carries ~0.48-0.61 coef on log2 mean count and IS the 'excess depth slope' (once log2(rl) enters, depth returns to 0.975). Left today to a starved 1056-param FiLM that cannot extrapolate to the 7-9/12 read_length-OOD eval targets; an arithmetic offset can. Frees the FiLM to spend rank on assay identity.

## Idea / Hypothesis

read_length as a fixed-coefficient physical exposure term completes the size-factor offset

## Verifiables

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
- [ ] macro CRPS (oracle-scale decomposed, H0) < wd0_on 1.341, gain concentrated on the read_length-OOD targets (arithmetic extrapolation removes a systematic scale error there)
- [ ] total told-depth slope stays |slope-1|<=0.10 (read_length orthogonal); macro Sp >= 0.56; ECE <= 0.053
- [ ] real-z read_length metadata-ablation dCRPS now LARGE and correct-signed (vs ~0 today because read_length rode a starved path)
- [ ] B5-safe: coefficient FIXED at the physical value 1 (no attribution among collinear covariates claimed or fitted); optional 2nd arm learns a no-decay coeff and it converges to ~1

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Findings

_(written by the PI/agent when the case is closed)_

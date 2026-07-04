---
id: h12
type: idea
title: Explicit assay_id metadata (platform dropped) improves prompt-conditioning
parent: q5
status: done
verdict: supported
metric: 
created: "2026-07-02T20:37:11"
updated: "2026-07-02T20:37:11"
---

# h12 — Explicit assay_id metadata (platform dropped) improves prompt-conditioning

Parent:: [[q5_does_query_based_decoding_fix_the_fixed_]]

## Problem Statement

_(why this is worth testing)_

## Idea / Hypothesis

Explicit assay_id metadata (platform dropped) improves prompt-conditioning

## Verifiables

- [x] covariates are now depth+assay_id+read_length+run_type with an assay_id embedding, cloze/missing sentinel split and a MaskStem   (found: model.py MetadataEncoder/QueryMetadataEncoder)

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Findings

Small but real method delta versus the submitted Methods, motivated by the prompt-invariance failure.

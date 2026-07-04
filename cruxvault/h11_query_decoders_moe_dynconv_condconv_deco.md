---
id: h11
type: idea
title: Query decoders (MoE / DynConv / CondConv) decode only queried assays from an assay-id-keyed decoder
parent: q5
status: done
verdict: partial
metric: 
created: "2026-07-02T20:37:11"
updated: "2026-07-02T20:37:11"
---

# h11 — Query decoders (MoE / DynConv / CondConv) decode only queried assays from an assay-id-keyed decoder

Parent:: [[q5_does_query_based_decoding_fix_the_fixed_]]

## Problem Statement

With fixed output channels a decoder can learn "channel 5 = H3K4me3" and ignore the output prompt — the supertrack root cause.

## Idea / Hypothesis

Query decoders (MoE / DynConv / CondConv) decode only queried assays from an assay-id-keyed decoder

## Verifiables

- [x] the query-decoder family is implemented and smoke-trained, backward-compatible with the fixed decoder   (found: models/*QueryDecoder_CondConv5_Hybrid*, Mar 2026)
- [ ] query decoding beats the fixed decoder on imputation or controllability   (found: full parity benchmark not yet run, spec flags it as needed)

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Findings

The mechanism to break the fixed-channel shortcut is built and runs; whether it actually won over the fixed baseline is unconfirmed.

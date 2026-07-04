---
id: h18
type: idea
title: A single-shot decoder FiLM beats per-layer decoder FiLM
parent: q9
status: done
verdict: supported
metric: 
created: "2026-07-02T20:37:12"
updated: "2026-07-02T20:37:12"
---

# h18 — A single-shot decoder FiLM beats per-layer decoder FiLM

Parent:: [[q9_how_should_conditioning_film_and_metadat]]

## Problem Statement

_(why this is worth testing)_

## Idea / Hypothesis

A single-shot decoder FiLM beats per-layer decoder FiLM

## Verifiables

- [x] one latent FiLM E7 is the best multi-head run in the sweep and was promoted default   (found: F8; beats linear FiLM E6)

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Findings

A single FiLM that makes the decoder a pure spatial upsampler wins; per-layer decoder FiLM over-conditions.

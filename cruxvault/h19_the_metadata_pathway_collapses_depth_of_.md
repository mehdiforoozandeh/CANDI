---
id: h19
type: idea
title: The metadata pathway collapses depth-of-coverage (depth ignored)
parent: q9
status: done
verdict: partial
metric: 
created: "2026-07-02T20:37:12"
updated: "2026-07-02T20:37:12"
---

# h19 — The metadata pathway collapses depth-of-coverage (depth ignored)

Parent:: [[q9_how_should_conditioning_film_and_metadat]]

## Problem Statement

_(why this is worth testing)_

## Idea / Hypothesis

The metadata pathway collapses depth-of-coverage (depth ignored)

## Verifiables

- [x] depth-of-coverage is ignored by the model   (found: F1 metadata collapse, dcr ~ 1)
- [ ] this is fixed on the production stack   (found: only a partial fix on v2 via the depth-offset head; production B8 open)

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Findings

Depth collapse (Q5) is the sandbox mirror of the supertrack failure; the E29/E30 depth-offset head is the fix, validated on v2 but not yet at production scale.

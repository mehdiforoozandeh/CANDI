---
id: h21
type: idea
title: Encoder-only JEPA yields usable Stage-1 latents only when warm-started from the CANDI encoder
parent: q11
status: done
verdict: partial
metric: 
created: "2026-07-02T20:37:12"
updated: "2026-07-02T20:37:12"
---

# h21 — Encoder-only JEPA yields usable Stage-1 latents only when warm-started from the CANDI encoder

Parent:: [[q11_does_jepa_sigreg_latent_pretraining_beat]]

## Problem Statement

_(why this is worth testing)_

## Idea / Hypothesis

Encoder-only JEPA yields usable Stage-1 latents only when warm-started from the CANDI encoder

## Verifiables

- [x] lambda=0.5 with pred_hidden=16 prevents collapse and is the best config   (found: E19 accepted for Stage 1)
- [ ] a purpose-built fresh encoder recovers CANDI-encoder geometry   (found: fresh encoder is the root cause of blob UMAPs; all 22 E23 fresh runs fail the v2 geometry gate)

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Findings

Encoder-only JEPA works from a warm start; a fresh-from-scratch encoder collapses geometry.

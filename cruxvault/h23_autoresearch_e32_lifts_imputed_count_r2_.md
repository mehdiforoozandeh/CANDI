---
id: h23
type: idea
title: Autoresearch (E32) lifts imputed-count R2 above zero via an eval fix plus loss reweighting
parent: q12
status: done
verdict: partial
metric: 
created: "2026-07-02T20:37:12"
updated: "2026-07-02T20:37:12"
---

# h23 — Autoresearch (E32) lifts imputed-count R2 above zero via an eval fix plus loss reweighting

Parent:: [[q12_in_the_v2_backbone_why_is_imputed_count_]]

## Problem Statement

_(why this is worth testing)_

## Idea / Hypothesis

Autoresearch (E32) lifts imputed-count R2 above zero via an eval fix plus loss reweighting

## Verifiables

- [x] vb_natural eval with imp_weight ~ 0.59 raises imp R2 to 0.122   (found: E32 partial, denoising peaked ~0.31)
- [ ] imp R2 clears the 0.15 validate gate   (found: 0.122 below the gate; E33 full-data confirm peaked 0.162 at ep44 then collapsed)

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Findings

The rank-magnitude decoupling is real; AR moved imp R2 positive but below the gate, with late-epoch collapse.

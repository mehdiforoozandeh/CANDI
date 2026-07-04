---
id: h26
type: idea
title: Isolated single-thesis loops (menu-AR) beat base candi_v2 without monoculture collapse
parent: q13
status: done
verdict: partial
metric: 
created: "2026-07-02T20:37:13"
updated: "2026-07-02T20:37:13"
---

# h26 — Isolated single-thesis loops (menu-AR) beat base candi_v2 without monoculture collapse

Parent:: [[q13_what_architecture_changes_move_held_out_]]

## Problem Statement

_(why this is worth testing)_

## Idea / Hypothesis

Isolated single-thesis loops (menu-AR) beat base candi_v2 without monoculture collapse

## Verifiables

- [x] all 5 loops beat base -0.1261 by +0.024 to +0.085, crps_calibration best at -0.0408   (found: CRPS proper score plus LOO-ref deviation-correlation lifts count-Pearson 0.165 to 0.292)
- [ ] any loop beats the marginal average-reference baseline   (found: best Q_imp 0.450 < 0.4857, S_A still negative)
- [ ] cross-loop consolidation stacks additively   (found: FALSIFIED — GroupNorm anti-synergizes with CRPS, crps alone stays best)

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Findings

The universal winning mechanism is leak-free deviation-from-average-reference modeling plus calibration; the ceiling is the chr19-only training data, not architecture.

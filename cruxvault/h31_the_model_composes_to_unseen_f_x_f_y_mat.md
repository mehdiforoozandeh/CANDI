---
id: h31
type: idea
title: The model composes to unseen (f_x, f_y) matrix cells
parent: q15
status: staged
verdict: 
metric: 
created: "2026-07-03T01:59:54"
updated: "2026-07-03T02:51:37"
---

# h31 — The model composes to unseen (f_x, f_y) matrix cells

Parent:: [[q15_can_candi_learn_dual_metadata_conditioni]]

## Problem Statement

Hold out individual matrix cells whose f_x and f_y each still appear in training (in other cells), so every transform is seen on both sides but specific pairings are unseen. Does the model generalize to the held-out pairings because it factorized input-understanding from output-steering, rather than memorizing each cell? Example: seen thinned inputs and seen x2 outputs, but never the thin->x2 pairing; can it do thin->x2?

## Idea / Hypothesis

The model composes to unseen (f_x, f_y) matrix cells

## Verifiables

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
- [ ] v1 unseen pairings work about as well as seen ones: held-out-cell M1 within delta-R2 <= 0.10 of seen-cell M1 (nearly as accurate on combinations never trained on).
- [ ] v2 beats a wrong-but-memorized-output baseline: for a held-out cell like thin->x2, beat a baseline that understood the thinned input but applied an output transform it HAD seen paired with thinned inputs (the wrong one); proves the model reads h_y and applies the correct output transform on the novel pairing rather than falling back on a memorized pairing.
- [ ] v3 easy pairings must compose, map which do not: held-out invertible x invertible pairings (x-h/+h/power combos) pass v1's bar; tabulate per-family which pairings generalize and which break (expectation: non-invertible input transforms thin/cap are hardest to compose).

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

- sandbox/diagnostics/dual_conditioning (impl complete, CPU-gated; awaiting GPU sweep)

## Findings

_(written by the PI/agent when the case is closed)_

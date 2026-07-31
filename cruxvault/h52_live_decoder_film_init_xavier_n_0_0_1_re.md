---
id: h52
type: idea
title: Live decoder-FiLM init (xavier + N(0,0.1)) removes the second half of the annihilation mechanism h47 left in place
parent: q20
status: idea
verdict:
metric:
created: 2026-07-24T11:11:29
updated: 2026-07-29T02:09:50
---

# h52 — Live decoder-FiLM init (xavier + N(0,0.1)) removes the second half of the annihilation mechanism h47 left in place

Parent:: [[q20_how_should_candi_architecture_and_traini]]

## Problem Statement

The h47 pathology had TWO ingredients: coupled-L2 decay (h47 fixed via wd=0) AND adaLN-zero, which gives film_proj EXACTLY zero gradient at step 0 so conditioning starts as a bit-exact identity while the arithmetic offset already fits the mean. Even with decay removed, a from-zero FiLM grows against a trunk already satisfying the depth objective, staying under-expressed (wd0_on (gamma,beta) effrank ~2 of 32). Production DELIBERATELY abandoned near-identity decoder-FiLM init ('allowed the model to ignore prompts') for xavier+N(0,0.1), live at step 0. Re-init film_proj to the production style; keep wd=0. h48/F2 relocates the blocker: the assay signal survives the fusion MLP intact and is destroyed at the **fusion LayerNorm** (`wd0_on` pre-LN activation norm 396 vs 8.9; post-LN 0.0091 vs 1.41), with `film_proj` attenuating a further ~18×. A live FiLM init addresses the second annihilation ingredient but **not** the LN scale — pre-register which one this arm is actually testing.

## Idea / Hypothesis

Live decoder-FiLM init (xavier + N(0,0.1)) removes the second half of the annihilation mechanism h47 left in place

## Verifiables

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
- [ ] full-152-state real-z (gamma,beta) effrank strictly > wd0_on ~2 AND **SENTINEL-FREE** real->real cross-target assay-ablation max|d_eta| >= **[PASS BAR PENDING PI RATIFICATION — h48 declines to name a threshold; h47's own functional bar was >= 0.10, which sentinel-free wd0_on misses by 43x at 0.0023, so the sentinel-free 0.0023 is cleared trivially by any arm]** — the whole-row assay permute is RETIRED, it slid the MISSING(-1) sentinel across unavailable slots and is what produced the withdrawn 0.833; real-z metadata-ablation degradation > wd0_on (more load-bearing). STRETCH REFERENCE, not a pass bar: the offset-OFF arms already reach 4.1772 (offoff) / 9.7144 (wd0_off) on this probe, so >= 4.1772 is what "solved" looks like. Report alongside: `assay_id == slot index` on this data, so this bounds the PROMPT pathway only.
- [ ] macro CRPS (oracle-scale decomposed) <= wd0_on capability 1.3077 - 0.09 (the target-clustered noise floor), macro Sp >= 0.5653, ECE <= 0.0533, on >=2 seeds (RNG-reinit breaks bit-comparability; compare on metrics)
- [ ] per-epoch per-field absmax + (gamma,beta) effrank logging confirms the pathway climbs from step 1 rather than crawling from zero

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Findings

_(written by the PI/agent when the case is closed)_

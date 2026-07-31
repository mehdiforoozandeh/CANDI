---
id: h55
type: idea
title: A grouped decoder trunk (groups=A), optionally with per-deconv per-assay FiLM, gives the revived conditioning genuine per-assay channels
parent: q20
status: idea
verdict:
metric:
created: 2026-07-24T11:12:17
updated: 2026-07-29T02:09:50
---

# h55 — A grouped decoder trunk (groups=A), optionally with per-deconv per-assay FiLM, gives the revived conditioning genuine per-assay channels

Parent:: [[q20_how_should_candi_architecture_and_traini]]

## Problem Statement

The deconv trunk is ungrouped (grouped=False), so feat.view(B,L,A,C) is an ARBITRARY partition of a fully-mixed 128-vector (gap-fill verified), and the trunk builds the entire 768-bin profile metadata-blind before the single post-trunk FiLM rescales it. Assay-specific SHAPE (peak width, background, dynamic range) forms inside the deconv and is unrecoverable by a position-constant affine. Set groups=num_assays=8 so each assay has independent deconv channels and the head/FiLM read genuinely per-assay features; optional 2nd arm adds per-assay (de-pooled) FiLM at every deconv layer (the A2-carveout: per-assay per-deconv is UNTESTED; do NOT cite the -0.788 pooled result). CAVEAT: the built-in grouped=True is groups=signal_dim=128 (wrong granularity) -- needs plumbing to groups=A. RISK: removes cross-assay strength-borrowing (Avocado/eDICE) -> hybrid fallback + pre-committed falsifier.

## Idea / Hypothesis

A grouped decoder trunk (groups=A), optionally with per-deconv per-assay FiLM, gives the revived conditioning genuine per-assay channels

## Verifiables

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
- [ ] macro CRPS (oracle-scale decomposed) <= wd0_on capability 1.3077 - 0.09 with the SHAPE-after-descale term improved; macro Sp >= ~~0.56~~ **0.5653** (aligned 2026-07-28 to q20's arbiter constraint, H48:L269-270) (this is the SHAPE arm)
- [ ] **SENTINEL-FREE** real->real cross-target assay-ablation max|d_eta| >= **[PASS BAR PENDING PI RATIFICATION — h48 declines to name a threshold (H48:L274-277 addresses h52/h55 jointly and names none); h47's own functional bar was >= 0.10, which sentinel-free wd0_on misses by 43x at 0.0023, so the sentinel-free 0.0023 is cleared trivially by any arm]** AND per-layer (gamma,beta)/feature effrank RISE above wd0_on ~2 (grouping supplies per-assay channels the FiLM can act on). The whole-row assay permute is RETIRED (MISSING(-1) sentinel contamination; it produced the withdrawn 0.833). STRETCH REFERENCE, not a pass bar: offset-OFF already reaches 4.1772 / 9.7144. CAUTION: embedding-table geometry is **anti-diagnostic** as a steering proxy — `offoff`, the strongest steerer, has mean pairwise ||.|| 0.67 and effective rank 1.79, ~10x SMALLER than wd0_on's 7.0 — so the effrank clause is a per-layer FEATURE claim, not an embedding-separation claim. Report alongside: `assay_id == slot index` on this data, so this bounds the PROMPT pathway only.
- [ ] PRE-COMMITTED FALSIFIER: if ~~imp Spearman drops below 0.56~~ **macro Sp drops below 0.5653** (aligned 2026-07-28 to q20's arbiter constraint, H48:L269-270 — same shape metric and threshold as V1 and as h52), cross-assay-sharing loss dominates -> pure grouping rejected in favour of the shared+grouped hybrid arm; ECE <= 0.0533

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Findings

_(written by the PI/agent when the case is closed)_

---
id: h54
type: idea
title: Conditioning dropout manufactures the missing 'use the metadata' gradient and yields a free inference-time guidance dial
parent: q20
status: idea
verdict:
metric:
created: 2026-07-24T11:11:31
updated: 2026-07-24T11:11:31
---

# h54 — Conditioning dropout manufactures the missing 'use the metadata' gradient and yields a free inference-time guidance dial

Parent:: [[q20_how_should_candi_architecture_and_traini]]

## Problem Statement

S15, untouched by h47: training always feeds the honest y_meta, so the loss is minimized identically whether the decoder reads y_meta or reconstructs from z. h47 revived the WEIGHTS but nothing manufactures a task GRADIENT rewarding their use. Randomly replace y_meta rows with the MISSING sentinel (p~0.15) on assay_id + read_length (KEEP depth out: B2; KEEP run_type out of the success bar: B1) so a subset of steps withholds the covariate -- a contrastive dose. Simultaneously trains the depth_missing/readlen_missing sentinels (denormal/at-init in BOTH arms today) and yields a classifier-free-guidance dial log2_mu = uncond + w*(cond-uncond) that can sharpen magnitude/calibration at inference with zero retrain. Zero new params.

## Idea / Hypothesis

Conditioning dropout manufactures the missing 'use the metadata' gradient and yields a free inference-time guidance dial

## Verifiables

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
- [ ] real-z metadata-ablation degradation (honest vs fully-MISSING y_meta) strictly > wd0_on (strictly more load-bearing) -- the axis h47 left weak
- [ ] missing-sentinel revival: decoder depth_missing_emb/readlen_missing_emb absmax from denormal/init -> >= 1e-2 (clean falsifiable side-prediction)
- [ ] CFG-w validation-CRPS curve has a minimum at w>=1 and macro CRPS at tuned w <= 1.341; macro Sp >= 0.56, ECE <= 0.053; NOTE verifiables reframed off S1-errors (no eta_slope, clustered CIs, real foreground)

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Findings

_(written by the PI/agent when the case is closed)_

---
id: h51
type: idea
title: A no-decay parameter group (decoupled AdamW, trunk-only weight decay) is the production-safe equivalent of h47's global wd=0
parent: q20
status: idea
verdict:
metric:
created: 2026-07-24T11:11:28
updated: 2026-07-24T11:11:28
---

# h51 — A no-decay parameter group (decoupled AdamW, trunk-only weight decay) is the production-safe equivalent of h47's global wd=0

Parent:: [[q20_how_should_candi_architecture_and_traini]]

## Problem Statement

h47's global wd=0 also disables decay on the 91%-of-params trunk, forfeiting regularization on the chr19-only, overfit-prone slice. The annihilation is specifically COUPLED-L2 x zero-task-gradient; decoupled AdamW's multiplicative shrink is trivially overpowered by any persistent gradient. Put the conditioning pathway (all *_embedding, film_proj, LayerNorm affines, biases) in a wd=0 group (identical protection to global wd=0) and keep decay on the trunk (the DiT/GPT-3/LLaMA split q19, the golden testbed, and production all omit). HONEST: on the conditioning pathway this is behaviourally identical to wd0_on, so it CANNOT out-steer h47; its value is generalization CRPS + production portability.

## Idea / Hypothesis

A no-decay parameter group (decoupled AdamW, trunk-only weight decay) is the production-safe equivalent of h47's global wd=0

## Verifiables

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
- [ ] conditioning-pathway weights absmax and real-z effrank MATCH wd0_on (identical zero decay there); a steering null vs wd0_on is the CORRECT prediction, not a failure
- [ ] macro CRPS (oracle-scale decomposed) within noise of, or better than, wd0_on 1.341 on >=3 seeds (effect is inside the ~0.12 CRPS seed floor, so seed replication is load-bearing); macro Sp >= 0.56; ECE <= 0.053
- [ ] clean param partition verified (name-filter puts embeddings/film_proj/norms/biases in wd=0 group, trunk+heads in decay group); ports to candi_v2 train.py unchanged

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Findings

_(written by the PI/agent when the case is closed)_

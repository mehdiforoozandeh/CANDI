---
id: h8
type: idea
title: The v1 NB count head is depth-controllable via the output prompt
parent: q4
status: done
verdict: refuted
metric: 
created: "2026-07-02T20:37:11"
updated: "2026-07-02T20:37:11"
---

# h8 — The v1 NB count head is depth-controllable via the output prompt

Parent:: [[q4_can_candi_s_counts_be_made_depth_control]]

## Problem Statement

Asking for a higher-depth supertrack did nothing because the count mean carried no depth dependence.

## Idea / Hypothesis

The v1 NB count head is depth-controllable via the output prompt

## Verifiables

- [ ] output count ratio responds to a +2 log2 depth prompt   (found: DCR ~ 1.0 on the default head — invariant to y_meta, depth collapse)

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Findings

Root cause of the supertrack failure, diagnosed honestly as a negative result — the NB mean was depth-blind.

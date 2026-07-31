---
id: h44
type: idea
title: Adding sequencing_platform + lab (with an UNKNOWN/OOV token) and pruning to the optimal metadata set improves imputation and denoising
parent: q19
status: done
verdict: refuted
metric: "Superseded (not disproven): relocated to h56 under q20; platform/lab identifiable (0.443/0.212 bits) unlike run_type (0 bits); to be tested vs wd0_on 1.341 with corrected instruments"
created: "2026-07-15T23:35:48"
updated: "2026-07-24T13:20:00"
---

# h44 — Adding sequencing_platform + lab (with an UNKNOWN/OOV token) and pruning to the optimal metadata set improves imputation and denoising

Parent:: [[q19_can_we_make_dual_conditioning_work_on_re]]

> **SUBSUMED (2026-07-24)** by [[h56_adding_sequencing_platform_lab_with_oov_|h56]] under [[q20_how_should_candi_architecture_and_traini|q20]] — h47 unblocked the dual-conditioning precondition h44 was deferred behind.

## Problem Statement

DEFERRED / banked -- do NOT run until dual conditioning is shown to work (h40-h43). CANDI currently conditions on 3 covariates besides assay_id (depth, read_length, run_type) but the data also carry sequencing_platform and lab, unused. Hypothesis: adding them (with a mandatory UNKNOWN/OOV field for platforms/labs unseen in pretraining) and pruning non-contributing fields yields the optimal metadata set and better imputation/denoising.

## Idea / Hypothesis

Adding sequencing_platform + lab (with an UNKNOWN/OOV token) and pruning to the optimal metadata set improves imputation and denoising

## Verifiables

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
- [ ] adding sequencing_platform + lab (with an UNKNOWN/OOV token) improves imp-count Spearman/CRPS by >= epsilon vs the 3-covariate model — NOT EVALUATED HERE (superseded): relocated to [[h56_adding_sequencing_platform_lab_with_oov_|h56]] under [[q20_how_should_candi_architecture_and_traini|q20]], to be tested with the corrected instruments (H0) and against the wd0_on 1.341 baseline.
- [ ] per-covariate ablation identifies the minimal sufficient metadata set (non-contributing fields pruned with no loss) — relocated to [[h56_adding_sequencing_platform_lab_with_oov_|h56]] (leave-one-covariate-out grid).
- [ ] OOV token: a held-out platform/lab at test time imputes with no significant degradation vs a seen platform/lab — relocated to [[h56_adding_sequencing_platform_lab_with_oov_|h56]] (OOV round-trip).

## Planned Intervention

**DEFERRED — do NOT run until h40–h43 give a positive dual-conditioning result.** Banked so the idea isn't lost; revisit before the production metadata design.

When run:
- Extend the metadata embedder to add **`sequencing_platform` + `lab`** (both present in `data/eic_metadata.csv`), each with a reserved **UNKNOWN / OOV embedding index** (alongside the existing `-1` missing / `-2` cloze specials) so a platform/lab unseen in pretraining is representable — a hard requirement for a pretrained CANDI meeting new labs/platforms.
- **Ablation grid**: {3-cov base} vs {+platform} vs {+lab} vs {+both}; plus **leave-one-covariate-out** over the full set to identify the **minimal sufficient** metadata set (prune non-contributing fields, add the ones that help).
- **OOV test**: hold out one platform (or lab) from training; at test, impute its data using the OOV token; require **no significant degradation** vs a seen platform/lab.
- **Metric**: imp-count Spearman / CRPS (M1) as primary; confirm no harm to M2/M3.

Rationale: CANDI currently conditions on 3 covariates besides assay_id; platform + lab are batch-defining fields likely to carry imputation-relevant signal, and the OOV mechanism is what makes the pretrained model robust to distribution shift at deployment.

**Tests (pre-GPU, written when h44 is picked up — deferred):** extend `RealMetaEmbedder` tests for the two new fields with sentinel + **OOV-token** handling (an unseen platform/lab id maps to the reserved OOV index, forward stays finite); an ablation-harness test that the leave-one-covariate-out grid runs and records per-covariate deltas; an OOV round-trip test (train without platform X, evaluate its data via the OOV token, assert no crash / finite metrics). Same gate discipline as h40–h43 before any GPU run.

## Run Links

_(none yet)_

## Findings

**REFUTED = SUPERSEDED (not empirically disproven).** h44 was banked pending a positive dual-conditioning result, which [[h47_the_offset_on_steering_null_is_a_weight_|h47]] delivered. Its content — add sequencing_platform + lab (which ARE identifiable on this slice: H(platform|assay,read_length)=0.443 bits, H(lab|...)=0.212 bits, vs run_type's 0.000) with a reserved OOV token — is relocated verbatim to [[h56_adding_sequencing_platform_lab_with_oov_|h56]] under [[q20_how_should_candi_architecture_and_traini|q20]], where it will be tested against the wd0_on 1.341 baseline with the corrected instruments (H0/[[h48_h0_fix_the_broken_q19_instruments_and_re|h48]]). This node is closed refuted only to retire it from q19's open set; the hypothesis itself is live in h56.

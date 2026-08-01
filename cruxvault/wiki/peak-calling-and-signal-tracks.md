---
type: wiki
title: Peak calling and signal tracks
summary: MACS/MACS2 and the local-Poisson model behind the −log10 p-value tracks that essentially all epigenome imputation methods consume as their target.
category: method
sources: raw/zhang-2008-macs.xml, raw/schreiber-2023-encode-imputation-challenge.pdf, raw/landt-2012-chip-seq-guidelines.xml, raw/amemiya-2019-encode-blacklist.xml
created: 2026-07-31T21:26:00
updated: 2026-07-31T21:26:00
---

# Peak calling and signal tracks

Understanding MACS's local-λ model matters beyond peak calling, because the −log10 p-value track it emits — not read counts, not fold enrichment — is the quantity nearly every imputation method has been trained to predict.

## MACS

`raw/zhang-2008-macs.xml` (Model-based Analysis of ChIP-Seq) has two defining features:

1. **Empirical modelling of the fragment shift *d***. ChIP-seq tags mark the ends of fragments and appear offset in the 3′ direction; MACS estimates *d* from the bimodal plus/minus strand pattern around enriched regions and shifts tags by *d*/2 to approximate the true protein–DNA interaction site.
2. **A dynamic local λ**. Rather than one genome-wide Poisson rate, MACS estimates the background rate *locally* from the control experiment, which captures regional biases (copy number, chromatin accessibility, sonication non-uniformity) that a global rate would miss. With a control, MACS first linearly scales total control tags to match total ChIP tags. See [[chip-seq-assay-and-controls]].

MACS also caps redundant tags at the number expected under a random genome-wide distribution, since excess duplicates come from amplification bias — the same concern handled by deduplication in [[read-processing-and-artifact-regions]].

## What the ENCODE pipelines emit

`raw/schreiber-2023-encode-imputation-challenge.pdf` documents the exact convention used for the challenge, and it is the convention the field inherited:

- **ATAC/DNase**: MACSv2 applied to smoothed counts of read **starts** (5′ ends) in a 150 bp smoothing window at each position, relative to the expected number of reads from a **local Poisson-simulated background**.
- **Histone ChIP-seq**: fold enrichment and statistical significance of counts of **extended** reads (extended 5′→3′ by the predominant fragment length), relative to extended reads from the control, with the local Poisson null parameterised from the control.
- Both filter peaks overlapping the ENCODE exclusion list (`raw/amemiya-2019-encode-blacklist.xml`).
- The distributed product is a genome-wide track of the **−log10 p-value of enrichment at each base pair**, later binned to 25 bp.

The paper is explicit about the choice: read counts and fold change were both available, but −log10 p-value was chosen "to be consistent with previous imputation literature." So the field's default target is a *significance* statistic, not a physical measurement — a p-value already folds in depth, background, and the control.

## Binarisation and peaks as an evaluation target

For evaluation, `raw/schreiber-2023-encode-imputation-challenge.pdf` binarises imputations at `Y ≥ 2` (p = 0.01) and uses **MACS2 peak calls** as the binarised experimental truth; its peak-correlation measure restricts DNase correlation to peak regions precisely because peaks — unlike promoters — are cell-type-specific. See [[imputation-evaluation-measures]].

`raw/landt-2012-chip-seq-guidelines.xml` notes several peak callers were used across ENCODE (SPP, PeakSeq, MACS), so "the peak set" is itself pipeline-dependent.

## See also

Related:: [[chip-seq-assay-and-controls]], [[count-distributions-for-sequencing-data]], [[imputation-evaluation-measures]], [[read-processing-and-artifact-regions]], [[signal-normalization-in-epigenomics]]

---
type: wiki
title: Peak calling and signal tracks
summary: MACS/MACS2 and the local-Poisson model behind the −log10 p-value tracks that essentially all epigenome imputation methods consume as their target.
category: method
sources: raw/zhang-2008-macs.xml, raw/schreiber-2023-encode-imputation-challenge.pdf, raw/landt-2012-chip-seq-guidelines.xml, raw/amemiya-2019-encode-blacklist.xml, raw/li-2011-idr.pdf, raw/pampari-2024-chrombpnet.pdf, raw/barbadilla-martinez-2025.pdf
created: 2026-07-31T21:26:00
updated: 2026-08-01T00:51:15
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

## Reproducibility as the peak-calling criterion

`raw/li-2011-idr.pdf` (IDR — irreproducible discovery rate) supplies the statistical basis for how ENCODE actually decides which peaks are real. Rather than thresholding one replicate's significance, it measures the **reproducibility of ranked findings across replicate experiments**. Its central object is not a scalar but a **curve** — the correspondence-at-the-top curve — which plots how the overlap between two replicates' top-ranked findings evolves as one descends the ranking, and so quantitatively identifies the point at which findings stop being consistent across replicates.

Two uses follow. As a **peak-calling criterion**, IDR selects the threshold where reproducibility degrades, which is how ENCODE's conservative and optimistic IDR peak sets are produced — meaning narrowPeak labels are reproducibility-derived, not significance-derived. As an **evaluation instrument**, the correspondence curve compares any two rankings of the same loci, including a predicted ranking against an observed one.

## Modelling the assay's own bias

`raw/pampari-2024-chrombpnet.pdf` (ChromBPNet) treats the enzyme's sequence preference as a component to be **modelled and factored out** rather than normalised away. Tn5 (ATAC) and DNase have intrinsic sequence biases that appear in base-resolution accessibility profiles; ChromBPNet learns a bias model and deconvolves it from the regulatory sequence determinants, recovering compact TF motif lexicons and precision footprints that are otherwise contaminated. The paper reports pervasive Tn5 bias motifs in **profile** contribution scores but not in **count** contribution scores — i.e. the bias distorts the shape of the signal more than its total.

Two transferable points: the assay's technical signature is **structured and learnable**, not noise; and ChromBPNet's models hold **across sequencing depths**, so a bias model fitted once transfers between experiments of different depth.

## What a peak call is evidence of

`raw/barbadilla-martinez-2025.pdf` is worth reading against the pipeline described above,
because it questions what the output actually measures. Maps of open chromatin and histone
modifications are **intrinsically correlative** and "do not offer direct measurements of
causal regulatory activity." Three specific cautions follow:

- The genome is partitioned into **large domains of autocorrelated histone modifications**,
  so a peak's boundaries do not cleanly localise the causally relevant sequence.
- **~15–50% of regions marked by open chromatin are not detectably active as enhancers** in
  reporter assays — accessibility is a necessary-ish but far from sufficient signal.
- Curated lists of regulatory elements derived from these features are used extensively as
  training and evaluation targets, yet the vast majority have **never been experimentally
  validated**, and it is likely only a subset has a regulatory role.

There is also a causal-direction problem that no peak caller resolves: sequence and
accessibility dictate TF binding, some TFs open chromatin or recruit histone-modifying
enzymes, and transcription is controlled by these features but **also alters chromatin state
in return**. The features a peak call summarises are mutually causal, so a peak is a marker
of a regulatory process rather than a measurement of one.

None of this argues against using peak calls as targets — it argues for treating agreement
with a narrowPeak set as agreement with a **noisy, partly unvalidated proxy**, and for
pairing it with an orthogonal functional readout where one exists.

## See also

Related:: [[chip-seq-assay-and-controls]], [[count-distributions-for-sequencing-data]], [[imputation-evaluation-measures]], [[read-processing-and-artifact-regions]], [[signal-normalization-in-epigenomics]]

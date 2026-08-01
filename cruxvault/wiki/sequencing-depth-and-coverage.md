---
type: wiki
title: Sequencing depth and coverage
summary: How many reads an epigenomic experiment needs, why the answer differs by mark, and why depth is a first-class experimental covariate rather than a nuisance to be scaled away.
category: concept
sources: raw/jung-2014-sequencing-depth-chip-seq.xml, raw/landt-2012-chip-seq-guidelines.xml, raw/schreiber-2023-encode-imputation-challenge.pdf, raw/xiang-2020-s3norm.xml, raw/karimzadeh-2018-umap-mappability.xml
created: 2026-07-31T21:26:00
updated: 2026-07-31T23:27:28
---

# Sequencing depth and coverage

Depth is the covariate most often assumed away by a scale factor and least well handled by one, because its effect on the signal is mark-dependent and saturating rather than linear.

## How much depth is enough

`raw/jung-2014-sequencing-depth-chip-seq.xml` proposes an operational definition: **sufficient depth is the point at which detected enriched regions increase by <1% per additional million reads**. Evaluated on deeply sequenced H3K4me3, H3K36me3, H3K27me3 and H3K9me2/me3 datasets in human and fly:

- Fly typically saturates below **20 million reads**.
- Human datasets showed **no clear saturation point**; the practical recommendation is **40–50 million reads** for most marks.
- The requirement depends on the nature of the mark and the state of the cell. Point-source factors (TFs) produce sharp peaks and are comparatively easy; broad marks such as H3K27me3 and H3K9me3 spread over large domains and need far more depth. Broad repressive marks are the binding case.

The paper also derives a model of genomic coverage as a function of depth by assuming tag positions are Poisson-distributed along the genome — see [[count-distributions-for-sequencing-data]].

## Depth in practice

`raw/schreiber-2023-encode-imputation-challenge.pdf` normalised depth across the challenge's datasets by **subsampling each to a maximum of 50 million reads** (after excluding mitochondrial reads), citing this as consistent with ChIP-seq best practice — i.e. the 40–50M figure above. Note what this implies: a compendium's tracks were generated at heterogeneous depths, and harmonising them requires either downsampling to the minimum (discarding information) or modelling depth explicitly.

`raw/landt-2012-chip-seq-guidelines.xml` ties depth to **library complexity**: sequencing more deeply from a low-complexity library adds duplicates rather than information, which is why deduplication and library-complexity metrics accompany depth in QC. See [[read-processing-and-artifact-regions]].

## Depth is not a pure scale factor

`raw/xiang-2020-s3norm.xml` separates depth from **signal-to-noise ratio** and shows they must be normalised together: rescaling for depth alone leaves experiments differing in what fraction of reads fall in peaks. Two experiments at identical depth can therefore have very different usable signal. This is the argument for treating depth as a covariate to condition on rather than a constant to divide out — see [[signal-normalization-in-epigenomics]] and [[film-conditioning]].

## Read length changes which positions exist

`raw/karimzadeh-2018-umap-mappability.xml` (Umap and Bismap) makes precise a property that a scalar depth covariate cannot capture. Every region of a genome assembly has a **mappability**: the extent to which it can be uniquely mapped by sequence reads. In regions of lower mappability, estimates from any sequencing assay are less reliable, and those regions are more susceptible to spurious mapping from reads originating elsewhere that carry sequencing errors or unexpected genetic variation.

The load-bearing detail is that mappability is defined **per read length**. Umap computes, for a given read length *k*, all *k*-mers of the genome and identifies which positions are uniquely mappable at that length. Longer reads are uniquely mappable at more positions; shorter reads lose positions to repeats.

Two consequences:

- **Read length is not just a scalar exposure term.** Changing read length changes *which positions can be observed at all*, so its effect on the expected count is position-dependent, not a uniform multiplicative factor. A model treating read length as a global coefficient will be wrong specifically in low-mappability regions — which are also where the [[read-processing-and-artifact-regions]] exclusion list concentrates.
- **Mappability is computable in advance.** Unlike most nuisance variables it is a deterministic function of the reference and the read length, so it can be supplied as a per-position feature rather than inferred.

Bismap extends the same computation to bisulfite-converted genomes, where C→T conversion collapses sequence diversity and worsens the problem.

## See also

Related:: [[chip-seq-assay-and-controls]], [[signal-normalization-in-epigenomics]], [[count-distributions-for-sequencing-data]], [[distributional-shift-and-batch-effects]]

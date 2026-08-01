---
type: wiki
title: Distributional shift and batch effects
summary: Processing and protocol differences across a compendium shift the signal distribution site-specifically; the ENCODE challenge's deduplication artefact is the canonical worked example.
category: concept
sources: raw/schreiber-2023-encode-imputation-challenge.pdf, raw/teng-2021-chip-seq-batch-effects.xml, raw/bonhoure-2014-chip-seq-spiking.xml, raw/reske-2020-atac-seq-normalization.xml, raw/mckenna-2010-gatk.pdf
created: 2026-07-31T21:26:00
updated: 2026-07-31T21:26:00
---

# Distributional shift and batch effects

The recurring finding across these sources is that batch effects in sequencing assays are **not** a global scale factor — they vary across genomic sites, which is exactly why rescaling cannot remove them.

## The deduplication artefact

`raw/schreiber-2023-encode-imputation-challenge.pdf` traces the challenge's invalidating shift to the deduplication step of the ENCODE pipelines. Duplicates are marked with Picard MarkDuplicates (`raw/mckenna-2010-gatk.pdf` supplies the surrounding toolkit) and removed; for **single-end** datasets a single read is chosen from each duplicate set, whereas for **paired-end** datasets a read-pair is kept if *either* read in the pair is unique. Because the training data was predominantly older single-end experiments and the prospectively collected test data was paired-end, the two ended up with materially different signal distributions.

The authors' initial hypothesis — that paired-end data is simply higher quality — was wrong. The shift was a processing artefact, and it was large enough that the [[average-activity-baseline]] beat all but two of 23 submissions until it was corrected. See [[read-processing-and-artifact-regions]].

## Site-specific variability

`raw/teng-2021-chip-seq-batch-effects.xml` quantifies this directly on 211 CTCF ChIP-seq samples spanning 90 cell types across **three laboratories**. Fitting a mixed-effects model with fixed effects for cell condition and random effects for laboratory and replicate, it finds that both the **laboratory batch effect** and the **biological-replicate variability** differ across genomic sites. The consequence stated in the abstract is that observed differences between conditions must be assessed statistically against a site-aware noise distribution, not by direct comparison of normalised values.

`raw/reske-2020-atac-seq-normalization.xml` shows the same structure in ATAC-seq: the choice of normalisation method changes which regions are called differentially accessible, most severely when a **global** chromatin change is present — precisely the case a scale factor is supposed to handle and does not.

## Why spike-ins exist

`raw/bonhoure-2014-chip-seq-spiking.xml` argues the problem is unfixable post hoc in one important case. Normalisation methods that include a quantile step behave well when occupancy changes at a *subset* of sites, but **miss uniform genome-wide increases or decreases**, since a uniform shift is indistinguishable from a scaling artefact once the data are on the analyst's desk. Their spike adjustment procedure (SAP) therefore intervenes **experimentally**: a constant, low amount of chromatin from a foreign genome (human into mouse) is added before immunoprecipitation and serves as an internal reference. The spike also doubles as a QC signal, since its quality reflects technical rather than biological variation.

The general lesson for modelling: if the covariates that generate the shift (depth, read length, run type, platform, lab, pipeline version) are recorded, they can be conditioned on; if they are not, the shift is confounded with biology. See [[signal-normalization-in-epigenomics]].

## See also

Related:: [[encode-imputation-challenge]], [[signal-normalization-in-epigenomics]], [[quantile-normalization]], [[read-processing-and-artifact-regions]], [[chip-seq-assay-and-controls]], [[sequencing-depth-and-coverage]]

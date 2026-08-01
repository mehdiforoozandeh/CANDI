---
type: wiki
title: Imputation evaluation measures
summary: Why MSE and global correlation are assay-dependent and mutually redundant, and the challenge's replacement measures partitioned by signal strength, cell-type specificity, promoters, and peaks.
category: concept
sources: raw/schreiber-2023-encode-imputation-challenge.pdf, raw/harrow-2012-gencode.xml, raw/zhang-2008-macs.xml
created: 2026-07-31T21:26:00
updated: 2026-07-31T21:26:00
---

# Imputation evaluation measures

The challenge's measure critique reduces to one structural insight: a measure computed over the whole genome at once is dominated by background, and background is both the easiest part to predict and the part where prediction matters least.

## What fails

`raw/schreiber-2023-encode-imputation-challenge.pdf` reports that performance depends heavily on the imputed assay — most models showed **four orders of magnitude higher MSE on H3K4me3 than on H3K9me3**, purely because the marks differ in dynamic range and punctateness. Aggregating such measures across assays is therefore close to meaningless without stratification.

The redundancy argument is sharper: scale-based measures that are appropriate when predictions and targets share a scale become **increasingly redundant with one another as scale differences increase**. So a battery of measures designed without controlling for [[distributional-shift-and-batch-effects]] collapses to fewer effective dimensions than its designers intended, giving false confidence that many aspects of quality have been checked.

## The replacement measures

All are computed by **partitioning the genome and then applying standard metrics per partition**, averaging over partitions so each partition is weighted equally rather than each locus (`raw/schreiber-2023-encode-imputation-challenge.pdf`). Binarised versions are defined as `Yᵇ = Yᶜ ≥ 2` (a signal p-value of 0.01) for imputations, and MACS2 peak membership for experimental data (`raw/zhang-2008-macs.xml`).

1. **Partition by signal strength.** Bin the experimental signal into logarithmic bins of size 0.1 from 10⁻¹ to 10^2.5; compute accuracy of binarised prediction vs binarised truth within each bin. Repeat with bins derived from the imputed signal. This exposes models that are accurate only where signal is high (or only where it is absent).
2. **Partition by cell-type specificity.** For each locus, the specificity score is the number of cell types in which the binarised signal is 1 for a given assay (a column sum over the cell-type × locus binary matrix). Group loci by equal specificity and compute **precision** and **recall** per group. This directly measures whether a model captures cell-type-specific signal or only the [[average-activity-baseline]].
3. **Promoter correlation.** Mean Pearson correlation of H3K4me3 signal over ±2 kb windows centred on gene starts, using the **GENCODE v38** annotation (`raw/harrow-2012-gencode.xml`); averaged over genes.
4. **Peak correlation.** Mean correlation of DNase signal restricted to MACS2 peak calls. Unlike promoters, peaks are cell-type-specific, so this measure probes cell-type-specific accuracy in the regions that matter.

## Beyond the challenge

`raw/schreiber-2023-encode-imputation-challenge.pdf` does not evaluate distributional or uncertainty-aware outputs — every submission produced point estimates, so calibration and ranking quality were outside its scope. Methods that emit predictive distributions need the additional measures described in [[uncertainty-calibration]].

## See also

Related:: [[encode-imputation-challenge]], [[average-activity-baseline]], [[peak-calling-and-signal-tracks]], [[uncertainty-calibration]], [[cross-cell-type-generalization-pitfall]]

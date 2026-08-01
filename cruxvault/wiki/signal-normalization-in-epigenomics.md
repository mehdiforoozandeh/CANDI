---
type: wiki
title: Signal normalisation in epigenomics
summary: The methods that try to make epigenomic tracks comparable across experiments — scaling, S3norm, spike-in, CHIPIN, quantile variants — and the assumption each one makes.
category: concept
sources: raw/xiang-2020-s3norm.xml, raw/polit-2021-chipin.xml, raw/bonhoure-2014-chip-seq-spiking.xml, raw/angelini-2015-chip-seq-normalization-diagnostic.xml, raw/reske-2020-atac-seq-normalization.xml, raw/schreiber-2023-encode-imputation-challenge.pdf
created: 2026-07-31T21:26:00
updated: 2026-07-31T21:26:00
---

# Signal normalisation in epigenomics

Every method here is defined by the invariant it assumes — a set of regions, genes, or quantiles whose true signal is unchanged across samples — and it fails exactly when that invariant does not hold.

## The two quantities that differ between experiments

`raw/xiang-2020-s3norm.xml` names them: **sequencing depth (SD)** and **signal-to-noise ratio (SNR)**. Most normalisation methods rescale either background regions or peak regions and assume one scale factor serves both. That adjusts for depth but not for SNR, so two experiments matched on total reads can still differ systematically in how much of that depth landed in peaks.

**S3norm** matches both simultaneously by fitting a **monotonic non-linear (two-factor power) transformation** from a target dataset to a reference, chosen so that both the background mean and the peak mean align. Evaluated on the VISION compendium (8 marks × 20 hematopoietic cell types), with downstream checks on gene-expression prediction from histone marks and on MACS2 peak calls.

## The invariant-set family

- **CHIPIN** (`raw/polit-2021-chipin.xml`) — assumes that, on average, ChIP-seq signal in the regulatory regions of genes whose **expression is constant across conditions** should not differ. It uses external gene-expression data to define those "constant genes," computes signal density over them (±4 kb by default), and fits either a quantile or linear normalisation from that. Designed for the case where spike-in information is unavailable but expression data is.
- **Spike-in / SAP** (`raw/bonhoure-2014-chip-seq-spiking.xml`) — the invariant is supplied experimentally rather than assumed: a fixed quantity of foreign-genome chromatin added pre-IP. The only approach in this set that survives a **uniform genome-wide** occupancy change. See [[distributional-shift-and-batch-effects]].
- **ChIP-vs-input scale factor** (`raw/angelini-2015-chip-seq-normalization-diagnostic.xml`) — the ratio *r* of ChIP to input reads in background windows. CisGenome estimates it from low-count 100 bp windows; NCIS uses data-adaptive window size and threshold; CCAT iterates using strand symmetry. MACS and SICER instead use the naive total-read ratio.

## Diagnosing whether it worked

`raw/angelini-2015-chip-seq-normalization-diagnostic.xml` fills a real gap: there was no tool to check whether a chosen normalisation constant is appropriate. Its diagnostic plots empirical densities of log relative risks in bins of equal read count alongside the estimated constant (log-transformed); a misestimated constant shows up as a systematic offset. The plot is valid for Poisson bin counts and for anything more dispersed, e.g. negative binomial — see [[count-distributions-for-sequencing-data]]. The paper also shows the normalisation constant materially changes MACS and SICER peak calls, and proposes a sample-swapping FDR procedure that gains power over the naive constant.

## The choice is not neutral

`raw/reske-2020-atac-seq-normalization.xml` compares 8 differential-accessibility pipelines (MACS2, DiffBind, csaw, voom, limma, edgeR, DESeq2) on the same ATAC-seq data and finds the **normalisation method changes the biological conclusion**, especially under global chromatin alterations. Its proposed workflow standardises molecular complexity before quantifying differences.

## What the challenge recommends

`raw/schreiber-2023-encode-imputation-challenge.pdf` concludes that when processing differences cannot be undone, the correction must be more than a rescaling, and recommends **quantile normalisation applied separately to signal in peaks and signal in background** — structurally the same peak/background split as S3norm, implemented non-parametrically. See [[quantile-normalization]].

## See also

Related:: [[quantile-normalization]], [[distributional-shift-and-batch-effects]], [[chip-seq-assay-and-controls]], [[count-distributions-for-sequencing-data]], [[peak-calling-and-signal-tracks]], [[sequencing-depth-and-coverage]]

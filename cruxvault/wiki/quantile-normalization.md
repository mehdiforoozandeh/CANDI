---
type: wiki
title: Quantile normalisation
summary: Forcing samples to share an empirical distribution — the default cross-sample normaliser, its failure when class or batch composition differs, and the smooth and peak/background variants.
category: method
sources: raw/zhao-2020-quantile-normalization-correctly.xml, raw/hicks-2018-smooth-quantile-normalization.pdf, raw/townes-2020-quantile-normalization-scrnaseq.xml, raw/schreiber-2023-encode-imputation-challenge.pdf
created: 2026-07-31T21:26:00
updated: 2026-07-31T21:26:00
---

# Quantile normalisation

Quantile normalisation's assumption — that the true distribution of values is the same in every sample — is exactly the assumption that fails whenever the biology involves a global shift, and every variant below is a way to relax it.

## The procedure

Rank the features within each sample, average across samples the values occupying each rank, and substitute that average for every feature at that rank (`raw/zhao-2020-quantile-normalization-correctly.xml`). The result: all samples share an identical empirical distribution, differing only in which feature sits at which rank.

## When it goes wrong

`raw/zhao-2020-quantile-normalization-correctly.xml` identifies two failure drivers: the **class-effect proportion** (what fraction of features genuinely differ between classes) and **batch effects**. Applying QN blindly to a whole dataset raises both false-positive and false-negative rates. Their evaluation of five strategies concludes that **splitting the data by class label first and quantile-normalising each split independently** ("class-specific") recovers good batch correction and feature selection, and — importantly — remains robust when separately normalised datasets are later combined.

## Smooth quantile normalisation (qsmooth)

`raw/hicks-2018-smooth-quantile-normalization.pdf` addresses the same tension by making the choice continuous rather than binary. Instead of committing to either full QN (assume one common distribution) or per-group normalisation (assume none), it estimates a weight that interpolates between the two, based on how much of the variability in the data is between-group versus within-group. Where global differences are genuine, it shrinks towards group-specific distributions; where they are technical, towards a common one.

## Quantile normalisation as distribution matching

`raw/townes-2020-quantile-normalization-scrnaseq.xml` uses QN in a different mode: not to make samples match *each other*, but to make them match a **target distribution derived from a better assay**. scRNA-seq read counts without UMIs are quantile-normalised to a compound Poisson (Poisson-lognormal) distribution empirically fitted from UMI datasets, yielding "quasi-UMIs" that are more accurate than competing normalisations and that let UMI-specific downstream methods run on non-UMI data. The pattern — QN as a bridge between two protocols measuring the same quantity — is directly analogous to correcting a single-end/paired-end shift.

## The peak/background split

`raw/schreiber-2023-encode-imputation-challenge.pdf` recommends quantile-normalising **signal in peaks and signal in background separately** when processing differences across a compendium cannot be undone. This is a structural response to the same problem `raw/zhao-2020-quantile-normalization-correctly.xml` and `raw/hicks-2018-smooth-quantile-normalization.pdf` treat: the genome contains two populations with very different distributions, and pooling them into one quantile map lets a change in the peak/background mixture masquerade as a change in signal.

Note that quantile normalisation was also one of the four preprocessing choices used by challenge entrants (alongside `arcsinh`, `log1p`, and Cauchy) — see [[encode-imputation-challenge]].

## See also

Related:: [[signal-normalization-in-epigenomics]], [[distributional-shift-and-batch-effects]], [[encode-imputation-challenge]], [[count-distributions-for-sequencing-data]]

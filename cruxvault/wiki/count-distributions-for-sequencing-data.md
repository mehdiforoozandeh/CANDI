---
type: wiki
title: Count distributions for sequencing data
summary: Poisson and negative binomial models of read counts, the overdispersion that forces the NB, and the variance-stabilising transforms used when counts are fed to a neural network.
category: concept
sources: raw/anders-2010-deseq.xml, raw/zhang-2008-macs.xml, raw/jung-2014-sequencing-depth-chip-seq.xml, raw/townes-2020-quantile-normalization-scrnaseq.xml, raw/hoffman-2012-segway.xml, raw/avsec-2021-enformer.xml, raw/angelini-2015-chip-seq-normalization-diagnostic.xml, raw/schreiber-2023-encode-imputation-challenge.pdf
created: 2026-07-31T21:26:00
updated: 2026-07-31T21:26:00
---

# Count distributions for sequencing data

The negative binomial is not a modelling flourish in this literature — it is the minimum distribution that fits, because biological replicates are more variable than Poisson sampling allows.

## Poisson as the null

Read positions along the genome are conventionally modelled as Poisson. `raw/zhang-2008-macs.xml` uses a **local Poisson** null to compute enrichment significance, and `raw/jung-2014-sequencing-depth-chip-seq.xml` derives genomic coverage as a function of sequencing depth by assuming the observed tag distribution is Poisson. Poisson is adequate for *technical* sampling: given a fixed underlying abundance, the count of reads drawn is Poisson with mean proportional to depth.

## Why the negative binomial

`raw/anders-2010-deseq.xml` (DESeq) makes the case that Poisson is insufficient once **biological** variability is present: counts are **overdispersed**, with variance exceeding the mean. The model is

    K_ij ~ NB(μ_ij, σ²_ij)

with the mean–variance relationship estimated by **local regression** rather than assumed parametric, and three parameter groups: per-sample **size factors** *s_j* (all counts from sample *j* are expected to be proportional to *s_j*), per-condition expression strengths *q_iρ*, and the fitted variance function. The size factor is the paper's formalisation of what "sequencing depth normalisation" means — an offset on the mean, not a rescaling of the data. See [[sequencing-depth-and-coverage]].

The NB matters here for two reasons: it is the correct likelihood for raw epigenomic counts, and — `raw/angelini-2015-chip-seq-normalization-diagnostic.xml` argues — any method valid for Poisson bin counts remains valid for more-dispersed distributions, so NB is the safe default.

`raw/townes-2020-quantile-normalization-scrnaseq.xml` provides the compound-Poisson counterpart: UMI counts are well fit by a **Poisson-lognormal** distribution, characterised per cell by just a scale and a shape parameter — which is what makes distribution-matching normalisation tractable there.

## Transforms when counts feed a network

Raw counts span orders of magnitude, so models transform them:

- **arcsinh**, `asinh(x) = ln(x + √(x²+1))`. `raw/hoffman-2012-segway.xml` adopts it explicitly "to reduce the distorting effects of high data values in sequence census assay data," noting it compresses like `ln x` for large values but — unlike `log`— is defined and near-linear at zero, so zero counts need no pseudocount. This is why `arcsinh` is the standard transform for epigenomic count data.
- **log1p** and **quantile** were the other preprocessing choices among challenge entrants (`raw/schreiber-2023-encode-imputation-challenge.pdf`; see [[encode-imputation-challenge]]).

## Count likelihoods as training objectives

`raw/avsec-2021-enformer.xml` trains Enformer with a **Poisson negative log-likelihood** loss on binned coverage rather than a squared error, i.e. treating the prediction task as fitting a count distribution's rate. This is the sequence-model lineage's version of the same argument DESeq makes for differential expression: the noise model should match the data-generating process. Squared error implicitly assumes homoscedastic Gaussian noise, which count data violates badly at both ends of the dynamic range.

## See also

Related:: [[peak-calling-and-signal-tracks]], [[signal-normalization-in-epigenomics]], [[uncertainty-calibration]], [[sequencing-depth-and-coverage]], [[sequence-conditioned-epigenome-models]]

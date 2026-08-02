---
type: wiki
title: Count distributions for sequencing data
summary: Poisson and negative binomial models of read counts, the overdispersion that forces the NB, and the variance-stabilising transforms used when counts are fed to a neural network.
category: concept
sources: raw/anders-2010-deseq.xml, raw/zhang-2008-macs.xml, raw/jung-2014-sequencing-depth-chip-seq.xml, raw/townes-2020-quantile-normalization-scrnaseq.xml, raw/hoffman-2012-segway.xml, raw/avsec-2021-enformer.xml, raw/angelini-2015-chip-seq-normalization-diagnostic.xml, raw/schreiber-2023-encode-imputation-challenge.pdf, raw/young-2024-ddpn.pdf, raw/rigby-2005-gamlss.xml, raw/kingma-2013-vae.pdf, raw/choudhary-2022-sctransform-v2.xml, raw/svensson-2020.pdf
created: 2026-07-31T21:26:00
updated: 2026-08-01T18:05:24
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

## What the overdispersion evidence actually covers

> **Gap note.** Every direct measurement of overdispersion held in `raw/` is from **RNA-seq or
> scRNA-seq**, not ChIP-seq or ATAC-seq. `raw/anders-2010-deseq.xml` is RNA-seq;
> `raw/choudhary-2022-sctransform-v2.xml`'s 59-dataset survey is scRNA-seq; `raw/svensson-2020.pdf`
> is droplet scRNA-seq. No source in this vault measures the mean–variance relationship of binned
> ChIP-seq counts across biological replicates. The NB case for epigenomic counts as stated above
> is therefore **transferred, plus a safety argument** (`raw/angelini-2015-chip-seq-normalization-diagnostic.xml`:
> Poisson-valid methods stay valid under greater dispersion) — not a measurement. Treat "NB is the
> correct likelihood for raw epigenomic counts" as a well-motivated default with an open empirical
> question behind it, and note that the safety argument is one-directional: it licenses NB over
> Poisson, and says nothing about NB versus a distribution that permits **under**-dispersion (see
> the DDPN critique below).

Two transferable results do survive the domain change, because they are about *estimation* rather
than about RNA:

- **Shallow sequencing masks overdispersion.** `raw/choudhary-2022-sctransform-v2.xml` finds
  Poisson looks adequate for sparse data, while at sufficient depth overdispersion is evident in
  every biological system. A "Poisson fits fine" result at low depth is therefore uninformative —
  which is exactly the regime most epigenomic 25 bp bins occupy.
- **Excess zeros are not evidence of zero-inflation.** `raw/svensson-2020.pdf` shows the zero
  counts in droplet data are what a plain NB already predicts at low mean. The same reasoning
  applies to sparse epigenomic bins: sparsity is not by itself a reason to reach for a
  zero-inflated head. See [[count-models-in-single-cell-genomics]].

## Transforms when counts feed a network

Raw counts span orders of magnitude, so models transform them:

- **arcsinh**, `asinh(x) = ln(x + √(x²+1))`. `raw/hoffman-2012-segway.xml` adopts it explicitly "to reduce the distorting effects of high data values in sequence census assay data," noting it compresses like `ln x` for large values but — unlike `log`— is defined and near-linear at zero, so zero counts need no pseudocount. This is why `arcsinh` is the standard transform for epigenomic count data.
- **log1p** and **quantile** were the other preprocessing choices among challenge entrants (`raw/schreiber-2023-encode-imputation-challenge.pdf`; see [[encode-imputation-challenge]]).

## Count likelihoods as training objectives

`raw/avsec-2021-enformer.xml` trains Enformer with a **Poisson negative log-likelihood** loss on binned coverage rather than a squared error, i.e. treating the prediction task as fitting a count distribution's rate. This is the sequence-model lineage's version of the same argument DESeq makes for differential expression: the noise model should match the data-generating process. Squared error implicitly assumes homoscedastic Gaussian noise, which count data violates badly at both ends of the dynamic range.

## Heteroscedastic counts

`raw/young-2024-ddpn.pdf` (Deep Double Poisson Networks) is the count-specific counterpart to the Gaussian heteroscedasticity literature. Its argument: the standard count heads **couple mean and dispersion**. Poisson forces variance = mean exactly; negative binomial allows overdispersion but ties it to the mean through a single dispersion parameter, and can represent variance only *above* the mean. A network that must express under-dispersion, or dispersion varying independently of the mean, cannot do so — and its predictive distributions are miscalibrated as a result. DDPN uses the double Poisson to give **fully heteroscedastic** count regression, with mean and dispersion free to vary independently.

This is the sharpest available critique of an NB output head: the failure is not accuracy of the mean but the shape of the predicted distribution where the mean–dispersion coupling is wrong.

## Separating location, scale and shape

`raw/rigby-2005-gamlss.xml` (GAMLSS) is the statistical framework for exactly that concern. A classical GLM lets covariates drive the **mean** and treats the remaining distributional parameters as nuisance constants. GAMLSS lets **different covariates drive location, scale and shape separately**, each through its own additive predictor. Applied to a count model, this means the covariates that set the expected count need not be the covariates that set the dispersion — and forcing them to be the same is a modelling assumption, not a necessity.

For experimental covariates this distinction is substantive: sequencing depth is a location-scale quantity acting through an offset, while read length or protocol may plausibly affect dispersion without shifting the mean. GAMLSS is the framework in which "route this covariate to the dispersion but not the mean" is a well-posed statement rather than an implementation hack. See [[count-models-in-single-cell-genomics]] for how the single-cell field parameterises dispersion in practice.

## Latent-variable models over counts

`raw/kingma-2013-vae.pdf` supplies the two mechanisms that any model with a probabilistic latent uses: the **reparameterisation trick**, which makes a stochastic latent differentiable by sampling `z = μ + σ⊙ε` with `ε ~ N(0, I)` so gradients flow through μ and σ; and the **ELBO**, a reconstruction term plus a KL divergence pulling the approximate posterior toward the prior. The KL term is the object that trades representation capacity against prior conformity — weight it too heavily and the latent collapses to the prior, carrying no information about the input. That failure mode, and the practice of annealing the KL weight up over training to avoid it, are the two things worth knowing before enabling a KL term on a latent.

## See also

Related:: [[peak-calling-and-signal-tracks]], [[signal-normalization-in-epigenomics]], [[uncertainty-calibration]], [[sequencing-depth-and-coverage]], [[sequence-conditioned-epigenome-models]], [[count-models-in-single-cell-genomics]], [[training-mechanics]], [[regression-likelihoods]]

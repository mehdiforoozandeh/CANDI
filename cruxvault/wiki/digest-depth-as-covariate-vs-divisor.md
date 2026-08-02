---
type: wiki
title: Digest — is sequencing depth a covariate or a divisor?
summary: Filed-back answer: the literature is one-sided. Depth enters the mean as an offset, not the data as a divisor — and it is not a pure scale factor, so dividing it out is lossy in a way that is measurable.
category: digest
sources: raw/anders-2010-deseq.xml, raw/jung-2014-sequencing-depth-chip-seq.xml, raw/hafemeister-2019-sctransform.xml, raw/lopez-2018-scvi.xml, raw/karimzadeh-2018-umap-mappability.xml, raw/angelini-2015-chip-seq-normalization-diagnostic.xml
created: 2026-08-01T18:18:11
updated: 2026-08-01T18:18:11
---

# Digest — is sequencing depth a covariate or a divisor?

> **Digest.** A question-shaped synthesis, filed back so the reconstruction does not have to be
> repeated. It draws only on sources in `raw/` and pages in this wiki; it records no findings from
> this project. Answers previously required opening five pages.

**Short answer: as a covariate, and the literature is unusually one-sided about it.** Every source
here that models depth explicitly puts it in the *mean*, and none divides it out of the data.

## The three positions, and why only one survives

| Treatment | What it assumes | Where it appears |
|---|---|---|
| **Divisor** — scale counts to a common depth | depth acts as a pure multiplicative constant on every position equally | the implicit assumption behind RPKM-style scaling and behind consuming pre-normalised tracks |
| **Offset on the mean** — `μ_ij ∝ s_j` | depth sets the expected count; the observation stays integral and its sampling noise stays honest | `raw/anders-2010-deseq.xml`, `raw/hafemeister-2019-sctransform.xml`, `raw/lopez-2018-scvi.xml` |
| **Conditioning covariate** — feed depth to the model | depth may act non-linearly and interact with other covariates | the general case; strictly more expressive than an offset |

`raw/anders-2010-deseq.xml` formalises the middle row: per-sample **size factors** *s_j*, with all
counts from sample *j* expected proportional to *s_j*. The size factor is the paper's definition of
what depth normalisation *means* — an offset on the mean, not a rescaling of the data.
`raw/hafemeister-2019-sctransform.xml` does the same operationally, fitting an NB GLM per feature
with **log sequencing depth as a covariate**. `raw/lopez-2018-scvi.xml` goes one step further and
gives the model a **library-size latent** alongside batch annotations fed to the decoder.

The reason the first row loses is a statistical one that nothing in the divisor approach recovers:
dividing a count by a constant destroys the count's own noise model. A raw count of 3 and a
depth-normalised value of 3.0 carry different information about how certain you should be, and only
the former is a draw from a distribution whose variance is known. See
[[count-distributions-for-sequencing-data]].

## Why depth is not a pure scale factor anyway

Even granting the divisor its own assumption, the assumption is false, and two sources say so from
different directions:

- **Coverage saturates non-linearly.** `raw/jung-2014-sequencing-depth-chip-seq.xml` derives
  genomic coverage as a function of depth and finds the requirement differs by mark — punctate and
  broad marks do not reach usable coverage at the same depth. Doubling reads does not double
  information, and it does not do so equally across assays.
- **Depth interacts with mappability.** `raw/karimzadeh-2018-umap-mappability.xml` shows which
  positions are measurable at all depends on **read length**, which co-varies with protocol and
  therefore with depth. A global divisor is wrong specifically in low-mappability regions, where
  the shortfall is not a scale effect but a missing-position effect.

Both are the same structural point: depth changes *which positions carry signal*, not just *how
large the numbers are*. See [[sequencing-depth-and-coverage]].

## The one argument on the other side

The strongest case for normalising is comparability, and `raw/angelini-2015-chip-seq-normalization-diagnostic.xml`
is the honest version of it: methods valid under a Poisson bin-count assumption remain valid under
greater dispersion, so a normalisation that preserves the Poisson-ish structure is not
*invalidating*. That is a safety argument, not an accuracy argument — it licenses normalisation
where you must, and does not claim it loses nothing. Where the model can consume raw counts, the
offset formulation strictly dominates.

## What this leaves open

Nothing here settles whether depth should additionally be routed to the **dispersion** rather than
only the mean. That is a separate question, well-posed under GAMLSS, and unanswered by the sources
above — see [[regression-likelihoods]] and [[count-distributions-for-sequencing-data]].

## See also

Related:: [[sequencing-depth-and-coverage]], [[count-distributions-for-sequencing-data]], [[signal-normalization-in-epigenomics]], [[count-models-in-single-cell-genomics]], [[covariate-conditioning-and-counterfactuals]], [[digest-normalization-assumptions-of-prior-imputation-methods]]

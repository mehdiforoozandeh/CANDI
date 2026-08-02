---
type: wiki
title: Digest — which likelihood for raw epigenomic counts?
summary: Filed-back answer, honestly partial: NB is the well-motivated default and every direct overdispersion measurement in this vault is from RNA-seq. What the choice actually costs, and what would settle it.
category: digest
sources: raw/anders-2010-deseq.xml, raw/angelini-2015-chip-seq-normalization-diagnostic.xml, raw/choudhary-2022-sctransform-v2.xml, raw/svensson-2020.pdf, raw/townes-2019-glmpca.xml, raw/young-2024-ddpn.pdf, raw/avsec-2021-enformer.xml
created: 2026-08-01T18:18:11
updated: 2026-08-01T18:18:11
---

# Digest — which likelihood for raw epigenomic counts?

> **Digest.** A question-shaped synthesis, filed back so the reconstruction does not have to be
> repeated. It draws only on sources in `raw/` and pages in this wiki; it records no findings from
> this project. **This answer is partial by construction** — the decisive measurement is not in
> `raw/`, and saying so is the point of writing it down.

**Short answer: negative binomial, as a well-motivated default rather than a measured result.**

## What is actually established

- **Poisson is right for technical sampling, wrong once biological variability enters.**
  `raw/anders-2010-deseq.xml` makes the case: counts are overdispersed, variance exceeds mean, and
  the mean–variance relationship is better estimated than assumed.
- **NB is the safe direction to be wrong in.** `raw/angelini-2015-chip-seq-normalization-diagnostic.xml`
  argues any method valid for Poisson bin counts remains valid under greater dispersion. Note the
  asymmetry: this licenses NB **over Poisson** and says nothing about NB versus a distribution
  permitting **under**-dispersion.
- **"Poisson fits" can be an artefact of depth.** `raw/choudhary-2022-sctransform-v2.xml` finds
  Poisson looks adequate on sparse data, while at sufficient depth overdispersion appears in every
  biological system surveyed. Sparse 25 bp bins are exactly the regime where the test is
  underpowered.
- **Excess zeros are not evidence of zero-inflation.** `raw/svensson-2020.pdf` and
  `raw/townes-2019-glmpca.xml` both conclude the zeros are what a plain NB (or multinomial) already
  predicts. Reach for ZINB only on evidence, not on sparsity.

## What is not established

Every one of those measurements is **RNA-seq or scRNA-seq**. No source in `raw/` measures the
mean–variance relationship of binned ChIP-seq or ATAC-seq counts across biological replicates. The
NB case for epigenomic counts is therefore a transfer plus a safety argument — sound, but not the
same thing as a measurement, and the honest way to hold it is as a default with an open question
behind it.

Two features of epigenomic data make the transfer non-obvious rather than routine: the control
channel already absorbs some of the regional variability that drives overdispersion in RNA-seq, and
25 bp bins are spatially autocorrelated in a way gene-level counts are not, so "biological
replicate variance" is not measuring the same object.

## What the choice costs if it is wrong

`raw/young-2024-ddpn.pdf` states the cost precisely: NB **couples mean and dispersion** through a
single parameter and can represent variance only *above* the mean. If any regime of epigenomic
counts is genuinely under-dispersed — plausible where a local background is near-deterministic —
an NB head cannot represent it, and the failure appears as **miscalibration rather than as bias in
the mean**. That is the failure mode to watch, and it is invisible to MSE and correlation.

Worth noting that the sequence lineage made the other choice and lives with it:
`raw/avsec-2021-enformer.xml` trains on a **Poisson** NLL over binned coverage, accepting
variance = mean exactly.

## What would settle it

A direct measurement, on this data: bin ChIP-seq counts at 25 bp, group biological replicates of
the same assay and biosample, and fit the mean–variance relationship — the same procedure
`raw/anders-2010-deseq.xml` performs for RNA-seq. The outcome distinguishes Poisson, NB, and the
double-Poisson case cleanly, and until it is run the default should be labelled as such.

## See also

Related:: [[count-distributions-for-sequencing-data]], [[count-models-in-single-cell-genomics]], [[regression-likelihoods]], [[uncertainty-calibration]], [[sequencing-depth-and-coverage]], [[digest-depth-as-covariate-vs-divisor]]

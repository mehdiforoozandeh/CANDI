---
type: wiki
title: Digest — what normalisation do prior imputation methods assume?
summary: Filed-back answer: nearly all of them assume it already happened upstream. They consume −log10 p-value tracks and inherit whatever the pipeline did, which is why a processing difference invalidated an entire benchmark.
category: digest
sources: raw/ernst-2015-chromimpute.xml, raw/durham-2018-predictd.xml, raw/schreiber-2020-avocado.xml, raw/hawkins-hooker-2023-edice.xml, raw/schreiber-2023-encode-imputation-challenge.pdf, raw/zhang-2008-macs.xml, raw/xiang-2020-s3norm.xml
created: 2026-08-01T18:18:11
updated: 2026-08-01T18:18:11
---

# Digest — what normalisation do prior imputation methods assume?

> **Digest.** A question-shaped synthesis, filed back so the reconstruction does not have to be
> repeated. It draws only on sources in `raw/` and pages in this wiki; it records no findings from
> this project. Answers previously required reconstructing across four pages with no landing page.

**Short answer: they assume it already happened, and none of them models it.** The imputation
lineage takes normalisation as a property of its input rather than as something to be learned,
which is a single shared assumption and a single shared point of failure.

## The chain every method inherits

1. Reads are aligned, deduplicated, and filtered.
2. MACS2 (`raw/zhang-2008-macs.xml`) computes enrichment against a **local Poisson background**
   estimated from the control, emitting a −log10 p-value per position.
3. The imputation method consumes that track.

Step 2 is where normalisation happens, and it is doing more than the word suggests: a p-value has
already folded in depth, local background, and the control experiment. `raw/ernst-2015-chromimpute.xml`,
`raw/durham-2018-predictd.xml`, `raw/schreiber-2020-avocado.xml` and `raw/hawkins-hooker-2023-edice.xml`
all consume this quantity. What varies between them is only the further transform applied on top —
`raw/schreiber-2023-encode-imputation-challenge.pdf` (Table 1) records that almost every challenge
entrant added one of `arcsinh`, `log1p`, quantile, or Cauchy.

So the answer to "what normalisation does method X assume?" is the same for all of them: **whatever
the ENCODE pipeline did**, plus a variance-stabilising transform chosen by taste.

## Why that assumption is load-bearing

`raw/schreiber-2023-encode-imputation-challenge.pdf` is the natural experiment. The presumption is
that upstream processing has removed batch and depth effects and produced an idealised "signal
strength". It had not: a difference in the **deduplication rule** between single-end and paired-end
data (single-end keeps one read per duplicate set; paired-end keeps a pair if *either* mate is
unique) shifted the distribution enough that a naive baseline outperformed all but two of 23
submissions. Nothing in any method's design could have detected this, because the assumption is
made before the model sees the data.

The paper's own conclusion is the strongest available statement of the limit: the correction
required is **more than a simple rescaling**, and its recommendation is quantile normalisation
applied to **signal in peaks and signal in background separately**. A single global transform is
insufficient by the field's own post-mortem. See [[distributional-shift-and-batch-effects]] and
[[quantile-normalization]].

## The one method family that models it

`raw/xiang-2020-s3norm.xml` (S3norm) is the exception worth naming: it fits the transform rather
than assuming it, matching both the peak signal and the background of a target dataset to a
reference by a monotone transformation with fitted parameters. That makes normalisation a
**learned, two-component** object instead of a preprocessing step — which is structurally the same
move as conditioning on the covariates that caused the shift, but performed outside the model
rather than inside it. See [[signal-normalization-in-epigenomics]].

## The consequence for a raw-count model

A model that consumes **counts plus experimental covariates** is not making the assumption above at
all. That is a genuine difference in kind rather than degree, and it has one specific implication
worth stating: it cannot be compared to these methods on their own inputs without giving up the
property that distinguishes it. Benchmarks defined on p-value tracks measure agreement with the
pipeline's normalisation choices as much as they measure imputation skill.

## See also

Related:: [[epigenome-imputation]], [[signal-normalization-in-epigenomics]], [[distributional-shift-and-batch-effects]], [[quantile-normalization]], [[peak-calling-and-signal-tracks]], [[encode-imputation-challenge]], [[digest-depth-as-covariate-vs-divisor]]

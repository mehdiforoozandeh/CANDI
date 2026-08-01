---
type: wiki
title: Average activity baseline
summary: Predicting an assay's mean signal across training cell types at each locus — the naive baseline that outperformed all but two of 23 ENCODE Imputation Challenge submissions.
category: concept
sources: raw/schreiber-2023-encode-imputation-challenge.pdf, raw/schreiber-2020-pitfall-cross-cell-type.xml
created: 2026-07-31T21:26:00
updated: 2026-07-31T21:26:00
---

# Average activity baseline

The average activity is not a strawman: it is the cell-type-invariant component of the signal, and because most of the genome's epigenomic variance is cell-type-invariant, it is a genuinely strong predictor.

## Definition

For assay *a* and genomic position *ℓ*, the average activity is the mean of the observed signal for assay *a* at position *ℓ* over all **training** cell types. It carries no information about the target cell type at all (`raw/schreiber-2023-encode-imputation-challenge.pdf`).

## Why it is hard to beat

`raw/schreiber-2020-pitfall-cross-cell-type.xml` supplies the mechanism: when training and test sets share genomic loci, a model can appear to perform well by **memorising the average activity of each locus** across the training cell types, without learning anything cell-type-specific. Any evaluation that does not separate loci therefore credits the model for reproducing a baseline it did not need to learn. See [[cross-cell-type-generalization-pitfall]].

`raw/schreiber-2023-encode-imputation-challenge.pdf` reports the corollary: eight of the 23 challenge submissions used the average activity as an **explicit input**, but methods that did not use it explicitly did **not** consistently show lower correlation with it — because the average activity is directly derivable from the training set, many model families learn it implicitly whether or not they are told to.

## The challenge result

Under the pre-registered performance measures, the average activity baseline outperformed **all but two** of the 23 submissions, and those two only marginally. After correcting the [[distributional-shift-and-batch-effects]] between the single-end training data and the paired-end test data, more than half the participants beat the same baseline — so the baseline's dominance was an artefact of the shift, but the fact that it went undetected is the lesson.

## Practical rule

The paper's first recommendation to future challenge organisers is to ensure participants are compared against naive baselines such as the average activity. For a method to demonstrate cell-type-specific imputation ability, it must be shown to beat the average activity **on the same loci and the same evaluation split**, and ideally the reported metric should be the *improvement over* the baseline rather than an absolute correlation or MSE.

The two reference baselines used throughout the challenge's error analysis were the average activity and [[avocado]]'s imputations; entrant error patterns were characterised by which of these two they resembled.

## See also

Related:: [[encode-imputation-challenge]], [[imputation-evaluation-measures]], [[cross-cell-type-generalization-pitfall]], [[avocado]]

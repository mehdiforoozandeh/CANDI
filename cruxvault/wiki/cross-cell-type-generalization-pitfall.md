---
type: wiki
title: Cross-cell-type generalisation pitfall
summary: Sharing genomic loci between train and test lets a model score well by memorising per-locus average activity; and performance on well-characterised cell types does not predict performance on poorly characterised ones.
category: concept
sources: raw/schreiber-2020-pitfall-cross-cell-type.xml, raw/schreiber-2023-encode-imputation-challenge.pdf
created: 2026-07-31T21:26:00
updated: 2026-07-31T21:26:00
---

# Cross-cell-type generalisation pitfall

Two distinct failures wear the same name, and only one of them has a fix.

## Failure 1 — locus leakage (has a fix)

`raw/schreiber-2020-pitfall-cross-cell-type.xml` demonstrates that when the training and test sets contain the **same genomic loci** (differing only in cell type), a model may falsely appear to perform well by effectively memorising the average activity associated with each locus across the training cell types. The paper shows this concretely for predicting gene expression and for predicting chromatin domain boundaries.

Diagnosis and avoidance:
- Compare against the [[average-activity-baseline]] computed on the same split — if the model does not beat it, the apparent performance is leakage.
- Split by genomic locus (e.g. hold out whole chromosomes), not only by cell type, so that the test loci are unseen.
- Report per-locus *residual* performance after regressing out the average activity.

The paper's closing warning is that the risk **grows** as more data becomes available, because larger compendia make the memorisable average more accurate.

## Failure 2 — characterisation imbalance (has no simple fix)

`raw/schreiber-2023-encode-imputation-challenge.pdf` identifies the second, harder problem: k-fold or leave-one-out cross-validation applied to a whole compendium **over-emphasises well-characterised cell types**, because those cell types contribute most of the (cell type, assay) pairs. Good performance on a cell type with 30 observed assays is not an indicator of good performance on one with two — and the latter is where imputation is actually needed.

The paper is explicit that, unlike the distributional-shift and measure-redundancy problems, this one "does not have a simple fix." The only remedy offered is design: **explicitly include both well-characterised and poorly characterised cell types in the evaluation**, and ensure at least one evaluation setting matches how the method is expected to be used in practice. The challenge itself operationalised this by drawing its blind test set almost entirely from poorly characterised cell types (9 of 12 test cell types had ≤2 training experiments).

## See also

Related:: [[average-activity-baseline]], [[imputation-evaluation-measures]], [[encode-imputation-challenge]], [[epigenome-imputation]]

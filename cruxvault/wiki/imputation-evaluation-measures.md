---
type: wiki
title: Imputation evaluation measures
summary: Why MSE and global correlation are assay-dependent and mutually redundant, and the challenge's replacement measures partitioned by signal strength, cell-type specificity, promoters, and peaks.
category: concept
sources: raw/schreiber-2023-encode-imputation-challenge.pdf, raw/harrow-2012-gencode.xml, raw/zhang-2008-macs.xml, raw/neufeld-2023-thinning.pdf, raw/cameron-2008-cgm-bootstrap.pdf, raw/toneyan-2022.xml, raw/rafi-2024-dream.xml, raw/zhou-2026-degu.xml, raw/saito-2015-precision-recall-plot.xml
created: 2026-07-31T21:26:00
updated: 2026-08-01T18:14:33
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

## Splitting counts instead of samples

`raw/neufeld-2023-thinning.pdf` (data thinning) solves a problem that sample-splitting cannot. It splits a **single observation** into two or more independent parts that sum to the original and follow the same distribution up to a known scaling of a parameter. This works for any **convolution-closed** distribution — Gaussian, Poisson, **negative binomial**, gamma, binomial.

For count data this is the principled way to obtain independent train and test copies of the *same* experiment: thin a count track into two independent tracks, fit on one, evaluate on the other. It is the correct instrument whenever evaluation would otherwise reuse the same counts for input and target, and it is also how one estimates an irreducible noise floor — thin an experiment and measure how well one half predicts the other, which bounds what any model can achieve.

The direction is worth noting: thinning splits a count **downward**. It can simulate a shallower experiment from a deeper one, but not the reverse, so it supplies supervision for lower-depth targets only.

## Inference with few clusters

`raw/cameron-2008-cgm-bootstrap.pdf` addresses a regime that genomics evaluations routinely occupy without acknowledging it. Cluster-robust standard errors — the standard fix for within-group dependence, e.g. positions within a chromosome or experiments within a biosample — **assume the number of clusters is large**. With few clusters (the paper's stated range is **5–30**), standard asymptotic tests over-reject substantially: nominal 5% tests reject at around 10%.

The remedy is a **cluster bootstrap-t** with asymptotic refinement (the wild cluster bootstrap), which the paper shows restores the nominal rate. Any comparison resting on a handful of biosamples or target experiments is in this regime, and a plain cluster bootstrap is not sufficient.

## Evaluation frameworks for epigenomic profile prediction

`raw/toneyan-2022.xml` builds a unified evaluation framework (GOPHER) and uses it to compare binary and quantitative models of chromatin accessibility. Its findings are methodological warnings that apply directly to any track-prediction comparison:

- Models using **different target resolutions cannot be compared directly**, because larger bins smooth high-frequency noise and mechanically inflate correlation-based metrics. A resolution difference alone can look like a performance difference.
- **Dataset selection** — coverage thresholds, peak-centred versus whole-chromosome test sets — changes the apparent generalisability substantially.
- Robustness to input perturbation is a distinct axis that identifies fragile models which score well on correlation.

`raw/rafi-2024-dream.xml` (the Random Promoter DREAM Challenge) is the community critical assessment for sequence-to-expression models, and contributes the **Prix Fixe** framework: decompose competing models into modular building blocks — architecture, training strategy, data processing — and test all combinations, so that a performance difference can be attributed to a specific component rather than to the model as a whole. It also builds a benchmark suite of multiple **sequence-type subsets** rather than one aggregate score, and reports that subset-level analysis exposes disparities that a single number hides — the same argument for partitioned measures made above.

## Rare positives break AUROC

The challenge's binarised measures above are precision- and recall-based for a reason worth
stating, and `raw/saito-2015-precision-recall-plot.xml` is the source that states it: the
precision–recall plot is more informative than the ROC plot when evaluating binary classifiers on
**imbalanced** datasets. Peak labels are
**heavily imbalanced** — positives are a small minority of loci — and ROC-AUC's false-positive
rate has the (large) negative count in its denominator, so a model can improve its FPR from
0.02 to 0.01 and move AUROC visibly while still returning mostly false positives among its
predicted peaks. Average precision / the precision–recall curve conditions on the predicted
positives instead, so it moves only when the positives get better. Report both, or report AUPRC
alone; AUROC on rare positives flatters every model and compresses the differences between them.
The challenge's own instrument — precision and recall **per cell-type-specificity group** — is
the stratified version of the same correction.

## Coverage guarantees and evaluation under shift

`raw/zhou-2026-degu.xml` (DEGU) supplies the evaluation counterpart to [[uncertainty-calibration]]'s
modelling story. Two contributions bear on measurement rather than on architecture:

- **Conformal prediction** gives interval coverage guarantees "under minimal assumptions" — a
  distribution-free wrapper that converts any point predictor into one with a calibrated
  interval, evaluated against a nominal target. This is the honest fallback when a parametric
  head's own calibration is in doubt.
- It names the assumption that in-distribution held-out evaluation quietly makes: held-out
  sequences "come from different genomic regions" but are otherwise typical of the training
  experiment, so they do not test **covariate shift**. DEGU reports its distilled uncertainty
  improves generalisation precisely in the shifted regime. This is the measurement-side statement
  of [[cross-cell-type-generalization-pitfall]] — a held-out chromosome is not a held-out
  condition.

## See also

Related:: [[encode-imputation-challenge]], [[average-activity-baseline]], [[peak-calling-and-signal-tracks]], [[uncertainty-calibration]], [[cross-cell-type-generalization-pitfall]], [[imbalance-aware-objectives]]

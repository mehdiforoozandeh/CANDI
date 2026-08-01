---
type: wiki
title: Uncertainty and calibration
summary: Modern networks are systematically overconfident; calibration is measurable, and in genomics the posterior probabilities emitted by SAGA models are a worked example of miscalibration with real consequences.
category: concept
sources: raw/guo-2017-calibration-modern-neural-networks.pdf, raw/shahraki-2024-robust-chromatin-state-annotation.xml, raw/schreiber-2023-encode-imputation-challenge.pdf
created: 2026-07-31T21:26:00
updated: 2026-07-31T21:26:00
---

# Uncertainty and calibration

Calibration is a property distinct from accuracy — a model can be more accurate and simultaneously more wrong about how confident it should be — and both sources here document exactly that dissociation.

## The general phenomenon

`raw/guo-2017-calibration-modern-neural-networks.pdf` reports that modern neural networks exhibit "a strange phenomenon: probabilistic error and miscalibration worsen even as classification error is reduced." The drivers identified are architectural and procedural — **model capacity, normalisation, and regularisation** — meaning the same changes that improved accuracy over the preceding decade degraded calibration.

The practical consequences: a network's softmax output cannot be read as a probability without correction; calibration must be measured explicitly (reliability diagrams, expected calibration error) rather than assumed; and post-hoc correction on a held-out set is often sufficient.

## The genomics instance

`raw/shahraki-2024-robust-chromatin-state-annotation.xml` supplies the domain-specific version. SAGA methods (ChromHMM, Segway — see [[chromatin-state-annotation]]) emit a **posterior probability** for the state assigned at each genomic position, which users naturally read as a confidence. The paper finds these posteriors are "often vastly too confident": most positions receive near-certain posteriors, yet applying the same SAGA method to two replicate datasets of the same cell type yields annotations that agree at only ~80% of genomic bins.

The paper's diagnosis separates two causes of irreproducibility that a posterior does not distinguish: models learn **multiple similar states not confidently distinguishable from each other**, and **spatial misalignment** of segment boundaries between replicates. Its remedy, SAGAconf, assigns *calibrated* confidence scores derived from measured reproducibility rather than from the model's own posterior — i.e. it recalibrates against an empirical replicate-agreement target.

## The gap in imputation evaluation

`raw/schreiber-2023-encode-imputation-challenge.pdf` evaluates only point predictions; every one of its measures (MSE, correlations, binarised accuracy/precision/recall — see [[imputation-evaluation-measures]]) compares a single predicted value to a single observed value. A method that emits a predictive **distribution** rather than a point estimate is not distinguishable from one that does not under these measures, so calibration and distributional ranking quality were untested in the field's most rigorous benchmark to date.

The two natural additions, by analogy with the sources above: **empirical coverage versus nominal credible-interval width** (the reliability diagram for a continuous predictive distribution), and a **ranking/concordance** measure asking whether the distribution orders loci correctly by expected signal.

## See also

Related:: [[count-distributions-for-sequencing-data]], [[chromatin-state-annotation]], [[imputation-evaluation-measures]], [[encode-imputation-challenge]]

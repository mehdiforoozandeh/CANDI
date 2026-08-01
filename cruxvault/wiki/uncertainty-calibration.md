---
type: wiki
title: Uncertainty and calibration
summary: Modern networks are systematically overconfident; calibration is measurable, and in genomics the posterior probabilities emitted by SAGA models are a worked example of miscalibration with real consequences.
category: concept
sources: raw/guo-2017-calibration-modern-neural-networks.pdf, raw/shahraki-2024-robust-chromatin-state-annotation.xml, raw/schreiber-2023-encode-imputation-challenge.pdf, raw/gneiting-2007-gneiting-raftery.pdf, raw/seitzer-2022-seitzer-betanll.pdf, raw/kuleshov-2018.pdf, raw/zhou-2026-degu.xml, raw/kendall-2017-kendall-uw.pdf
created: 2026-07-31T21:26:00
updated: 2026-07-31T23:27:28
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

## Proper scoring rules — the missing foundation

`raw/gneiting-2007-gneiting-raftery.pdf` is the reference that legitimises distributional evaluation. A scoring rule assigns a numerical score to a predictive distribution given the value that materialises. It is **proper** if the forecaster maximises expected score by reporting their true belief `F` rather than any other `G`, and **strictly proper** if that maximum is unique. Propriety is what makes a score non-gameable: an improper score can be improved by misreporting uncertainty.

The two consequences that matter operationally:

- **CRPS** (continuous ranked probability score) is a strictly proper score for real-valued outcomes, expressible as the integrated squared difference between the predicted CDF and the step function at the observation, and computable in closed form for many distributions. It reduces to absolute error for a point forecast, so point and distributional predictions are directly comparable on one scale.
- The paper formalises **calibration versus sharpness**: the goal is maximising sharpness *subject to* calibration. A distribution can be perfectly calibrated and useless (predict the marginal every time), or very sharp and badly wrong. Any single scalar that trades these off is hiding which one moved.

Note that a proper score is not decomposable by inspection into "how good is the model" and "how badly is its scale off"; that split has to be constructed deliberately.

## Why a learned-variance head under-fits

`raw/seitzer-2022-seitzer-betanll.pdf` diagnoses a specific, common failure of heteroscedastic Gaussian heads trained by maximising log-likelihood. The NLL gradient with respect to the mean is scaled by the **inverse predicted variance**, so regions the model currently believes are high-variance receive **down-weighted** mean gradients. Early in training this is self-reinforcing: poorly fit regions get high predicted variance, which suppresses the gradient that would fix them, so they stay poorly fit. The result is a mean that is systematically worse than a homoscedastic model's would have been.

The fix is one line — **β-NLL**, multiplying the per-example loss by a stop-gradient copy of the predicted variance raised to β — which interpolates between standard NLL (β=0) and MSE-like uniform weighting (β=1). This is the principled replacement for clamping the predicted variance at a floor, which suppresses the symptom while distorting the likelihood.

## Post-hoc recalibration

`raw/kuleshov-2018.pdf` gives the continuous-output analogue of temperature scaling: fit a monotone mapping on a held-out set so that the predicted quantiles match empirical frequencies — if the model claims 90% intervals, 90% of observations should fall inside them. It is cheap, model-agnostic, applies after training, and does not require changing the loss. For any model whose coverage curve is smooth but offset, this is strictly easier than adding calibration terms to the training objective.

## Uncertainty in genomics

`raw/zhou-2026-degu.xml` (DEGU) is the closest genomics comparator. It distils an **ensemble** of genomic DNNs into a single model that predicts both the ensemble mean and the ensemble variance — the latter capturing **epistemic** (model) uncertainty — with an optional auxiliary head for **aleatoric** (data) uncertainty, and evaluates against 95% coverage targets. It reports that the distilled uncertainty improves generalisation under covariate shift, the regime where uncertainty estimates matter most.

The separation is worth holding onto: a single distributional head captures aleatoric uncertainty only. Epistemic uncertainty — the model's uncertainty about its own parameters — requires ensembling or distillation of an ensemble, and no amount of calibration of a single head recovers it.

## Uncertainty as a loss weighter

`raw/kendall-2017-kendall-uw.pdf` uses the same machinery for a different purpose. Rather than reporting uncertainty, it **learns a per-task log-variance** and weights each task's loss by its inverse, adding a log-variance penalty to prevent the trivial solution of inflating all variances:

    L = Σ_t exp(−s_t)·L_t + ½·s_t

This is homoscedastic (task-level, not input-level) uncertainty, and it replaces hand-tuned loss weights with learned ones. The paper's finding is that multi-task performance is **highly dependent** on the weighting and that manual search is prohibitive.

One interaction is worth flagging when this is combined with the heads above: a task whose head **already predicts its own variance** has two variance-like quantities in play — the per-example predicted σ² inside the NLL, and the per-task learned s_t outside it. They are not the same object, and the `raw/seitzer-2022-seitzer-betanll.pdf` gradient pathology applies to the inner one regardless of what the outer one does.

## Ranking versus calibration

Harrell's concordance index measures **rank discrimination** — whether the predicted distributions order observations correctly — and is blind to calibration: a model can rank perfectly while being systematically overconfident. It is therefore a complement to coverage and PIT-based measures, never a substitute. (The originating paper, Harrell 1982, is not yet in `raw/`.)

## See also

Related:: [[count-distributions-for-sequencing-data]], [[chromatin-state-annotation]], [[imputation-evaluation-measures]], [[encode-imputation-challenge]], [[training-mechanics]], [[count-models-in-single-cell-genomics]]

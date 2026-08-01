---
type: wiki
title: Covariate conditioning and counterfactuals
summary: Making a model respond to an explicit covariate rather than absorbing it into the latent — adversarial disentangling, compositional embeddings, and how to test that conditioning is actually used.
category: concept
sources: raw/lotfollahi-2023-cpa.xml, raw/boyeau-2025-mrvi.xml, raw/mller-2025-deepdive.pdf, raw/elazar-2020-amnesic.pdf, raw/vafa-2025-steerability.pdf
created: 2026-07-31T23:27:28
updated: 2026-07-31T23:27:28
---

# Covariate conditioning and counterfactuals

The recurring failure is not that a conditional model ignores its covariate outright, but that it recovers the covariate's information from the input and routes it through the latent instead — leaving the conditioning input decorative.

## Why conditioning gets ignored

If a covariate is predictable from the data itself, an unconstrained encoder will encode it, and the decoder will read it from the latent rather than from the conditioning vector. The conditioning pathway then carries no gradient and does nothing at inference. Every method below is a way to break that redundancy.

## Compositional conditioning

`raw/lotfollahi-2023-cpa.xml` (Compositional Perturbation Autoencoder) is the canonical construction. It decomposes a cell's representation into a **basal state** plus additive **perturbation** and **covariate** embeddings, with a nonlinear dose scalar. The basal latent is adversarially forced to be *free* of the perturbation and covariate information, so those factors can only enter through their explicit embeddings. Because the embeddings compose additively, the model predicts responses for **unseen combinations** — unseen dosages, cell types, time points, and species — and validates on held-out drug combinations. It runs on count data.

Two properties matter: the **adversarial term is what makes the conditioning load-bearing**, and **compositionality is what buys generalisation to unseen covariate combinations**. Neither comes free from simply concatenating a covariate.

## Separating nuisance from target covariates

`raw/boyeau-2025-mrvi.xml` (MrVI) uses two hierarchy levels to distinguish two *kinds* of sample-level covariate: **nuisance** (site, preparation technology, study) and **target** (the sample identity or condition you actually want to reason about). The nuisance covariates are corrected out; the target covariate is retained and can be **substituted at inference to generate counterfactuals** — what would this cell look like if it came from that sample?

This two-tier distinction is the cleanest available framing for a covariate vector whose entries are not all the same kind of thing.

## The epigenomic instance

`raw/mller-2025-deepdive.pdf` (DeepDive) applies the same idea to single-cell-resolved epigenomes: an end-to-end probabilistic framework that **disentangles known covariate effects** from residual variation, explicitly to enable accurate **counterfactual prediction**. It is the nearest published precedent for treating experimental covariates in a chromatin assay as a steerable input rather than a batch effect to be removed.

## Testing whether conditioning is used

Two sources address the measurement problem, which is harder than the modelling problem.

`raw/elazar-2020-amnesic.pdf` (Amnesic Probing) makes an argument that generalises well beyond NLP: **you cannot infer behavioural conclusions from probing results.** Showing that a property is *recoverable* from a representation does not show the model *uses* it. Their alternative measures the influence of a **causal intervention that removes the property from the representation** and observes the degradation in task behaviour. Applied to conditioning: a probe that recovers a covariate from the latent is not evidence the covariate drives the output; ablating it and measuring what changes is.

`raw/vafa-2025-steerability.pdf` supplies the complementary framing, decomposing **steerability** from **producibility**. A model may be able to *produce* an output — it lies within the model's range — while a user with a specific goal cannot *reach* it by setting the conditioning. These are different properties and the second is what a conditioning interface is for. The paper gives a mathematical decomposition separating the two and notes steerability is harder to evaluate because it requires knowing the user's goal.

Together they define the right test for a conditioned model: not "is the covariate encoded" (probing), but "does intervening on the conditioning move the output to the requested place" (steerability), with ablation as the control.

## See also

Related:: [[film-conditioning]], [[jepa-and-collapse-prevention]], [[count-models-in-single-cell-genomics]], [[distributional-shift-and-batch-effects]], [[sequence-conditioned-epigenome-models]]

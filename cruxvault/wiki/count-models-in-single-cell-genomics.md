---
type: wiki
title: Count models in single-cell genomics
summary: The single-cell field's decade-long argument about how to model raw counts — NB vs Poisson, zero-inflation, library-size offsets, and whether to transform at all.
category: comparison
sources: raw/lopez-2018-scvi.xml, raw/eraslan-2019-dca.xml, raw/hafemeister-2019-sctransform.xml, raw/choudhary-2022-sctransform-v2.xml, raw/martens-2023.xml, raw/svensson-2020.pdf, raw/townes-2019-glmpca.xml, raw/ahlmann-eltze-2023.xml, raw/ashuach-2022-peakvi.xml, raw/ashuach-2023-multivi.xml, raw/chen-2025-epiagent.pdf
created: 2026-07-31T23:27:28
updated: 2026-08-01T18:05:24
---

# Count models in single-cell genomics

Single-cell genomics ran the raw-counts-versus-normalised-signal experiment a decade before epigenome imputation did, and the conclusions transfer directly — including one strong dissent.

## The case against transforming

`raw/townes-2019-glmpca.xml` shows with negative controls that UMI counts follow **multinomial sampling with no zero inflation**, and that current practice — log of counts-per-million, then feature selection by highly variable genes — **produces false variability** in dimension reduction. Its remedy is to model counts directly (GLM-PCA, a generalised PCA for non-normal likelihoods) and select features by deviance.

`raw/martens-2023.xml` makes the same argument in the chromatin domain, and it is the most directly transferable result here: for scATAC-seq, **binarisation is an unnecessary step** that improves neither goodness of fit, clustering, cell-type identification, nor batch integration. The prescription has a subtlety worth carrying: model **fragment counts, not read counts** — the Tn5 insertion event is the physical unit, and read counts double-count it.

## What likelihood

- **Negative binomial.** `raw/lopez-2018-scvi.xml` (scVI) models expression as zero-inflated negative binomial with a **library-size latent** and batch annotations fed to the decoder, explicitly decoupling biological signal from sequencing depth and categorical nuisance factors. `raw/eraslan-2019-dca.xml` (DCA) uses an NB (optionally zero-inflated) autoencoder to **denoise** counts, taking count distribution, overdispersion and sparsity into account, scaling linearly in cells.
- **Against zero-inflation.** `raw/svensson-2020.pdf` examines technical controls and concludes droplet scRNA-seq is **not zero-inflated** — the excess zeros are what a plain NB predicts at low mean. `raw/townes-2019-glmpca.xml` agrees from the multinomial side. Together these are the standard justification for a plain NB rather than ZINB.
- **When Poisson suffices.** `raw/choudhary-2022-sctransform-v2.xml` analyses 59 datasets and gives the decisive answer: a Poisson model looks adequate for **sparse** data, but with sufficient sequencing depth there is clear evidence of overdispersion in every biological system, **necessitating NB**. Shallow sequencing masks overdispersion — so "Poisson fits" can be an artefact of depth, not a property of the assay.

## Parameterising dispersion

This is where the single-cell literature is most useful and most often skipped. `raw/hafemeister-2019-sctransform.xml` (sctransform) fits a **negative binomial GLM per gene with log sequencing depth as a covariate**, and shows that an **unconstrained NB overfits** scRNA-seq: per-gene dispersion estimates are unstable. The fix is regularisation by **pooling information across genes of similar abundance** — fit the per-gene parameters, then smooth them as a function of mean abundance. `raw/choudhary-2022-sctransform-v2.xml` confirms and sharpens this: overdispersion θ is not a constant and not free per feature, but **varies systematically as a function of abundance**.

The general lesson: a per-feature dispersion parameter is estimable only with pooling; depth belongs in the model as an **offset on the mean**, not as a divisor applied to the data. See [[count-distributions-for-sequencing-data]].

## Multiple likelihoods off one latent

`raw/ashuach-2022-peakvi.xml` (PeakVI) models scATAC accessibility as a **Bernoulli** probability per region, factorised into a per-cell library-size factor and a per-region bias, with batch conditioned into the decoder — the closest published precedent for a peak head that is depth- and batch-aware rather than depth-normalised. `raw/ashuach-2023-multivi.xml` (MultiVI) runs **NB (expression) and Bernoulli (accessibility) heads off a single shared latent**, and imputes a missing modality for cells where it was not measured — structurally the same move as predicting several distributional heads for an assay from one shared representation.

## The tokenisation escape route

`raw/chen-2025-epiagent.pdf` (EpiAgent) sidesteps the whole likelihood argument, which makes it a
useful boundary case. Facing the same three problems this page addresses — "the abundance of
features, high data sparsity, and the **quasi-binary** nature of these data" — it neither models
counts nor binarises them into a Bernoulli field. It **ranks** accessible cCREs by their
TF-IDF-transformed values and emits the ordered list as a "cell sentence", then applies
bidirectional attention over that sequence.

The move is to convert a magnitude problem into an ordering problem: rank information survives,
absolute scale is discarded, and the depth-normalisation question dissolves because a ranking is
depth-invariant by construction. The cost is exactly what CANDI needs and this representation
cannot give — no predictive distribution over a count, so no calibration, no uncertainty, and no
way to express "this position has 3 reads and I am confident" versus "3 reads and I am not".

It is worth registering as the strongest available argument that count modelling is a *choice*
rather than a necessity, and worth registering equally that the choice is forced once calibrated
per-position prediction is the goal. See [[set-conditioned-modelling-and-missingness]].

## The dissent

`raw/ahlmann-eltze-2023.xml` compared 22 transformations across four families and found that the simplest — **shifted logarithm `log(y/s + 1)` followed by PCA** — performs as well as GLM-PCA and latent-expression models at recovering latent structure, despite the latter having better theoretical properties. This is the strongest counter-argument to the raw-counts position and should be engaged rather than ignored. Its scope is important: the benchmark is **scRNA-seq dimension reduction**, judged on recovery of latent cell structure — not calibrated per-position prediction, not denoising, and not a task where the count scale itself is the quantity of interest.

## See also

Related:: [[count-distributions-for-sequencing-data]], [[signal-normalization-in-epigenomics]], [[epigenome-denoising]], [[covariate-conditioning-and-counterfactuals]], [[sequencing-depth-and-coverage]]

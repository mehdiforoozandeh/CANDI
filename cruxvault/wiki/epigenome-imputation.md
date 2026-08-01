---
type: wiki
title: Epigenome imputation
summary: Predicting unperformed epigenomic experiments from the correlation structure of observed ones; the task definition, its history, and why imputed data is often used for denoising.
category: overview
sources: raw/ernst-2015-chromimpute.xml, raw/durham-2018-predictd.xml, raw/schreiber-2020-avocado.xml, raw/schreiber-2020-encode3-compendium.xml, raw/hawkins-hooker-2023-edice.xml, raw/schreiber-2023-encode-imputation-challenge.pdf, raw/zhang-2023-epcot.xml, raw/wen-2024-discriminative-histone-imputation.pdf, raw/boix-2021-regulatory-genomic-circuitry.xml, raw/chen-2025-epiagent.pdf
created: 2026-07-31T21:26:00
updated: 2026-07-31T23:27:28
---

# Epigenome imputation

The field's organising abstraction is a **3-D tensor** — cell types × assays × genomic positions — that is mostly empty, and every imputation method is a different bet about how to fill it.

## The task

Reference compendia measure assay *a* in cell type *c* only for a small fraction of (c, a) pairs (`raw/schreiber-2023-encode-imputation-challenge.pdf`, `raw/durham-2018-predictd.xml`). Imputation predicts the signal track for a missing (c, a) pair from the tracks that *were* performed, exploiting two correlation structures: across assays within a cell type, and across cell types within an assay. See [[reference-epigenome-compendia]] for the data these methods consume.

Nearly all published methods predict **processed signal**, specifically the −log10 p-value of enrichment over a local background, at 25 bp resolution (`raw/ernst-2015-chromimpute.xml`, `raw/schreiber-2023-encode-imputation-challenge.pdf`). This choice is load-bearing: it presumes upstream processing has already removed batch and depth effects and produced an idealised "signal strength," an assumption [[distributional-shift-and-batch-effects]] shows is false in practice.

## Lineage of methods

- **[[chromimpute]]** (2015) — ensembles of regression trees, one model per target experiment; the first large-scale demonstration, and still the only method that can denoise an existing experiment without retraining.
- **[[predictd]]** (2018) — parallel tensor decomposition; imputes all missing entries jointly from one factorisation.
- **[[avocado]]** (2020) — deep tensor factorisation with multi-scale genomic factors; produces a reusable latent representation.
- **[[edice]]** (2023) — attention over observed tracks; pushes into individual-specific (donor-level) imputation.
- **[[sequence-conditioned-epigenome-models]]** — a separate lineage that predicts epigenomic signal from DNA sequence (± chromatin accessibility) rather than from other assays.

## Imputation as denoising

`raw/ernst-2015-chromimpute.xml` reports that imputed tracks *surpass* the observed experiments they replace on consistency, recovery of gene annotations, and enrichment for disease-associated variants — because averaging across correlated experiments suppresses experiment-specific noise. `raw/boix-2021-regulatory-genomic-circuitry.xml` (EpiMap, ~15,000 tracks over 833 biosamples) reports the same effect at scale: imputed datasets clustered more cleanly than observed ones and their pairwise distances were *less* affected by technical covariates. This is why imputation is routinely applied as a denoiser even when the observed experiment exists.

The practical limitation is procedural: to denoise experiment *x* with a tensor-factorisation or attention method, *x* must be held out of training and re-imputed, which means retraining. Only [[chromimpute]]'s per-target architecture avoids this. Re-imputation also discards the target experiment's own measurement entirely, rather than treating it as noisy evidence.

## Evaluation is the unsolved part

`raw/schreiber-2023-encode-imputation-challenge.pdf` is the field's post-mortem: 23 methods evaluated prospectively, and under the pre-registered measures a naive [[average-activity-baseline]] outperformed all but two. See [[encode-imputation-challenge]], [[imputation-evaluation-measures]], and [[cross-cell-type-generalization-pitfall]].

## Single-cell foundation models

`raw/chen-2025-epiagent.pdf` (EpiAgent) is the single-cell mirror of this task. Pretrained on ~5 million cells and 35 billion tokens of scATAC data, it encodes each cell's accessibility pattern as a "cell sentence" and uses bidirectional attention over it. Among its downstream tasks are **zero-shot cell-type annotation and data imputation on unseen scATAC data**, plus prediction of cellular responses to unseen stimuli and genetic perturbations.

Two things transfer. The problem framing is the same — sparse, quasi-binary observations over a large feature space, with imputation as a first-class downstream task rather than an afterthought. And the tokenisation choice is instructive: representing a cell by the *set of its accessible regions* rather than a dense vector over all regions is the single-cell analogue of treating the observed assays as a set (see [[set-conditioned-modelling-and-missingness]]).

## Denoising as a supervised task

The framing of imputation-as-denoising described above has a parallel literature that trains directly on paired low-quality and high-quality experiments rather than imputing a held-out track — see [[epigenome-denoising]].

## See also

Related:: [[encode-imputation-challenge]], [[reference-epigenome-compendia]], [[signal-normalization-in-epigenomics]], [[chromatin-state-annotation]], [[masked-self-supervised-learning]], [[epigenome-denoising]], [[set-conditioned-modelling-and-missingness]]

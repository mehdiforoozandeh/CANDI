---
type: wiki
title: Epigenome denoising
summary: Coda and AtacWorks: learning a mapping from low-quality to high-quality chromatin signal, generalising to unseen cell types — the direct ancestors of denoising-by-model rather than by imputation.
category: method
sources: raw/koh-2017-coda.xml, raw/lal-2021-atacworks.xml
created: 2026-07-31T23:27:28
updated: 2026-07-31T23:27:28
---

# Epigenome denoising

These two papers define denoising as a **supervised paired-data problem** — train on (low-quality, high-quality) pairs of the same experiment — which is a fundamentally different construction from denoising by re-imputation.

## Coda

`raw/koh-2017-coda.xml` (Convolutional Denoising Algorithm) is the seminal work. It takes a pair of matched ChIP-seq datasets for the same histone modification in the same cell type — one high-quality, one noisy — and trains a CNN to learn the **mapping from noisy to high-quality signal**. Inputs are noisy signal measurements of *multiple* histone marks, so the model exploits cross-mark correlation as well as within-track structure.

The framing of noise is the useful part. Coda is applied to three distinct, separately-manipulable sources:

1. **low sequencing depth** — motivated by the 40–50M read recommendation for human histone ChIP-seq (see [[sequencing-depth-and-coverage]]);
2. **low cell input**;
3. **low ChIP enrichment** (i.e. poor signal-to-noise, distinct from depth).

That decomposition matters: these degrade the signal in different ways, and a model trained on one does not automatically handle another. The paper demonstrates transfer across individuals, cell types and species.

## AtacWorks

`raw/lal-2021-atacworks.xml` extends the idea to chromatin accessibility with two additions relevant to any denoising model with a peak output:

- **Base-pair resolution** denoising of sequencing coverage, and **joint peak calling** from the denoised signal — one model producing both the continuous track and the discrete calls, rather than denoising and then running a separate caller.
- **Generalisation to cell types not seen in training**, and across sample preparations and experimental platforms.

Applied to droplet single-cell ATAC-seq, AtacWorks produces results from low cell counts on par with conventional methods on much larger inputs, and the authors extend it to transcription-factor footprinting as a cross-modality prediction.

## Relation to imputation-as-denoising

The [[epigenome-imputation]] lineage denoises by **holding an experiment out and re-imputing it** — the denoised output never sees the observed measurement. Coda and AtacWorks instead **consume the noisy measurement** and map it to what a better experiment would have produced. The two differ in what they can do:

- Imputation-based denoising needs no paired low/high-quality data, but discards the target experiment's own evidence and (except for [[chromimpute]]) requires retraining per target.
- Paired denoising uses the target measurement, but needs matched low/high-quality pairs to train on — obtainable by **downsampling** a deep experiment for the depth axis, but not straightforwardly for the enrichment or cell-input axes, where the degradation cannot be simulated from a good experiment.

Neither line emits a predictive distribution; both produce point estimates of the improved signal.

## See also

Related:: [[epigenome-imputation]], [[sequencing-depth-and-coverage]], [[peak-calling-and-signal-tracks]], [[count-models-in-single-cell-genomics]], [[chip-seq-assay-and-controls]]
